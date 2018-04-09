import os
import requests
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

"""
download data from web
cluster sparse data based on shared columns
save cluster to .csv
"""

def run(logger):
    unzip_name = os.path.join('..','data', 'train_numeric.csv')
    logger.info('STARTING - load_data()')
    df, labels = load_data(unzip_name, logger)
    logger.info('STARTING - list_nulls()')
    row_indices = list_nulls(df)
    logger.info('STARTING - rank_columns_similarity()')
    mat = rank_column_similarity(df, row_indices, os.path.join('..','data','similair_columns.npy'), logger)
    logger.info('STARTING - make_clusters()')
    clusters = make_clusters(mat)
    logger.info('STARTING - make_dfs()')
    data_dict, col_counts = make_dfs(clusters, df, logger)
    logger.info('STARTING - assign_row_to_cluster()')
    keepers, empties = assign_row_to_cluster(col_counts)
    logger.info('STARTING - assign_rows_for_cluster')
    cluster_idx = assign_rows_for_cluster(data_dict, keepers)
    logger.info('SAVING CLUSTERS')
    save_dfs(data_dict, cluster_idx, empties, os.path.join('..','data','clusters','df{}.csv'), df, labels)

def download_data(data_url, zip_name, unzip_dir, logger):
    uname = os.getenv('KAGGLE_UNAME')
    pword = os.getenv('KAGGLE_PWORD')
    kaggle_info = {'UserName': uname, 'Password': pword}

    r = requests.get(data_url)
    r = requests.post(r.url, data=kaggle_info, stream=True)
    logger.info('Downloading from %s' % r.url)
    with open(zip_name, 'wb') as f:
        for chunk in tqdm(r.iter_content(chunk_size=512*1024), ascii=True, desc='DOWNLOADING...'):
            if chunk:
                f.write(chunk)
    zip_ref = zipfile.ZipFile(zip_name, 'r')
    zip_ref.extractall(unzip_dir)
    zip_ref.close()

def load_data(unzip_name, logger):
    if not os.path.isfile(unzip_name):
        data_url = 'https://www.kaggle.com/c/5357/download/train_numeric.csv.zip'
        zip_name = os.path.join('..','data', 'train_numeric.csv.zip')
        unzip_dir = os.path.join('..','data')
        download_data(data_url, zip_name, unzip_dir, logger)
    tp = pd.read_csv(unzip_name, iterator=True, chunksize=1000)
    df = pd.concat(tp, ignore_index=True)
    df.set_index('Id', inplace=True, drop=True)
    labels = df['Response']
    df.drop('Response', inplace=True, axis=1)
    return df, labels

def list_nulls(df):
    '''
    make list of sets of row indices that are not null for every column
    the index of the set in 'row_indices' corresponds to the index of the column in 'df'

    INPUT: pandas.DataFrame
    OUTPUT: list
    '''
    row_indices = []
    for x in range(df.shape[1]):
        row_indices.append(set(np.where(df.iloc[:,x].notnull())[0].tolist()))
    return row_indices

def rank_column_similarity(df, ri, path, logger):
    '''
    create matrix of number of shared rows between columns

          col1 col2  col3 col4
          --------------------
    col1 | 0    int  int int  |
    col2 | 0     0   int int  |
    col3 | 0     0    0  int  |
    col4 | 0     0    0   0   |

    INPUT: pandas.DataFrame, list, string
    OUTPUT: numpy.array
    '''
    if os.path.isfile(path):
        mat = np.load(path)
        logger.info('similair_columns.npy successfully loaded')
    else:
        logger.info('similair_columns.npy not loaded, must build ...')
        mat = np.zeros(df.shape[1]**2).reshape(df.shape[1], df.shape[1])
        counter = 0
        for a in range(mat.shape[0]): #this takes a long time to execute
            for b in range(a+1, mat.shape[0]):
                mat[a,b] = len(ri[a].intersection(ri[b]))
                counter += 1
                if counter % 1000 == 0:
                    per_done = round(counter/467544.*100, 4)
                    logger.info('{}% done')
        np.save(path, mat)
    return mat

def make_clusters(mat, logger):
    '''
    locate indices of highest ranked column pairs and group them together

    INPUT: numpy.array
    OUTPUT: list of sets
    '''
    keys = set(range(mat.shape[0]))
    clusters = []
    while len(keys) > 0:
        a = np.where(mat == mat.max())
        a = np.vstack((a[0],a[1])).T
        a.sort()
        ttl = len(a)
        for x, tup in enumerate(a):
            logger.info('{} of {} ... {}%'.format(str(x+1), ttl, round(100*x+1/float(ttl),2)))
            if tup[0] in keys and tup[1] in keys:
                clusters.append({tup[0], tup[1]})
                keys.difference_update([tup[0],tup[1]])
            elif tup[0] in keys and tup[1] not in keys:
                for cluster in clusters:
                    if tup[1] in cluster:
                        cluster.add(tup[0])
                        break
                keys.remove(tup[0])
            elif tup[0] not in keys and tup[1] in keys:
                for cluster in clusters:
                    if tup[0] in cluster:
                        cluster.add(tup[1])
                        break
                keys.remove(tup[1])
            mat[tup[0], tup[1]] = 0
    return clusters

def make_dfs(clusters, df, logger):
    '''
    create subsetted dataframes, drop Null rows and store in 'data_dict'
    count columns per row per cluster and store in dict

    INPUT: list, pandas.DataFrame
    OUTPUT: dict, dict
    '''
    data_dict = {}
    logger.info('instantiating col_counts dict')
    col_counts = {key:[] for key in df.index.values}
    ttl = len(clusters)
    for x, cluster in enumerate(clusters):
        logger.info('{} of {} ... {}%'.format(str(x+1), ttl, round(100*x+1/float(ttl),2)))
        dfx = 'df%s' % x
        col_idx = list(cluster)
        logger.info('instantiating pandas.DataFrame(%s)' % dfx)
        dataframe = df.iloc[:,col_idx]
        logger.info('STARTING - %s.dropna' % dfx)
        dataframe.dropna(how='all', inplace=True)
        logger.info('STARTING - %s.fillna()' % dfx)
        dataframe.fillna(0, inplace=True)
        data_dict[dfx] = dataframe
        logger.info('{} stored'.format(dfx))
        #count columns per row per cluster store in dict
        for i in dataframe.index.tolist():
            col_counts[i].append((len(col_idx), dfx))
        # temp = np.zeros(dataframe.shape[0])
        # temp.fill(len(col_idx))
        # count_full = np.array([dataframe.index.values, temp]).T
        # for entry in count_full:
        #     col_counts[int(entry[0])].append((int(entry[1]), dfx))
    return data_dict, col_counts

def assign_row_to_cluster(col_counts):
    '''
    make dictionary with key/value pairs being the row and which cluster that row is mostly represented in

    INPUT: dict
    OUTPUT: dict, set
    '''
    keepers = {}
    empties = set()
    for row, lst in col_counts.items():
        if len(lst) == 0:
            empties.add(row)
            continue
        lst.sort()
        keepers[row] = [tup[1] for tup in lst[-len(set(np.where(np.array(lst) == max(lst))[0])):]]
    return keepers, empties

def assign_rows_for_cluster(data_dict, keepers):
    '''
    transpose 'keepers' to associate each cluster with a unique list of rows

    INPUT: dict, dict
    OUTPUT: dict
    '''
    cluster_idx = {'df{}'.format(x):[] for x in range(1, len(data_dict)+1)}
    for row, lst in keepers.items():
        for dfn in lst:
            cluster_idx[dfn].append(row)
    return cluster_idx

def save_dfs(data_dict, cluster_idx, empties, filepath, df, labels):
    '''
    INPUT: dict, dict, set, string, pandas.DataFrame
    '''
    counter = 0
    dfx = df.loc[list(empties)]
    dfx['Response'] = labels
    dfx.to_csv(filepath.format(counter))
    for name, df_ in data_dict.items():
        counter += 1
        final = df_.loc[cluster_idx[name]]
        if len(final)>0:
            final['Response'] = labels
            final.to_csv(filepath.format(counter)+'.csv')
