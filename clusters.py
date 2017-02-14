import numpy as np
import pandas as pd

def read_data(filename):
    df = pd.read_csv(filename)
    df.set_index('Id', drop=True, inplace=True)
    df_l = df
    df = df.drop('Response', axis=1)
    return df_l, df

def list_nulls(df):
    '''
    make list of sets of row indices that are not null for every column
    the index of the set in 'row_indices' corresponds to the index of the column in 'df'

    INPUT: pd.DataFrame
    OUTPUT: list
    '''
    row_indices = []
    for x in xrange(df.shape[1]):
        row_indices.append(set(np.where(df.iloc[:,x].notnull())[0].tolist()))
    return row_indices

def rank_column_similarity(df, row_indices, path):
    '''
    create matrix of number of shared rows between columns

          col1 col2  col3 col4
          --------------------
    col1 | 0    int  int int  |
    col2 | 0     0   int int  |
    col3 | 0     0    0  int  |
    col4 | 0     0    0   0   |

    INPUT: pd.DataFrame, list, string
    OUTPUT: np.array
    '''
    try:
        mat = np.load(path)
        print '\x1b[6;30;42m'+'similair_columns successfully loaded'+'\x1b[0m'
    except:
        print '\x1b[6;30;41m'+'similair_columns not loaded'+'\x1b[0m'
        mat = np.zeros(df.shape[1]**2).reshape(df.shape[1], df.shape[1])
        for a in xrange(mat.shape[0]): #this takes a long time to execute
            for b in xrange(a+1, mat.shape[0]):
                mat[a,b] = len(row_indices[a].intersection(row_indices[b]))
        np.save(path, mat)
    return mat

def make_clusters(mat):
    '''
    locate indices of highest ranked column pairs and group them together

    INPUT: np.array
    OUTPUT: list
    '''
    keys = set(range(mat.shape[0]))
    clusters = []
    while len(keys) > 0:
        a = np.where(mat == mat.max())
        a = np.vstack((a[0],a[1])).T
        a.sort()
        for tup in a:
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

def make_dfs(clusters, df_l, filepath):
    '''
    create subsetted dataframes, drop null rows and store in 'data_dict'

    INPUT: list, pd.DataFrame, string
    '''
    for cluster in clusters:
        dfx = 'df{}'.format(clusters.index(cluster)+1)
        col_idx = list(cluster)
        col_idx.append(df_l.shape[1]-1) #append the index of 'Response'
        dataframe = df_l.iloc[:,col_idx]
        dataframe2 = dataframe.dropna(how='any')
        dataframe2.to_csv(filepath.format(dfx)+'.csv')

df_l, df = read_data('data/train_numeric.csv')
row_indices = list_nulls(df)
mat = rank_column_similarity(df, row_indices, 'data/similair_columns.npy')
clusters = make_clusters(mat)
make_dfs(clusters, df_l, 'data/clusters/{}')
