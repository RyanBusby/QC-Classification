import pickle
import os
import json
import pandas as pd
import csv

def run(logger):
    df = load_data(os.path.join('..','data','test_numeric.csv'), logger)
    models, pca_map, good_dfs = load_models()
    col_idx = make_col_idx(good_dfs)
    logger.info('STARTING - assign_row_to_cluster()')
    assignments = assign_row_to_cluster(df, col_idx)
    logger.info('STARTING - make_predictions()')
    preds = make_predictions(assignments, col_idx, df, modelchoices, pca_map, models, logger)
    pd.DataFrame(preds).to_csv(os.path.join('..','data','submission.csv'))

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
        data_url = 'https://www.kaggle.com/c/5357/download/test_numeric.csv.zip'
        zip_name = os.path.join('..','data', 'test_numeric.csv.zip')
        unzip_dir = os.path.join('..','data')
        download_data(data_url, zip_name, unzip_dir, logger)
    tp = pd.read_csv(unzip_name, iterator=True, chunksize=1000)
    df = pd.concat(tp, ignore_index=True)
    df.set_index('Id', inplace=True, drop=True)
    return df

def load_models():
    models, pca_map = {}, {}
    with open(os.path.join('..','data','models','modelchoices.json'), 'rb') as f:
        modelchoices = json.load(f)
    for name, d in modelchoices.items():
        with open(os.path.join('..','data','models',d['munge'],name), 'rb') as f:
            models[name] = pickle.load(f)
        if d['munge'] == 'pca':
            with open(os.path.join('..','data','pca_map', name+'.pickle'), 'rb') as f:
                pca_map[name] = pickle.load(f)
    good_dfs = list(modelchoices.keys())
    return models, pca_map, good_dfs

def make_col_idx(good_dfs):
    '''
    a dictionary that has the columns represented in each cluster
    '''
    col_idx = {df: None for df in good_dfs}
    for df in good_dfs:
        with open(os.path.join('..','data','clusters','%s.csv' % df), newline='') as f:
            reader = csv.reader(f)
            row1 = set(next(reader)).difference(['Id', 'Response'])
            col_idx[df] = row1
    return col_idx

def assign_row_to_cluster(DF, col_idx):
    assignments = {}
    for idx, row in DF.iterrows():
        shared_col_counts = []
        notnull_cols = set(DF.columns[row.notnull()])
        for df, cols in col_idx.items():
            shared_col_counts.append((len(cols.intersection(notnull_cols),df)))
        shared_col_counts.sort(reverse=True)
        assignments[idx] = shared_col_counts[0][1]
    return assignments

def make_predictions(assignments, col_idx, DF, modelchoices, pca_map, models, logger):
    preds = []
    ttl = len(assignments)
    n = 1
    for idx, assigned in assignments.items():
        if n % 1000 == 0:
            logger.info('{} of {} predicitons made ... {}%'.format(n, ttl, round(((n/float(ttl))*100), 2)))
        n+=1
        if not assigned:
            preds.append({'Id':idx,'Response':0})
        cols = list(col_idx[assigned])
        X = DF.loc[idx, cols].fillna(0).values
        n = X.shape[0]
        X = X.reshape(1,n)
        if modelchoices[assigned]['munge'] == 'pca':
            X = pca_map[assigned].transform(X)
        response = models[assigned].predict(X)[0]
        preds.append({'Id':idx,'Response':response})
    return preds
