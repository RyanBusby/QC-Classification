import os
import pickle
from glob import glob
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA

"""
perform SVD on each cluster and save as seperate cluster
in choose_k, last karg is set to .95, that means 95%
of the variance of each cluster is being explained by k components,
95% was chosen for conventionality, but might be worth tweaking and seeing
if there is any improvement
"""

def run(logger):
    data, indices, labels = load_clusters()
    dfs = drop_non_fails(data)
    X = center_data(dfs, data)
    logger.info('STARTING - transform_clusters()')
    Z, pca_map = transform_clusters(dfs, X, logger)
    save_Z_pca_maps(Z, pca_map, dfs, indices, labels)

def load_clusters():
    """
    load clustered data, not decomposed
    """
    fnames = glob(os.path.join('..','data','clusters','*.csv'))
    data = {}
    indices = {}
    labels = {}
    for name in fnames:
        dfx = name.split('/')[-1][:-4]
        df = pd.read_csv(name, index_col='Id')
        data[dfx] = df
        indices[dfx] = df.index
        labels[dfx] = df.Response
    return data, indices, labels

def drop_non_fails(data):
    dfs = []
    for name, df in data.items():
        if df.Response.sum() > 2:
            dfs.append(name)
    return dfs

def center_data(dfs, data):
    A = {name: data[name].drop('Response', axis=1).values for name in dfs}
    X = {name: preprocessing.scale(A[name]) for name in dfs}
    return X

def choose_k(singular_values, expalained_variance=.95):
    total = np.nansum(singular_values)
    k = 1
    exp_var = (singular_values[:k].sum()/total)
    while exp_var < expalained_variance:
        k += 1
        exp_var = (singular_values[:k].sum()/total)
    return k

def transform_clusters(dfs, X, logger):
    Z = {}
    pca_map = {}
    for n, df in enumerate(dfs):
        logger.info('{} of {}'.format(n+1, len(dfs)))
        model = PCA(svd_solver='full', random_state=11).fit(X[df])
        k = choose_k(model.singular_values_)
        pca_map[df] = PCA(n_components=k, random_state=11)
        logger.info('fitting cluster {}x{} with {} components'.format(k, X[df].shape[0], X[df].shape[1]))
        Z[df] = pca_map[df].fit_transform(X[df])
    return Z, pca_map

def save_Z_pca_maps(Z, pca_map, dfs, indices, labels):
    for df in dfs:
        dfx = pd.DataFrame(Z[df], index=indices[df])
        dfx['Response'] = labels[df]
        dfx.to_csv(os.path.join('..','data','pca',df+'.csv'))
        with open(os.path.join('..','data','pca_map',df+'.pickle'), 'wb') as f:
            pickle.dump(pca_map[df], f)
