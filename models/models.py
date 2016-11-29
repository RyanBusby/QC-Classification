import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
import cPickle as pickle

data_dict = {}

for n in xrange(1, 20):
    dfn = 'df{}'.format(n)
    data_dict[dfn] = pd.read_csv('../data/data/'+dfn+'.csv')
    data_dict[dfn].set_index('Id', drop=True, inplace=True)
    data_dict[dfn].sort_index(inplace=True)

data_dict['df15'].fillna(data_dict['df15'].L3_S30_F3499.mean(), inplace=True)

info_dict = {
            'df10': ('If', 7),
            'df5': ('If', 6),
            'df8': ('If', 6),
            'df17': ('If', 5),
            'df18': ('SVM', 1),
            'df6': ('SVM', 8),
            'df14': None,
            'df9': None,
            'df19': None,
            'df4': ('dt', 8),
            'df1': ('dt', 6),
            'df11': ('dt', 7),
            'df12': ('dt', 6),
            'df13': ('dt', 19),
            'df15': ('dt', 10),
            'df16': ('dt', 1),
            'df7': ('dt', 7),
            'df2': ('dt', 7),
            'df3': ('dt', 9)
            }

for name, dataframe in data_dict.iteritems():
    if info_dict[name] == None:
        continue
    num = name[2:]
    pca = PCA(n_components = info_dict[name][1])
    y = dataframe['Response'].values
    X = dataframe.drop('Response', axis=1).values

    fail = dataframe[dataframe['Response'] == 1].shape[0]/float(dataframe.shape[0])
    accept = 1 - fail
    class_weight = {1:fail, 0:accept}

    pca.fit(X)

    with open('pca/pca'+num+'.pkl', 'wb') as f:
        pickle.dump(pca, f)

    X = pca.transform(X)

    if info_dict[name][0] == 'If':
        If = IsolationForest()
        If.fit(X, y)
        with open('models/mod'+num+'.pkl', 'wb') as f:
            pickle.dump(If, f)

    elif info_dict[name][0] == 'SVM':
        svm = OneClassSVM()
        svm.fit(X, y)
        with open('models/mod'+num+'.pkl', 'wb') as f:
            pickle.dump(svm, f)

    elif info_dict[name][0] == 'dt':
        dt = DecisionTreeClassifier(class_weight=class_weight)
        dt.fit(X, y)
        with open('models/mod'+num+'.pkl', 'wb') as f:
            pickle.dump(dt, f)
