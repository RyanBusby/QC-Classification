import os
import pandas as pd
import pickle
from glob import glob
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef as mc

"""
run clusters through different classifier algorithms
choose best combination of munge technique (decomposed or not)
and classification, persist relevant models for prediction
"""

def run(logger):
    logger.info('STARTING - get_result()')
    result, dfs, models = get_result(['clusters', 'pca'], logger)
    final = {}
    hi_scores = {df:0.0 for df in dfs}
    for df in dfs:
        print(df)
        for d in result:
            dfx = list(d.keys())[0]
            if  dfx == df and d[dfx]['score'] > hi_scores[dfx]:
                hi_scores[dfx] = d[dfx]['score']
                final[dfx] = d[dfx]
    logger.info('SAVING MODELS')
    for name, d in final.items():
        with open(os.path.join('..','data','models', d['munge'], name), 'wb') as f:
            pickle.dump(models[d['munge']][d['classifier']], f)
    modelchoicef = os.path.join('..','data','models','modelchoices')
    if os.path.isfile(modelchoicef):
        with open(modelchoicef, 'rb') as f:
            old_final = json.load(f)
        for name, d in final.items():
            old_final[name] = d
        final = old_final
    with open(modelchoicef, 'wb') as f:
        json.dump(final, f)

def get_result(munges, logger):
    '''
    INPUT: lst of directories containing .csv files,
    each entry cooresponding to a different munge
    OUTPUT: lst of dicts
    '''
    result = []
    models = {}
    for munge in munges:
        logger.info(munge)
        logger.info('STARTING - make_data_dict()')
        data, dfs = make_data_dict(munge)
        class_weights = get_class_weights(data, dfs)
        X_trains, X_tests, y_trains, y_tests = split_training(data, dfs)
        logger.info('STARTING - train_classifiers()')
        classifiers = train_classifiers(dfs, class_weights, X_trains, X_tests, y_trains, y_tests, munge, logger)
        models[list(classifiers.keys())[0]] = classifiers[munge]
        logger.info('STARTING - make_preds()')
        d_preds = make_preds(dfs, classifiers[munge], X_tests)
        logger.info('STARTING - score()')
        d_scores = score(dfs, d_preds, y_tests)
        logger.info('STARTING - choose_classifier()')
        d_choice = choose_classifier(d_scores, dfs)
        scores = {df: d_scores[df][d_choice[df]] for df in dfs}
        for df in dfs:
            d = {df:{}}
            d[df]['score'] = d_scores[df][d_choice[df]]
            d[df]['classifier'] = d_choice[df]
            d[df]['munge'] = munge
            result.append(d)
    return result, dfs, models

def make_data_dict(file):
    fnames = glob(os.path.join('..','data',file,'*.csv'))
    data = {name.split('/')[-1][:-4]: pd.read_csv(name, index_col='Id') for name in fnames}
    #some of the clusters don't have any failuers, can't model without both classes
    dfs = [d for d in data.keys() if data[d]['Response'].sum()>1]
    return data, dfs

def get_class_weights(data, dfs):
    class_weights = {}
    for df in dfs:
        dfx = data[df]
        n_failures = dfx.Response.sum()
        class_weights[df] = n_failures/dfx.shape[0]
    return class_weights

def split_training(data, dfs):
    X_trains, X_tests, y_trains, y_tests = {},{},{},{}
    for df in dfs:
        X_trains[df], X_tests[df], y_trains[df], y_tests[df] = \
        train_test_split(data[df].drop('Response', axis=1).values, \
                         data[df]['Response'], \
                         stratify=data[df]['Response'])
    return X_trains, X_tests, y_trains, y_tests

def train_classifiers(dfs, class_weights, X_trains, X_tests, y_trains, y_tests, munge, logger):
    logger.info('TRAINING LogisticRegression on {} clusters'.format(len(dfs)))
    lrs = {df: LogisticRegression(fit_intercept=True).fit(X_trains[df], y_trains[df]) for df in dfs}
    logger.info('TRAINING RandomForestClassifier on {} clusters'.format(len(dfs)))
    rfs = {df: RandomForestClassifier(max_features='sqrt',\
                                      class_weight={0:1-class_weights[df],\
                                      1:class_weights[df]}).fit(X_trains[df], y_trains[df])\
                                      for df in dfs}
    logger.info('TRAINING DecisionTreeClassifier on {} clusters'.format(len(dfs)))
    dts = {df: DecisionTreeClassifier(max_features='sqrt',\
                                      class_weight={0:1-class_weights[df],\
                                      1:class_weights[df]}).fit(X_trains[df], y_trains[df])\
                                      for df in dfs}

    classifiers = {munge:{'lrs': lrs, 'rfs': rfs, 'dts': dts}}
    return classifiers

def make_preds(dfs, classifiers, X_tests):
    d_preds = {df: {} for df in dfs}
    for df in dfs:
        for name, classifier in classifiers.items():
            d_preds[df][name] = classifier[df].predict(X_tests[df])
    return d_preds

def score(dfs, d_preds, y_tests):
    d_scores = {df: {} for df in dfs}
    for df in dfs:
        for name, preds in d_preds[df].items():
            d_scores[df][name] = mc(y_tests[df], preds)
    return d_scores

def choose_classifier(d_scores, dfs):
    d_choice = {}
    for df in dfs:
        best = -1
        for name, score in d_scores[df].items():
            if score > best:
                best = score
                choice = name
        d_choice[df] = choice
    return d_choice
