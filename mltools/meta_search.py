"""
Multi-core support for meta-parameters search of
machine learning models based on cross validation

use command to start several multicore support, e.g.
$ ipcluster -n 4 

the implementation is inspired by Grisel's presentation
https://github.com/ogrisel/parallel_ml_tutorial
"""
__all__ = ['persist_data_dict',
        'persist_cv_splits',
]

from sklearn.externals import joblib
from sklearn.grid_search import IterGrid
from IPython.parallel import Client
from sklearn.cross_validation import ShuffleSplit
import os

############### Run on master ################

def persist_data_dict(data_dict, data_folder = './'):
    filenames = {}
    for (name, data) in data_dict.items():
        filename = os.path.abspath(data_folder + name + '.pkl')
        joblib.dump(data, filename)
        filenames[name] = filename
    return filenames
    
def persist_cv_splits(name, X, y, data_folder = './',
            n_iter=5, suffix = '_cv_%03d',
            test_size = 0.2, random_state = 0):
    cv = ShuffleSplit(X.shape[0], n_iter = n_iter, 
                    test_size = test_size, random_state = random_state)
    datadict = {}
    for (i, (train, test)) in enumerate(cv):
        dataname = name + suffix % i
        datablock = (X[train], y[train], X[test], y[test])
        datadict[dataname] = datablock
    return persist_data_dict(datadict, data_folder)
    
############### Run on workers in parallel ################
def evaluate_model_on_data(model, params, datafile):
    from sklearn.externals import joblib
    X_train, y_train, X_validation, y_validation = joblib.load(datafile)
    model.set_params(**params)
    model.fit(X_train, y_train)