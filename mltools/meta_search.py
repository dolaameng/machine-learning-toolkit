"""
Multi-core support for meta-parameters search of
machine learning models based on cross validation

use command to start several multicore support, e.g.
$ ipcluster start -n 4 

the implementation is inspired by Grisel's presentation
https://github.com/ogrisel/parallel_ml_tutorial
"""
__all__ = []

import dataio
import multicore
from sklearn.cross_validation import ShuffleSplit
import os

class CVSearch(object):
    """search of meta parameters based on cross validation
    WARNING: this class should NOT be used directly, use the derived classes instead
    """
    def __init__(self, name, X, y, data_folder, 
                n_iter = 5, train_size = None, test_size = 0.2, random_state = 0):
        self.n_iter = n_iter
        self.train_size = train_size
        self.test_size = test_size
        self.random_state = random_state
        self.suffix = '_cv_%03d'
        self.datafiles = self.__persist_cv_splits(name, X, y, data_folder)
        ##TODO
    def __persist_cv_splits(self, name, X, y, data_folder):
        cv = ShuffleSplit(X.shape[0], n_iter = self.n_iter,
                                train_size = self.train_size, test_size = self.test_size,
                                random_state = self.random_state)
        named_data = {}
        for (i, (train, test)) in enumerate(cv):
            dataname = name + self.suffix % i
            data = (X[train], y[train], X[test], y[test])
            named_data[dataname] = data
        return dataio.persist_named_data(data_folder, named_data)
    


def test():
    from sklearn.datasets import load_digits
    digits = load_digits()
    X, y = digits.data, digits.target
    cvsearch = CVSearch('digits', X, y, '../tmp/')
    print cvsearch.datafiles
    print 'all tests passed ...'

if __name__ == '__main__':
    test()