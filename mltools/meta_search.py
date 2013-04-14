"""
Multi-core support for meta-parameters search of
machine learning models based on cross validation

use command to start several multicore support, e.g.
$ ipcluster start -n 4 

the implementation is inspired by Grisel's presentation
https://github.com/ogrisel/parallel_ml_tutorial
"""
__all__ = []

import multicore
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import IterGrid
import os, random
from os import path
import numpy as np

class CVSearch(object):
    """search of meta parameters based on cross validation
    WARNING: this class should NOT be used directly, use the derived classes instead
    """
    def __init__(self, datafiles = None,
                n_iter = 5, train_size = None, test_size = 0.2, random_state = 0):
        self.n_iter = n_iter
        self.train_size = train_size
        self.test_size = test_size
        self.random_state = random_state
        self.suffix = '_cv_%03d'
        ## do the persistence on hard disk if datafiles not passed in
        ## if datafiles, then no need to call persist_cv_splits again
        if datafiles:
            self.datafiles = datafiles
    def persist_cv_splits(self, name, X, y, data_folder):
        print 'persisting cv data ', name, 'in folder', data_folder
        self.datafiles = self.__persist_cv_splits(name, X, y, data_folder)
        return self
    def __persist_cv_splits(self, name, X, y, data_folder):
        if hasattr(X, 'shape'):
            n_samples = X.shape[0]
        else: 
            n_samples = len(X) ## for text data
            X = np.array(X)
        cv = ShuffleSplit(n_samples, n_iter = self.n_iter,
                                train_size = self.train_size, test_size = self.test_size,
                                random_state = self.random_state)
        named_data = {}
        for (i, (train, test)) in enumerate(cv):
            dataname = name + self.suffix % i
            data = (X[train], y[train], X[test], y[test])
            named_data[dataname] = data
        return self.__persist_named_data(data_folder, named_data)
    def __persist_named_data(self, folder, named_data):
        """usually used in one process (machine) only
        """
        from sklearn.externals import joblib
        data_files = {}
        for (name, data) in named_data.items():
            data_file = path.abspath(path.join(folder, name + '.pkl'))
            joblib.dump(data, data_file)
            data_files[name] = data_file
        return data_files
    @staticmethod
    def evaluate_model_on_params(model, params, datafile):
        """Evaluate  a certain model on a combination 
        of parameter, train_test_data 
        """
        from sklearn.externals import joblib
        X_train, y_train, X_test, y_test = joblib.load(datafile)
        model.set_params(**params)
        model.fit(X_train, y_train)
        validation_score = model.score(X_test, y_test)
        return validation_score
    def isready(self):
        return self.workers and self.workers.isready()
    def progress(self):
        return self.workers.progress() if self.workers else 1.0
    def best_params_so_far(self):
        presult = self.partial_result()
        return presult[0] if presult else None  
    def abort(self):
        print 'abort meta search task'
        self.workers.abort()
        return self

class GridSearch(CVSearch):
    def __init__(self,  datafiles = None,
                    n_iter = 5, train_size = None, test_size = 0.2, random_state = 0):
        super(GridSearch, self).__init__(datafiles, 
                                        n_iter, train_size, test_size, random_state)
        self.jobs = None
    def search(self, model, param_grid):
        self.parameters = list(IterGrid(param_grid)) 

        tasks = {}
        for (iparam, params) in enumerate(self.parameters):
            for (icv, (name, datafile)) in enumerate(self.datafiles.items()):
                tasks[(iparam, icv)] = {'model': model, 'params': params, 'datafile': datafile}
        self.workers = multicore.MulticoreJob().apply(CVSearch.evaluate_model_on_params, tasks)
        return self
    def partial_result(self):
        if not self.workers: return None
        results = self.workers.partial_result()
        partial_param_scores = [(params, np.mean([results[(iparam, icv)] for icv in xrange(self.n_iter)])) 
                for (iparam,params) in enumerate(self.parameters)
                if all([(iparam, icv) in results for icv in xrange(self.n_iter)])]
        return sorted(partial_param_scores, key = lambda (p, s): s, reverse = True)

class RandomSearch(CVSearch):
    def __init__(self,  datafiles = None,
                    n_iter = 5, train_size = None, test_size = 0.2, random_state = 0):
        super(RandomSearch, self).__init__(datafiles, 
                                                n_iter, train_size, test_size, random_state)
        self.jobs = None
    def search(self, model, param_grid):
        self.parameters = list(IterGrid(param_grid))
        random.shuffle(self.parameters) 

        tasks = {}
        for (iparam, params) in enumerate(self.parameters):
            for (icv, (name, datafile)) in enumerate(self.datafiles.items()):
                tasks[(iparam, icv)] = {'model': model, 'params': params, 'datafile': datafile}
        self.workers = multicore.MulticoreJob().apply(CVSearch.evaluate_model_on_params, tasks)
        return self
    def partial_result(self):
        if not self.workers: return None
        results = self.workers.partial_result()
        partial_param_scores = [(params, np.mean([results[(iparam, icv)] for icv in xrange(self.n_iter)
                                                                        if (iparam, icv) in results])) 
                for (iparam,params) in enumerate(self.parameters)
                if any([(iparam, icv) in results for icv in xrange(self.n_iter)])]
        return sorted(partial_param_scores, key = lambda (p, s): s, reverse = True)


def test():
    from sklearn.datasets import load_digits
    digits = load_digits()
    X, y = digits.data, digits.target
    ## test gridsearch - WARNING: should NOT be used directly
    searcher = GridSearch('digits', X, y, '../tmp/', n_iter = 10)
    print searcher.datafiles
    from sklearn.svm import SVC
    searcher.search(SVC(), {'C': np.logspace(-1, 2, 4), 'gamma': np.logspace(-4, 0, 5)})
    import time
    while not searcher.isready():
        print time.sleep(2)
        print 'progress:', searcher.progress()
        print 'best result:', searcher.best_params_so_far()
        if searcher.best_params_so_far():
            pass#searcher.abort()
    print len(searcher.partial_result())
    ## random search 
    rsearcher = RandomSearch('digits', X, y, '../tmp/', n_iter = 10)
    rsearcher.search(SVC(), {'C': np.logspace(-1, 2, 4), 'gamma': np.logspace(-4, 0, 5)})
    while not rsearcher.isready():
        print time.sleep(2)
        print 'progress:', rsearcher.progress()
        print 'best result:', rsearcher.best_params_so_far()
        if rsearcher.best_params_so_far():
            pass#rsearcher.abort()
    print rsearcher.partial_result()
    print 'all tests passed ...'

if __name__ == '__main__':
    test()