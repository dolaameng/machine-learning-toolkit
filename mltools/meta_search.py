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
import os
from os import path

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
        ## do the persistence on hard disk
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

class GridSearch(CVSearch):
    def __init__(self, name, X, y, data_folder, 
                    n_iter = 5, train_size = None, test_size = 0.2, random_state = 0):
        super(GridSearch, self).__init__(name, X, y, data_folder, 
                                        n_iter, train_size, test_size, random_state)
    def search(self, model, param_grid):
        parameters = list(IterGrid(param_grid)) 

        tasks = {}
        for (iparam, params) in enumerate(parameters):
            for (icv, (name, datafile)) in enumerate(self.datafiles.items()):
                tasks[(iparam, icv)] = {'model': model, 'params': params, 'datafile': datafile}
        print tasks
        self.jobs = multicore.MulticoreJob().apply(CVSearch.evaluate_model_on_params, tasks)
        import time
        while not self.jobs.isready():
            print self.jobs.progress()
            time.sleep(2)
        return self.jobs.partial_result()

def test():
    from sklearn.datasets import load_digits
    digits = load_digits()
    X, y = digits.data, digits.target
    ## test cvsearch - WARNING: should NOT be used directly
    searcher = GridSearch('digits', X, y, '../tmp/')
    print searcher.datafiles
    from sklearn.svm import SVC
    print searcher.search(SVC(), {'C': [1, 10], 'gamma': [0.01, 0.1]})
    ## cv search 
    print 'all tests passed ...'

if __name__ == '__main__':
    test()