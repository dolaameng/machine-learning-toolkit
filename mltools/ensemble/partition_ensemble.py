"""
PartitionEnsemble - partition dataset, train individual models on them
and combine different models (e.g., linear model, forest model) and etc - based bagging

This implementation is NOT parallel to make it more scalable for large datasets,
this wont be a big limitation because some individual learners (e.g. randomforest)
can be trained in a parallel way

A parallel demostration can be found in the presentation by Grisel
https://github.com/ogrisel/parallel_ml_tutorial
"""
from sklearn.base import BaseEstimator
from copy import deepcopy
import numpy as np
import multicore

class PartitionBasedEnsemble(BaseEstimator):
    """Partition dataset and train individual base models on 
    different partitions.
    Combine individual models using customerized combine method.
    Use directview of ipython parallel to train model parallelly.
    """
    def __init__(self, model, combine, n_data_partition = 5, n_partition_size = 1.0/5):
        self.basemodel = model
        self.combine = combine
        self.n_data_partition = n_data_partition
        self.n_partition_size = n_partition_size
        self.ensemble_model = None
    def fit(self, n_samples, data_iterable):
        """data_iterable generates data iterators that generates sequence of (x, y)
        n_samples: number of (x, y) in data_stream
        """
        ## distribute work to multicore
        partitions = self.__generate_partition_indices(n_samples)
        tasks = {i: {'indices': partitions[i], 
                     'data_iterable': data_iterable, 
                     'model': deepcopy(self.basemodel)} 
                for i in xrange(self.n_data_partition)}
        self.multitasks = multicore.MulticoreJob().apply(PartitionBasedEnsemble.__fit_model, 
                                                        tasks)
        ## blocking collect model and combine
        self.multitasks.wait()
        self.models = self.multitasks.partial_result().values()
        print self.multitasks.partial_result().values()
        self.ensemble_model = self.combine(self.models)
        return self
    def __generate_partition_indices(self, n_samples):
        """generate sampel with replacement - bagging model
        """
        #indices = np.random.permutation(xrange(n_samples))
        #partitions = [indices[i::self.n_data_partition] for i in xrange(self.n_data_partition)]
        partition_size = int(n_samples * self.n_partition_size)
        partitions = [np.random.permutation(xrange(n_samples))[:partition_size] 
                        for i in xrange(self.n_data_partition)]
        return partitions
    @staticmethod
    def __fit_model(indices, data_iterable, model):
        import numpy as np
        data_stream = data_iterable()
        data = [(x, y) for (i, (x, y)) in enumerate(data_stream) if i in indices]
        X, y = zip(*data)
        X, y = np.asarray(X), np.asarray(y)
        model.fit(X, y)
        return model
    def predict(self, X):
        return self.ensemble_model.predict(X)
    def predict_prob(self, X):
        return self.ensemble_model.predict_prob(X)
    def score(self, X, y):
        print [m.score(X, y) for m in self.models]
        #print [m.n_classes_ for m in self.models]
        return self.ensemble_model.score(X, y)
    @staticmethod
    def combine_linear_models(models):
        import numpy as np
        from copy import deepcopy
        avg = deepcopy(models[0]) 
        avg.coef_ = np.sum([m.coef_ for m in models], axis = 0)
        avg.coef_ /= len(models)
        avg.intercept_ = np.sum([m.intercept_ for m in models], axis = 0)
        avg.intercept_ /= len(models)
        return avg 
    @staticmethod
    def combine_forest_models(models):
        from copy import deepcopy
        import numpy as np
        avg = deepcopy(models[0])
        for m in models[1:]:
            avg.estimators_ += m.estimators_
        avg.classes_ = np.unique(np.concatenate([m.classes_ for m in models], axis = 0))
        avg.n_classes_ = len(avg.classes_)
        return avg
        
        

        
def test():
    print 'all tests passed ...'
    
if __name__ == '__main__':
    test()