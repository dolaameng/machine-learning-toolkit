from .. import ensemble

from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



def test_ensemble_on_iris():
    ## setup
    iris = load_iris()
    n_samples = iris.data.shape[0]
    def iris_iterable():
        from sklearn.datasets import load_iris
        iris = load_iris()
        X, y = iris.data, iris.target
        n_samples = X.shape[0]
        for i in xrange(n_samples):
            yield (X[i], y[i])        
    model = RandomForestClassifier()#RidgeClassifier()
    ensemble_model = ensemble.PartitionBasedEnsemble(model, ensemble.PartitionBasedEnsemble.combine_forest_models)
    ensemble_model.fit(n_samples, iris_iterable)
    print ensemble_model.score(iris.data, iris.target)
    print 'done...'