## IDEA based on the paper "ensemble selection from libraries of models"
## available at http://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf

## Steps:
## 1. build different models with different params, on different data subsets and feature subsets
## 2. serialize the model, and register it to the configuration.json with train, valid information
## 3. use the LibraryEnsemble class to "greedily" search for the models involved in ensemlbe - based on validation result
## 4. the built LibraryEnsemble model can be used to make new predictions by passing in another configuration param for 
## different test datasets used in each model

## configuration is currently supported by json and searizliation is supported by cPickle
## future implementation could involve database


__all__ = ['persist_train_valid_split', 'persist_subset_split', 
            'persist_models', 'LibraryEnsemble']

from sklearn.base import BaseEstimator
import numpy as np
import cPickle, json
from sklearn.externals import joblib
from sklearn import metrics
from sklearn import cross_validation
from scipy.stats import mode
import os

##################### Helper functions to generate a library of #################
## machine learning methods

def persist_train_valid_split(X_path, y_path,
                            train_X_path, valid_X_path,
                            train_y_path, valid_y_path, 
                            test_size=0.2, random_state=0):
    X, y = joblib.load(X_path), joblib.load(y_path)
    train_X, valid_X, train_y, valid_y = cross_validation.train_test_split(X, y, 
                                                test_size=test_size,
                                                random_state=random_state)
    joblib.dump(train_X, train_X_path)
    joblib.dump(valid_X, valid_X_path)
    joblib.dump(train_y, train_y_path)
    joblib.dump(valid_y, valid_y_path)

def persist_subset_split(X_path, y_path, X_sub_path, y_sub_path, suffix = '_%03d.pkl', n_iter=3):
    X, y = joblib.load(X_path), joblib.load(y_path)
    n_samples = X.shape[0]
    assert n_samples == y.shape[0], 'X, y must have the same shape for persist_subset_split'
    index = range(n_samples)
    np.random.shuffle(index)
    chunks = np.array_split(index, n_iter)
    for (i, chunk) in enumerate(chunks):
        sub_X, sub_y = X[chunk, :], y[chunk]
        joblib.dump(sub_X, X_sub_path + suffix % i)
        joblib.dump(sub_y, y_sub_path + suffix % i)



def persist_models(modelname, model, param_set, library_path, configure_file, 
                    train_X_path, train_y_path, 
                    valid_X_path, valid_y_path):
    """
    model: a model to fit - e.g. a sklearn model with fit(), predict(), predict_proba()
    param_set: set of parameters for model to try
    library_path: the path where all the pickle files stored 
    configure_file: the configuration files for results to be put in 
        - like a database, results will be appended
    """
    ## load data
    train_X, train_y = joblib.load(train_X_path), joblib.load(train_y_path)
    valid_X, valid_y = joblib.load(valid_X_path), joblib.load(valid_y_path)
    ## for each parameter setting
    configurations = {}
    for params in param_set:
        model.set_params(**params)
        fullname = modelname + '__' + '__'.join(['%s_%g' % (k, v) for (k,v) in params.items()])
        print 'training model ', fullname
        model.fit(train_X, train_y)
        train_score = model.score(train_X, train_y)
        valid_score = model.score(valid_X, valid_y)
        modelpath = os.path.abspath(os.path.join(library_path, fullname + '.pkl'))
        joblib.dump(model, modelpath)
        configurations[fullname] = {
              'model': modelpath
            , 'train_X': train_X_path
            , 'train_y': train_y_path
            , 'valid_X': valid_X_path
            , 'valid_y': valid_y_path
            , 'train_score': train_score
            , 'valid_score': valid_score
        }
    ## append the configuration into configuration file
    if os.path.exists(configure_file):
        conf_from_file = json.load(open(configure_file, 'rb'))
    else:
        conf_from_file = {}
    conf_from_file.update(configurations)
    json.dump(conf_from_file, open(configure_file, 'wb'))
    return conf_from_file

##################### LibraryEnsemble Class #####################################

class LibraryEnsemble(BaseEstimator):
    """
    """
    def __init__(self, voting, scoring, ensemble = None):
        """
        voting: voting method for ensemble, {'regression', 'probability', 'classification'},
            regression: averaged value for regression
            probability: avearged posterior probability for classification
            classification: label prediction for classification (not accurate)

        scoring: scoring method for ensemble, possible values are 
        {'regression', 'probability', 'classification'}
        regression: using the R^2 coefficients score
        probability: using the auc_score
        classification (ys are labels): using the classification rate score 

        ensemble: dict entries of {modelname: {
                                                    'model':model_pickle_file,
                                                    'train_X':train_X_file, 
                                                    'train_y':train_y_file,
                                                    'valid_X':valid_X_file,
                                                    'valid_y':valid_y_file, 
                                                    ...}}
        row_ids in valid_X, valid_y must be common for all models though train_X, train_y can be different
        e.g., train_X can be on different subset of data, different feature space of data
        if ensemble is None, it can be learned by fit on a library. 
        in this setting, it is easier to seralize the LibraryEnsemble model
        """
        VOTING_METHODS = {
            'regression':self.__regression_vote
            , 'probability': self.__probability_vote
            , 'classification': self.__classification_vote
        }
        SCORING_METHODS = {
            'regression': self.__regression_score ## same as SVR.score
            , 'probability': self.__probability_score
            , 'classification': self.__classification_score
        }
        self.voting = VOTING_METHODS[voting]
        self.scoring = SCORING_METHODS[scoring]
        self.ensemble = ensemble
    def __regression_vote(self, ys):
        """each y in ys could be of shape (nrow, ntarget)
        """
        return sum(ys) * 1. / len(ys)
    def __probability_vote(self, ys):
        return sum(ys) * 1. / len(ys)
    def __classification_vote(self, ys):
        """
        ys is the label of classes such as [1, 2, 3, 1]
        """
        y = np.vstack(ys)
        y_mode = mode(y)[0][0]
        return y_mode.astype(np.int) 
    def __regression_score(self, y_true, y_pred):
        u = ((y_true - y_pred) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()
        return (1 - u/v)
    def __probability_score(self, y_true, y_pred):
        ## use auc_score 
        y_pred_label = np.argmax(y_pred, axis = 1)
        #print y_true.shape, y_pred_label.shape, y_pred.shape
        return metrics.accuracy_score(y_true, y_pred_label)
    def __classification_score(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)
    def fit(self, library):
        """
        library -- file of same json structure of the same format as in self.ensemble
        select ensemble from library greedily to make the valid_score the highest
        """
        with open(library, 'rb') as f:
            configs = json.load(f)
        ## calcualte (model, valid_yhat) for each model
        modelname2yhat = {}
        for modelname in configs.keys():
            model = joblib.load(configs[modelname]['model'])
            valid_X = joblib.load(configs[modelname]['valid_X'])
            if hasattr(model, 'predict_proba'):
                valid_yhat = model.predict_proba(valid_X)
            else:
                valid_yhat = model.predict(valid_X)
            modelname2yhat[modelname] = valid_yhat
        valid_y = joblib.load(configs[modelname]['valid_y'])
        ## greedy search for ensemble
        ensemble, ensemble_score = [], 0.0
        candidates = set(modelname2yhat.keys())
        while candidates:
            next_candidate, next_score = max([(m, self.scoring(valid_y, 
                                                            self.voting(map(modelname2yhat.get, ensemble+[m])))) 
                                                for m in candidates if m not in ensemble] , 
                                            key = lambda (m, s): s)
            if next_score < ensemble_score:
                break
            else:
                ensemble_score = next_score
                ensemble.append(next_candidate)
        self.ensemble = {m:configs[m] for m in ensemble}
        #print ensemble_score
        #print self.ensemble.keys()
        #print len(self.ensemble)
    def predict(self, test_config_file):
        """
        test_config: json file of dictionary of {modelname: 
                            'test_X': test_X_file,
                            'test_y': test_y_file}
        'test_y' is not necessary for predict, but necessary for score
        """
        with open(test_config_file, 'rb') as f:
            configs = json.load(f)
        assert set(configs.keys()) == set(self.ensemble.keys()), 'model names must match in test_config and ensemlbe'
        modelname2yhat = {}
        for modelname in self.ensemlbe:
            model = joblib.load(self.ensemlbe[modelname]['model'])
            test_X = configs[modelname]['test_X']
            if hasattr(model, 'predict_proba'):
                test_yhat = model.predict_proba(test_X)
            else:
                test_yhat = model.predict(test_X)
            modelname2yhat[modelname] = test_yhat
        return self.voting(map(modelname2yhat.get, self.ensemble))


