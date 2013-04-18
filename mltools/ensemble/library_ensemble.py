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


from sklearn.base import BaseEstimator
import numpy as np
import cPickle, json
from sklearn.external import joblib
from sklearn import metrics

class LibraryEnsemble(BaseEstimator):
    """
    """
    def __init__(self, voting, score, ensemble = None):
        """
        voting: voting method for ensemble, {'regression', 'probability', 'classification'},
            average: averaged value for regression
            probability: avearged posterior probability for classification
            classification: 0 / 1 average for binary classification (not accurate)

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
            'average':self.__average_vote
            , 'probability': self.__probability_vote
            , 'biclass': __biclass_vote
        }
        SCORE_METHODS = {
            'coefficient': self.__coefficient_score ## same as SVR.score
            , 'classification_rate': self.__classification_rate_score
        }
        self.voting = VOTING_METHODS[voting]
        self.score = SCORE_METHODS[score]
        self.ensemble = ensemble
    def __average_vote(self, ys):
        ##TODO
        ??return np.mean(np.vstack(ys), axis = 0)
    def __probability_vote(self, ys):
        ##TODO
        ??return np.mean(np.vstack(ys), axis = 0)
    def __biclass_vote(self, ys):
        """
        0/1 coding for binary classification
        """
        ##TODO
        ??return np.mean(np.vstack(ys), axis = 0) > 0.5
    def __coefficient_score(self, y_true, y_pred):
        ##TODO
        ??
        u = ((y_true - y_pred) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()
        return (1 - u/v)
    def __classification_rate_score(self, y_true, y_pred):
        ##TODO
        ??return metrics.accuracy_score(y_true, y_pred)
    def fit(self, library):
        """
        configs -- the same format as in self.ensemble
        select ensemble from library greedily to make the valid_score the highest
        """
        configs = json.loads(library)
        ## calcualte (model, valid_yhat) for each model
        modelname2score = {}
        for modelname in configs.keys():
            model = joblib.load(configs[modelname]['model'])
            valid_X = joblib.load(configs[modelname]['valid_X'])
            if hasattr(model, 'predict_proba'):
                valid_yhat = model.predict_proba(valid_X)
            else:
                valid_yhat = model.predict(valid_X)
            modelname2score[modelname] = valid_yhat
        valid_y = joblib.load(configs[modelname]['valid_y'])
        ## greedy search for ensemble
        ensemble, ensemble_score = [], 0.0
        candidates = set(modelname2score.keys())
        while candidates:
            next_candidate, next_score = max([(m, self.score(valid_y, self.voting(map(modelname2score.get, ensemble+[m])))) 
                                                for m in candidates if m not in ensemble] , 
                                            key = lambda (m, s): s)
            if next_score < ensemble_score:
                break
            else:
                ensemble_score = next_score
                ensemble.append(next_candidate)
        self.ensemble = {m:configs[m] for m in ensemble}
    def predict(self, test_config):
        pass