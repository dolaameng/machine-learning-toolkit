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
from scipy.stats import mode

##################### Helper functions to generate a library of #################
## machine learning methods

def persist_train_valid_split(X, y):
    ## TODO
    pass

def build_models(model, param_set, library_path, configure_file, 
                    train_X_path, train_y_path, 
                    valid_X_path, valid_y_path):
    """
    model: a model to fit - e.g. a sklearn model with fit(), predict(), predict_proba()
    param_set: set of parameters for model to try
    library_path: the path where all the pickle files stored 
    configure_file: the configuration files for results to be put in 
        - like a database, results will be appended
    """
    configurations = {}
    for params in param_set:
        model.set_params(**params)

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
            , 'classification': __classification_vote
        }
        SCORING_METHODS = {
            'regression': self.__regression_score ## same as SVR.score
            , 'probability': self.__probability_score,
            , 'classification': self.__classification_score
        }
        self.voting = VOTING_METHODS[voting]
        self.scoring = SCORING_METHODS[scoring]
        self.ensemble = ensemble
    def __regression_vote(self, ys):
        """each y in ys could be of shape (nrow, ntarget)
        """
        return sum(ys) * 1. / len(ys)1 
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
        y_pred_label = np.argmax(y_pred, axis = 0)
        return metrics.accuracy_score(y_true, y_pred_label)
    def __classification_score(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)
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