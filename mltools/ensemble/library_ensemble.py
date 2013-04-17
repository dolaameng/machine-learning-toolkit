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

class LibraryEnsemble(BaseEstimator):
    """
    """
    def __init__(self, ensemble = None):
        """
        ensemble: list of dict entries of {modelname: [
                                                        model_pickle_file,
                                                        train_X_file, 
                                                        train_y_file,
                                                        valid_X_file,
                                                        valid_y_file, 
                                                        ...]}
        if ensemble is None, it can be learned by fit on a library. 
        in this setting, it is easier to seralize the LibraryEnsemble model
        """
        self.ensemble = ensemble
    def fit(self, library):
        pass