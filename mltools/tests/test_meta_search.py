
from .. import meta_search

from sklearn.datasets import load_digits
import os


def test_persist_cv_splits():
    """load digits and persist it in the ../data folder"""
    digits = load_digits()
    X, y = digits.data, digits.target
    print meta_search.persist_cv_splits('digits', X, y, data_folder = './tmp/')
    #assert False
    
