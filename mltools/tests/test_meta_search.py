
from .. import meta_search

from sklearn.datasets import load_digits
from sklearn.datasets import load_files
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import os


def test_random_search_cv_on_newsgroup():
    ## load news group data
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
    twenty_train_small = load_files('./data/20news-bydate-train/',
        categories=categories, charset='latin-1')
    twenty_test_small = load_files('./data/20news-bydate-test/',
        categories=categories, charset='latin-1')
    ## model pipeline using tfidf and passive aggresive
    pipeline = Pipeline((
        ('vec', TfidfVectorizer(min_df=1, max_df=0.8, use_idf=True)),
        ('clf', PassiveAggressiveClassifier(C=1)),
    ))
    param_grid = {
        'vec__min_df': [1, 2],
        'vec__max_df': [0.8, 1.0],
        'vec__ngram_range': [(1, 1), (1, 2)],
        'vec__use_idf': [True, False]
    }
    X, y = twenty_train_small.data, twenty_train_small.target
    ## cross validation on n_iter = 5
    rnd_searcher = meta_search.RandomSearch()
    # persist only once
    rnd_searcher.persist_cv_splits('text_classification', X, y, './tmp/')
    rnd_searcher.search(pipeline, param_grid)
    import time
    while not rnd_searcher.isready():
        print time.sleep(2)
        print 'progress:', rnd_searcher.progress()
        print 'best result:', rnd_searcher.best_params_so_far()
        if rnd_searcher.best_params_so_far():
            pass#rnd_searcher.abort()
    print len(rnd_searcher.partial_result())
    ## run again with naive bayesian
    ## no need to persist_cv_splits
    pipeline = Pipeline((
        ('vec', TfidfVectorizer(min_df=1, max_df=0.8, use_idf=True)),
        ('clf', MultinomialNB()),
    ))
    rnd_searcher10 = meta_search.RandomSearch(datafiles = rnd_searcher.datafiles)
    rnd_searcher10.search(pipeline, param_grid)
    while not rnd_searcher10.isready():
        print time.sleep(2)
        print 'progress:', rnd_searcher10.progress()
        print 'best result:', rnd_searcher10.best_params_so_far()
        if rnd_searcher10.best_params_so_far():
            pass#rnd_searcher10.abort()
    print len(rnd_searcher10.partial_result())

def test_grid_search_cv_on_newsgroup():
    ## load news group data
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
    twenty_train_small = load_files('./data/20news-bydate-train/',
        categories=categories, charset='latin-1')
    twenty_test_small = load_files('./data/20news-bydate-test/',
        categories=categories, charset='latin-1')
    ## model pipeline using tfidf and passive aggresive
    pipeline = Pipeline((
        ('vec', TfidfVectorizer(min_df=1, max_df=0.8, use_idf=True)),
        ('clf', PassiveAggressiveClassifier(C=1)),
    ))
    param_grid = {
        'vec__min_df': [1, 2],
        'vec__max_df': [0.8, 1.0],
        'vec__ngram_range': [(1, 1), (1, 2)],
        'vec__use_idf': [True, False]
    }
    X, y = twenty_train_small.data, twenty_train_small.target
    ## cross validation on n_iter = 5
    grid_searcher = meta_search.GridSearch()
    # persist only once
    grid_searcher.persist_cv_splits('text_classification', X, y, './tmp/')
    grid_searcher.search(pipeline, param_grid)
    import time
    while not grid_searcher.isready():
        print time.sleep(2)
        print 'progress:', grid_searcher.progress()
        print 'best result:', grid_searcher.best_params_so_far()
        if grid_searcher.best_params_so_far():
            pass#grid_searcher.abort()
    print len(grid_searcher.partial_result())
    ## run again with naive bayesian
    ## no need to persist_cv_splits
    pipeline = Pipeline((
        ('vec', TfidfVectorizer(min_df=1, max_df=0.8, use_idf=True)),
        ('clf', MultinomialNB()),
    ))
    grid_searcher10 = meta_search.GridSearch(datafiles = grid_searcher.datafiles)
    grid_searcher10.search(pipeline, param_grid)
    while not grid_searcher10.isready():
        print time.sleep(2)
        print 'progress:', grid_searcher10.progress()
        print 'best result:', grid_searcher10.best_params_so_far()
        if grid_searcher10.best_params_so_far():
            pass#grid_searcher10.abort()
    print len(grid_searcher10.partial_result())    
