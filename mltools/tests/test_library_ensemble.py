from ..ensemble.library_ensemble import *

from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.grid_search import IterGrid
import numpy as np

def test_full_data_training():
	retrain = True
	
	## constants
	dataset = 'digits'#'mnist' #'digits'
	X_path = 'tmp/%s/data/X.pkl' % dataset
	y_path = 'tmp/%s/data/y.pkl' % dataset
	train_X_path = 'tmp/%s/data/train_X.pkl' % dataset
	valid_X_path = 'tmp/%s/data/valid_X.pkl' % dataset
	train_y_path = 'tmp/%s/data/train_y.pkl' % dataset
	valid_y_path = 'tmp/%s/data/valid_y.pkl' % dataset
	library_path =  'tmp/%s/models/' % dataset
	configure_file = 'tmp/%s/models/library.conf' % dataset

	if retrain:
		## persist data set
		#### generate X, y file
		digits = load_digits()
		X, y = digits.data, digits.target
		#mnist = fetch_mldata('MNIST original', data_home = 'tmp')
		#X, y = mnist.data, mnist.target
		joblib.dump(X, X_path)
		joblib.dump(y, y_path)
		#### split data into train and valid
		persist_train_valid_split(X_path, y_path, train_X_path, valid_X_path,
	    							train_y_path, valid_y_path)
		
		
		## persist models
		svc = SVC(probability = True)
		param_set = list(IterGrid({
			  'C': np.logspace(-3, 0, 4)
			, 'gamma': np.logspace(-3, -1, 3)
		}))
		persist_models('svc', svc, param_set, library_path, configure_file, 
	                    train_X_path, train_y_path, 
	                    valid_X_path, valid_y_path)

	## build ensemble
	ensemble = LibraryEnsemble(voting = 'probability', scoring = 'probability')
	ensemble.fit(configure_file)
	##TODO test prediction
	
	print 'all tests passed...'


def test_partition_data_training():
	retrain = True
	## constants
	dataset = 'digits' #'mnist'
	X_path = 'tmp/%s/data/X.pkl' % dataset
	y_path = 'tmp/%s/data/y.pkl' % dataset
	#X1_path = 'tmp/%s/data/X1.pkl' % dataset
	#X2_path = 'tmp/%s/data/X2.pkl' % dataset
	#y1_path = 'tmp/%s/data/y1.pkl' % dataset
	#y2_path = 'tmp/%s/data/y2.pkl' % dataset
	train_X_path = 'tmp/%s/data/train_X.pkl' % dataset
	valid_X_path = 'tmp/%s/data/valid_X.pkl' % dataset
	train_y_path = 'tmp/%s/data/train_y.pkl' % dataset
	valid_y_path = 'tmp/%s/data/valid_y.pkl' % dataset

	train1_X_path = 'tmp/%s/data/train_X_000.pkl' % dataset
	train2_X_path = 'tmp/%s/data/train_X_001.pkl' % dataset
	train1_y_path = 'tmp/%s/data/train_y_000.pkl' % dataset
	train2_y_path = 'tmp/%s/data/train_y_001.pkl' % dataset

	valid_X_path = 'tmp/%s/data/valid_X.pkl' % dataset
	valid_y_path = 'tmp/%s/data/valid_y.pkl' % dataset
	library_path =  'tmp/%s/models/' % dataset
	configure_file = 'tmp/%s/models/library.json' % dataset

	if retrain:
		## persist data set
		#### generate X, y file
		digits = load_digits()
		X, y = digits.data, digits.target

		#mnist = fetch_mldata('MNIST original', data_home = 'tmp')
		#X, y = mnist.data, mnist.target
		joblib.dump(X, X_path)
		joblib.dump(y, y_path)
		#### split data into train and valid
		persist_train_valid_split(X_path, y_path, train_X_path, valid_X_path,
	    							train_y_path, valid_y_path)
		#### split train data into 2 subsets
		persist_subset_split(train_X_path, train_y_path, 
					'tmp/%s/data/train_X' % dataset, 
					'tmp/%s/data/train_y' % dataset, n_iter=2)
		
		
		## persist models
		svc = SVC(probability = True)
		param_set = list(IterGrid({
			  'C': np.logspace(-3, 0, 4)
			, 'gamma': np.logspace(-3, -1, 3)
		}))
		persist_models('svc000', svc, param_set, library_path, configure_file, 
	                    train1_X_path, train1_y_path, 
	                    valid_X_path, valid_y_path)
		persist_models('svc001', svc, param_set, library_path, configure_file, 
	                    train2_X_path, train2_y_path, 
	                    valid_X_path, valid_y_path)

	## build ensemble
	ensemble = LibraryEnsemble(voting = 'probability', scoring = 'probability')
	ensemble.fit(configure_file)

	##TODO test prediction
	print 'all tests passed...'
