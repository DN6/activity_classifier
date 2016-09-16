import os

import numpy as np
import pandas as pd 

from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn2pmml import sklearn2pmml


class Config(object):

	PATH = os.getcwd() + '/data/'
	MODEL_PATH =  os.getcwd() + '/models/'


class TrainConfig(Config):

	def __init__(self):
		self.HAR_PATH = Config.PATH + 'HAR_train.csv'
		self.DF = pd.read_csv(self.HAR_PATH, sep=',', header=0, index_col=False)


class TestConfig(Config):

	def __init__(self):
		self.HAR_PATH = Config.PATH + 'HAR_test.csv'
		self.DF = pd.read_csv(self.HAR_PATH, sep=',', header=0, index_col=False)
	


class RawTrainConfig(object):

	NAME = 'HAR_train.csv'
	X_NAME = 'body_acc_x_train.csv'
	Y_NAME = 'body_acc_y_train.csv'
	Z_NAME = 'body_acc_z_train.csv'

	PATH = os.getcwd() + '/data/'
	#PATH = 'C:/Users/suresh/Documents/GitHub/Pipeline/data/'
	
	HAR_PATH = PATH + NAME
	DF = pd.read_csv(HAR_PATH, sep=',', header=0, index_col=False)
		
	X_PATH = PATH + X_NAME
	Y_PATH = PATH + Y_NAME
	Z_PATH = PATH + Z_NAME

	X_DF = pd.read_csv(X_PATH, sep=',', header=0, index_col=False)
	Y_DF = pd.read_csv(Y_PATH, sep=',', header=0, index_col=False)
	Z_DF = pd.read_csv(Z_PATH, sep=',', header=0, index_col=False)


class RawTestConfig(object):

	NAME = 'HAR_test.csv'
	X_NAME = 'body_acc_x_test.csv' 
	Y_NAME = 'body_acc_y_test.csv'
	Z_NAME = 'body_acc_z_test.csv'

	PATH = os.getcwd() + '/data/'
	#PATH = 'C:/Users/suresh/Documents/GitHub/Pipeline/data/'
	
	HAR_PATH = PATH + NAME
	DF = pd.read_csv(HAR_PATH, sep=',', header=0, index_col=False)

	X_PATH = PATH + X_NAME
	Y_PATH = PATH + Y_NAME
	Z_PATH = PATH + Z_NAME

	X_DF = pd.read_csv(X_PATH, sep=',', header=0, index_col=False)
	Y_DF = pd.read_csv(Y_PATH, sep=',', header=0, index_col=False)
	Z_DF = pd.read_csv(Z_PATH, sep=',', header=0, index_col=False)


	
train = TrainConfig()
print train.PATH