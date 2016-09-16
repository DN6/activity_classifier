import os
import numpy as np 
import pandas as pd
import sklearn_pandas

from sklearn.preprocessing import Imputer
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.externals import joblib


import logging
import datetime
import errno

from config import Config, TrainConfig, TestConfig, ClassifierConfig

config = Config()

features = [ 'tBody_AI', 'tBody_VI', 'tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 
			'tBodyAcc-mean()-Z', 'tBodyAcc-sma()' ]
label = ['Class_1']

mapper = sklearn_pandas.DataFrameMapper([
		(['AI', 'VI', 'Mean_X', 'Mean_Y', 'Mean_Z', 'SMA'], None)
])


columns = ['tBody_AI', 'tBody_VI', 'tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 
			'tBodyAcc-mean()-Z', 'tBodyAcc-sma()', 'ActivityID']

tree = tree.DecisionTreeClassifier()
boosted_tree = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
											max_depth=1, random_state=0)


forest = RandomForestClassifier(n_estimators=10)

joblib.dump(mapper, 'mapper.pkl')

def train():

	train = TrainConfig()
		
	df = train.DF
	df = shuffle(df[columns])

	temp = df[features]
	X = np.asarray(temp[temp['ActivityID' < 4]])
	#X = np.asarray(df[features])
	y = np.asarray(df[label])

	boosted_tree.fit(X,y.ravel())
	joblib.dump(boosted_tree,  config.MODEL_PATH + 'class_2_boosted_tree.pkl', compress=9)



def test():

	config = TestConfig()
	
	df = config.DF  
	df = shuffle(df[columns])

	X = np.asarray(df[features])
	y = np.asarray(df[label])

	imp = Imputer(missing_values='NaN', strategy='median', axis=0)
	X = imp.fit_transform(X)

	clf = joblib.load(config.MODEL_PATH + 'boosted_tree.pkl')
	score = clf.score(X, y.ravel())
	print score



if __name__=='__main__':
	pass
	#train()
	#test()