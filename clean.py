import numpy as np 
import pandas as pd 

import re
import warnings 

from config import TrainConfig, TestConfig


train = TrainConfig()
test = TestConfig()

configs = [train, test]

'''
for config in configs:
	x_df = config.X_DF
	y_df = config.Y_DF
	z_df = config.Z_DF

	print x_df.shape

	x_df.iloc[:,0:565] = x_df.iloc[:,0:565].astype(np.float32)
	y_df.iloc[:,0:565] = y_df.iloc[:,0:565].astype(np.float32)
	z_df.iloc[:,0:565] = z_df.iloc[:,0:565].astype(np.float32)

	x_df.to_csv(config.X_PATH + config.X_NAME)
	y_df.to_csv(config.Y_PATH + config.Y_NAME)
	z_df.to_csv(config.Z_PATH + config.Z_NAME)
'''

train_df = train.DF
test_df = test.DF

train_label_1 = [0]*train_df.shape[0]
test_label_1 = [0]*test_df.shape[0]

for index, row in train_df.iterrows():
	if row['ActivityID'] < 4:
		train_label_1[index] = 1
	else:
		train_label_1[index] = -1

train_df['Class_1'] = train_label_1
train_df.to_csv('HAR_train.csv')


for index, row in test_df.iterrows():
	if row['ActivityID'] < 4:
		test_label_1[index] = 1
	else:
		test_label_1[index] = -1

test_df['Class_1'] = test_label_1
test_df.to_csv('HAR_test.csv')

'''

