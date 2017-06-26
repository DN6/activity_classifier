import os
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit

''' From:
https://mashimo.wordpress.com/2013/09/28/read-and-clean-data-with-python-pandas/
'''

DATA_PATH = "%s/%s/%s"%(os.path.dirname(os.getcwd()), 'data', 'UCI_HAR_Dataset')

activity = pd.read_table("%s/%s"%(DATA_PATH, 'activity_labels.txt') , sep=" ", header=None)
features = pd.read_table("%s/%s"%(DATA_PATH, 'features.txt'), sep=" ", header=None, names=('ID','Sensor'))

test_X = pd.read_table("%s/%s"%(DATA_PATH, 'test/X_test.txt'), sep='\s+', header=None)
test_Y = pd.read_table("%s/%s"%(DATA_PATH, 'test/y_test.txt'), sep='\s+', header=None, names=['ActivityID'])

train_X = pd.read_table("%s/%s"%(DATA_PATH, 'train/X_train.txt'), sep='\s+', header=None)
train_Y = pd.read_table("%s/%s"%(DATA_PATH, 'train/y_train.txt'), sep='\s+', header=None, names=['ActivityID'])

_all_X = pd.concat([train_X, test_X], ignore_index = True)
_all_Y = pd.concat([train_Y, test_Y], ignore_index = True)

_all_X.columns = features['Sensor']

_all_Y['Node_1'] = 1

for index, row in _all_Y.iterrows():
    if row['ActivityID'] > 3 :
        _all_Y['Node_1'][index] = -1

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
_index = sss.split(_all_X, _all_Y['Node_1'])


for train_index, test_index in sss.split(_all_X, _all_Y['Node_1']):
    train, test = _all_X.loc[train_index, :], _all_X.loc[test_index, :]
    label_train, label_test = _all_Y.loc[train_index, :], _all_Y.loc[test_index, :]

WRITE_PATH = "%s/%s/%s"%(os.path.dirname(os.getcwd()), 'data', 'tidy')

train.to_csv('%s/%s'%(WRITE_PATH,'tidy_train.csv'))
test.to_csv('%s/%s'%(WRITE_PATH, 'tidy_test.csv'))

label_train.to_csv('%s/%s'%(WRITE_PATH, 'tidy_label_train.csv'))
label_test.to_csv('%s/%s'%(WRITE_PATH, 'tidy_label_test.csv'))
