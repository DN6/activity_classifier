import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

""" Generate train/validation sets with adversairal validation

from:

https://github.com/zygmuntz/adversarial-validation/blob/master/numerai/sort_train.py

"""


def split(train, test, conf):
    """ Assign target labels to identify train/test data

    Returns: holdout_index
    """
    train['adv_target'] = 0
    test['adv_target'] = 1

    _test = test.drop(['id', 'era', 'data_type', 'target'], axis=1)
    assert(np.all(train.columns == _test.columns))

    """ Concatenate and shuffle data
    """
    data = pd.concat((train, _test))
    data = data.iloc[np.random.permutation(len(data))]

    data.reset_index(drop=True, inplace=True)

    X = data.drop(['adv_target'], axis=1)
    y = data['adv_target']

    predictions = np.zeros(y.shape)

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=5678)
    clf = RandomForestClassifier(n_estimators=100)

    for train_i, test_i in cv.split(X, y):
        X_train = X.iloc[train_i]
        X_test = X.iloc[test_i]

        y_train = y.iloc[train_i]
        y_test = y.iloc[test_i]

        clf.fit(X_train, y_train)
        p = clf.predict_proba(X_test)[:, 1]

        predictions[test_i] = p

    i = predictions.argsort()
    train_sorted = data.iloc[i]
    train_sorted = train_sorted.loc[train_sorted['adv_target'] == 0]

    assert(train_sorted.target.sum() == orig_train.target.sum())

    return train_sorted.index.values
