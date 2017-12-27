import os
import datetime
import json
import numpy as np
import pandas as pd

import argparse
import optparse

from operator import itemgetter

from sklearn.externals import joblib
from sklearn import linear_model
from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline


def get_config(path):
    with open(path) as f:
        return json.loads(f.read())


def load_data(path):
    return pd.read_csv(path, sep=',', header=0, index_col=False)


def load_models(path):
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            yield joblib.load("%s/%s" % (dirpath, filename))


def load_model(conf, model):
    run_id = '%s-%s-%s' % (conf.get("name"),
                           conf.get("model_no"),
                           model)
    path = "%s/%s/%s.pkl" % (conf.get("model_path"), conf.get("name"), run_id)
    return joblib.load(path)


def get_label(label_path, _class):
    df = pd.read_csv(label_path, sep=',', header=0, index_col=False)
    return df[_class]


def get_features(df):
    return df.filter(regex=("feature"))


def get_cv(spliter, n_splits):
    return {
        "StratifiedShuffleSplit": StratifiedShuffleSplit(n_splits=n_splits),
        "StratifiedKFold": StratifiedKFold(n_splits=n_splits)
    }.get(spliter)


def get_classifier(classifier):
    return {
        'rf': RandomForestClassifier(),
        'boosted_tree': GradientBoostingClassifier(),
        'logreg': linear_model.LogisticRegression(),
        'svm': SVC(),
        'gp': GaussianProcessClassifier(),
        'naive_bayes': GaussianNB(),
        'sgd':  SGDClassifier(),
        'knn': KNeighborsClassifier(),
        'mlp': MLPClassifier()

    }.get(classifier)


def get_selector(selector, estimator):
    return {
        "rfe": RFE(estimator, 5, step=1)
    }.get(selector)


def make_pipeline(conf):
    return Pipeline([
        ('classifier', get_classifier(conf.get("classifier")))
    ])


def get_run_id(conf, param):
    return '%s-%s-%s' % (conf.get("name"),
                         conf.get("model_no"), param.get("classifier"))


def preprocess(index, steps, features):
    def _preprocess(_features, step):
        return {
            "StandardScaler":  preprocessing.StandardScaler().fit_transform(_features),
            "MinMaxScaler": preprocessing.MinMaxScaler().fit_transform(_features),
            "Polynomial": preprocessing.PolynomialFeatures().fit_transform(_features),
            "none": None
        }.get(step)

    if index >= len(steps):
        return features

    else:
        _X = _preprocess(features, steps[index])
        index += 1
        return preprocess(index, steps, _X)
