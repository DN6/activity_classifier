import os
import datetime
import json
import numpy as np
import pandas as pd

import argparse
import optparse

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA


def get_config(path):
    with open(path) as f:
        return json.loads(f.read())

def load_data(path):
    return pd.read_csv(path, sep=',', header=0, index_col=False)

def load_model(path):
    return joblib.load(path)

def get_label(label_path, _class):
    df = pd.read_csv(label_path, sep=',', header=0, index_col=False)
    return df[_class]

def get_classifier(classifier):
    return {
        'rf' : RandomForestClassifier(n_estimators=10),
        'boosted_tree' : GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
        											max_depth=1, random_state=0),
        'logreg' : linear_model.LogisticRegression(C=1e5),
        'svm' : SVC(),
        'gp' : GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
    }.get(classifier)

def get_selector(selector, estimator):
    return {
        "rfe" : RFE(estimator, 5, step=1)
    }.get(selector)

def get_transformation(transformation):
    return {
        "pca" : PCA(n_components=20)
    }.get(transformation)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config')

    return parser.parse_args()

def get_run_id(conf):
    return '%s_%s'%(conf.get("class"), conf.get("classifier").get("id"))

def train(conf):
    run_id = get_run_id(conf)
    'Starting run id %s'%(run_id)

    data = load_data(conf.get("data"))
    y = get_label(conf.get("labels"), conf.get("class"))
    clf = get_classifier(conf.get("classifier").get("id"))

    if conf.get("transformation"):
        transformation = get_transformation(conf.get("transformation"))
        X = transformation.fit_transform(data)
        clf.fit(X,y)

        joblib.dump(clf, filename="%s/%s.pkl"%(conf.get("model_path"), run_id))

    if conf.get("feature_selection"):
        selector = get_selector(conf.get("feature_selection"), clf)
        selector.fit(data,y)

        joblib.dump(selector, filename="%s/%s.pkl"%(conf.get("model_path"), run_id))

def evaluate(conf):
    clf = load_model(conf.get("model_path"))
    data = load_data(conf.get("data"))

def main():
    args = get_args()
    conf = get_config(args.config)

    if conf.get("mode") == "train":
        train(conf)

    if conf.get("mode") == "evaluate":
        pass

if __name__=='__main__':
    main()
