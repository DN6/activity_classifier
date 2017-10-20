import os
import datetime
import json
import numpy as np
import pandas as pd

import argparse

from utils.voting_classifer import VotingClassifier
from utils import utils

from sklearn.externals import joblib

def get_config(path):
    with open(path) as f:
        return json.loads(f.read())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')

    return parser.parse_args()


def evaluate(conf):
    test_data = "%s/%s/%s"%(conf.get("data"),
                        conf.get("name"), conf.get("test"))

    data = utils.load_data(test_data)
    features = utils.get_features(data, conf.get("features"))

    X = utils.preprocess(0, conf.get('preprocess'), features)

    _flag = "soft"
    weights = [1, 1]

    estimators = [model for model in utils.load_models(conf.get("model_path"))]
    _vc = VotingClassifier(estimators=estimators, weights=weights, flag=_flag)

    predictions = pd.DataFrame()
    predictions['id'] = data['id']
    predictions['probability'] = _vc.predict_proba(X)

    predictions.to_csv("%s/%s-%s.csv"%("./predictions", conf.get("name"), _flag),
                        sep=',', index=False)

def main():

    args = get_args()
    conf = get_config(args.config)

    evaluate(conf)

if __name__ == '__main__':
    main()
