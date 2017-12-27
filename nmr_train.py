import os
import datetime
import json
import numpy as np
import pandas as pd

import argparse

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

from sklearn.externals import joblib

from utils import utils


def get_config(path):
    with open(path) as f:
        return json.loads(f.read())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')

    return parser.parse_args()


def train(conf, filename):
    path = "%s/%s" % (conf.get("data"), filename)

    data = utils.load_data(path)

    X = utils.get_features(data)
    y = data[conf.get("labels")]

    for param in conf.get("training").get("parameters"):
        run_id = utils.get_run_id(conf, param)
        print('Starting run id %s' % (run_id))

        pipeline = utils.make_pipeline(param)

        grid = GridSearchCV(pipeline, param.get("grid"))
        print("Starting Grid Search")
        grid.fit(X, y)

        model_path = "%s/%s" % (conf.get("model_path"), conf.get("name"))
        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        joblib.dump(grid.best_estimator_,
                    filename="%s/%s.pkl" % (model_path, run_id))


def main():
    args = get_args()
    conf = get_config(args.config)

    #train(conf)


if __name__ == '__main__':
    main()
