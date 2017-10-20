import numpy as np

class VotingClassifier(object):

    def __init__(self, estimators, weights, flag="hard"):
        self.estimators = estimators
        self.weights = weights
        self.flag = flag

    def predict(self):
        # Hard voting
        if self.flag=="hard":
            self.pred = np.asarray([est.predict(X) for est in self.estimators])
            self.pred = np.apply_along_axis(lambda x:
                            np.argmax(np.bincount(x, weights=self.weights)),
                            axis=1,
                            arr=predictions.astype('int'))
        # Soft voting
        else:
            self.pred = np.asarray([est.predict(X) for est in self.estimators])
            self.pred = np.average(self.pred, axis=0, weights=self.weights)
            self.pred = np.argmax(self.pred, axis=1)

        return self.pred

    def predict_proba(self, X):
        # Hard voting
        if self.flag=="hard":
            self.pred = np.asarray([est.predict_proba(X) for est in self.estimators])
            self.pred = np.apply_along_axis(lambda x:
                            np.argmax(np.bincount(x, weights=self.weights)),
                            axis=1,
                            arr=self.pred.astype('int'))
        # Soft voting
        else:
            self.pred = np.asarray([est.predict_proba(X) for est in self.estimators])
            self.pred = np.average(self.pred, axis=0, weights=self.weights)
            self.pred = np.argmax(self.pred, axis=1)

        return self.pred
