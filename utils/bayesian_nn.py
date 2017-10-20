import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal

def neural_network(x):
    h = tf.tanh(tf.matmul(x, W_0) + b_0)
    h = tf.tanh(tf.matmul(h, W_1) + b_1)
    h = tf.matmul(h, W_2) + b_2
    return tf.reshape(h, [-1])

class BayesianNN(object):

    def __init__(self, X, y, conf):
        self.X = X
        self.y = y
