__author__ = 'diegoinsydo'

import numpy as np
import math


class Neuron(object):
    def __init__(self, weights, bias):
            self.weights = np.zeros([10,3072]),
            self.bias = np.ones([10])

    def forward(self, inputs):
        cell_body_sum = np.sum(inputs * self.weights) + self.bias
        firing_rate = 1.0 / (1.0 + math.exp(-cell_body_sum))
        return firing_rate
