#! -*- encoding: utf8 -*-

import pandas as pd
import numpy as np
from filter import Filter
from feature_extractor import FeatureExtractor
import logging

logger = logging.getLogger('server')

class SimpleModel(object):
    """
    tradditional ML model for signal classify
    """
    
    def __init__(self):
        self.featureExtractor = FeatureExtractor()

    def predict(self, signal, params):
        """
        0 is good, otherwise 1
        binary prediction
        """
        predict_result = 0
        return predict_result

    def train(self, signals, labels):
        """
        signals is two dimensional signal arrays
        label is the expect result
        """
        # transform singals into features

        # split into train-set & eval-set

        # training & get training-eval & loss

        # evaluation
        return model

    def getFeatures(self, signals):
        """
        """
    
    def getFeature(self, signals):






