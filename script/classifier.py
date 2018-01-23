#! -*- encoding: utf8 -*-

import pandas as pd
import numpy as np
from filter import Filter
from feature_extractor import FeatureExtractor

class Classifier(object):
    """
    Basic Signal Classifier
    """
    NORMAL_TYPE = 0
    FLAW_TYPE_MISSING_PEAK = 1
    FLAW_TYPE_UNSYMMENTRIC_SHOULDER = 2
    def __init__(self):
        self.featureExtractor = FeatureExtractor()

    def predict(self, signals, params):
        """
        return 0 if signal is normal, otherwise -1
        """
        return self.predictWithReason(signals, params)

    def predictWithReason(self, signals, params):
        """
        return a tupple consists of status and reasons
        """
        signal_length = len(signals)
       
        # get all peak points
        self.peakLocations = self.getPeakLoc_(signals, params)
        # get all upwards edges
        self.upwardsEdges = self.getUpEdges_(signals, params)

        # get all downwards edges
        self.downwardsEdges = self.getDownEdges_(signals, params)

        # start analysis
        (result, reason) = self.signalDiagnosis(signals, params)
        
        retParam = dict()
        retParam['stat'] = result
        retParam['reason'] = reason
        return retParam

    def calcRotationSpeed(self, signals, params):
        """
        return predict real speed of current signals
        """
        return 0

    #### get signal features ####
    def getPeakLoc_(self, signals, params):
        """
        return all peak pointer location within signals
        """
        _peak_window_size = params['PEAK_WINDOW_SIZE']
        _peak_threshold = params['PEAK_THRESHOLD']
        return self.featureExtractor.peakPointers(signals, _peak_window_size, _peak_threshold)
    
    def getUpEdges_(self, signals, params):
        """
        return [start, end] for upwards edges
        """
        _edge_window_size = params['EDGE_WINDOW_SIZE']
        _edge_threshold_H = params['EDGE_THRESHOLD_HIGH']
        _edge_threshold_L = params['EDGE_THRESHOLD_LOW']
        
        return self.featureExtractor.upwardsEdges(signals, _edge_window_size, _edge_threshold_H)

    def getDownEdges_(self, signals, params):
        """
        return [start, end] for downwards edges
        """
        
        _edge_window_size = params['EDGE_WINDOW_SIZE']
        _edge_threshold_H = params['EDGE_THRESHOLD_HIGH']
        _edge_threshold_L = params['EDGE_THRESHOLD_LOW']

        return self.featureExtractor.downwardsEdges(signals, _edge_window_size, _edge_threshold_H)
    
    # TODO: get signal shoulders
    def getShoulders(self, signals, params):
        """
        get shoulder informations
        """
    
    #### abnormal signal reasoning ####
    def isLackOfPeaks(self, params):
        """ 
        detects whether peak exists or missing
        """
        _peak_missing_ratio = params['PEAK_MISSING_RATIO']
        peak_num = len(self.peakLocations)
        up_edge_num = len(self.upwardsEdges)
        downward_edge_num = len(self.downwardsEdges)
        expect_num = (up_edge_num + downward_edge_num) / 2.0
        missing_ratio = abs(expect_num - peak_num) * 1.0 / expect_num
        #print "peakNum: %d upEdge: %d downEdge: %d expect_num: %d, missing_ratio: %.2lf" % (peak_num, up_edge_num, downward_edge_num, expect_num, missing_ratio)
        return missing_ratio >= _peak_missing_ratio
    
    def isEdgePeakCompatible(self):
        """
        detects edge number and different things
        """
        return

    def signalDiagnosis(self, signals, params):
        """
        Rule assembled to classify & recognize signals
        """
        isFlawSignal = False
        flawType = Classifier.NORMAL_TYPE
        if self.isLackOfPeaks(params):
            isFlawSignal = True
            flawType = Classifier.FLAW_TYPE_MISSING_PEAK 

        if isFlawSignal:
            result = 1
        else:
            result = 0
        return (result, flawType)
