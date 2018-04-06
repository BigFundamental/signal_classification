#! -*- encoding: utf8 -*-

import pandas as pd
import numpy as np
from filter import Filter
from feature_extractor import FeatureExtractor
import logging
import os
from sklearn.externals import joblib

logger = logging.getLogger('server')

class Classifier(object):
    """
    Basic Signal Classifier
    """
    NORMAL_TYPE = 0
    FLAW_TYPE_MISSING_PEAK = 1
    FLAW_TYPE_UNSYMMENTRIC_SHOULDER = 2
    FLAW_TYPE_HEIGHT_VARIANCE = 3
    FLAW_TYPE_WIDTH_VARIANCE = 4
    FLAW_TYPE_SPEED_INVALID = 5
    FLAW_TYPE_TWO_MANY_DOWN_PEAKS = 6
    ERR_RET_PARAM = dict({'stat': 1, 'reason': 0, 'speed': 0})

    def __init__(self, model_path=''):
        self.featureExtractor = FeatureExtractor()
        self.features = dict()
        if model_path != '':
            self.model = joblib.load(model_path)
        else:
            self.model = joblib.load(os.path.abspath('..') + os.sep + "model" + os.sep + "ada.pkl")

    def predict(self, signals, params, request_params = dict()):
        """
        return 0 if signal is normal, otherwise -1
        """
        #return self.predictWithReason(signals, params, request_params)
        return self.predictWithModel(signals, params, request_params)

    def predictWithModel(self, signals, params, request_params = dict()):
        f = self.get_features(signals, params, request_params)
        self.upwardsEdges = f['up_edges']
        self.downwardsEdges = f['down_edges']
        feature = np.array([f['peaks_num'], f['up_edges_num'], f['down_edges_num'], f['down_peaks_num'], f['peak_edge_ratio'], f['down_peak_edge_ratio']]).reshape(1, -1)
        result = int(self.model.predict(feature)[0])
        
        retParam = dict()
        retParam['stat'] = result
        retParam['reason'] = -1

        if result != 0:
            retParam['speed'] = 0
        else:
            # calculate speed
            samplerate = request_params.get('samplerate', [params['SAMPLING_DT']])[0]
            #samplerate = request_params.get('samplerate', params['SAMPLING_DT']) 
            retParam['speed'] = self.calcSpeed(signals, params, float(samplerate))
            if retParam['speed'] < 12300 or retParam['speed'] > 15500:
                retParam['stat']= 1
                retParam['reason'] = Classifier.FLAW_TYPE_SPEED_INVALID
        return retParam

    def get_features(self, signals, params, request_params = dict()):
        """
        calculate features dicts
        """
        signal_length = len(signals)
        feature_dict = dict()
        if 0 == signal_length:
            return feature_dict

        # get peaks / edges
        feature_dict['peaks'] = self.getPeakLoc_(signals, params)
        feature_dict['down_peaks'] = self.getDownPeakLoc_(signals, params)
        feature_dict['up_edges'] = self.getUpEdges_(signals, params)
        feature_dict['down_edges'] = self.getDownEdges_(signals, params)
        feature_dict['peaks_num'] = len(feature_dict['peaks'])
        feature_dict['down_peaks_num'] = len(feature_dict['down_peaks'])
        feature_dict['up_edges_num'] = len(feature_dict['up_edges'])
        feature_dict['down_edges_num'] = len(feature_dict['down_edges'])
        if feature_dict['up_edges_num'] + feature_dict['down_edges_num'] != 0:
            feature_dict['peak_edge_ratio'] = feature_dict['peaks_num'] * 1.0 / ((feature_dict['up_edges_num'] + feature_dict['down_edges_num']) / 2.0)
            feature_dict['down_peak_edge_ratio'] = feature_dict['down_peaks_num'] * 1.0 / ((feature_dict['up_edges_num'] + feature_dict['down_edges_num']) / 2.0)
        else:
            feature_dict['peak_edge_ratio'] = 0.0
            feature_dict['down_peak_edge_ratio'] = 0.0

        return feature_dict

    def predictWithReason(self, signals, params, request_params = dict()):
        """
        return a tupple consists of status and reasons
        """
        signal_length = len(signals)
        
        if signal_length == 0:
            return Classifier.ERR_RET_PARAM

        # get all peak points
        self.peakLocations = self.getPeakLoc_(signals, params)
        # get all down-peak points
        self.downPeakLocations = self.getDownPeakLoc_(signals, params)
        # get all upwards edges
        self.downwardsEdges = self.getDownEdges_(signals, params)
        # get all downwards edges
        self.upwardsEdges = self.getUpEdges_(signals, params)

        # start analysis
        (result, reason) = self.signalDiagnosis(signals, params)
        
        retParam = dict()
        retParam['stat'] = result
        retParam['reason'] = reason

        if result != 0:
            retParam['speed'] = 0
        else:
            # calculate speed
            logger.debug('request_params: %s' % str(request_params))
            samplerate = request_params.get('samplerate', [params['SAMPLING_DT']])[0]
            #samplerate = request_params.get('samplerate', params['SAMPLING_DT']) 
            retParam['speed'] = self.calcSpeed(signals, params, float(samplerate))
            if retParam['speed'] < 12300 or retParam['speed'] > 15500:
                retParam['stat']= 1
                retParam['reason'] = Classifier.FLAW_TYPE_SPEED_INVALID

        if request_params.get('debug', False):
            retParam['debug'] = dict()
            # adding noarmalized information
            retParam['debug']['normalized_signal'] = signals
            retParam['debug']['peaks'] = self.peakLocations
            retParam['debug']['up_edges'] = self.upwardsEdges
            retParam['debug']['down_edges'] = self.downwardsEdges
            retParam['debug']['down_peaks'] = self.downPeakLocations
            retParam['debug']['shoulder_height'] = self.shoulder_mean_heights
            #retParam['debug']['height_delta'] = self.edge_deltas
        return retParam

    def calcSpeed(self, signals, params, sampling_dt):
        """
        round per minute
        dt * edge_pairs
        """
        #sampling_dt = params['SAMPLING_DT'] # second
        total_secs = len(signals) * sampling_dt
        cycle_num = (len(self.upwardsEdges) + len(self.downwardsEdges)) / 2.0 + 1
        rpm = cycle_num / 4.0 / total_secs * 60.0
        #print "total_time:%.7lf cycle_num:%d" % (total_secs, cycle_num)
        return rpm

    #### get signal features ####
    def getPeakLoc_(self, signals, params):
        """
        return all peak pointer location within signals
        """
        _peak_window_size = params['PEAK_WINDOW_SIZE']
        _peak_threshold = params['PEAK_THRESHOLD']
        return self.featureExtractor.peakPointers(signals, _peak_window_size, _peak_threshold)
    
    def getDownPeakLoc_(self, signals, params):
        """
        return all extreme down-peak locations
        """
        _bottom_window_size = params['DOWN_PEAK_WINDOW_SIZE']
        _bottom_threshold = params['DOWN_PEAK_THESHOLD']
        return self.featureExtractor.peakPointers(signals, _bottom_window_size, _bottom_threshold, True)

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

    def getPairedEdges_(self, params):
        """
        group upwards edges & downwards edge list
        """
        paired_edges = list()
        up_idx = 0
        down_idx = 0
        logger.debug("upwardsEdges: %d downwardsEdges: %d" % (len(self.upwardsEdges), len(self.downwardsEdges)))
        while up_idx < len(self.upwardsEdges) and down_idx < len(self.downwardsEdges):
           up = self.upwardsEdges[up_idx]
           down = self.downwardsEdges[down_idx]
           #print "up_idx:%d down_idx:%d up:%s down:%s" % (up_idx, down_idx, str(up), str(down))
           if up[1] > down[0]:
               down_idx += 1
               continue
           paired_edges.append((up, down))
           up_idx += 1
           down_idx +=1
        return paired_edges

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
        
        if 0 == peak_num or 0 == up_edge_num or 0 == downward_edge_num or 0 == expect_num:
            return True
        missing_ratio = abs(expect_num - peak_num) * 1.0 / expect_num
        #print "peak_num:%d up_edge_num:%d downward_edge_num:%d expect_peak_num:%.2lf missing_ratio:%.2lf expect_missing_ratio:%.2f" % (peak_num, up_edge_num, downward_edge_num, expect_num, missing_ratio, _peak_missing_ratio)
        #print "peak_num:%d up_edge_num:%d downward_edge_num:%d peaks:%s" % (peak_num, up_edge_num, downward_edge_num, str(self.peakLocations))
        return missing_ratio >= _peak_missing_ratio
    
    def isTooManyDownPeaks(self, params):
        """
        detects normal peaks
        """
        _down_peak_appear_ratio = params['DOWN_PEAK_APPEARING_RATIO']
        down_peak_num = len(self.downPeakLocations)
        up_edge_num = len(self.upwardsEdges)
        downward_edge_num = len(self.downwardsEdges)
        if up_edge_num + downward_edge_num == 0:
            return True
        appearing_ratio = down_peak_num / ((up_edge_num + downward_edge_num) / 2.0)
        return appearing_ratio >= _down_peak_appear_ratio
    
    def isShoulderWidthAbnormal(self, params):
        """ 
        detect edge width
        """
        _shoulder_symmentric_mean_threshold = params['SHOULDER_SYMMENTRIC_MEAN_THRESHOLD']
        _shoulder_symmentric_variance_threshold = params['SHOULDER_SYMMENTRIC_VARIANCE_THRESHOLD']
        paired_edges = self.getPairedEdges_(params)
        widths = list()
        for (up, down) in paired_edges:
            up_width = abs(up[0] - up[1])
            down_width = abs(down[0] - down[1])
            width_diff = abs(up_width - down_width)
            widths.append(width_diff)
        global_width_mean = np.mean(widths)
        global_width_std = np.std(widths)
        
        logger.debug("width mean: %.2lf width variance: %.2lf" % (global_width_mean, global_width_std))
        if global_width_mean > _shoulder_symmentric_mean_threshold or global_width_std > _shoulder_symmentric_variance_threshold:
            return True 
        return False

    def isShoulderHeightAbnormal(self, signals, params):
        """
        detect shoulder height variances
        """
        _height_variances_error_threshold = params['SHOULDER_HEIGHT_VARIANCE_THRESHOLD']
        self.paired_edges = self.getPairedEdges_(params)
        heights = list()
        for (up, down) in self.paired_edges:
            mean_height = (signals[up[0]] + signals[down[1]]) / 2.0
            heights.append(mean_height)
        #norm_height = self.standardGuassianNormalize(height)
        global_height_mean = np.mean(heights)
        global_height_std = np.std(heights)
       
        self.shoulder_mean_heights = heights
        return global_height_std > _height_variances_error_threshold

    def standardGuassianNormalize(self, input_val):
        mean = np.mean(input_val)
        dev = np.std(input_val)
        return (input_val - mean) / dev

    def isShoulderNotSymmetric(self, signals, params):
        """
        check whether the diff of max(upwardsEdge) - max(downwardsEdge) is in appropriate margin
        """
        _unsymmetric_ratio = params['SHOULDER_UNSYMMETRIC_RATIO']
        _unsymmetric_threshold = params['SHOULDER_UNSYMMETRIC_THRESHOLD']
        _unsymmetric_var = params['SHOULDER_UNSYMMETRIC_VAR']
        paired_edges = self.getPairedEdges_(params)
        if len(paired_edges) == 0:
            return True
        invalid_cnt = 0
        edge_deltas = list()
        for (up, down) in paired_edges:
            l_height_idx = np.argmax(up)
            r_height_idx = np.argmax(down)
            delta = abs(signals[up[l_height_idx]] - signals[down[r_height_idx]])
            edge_deltas.append(delta)
        #    if delta >= _unsymmetric_threshold:
        #        invalid_cnt += 1

        #invalid_ratio = invalid_cnt * 1.0 / len(paired_edges)
        self.edge_deltas = edge_deltas
        return np.std(edge_deltas) >= _unsymmetric_var
    def signalDiagnosis(self, signals, params):
        """
        Rule assembled to classify & recognize signals
        """
        isFlawSignal = False
        flawType = Classifier.NORMAL_TYPE
        if self.isLackOfPeaks(params):
            isFlawSignal = True
            flawType = Classifier.FLAW_TYPE_MISSING_PEAK
        if self.isTooManyDownPeaks(params):
            isFlawSignal = True
            flawType = Classifier.FLAW_TYPE_TWO_MANY_DOWN_PEAKS
        if self.isShoulderHeightAbnormal(signals, params):
            isFlawSignal = True
            flawType = Classifier.FLAW_TYPE_HEIGHT_VARIANCE
       # if self.isShoulderNotSymmetric(signals, params):
       #     isFlawSignal = True
       #     flawType = Classifier.FLAW_TYPE_UNSYMMENTRIC_SHOULDER
        if isFlawSignal:
            result = 1
        else:
            result = 0
        return (result, flawType)
