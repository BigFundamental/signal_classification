#! -*- encoding: utf8 -*-

import pandas as pd
import numpy as np
from filter import Filter
from feature_extractor import FeatureExtractor
import logging

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
    ERR_RET_PARAM = dict({'stat': 1, 'reason': 0, 'speed': 0})

    def __init__(self):
        self.featureExtractor = FeatureExtractor()

    def predict(self, signals, params, request_params = dict()):
        """
        return 0 if signal is normal, otherwise -1
        """
        return self.predictWithReason(signals, params, request_params)

    def predictWithReason(self, signals, params, request_params = dict()):
        """
        return a tupple consists of status and reasons
        """
        signal_length = len(signals)
        
        if signal_length == 0:
            return Classifier.ERR_RET_PARAM

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

        if result != 0:
            retParam['speed'] = 0
        else:
            # calculate speed
            logger.debug('request_params: %s' % str(request_params))
            samplerate = request_params.get('samplerate', [params['SAMPLING_DT']])[0]
            #samplerate = request_params.get('samplerate', params['SAMPLING_DT']) 
            retParam['speed'] = self.calcSpeed(signals, params, float(samplerate))
        return retParam

    def calcSpeed(self, signals, params, sampling_dt):
        """
        round per minute
        dt * edge_pairs
        """
        #sampling_dt = params['SAMPLING_DT'] # second
        total_secs = len(signals) * sampling_dt
        cycle_num = (len(self.upwardsEdges) + len(self.downwardsEdges)) / 2.0
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
        #print "peak_num:%d up_edge_num:%d downward_edge_num:%d peaks:%s" % (peak_num, up_edge_num, downward_edge_num, str(self.peakLocations))
        return missing_ratio >= _peak_missing_ratio

    def isShoulderWidthAbnormal(self, params):
        """ 
        detect edge width
        """
        _shoulder_symmentric_mean_threshold = params['SHOULDER_SYMMENTRIC_MEAN_THRESHOLD']
        _shoulder_symmentric_variance_threshold = params['SHOULDER_SYMMENTRIC_VARIANCE_THRESHOLD']
        self.paired_edges = self.getPairedEdges_(params)
        widths = list()
        for (up, down) in self.paired_edges:
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

    def isShoulderHeightNormal(self, signals, params):
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
        
        delta_cnt = 0
        ddelta_cnt = 0
        tdelta_cnt = 0
        for height in heights:
            if abs(height) <= 1:
                delta_cnt += 1
            elif abs(height) <= 2:
                ddelta_cnt +=1
            else:
                tdelta_cnt +=1

        #print "height mean: %.2lf height variance: %.2lf" % (global_height_mean, global_height_std)
        #print "delta_cnt: %d ddelta_cnt: %d tdelta_cnt: %d" % (delta_cnt, ddelta_cnt, tdelta_cnt)
        return True

    def standardGuassianNormalize(self, input_val):
        mean = np.mean(input_val)
        dev = np.std(input_val)
        return (input_val - mean) / dev

    def isShoulderSymantric(self, params):
        """
        check whether the diff of max(upwardsEdge) - max(downwardsEdge) is in appropriate margin
        """
        _unsymmetric_ratio = params['SHOULDER_UNSYMANTRIC_RATIO']
        up_idx = 0
        down_idx = 0
        shoulder_diff = list()
        #print "upwardsEdges: %d downwardsEdges: %d" % (len(self.upwardsEdges), len(self.downwardsEdges))
        while up_idx < len(self.upwardsEdges) and down_idx < len(self.downwardsEdges):
           up = self.upwardsEdges[up_idx]
           down = self.downwardsEdges[down_idx]
           #print "up_idx:%d down_idx:%d up:%s down:%s" % (up_idx, down_idx, str(up), str(down))
           if up[1] > down[0]:
               down_idx += 1
               continue
           shoulder_diff.append(abs(np.max(up) - np.max(down)))
           up_idx += 1
           down_idx +=1
        logger.debug("shouder_diff: %s" % (str(shoulder_diff)))
        unsymmetric_cnt = 0
        
        #normalize shoulder diff
        #shoulder_diff = np.array(self.standardGuassianNormalize(shoulder_diff))
        #print "diff_array: %s mean: %.2lf dev: %.2lf" % (str(shoulder_diff), np.mean(shoulder_diff), np.std(shoulder_diff))
        mean = np.mean(shoulder_diff)
        delta = np.std(shoulder_diff)
        shoulder_diff = (np.array(shoulder_diff) - np.mean(shoulder_diff)) / np.std(shoulder_diff)

        delta = np.std(shoulder_diff)
        mean = np.mean(shoulder_diff)
        delta_within = 0
        delta_cnt = 0
        delta_double_cnt = 0
        delta_tripple_cnt = 0
        for abs_diff in shoulder_diff:
            diff = abs(abs_diff - mean)
            if diff > delta:
                delta_cnt += 1
            else:
                delta_within += 1

            if diff >= 2 * delta:
                delta_double_cnt += 1
            if diff >= 3 * delta:
                delta_tripple_cnt += 1

       # print "diff_array: %s delta_within: %d 1dev: %d 2dev: %d 3dev: %d" % (str(shoulder_diff), delta_within, delta_cnt, delta_double_cnt, delta_tripple_cnt)
       # unsymmentric_ratio = unsymmetric_cnt * 1.0 / ((len(self.upwardsEdges) + len(self.downwardsEdges)) / 2.0)
       # print "unsymmentric_ratio: %.5lf" % (unsymmentric_ratio)
       # if unsymmentric_ratio >= _unsymmetric_ratio:
       #     return False
        return True

    def signalDiagnosis(self, signals, params):
        """
        Rule assembled to classify & recognize signals
        """
        isFlawSignal = False
        flawType = Classifier.NORMAL_TYPE
        if self.isLackOfPeaks(params):
            isFlawSignal = True
            flawType = Classifier.FLAW_TYPE_MISSING_PEAK 
        elif self.isShoulderWidthAbnormal(params):
            isFlawSignal = True
            flawType = Classifier.FLAW_TYPE_WIDTH_VARIANCE

        if isFlawSignal:
            result = 1
        else:
            result = 0
        return (result, flawType)
