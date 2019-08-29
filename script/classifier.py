#! -*- encoding: utf8 -*-

import pandas as pd
import numpy as np
from filter import Filter
from feature_extractor import FeatureExtractor
import logging
import os
from model import ModelVersionFeatureConfig
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

    def __init__(self, model_path='', model_version=''):
        self.featureExtractor = FeatureExtractor()
        self.features = dict()
        self.wanted_features = []

        if model_version != '' and ModelVersionFeatureConfig.has_key(model_version):
            self.wanted_features = ModelVersionFeatureConfig[model_version]['features']
            model_path = os.path.sep.join([os.path.dirname(os.path.abspath(__file__)), '..', 'production', ModelVersionFeatureConfig[model_version]['path'],'model.pkl'])

        if model_path == 'train' or model_path == None:
            return
        if model_path != '':
            self.model = joblib.load(model_path)
        else:
            self.model = joblib.load(os.path.abspath('..') + os.sep + "model" + os.sep + "model.pkl")

    def normalize_signals(self, signals):
        """
        N(0, 1) normalization of input signals
        """
        mean = np.mean(signals)
        delta = np.std(signals)
        return (signals - mean) / delta

    def predict(self, signals, params, request_params = dict()):
        """
        return 0 if signal is normal, otherwise -1
        """
        if 'mode' in request_params.keys() and request_params['mode'] == 'speed':
            return self.predictSpeedOnly(signals, params, request_params)
        
        return self.predictWithModel(signals, params, request_params)

    def predictSpeedOnly(self, raw_signals, params, request_params = dict()):
        """
        predict signals' speed only
        """
        feature_masks = []
        f = self.get_speed_features(raw_signals, params)
        self.upwardsEdges = f['up_edges']
        self.downwardsEdges = f['down_edges']
        
        retParam = dict()
        retParam['stat'] = 0
        retParam['reason'] = -1
        retParam['speed'] = 0
        retParam['speedResult'] = 0
        retParam['waveResult'] = 0
        
        # calculate speed
        samplerate = request_params.get('samplerate', [params['SAMPLING_DT']])[0]
        retParam['speed'] = self.calcSpeed(raw_signals, params, float(samplerate))

        #judge speeds
        speed_lower_bound = int(request_params.get('speed_lower_bound', [params['SPEED_LOWER_BOUND']])[0])
        speed_upper_bound = int(request_params.get('speed_upper_bound', [params['SPEED_UPPER_BOUND']])[0])
        if retParam['speed'] < speed_lower_bound or retParam['speed'] > speed_upper_bound:
            retParam['speedResult']= 1
        
        if retParam['speedResult'] == 1:
            retParam['stat']= 1
            retParam['reason'] = Classifier.FLAW_TYPE_SPEED_INVALID
        return retParam

    def predictWithModel(self, raw_signals, params, request_params = dict()):
        feature_masks = self.wanted_features if len(self.wanted_features) > 0 else  None
        f = self.get_features(raw_signals[0:1024], params, request_params, feature_masks=feature_masks)
        feature = self.get_feature_vec(f)
        result = int(self.model.predict(feature)[0])
        
        retParam = dict()
        retParam['stat'] = result
        retParam['reason'] = -1
        retParam['speed'] = 0
        retParam['speedResult'] = 0
        retParam['waveResult'] = result

        speed_params = self.predictSpeedOnly(raw_signals, params, request_params)
        retParam['speed'] = speed_params['speed']
        retParam['speedResult'] = speed_params['speedResult']
       # calculate speed
       # samplerate = request_params.get('samplerate', [params['SAMPLING_DT']])[0]
       # #samplerate = request_params.get('samplerate', params['SAMPLING_DT']) 
       # retParam['speed'] = self.calcSpeed(raw_signals, params, float(samplerate))

       # #judge speeds
       # speed_lower_bound = int(request_params.get('speed_lower_bound', [params['SPEED_LOWER_BOUND']])[0])
       # speed_upper_bound = int(request_params.get('speed_upper_bound', [params['SPEED_UPPER_BOUND']])[0])
       # if retParam['speed'] < speed_lower_bound or retParam['speed'] > speed_upper_bound:
       #     retParam['speedResult']= 1
       # 
        if result == 0 and retParam['speedResult'] == 1:
            retParam['stat']= 1
            retParam['reason'] = Classifier.FLAW_TYPE_SPEED_INVALID
        return retParam
    
    def get_feature_vec(self, features):
        if len(self.wanted_features) > 0:
            feature_list = sorted(self.wanted_features)
        else:
            feature_list = sorted(self.get_feature_list())
        fea_vec = np.zeros(len(feature_list))
        for i in xrange(len(feature_list)):
            fea_vec[i] = features[feature_list[i]]
        return fea_vec.reshape(1, -1)

    def get_feature_list(self):
        """
        predefined feature lists, order is sensitive
        """
        return ['peaks_num', 'up_edges_num', 'down_edges_num', 'down_peaks_num', 'peak_edge_ratio', 'down_peak_edge_ratio', 'edge_diff_10', 'edge_diff_20', 'edge_diff_50', 'width_diff_10', 'negative_peak_num']
    
    def get_speed_features(self, raw_signals, params, enable_normalization=True):
        """
        shortened feature extractions
        """
        signals_length = len(raw_signals)
        feature_dict = dict()
        if enable_normalization:
            signals = self.normalize_signals(raw_signals)
        else:
            signals = raw_signals
        feature_dict['normalize_signals'] = signals
        feature_dict['up_edges'] = self.getUpEdges_(signals, params)
        feature_dict['down_edges'] = self.getDownEdges_(signals, params)
        return feature_dict

    def get_features(self, raw_signals, params, enable_normalization=True, request_params=dict(), feature_masks=None):
        """
        calculate features dicts
        """
        raw_signals = raw_signals[0:1024]
        signal_length = len(raw_signals)
        feature_dict = dict()
        if feature_masks != None:
            feature_masks = set(feature_masks)
        if 0 == signal_length:
            return feature_dict
        
        if enable_normalization:
            signals = self.normalize_signals(raw_signals[0:1024])
        else:
            signals = raw_signals[0:1024]
        # tracing & debuging features, for visualizations
        feature_dict['normalized_signals'] = signals
        # get peaks / edges, basic features:
        feature_dict['peaks'] = self.getPeakLoc_(signals, params)
        feature_dict['down_peaks'] = self.getDownPeakLoc_(signals, params)
        feature_dict['negative_peak_num'] = self.getNegativePeakNum_(raw_signals, params)
        feature_dict['max_down_peak_point'] = self.getExtremeDownPeakVal_(raw_signals, params)
        feature_dict['up_edges'] = self.getUpEdges_(signals, params)
        feature_dict['down_edges'] = self.getDownEdges_(signals, params)

        # combined features
        if not feature_masks or 'peaks_num' in feature_masks:
            feature_dict['peaks_num'] = len(feature_dict['peaks'])
        if not feature_masks or 'down_peaks_num' in feature_masks:
            feature_dict['down_peaks_num'] = len(feature_dict['down_peaks'])
        if not feature_masks or 'up_edges_num' in feature_masks:
            feature_dict['up_edges_num'] = len(feature_dict['up_edges'])
        if not feature_masks or 'down_edges_num' in feature_masks:
            feature_dict['down_edges_num'] = len(feature_dict['down_edges'])
        if (not feature_masks or ['up_edges_num', 'down_edges_num'] <= feature_masks) and feature_dict['up_edges_num'] + feature_dict['down_edges_num'] != 0:
            if not feature_masks or ['peak_edge_ratio', 'peaks_num'] <= feature_masks:
                feature_dict['peak_edge_ratio'] = feature_dict['peaks_num'] * 1.0 / ((feature_dict['up_edges_num'] + feature_dict['down_edges_num']) / 2.0)
            if not feature_masks or ['down_peak_edge_ratio', 'down_peaks_num'] <= feature_masks:
                feature_dict['down_peak_edge_ratio'] = feature_dict['down_peaks_num'] * 1.0 / ((feature_dict['up_edges_num'] + feature_dict['down_edges_num']) / 2.0)
        else:
            feature_dict['peak_edge_ratio'] = 0.0
            feature_dict['down_peak_edge_ratio'] = 0.0

        if not feature_masks or ['up_edges', 'up_edge_height'] <= feature_masks:
            feature_dict['up_edge_height'] = self.getEdgeHeight_(signals, feature_dict['up_edges'])
        if not feature_masks or ['down_edge_height', 'down_edges'] <= feature_masks:
            feature_dict['down_edge_height'] = self.getEdgeHeight_(signals, feature_dict['down_edges'])
        if not feature_masks or ['up_edges', 'down_edges'] <= feature_masks:
            feature_dict['paired_edges'] = self.getPairedEdges_(params, feature_dict['up_edges'], feature_dict['down_edges'])
        # 上下沿对比数据
        feature_dict['paired_edge_height'] = self.getPairedEdgeHeight_(signals, feature_dict['paired_edges'])
        feature_dict['paired_edge_height_diff'] = sorted(self.getPairedEdgeDifference_(feature_dict['paired_edge_height']), reverse=True)

        # 下拉及无头等缺陷的序列周期性检测
        feature_dict['cyclic_nopeak_seq'] = self.unitMaskGenerate(feature_dict['peaks'], feature_dict['paired_edges'], flip=True)
        feature_dict['cyclic_downpeak_seq'] = self.unitMaskGenerate(feature_dict['down_peaks'], feature_dict['paired_edges'])
        feature_dict['cyclic_intense_nopeak'] =  self.cyclicIntense(feature_dict['cyclic_nopeak_seq'], params['PHRASE_NUM'])
        feature_dict['cyclic_intense_downpeak'] = self.cyclicIntense(feature_dict['cyclic_downpeak_seq'], params['PHRASE_NUM'])

        # 获取波形单元间隔宽度数据
        feature_dict['unit_interviene_length'] = self.getIntervieneLength_(feature_dict['paired_edges'])
        feature_dict['unit_interviene_length_diff'] = self.getIntervieneLengthDifference_(feature_dict['unit_interviene_length'])
        if len(feature_dict['unit_interviene_length_diff']) > 0:
            feature_dict['inter_diff_mean'] = np.mean(feature_dict['unit_interviene_length_diff'])
            feature_dict['inter_diff_delta'] = np.std(feature_dict['unit_interviene_length_diff'])
        else:
            feature_dict['inter_diff_mean'] = 0
            feature_dict['inter_diff_delta'] = 0

        # 获取波形单元间隔底部的不对称性角度
        feature_dict['unit_interviene_skewness'] = self.getIntervieneSkewness_(signals, feature_dict['paired_edges'])
        if len(feature_dict['unit_interviene_skewness']) > 0:
            feature_dict['skewness_mean'] = np.mean(feature_dict['unit_interviene_skewness'])
            feature_dict['skewness_delta'] = np.std(feature_dict['unit_interviene_skewness'])
        else:
            feature_dict['skewness_mean'] = 0
            feature_dict['skewness_delta'] = 0

        # 获取上下沿边的长度diff分位数据
        if len(feature_dict['paired_edge_height_diff']) != 0:
            if not feature_masks or 'edge_diff_10' in feature_masks:
                feature_dict['edge_diff_10'] = np.percentile(feature_dict['paired_edge_height_diff'], 90)
            if not feature_masks or 'edge_diff_20' in feature_masks:
                feature_dict['edge_diff_20'] = np.percentile(feature_dict['paired_edge_height_diff'], 80)
            if not feature_masks or 'edge_diff_30' in feature_masks:
                feature_dict['edge_diff_30'] = np.percentile(feature_dict['paired_edge_height_diff'], 70)
            if not feature_masks or 'edge_diff_50' in feature_masks:
                feature_dict['edge_diff_50'] = np.percentile(feature_dict['paired_edge_height_diff'], 50)
        else:
            feature_dict['edge_diff_10'] = 100
            feature_dict['edge_diff_20'] = 100
            feature_dict['edge_diff_30'] = 100
            feature_dict['edge_diff_50'] = 100
            

        # 上下边缘对比数据
        feature_dict['paired_edge_width'] = self.getPairedEdgeUpperBottomWidth_(signals, feature_dict['paired_edges'])
        feature_dict['paired_edge_width_diff'] = sorted(self.getPairedWidthDifference_(feature_dict['paired_edge_width']), reverse=True)

        if len(feature_dict['paired_edge_width_diff']) != 0:
            if not feature_masks or 'width_diff_10' in feature_masks:
                feature_dict['width_diff_10'] = np.percentile(feature_dict['paired_edge_width_diff'], 90)
            if not feature_masks or 'width_diff_20' in feature_masks:
                feature_dict['width_diff_20'] = np.percentile(feature_dict['paired_edge_width_diff'], 80)
            if not feature_masks or 'width_diff_30' in feature_masks:
                feature_dict['width_diff_30'] = np.percentile(feature_dict['paired_edge_width_diff'], 70)
            if not feature_masks or 'width_diff_50' in feature_masks:
                feature_dict['width_diff_50'] = np.percentile(feature_dict['paired_edge_width_diff'], 50)
        else:
            feature_dict['width_diff_10'] = 100
            feature_dict['width_diff_20'] = 100
            feature_dict['width_diff_30'] = 100
            feature_dict['width_diff_50'] = 100

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
            #logger.debug('request_params: %s' % str(request_params))
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
        if total_secs == 0:
            rpm = 0
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

    def getNegativePeakNum_(self, raw_signals, params):
        """
        return total negative downpeak numbers
        NOTICE: input signals should be raw signals
        """
        return self.featureExtractor.outlierPointNum(raw_signals, 0, lambda x, y: x <= y)

    def getExtremeDownPeakVal_(self, raw_signals, params):
        """
        return extreme down peak values
        """
        return self.featureExtractor.valley(raw_signals)

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

    def getPairedEdgeHeight_(self, signals, up_down_edge_pairs):
        """
        get paired edge's height
        """
        up_down_height_paired_list = list()
        for (up_idx, down_idx) in up_down_edge_pairs:
            up_height = self.featureExtractor.singleEdgeHeight(signals, up_idx)
            down_height = self.featureExtractor.singleEdgeHeight(signals, down_idx)
            up_down_height_paired_list.append((up_height, down_height))
        return up_down_height_paired_list

    def getPairedEdgeUpperBottomWidth_(self, signals, up_down_edge_pairs):
        """
        get paired edges's upper/bottom width
        """
        up_down_width_paired_list = list()
        for (up_idx, down_idx) in up_down_edge_pairs:
            upper_width = abs(up_idx[1] - down_idx[0]) + 1
            bottom_width = abs(up_idx[0] - down_idx[1]) + 1
            up_down_width_paired_list.append((bottom_width, upper_width))
        return up_down_width_paired_list

    def getIntervieneLength_(self, up_down_edge_pairs):
        """
        one up & one down edge forms a single unit
        the length between them should have the same size
        """
        interviene_length_list = list()
        for i in range(1, len(up_down_edge_pairs)):
            prev_down_idx = up_down_edge_pairs[i - 1][1]
            cur_up_idx = up_down_edge_pairs[i][0]
            interviene_length_list.append(abs(prev_down_idx[1] - cur_up_idx[0]))
        return interviene_length_list

    def getIntervieneLengthDifference_(self, inter_length_list):
        """
        input is the difference lengths
        the interviene length distribution should be Guassian
        we can use Guassian normalization
        """
        differences_ = list()
        for i in range(1, len(inter_length_list)):
            differences_.append(abs(inter_length_list[i - 1] - inter_length_list[i]))
        
        return differences_

    def getIntervieneSkewness_(self, signals, up_down_edge_pairs):
        """
        Interates through all edges
        """
        interviene_skewness = list()
        for i in range(1, len(up_down_edge_pairs)):
            down = up_down_edge_pairs[i - 1][1]
            up = up_down_edge_pairs[i][0]
            bottom1 = min(signals[up[0]], signals[up[1]])
            bottom2 = min(signals[down[0]], signals[down[1]])
            interviene_width = abs(up[0] - down[1]) + 1
            interviene_skewness.append(abs(bottom1 - bottom2) * 1.0 / interviene_width)

       # for (up, down) in up_down_edge_pairs:
       #     bottom1 = min(signals[up[0]], signals[up[1]])
       #     bottom2 = min(signals[down[0]], signals[down[1]])
       #     interviene_width = abs(up[1] - down[0]) + 1
       #     interviene_skewness.append(abs(bottom1 - bottom2) * 1.0 / interviene_width)
        return interviene_skewness

    def getPairedEdgeDifference_(self, up_down_edge_height_paired_list):
        """
        given paired height list, scale difference to [0, 1]
        """
        up_down_height_diff = list()
        for (up_height, down_height) in up_down_edge_height_paired_list:
            base = max(up_height, down_height)
            diff = abs(up_height - down_height)
            assert(base != 0)
            up_down_height_diff.append(diff / base)
        return up_down_height_diff

    def getPairedWidthDifference_(self, up_down_edge_width_paired_list):
        """
        given paired width list, scale difference to [0, 1]
        """
        up_down_width_diff = list()
        for (up_width, down_width) in up_down_edge_width_paired_list:
            assert(down_width != 0)
            up_down_width_diff.append(abs(up_width - down_width) * 1.0 / down_width)
        return up_down_width_diff

    def getPairedEdges_(self, params, up_edges = None, down_edges = None):
        """
        group upwards edges & downwards edge list
        """
        paired_edges = list()
        up_idx = 0
        down_idx = 0
        upwardsEdges = None
        downwardsEdges = None
        if up_edges != None:
            upwardsEdges = up_edges
        else:
            upwardsEdges = self.upwardsEdges

        if down_edges != None:
            downwardsEdges = down_edges
        else:
            downwardsEdges = self.downwardsEdges
        #logger.debug("upwardsEdges: %d downwardsEdges: %d" % (len(self.upwardsEdges), len(self.downwardsEdges)))
        while up_idx < len(upwardsEdges) and down_idx < len(downwardsEdges):
           up = upwardsEdges[up_idx]
           down = downwardsEdges[down_idx]
           #print "up_idx:%d down_idx:%d up:%s down:%s" % (up_idx, down_idx, str(up), str(down))
           if up[1] > down[0]:
               down_idx += 1
               continue
           paired_edges.append((up, down))
           up_idx += 1
           down_idx +=1
        return paired_edges
    
    def getEdgeHeight_(self, signals, edge_loc):
        """
        caculate edge's absolute height
        """
        return self.featureExtractor.edgeHeight(signals, edge_loc)

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
        
        #logger.debug("width mean: %.2lf width variance: %.2lf" % (global_width_mean, global_width_std))
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

    def unitMaskGenerate(self, eventAxis, paired_edges, flip=False):
        """
        paired_edges divide signals into units
        foreach unit we will give each signals a label-1 if eventAxis appears
        if flip equals true, we will use 0 label for positive events
        return list of masks
        """
        unit_num = len(paired_edges)
        positive_label = 1
        negtive_label = 0
        if flip == True:
            positive_label = 0
            negtive_label = 1

        unit_num = len(paired_edges) - 1
        # initialize intial labels
        masks = np.full(unit_num, negtive_label)
        for i in range(0, unit_num):
            left = paired_edges[i][0][0]
            right = paired_edges[i + 1][0][0]
            for j in eventAxis:
                if j >= left and j < right:
                    masks[i] = positive_label
                    break

        return masks.tolist()
    
    def cyclicIntense(self, seqs, interval):
        cyclic_pair = 0
        max_cyclic_pairs = 0
        for i in range(0, interval):
            cyclic_pair = 0
            for j in range(i, len(seqs), interval):
                if seqs[j] > 0:
                    cyclic_pair += 1
            max_cyclic_pairs = max(max_cyclic_pairs, cyclic_pair)
        return max_cyclic_pairs

   # def cyclicIntense(self, seqs, interval):
   #     """
   #     using auto self-correlation to calculate cyclic informations
   #     """
   #     periodic_pairs = 0.0
   #     for i in range(0, len(seqs) - interval):
   #         if seqs[i] == 1 and seqs[i + interval] == 1:
   #             periodic_pairs += 1
   #     #s = pd.Series(seqs)
   #     #ret = s.autocorr(lag = interval)
   #     #if pd.isna(ret):
   #     #    ret = 0
   #     return periodic_pairs
