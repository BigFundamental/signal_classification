#!-*- encoding: utf-8 -*-
"""
author: zhou lin
date: 2017-01-13
brief: signal pattern extractors
"""

import pandas as pd
import numpy as np
import scipy.signal as signal
#from matplotlib import pyplot as plt
from filter import Filter

class FeatureExtractor(object):
    """
    Signal Feature Extractors
    """

    def __init__(self):
        pass
    
    def guassianNormalize(self, signals):
        """
        normalize signals into N(0, 1) guassian distributions
        return a new distribution of normalized signals
        """
        mean = np.mean(signals)
        std = np.std(signals)
        return (signals - mean) / std

    def peakPointers(self, signals, window_size, threshold, reverse=False):
        """
        locates & count all peak / bottom pointers
        return a tuple (num_of_peaks, array_of_peak_indices)
        """
        # first normalize signals
        # normalize all peak / lows ranges
        #norm_signals = self.guassianNormalize(signals)

        # med-filter to give away noises & sharp peaks
        med_filterd_signals = Filter.medfilter(signals, window_size)

        # caculate residual, only reserve candidate peaks
        peak_candidates = signals - med_filterd_signals
#        plt.plot(peak_candidates)
        # non-max-suppress, reserve one highest pointers for peaks
        peaks = Filter.nms(peak_candidates, window_size, reverse)
#        plt.plot(peaks)
        # threshold check, remove suspecious peaks
        idx = np.arange(0, len(signals))
        if not reverse:
            masks = peaks > threshold
        else:
            masks = peaks < threshold
        peak_idx = idx[masks]
        num_of_peaks = masks.sum()
        return peak_idx

    def _merge_adjecent_segments(self, edges, adj_thres = 3):
        """
        merge adjacent segments, because of the the step-wise detections
        """
        if len(edges) <= 1:
            return edges
        merged_edges = list()
        merged_edges.append(edges[0])
        for i in xrange(1, len(edges)):
            last_segment = merged_edges[-1]
            if edges[i][0] - last_segment[1] <= adj_thres:
                #consecutive edges, doing merge
                merged_edges[-1] = (last_segment[0], edges[i][1])
            else:
                merged_edges.append(edges[i])
            #print "merge %d: %s" % (i, str(merged_edges))
        return merged_edges

    def _consecutive_segments(self, segment, threshold, direction, thres_l = 0.05):
        """
        select minimal consecutive edges in range
        Assumption: there is one sharp edge exists in segment
        """
        if len(segment) <= 1:
            return None
        # check biggest margin in segment
        peak = np.argmax(segment)
        bottom = np.argmin(segment)
        segment_margin = segment[peak] - segment[bottom]
        #print "peak:%.2lf, bottom:%.2lf, margin:%.2lf" % (peak, bottom, segment_margin)
        #print "peak_idx:%d, peak: %f, bottom_idx:%d, bottom: %f, margin: %f" % (peak, segment[peak], bottom, segment[bottom], segment_margin)
        if segment_margin < threshold:
            return None
        # upwards margin
        if direction > 0 and peak <= bottom:
            return None
        if direction < 0 and peak >= bottom:
            return None

        #upwards edge
        if direction > 0:
            begin = 0
            end = peak
        else:
            begin = peak
            end = len(segment) - 1
        #print "consec, begin: %d end: %d" % (begin, end)
        # shrink boundaries
        # at least two points
        if direction > 0:
            while begin + 1 < end:
                margin = abs(segment[begin] - segment[begin + 1])
                if margin < thres_l:
                    begin = begin + 1
                else:
                    break
        else:
            # at least two points
            while begin + 1 < end and direction < 0:
                margin = abs(segment[end] - segment[end - 1])
                if margin < thres_l:
                    end = end - 1
                else:
                    break

        return (begin, end)

    def detectEdges(self, signals, window_size, threshold, direction):
        """
        direction: 1 upwards edges / -1: downwards edges
        """
        # using medfilter to remove all noise / peak points
        #norm_signals = self.guassianNormalize(signals)

        # medfilt to remove sharp changes, smooth data for later edge detection
        #med_signals = Filter.medfilter(norm_signals, 21)
        #return med_signals
        # calculate derivatives
        # dev_signals = self.1stDerivative(med_signals)

        # sliding-window detects points
        window_size = min(len(signals), window_size)
        i = 0
        end = len(signals) - 1
        #print "i: %d end: %d" % (i, end)
        edges = list()
        while i <= end:
            ret = self._consecutive_segments(signals[i:i + window_size], threshold, direction)
            #print "range(%d, %d), ret:%s" % (i, i + window_size - 1, str(ret))
            if ret != None and len(ret) == 2:
                edges.append((i + ret[0], i + ret[1]))
                #i += ret[1] + 1
                i += ret[1]
            else:
                i += 1
            #print ("[%d, %d]: ") % (i, i + window_size), ret
            #print edges
        #print "before merge: %s" % (str(edges))
        return self._merge_adjecent_segments(edges)

    def upwardsEdges(self, signals, window_size, threshold):
        """
        wrapper for upwards edges
        """
        # using medfilter to remove all noise / peak points
        #norm_signals = self.guassianNormalize(signals)
        # medfilt to remove sharp changes, smooth data for later edge detection
        med_signals = Filter.medfilter(signals, 9)
        #plt.plot(med_signals)
        return self.detectEdges(med_signals, window_size, threshold, 1)

    def downwardsEdges(self, signals, window_size, threshold):
        """
        wrapper for downwards edges
        """
        # using medfilter to remove all noise / peak points
        #norm_signals = self.guassianNormalize(signals)

        # medfilt to remove sharp changes, smooth data for later edge detection
        med_signals = Filter.medfilter(signals, 9)
        return self.detectEdges(med_signals, window_size, threshold, -1)

    def singleEdgeHeight(self, signals, edge):
        """
        get sigle edge' height
        """
        l_b = edge[0]
        r_b = edge[1]
        med_signals = Filter.medfilter(signals, 9)
        return abs(max(med_signals[l_b:r_b+1]) - min(med_signals[l_b:r_b+1]))

    def edgeHeight(self, signals, edge_loc):
        """
        calcuate edge absolute heights
        """
        height_list = list()
        for (l_b, r_b) in edge_loc:
            height_list.append(abs(signals[r_b] - signals[l_b]))
        return height_list

    def valley(self, signals):
        """
        return valley point in signals
        """
        return np.min(signals)
    def outlierPointNum(self, signals, threshold, cmp_func):
        """
        outlier numbers
        """
        ret_list = []
        for i in signals:
            if cmp_func(i, threshold):
                ret_list.append(i)
        return len(ret_list)
