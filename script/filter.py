#!-*- encoding: utf-8 -*-
"""
author: zhou lin
date: 2017-01-13
brief: basic signal filters
"""

import pandas as pd
import numpy as np
import scipy.signal as signal

class Filter(object):
    """
    filter toolkits
    """

    def __init__(self):
        pass

    @classmethod
    def nms(self, signals, window_size = 5, minimun=False):
        """
        non max/minimun supress filter.
        only filter local maximun points
        window_size should be odd
        """
        if (window_size & 1) == 0:
            raise "window_size should be odd"
        step = max(1, (window_size - 1) / 2)
        for i in xrange(0, len(signals)):
            left = max(0, i - step)
            right = min(len(signals), i + step)
            target = np.argmax(signals[left:right]) + left
            if minimun == True:
                target = np.argmin(signals[left:right]) + left
            if target != i:
                signals[i] = 0
            
        return signals

    @classmethod
    def medfilter(self, signals, window_size = 5):
        """
        median filter to remove noises
        window_size should be odd
        """
        if (window_size & 1) == 0:
            raise "window_size should be odd"
        return signal.medfilt(signals, window_size)

