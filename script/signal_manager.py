#!-*- encoding: utf8 -*-
"""
Author: Lin Zhou
Date: 2018-01-22
Major Entry of signal diagnosis module
"""
import os, sys
import numpy as np
import pandas as pd
import logging
from classifier import Classifier

logger = logging.getLogger('server')

class SignalMgr(object):
   """
   Signal process entry class
   """
   signalParams = {
           'PEAK_WINDOW_SIZE': 9,
           'PEAK_THRESHOLD': 0.25,
           'PEAK_MISSING_RATIO': 0.2,
           'EDGE_WINDOW_SIZE': 7,
           'EDGE_THRESHOLD_HIGH': 0.2,
           'EDGE_THRESHOLD_LOW': 0.02,
           'SHOULDER_UNSYMANTRIC_RATIO': 0.25,
           'SHOULDER_HEIGHT_VARIANCE_THRESHOLD': 0.01,
           'SHOULDER_SYMMENTRIC_MEAN_THRESHOLD': 3,
           'SHOULDER_SYMMENTRIC_VARIANCE_THRESHOLD': 2.4,
           'WITH_HEADER': False,
           'COLUMN_NUM': 1,
           'SAMPLING_DT': 0.00002
   }

   def process(self, file_path, request_param = dict()):
       """
       httpserver process callback entry function
       dict result for outer server
       """
       # step1: read raw signals
       dt, raw_signals = self.parse_signals_from_file(file_path)
       logger.debug('dt:%s raw_signals:%s' % (str(dt), str(raw_signals)))
       # step2: normalize input signals using guassian normalization
       #        easing later threshold variance between different channels
       normalized_signals = self.normalize_signals(raw_signals)

       # step3: using classifier to detects potential signals with pitfalls
       classifier = Classifier()
       return classifier.predict(normalized_signals, SignalMgr.signalParams, request_param)
   
   def get_header_(self, fpath):
       """
       parse & extract headers
       """
       context_begin = False
       line_no = 0
       header = list()
       header_params = dict()
       with open(fpath, 'r') as fhandler:
           for line in fhandler:
               # extract header informations for later process
               if not context_begin:
                   if line.strip(', \n\r') != "":
                       header.append(line.strip(', \n\r'))
       #        if line.startswith("X_Value,AI Channel"):
               if line.startswith("0"):
                   context_begin = True
                   break
               line_no += 1
           
       # TODO: parse header information to dict()
       if not context_begin:
           line_no = 0
       header_params['lineno'] = line_no
       return header_params

   def normalize_signals(self, signals):
       """
       N(0, 1) normalization of input signals
       """
       mean = np.mean(signals)
       delta = np.std(signals)
       return (signals - mean) / delta
   
   def parse_signals_from_file(self, fpath):
       """
       parse signals from fhandler
       expect schema:
       dt, signal value
       """
       if SignalMgr.signalParams['WITH_HEADER']:
           header_params = self.get_header_(fpath)
           header_lines = header_params['lineno']
       else:
           header_lines = 0
       
       column_index = range(0, SignalMgr.signalParams['COLUMN_NUM'])
       dt = np.array([])
       raw_signals = np.array([])
       if SignalMgr.signalParams['COLUMN_NUM'] == 1:
           logger.debug(str(raw_signals))
           logger.debug(str(header_lines))
           logger.debug(str(column_index))
           logger.debug("path:[%s]" % (fpath))
           raw_signals = np.genfromtxt(fpath, unpack=True, skip_header=header_lines, dtype=np.float32, delimiter=',', usecols=column_index)
           logger.debug(str(raw_signals))
           dt = np.arange(0, len(raw_signals) * SignalMgr.signalParams['SAMPLING_DT'], SignalMgr.signalParams['SAMPLING_DT'])
       elif SignalMgr.signalParams['COLUMN_NUM'] == 2:
           dt, raw_signals = np.genfromtxt(fpath, unpack=True, skip_header=header_lines, dtype=np.float32, delimiter=',', usecols=column_index)
       return (dt, raw_signals)
