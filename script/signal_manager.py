#!-*- encoding: utf8 -*-
"""
Author: Lin Zhou
Date: 2018-01-22
Major Entry of signal diagnosis module
"""
import os, sys
import numpy as np
import pandas as pd
from classifier import Classifier

class SignalMgr(object):
   """
   Signal process entry class
   """
   signalParams = {
           'PEAK_WINDOW_SIZE': 9,
           'PEAK_THRESHOLD': 0.2,
           'EDGE_WINDOW_SIZE': 7,
           'EDGE_THRESHOLD_HIGH': 0.2,
           'EDGE_THRESHOLD_LOW': 0.02,
           'PEAK_MISSING_RATIO': 0.25,
           'WITH_HEADER': True,
           'COLUMN_NUM': 1,
           'SAMPLING_DT': 0.00002
   }

   def process(self, file_path):
       """
       httpserver process callback entry function
       dict result for outer server
       """
       # step1: read raw signals
       dt, raw_signals = self.parse_signals_from_file(file_path)

       # step2: normalize input signals using guassian normalization
       #        easing later threshold variance between different channels
       normalized_signals = self.normalize_signals(raw_signals)

       # step3: using classifier to detects potential signals with pitfalls
       classifier = Classifier()
       return classifier.predict(normalized_signals, SignalMgr.signalParams)
   
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
       print "header_Lines:%d" % (header_lines) 
       column_index = range(0, SignalMgr.signalParams['COLUMN_NUM'])
       dt = np.array([])
       raw_signals = np.array([])
       if SignalMgr.signalParams['COLUMN_NUM'] == 1:
           raw_signals = np.genfromtxt(fpath, unpack=True, skip_header=header_lines, dtype=np.float32, delimiter=',', usecols=column_index)
           dt = np.arange(0, len(raw_signals * SignalMgr.signalParams['SAMPLING_DT']), SignalMgr.signalParams['SAMPLING_DT'])
       elif SignalMgr.signalParams['COLUMN_NUM'] == 2:
           dt, raw_signals = np.genfromtxt(fpath, unpack=True, skip_header=header_lines, dtype=np.float32, delimiter=',', usecols=column_index)
       return (dt, raw_signals)
