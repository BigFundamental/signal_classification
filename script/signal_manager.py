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
           'PEAK_THRESHOLD': 0.5,
           'PEAK_MISSING_RATIO': 0.25,
           'EDGE_WINDOW_SIZE': 7,
           'EDGE_THRESHOLD_HIGH': 0.4,
           'EDGE_THRESHOLD_LOW': 0.05,
           'SHOULDER_UNSYMMETRIC_RATIO': 0.25,
           'SHOULDER_UNSYMMETRIC_VAR':0.42,
           'SHOULDER_UNSYMMETRIC_THRESHOLD': 0.01,
           'SHOULDER_HEIGHT_VARIANCE_THRESHOLD': 0.5,
           'SHOULDER_SYMMENTRIC_MEAN_THRESHOLD': 3,
           'SHOULDER_SYMMENTRIC_VARIANCE_THRESHOLD': 2.8,
           'DOWN_PEAK_APPEARING_RATIO': 0.25,
           'DOWN_PEAK_WINDOW_SIZE': 9,
           'DOWN_PEAK_THESHOLD': -1,
           'WITH_HEADER': False,
           'COLUMN_NUM': 1,
           'SAMPLING_DT': 0.00004,
           'SPEED_LOWER_BOUND':12300,
           'SPEED_UPPER_BOUND':15500,
           'DEFAULT_MODEL_VERSION':'gbdt_0830_BA'
   }

   def __init__(self):
       self.debug_info = dict()

   def get_features(self, file_path, request_param = dict()):
       dt, raw_signals = self.parse_signals_from_file(file_path, int(request_param.get('skip_row', [0])[0]))

       if request_param.has_key('model_path'):
           classifier = Classifier(model_path = request_param['model_path'][0])
       else:
           classifier = Classifier(model_version=SignalMgr.signalParams['DEFAULT_MODEL_VERSION'])
       return classifier.get_features(raw_signals, SignalMgr.signalParams, request_param)

   def process(self, file_path, request_param = dict()):
       """
       httpserver process callback entry function
       dict result for outer server
       """
       # step1: read raw signals
       dt, raw_signals = self.parse_signals_from_file(file_path,int(request_param.get('skip_row', [0])[0]))
       #logger.debug('dt:%s raw_signals:%s' % (str(dt), str(raw_signals)))
       # step2: normalize input signals using guassian normalization
       #        easing later threshold variance between different channels
       #normalized_signals = self.normalize_signals(raw_signals)
       # step3: using classifier to detects potential signals with pitfalls
       if request_param.has_key('mode') and request_param['mode'] == 'speed':
           # speed mode, no need to load model, specify model_path to None
           request_param['model_path'] = [None]
       
       if request_param.has_key('model_version'):
           classifier = Classifier(model_version = request_param['model_version'])
       elif request_param.has_key('model_path'):
           classifier = Classifier(model_path = request_param['model_path'][0])
       else:
           classifier = Classifier(model_version=SignalMgr.signalParams['DEFAULT_MODEL_VERSION'])
       
       return classifier.predict(raw_signals, SignalMgr.signalParams, request_param)
  
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
   
   def parse_signals_from_file(self, fpath, skip_rows = 0):
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
      
       header_lines += skip_rows
       column_index = range(0, SignalMgr.signalParams['COLUMN_NUM'])
       dt = np.array([])
       raw_signals = np.array([])
       if SignalMgr.signalParams['COLUMN_NUM'] == 1:
           #logger.debug(str(raw_signals))
           #logger.debug(str(header_lines))
           #logger.debug(str(column_index))
           #logger.debug("path:[%s]" % (fpath))
           raw_signals = np.genfromtxt(fpath, unpack=True, skip_header=header_lines, dtype=np.float32, delimiter=',', usecols=column_index)
           #logger.debug(str(raw_signals))
           dt = np.arange(0, len(raw_signals) * SignalMgr.signalParams['SAMPLING_DT'], SignalMgr.signalParams['SAMPLING_DT'])
       elif SignalMgr.signalParams['COLUMN_NUM'] == 2:
           dt, raw_signals = np.genfromtxt(fpath, unpack=True, skip_header=header_lines, dtype=np.float32, delimiter=',', usecols=column_index)
       return (dt, raw_signals)
