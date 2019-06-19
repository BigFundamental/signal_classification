#! -*- encoding: utf8 -*-
#! brief: read result.csv data & test data for test suites
import os
import csv
import pandas as pd
import numpy as np

def iconverter(x):
    if x == '--' or x == '':
        return -1
    else:
        return int(x)
    
class DataReader(object):
    """
    Standard data reader for structured directorieis
    ROOT
      -- timestamp directory
      -- timestamp directory
      -- timestamp directory
      -- result.csv
    """

    def search_label_files(self, directory, label_file_name='result.csv'):
        """
        detect label file
        """

        label_pathes = []
        for root, dirs, files in os.walk(directory):
            if label_file_name in files:
                label_pathes.append(os.sep.join([root, label_file_name]))
        return label_pathes

    def detect_schema(self, result_path):
        """
        detect schemas
        """
        old_schema=['case_name', 'sys_result', 'expect_result', 'reason']
        new_schema=['case_name', 'sys_result', 'expect_result', 'reason', 'channel_id']
        with open(result_path, 'r') as csv_f:
            csvreader = csv.reader(csv_f)
            for row in csvreader:
                col_num = len(row)
                break
            if col_num == 4:
                return old_schema
            return new_schema

    def get_signal_list(self, root_path, channel_num = 8, include_channels=[], exclude_channels=[]):
        test_suites = os.listdir(root_path)
        signals = list()
        for test in test_suites:
            #test suite dir, including multi-channels
    	    dir_path = os.path.join(root_path, test)
    	    if not os.path.isdir(dir_path):
    	        continue
            
            #collecting all 8 channels
            channels = [ i for i in xrange(1, channel_num + 1) ]
            if len(include_channels ) > 0:
                 channels = include_channels
            if len(exclude_channels) > 0:
                 channels = exclude_channels
            
            for channel in channels: 
                file_name = 'Channel_%d.csv' % (channel)
    	        case_path = os.path.join(dir_path, file_name)
    	        if not os.path.isfile(case_path):
    	            continue
    	        
                with open(case_path, 'r') as fhandler:
    	            ret = fhandler.readline().strip().split(',')[0]
    	            if int(ret) > 0:
    		        ret = 1
    	            else:
    		        ret = 0
            
                signals.append([test, channel, case_path, ret]) 
        return pd.DataFrame(signals, columns = ['case_name', 'channel_id', 'case_path', 'sys_result'])

    def create_index(self, result_path_list):
        df_ret = pd.DataFrame(columns=['case_name', 'expect_result', 'reason', 'channel_id'])
        for result_path in result_path_list:
            df_part = self.create_single_index(result_path)
            df_ret = df_ret.append(df_part)
            print result_path, len(df_part.index), len(df_ret.index)
        df_ret['expect_result'] = map(int,(~(df_ret['reason'] == -1)))
        return df_ret

    def create_single_index(self, result_path):
         """
         read csv files and create dataframe for this result.csv
         """
         case_root = os.path.dirname(result_path)
         case_info_df = self.get_signal_list(case_root)
         schemas = self.detect_schema(result_path)
         # schemas:整体比对结果，手动标定结果，波形结果，转速结果，转速值
         if 'channel_id' not in schemas:
             result_df = pd.read_csv(result_path, header = None, skiprows=1, names=schemas, usecols=[0, 2, 3], converters={'reason':iconverter, 'expect_result':iconverter})
             result_df['channel_id'] = 1
         else:
             result_df = pd.read_csv(result_path, header = None, skiprows=1, names=schemas, usecols=[0, 2, 3, 4], dtype={'channel_id':np.int32}, converters={'reason':iconverter, 'expect_result':iconverter})
         # merge scaned info into result.csv
         #print result_df.head()
         case_info_df = case_info_df.merge(result_df, on=['case_name', 'channel_id'], how='inner')
         return case_info_df 

