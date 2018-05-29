#! -*- encoding: utf8 -*-
#! brief: read result.csv data & test data for test suites
import os

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

    def search_batch_files(self, dir_list, label_file_name='result.csv'):
        """
        detect all label files within contraints
        """
        label_pathes = []
        for single_dir in dir_list:
            label_pathes.extend(self.seach_label_files(single_dir))

        return label_pathes

    def read_single_label(self, dir_path, channel=1):
        """
        read label file under single directory
        """
        label_pathes = []
        
