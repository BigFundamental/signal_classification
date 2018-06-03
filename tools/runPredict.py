#! -*- encoding:utf8 -*-
from __future__ import absolute_import
import sys, os
import getopt
from timeit import default_timer as timer
import numpy as np
import urllib, urllib2
import json
sys.path.append("./script/")
reload(sys)
from signal_manager import SignalMgr 
#import SignalMgr


def print_help():
    """
    print help informations
    """
    print "Usage: python runPredict.py [-o] -d test_case_root -f test_case_file"
    print "-d --dir root directory include all test cases"
    print "-o --output specified output file path to write test result"
    print "-f --file will specified filename of test case"
    print "-h --help print help functions"
    print "-t --enable_timing will enable performance evaluation"
    print "-s --server 'local/http' enable performance"

class Tester(object):
    """
    wrapper test cases
    """
    def __init__(self, params):
        self.input_file_name = params['file']
        self.input_dir_name = params['dir'].rstrip(os.path.sep)
        if params.has_key('output'):
            self.output_handler = open(params['output'], 'w')
        else:
            self.output_handler = sys.stdout

        self.enable_timing = False
        if params.has_key('enable_timing'):
            self.enable_timing = True
        if params.has_key('server'):
            self.server = params['server']
        else:
            self.server = 'local'
        self.sigMgr = SignalMgr()
        self.request_url = 'http://localhost:8000/detect?%s'
    
    def mk_request_params(self, input_signal, request_param):
        """
        request param to url params
        """
        request_param = {'skip_row':request_param['skip_row'][0], 'model_path':request_param['model_path'], 'speed_lower_bound':0, 'speed_upper_bound':20000, 'filepath':input_signal}
        return self.request_url % (urllib.urlencode(request_param))

    def search_test_files(self, root_path, file_name):
        """
        search recursively in test root
        """
        test_suites = os.listdir(root_path)
        test_cases_path = []
        for test in test_suites:
            dir_path = os.path.join(root_path, test)
            if not os.path.isdir(dir_path):
                continue
            case_path = os.path.join(dir_path, file_name)
            if not os.path.isfile(case_path):
                continue
            test_cases_path.append((test, case_path))
        return test_cases_path

    def get_test_files(self):
        """
        get files under dir
        """
        single_file_path = os.path.join(self.input_dir_name, self.input_file_name)
        if os.path.exists(single_file_path):
            return [(os.path.basename(self.input_dir_name), single_file_path)]
        else:
            # recursively search for subdirs
            return self.search_test_files(self.input_dir_name, self.input_file_name)

    def predict(self, input_signal, request_param):
        """
        wrapper for directly invoke class or via http-server
        """
        if self.server == 'http':
            #TODO issue http server request
            url_with_param = self.mk_request_params(input_signal, request_param)
            req = urllib2.Request(url = url_with_param)
            predict_result = json.loads(urllib2.urlopen(url_with_param).read())
            predict_result['stat'] = predict_result['resultCode']
        else:
            predict_result = self.sigMgr.process(input_signal, request_param)
        return predict_result
    
    def run(self):
        """
        run predict informations
        """
        input_signals = self.get_test_files()
        request_param = dict()
        request_param['skip_row'] = [1]
        request_param['model_path'] = ['/Users/changkong/ML/Signal Classification/project/model/ada.pkl']
        #request_param['model_path'] = ['/Users/changkong/ML/Signal Classification/project/model/xgb.pkl']
        #request_param['model_path'] = '/Users/changkong/ML/Signal Classification/project/production/model.ada.2018.04.10/ada.pkl'
        time_cost = dict()
        for (shortname, input_signal) in input_signals:
            if self.enable_timing:
                start = timer()
            predict_result = self.predict(input_signal, request_param)
            #predict_result = self.sigMgr.process(input_signal, request_param)
            if self.enable_timing:
                end = timer()
                time_cost[shortname] = end - start
            if self.enable_timing:
                print >> self.output_handler, "%s:%d:%f" % (shortname, predict_result['stat'], time_cost[shortname])
            else:
                print >> self.output_handler, "%s:%d" % (shortname, predict_result['stat'])

        if self.enable_timing:
            print "performace summary: [total_time:%f] [avg_time:%f] [max_time:%f] [min_time:%f]" % (np.sum(time_cost.values()), np.mean(time_cost.values()), np.max(time_cost.values()), np.min(time_cost.values()))
        return

def main(opts, args):
    params = dict()
    for (key, value) in opts:
        if key == '--help' or key == '-h':
            print_help()
            return
        if key == '--file' or key == '-f':
            params['file'] = value
        if key == '--output' or key == '-o':
            params['output'] = value
        if key == '--dir' or key == '-d':
            params['dir'] = value
        if key == '--enable_timing' or key == '-t':
            params['enable_timing'] = True
        if key == '--server' or key == '-s':
            params['server'] = value

    if not params.has_key('file'):
        print "specify [file] first"
        exit(1)
    if not params.has_key('dir'):
        print "specify [dir] first"
        exit(1)

#    if not params.has_key("dir") or not params.has_key("file")
    tester = Tester(params)
    tester.run()
    return

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:d:f:ts:", ["help", "output=", "dir=", "file=", "enable_timing","server="])
    except getopt.GetoptError:
        # print help information and exit:
        print_help()
        exit(1)
    
    main(opts, args)
