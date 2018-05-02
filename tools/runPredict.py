#! -*- encoding:utf8 -*-
from __future__ import absolute_import
import sys, os
import getopt

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

        self.sigMgr = SignalMgr()
    
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

    def run(self):
        """
        run predict informations
        """
        input_signals = self.get_test_files()
        request_param = dict()
        request_param['skip_row'] = [1]
        request_param['model_path'] = '/Users/changkong/ML/Signal Classification/project/model/ada.pkl'
        #request_param['model_path'] = '/Users/changkong/ML/Signal Classification/project/production/model.ada.2018.04.10/ada.pkl'
        for (shortname, input_signal) in input_signals:
            predict_result = self.sigMgr.process(input_signal, request_param)
            print >> self.output_handler, "%s:%d" % (shortname, predict_result['stat'])
        return

def main(opts, args):
    params = dict()
    for (key, value) in opts:
        if key == 'help':
            print_help()
            return
        if key == '--file' or key == '-f':
            params['file'] = value
        if key == '--output' or key == '-o':
            params['output'] = value
        if key == '--dir' or key == '-d':
            params['dir'] = value

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
        opts, args = getopt.getopt(sys.argv[1:], "ho:d:f:", ["help", "output=", "dir=", "file="])
    except getopt.GetoptError:
        # print help information and exit:
        print_help()
        exit(1)
    
    main(opts, args)
