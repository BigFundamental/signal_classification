#!-*- encoding: utf8 -*-
"""
author: Lin Zhou
date: 2018-01-22
Local http-server for system response
"""

from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from SocketServer import ThreadingMixIn
from urlparse import urlparse, parse_qs
from urllib import unquote
import threading
import json
import traceback

"""
https://pymotw.com/2/BaseHTTPServer/
"""
class Handler(BaseHTTPRequestHandler):

    def get_params_(self):
        query_components = parse_qs(urlparse(unquote(self.path)).query)
        return query_components

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        #message =  threading.currentThread().getName()
        jsonRetParam = dict()
        jsonRetParam['errorCode'] = 0
        jsonRetParam['resultCode'] = 0
        jsonRetParam['speed'] = 0
        try:
            params = self.get_params_()
            signal_mgr = SignalMgr()
            pred_ret = signal_mgr.process(params['filepath'])
            jsonRetParam['resultCode'] = pred_ret['signal_type']
            jsonRetParam['speed'] = pred_ret['speed']
        except:
            traceback.print_exc()
            jsonRetParam['errorCode'] = 1
        self.wfile.write(json.dumps(jsonRetParam))
        return

# using ForkingMixIn instead of ThreadingMixIn
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    allow_reuse_address = True

    def shutdown(self):
        self.socket.close()
        HTTPServer.shutdown(self) 

if __name__ == '__main__':
    server = ThreadedHTTPServer(('localhost', 8000), Handler)
    print 'Starting server, use <Ctrl-C> to stop'
    server.serve_forever()
