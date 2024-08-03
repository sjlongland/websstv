#!/usr/bin/env python3

"""
Web interface for websstv, back-end server
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

from tornado.web import HTTPServer, Application, RequestHandler

from . import defaults


class Webserver(object):
    def __init__(self, port=8888, loop=None, log=None):
        self._log = defaults.get_logger(log, self.__class__.__module__)
        self._loop = defaults.get_loop(loop)
        self._application = Application(handlers=[(r"/", RootHandler)])
        self._server = HTTPServer(self._application)
        self._port = port

    def start(self):
        self._server.listen(self._port)
        self._log.info("Listening on port %d", self._port)


class RootHandler(RequestHandler):
    def get(self):
        self.write("Hello, world")
