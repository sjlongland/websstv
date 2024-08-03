#!/usr/bin/env python3

"""
Web interface for websstv, back-end server
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

from tornado.web import (
    HTTPServer,
    Application,
    RequestHandler,
    StaticFileHandler,
)

from . import defaults


class Webserver(object):
    def __init__(self, image_dir, port=8888, loop=None, log=None):
        self._log = defaults.get_logger(log, self.__class__.__module__)
        self._loop = defaults.get_loop(loop)
        self._application = Application(
            handlers=[
                (r"/", RootHandler),
                (r"/rx/(.*)", StaticFileHandler, {"path": image_dir}),
            ]
        )
        self._server = HTTPServer(self._application)
        self._port = port
        self._image_dir = image_dir

    def start(self):
        self._server.listen(self._port)
        self._log.info("Listening on port %d", self._port)


class RootHandler(RequestHandler):
    def get(self):
        self.write("Hello, world")
