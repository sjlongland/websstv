#!/usr/bin/env python3

"""
Web interface for websstv, back-end server
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

import uuid
import asyncio
import json
import os.path

from tornado.web import (
    HTTPServer,
    Application,
    RequestHandler,
    StaticFileHandler,
)

from . import defaults


class Webserver(object):
    def __init__(self, image_dir, locator, port=8888, loop=None, log=None):
        self._log = defaults.get_logger(log, self.__class__.__module__)
        self._loop = defaults.get_loop(loop)
        self._image_dir = image_dir
        self._slowrxd_listeners = {}

        self._application = Application(
            handlers=[
                (r"/", RootHandler),
                (r"/gps", GPSLocatorHandler, {"locator": locator}),
                (
                    r"/slowrxd",
                    SlowRXDEventHandler,
                    {"listeners": self._slowrxd_listeners},
                ),
                (r"/rx/(.*)", StaticFileHandler, {"path": image_dir}),
            ]
        )
        self._server = HTTPServer(self._application)
        self._port = port

    def start(self):
        self._server.listen(self._port)
        self._log.info("Listening on port %d", self._port)

    def on_slowrxd_event(
        self, event, image=None, log=None, audio=None, **kwargs
    ):
        self._log.debug("Processing event %s", event.name)
        event = {"event": event.name}

        # Convert paths to relative

        for prop, path in (("image", image), ("log", log), ("audio", audio)):
            if path is not None:
                path = os.path.relpath(path, self._image_dir)

            event[prop] = path

        # Deliver to the clients
        self._log.debug("Delivering %r", event)

        for queue in list(self._slowrxd_listeners.values()):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop for that listener
                pass


class RootHandler(RequestHandler):
    def get(self):
        self.write("Hello, world")


class GPSLocatorHandler(RequestHandler):
    def initialize(self, locator):
        self._locator = locator

    def get(self):
        if self._locator is None:
            self.set_status(501)
            self.write(
                {
                    "mode": None,
                    "lat": None,
                    "lon": None,
                    "grid": None,
                    "status": "No GPS",
                }
            )
        else:
            self.set_status(200)
            body = {"grid": self._locator.maidenhead}
            body.update(self._locator.tpv)
            self.write(body)


class SlowRXDEventHandler(RequestHandler):
    def initialize(self, listeners):
        self._id = uuid.uuid4()
        self._listeners = listeners
        self._queue = asyncio.Queue()

    async def get(self):
        self.set_header("Content-Type", "application/x-ndjson")
        try:
            self._listeners[self._id] = self._queue
            while True:
                msg = await self._queue.get()
                self.write(json.dumps(msg).encode() + b"\n")
                self.flush()
        finally:
            self._listeners.pop(self._id, None)
