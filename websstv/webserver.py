#!/usr/bin/env python3

"""
Web interface for websstv, back-end server
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

import uuid
import asyncio
import json
import os
import os.path

from tornado.web import (
    HTTPServer,
    Application,
    RequestHandler,
    StaticFileHandler,
)

from . import defaults


class Webserver(object):
    def __init__(
        self, image_dir, locator, rigctl, port=8888, loop=None, log=None
    ):
        self._log = defaults.get_logger(log, self.__class__.__module__)
        self._loop = defaults.get_loop(loop)
        self._image_dir = image_dir
        self._rigctl = rigctl
        self._slowrxd_listeners = {}
        self._slowrxd_log = {}
        self._slowrxd_sent = []

        self._application = Application(
            handlers=[
                (r"/", RootHandler),
                (r"/gps", GPSLocatorHandler, {"locator": locator}),
                (
                    r"/slowrxd",
                    SlowRXDEventHandler,
                    {
                        "listeners": self._slowrxd_listeners,
                        "sent": self._slowrxd_sent,
                    },
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
        evt = {"event": event.name}

        # Convert paths to relative

        for prop, path in (("image", image), ("log", log), ("audio", audio)):
            if path is not None:
                path = os.path.relpath(path, self._image_dir)

            evt[prop] = path

        # Post the event as received
        self._post_slowrxd_event(evt)

        if (log is not None) or not os.path.isfile(log):
            self._scan_slowrxd_log(log)
        else:
            self._log.debug("No log file (log=%r)", log)

    def _scan_slowrxd_log(self, log):
        logger = self._log.getChild("slowrxd.%s" % os.path.basename(log))
        log_ino = os.stat(log).st_ino
        (prevname, prevpos) = self._slowrxd_log.get(log_ino, (log, 0))
        with open(log, "r") as f:
            # Skip past previously read data
            f.seek(prevpos)

            # Read in everything that is new, split into lines
            for msgtext in f.readlines():
                pos = f.tell()
                logger.debug("at %d: %r", pos, msgtext)
                if not msgtext.endswith("\n"):
                    # Incomplete line
                    pos -= len(msgtext.encode())
                    break

                try:
                    msg = json.loads(msgtext)
                except:
                    logger.debug("Malformed log: %r", msgtext, exc_info=1)
                    continue

                self._post_slowrxd_event(msg)

            logger.debug("Read upto position %d", pos)
            if prevname == log:
                self._slowrxd_log[log_ino] = (log, pos)
                # Poll for events
                self._loop.call_later(0.1, self._scan_slowrxd_log, log)
            else:
                logger.debug("File name changed: %r → %r", prevname, log)
                self._slowrxd_log.pop(log_ino, None)
                self._slowrxd_sent.clear()

    def _post_slowrxd_event(self, event):
        self._log.debug("Delivering %r", event)
        self._slowrxd_sent.append(event)
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
    def initialize(self, listeners, sent):
        self._listeners = listeners
        self._sent = sent

    async def get(self):
        listener_id = uuid.uuid4()
        queue = asyncio.Queue()
        for msg in list(self._sent):
            self.write(json.dumps(msg).encode() + b"\n")

        self.set_header("Content-Type", "application/x-ndjson")
        try:
            self._listeners[listener_id] = queue
            while True:
                msg = await queue.get()
                self.write(json.dumps(msg).encode() + b"\n")
                self.flush()
        finally:
            self._listeners.pop(listener_id, None)
