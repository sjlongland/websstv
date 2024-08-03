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
import time

from tornado.web import (
    HTTPServer,
    Application,
    RequestHandler,
    StaticFileHandler,
)

from . import defaults


class Webserver(object):
    def __init__(
        self,
        image_dir,
        template_dir,
        locator,
        rigctl,
        port=8888,
        loop=None,
        log=None,
    ):
        self._log = defaults.get_logger(log, self.__class__.__module__)
        self._loop = defaults.get_loop(loop)
        self._image_dir = image_dir
        self._rigctl = rigctl
        self._locator = locator
        self._slowrxd_listeners = {}
        self._slowrxd_log = {}
        self._slowrxd_sent = []
        self._slowrxd_log_lock = asyncio.Lock()
        self._log_idx = 0

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
        evt = {"timestamp": int(time.time() * 1000), "event": event.name}

        # Convert paths to relative

        for prop, path in (("image", image), ("log", log), ("audio", audio)):
            if path is not None:
                path = os.path.relpath(path, self._image_dir)

            evt[prop] = path

        if (log is not None) or not os.path.isfile(log):
            # An identical log should be in the log file, so look there
            self._loop.create_task(self._scan_slowrxd_log(log))
        else:
            # Post the event as received
            self._log.debug("No log file (log=%r)", log)
            self._post_slowrxd_event(evt)

        # Post frequency and S-meter data
        self._loop.create_task(self._poll_rig_stats())

        # Poll location data
        if self._locator is not None:
            loc_evt = {
                "timestamp": int(time.time() * 1000),
                "event": "LOCATION",
                "grid": self._locator.maidenhead,
            }
            loc_evt.update(self._locator.tpv)
            self._post_slowrxd_event(loc_evt)

    async def _poll_rig_stats(self):
        self._log.debug("Polling frequency / S-meter")
        s_meter = await self._rigctl.get_s_meter_pts()
        freq = await self._rigctl.get_freq_unit()
        self._post_slowrxd_event(
            {
                "timestamp": int(time.time() * 1000),
                "event": "RIG_STATUS",
                "frequency": freq,
                "s_meter": s_meter,
            }
        )

    async def _scan_slowrxd_log(self, log):
        async with self._slowrxd_log_lock:
            logger = self._log.getChild("slowrxd.%s" % os.path.basename(log))
            try:
                if not os.path.isfile(log):
                    logger.debug("File has disappeared?")
                    return

                log_ino = os.stat(log).st_ino
                logger.debug("Scanning %r (inode %s)", log, log_ino)
                (prevname, prevpos) = self._slowrxd_log.get(log_ino, (log, 0))
                with open(log, "r") as f:
                    # Skip past previously read data
                    f.seek(prevpos)

                    # Read in everything that is new, split into lines
                    pos = 0
                    for msgtext in f.readlines():
                        pos = f.tell()
                        logger.debug("at %d: %r", pos, msgtext)
                        if not msgtext.endswith("\n"):
                            # Incomplete line
                            log.debug("rewinding incomplete line at %d", pos)
                            pos -= len(msgtext.encode())
                            break

                        try:
                            msg = json.loads(msgtext)
                        except:
                            logger.debug(
                                "Malformed log: %r", msgtext, exc_info=1
                            )
                            continue

                        self._post_slowrxd_event(msg)

                    logger.debug("Read upto position %d", pos)
                    if prevname == log:
                        self._slowrxd_log[log_ino] = (log, pos)
                        # Poll for events
                        self._loop.call_later(
                            0.1, self._scan_slowrxd_log, log
                        )
                    else:
                        logger.debug(
                            "File name changed: %r → %r", prevname, log
                        )
                        self._slowrxd_log.clear()
                        self._slowrxd_sent.clear()
            except:
                logger.debug(
                    "Failed to process event log %r", log, exc_info=1
                )

    def _post_slowrxd_event(self, event):
        idx = self._log_idx
        self._log_idx += 1
        event["idx"] = idx

        self._log.debug("Delivering %r", event)
        self._slowrxd_sent.append(event)
        for queue in list(self._slowrxd_listeners.values()):
            try:
                queue.put_nowait((event["timestamp"], idx, event))
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
        queue = asyncio.PriorityQueue()
        for msg in sorted(
            list(self._sent), key=lambda e: e.get("timestamp", 0)
        ):
            msg = msg.copy()
            msg["replay"] = True
            self.write(json.dumps(msg).encode() + b"\n")

        self.set_header("Content-Type", "application/x-ndjson")
        try:
            self._listeners[listener_id] = queue
            while True:
                (_, _, msg) = await queue.get()
                self.write(json.dumps(msg).encode() + b"\n")
                self.flush()
        finally:
            self._listeners.pop(listener_id, None)
