#!/usr/bin/env python3

"""
Acquire the current maidenhead locator position and return it.
Arbitrary precision down to the sub-sub-square as made available
by the Python gpsd library.
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

from gps import gps

# Thanks to xssfox for pointing this part of the lib out
from gps.clienthelpers import maidenhead

import logging
import asyncio
import uuid
import traceback
from enum import Enum
from multiprocessing import Process, Pipe
from sys import exc_info


GPS_HOST = "localhost"
GPS_PORT = 2947


class _GPSMessage(Enum):
    VERSION = "version"
    ERROR = "error"
    EXIT = "exit"
    LOG = "log"
    MSG = "msg"
    RPC_RQ = "rpc_rq"
    RPC_RES = "rpc_res"
    RPC_ERR = "rpc_err"


class _ChildLogHandler(logging.Handler):
    def __init__(self, pipe, level=0):
        super().__init__(level=level)
        self._pipe = pipe

    def emit(self, record):
        self._pipe.send((_GPSMessage.LOG, record))


class GPSLocator(object):
    def __init__(
        self,
        hostname=GPS_HOST,
        port=GPS_PORT,
        precision=3,
        poll_interval=0.1,
        loop=None,
        log=None,
    ):
        if loop is None:
            loop = asyncio.get_event_loop()

        if log is None:
            log = logging.getLogger(self.__class__.__module__)

        self._log = log
        self._loop = loop
        self._hostname = hostname
        self._port = port
        self._precision = precision
        self._child_pipe = None
        self._child = None
        self._child_log = log.getChild("child")
        self._poll_interval = poll_interval
        self._version = None
        self._child_run = False
        self._tpv = None
        self._rpc_pending = {}

    @property
    def running(self):
        """
        Return true if the client is running.
        """
        return self._child is not None

    @property
    def version(self):
        """
        Return the version message reported by gpsd on connect.
        """
        return self._version

    @property
    def tpv(self):
        """
        Return the last position report from gpsd.
        """
        return self._tpv

    @property
    def maidenhead(self):
        """
        Return the maidenhead locator for the current position.  Return None
        if no fix is available.
        """
        tpv = self.tpv

        if tpv is None:
            return

        if tpv["mode"] < 2:
            return

        return maidenhead(tpv["lat"], tpv["lon"])[: self._precision * 2]

    def call(self, method, *args, **kwargs):
        """
        Call a method on the GPS client.
        """
        rq_id = uuid.uuid4()
        future = self._loop.create_future()

        self._rpc_pending[rq_id] = future
        self._child_pipe.send(
            (_GPSMessage.RPC_RQ, rq_id, method, args, kwargs)
        )
        return future

    def start(self):
        if self._child is not None:
            raise RuntimeError("Child already exists")

        (parent_pipe, child_pipe) = Pipe()
        self._child_pipe = parent_pipe
        self._child_run = True
        self._child = Process(target=self._child_main, args=(child_pipe,))
        self._child.start()
        self._loop.call_soon(self._parent_main)

    def stop(self):
        if self._child_pipe is None:
            raise RuntimeError("Child not running")

        self._log.debug("Sending EXIT request")
        self._child_pipe.send((_GPSMessage.EXIT,))

        self._log.debug("Waiting for exit to happen")
        self._child.join()

    def _on_child_exit(self):
        self._log.debug("Cleaning up child instance")
        self._child = None
        self._child_run = False
        self._child_pipe = None
        self._version = None
        self._tpv = None

    def _handle_message(self, msg):
        """
        Handle a message from gpsd.
        """
        if msg["class"] == "TPV":
            # Position report
            self._tpv = msg

    def _parent_main(self):
        if self._child is None:
            return

        if not self._child.is_alive():
            self._log.info("GPS client instance has exited")
            self._on_child_exit()
            return

        if self._child_pipe.poll(self._poll_interval):
            msg = self._child_pipe.recv()
            if msg[0] is _GPSMessage.LOG:
                # Log message from the child
                self._child_log.handle(msg[1])
            elif msg[0] is _GPSMessage.EXIT:
                # Child has announced it is exiting
                self._log.info("GPS client instance has announced an exit")
                self._child_run = False
                self._child.join()
                self._on_child_exit()
                return
            elif msg[0] is _GPSMessage.ERROR:
                # Child has died with a fatal error
                (ex_msg, ex_tb) = msg[1:]

                # Abort all pending requests
                rpc_pending = self._rpc_pending.copy()
                self._rpc_pending.clear()

                for future in rpc_pending.values():
                    if not future.done():
                        future.set_exception(RuntimeError(ex_msg))

                self._log.error("GPS client instance has died!\n%s", ex_tb)
                self._on_child_exit()
            elif msg[0] is _GPSMessage.VERSION:
                self._version = msg[1]
            elif msg[0] is _GPSMessage.MSG:
                self._handle_message(msg[1])
            elif msg[0] in (_GPSMessage.RPC_RES, _GPSMessage.RPC_ERR):
                rq_id = msg[1]
                try:
                    future = self._rpc_pending.pop(rq_id)
                except KeyError:
                    self._log.debug(
                        "Got RPC response to non-existant request %s", rq_id
                    )
                    return

                if future.done():
                    self._log.debug(
                        "Got RPC response to 'done' request %s", rq_id
                    )
                    return

                if msg[0] is _GPSMessage.RPC_RES:
                    future.set_result(msg[2])
                else:
                    # Child has failed the request
                    (ex_msg, ex_tb) = msg[2:]

                    self._log.warning(
                        "GPS client request %s failed:\n%s", rq_id, ex_tb
                    )
                    future.set_exception(RuntimeError(ex_msg))

        # Check again for events
        self._loop.call_soon(self._parent_main)

    def _child_main(self, parent_pipe):
        try:
            # Remove existing logger handlers so we don't cause issues
            # between the parent and child writing to the same place.
            rootlog = logging.getLogger()
            for handler in rootlog.handlers:
                rootlog.removeHandler(handler)

            # Set up logging with our own handler to pipe to the parent
            logging.getLogger().addHandler(
                _ChildLogHandler(parent_pipe, level=logging.DEBUG)
            )

            # Set up GPS client
            client = gps()
            client.connect(self._hostname, self._port)

            # Wait for the client to connect
            for msg in client:
                if msg["class"] == "VERSION":
                    parent_pipe.send((_GPSMessage.VERSION, dict(msg)))
                    break

            self._child_log.debug(
                "Connected to %s port %d", self._hostname, self._port
            )
            # Issue our watch command to enable reception
            client.send('?WATCH={"enable":true,"json":true}')

            # Enter our polling loop
            self._log.debug("Entering child main loop")
            while self._child_run:
                self._child_poll(parent_pipe, client)
                for msg in client:
                    parent_pipe.send((_GPSMessage.MSG, dict(msg)))
                    self._child_poll(parent_pipe, client)
                    if not self._child_run:
                        break

            self._log.debug("Exited child main loop")
        except:
            # Announce our failure to the parent
            (ex_type, ex_value, ex_tb) = exc_info()
            ex_str = "\n".join(
                traceback.format_exception(ex_type, value=ex_value, tb=ex_tb)
            )
            parent_pipe.send((_GPSMessage.ERROR, str(ex_value), ex_str))
            return

        # Announce we have exited to the parent
        parent_pipe.send((_GPSMessage.EXIT,))

    def _child_poll(self, parent_pipe, client):
        processed = 0
        while parent_pipe.poll(self._poll_interval):
            # There is a message from the parent
            msg = parent_pipe.recv()
            processed += 1
            self._child_log.debug("Received parent message %r", msg)
            if msg[0] is _GPSMessage.EXIT:
                # Exit request submitted.  No further arguments.
                self._child_log.info("Graceful exit requested")
                self._child_run = False
            elif msg[0] is _GPSMessage.RPC_RQ:
                # RPC request made of the client.
                # Arg 1: request ID
                # Arg 2:
                rq_id = msg[1]
                try:
                    method = getattr(client, msg[2])
                    response = (
                        _GPSMessage.RPC_RES,
                        rq_id,
                        method(*msg[3], **msg[4]),
                    )
                except:
                    (ex_type, ex_value, ex_tb) = exc_info()
                    ex_str = "\n".join(
                        traceback.format_exception(
                            ex_type, value=ex_value, tb=ex_tb
                        )
                    )
                    response = (
                        _GPSMessage.RPC_ERR,
                        rq_id,
                        str(ex_value),
                        ex_str,
                    )

                parent_pipe.send(response)
            else:
                raise ValueError("Unsupported message type %r" % msg[0])

        if processed > 0:
            self._child_log.debug("Processed %d messages", processed)


if __name__ == "__main__":
    import argparse

    async def main():
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "--hostname", type=str, default=GPS_HOST, help="gpsd hostname"
        )
        ap.add_argument(
            "--port", type=int, default=GPS_PORT, help="gpsd port"
        )

        args = ap.parse_args()

        logging.basicConfig(level=logging.DEBUG)

        client = GPSLocator(hostname=args.hostname, port=args.port)
        client.start()

        print("Connecting to gpsd…")
        while client.version is None:
            if not client.running:
                raise RuntimeError("client has died")

            await asyncio.sleep(0.1)

        print("Version: %r" % client.version)
        while client.maidenhead is None:
            if not client.running:
                raise RuntimeError("client has died")

            await asyncio.sleep(0.1)

        print("Locator: %r" % client.maidenhead)
        client.stop()

    asyncio.run(main())
