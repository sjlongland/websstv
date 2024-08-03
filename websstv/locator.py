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

from .observer import Signal
from .subproc import ChildProcessWrapper

import logging
import asyncio
from enum import Enum


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


class GPSLocator(ChildProcessWrapper):
    _Message = _GPSMessage

    def __init__(
        self,
        hostname=GPS_HOST,
        port=GPS_PORT,
        precision=3,
        poll_interval=0.1,
        start_delay=0,
        loop=None,
        log=None,
    ):
        super().__init__(poll_interval=poll_interval, loop=loop, log=log)

        self._hostname = hostname
        self._port = port
        self._precision = precision
        self._version = None
        self._tpv = None

        self.tpv_received = Signal()

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

    def _handle_message(self, msg):
        if msg[0] is _GPSMessage.VERSION:
            self._version = msg[1]
        elif msg[0] is _GPSMessage.MSG:
            self._handle_gpsd_message(msg[1])
        else:
            super()._handle_message(msg)

    def _handle_gpsd_message(self, msg):
        """
        Handle a message from gpsd.
        """
        if msg["class"] == "TPV":
            # Position report
            self._tpv = msg

            # We're getting TPV messages, mark the child as alive
            self._mark_child_up()

            self.tpv_received.emit(tpv=msg, maidenhead=self.maidenhead)

    def _child_init(self, parent_pipe):
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

        return client

    def _child_poll_tasks(self, parent_pipe, client):
        for msg in client:
            parent_pipe.send((_GPSMessage.MSG, dict(msg)))
            self._child_poll_parent(parent_pipe, client)
            if not self._child_run:
                break


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
