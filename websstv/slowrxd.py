#!/usr/bin/env python3

"""
Wrap the slowrx daemon up so it can be easily managed via an asyncio event
loop.

slowrxd can be obtained from
https://github.com/sjlongland/slowrx/tree/slowrx-daemon
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

import asyncio
import os
import os.path
import enum
import json
import socket

from .extproc import ExternalProcess
from .path import get_app_runtime_dir, get_cache_dir
from .notify import SOCKET_ENV_VAR
from .observer import Signal


class SlowRXDaemonChannel(enum.Enum):
    LEFT = "l"
    RIGHT = "r"
    MONO = "m"


class SlowRXDaemonEvent(enum.Enum):
    """
    All recognised slowrxd events.
    """

    RECEIVE_START = "RECEIVE_START"
    VIS_DETECT = "VIS_DETECT"
    SIG_STRENGTH = "SIG_STRENGTH"
    IMAGE_REFRESHED = "IMAGE_REFRESHED"
    IMAGE_FINISH = "IMAGE_FINISHED"
    FSK_DETECT = "FSK_DETECT"
    FSK_RECEIVED = "FSK_RECEIVED"
    RECEIVE_END = "RECEIVE_END"
    STATUS = "STATUS"
    WARNING = "WARNING"


class SlowRXDaemon(ExternalProcess):
    """
    An instance of the slowrx daemon.  This wraps the daemon process up to
    make integration into ``websstv`` easier.  The full set of features is
    exposed.
    """

    def __init__(
        self,
        slowrxd_path="slowrxd",
        image_dir=None,
        inprogress_audio="inprogress.au",
        inprogress_image="inprogress.png",
        inprogress_log="inprogress.ndjson",
        latest_audio="latest.au",
        latest_image="latest.png",
        latest_log="latest.ndjson",
        pcm_rate=44100,
        pcm_device=None,
        pcm_channel=SlowRXDaemonChannel.LEFT,
        event_script=None,
        fsk_detect=True,
        slant_correct=True,
        socket_path=None,
        cwd=None,
        loop=None,
        log=None,
    ):
        if image_dir is None:
            image_dir = os.path.join(get_cache_dir(), "incoming")

        if socket_path is None:
            socket_path = os.path.join(get_app_runtime_dir(), "event.sock")

        args = [
            "-d",
            image_dir,
            "-A",
            inprogress_audio,
            "-I",
            inprogress_image,
            "-L",
            inprogress_log,
            "-a",
            latest_audio,
            "-i",
            latest_image,
            "-l",
            latest_log,
            "-c",
            pcm_channel.value,
            "-r",
            pcm_rate,
            # We supply our own that will report to us via a unix domain
            # socket that we pass down to it through the environment.
            "-x",
            os.path.join(
                os.path.realpath(os.path.dirname(__file__)), "notify.py"
            ),
        ]

        if pcm_device:
            args += ["-p", pcm_device]

        if not fsk_detect:
            args += ["-F"]

        if not slant_correct:
            args += ["-S"]

        super().__init__(
            proc_path=slowrxd_path,
            proc_args=args,
            proc_env={SOCKET_ENV_VAR: socket_path},
            shell=False,
            inherit_env=True,
            cwd=cwd,
            loop=loop,
            log=log,
        )

        if event_script is not None:
            self._event_proc = ExternalProcess(
                proc_path=event_script,
                proc_args=None,
                proc_env=None,
                shell=False,
                inherit_env=True,
                loop=self._loop,
                log=self._log.getChild("event_script"),
            )
        else:
            self._event_proc = None

        self._image_dir = image_dir
        self._socket_path = socket_path

        # Signals
        self.slowrxd_event = Signal()

    @property
    def pid(self):
        """
        Return the PID of the subprocess, returns None if not running.
        """
        if self._transport is None:
            return
        return self._transport.get_pid()

    def trigger_event_script(self, event, image_file, log_file, audio_file):
        """
        Trigger the receive script given in the constructor.  This is done
        automatically when we receive an event from ``slowrxd``, but may be
        called on transmit to "inject" an outgoing transmitted image into the
        output stream.
        """
        if self._event_proc:
            self._loop.create_task(
                self._event_proc.start(
                    extra_args=[
                        event.value,
                        os.path.realpath(image_file),
                        os.path.realpath(log_file),
                        os.path.realpath(audio_file),
                    ],
                )
            )

    def _make_event_protocol(self):
        """
        Return a Protocol instance that will handle event notifications.
        """
        return _SlowRXDEventProtocol(self)

    def _on_proc_event_receive(self, data):
        try:
            data = json.loads(data)
        except:
            self._log.exception("Received malformed event %r", data)
            return

        event = SlowRXDaemonEvent(data[0])
        image, log, audio = data[1:4]
        self._log.debug("Received event %s", event.name)

        self.trigger_event_script(event, image, log, audio)
        self.slowrxd_event.emit(
            event=event, image=image, log=log, audio=audio
        )

    async def start(self):
        socket_dir = os.path.dirname(self._socket_path)
        for dirname in (self._image_dir, socket_dir):
            if not os.path.exists(dirname):
                os.makedirs(dirname)
                self._log.debug("Created %s", dirname)

        await super().start()

        self._log.info("Opening datagram socket %r", self._socket_path)
        await self._loop.create_datagram_endpoint(
            self._make_event_protocol,
            local_addr=self._socket_path,
            family=socket.AF_UNIX,
        )


class _SlowRXDEventProtocol(asyncio.DatagramProtocol):
    def __init__(self, daemon):
        super().__init__()

        self._daemon = daemon
        self._log = daemon._log.getChild("event_protocol")

    def datagram_received(self, data, addr):
        try:
            # NB: addr is usually not meaningful.
            self._daemon._on_proc_event_receive(data)
        except:
            self._log.exception(
                "Failed to handle incoming data %r from addr=%s", data, addr
            )
