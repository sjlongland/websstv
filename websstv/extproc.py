#!/usr/bin/env python3

"""
Wrap an external application in an asyncio wrapper, piping stdout/stderr
to a log.
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

import logging
import asyncio
import os
import shlex

from . import defaults
from .signal import Signal


class ExternalProcess(object):
    """
    An instance of an external process.
    """

    _STREAM_NAMES = {1: "stdout", 2: "stderr"}

    _STREAM_LEVELS = {1: logging.DEBUG, 2: logging.WARNING}

    def __init__(
        self,
        proc_path,
        proc_args=None,
        proc_env=None,
        shell=False,
        inherit_env=True,
        cwd=None,
        loop=None,
        log=None,
    ):
        self._proc_path = proc_path
        self._proc_args = proc_args
        self._proc_env = proc_env
        self._shell = shell
        self._inherit_env = inherit_env
        self._cwd = cwd
        self._transport = None
        self._exit_status = None
        self._loop = defaults.get_loop(loop)
        self._log = defaults.get_logger(log, self.__class__.__module__)
        self._stream_logs = {}

        # Signals
        self.started = Signal()
        self.stopped = Signal()

    @property
    def pid(self):
        """
        Return the PID of the subprocess, returns None if not running.
        """
        if self._transport is None:
            return
        return self._transport.get_pid()

    @property
    def exit_status(self):
        """
        Return the exit status of the process when it stopped.  Returns None
        if the process is running (or never started).
        """
        return self._exit_status

    def _make_protocol(self):
        """
        Return a SubprocessProtocol instance that will handle the traffic from
        the subprocess stdout/stderr.
        """
        return _StdioForwardProtocol(self)

    def _on_proc_connect(self, transport):
        self._transport = transport
        self._log.info("external process started, PID %d", self.pid)
        self.started.emit()

    def _get_stream_log(self, fd):
        try:
            stream = self._STREAM_NAMES[fd]
        except KeyError:
            stream = "fd%d" % fd

        try:
            return self._stream_logs[stream]
        except KeyError:
            pass

        log = self._log.getChild(stream)
        self._stream_logs[stream] = log
        return log

    def _on_proc_receive(self, fd, data):
        data = data.decode().rstrip()
        try:
            level = self._STREAM_LEVELS[fd]
        except KeyError:
            level = logging.DEBUG

        self._get_stream_log(fd).log(level, data)

    def _on_proc_close(self):
        if self._transport is not None:
            self._log.info("Connection closed")
            self._exit_status = self._transport.get_returncode()
            self._transport = None
            self.stopped.emit()

    def _on_proc_connection_lost(self, fd, exc):
        self._log.error("Connection lost (FD %d): %s", fd, exc)
        self._on_proc_close()

    def _get_command_and_args(self, extra_args):
        command = [self._proc_path]

        if self._proc_args:
            command.extend(str(a) for a in self._proc_args)

        if extra_args:
            command.extend(str(a) for a in extra_args)

        return command

    def _get_environment(self, extra_env):
        if self._inherit_env:
            env = dict(os.environ)
        else:
            env = {}

        if self._proc_env:
            for var, val in self._proc_env.items():
                env[var] = str(val)

        if extra_env:
            for var, val in extra_env.items():
                env[var] = str(val)

        return env

    async def start(self, extra_args=None, extra_env=None):
        """
        Start the process, but do not wait for it to finish (run in
        background).
        """
        command = self._get_command_and_args(extra_args)
        env = self._get_environment(extra_env)

        if env and self._log.isEnabledFor(logging.DEBUG):
            self._log.debug(
                "Starting with environment:\n%s",
                "\n".join(
                    "\t%s = %r" % (var, val) for var, val in env.items()
                ),
            )

        if self._shell:
            command = shlex.join(command)
            self._log.info("Starting shell command (%r)", command)
            await self._loop.subprocess_shell(
                self._make_protocol, command, env=env, cwd=self._cwd
            )
        else:
            self._log.info(
                "Starting external process (%s)",
                " ".join(repr(a) for a in command),
            )
            await self._loop.subprocess_exec(
                self._make_protocol, *command, env=env, cwd=self._cwd
            )


class OneShotExternalProcess(ExternalProcess):
    """
    An external process that runs exactly once.
    """

    def __init__(
        self,
        proc_path,
        proc_args=None,
        proc_env=None,
        shell=False,
        inherit_env=True,
        cwd=None,
        loop=None,
        log=None,
    ):
        super().__init__(
            proc_path=proc_path,
            proc_args=proc_args,
            proc_env=proc_env,
            shell=shell,
            inherit_env=inherit_env,
            cwd=cwd,
            loop=loop,
            log=log,
        )
        self._future = None

    def _on_proc_close(self):
        super()._on_proc_close()
        if not self._future.done():
            self._future.set_result(self.exit_status)

    async def run(self, extra_args=None, extra_env=None):
        """
        Run the process, wait for it to finish.  Raise error on non-zero exit
        status.
        """
        self._future = self._loop.create_future()
        await self.start(extra_args, extra_env)
        self._log.info("Waiting for process to finish")
        res = await self._future
        self._log.info("Exit status: %r", res)
        if res is None:
            raise RuntimeError("No exit status received")
        elif res < 0:
            raise IOError("Subprocess was terminated")
        elif res > 0:
            raise IOError("Subprocess exited with status %d" % res)


class _StdioForwardProtocol(asyncio.SubprocessProtocol):
    """
    _StdioForwardProtocol proxies subprocess events to the
    ``ExternalProcess`` object.
    """

    def __init__(self, daemon):
        super().__init__()

        self._daemon = daemon
        self._log = daemon._log.getChild("stdio_protocol")

    def connection_made(self, transport):
        try:
            self._log.debug("Announcing connection: %r", transport)
            self._daemon._on_proc_connect(transport)
        except:
            self._log.exception("Failed to handle connection establishment")
            transport.close()

    def pipe_connection_lost(self, fd, exc):
        try:
            self._daemon._on_proc_connection_lost(fd, exc)
        except:
            self._log.exception(
                "Failed to handle connection loss on fd=%d", fd
            )

    def pipe_data_received(self, fd, data):
        try:
            self._daemon._on_proc_receive(fd, data)
        except:
            self._log.exception(
                "Failed to handle incoming data %r on fd=%d", data, fd
            )

    def process_exited(self):
        try:
            self._daemon._on_proc_close()
        except:
            self._log.exception("Failed to handle process exit")
