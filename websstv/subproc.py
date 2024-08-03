#!/usr/bin/env python3

"""
Sub-process wrapper.  This is used to wrap communications with a synchronous
(and possibly not thread-safe) library in a sub-process so it can be
interacted with asynchronously without blocking.
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

import logging
import uuid
import traceback
from functools import partial
from multiprocessing import Process, Pipe
from sys import exc_info

from . import defaults
from .observer import Signal


class _ChildLogHandler(logging.Handler):
    def __init__(self, msg_type, pipe, level=0):
        super().__init__(level=level)
        self._msg_type = msg_type
        self._pipe = pipe

    def emit(self, record):
        self._pipe.send((self._msg_type, record))


class ChildError(RuntimeError):
    """
    Class to capture and represent an exception that happened client-side.
    """

    pass


class ChildProcessWrapper(object):
    def __init__(
        self,
        poll_interval=0.1,
        start_delay=3,
        loop=None,
        log=None,
    ):
        self._log = defaults.get_logger(log, self.__class__.__module__)
        self._loop = defaults.get_loop(loop)
        self._poll_interval = poll_interval
        self._child_pipe = None
        self._child = None
        self._child_log = log.getChild("child")
        self._child_run = False
        self._rpc_pending = {}
        self._start_delay = start_delay
        self._start_future = None

        self.started = Signal()
        self.stopped = Signal()

    @property
    def running(self):
        """
        Return true if the client is running.
        """
        return self._child is not None

    def start(self):
        if self._child is not None:
            raise RuntimeError("Child already exists")

        (parent_pipe, child_pipe) = Pipe()
        self._start_future = self._loop.create_future()
        self._child_pipe = parent_pipe
        self._child_run = True
        self._child = Process(target=self._child_main, args=(child_pipe,))
        self._child.start()
        self._loop.call_soon(self._parent_poll_child)
        self._loop.call_later(self._start_delay, self._parent_check_child)
        return self._start_future

    def stop(self):
        if self._child_pipe is None:
            raise RuntimeError("Child not running")

        self._log.debug("Sending EXIT request")
        self._child_pipe.send((self._Message.EXIT,))

        self._log.debug("Waiting for exit to happen")
        self._child.join()

    def _call(self, method, *args, **kwargs):
        """
        Call a method on the wrapped client.
        """
        rq_id = uuid.uuid4()
        future = self._loop.create_future()

        self._rpc_pending[rq_id] = future
        self._child_pipe.send(
            (self._Message.RPC_RQ, rq_id, method, args, kwargs)
        )
        return future

    def _on_child_exit(self):
        self._log.debug("Cleaning up child instance")
        self._child = None
        self._child_run = False
        self._child_pipe = None
        self._version = None
        self._tpv = None
        if (self._start_future is not None) and not self._start_future.done():
            self._start_future.set_exception(RuntimeError("Child has died"))

    def _handle_message(self, msg):
        self._log.debug("Received unhandled message %r", msg)

    def _parent_check_child(self):
        if self._child.is_alive():
            self._mark_child_up()

    def _parent_poll_child(self):
        if self._child is None:
            return

        if not self._child.is_alive():
            self._log.info("Child has exited")
            self._on_child_exit()
            return

        if self._child_pipe.poll(self._poll_interval):
            msg = self._child_pipe.recv()
            if msg[0] is self._Message.LOG:
                # Log message from the child
                self._child_log.handle(msg[1])
            elif msg[0] is self._Message.EXIT:
                # Child has announced it is exiting
                self._log.info("Child has announced an exit")
                self._child_run = False
                self._child.join()
                self._on_child_exit()
                return
            elif msg[0] is self._Message.ERROR:
                # Child has died with a fatal error
                (ex_msg, ex_tb) = msg[1:]

                # Abort all pending requests
                rpc_pending = self._rpc_pending.copy()
                self._rpc_pending.clear()

                for future in rpc_pending.values():
                    if not future.done():
                        future.set_exception(ChildError(ex_msg))

                self._log.error("Child has died!\n%s", ex_tb)
                self._on_child_exit()
            elif msg[0] in (self._Message.RPC_RES, self._Message.RPC_ERR):
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

                if msg[0] is self._Message.RPC_RES:
                    future.set_result(msg[2])
                else:
                    # Child has failed the request
                    (ex_msg, ex_tb) = msg[2:]

                    self._log.warning(
                        "Child request %s failed:\n%s", rq_id, ex_tb
                    )
                    future.set_exception(ChildError(ex_msg))
            else:
                self._handle_message(msg)

        # Check again for events
        self._loop.call_soon(self._parent_poll_child)

    def _mark_child_up(self):
        if (self._start_future is not None) and not self._start_future.done():
            self._log.debug("Marking process as up")
            self._start_future.set_result(None)
            self._start_future = None

    def _child_main(self, parent_pipe):
        try:
            # Remove existing logger handlers so we don't cause issues
            # between the parent and child writing to the same place.
            rootlog = logging.getLogger()
            for handler in rootlog.handlers:
                rootlog.removeHandler(handler)

            # Set up logging with our own handler to pipe to the parent
            logging.getLogger().addHandler(
                _ChildLogHandler(
                    self._Message.LOG, parent_pipe, level=logging.DEBUG
                )
            )

            # Initialise the child process, initialise our client
            client = self._child_init(parent_pipe)

            # Enter our polling loop
            self._log.debug("Entering child main loop")
            while self._child_run:
                self._child_poll_parent(parent_pipe, client)
                self._child_poll_tasks(parent_pipe, client)

            self._log.debug("Exited child main loop")
        except:
            # Announce our failure to the parent
            (ex_type, ex_value, ex_tb) = exc_info()
            ex_str = "\n".join(
                traceback.format_exception(ex_type, value=ex_value, tb=ex_tb)
            )
            parent_pipe.send((self._Message.ERROR, str(ex_value), ex_str))
            return

        # Announce we have exited to the parent
        parent_pipe.send((self._Message.EXIT,))

    def _child_poll_parent(self, parent_pipe, client):
        processed = 0
        while parent_pipe.poll(self._poll_interval):
            # There is a message from the parent
            msg = parent_pipe.recv()
            processed += 1
            self._child_log.debug("Received parent message %r", msg)
            if msg[0] is self._Message.EXIT:
                # Exit request submitted.  No further arguments.
                self._child_log.info("Graceful exit requested")
                self._child_run = False
            elif msg[0] is self._Message.RPC_RQ:
                # RPC request made of the client.
                # Arg 1: request ID
                # Arg 2: method
                # Arg 3: positional arguments tuple
                # Arg 4: keyword arguments dict
                rq_id = msg[1]
                try:
                    method = getattr(client, msg[2])
                    response = (
                        self._Message.RPC_RES,
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
                        self._Message.RPC_ERR,
                        rq_id,
                        str(ex_value),
                        ex_str,
                    )

                parent_pipe.send(response)
            else:
                raise ValueError("Unsupported message type %r" % msg[0])

        if processed > 0:
            self._child_log.debug("Processed %d messages", processed)
