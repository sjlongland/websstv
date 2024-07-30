#!/usr/bin/env python3

"""
Helper routines for setting default I/O loops and loggers
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

import logging
import asyncio


def get_loop(loop):
    """
    If given an event loop, return it, otherwise use
    ``asyncio.get_event_loop()``.
    """
    if loop is None:
        loop = asyncio.get_event_loop()

    return loop


def get_logger(log, name):
    """
    If given a logger, return it, otherwise create a new one with the name
    given.
    """
    if log is None:
        log = logging.getLogger(name)

    return log
