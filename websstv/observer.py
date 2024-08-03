#!/usr/bin/env python3

"""
A very simple signalslot work-alike
"""

# Why re-invent the wheel?  Because the existing wheel hasn't been worked on
# in a while and will stop working in Python some day.

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later


class Signal(object):
    def __init__(self):
        self._slots = []
        self._oneshot_slots = []

    def connect(self, slot, oneshot=False):
        if oneshot:
            self._oneshot_slots.append(slot)
        else:
            self._slots.append(slot)

    def disconnect(self, slot):
        try:
            self._slots.remove(slot)
        except ValueError:
            pass

        try:
            self._oneshot_slots.remove(slot)
        except ValueError:
            pass

    def emit(self, *args, **kwargs):
        slots = list(self._slots) + list(self._oneshot_slots)
        self._oneshot_slots = []
        for slot in slots:
            self._emit(slot, args, kwargs)

    def _emit(self, slot, args, kwargs):
        slot(*args, **kwargs)
