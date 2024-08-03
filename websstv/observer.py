#!/usr/bin/env python3

"""
A very simple signalslot work-alike
"""

# Why re-invent the wheel?  Because the existing wheel hasn't been worked on
# in a while and will stop working in Python some day.

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later


class Slot(object):
    def __init__(self, callback):
        self._callback = callback

    @property
    def callback(self):
        return self._callback

    def __call__(self, *args, **kwargs):
        self._callback(*args, **kwargs)

    def __eq__(self, other):
        if other is self:
            return True

        if isinstance(other, Slot):
            return self.callback is other.callback
        else:
            return self.callback is other


class Signal(object):
    def __init__(self):
        self._slots = []
        self._oneshot_slots = []

    def connect(self, slot, oneshot=False):
        if not isinstance(slot, Slot):
            slot = Slot(slot)

        if oneshot:
            self._oneshot_slots.append(slot)
        else:
            self._slots.append(slot)

        return slot

    def disconnect(self, slot):
        self._slots = [s for s in self._slots if s != slot]
        self._oneshot_slots = [s for s in self._oneshot_slots if s != slot]

    def emit(self, *args, **kwargs):
        slots = list(self._slots) + list(self._oneshot_slots)
        self._oneshot_slots = []
        for slot in slots:
            self._emit(slot, args, kwargs)

    def _emit(self, slot, args, kwargs):
        slot(*args, **kwargs)
