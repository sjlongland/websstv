#!/usr/bin/env python3

"""
Async-friendly ring buffer.
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

import array
import asyncio

from .defaults import get_logger


class RingBuffer(object):
    """
    A RingBuffer object is a wrapper around an Array type that tries to avoid
    blocking the I/O loop whilst filling and consuming the buffer content.

    On the receive side, we pass in a synchronous generator, and as we enqueue
    the samples, we periodically "yield" to the I/O loop via a zero-length
    sleep to ensure other tasks have a chance to run.

    On the transmit side, we emit samples as quickly as we can until we run
    out.  The consumer, if it finds it needs more, can asynchronously wait for
    additional data where required.
    """

    def __init__(self, typecode, capacity, log=None):
        """
        Initialise a ring buffer with the given type-code and capacity.
        """
        self._log = get_logger(log, self.__class__.__module__)
        self._abort = False
        self._array = array.array(typecode, [0] * capacity)
        self._rx = 0
        self._tx = 0
        self._level = 0
        self._capacity = capacity
        self._readable = asyncio.Event()
        self._writable = asyncio.Event()
        self._writable.set()

    @property
    def itemsize(self):
        """
        Return the size, in bytes, for each element in the ring buffer.
        """
        return self._array.itemsize

    @property
    def typecode(self):
        """
        Return the type code used to initialise the array in the ring buffer.
        """
        return self._array.typecode

    @property
    def level(self):
        """
        Return the number of samples waiting.
        """
        return self._level

    @property
    def capacity(self):
        """
        Return the number of samples the ring buffer can store, total.
        """
        return self._capacity

    @property
    def space(self):
        """
        Return the amount of free space in the ring buffer.
        """
        return self.capacity - self.level

    def abort(self):
        """
        Abort the enqueue task as quickly as possible.
        """
        self._log.debug("Abort called")
        self._abort = True

    async def enqueue(self, source, yield_interval=1.0, yield_period=0.0):
        """
        Read samples from the given source until it is exhausted.
        Periodically yield to the event loop to ensure we never block it for
        long periods.
        """
        try:
            loop = asyncio.get_event_loop()
            next_yield = loop.time() + yield_interval
            self._log.debug(
                "Reading from %r, yielding every %f sec for %f sec",
                source,
                yield_interval,
                yield_period,
            )

            for sample in source:
                # Check for available space
                if self.space == 0:
                    # We are full, wait for space
                    self._log.debug("Ring buffer is full")
                    self._writable.clear()
                    await self._writable.wait()
                    self._log.debug("Resuming write")

                    # Reset the yield timer, since we just returned from a
                    # yield, we can wait until later.
                    next_yield = loop.time() + yield_interval

                # Copy the sample in
                self._array[self._tx] = sample
                self._tx = (self._tx + 1) % self.capacity
                self._level += 1

                # Mark the stream as readable
                if not self._readable.is_set():
                    self._log.debug("Samples have been enqueued.")
                    self._readable.set()

                # Check for abort request
                if self._abort:
                    self._log.debug("Aborting here")
                    break

                # Yield if its time
                if loop.time() > next_yield:
                    await asyncio.sleep(yield_period)
                    next_yield = loop.time() + yield_interval

            self._log.debug("Enqueue finished")
        except:
            self._log.debug("Exception during generator read", exc_info=1)
            raise
        finally:
            # Reset the abort flag
            self._abort = False

    async def wait_readable(
        self, samples=1, poll_interval=0.2, duration=None
    ):
        """
        Wait until there are samples in the ring buffer.
        """
        if samples > self.capacity:
            # Clamp to capacity!
            samples = self.capacity

        if duration is not None:
            loop = asyncio.get_event_loop()
            deadline = loop.time() + duration

        self._log.debug("Waiting for %d samples", samples)
        while self.level < samples:
            await self._readable.wait()
            await asyncio.sleep(poll_interval)
            self._log.debug(
                "Have %d samples, waiting for %d samples", self.level, samples
            )
            if (duration is not None) and (deadline < loop.time()):
                self._log.warning(
                    "Deadline reached, have %d samples, expected %d",
                    self.level,
                    samples,
                )
                break

    def dequeue(self):
        """
        Dequeue samples from the ring buffer.
        """
        while self.level:
            # Dequeue and yield the sample
            sample = self._array[self._rx]
            self._rx = (self._rx + 1) % self.capacity
            self._level -= 1

            yield sample

            # Should be writable space now, mark as writable
            if not self._writable.is_set():
                self._log.debug("Space has been made")
                self._writable.set()

        # We've run out, mark it as not readable
        self._log.debug("Ring buffer is empty")
        self._readable.clear()
        self._writable.set()
