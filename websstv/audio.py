#!/usr/bin/env python3

"""
Asynchronous audio output interface.
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

import array
import enum
import struct

from . import defaults
from .extproc import ExternalProcess
from .signal import Signal


class AudioFormat(enum.Enum):
    LINEAR_8BIT = "b"
    LINEAR_16BIT = "h"
    LINEAR_32BIT = "l"
    FLOAT_32BIT = "f"
    FLOAT_64BIT = "d"


class AudioEndianness(enum.Enum):
    LITTLE = 0
    BIG = 1


class AudioInterface(object):
    def __init__(
        self,
        sample_rate=48000,
        channels=1,
        sample_format=AudioFormat.LINEAR_16BIT,
        endianness=AudioEndianness.LITTLE,
        buffer_sz=1048576,
        read_threshold=0.25,
        loop=None,
        log=None,
    ):
        self._loop = defaults.get_loop(loop)
        self._log = defaults.get_logger(log, self.__class__.__module__)
        self._sample_rate = int(sample_rate)
        self._channels = int(channels)
        self._sample_format = AudioFormat(sample_format)
        self._buffer = array.array(
            self._sample_format.value, bytes([0]) * buffer_sz
        )
        self._rd_ptr = 0
        self._wr_ptr = 0
        self._src = None
        self._queue = []
        self._drain = False

        self._read_threshold = int(
            (len(self._buffer) // self._channels)
            * read_threshold
        )

        # Determine if we need to swap bytes or not
        if self._sample_format is not AudioFormat.LINEAR_8BIT:
            endianness = AudioEndianness(endianness)
            host_endianness = AudioEndianness(struct.pack("@H", 0x0100)[0])
            self._swapped = endianness is not host_endianness
        else:
            # 8-bit audio has no endianness
            self._swapped = False

        # Signals
        self.underrun = Signal()

    def enqueue(self, gen):
        """
        Enqueue an audio sample generator to be played.
        """
        self._drain = False
        self._queue.append(gen)

    @property
    def _len_samples_buffered(self):
        """
        Return the number of samples buffered.
        """
        if self._rd_ptr == self._wr_ptr:
            # Nothing is buffered
            return 0
        elif self._rd_ptr < self._wr_ptr:
            # [       R         W      ]
            #         '--------' <-- buffered
            return self._wr_ptr - self._rd_ptr
        else:
            # [       W         R      ]
            # -------'          '------- <-- buffered
            return len(self._buffer) - self._rd_ptr + self._wr_ptr

    @property
    def _len_frames_buffered(self):
        """
        Return the number of frames buffered
        """
        return self._len_samples_buffered // self._channels

    def _buffer_wr(self, samples):
        """
        Write samples from the given sequence into the buffer until
        we run out of samples or space.  Return whether we stopped because
        our buffer filled up.
        """

        for sample in samples:
            next_wr = (self._wr_ptr + 1) % len(self._buffer)
            if next_wr == self._rd_ptr:
                # Buffer is full
                return True

            # There is space, write
            self._buffer[next_wr] = sample
            self._wr_ptr = next_wr

        # We got to the end of the sequence without filling up
        return False

    def _buffer_rd(self, frames):
        """
        Read up to ``frames`` frames of samples from the buffer.
        """
        # Make space for the frames
        output = array.array(
            self._sample_format.value,
            bytes([0]) * frames * self._buffer.itemsize * self._channels,
        )
        pos = 0
        remain = frames * self._channels
        buffer_sz = len(self._buffer)

        while remain:
            if self._len_frames_buffered < self._read_threshold:
                # We are low on samples, read some data in
                self._log.debug("Low watermark reached, performing a read")
                while self._queue or (self._src is not None):
                    if self._src is None:
                        self._log.debug("Next audio source")
                        self._src = self._queue.pop(0)

                    full = self._buffer_wr(self._src)
                    if not full:
                        # This source is depleted
                        self._log.debug("Audio source finished")
                        self._src = None

            if self._rd_ptr == self._wr_ptr:
                # We're out of data
                if self._drain:
                    self._log.debug("Playback complete")
                    self._log.call_soon(self._end_playback)
                else:
                    self._log.warning("Underrun detected")
                    self.underrun.emit()
                break
            elif self._rd_ptr < self._wr_ptr:
                # [       R         W      ]
                #         '--------' <-- buffered
                sz = min(self._wr_ptr - self._rd_ptr, remain)
            else:
                # [       W         R      ]
                # -------'          '------- <-- buffered
                # Read the first part up to the end of the buffer.
                # sz will wrap us around to 0 to get the rest on
                # the next cycle.
                sz = min(buffer_sz - self._rd_ptr, remain)

            output[pos : pos + sz] = self._buffer[
                self._rd_ptr : self._rd_ptr + sz
            ]
            self._rd_ptr = (self._rd_ptr + sz) % buffer_sz
            pos += sz
            remain -= sz

        if remain > 0:
            # Truncate the array we have
            output = output[0:-remain]

        # Perform byte swap if appropriate
        if self._swapped:
            output.byteswap()

        return output
