#!/usr/bin/env python3

"""
SSTV encoder with thread-based wrapper.
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

from . import defaults
from .libsstvenc import LibSSTVEnc
from .imgreader import read_image
from .slowrxd import SlowRXDaemonEvent
from .sunaudio import SunAudioEncoder, SunAudioEncoding
from .threadpool import ThreadPool
from .raster import (
    RasterHJustify,
    RasterVJustify,
)

from collections import namedtuple
import enum
import json
import time


# All modes supported by libsstvenc
MODES = dict(
    (m.name, m)
    for m in (
        LibSSTVEnc.get_instance().get_sstv_mode_by_idx(idx)
        for idx in range(LibSSTVEnc.get_instance().get_sstv_mode_count())
    )
)


class SSTVEncoder(object):
    """
    A wrapper around the libsstvenc encoder that handles the reading of the
    image and conversion into the correct format.
    """

    def __init__(
        self,
        mode,
        imagefile,
        logfile,
        audiofile,
        sample_rate,
        sample_encoding,
        fsk_id=None,
        fill=False,
        hjust=RasterHJustify.CENTRE,
        vjust=RasterVJustify.CENTRE,
        resample=None,
        loop=None,
        log=None,
    ):
        if isinstance(mode, str):
            mode = MODES[mode]

        self._log = defaults.get_logger(log, self.__class__.__module__)
        self._loop = defaults.get_loop(loop)
        self._imagefile = imagefile
        self._logfile = logfile
        self._audiofile = audiofile
        self._sample_rate = sample_rate
        self._sample_encoding = sample_encoding
        self._fsk_id = fsk_id
        self._mode = mode
        self._fill = fill
        self._hjust = hjust
        self._vjust = vjust
        self._resample = resample

    def encode(self):
        """
        Encode the image in a worker thread.
        """
        future = self._loop.create_future()
        threadpool = ThreadPool.get_instance()
        threadpool.submit(self._encode, future)
        return future

    def _encode(self, future):
        try:
            # Record start time (msec)
            start_time = int(time.time() * 1000)

            # Instantiate a log
            log_records = [
                dict(
                    timestamp=start_time,
                    type=SlowRXDaemonEvent.VIS_DETECT.value,
                    msg="Detected mode %s (VIS code 0x%02x)"
                    % (self._mode.description, self._mode.vis_code),
                    code=self._mode.vis_code,
                    mode=self._mode.name,
                    desc=self._mode.description,
                ),
                dict(
                    timestamp=start_time,
                    type=SlowRXDaemonEvent.RECEIVE_START.value,
                    msg="Receive started",
                ),
            ]

            # Load and scale the image, extract the RGB framebuffer
            framebuffer = read_image(
                imagefile=self._imagefile,
                width=self._mode.width,
                height=self._mode.height,
                fill=self._fill,
                hjust=self._hjust,
                vjust=self._vjust,
                resample=self._resample,
            )

            mod = LibSSTVEnc.get_instance().init_mod(
                mode=self._mode,
                fsk_id=self._fsk_id,
                framebuffer=framebuffer,
                sample_rate=self._sample_rate,
            )

            if self._fsk_id is not None:
                timestamp = int(time.time() * 1000)
                log_records.extend(
                    [
                        dict(
                            timestamp=timestamp,
                            type=SlowRXDaemonEvent.FSK_DETECT.value,
                        ),
                        dict(
                            timestamp=timestamp,
                            type=SlowRXDaemonEvent.FSK_RECEIVED.value,
                            id=self._fsk_id,
                        ),
                    ]
                )

            # Open output audio file
            audio = SunAudioEncoder(
                self._audiofile, self._sample_rate, 1, self._sample_encoding
            )

            # Encode to audio
            done = False
            while not done:
                buffer = mod.read(self._sample_rate)
                self._log.debug("Read %d samples", len(buffer))
                if len(buffer) < self._sample_rate:
                    self._log.debug("We have reached the end")
                    done = True
                audio.write_samples(
                    buffer, encoding=SunAudioEncoding.FLOAT_64BIT
                )

            # Finish up
            audio.close()

            # Make a note of the finish time
            end_time = int(time.time() * 1000)
            log_records.append(
                dict(
                    timestamp=end_time,
                    type=SlowRXDaemonEvent.RECEIVE_END.value,
                )
            )

            # Write out an event log for slowrxd script compatibility
            with open(self._logfile, "w") as f:
                for entry in log_records:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            # Signal failure
            self._loop.call_soon_threadsafe(future.set_exception, e)
            return

        # Signal completion
        self._loop.call_soon_threadsafe(future.set_result, None)
