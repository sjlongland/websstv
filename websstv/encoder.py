#!/usr/bin/env python3

"""
SSTV encoder with thread-based wrapper.
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

import pysstv.color
import pysstv.grayscale
from PIL import Image

from . import defaults
from .slowrxd import SlowRXDaemonEvent
from .sunaudio import SunAudioEncoder, get_spec
from .threadpool import ThreadPool
from .raster import (
    RasterDimensions,
    RasterHJustify,
    RasterVJustify,
    scale_image,
)

from collections import namedtuple
import enum
import json
import time


class SSTVColourSpace(enum.Enum):
    """
    Description of the colour space used by the SSTV mode.  This basically
    is an approximation without regard to channel sequence order, meant to
    guide user selection in a UI.
    """

    MONO = "mono"
    RGB = "rgb"
    YUV = "yuv"


class SSTVMode(
    namedtuple(
        "_SSTVMode", ["name", "shortname", "colourspace", "txtime", "encoder"]
    )
):
    """
    A helper class for storing what modes are available.
    """

    @property
    def dimensions(self):
        return RasterDimensions(
            width=self.encoder.WIDTH, height=self.encoder.HEIGHT
        )


# All modes supported by both pysstv and slowrx.
#
# Not certain the txtimes are correct in all cases, it was a quick
# calculation by a quick and dirty C program written againt modespec.c in
# slowrx.
MODES = dict(
    (m.shortname, m)
    for m in [
        SSTVMode(
            name="Martin M1",
            shortname="M1",
            colourspace=SSTVColourSpace.RGB,
            txtime=114.290176,
            encoder=pysstv.color.MartinM1,
        ),
        SSTVMode(
            name="Martin M2",
            shortname="M2",
            colourspace=SSTVColourSpace.RGB,
            txtime=58.060442,
            encoder=pysstv.color.MartinM2,
        ),
        SSTVMode(
            name="Scottie S1",
            shortname="S1",
            colourspace=SSTVColourSpace.RGB,
            txtime=109.665280,
            encoder=pysstv.color.ScottieS1,
        ),
        SSTVMode(
            name="Scottie S2",
            shortname="S2",
            colourspace=SSTVColourSpace.RGB,
            txtime=71.089152,
            encoder=pysstv.color.ScottieS2,
        ),
        SSTVMode(
            name="Scottie DX",
            shortname="SDX",
            colourspace=SSTVColourSpace.RGB,
            txtime=268.876800,
            encoder=pysstv.color.ScottieDX,
        ),
        SSTVMode(
            name="Robot 36",
            shortname="R36",
            colourspace=SSTVColourSpace.YUV,
            txtime=36.000000,
            encoder=pysstv.color.Robot36,
        ),
        SSTVMode(
            name="Robot 24 B/W",
            shortname="R24BW",
            colourspace=SSTVColourSpace.MONO,
            txtime=24.000000,
            encoder=pysstv.grayscale.Robot24BW,
        ),
        SSTVMode(
            name="Robot 8 B/W",
            shortname="R8BW",
            colourspace=SSTVColourSpace.MONO,
            txtime=8.040000,
            encoder=pysstv.grayscale.Robot8BW,
        ),
        SSTVMode(
            name="PD-90",
            shortname="PD90",
            colourspace=SSTVColourSpace.YUV,
            txtime=179.978240,
            encoder=pysstv.color.PD90,
        ),
        SSTVMode(
            name="PD-120",
            shortname="PD120",
            colourspace=SSTVColourSpace.YUV,
            txtime=252.206080,
            encoder=pysstv.color.PD120,
        ),
        SSTVMode(
            name="PD-160",
            shortname="PD160",
            colourspace=SSTVColourSpace.YUV,
            txtime=321.766400,
            encoder=pysstv.color.PD160,
        ),
        SSTVMode(
            name="PD-180",
            shortname="PD180",
            colourspace=SSTVColourSpace.YUV,
            txtime=374.103040,
            encoder=pysstv.color.PD180,
        ),
        SSTVMode(
            name="PD-240",
            shortname="PD240",
            colourspace=SSTVColourSpace.YUV,
            txtime=496.000000,
            encoder=pysstv.color.PD240,
        ),
        SSTVMode(
            name="PD-290",
            shortname="PD290",
            colourspace=SSTVColourSpace.YUV,
            txtime=577.364480,
            encoder=pysstv.color.PD290,
        ),
        SSTVMode(
            name="Pasokon P3",
            shortname="P3",
            colourspace=SSTVColourSpace.RGB,
            txtime=203.050000,
            encoder=pysstv.color.PasokonP3,
        ),
        SSTVMode(
            name="Pasokon P5",
            shortname="P5",
            colourspace=SSTVColourSpace.RGB,
            txtime=304.576240,
            encoder=pysstv.color.PasokonP5,
        ),
        SSTVMode(
            name="Pasokon P7",
            shortname="P7",
            colourspace=SSTVColourSpace.RGB,
            txtime=406.098512,
            encoder=pysstv.color.PasokonP7,
        ),
        SSTVMode(
            name="Wraase SC-2 120",
            shortname="W2120",
            colourspace=SSTVColourSpace.RGB,
            txtime=121.735685,
            encoder=pysstv.color.WraaseSC2120,
        ),
        SSTVMode(
            name="Wraase SC-2 180",
            shortname="W2180",
            colourspace=SSTVColourSpace.RGB,
            txtime=182.021760,
            encoder=pysstv.color.WraaseSC2180,
        ),
    ]
)


class SSTVEncoder(object):
    """
    A wrapper around the pysstv encoder that handles the encoding of the
    slow-scan image in a background thread.  For compatibility with slowrxd
    hook scripts, we also mimic slowrxd's behaviour.
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
        # Function credit: András Veres-Szentkirályi and pySSTV contributors
        try:
            # Record start time (msec)
            start_time = int(time.time() * 1000)

            # Obtain the encoder class
            encoder_cls = self._mode.encoder

            # Instantiate a log
            log_records = [
                dict(
                    timestamp=start_time,
                    type=SlowRXDaemonEvent.VIS_DETECT.value,
                    msg="Detected mode %s (VIS code 0x%02x)"
                    % (self._mode.name, encoder_cls.VIS_CODE),
                    code=encoder_cls.VIS_CODE,
                    mode=self._mode.shortname,
                    desc=self._mode.name,
                ),
                dict(
                    timestamp=start_time,
                    type=SlowRXDaemonEvent.RECEIVE_START.value,
                    msg="Receive started",
                ),
            ]

            # Load and scale the image
            image = scale_image(
                image=Image.open(self._imagefile),
                dimensions=self._mode.dimensions,
                fill=self._fill,
                hjust=self._hjust,
                vjust=self._vjust,
                resample=self._resample,
            )

            # Instantiate and configure the encoder
            encoder = encoder_cls(
                image, self._sample_rate, get_spec(self._sample_encoding).bits
            )
            if self._fsk_id is not None:
                encoder.add_fskid_text(self._fsk_id)
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
            audio.write_samples(encoder.gen_samples())

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
