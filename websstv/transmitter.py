#!/usr/bin/env python3

"""
SSTV transmitter sequencer/helper.  This class is responsible for generating
a preview image for the user to review, then on approval, encode it and
transmit it live on air.
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

import datetime
import os
import os.path
import shutil
import zoneinfo

from . import defaults
from .raster import scale_image, RasterHJustify, RasterVJustify
from .encoder import MODES, SSTVEncoder
from .audio import init_audio, get_channel, AudioMixer, AudioFormat
from .sunaudio import SunAudioDecoder, SunAudioEncoding
from .cw import CWString, Prosign
from .oscillator import Oscillator
from .registry import Registry
from .observer import Signal
from .path import get_cache_dir

_STEP_REGISTRY = Registry(typeprop="action")


class TXSequenceStep(object):
    def emit(self, **kwargs):
        raise NotImplementedError("Implement in %s" % self.__class__.__name__)


@_STEP_REGISTRY.register
class TXSequenceToneStep(object):
    ALIASES = ("tone",)
    REQUIRED_INPUTS = ("oscillator",)

    def __init__(
        self,
        duration,
        frequency,
        amplitude=1.0,
        risetime=None,
        falltime=None,
    ):
        self._amplitude = float(amplitude)
        self._duration = float(duration)
        self._frequency = float(frequency)
        self._risetime = float(risetime) if (risetime is not None) else None
        self._falltime = float(falltime) if (falltime is not None) else None

    def emit(self, oscillator, **kwargs):
        oscillator.amplitude = self._amplitude
        yield from oscillator.generate(
            frequency=self._frequency,
            duration=self._duration,
            risetime=self._risetime,
            falltime=self._falltime,
        )


@_STEP_REGISTRY.register
class TXSequenceSilenceStep(object):
    ALIASES = ("silence",)
    REQUIRED_INPUTS = ("oscillator",)

    def __init__(self, duration):
        self._duration = float(duration)

    def emit(self, oscillator, **kwargs):
        yield from oscillator.silence(duration=self._duration)


@_STEP_REGISTRY.register
class TXSequenceCWStep(object):
    ALIASES = ("cw",)
    REQUIRED_INPUTS = ("oscillator",)

    def __init__(
        self,
        message=None,
        prosign=None,
        tokens=None,
        frequency=800,
        amplitude=1.0,
        dit_period=None,
        risetime=None,
        falltime=None,
    ):
        self._amplitude = float(amplitude)
        self._frequency = float(frequency)
        self._dit_period = (
            float(dit_period) if (dit_period is not None) else None
        )
        self._risetime = float(risetime) if (risetime is not None) else None
        self._falltime = float(falltime) if (falltime is not None) else None

        if message is not None:
            self._cwstring = CWString.from_string(message)
        elif prosign is not None:
            self._cwstring = CWString(Prosign[str(prosign).upper()])
        elif tokens is not None:
            self._cwstring = CWString.from_tokens(*tokens)
        else:
            raise ValueError(
                "At least one of message, prosign or tokens must be given."
            )

    def emit(self, oscillator, **kwargs):
        oscillator.amplitude = self._amplitude
        yield from self._cwstring.modulate(
            oscillator=oscillator,
            frequency=self._frequency,
            dit_period=self._dit_period,
            risetime=self._risetime,
            falltime=self._falltime,
        )


@_STEP_REGISTRY.register
class TXSequenceSSTVStep(object):
    ALIASES = ("sstv",)
    REQUIRED_INPUTS = (
        "sample_rate",
        "sample_encoding",
        "imagefile",
        "logfile",
        "audiofile",
        "mode",
        "fsk_id",
        "loop",
        "log",
    )

    def __init__(
        self,
        fsk_id=True,
        fill=False,
        hjust="centre",
        vjust="centre",
        resample=None,
    ):
        self._fsk_id = fsk_id
        self._fill = fill
        self._hjust = RasterHJustify.from_string(hjust)
        self._vjust = RasterVJustify.from_string(vjust)
        self._resample = resample.upper() if resample else None

    @property
    def fsk_id(self):
        return self._fsk_id

    async def prepare(
        self,
        sample_rate,
        sample_encoding,
        imagefile,
        logfile,
        audiofile,
        mode,
        fsk_id,
        loop,
        log,
        **kwargs
    ):
        """
        Encode the image file to an output audio file for efficient
        transmission.  Generate the slowrxd log for script-compatibility.
        """
        if self._fsk_id is not True:
            fsk_id = None

        encoder = SSTVEncoder(
            mode=mode,
            imagefile=imagefile,
            logfile=logfile,
            audiofile=audiofile,
            sample_rate=sample_rate,
            sample_encoding=sample_encoding,
            fsk_id=fsk_id or None,
            hjust=self._hjust,
            vjust=self._vjust,
            fill=self._fill,
            resample=self._resample,
            loop=loop,
            log=log,
        )
        await encoder.encode()

    def emit(self, audiofile, **kwargs):
        """
        Emit the encoded audio file samples to the audio output device.
        """
        decoder = SunAudioDecoder(audiofile)
        yield from decoder.read()


class TXSequence(object):
    @classmethod
    def from_cfg(cls, config):
        """
        Instantiate a transmit sequence from the given configuration file.
        """
        sequence = cls(
            config.pop("description"), default=config.pop("default", False)
        )
        for step in config.pop("steps"):
            sequence.add_step(**step)
        return sequence

    def __init__(self, description, default=False):
        self._default = default
        self._description = description
        self._required_inputs = set()
        self._steps = []
        self._fsk_id = False

    @property
    def description(self):
        return self._description

    @property
    def required_inputs(self):
        return self._required_inputs

    @property
    def fsk_id(self):
        return self._fsk_id

    @property
    def default(self):
        return self._default

    def add_step(self, **action):
        action = _STEP_REGISTRY.init_instance(**action)
        self._required_inputs.update(set(action.REQUIRED_INPUTS))
        self._steps.append(action)

        try:
            if action.fsk_id is True:
                self._fsk_id = True
        except AttributeError:
            pass

        return self

    async def prepare(self, **kwargs):
        """
        Prepare all data for a transmission.
        """
        # Check for required inputs!
        missing = self._required_inputs - set(kwargs.keys())
        if missing:
            raise ValueError(
                "Inputs missing: %s" % ", ".join(sorted(missing))
            )

        # All good, begin a prepare
        for step in self._steps:
            if hasattr(step, "prepare"):
                await step.prepare(**kwargs)

    def emit(self, **kwargs):
        """
        Emit audio samples from each step of the encoding process.
        """
        for step in self._steps:
            yield from step.emit(**kwargs)


class Transmitter(object):
    def __init__(
        self,
        rigctl,
        locator,
        rasteriser,
        audio_cfg,
        txsequence=None,
        channel="m",
        fsk_id=None,
        preview_path=None,
        outgoing_path=None,
        txfile_path=None,
        txfile_name="%(timestamp)s-%(mode)s",
        txfile_append_fskid="-",
        txfile_timestamp_format="%Y-%m-%dT%H-%MZ",
        txfile_timestamp_tz="UTC",
        loop=None,
        log=None,
    ):
        self._log = defaults.get_logger(log, self.__class__.__module__)
        self._loop = defaults.get_loop(loop)
        self._rig = rigctl
        self._locator = locator
        self._rasteriser = rasteriser
        self._fsk_id = fsk_id
        self._txsequence = {}
        self._defaultsequence = None
        self._channel = get_channel(channel)

        # Initialise an audio interface
        self._audio = init_audio(
            loop=self._loop, log=self._log.getChild("audio"), **audio_cfg
        )

        if txsequence:
            for name, seqcfg in txsequence.items():
                txseq = TXSequence.from_cfg(seqcfg)
                self._txsequence[name] = txseq
                if txseq.default or (self._defaultsequence is None):
                    self._defaultsequence = txseq
        else:
            # Initialise a default minimal config
            self._defaultsequence = TXSequence.from_cfg(
                {
                    "description": "Picture transmission",
                    "default": True,
                    "steps": [
                        {"action": "sstv", "fsk_id": fsk_id is not None}
                    ],
                }
            )
            self._txsequence["pic"] = self._defaultsequence

        if preview_path is None:
            # File name without extension
            preview_path = os.path.join(get_cache_dir(), "preview", "preview")
        else:
            (preview_path, _) = os.path.splitext(
                os.path.realpath(
                    os.path.expanduser(os.path.expandvars(preview_path))
                )
            )

        if outgoing_path is None:
            # File name without extension
            outgoing_path = os.path.join(
                get_cache_dir(), "outgoing", "outgoing"
            )
        else:
            (outgoing_path, _) = os.path.splitext(
                os.path.realpath(
                    os.path.expanduser(os.path.expandvars(outgoing_path))
                )
            )

        if txfile_path is not None:
            (txfile_path, _) = os.path.splitext(
                os.path.realpath(
                    os.path.expanduser(os.path.expandvars(txfile_path))
                )
            )

        for f in (preview_path, outgoing_path):
            d = os.path.dirname(f)
            if not os.path.isdir(d):
                self._log.debug("Creating %r", d)
                os.makedirs(d)

        # Preview image file outputs
        self._preview_svg = preview_path + ".svg"
        self._preview_png = preview_path + ".png"
        self._preview_mode = None

        # Outgoing file path
        self._outgoing_png = outgoing_path + ".png"
        self._outgoing_au = outgoing_path + ".au"
        self._outgoing_log = outgoing_path + ".ndjson"

        # Transmit file outputs for slowrxd script compatibility
        self._txfile_path = txfile_path
        self._txfile_name = txfile_name
        self._txfile_append_fskid = txfile_append_fskid
        self._txfile_timestamp_format = txfile_timestamp_format
        self._txfile_timestamp_tz = zoneinfo.ZoneInfo(txfile_timestamp_tz)

        self.transmitted = Signal()

    async def _fillin(self, mode, fieldname, fields, values):
        if fieldname not in fields:
            return

        method = "_fillin_field_%s" % fieldname
        if hasattr(self, method):
            value = await getattr(self, method)(
                mode=mode, fields=fields, values=values
            )
        else:
            self._log.debug("Leaving template field %r unfilled", fieldname)
            return

        self._log.debug(
            "Filling in template field %r with %r", fieldname, value
        )
        values[fieldname] = value

    async def _fillin_field_mode(self, mode, **kwargs):
        return mode.shortname

    async def _fillin_field_mode_short(self, mode, **kwargs):
        return mode.shortname

    async def _fillin_field_mode_full(self, mode, **kwargs):
        return mode.name

    async def _fillin_field_frequency_unit(self, **kwargs):
        try:
            return await self._rig.get_freq_unit()
        except NotImplementedError:
            self._log.debug("Rig does not implement frequency output")
            return "N/A"

    async def _fillin_field_grid(self, **kwargs):
        if self._locator is None:
            return "N/A"
        return self._locator.maidenhead or "N/A"

    async def render(self, mode, template, values=None):
        """
        Render the given template file with the provided input fields.
        """
        if isinstance(mode, str):
            mode = MODES[mode]

        self._log.debug("Filling in template fields")
        fields = template.fields

        if values is None:
            values = {}
        else:
            values = values.copy()

        for fieldname, field in fields.items():
            if fieldname not in values:
                # Try to fill in what we can with what we know
                await self._fillin(mode, fieldname, fields, values)

            # Handle enumerations
            if hasattr(field, "options"):
                # Validate this is one of the valid choices
                options = dict(
                    (label, value) for (value, label) in field.options
                )
                values[fieldname] = options[values[fieldname]]

        # Instantiate the template
        instance = template.get_instance(defaults=values)

        # Apply the parameters
        instance.apply()

        # Write out the preview SVG
        self._log.debug("Generating output SVG to %r", self._preview_svg)
        instance.write(self._preview_svg)

        # Rasterise the image
        self._log.debug("Rasterising for %s", mode.name)
        await self._rasteriser.render(
            self._preview_svg, self._preview_png, mode.dimensions
        )

        # Return the rendered image path
        self._log.debug("Rasterised output is in %r", self._preview_png)
        self._preview_mode = mode
        return self._preview_png

    def clear_preview(self):
        """
        Clear the preview file.
        """
        for file in (self._preview_png, self._preview_svg):
            if os.path.exists(file):
                os.unlink(file)

        self._preview_mode = None
        self._log.info("Preview cleared")

    async def transmit(self, sequence=None):
        """
        Transmit the sequence specified.
        """
        if sequence is None:
            seq = self._defaultsequence
        else:
            seq = self._txsequence[sequence]

        if self._preview_mode is not None:
            filename = self._get_txfile_name(self._preview_mode, seq.fsk_id)
            log = self._log.getChild(filename)
        else:
            filename = None
            log = self._log

        args = {
            "log": log.getChild("sequence"),
            "loop": self._loop,
            "imagefile": self._outgoing_png,
            "audiofile": self._outgoing_au,
            "logfile": self._outgoing_log,
            "sample_rate": self._audio.sample_rate,
            "sample_encoding": self._audio.sample_format.sun_encoding,
            "fsk_id": self._fsk_id,
        }

        if "oscillator" in seq.required_inputs:
            args["oscillator"] = Oscillator(
                sample_rate=self._audio.sample_rate,
                encoding=self._audio.sample_format.sun_encoding,
            )

        if "mode" in seq.required_inputs:
            if self._preview_mode is None:
                raise RuntimeError("No preview has been generated")

            args["mode"] = self._preview_mode

        if "imagefile" in seq.required_inputs:
            # Copy the preview to the output
            log.debug("Copying preview to outgoing staging area")
            shutil.copy(self._preview_png, self._outgoing_png)

        # Prepare the transmission sequencer
        await seq.prepare(**args)
        stream = seq.emit(**args)

        # Prepare the audio interface
        self._audio.reset()
        if self._audio.channels > 1:
            # Audio mapping required
            mixer = AudioMixer(
                channels=self._audio.channels,
                sample_format=self._audio.sample_format,
            )
            if self._channel < 0:
                # Mono mapping
                mixer.add_source(source=stream, channels=1, wrap=True)
            else:
                mixer.add_source(source=stream, channelmap={0: self._channel})
            self._audio.enqueue(mixer.generate(), finish=True)
        else:
            self._audio.enqueue(stream, finish=True)

        # Engage PTT
        log.info("Engaging PTT")
        await self._rig.ptt.set_ptt_state(True)
        try:
            log.info("Transmitting")
            await self._audio.start(wait=True)
            log.info("Transmission finished")
        except:
            log.exception("Failed to perform transmission")
            raise
        finally:
            # Disengage PTT
            log.info("Disengaging PTT")
            await self._rig.ptt.set_ptt_state(False)

        if filename is not None:
            if self._txfile_path is not None:
                imagefile = os.path.join(self._txfile_path, filename + ".png")
                audiofile = os.path.join(self._txfile_path, filename + ".au")
                logfile = os.path.join(
                    self._txfile_path, filename + ".ndjson"
                )
                log.info("Moving files to output")
                shutil.move(self._outgoing_png, imagefile)
                shutil.move(self._outgoing_au, audiofile)
                shutil.move(self._outgoing_log, logfile)
            else:
                imagefile = self._outgoing_png
                audiofile = self._outgoing_au
                logfile = self._outgoing_log

            self.transmitted.emit(
                imagefile=imagefile,
                logfile=logfile,
                audiofile=audiofile,
            )

        # Success, remove the preview image
        self.clear_preview()

    def _get_txfile_name(self, mode, fsk_id=True):
        """
        Return the name of the transmit file.
        """
        timestamp = datetime.datetime.now(
            tz=self._txfile_timestamp_tz
        ).strftime(self._txfile_timestamp_format)

        filename = self._txfile_name % dict(
            mode=mode.shortname, timestamp=timestamp
        )
        if (
            fsk_id
            and (self._fsk_id is not None)
            and (self._txfile_append_fskid is not None)
            and (self._txfile_append_fskid is not False)
        ):
            filename += self._txfile_append_fskid
            filename += self._fsk_id

        return os.path.join(self._txfile_path, filename)
