#!/usr/bin/env python3

"""
Asynchronous audio output interface.
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

import array
import enum
import struct
import json
import tempfile
import os
import asyncio

from . import defaults
from .extproc import ExternalProcess
from .observer import Signal
from .registry import Registry
from .sunaudio import SunAudioEncoding

# The 'array' class type codes differ in size depending on the host
# architecture (possibly compiler related too).  A proposal has been
# put forward to fix this, but right now, we must DIY.  These names
# might not match what ultimately is used.
#
# https://github.com/python/cpython/issues/96467
try:
    from array import INT8, INT16, INT32, FLOAT, DOUBLE
except ImportError:
    INT8 = "b"
    INT16 = "h"
    FLOAT = "f"
    DOUBLE = "d"

    if array.array("l").itemsize == 4:
        INT32 = "l"
    else:
        INT32 = "i"


class AudioFormat(enum.Enum):
    LINEAR_8BIT = INT8
    LINEAR_16BIT = INT16
    LINEAR_32BIT = INT32
    FLOAT_32BIT = FLOAT
    FLOAT_64BIT = DOUBLE

    @classmethod
    def parse(cls, encoding):
        if isinstance(encoding, str):
            return cls[encoding]
        else:
            return cls(encoding)

    @classmethod
    def from_sun_encoding(cls, encoding):
        return cls[encoding.name]

    @property
    def sun_encoding(self):
        return SunAudioEncoding[self.name]


AudioEndianness = enum.Enum(
    "AudioEndianness",
    {
        "LITTLE": 0,
        "BIG": 1,
        # struct.pack will return either b"\x01\x00" (big-endian) or
        # b"\x00\x01" (little-endian)… first byte value will match
        # AudioEndianness values.
        "HOST": struct.pack("@H", 0x0100)[0],
    },
)


def parse_endianness(endianness):
    if isinstance(endianness, str):
        return AudioEndianness[endianness]
    else:
        return AudioEndianness(endianness)


_REGISTRY = Registry(
    defaults={
        "type": "aplay",
        "sample_rate": 48000,
        "sample_format": AudioFormat.LINEAR_16BIT,
        "endianness": AudioEndianness.HOST,
        "channels": 1,
        "stream_interval": 10,
    }
)


def init_audio(**kwargs):
    """
    Initialise an audio interface from the given parameters.
    """
    return _REGISTRY.init_instance(**kwargs)


def get_channel(channel):
    """
    Determine which channel is selected.  Input may be:
    - an integer: 0 is left, 1 is right… etc.  -1 is all channels (mono).
    - a numeric string: will be cast to an integer.
    - a string starting with 'l'/'L' or 'r'/'R': will be decoded to 0 or 1.
    - a string starting with 'm'/'M' or 'a'/'A': will be decoded to -1
    """
    if isinstance(channel, str):
        # Try decoding as an integer
        try:
            channel = int(channel)
        except ValueError:
            pass

    if isinstance(channel, int):
        # We have an integer already
        if channel < -1:
            raise ValueError("channel %d is not valid" % channel)

        return channel

    # This is a string, lower-case it
    channel = str(channel).lower()
    if channel == "l":
        return 0  # Left
    elif channel == "r":
        return 1  # Right
    elif channel in ("a", "m"):
        return -1  # All/Any/Mono
    else:
        raise ValueError("%r is not a recognised channel" % channel)


class AudioPlaybackInterface(object):
    """
    Base class for an audio playback interface.  This implements the buffering
    logic needed to stream audio to an underlying application.  Sub-classes
    must implement the actual playback logic.
    """

    def __init__(
        self,
        sample_rate=48000,
        channels=1,
        sample_format=AudioFormat.LINEAR_16BIT,
        endianness=AudioEndianness.HOST,
        stream_interval=10,
        loop=None,
        log=None,
    ):
        self._loop = defaults.get_loop(loop)
        self._log = defaults.get_logger(log, self.__class__.__module__)
        self._sample_rate = int(sample_rate)
        self._channels = int(channels)
        self._sample_format = AudioFormat.parse(sample_format)
        self._src = None
        self._queue = []
        self._drain = False
        self._future = None
        self._started = False
        self._finished = False
        self._stream_interval = stream_interval
        self._stream_sz = int(sample_rate * stream_interval)
        self._next_write = 0
        self._tempfile = None

        # Create an empty zero-sized buffer so we can figure out the
        # sample size.
        self._buffer = array.array(self._sample_format.value)

        # Compute the frame size
        frame_sz = channels * self._buffer.itemsize
        self._log.debug("Frame size is %d bytes", frame_sz)

        # Allocate a buffer big enough for two stream intervals
        buffer_sz = frame_sz * int(
            ((stream_interval * sample_rate * frame_sz * 2) + (frame_sz - 1))
            / frame_sz
        )
        self._log.debug(
            "%f sec at %d Hz is %d frames, (doubled) buffer size is %d bytes",
            stream_interval,
            sample_rate,
            frame_sz,
            buffer_sz,
        )
        self._buffer.extend(bytes([0]) * buffer_sz)

        # Determine if we need to swap bytes or not
        if self._sample_format is not AudioFormat.LINEAR_8BIT:
            # Cast the input.
            endianness = parse_endianness(endianness)

            # We only care if it's opposite to the native host endianness.
            self._swapped = endianness is not AudioEndianness.HOST
        else:
            # 8-bit audio has no endianness.
            self._swapped = False

        # Signals
        self.underrun = Signal()
        self.lowbuffer = Signal()

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def sample_format(self):
        return self._sample_format

    @property
    def channels(self):
        return self._channels

    @property
    def busy(self):
        """
        Audio interface is playing audio right now.
        """
        return self._started and not self._drain

    @property
    def more(self):
        """
        Returns whether there is more audio in the buffer.
        """
        return (self._src is not None) or (len(self._queue) > 0)

    def reset(self):
        """
        Reset the state of the audio interface ready to accept a new
        sequence of samples.  Must not be playing at the time!
        """
        if self.busy:
            raise RuntimeError("Trying to reset during audio playback!")

        self._log.debug("Resetting audio interface state")
        self._buffer_sz = 0
        self._rd_ptr = 0
        self._wr_ptr = 0
        self._src = None
        self._queue.clear()
        self._drain = False
        self._future = None
        self._started = False
        self._finished = False

    def enqueue(self, gen, finish=False):
        """
        Enqueue an audio sample generator to be played.  If ``finish``
        is true, set the drain flag to indicate this is the end of the
        recording being played.
        """
        if self._drain:
            raise BufferError("Finish flag is set")
        else:
            self._drain = finish
            self._queue.append(gen)

    def stop(self):
        """
        Stop playback as soon as possible.
        """
        self._queue.clear()
        self._rd_ptr = 0
        self._wr_ptr = 0
        self._buffer_sz = 0
        self._drain = True

    async def generate(self):
        """
        Generate the raw audio file to pipe to the audio process.
        """
        if self._tempfile is not None:
            self._log.debug("Removing stale temp file %r", self._tempfile)
            os.unlink(self._tempfile)

        (fd, self._tempfile) = tempfile.mkstemp(suffix=".snd")
        self._log.debug("Writing raw audio to %r", self._tempfile)

        try:
            # Asynchronously pull samples from the generator until we run out
            # of audio and write each to the file descriptor above.
            while self.more:
                # Fill the buffer with the requisite frames
                samples_rem = self._stream_sz * self.channels
                pos = 0
                self._log.debug(
                    "Reading %d frames (%d samples)",
                    self._stream_sz,
                    samples_rem,
                )
                while samples_rem:
                    if self._src is None:
                        self._log.debug(
                            "Next audio source (%d samples remain)",
                            samples_rem,
                        )
                        try:
                            self._src = self._queue.pop(0)
                        except IndexError:
                            self._log.debug("No more audio streams")
                            break

                    try:
                        sample = next(self._src)
                    except StopIteration:
                        # End of source
                        self._log.debug("Existing source is finished")
                        self._src = None
                        continue

                    self._buffer[pos] = sample
                    pos += 1
                    samples_rem -= 1

                self._log.debug("Read %d samples", pos)
                if (pos % self._channels) > 0:
                    self._log.warning("Partial frame read!")

                # Byte-swap if required
                if self._swapped:
                    self._buffer.byteswap()

                # Extract the buffer content
                block = self._buffer[0:pos].tobytes()

                # Write these to the audio file
                os.write(fd, block)

                # Yield to other event loop events
                await asyncio.sleep(0)
        finally:
            os.close(fd)

    async def start(self, wait=False):
        """
        Start playback.  Optionally wait for it to finish.  Sub-classes should
        extend this method to actually begin playback.
        """
        if self._tempfile is None:
            raise BufferError("Call .generate() first!")

        self._log.debug(
            "Beginning playback from %r",
            self._tempfile,
        )
        self._started = True
        if wait:
            self._future = self._loop.create_future()
            await self._future

    def _on_stream_end(self):
        """
        End of stream has been reached, finish up.
        """
        self._loop.call_soon(self._on_finish)

    def _on_finish(self, result=None, ex=None):
        """
        Signal the end of playback.
        """
        self._loop.create_task(self._finish(result, ex))

    async def _finish(self, result=None, ex=None):
        """
        Finish up the playback, clean up any processes.
        """
        self._drain = True

        if self._tempfile:
            self._log.debug("Removing tempfile %r", self._tempfile)
            tempfile = self._tempfile
            self._tempfile = None
            os.unlink(tempfile)

        if not self._future:
            return

        if self._future.done():
            self._log.warning("Finish event after future is resolved")
        else:
            self._log.debug("Playback is finished")
            if ex is not None:
                self._future.set_exception(ex)
            else:
                self._future.set_result(result)


class ExtProcAudioPlayback(ExternalProcess, AudioPlaybackInterface):
    """
    Implementation of the audio playback interface that uses an external
    command to play back audio.
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
        **kwargs,
    ):
        # Using explicit constructors because we need to pass different
        # things to each base class.
        ExternalProcess.__init__(
            self,
            proc_path=proc_path,
            proc_args=proc_args,
            proc_env=proc_env,
            shell=shell,
            inherit_env=inherit_env,
            cwd=cwd,
            loop=loop,
            log=log,
        )
        AudioPlaybackInterface.__init__(
            self,
            loop=loop,
            log=log,
            **kwargs,
        )

    @property
    def busy(self):
        """
        Audio interface is playing audio right now.
        """
        return super().busy or (
            (self._transport is not None)
            and (self._transport.returncode is not None)
        )

    def reset(self):
        """
        Reset the state of the audio interface ready to accept a new
        sequence of samples.  Must not be playing at the time!
        """
        super().reset()
        self._transport = None
        self._exit_status = None

    async def start(self, wait=False):
        """
        Start playback.  Optionally wait for it to finish.  Sub-classes should
        extend this method to actually begin playback.
        """
        # Explicit base classes here, because we need to call both base
        # classes' start() methods.

        # Start the subprocess
        await ExternalProcess.start(self)

        # Announce playback has started
        return await AudioPlaybackInterface.start(self, wait=wait)

    def _on_stream_end(self):
        """
        End of stream has been reached, finish up.
        """
        self._transport.get_pipe_transport(0).write_eof()
        self._log.debug("Waiting for player process to finish")

    def _on_proc_connection_lost(self, fd, exc):
        if self._finished:
            # This is expected
            self._log.debug("Connection closed for fd %d", fd)
            return
        super()._on_proc_connection_lost(fd, exc)

    def _on_proc_close(self):
        if self._transport is not None:
            self._log.debug("Cleaning up")
            self._on_finish()
        super()._on_proc_close()

    def _on_finish(self, result=None, ex=None):
        super(ExtProcAudioPlayback, self)._on_finish(result=result, ex=ex)

    def _get_command_and_args(self, extra_args):
        # Most CLI tools will expect the file name at the end, override
        # if this is not the case.
        command = super()._get_command_and_args(extra_args)
        command.append(self._tempfile)
        return command


@_REGISTRY.register
class APlayAudioPlayback(ExtProcAudioPlayback):
    """
    Implementation of the audio playback interface using the ALSA-utils
    `aplay` command.
    """

    ALIASES = ("aplay", "alsa")

    # Mapping between audio format / endianness and the ``-f`` flag used
    # by aplay
    _AUDIO_FORMATS = {
        (AudioFormat.LINEAR_8BIT, AudioEndianness.LITTLE): "S8",
        (AudioFormat.LINEAR_16BIT, AudioEndianness.LITTLE): "S16_LE",
        (AudioFormat.LINEAR_32BIT, AudioEndianness.LITTLE): "S32_LE",
        (AudioFormat.FLOAT_32BIT, AudioEndianness.LITTLE): "FLOAT_LE",
        (AudioFormat.FLOAT_64BIT, AudioEndianness.LITTLE): "FLOAT64_LE",
        (AudioFormat.LINEAR_8BIT, AudioEndianness.BIG): "S8",
        (AudioFormat.LINEAR_16BIT, AudioEndianness.BIG): "S16_BE",
        (AudioFormat.LINEAR_32BIT, AudioEndianness.BIG): "S32_BE",
        (AudioFormat.FLOAT_32BIT, AudioEndianness.BIG): "FLOAT_BE",
        (AudioFormat.FLOAT_64BIT, AudioEndianness.BIG): "FLOAT64_BE",
    }

    def __init__(
        self,
        aplay_path="aplay",
        device="plug:default",
        sample_rate=48000,
        channels=1,
        sample_format=AudioFormat.LINEAR_16BIT,
        endianness=AudioEndianness.HOST,
        loop=None,
        log=None,
        **kwargs,
    ):
        endianness = parse_endianness(endianness)
        sample_format = AudioFormat.parse(sample_format)

        # Figure out arguments
        aplay_args = [
            "-D",
            device,
            "-t",
            "raw",
            "-f",
            self._AUDIO_FORMATS[(sample_format, endianness)],
            "-r",
            str(sample_rate),
            "-c",
            str(channels),
        ]

        super().__init__(
            proc_path=aplay_path,
            proc_args=aplay_args,
            proc_env=None,
            shell=False,
            inherit_env=True,
            cwd=None,
            sample_rate=sample_rate,
            channels=channels,
            sample_format=sample_format,
            endianness=endianness,
            loop=loop,
            log=log,
            **kwargs,
        )


@_REGISTRY.register
class SoXAudioPlayback(ExtProcAudioPlayback):
    """
    Implementation of the audio playback interface using SoX.
    """

    ALIASES = ("sox",)

    # Mapping between audio format and flags needed by SoX
    _AUDIO_FORMATS = {
        AudioFormat.LINEAR_8BIT: [
            "--bits",
            "8",
            "--encoding",
            "signed-integer",
        ],
        AudioFormat.LINEAR_16BIT: [
            "--bits",
            "16",
            "--encoding",
            "signed-integer",
        ],
        AudioFormat.LINEAR_32BIT: [
            "--bits",
            "32",
            "--encoding",
            "signed-integer",
        ],
        AudioFormat.FLOAT_32BIT: [
            "--bits",
            "32",
            "--encoding",
            "floating-point",
        ],
        AudioFormat.FLOAT_64BIT: [
            "--bits",
            "64",
            "--encoding",
            "floating-point",
        ],
    }

    def __init__(
        self,
        sox_path="sox",
        device_type=None,
        device=None,
        sample_rate=48000,
        channels=1,
        sample_format=AudioFormat.LINEAR_16BIT,
        endianness=AudioEndianness.HOST,
        loop=None,
        log=None,
        **kwargs,
    ):
        endianness = parse_endianness(endianness)
        sample_format = AudioFormat.parse(sample_format)

        # Figure out arguments
        sox_args = (
            [
                "--no-show-progress",
            ]
            + self._AUDIO_FORMATS[sample_format]
            + [
                "--channels",
                str(channels),
                "--rate",
                str(sample_rate),
                "--endian",
                endianness.name.lower(),
                "--type",
                "raw",
            ]
        )

        if (device is not None) and (device_type is not None):
            sox_args.extend(("-t", device_type, device))
        else:
            sox_args.append("-d")

        super().__init__(
            proc_path=sox_path,
            proc_args=sox_args,
            proc_env=None,
            shell=False,
            inherit_env=True,
            cwd=None,
            sample_rate=sample_rate,
            channels=channels,
            sample_format=sample_format,
            endianness=endianness,
            loop=loop,
            log=log,
            **kwargs,
        )


@_REGISTRY.register
class PWCatAudioPlayback(ExtProcAudioPlayback):
    """
    Implementation of the audio playback interface using the pipewire
    ``pw-cat`` command.
    """

    ALIASES = ("pw-cat", "pipewire")

    # Mapping between audio format and the ``--format`` flag used by pw-cat.
    _AUDIO_FORMATS = {
        AudioFormat.LINEAR_8BIT: "s8",
        AudioFormat.LINEAR_16BIT: "s16",
        AudioFormat.LINEAR_32BIT: "s32",
        AudioFormat.FLOAT_32BIT: "f32",
        AudioFormat.FLOAT_64BIT: "f64",
    }

    def __init__(
        self,
        pwcat_path="pw-cat",
        target="auto",
        mediatype="Audio",
        mediarole="Music",
        mediacategory="Playback",
        latency=0.5,
        sample_rate=48000,
        channels=1,
        properties=None,
        quality=4,
        volume=1.0,
        sample_format=AudioFormat.LINEAR_16BIT,
        # pw-cat uses the same endianness as the host computer
        # see https://fosstodon.org/@pipewire/112910467772565426
        endianness=AudioEndianness.HOST,
        remote=None,
        loop=None,
        log=None,
        **kwargs,
    ):
        sample_format = AudioFormat.parse(sample_format)

        # Figure out arguments
        pwcat_args = [
            "--target=%s" % target,
            "--media-type=%s" % mediatype,
            "--media-role=%s" % mediarole,
            "--media-category=%s" % mediacategory,
            "--latency=%d" % int((latency * 1000) + 0.5),
            "--quality=%d" % int(quality),
            "--format=%s" % self._AUDIO_FORMATS[sample_format],
            "--rate=%d" % int(sample_rate),
            "--channels=%d" % int(channels),
            "--volume=%f" % float(volume),
        ]

        if properties is not None:
            pwcat_args.append("--properties=%s" % json.dumps(properties))

        pwcat_args.append("--playback")

        super().__init__(
            proc_path=pwcat_path,
            proc_args=pwcat_args,
            proc_env=None,
            shell=False,
            inherit_env=True,
            cwd=None,
            sample_rate=sample_rate,
            channels=channels,
            sample_format=sample_format,
            endianness=endianness,
            loop=loop,
            log=log,
            **kwargs,
        )


class ChannelMap(object):
    """
    Base class for a channel map.  Abstract class.
    """

    def __init__(self, amplitude=1.0):
        """
        Define a channel mapping with the master amplitude given.
        """
        self._amplitude = float(amplitude)

    def get_mapping(self, input_channels, output_channels):
        """
        Returns a generator that yields tuples of the form:

        ``(input_ch, {output_ch1: amplitude, …})``
        """
        raise NotImplementedError("Implement in %s" % self.__class__.__name__)


class DirectChannelMap(ChannelMap):
    """
    Directly map the input channels to the output.  Optionally wrap back to
    the first input channel when we run out of outputs.
    """

    def __init__(self, start_ch=0, wrap=False, **kwargs):
        super().__init__(**kwargs)
        self._wrap = wrap
        self._start_ch = start_ch

    def get_mapping(self, input_channels, output_channels):
        if self._wrap:
            channels = output_channels
        else:
            channels = min(input_channels, output_channels)

        for ch in range(channels):
            yield (
                ch % input_channels,
                {
                    (ch + self._start_ch) % output_channels: self._amplitude,
                },
            )


class DictChannelMap(ChannelMap):
    """
    Arbitrary mapping specified by a dict.  Keys are the input channel
    numbers starting at 0.  Output channels may be given in one of a few
    forms:

    - ``None``: Drop channel, do not map
    - ``ch``: Map to output channel, amplitude is the master amplitude
    - ``(ch,)``: Treated the same as a bare integer, map to that channel.
    - ``(ch, amplitude)``: Map to output channel, amplitude is the product
      of this amplitude and the master amplitude
    - ``[…]``: Lists of integers and tuples maps the channel to multiple
      places simultaneously.
    """

    def __init__(self, mapping=None, **kwargs):
        super().__init__(**kwargs)
        self._mapping = {}
        if mapping is not None:
            for ch, val in mapping.items():
                if val is None:
                    continue

                if isinstance(val, int):
                    val = (val,)

                if isinstance(val, list):
                    val = [tuple(v) for v in val]
                else:
                    val = [tuple(val)]

                for v in val:
                    if v is None:
                        self.unmap(ch)
                    elif len(v) == 1:
                        self.map(
                            input_channel=ch,
                            output_channel=v[0],
                            amplitude=1.0,
                        )
                    else:
                        self.map(
                            input_channel=ch,
                            output_channel=v[0],
                            amplitude=v[1],
                        )

    def map(self, input_channel, output_channel, amplitude=1.0):
        """
        Map a given input channel to the specified output channel at the
        specified amplitude.
        """
        in_map = self._mapping.setdefault(input_channel, {})
        in_map[output_channel] = amplitude
        return self

    def unmap(self, input_channel):
        """
        Remove all mappings to an input channel.
        """
        self._mapping.pop(input_channel, None)

    def get_mapping(self, input_channels, output_channels):
        for in_channel in range(input_channels):
            yield (
                in_channel,
                dict(self._get_mapping(in_channel, output_channels)),
            )

    def _get_mapping(self, in_channel, output_channels):
        try:
            in_map = self._mapping[in_channel]
        except KeyError:
            return

        for out_channel in range(output_channels):
            try:
                yield (out_channel, in_map[out_channel])
            except KeyError:
                pass


class AudioMixer(object):
    """
    A simple audio mixer object.  This takes multiple audio generator sources
    and sums them together into a single multiplexed stream with the
    designated channel count.  All sources are assumed to be the same format
    and sample rate.
    """

    def __init__(self, channels=1, sample_format=AudioFormat.LINEAR_16BIT):
        self._channels = channels
        self._sources = {}
        self._source_map = {}
        self._src_idx = 0
        self._float = sample_format in (
            AudioFormat.FLOAT_32BIT,
            AudioFormat.FLOAT_64BIT,
        )

    def add_source(
        self,
        source,
        channels,
        channelmap=None,
        amplitude=1.0,
        wrap=False,
        start_ch=0,
    ):
        """
        Add the given source, optionally specifying a channel map
        that maps the input channels to the output.
        """
        if channelmap is None:
            channelmap = DirectChannelMap(
                amplitude=amplitude, wrap=wrap, start_ch=start_ch
            )
        elif isinstance(channelmap, dict):
            channelmap = DictChannelMap(channelmap, amplitude=amplitude)

        src_idx = self._src_idx
        self._src_idx += 1
        self._sources[src_idx] = (source, channels)

        for in_channel, outputs in channelmap.get_mapping(
            channels, self._channels
        ):
            for out_ch, amplitude in outputs.items():
                self._source_map.setdefault(out_ch, {}).setdefault(
                    src_idx, {}
                )[in_channel] = amplitude

        return self

    def generate(self):
        """
        Generate the mixed samples.
        """
        todo = set(self._sources.keys())
        while todo:
            # Fetch all the samples for this frame
            samples = {}
            for src_idx, (source, channels) in self._sources.items():
                try:
                    samples[src_idx] = [next(source) for n in range(channels)]
                except StopIteration:
                    # This one is finished
                    todo.discard(src_idx)
                    continue

            for channel in range(self._channels):
                try:
                    ch_map = self._source_map[channel]
                except KeyError:
                    # Nothing here
                    if self._float:
                        yield 0.0
                    else:
                        yield 0
                    continue

                # Gather all the samples for this channel
                ch_samples = []
                for src_idx, src_map in ch_map.items():
                    try:
                        src_samples = samples[src_idx]
                    except KeyError:
                        continue

                    for in_channel, amplitude in src_map.items():
                        try:
                            src_sample = src_samples[in_channel]
                        except IndexError:
                            src_sample = 0

                        ch_samples.append(src_sample * amplitude)

                # Sum them
                if ch_samples:
                    out = sum(ch_samples) / len(ch_samples)
                    if self._float:
                        yield out
                    else:
                        yield int(out + 0.5)
                elif self._float:
                    yield 0.0
                else:
                    yield 0


if __name__ == "__main__":
    # Demo program to test the interface
    import argparse
    import logging
    from .sunaudio import SunAudioDecoder
    from .oscillator import Oscillator
    from .cw import CWString

    async def main():
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "--audiocfg",
            nargs=2,
            action="append",
            default=[],
            help="Set an audio configuration option",
        )

        ap_sub = ap.add_subparsers(required=True)

        player_ap = ap_sub.add_parser(
            "play", help="Play a Sun Audio (.au) file"
        )
        player_ap.set_defaults(mode="play")
        player_ap.add_argument("audiofile", type=str, help="Audio file")

        tonegen_ap = ap_sub.add_parser("tone", help="Generate a tone")
        tonegen_ap.set_defaults(mode="tone")
        tonegen_ap.add_argument("--sample-rate", type=int, default=48000)
        tonegen_ap.add_argument(
            "--sample-fmt",
            type=str,
            default="LINEAR_16BIT",
            choices=(
                "LINEAR_8BIT",
                "LINEAR_16BIT",
                "LINEAR_32BIT",
                "FLOAT_32BIT",
                "FLOAT_64BIT",
            ),
        )
        tonegen_ap.add_argument(
            "--freq", type=int, default=800, help="Frequency"
        )
        tonegen_ap.add_argument("duration", type=float, help="Duration")

        cwgen_ap = ap_sub.add_parser("cw", help="Generate CW")
        cwgen_ap.set_defaults(mode="cw")
        cwgen_ap.add_argument("--sample-rate", type=int, default=48000)
        cwgen_ap.add_argument(
            "--sample-fmt",
            type=str,
            default="LINEAR_16BIT",
            choices=(
                "LINEAR_8BIT",
                "LINEAR_16BIT",
                "LINEAR_32BIT",
                "FLOAT_32BIT",
                "FLOAT_64BIT",
            ),
        )
        cwgen_ap.add_argument(
            "--freq", type=int, default=800, help="Frequency"
        )
        cwgen_ap.add_argument(
            "--rise", type=float, default=None, help="Rise time in seconds"
        )
        cwgen_ap.add_argument(
            "--fall", type=float, default=None, help="Fall time in seconds"
        )
        cwgen_ap.add_argument(
            "--dit-period",
            type=float,
            default=0.120,
            help="dit period in seconds",
        )
        cwgen_ap.add_argument("text", type=str, help="CW text to encode")

        args = ap.parse_args()

        logging.basicConfig(level=logging.DEBUG)

        if args.mode == "play":
            inputstream = SunAudioDecoder(args.audiofile)
            sample_rate = inputstream.header.sample_rate
            channels = inputstream.header.channels
            try:
                fmt = getattr(AudioFormat, inputstream.header.encoding.name)
            except AttributeError:
                logging.error(
                    "Unsupported sample format %s",
                    inputstream.header.encoding.name,
                )
                raise

        elif args.mode in ("tone", "cw"):
            genfmt = getattr(SunAudioEncoding, args.sample_fmt.upper())
            fmt = getattr(AudioFormat, args.sample_fmt.upper())
            oscillator = Oscillator(args.sample_rate, genfmt)
            sample_rate = args.sample_rate
            channels = 1

        player_cfg = dict(args.audiocfg)
        logging.debug("Audio config: %r", player_cfg)

        player = init_audio(
            sample_rate=sample_rate,
            channels=channels,
            sample_format=fmt,
            **player_cfg,
        )
        if args.mode == "play":
            player.enqueue(inputstream.read(), finish=True)
        elif args.mode == "tone":
            if not args.duration:
                # Indefinitely-long tone
                asyncio.get_event_loop().add_reader(
                    0, lambda *a, **kwa: oscillator.stop()
                )
                logging.info("Indefinite length tone, press ENTER to stop.")
            player.enqueue(
                oscillator.generate(args.freq, args.duration or None),
                finish=True,
            )
        elif args.mode == "cw":
            player.enqueue(
                CWString.from_string(args.text).modulate(
                    oscillator=oscillator,
                    frequency=args.freq,
                    dit_period=args.dit_period,
                    risetime=args.rise,
                    falltime=args.fall,
                ),
                finish=True,
            )

        await player.generate()
        await player.start(wait=True)
        logging.info("Done")

    asyncio.run(main())
