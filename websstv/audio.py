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
        endianness=AudioEndianness.LITTLE,
        buffer_sz=None,
        stream_interval=0.1,
        loop=None,
        log=None,
    ):
        self._loop = defaults.get_loop(loop)
        self._log = defaults.get_logger(log, self.__class__.__module__)
        self._sample_rate = int(sample_rate)
        self._channels = int(channels)
        self._sample_format = AudioFormat(sample_format)
        self._buffer = array.array(self._sample_format.value)

        if buffer_sz is None:
            # Buffer four seconds worth of audio
            buffer_sz = int(
                sample_rate
                * channels
                * self._buffer.itemsize
                * 4
                * stream_interval
            )
            self._log.debug("Using buffer size of %d bytes", buffer_sz)

        self._buffer.extend(bytes([0]) * buffer_sz)
        self._buffer_sz = 0
        self._rd_ptr = 0
        self._wr_ptr = 0
        self._src = None
        self._queue = []
        self._drain = False
        self._future = None
        self._first_write = False
        self._finished = False
        self._stream_interval = stream_interval
        self._stream_sz = int(sample_rate * stream_interval)

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
        self.lowbuffer = Signal()

    @property
    def more(self):
        """
        Returns whether there is more audio in the buffer.
        """
        return (self._src is not None) or (len(self._queue) > 0)

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

    async def start(self, wait=False):
        """
        Start playback.  Optionally wait for it to finish.  Sub-classes should
        extend this method to actually begin playback.
        """
        if not self.more:
            raise BufferError("Enqueue an audio source first!")

        self._log.debug(
            "Beginning playback, sending %d frames every %f sec",
            self._stream_sz,
            self._stream_interval,
        )
        self._first_write = False
        self._loop.call_soon(self._send_next)
        if wait:
            self._future = self._loop.create_future()
            await self._future

    def _send_next(self):
        """
        Send the next block of audio.
        """
        try:
            start_ts = self._loop.time()
            read_sz = self._stream_sz
            if not self._first_write:
                # Read double for the first block
                self._log.debug("Performing double-size read for first block")
                read_sz *= 2
                self._first_write = True

            # Read a block of samples from the buffer
            block = self._buffer_rd(read_sz).tobytes()

            # Write these to the audio device
            self._log.debug("Sending %d bytes of data", len(block))
            self._write_audio(block)

            if self.more:
                # More to come, re-schedule, giving ourselves a little buffer
                # time to avoid underruns.
                duration = self._loop.time() - start_ts
                delay = self._stream_interval - (duration * 3)
                self._log.debug(
                    "More to send, call again in %f sec",
                    delay,
                )
                self._loop.call_later(delay, self._send_next)
            else:
                self._log.debug("Audio stream has finished")
                self._finished = True
                self._loop.call_soon(self._on_stream_end)
        except Exception as ex:
            self._log.exception("Failed to send audio block")
            self._on_finish(ex=ex)

    def _write_audio(self, audiodata):
        """
        Write the specified audio samples to the audio device/subsystem.
        This must be implemented by a subclass.
        """
        raise NotImplementedError("Implement in %s" % self.__class__.__name__)

    @property
    def _buffer_full(self):
        """
        Return true if the buffer is full
        """
        return self._buffer_sz == len(self._buffer)

    @property
    def _len_frames_buffered(self):
        """
        Return the number of frames buffered
        """
        return self._buffer_sz // self._channels

    def _buffer_wr(self, samples, limit=None):
        """
        Write samples from the given sequence into the buffer until
        we run out of samples or space.  Return whether we stopped because
        our buffer filled up.
        """
        if limit is None:
            self._log.debug(
                "Read all possible samples from source "
                "(rd=%d wr=%d sz=%d)",
                self._rd_ptr,
                self._wr_ptr,
                self._buffer_sz,
            )
        else:
            self._log.debug(
                "Read %d samples from source (rd=%d wr=%d sz=%d)",
                limit,
                self._rd_ptr,
                self._wr_ptr,
                self._buffer_sz,
            )

        for sample in samples:
            next_wr = (self._wr_ptr + 1) % len(self._buffer)
            if self._buffer_full:
                # Buffer is full
                self._log.debug(
                    "Buffer is now full (rd=%d wr=%d sz=%d)",
                    self._rd_ptr,
                    self._wr_ptr,
                    self._buffer_sz,
                )
                return True

            # There is space, write
            self._buffer[next_wr] = sample
            self._wr_ptr = next_wr
            self._buffer_sz += 1
            if limit is not None:
                limit -= 1
                if limit <= 0:
                    self._log.debug(
                        "Read limit reached (rd=%d wr=%d sz=%d)",
                        self._rd_ptr,
                        self._wr_ptr,
                        self._buffer_sz,
                    )
                    return True

        # We got to the end of the sequence without filling up
        self._log.debug(
            "Generator has finished (rd=%d wr=%d sz=%d)",
            self._rd_ptr,
            self._wr_ptr,
            self._buffer_sz,
        )
        return False

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

    def _buffer_rd(self, frames):
        """
        Read up to ``frames`` frames of samples from the buffer.
        """
        self._log.debug(
            "Reading %d frames (rd=%d wr=%d sz=%d)",
            frames,
            self._rd_ptr,
            self._wr_ptr,
            self._buffer_sz,
        )

        # Make space for the frames
        output = array.array(
            self._sample_format.value,
            bytes([0]) * frames * self._buffer.itemsize * self._channels,
        )
        pos = 0
        remain = frames * self._channels
        buffer_sz = len(self._buffer)

        while remain:
            if self._len_frames_buffered < self._stream_sz:
                # We are low on samples, read some data in
                self._log.debug("Low watermark reached, performing a read")
                while self.more:
                    if self._src is None:
                        self._log.debug("Next audio source")
                        self._src = self._queue.pop(0)

                    full = self._buffer_wr(self._src)
                    if full:
                        break
                    else:
                        # This source is depleted
                        self._log.debug("Audio source finished")
                        self._src = None
            elif (not self._buffer_full) and self.more:
                self._buffer_wr(self._src, self._stream_sz)

            if not self._drain and (
                self._len_frames_buffered < self._stream_sz
            ):
                self._log.debug("Buffer still low")
                self.lowbuffer.emit()

            if self._buffer_sz == 0:
                # We're out of data
                if self._drain:
                    self._log.debug("Playback complete")
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
            self._buffer_sz -= sz

        if remain > 0:
            # Truncate the array we have
            output = output[0:-remain]

        # Perform byte swap if appropriate
        if self._swapped:
            output.byteswap()

        return output


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

    def _write_audio(self, audiodata):
        """
        Write the specified audio samples to the audio device/subsystem.
        """
        # Write to standard input
        self._transport.get_pipe_transport(0).write(audiodata)

    def _on_finish(self, result=None, ex=None):
        super(ExtProcAudioPlayback, self)._on_finish(result=result, ex=ex)


class APlayAudioPlayback(ExtProcAudioPlayback):
    """
    Implementation of the audio playback interface using the ALSA-utils
    `aplay` command.
    """

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
        endianness=AudioEndianness.LITTLE,
        loop=None,
        log=None,
        **kwargs,
    ):
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
            "-",
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
            log=loop,
            **kwargs,
        )


if __name__ == "__main__":
    # Demo program to test the interface
    import argparse
    import asyncio
    import logging
    from .sunaudio import SunAudioDecoder

    async def main():
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "--aplay-path",
            default="aplay",
            type=str,
            help="Path to aplay binary",
        )
        ap.add_argument(
            "--device",
            default="plug:default",
            type=str,
            help="Audio device to send audio to",
        )
        ap.add_argument("audiofile", type=str, help="Audio file")

        args = ap.parse_args()

        logging.basicConfig(level=logging.DEBUG)

        inputstream = SunAudioDecoder(args.audiofile)
        try:
            fmt = getattr(AudioFormat, inputstream.header.encoding.name)
        except AttributeError:
            logging.error(
                "Unsupported sample format %s",
                inputstream.header.encoding.name,
            )
            raise

        player = APlayAudioPlayback(
            aplay_path=args.aplay_path,
            device=args.device,
            sample_rate=inputstream.header.sample_rate,
            channels=inputstream.header.channels,
            sample_format=fmt,
        )
        player.enqueue(inputstream.read())
        await player.start(wait=True)
        logging.info("Done")

    asyncio.run(main())
