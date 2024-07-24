#!/usr/bin/env python3

"""
Very rough Sun audio file generator.  To plug a hole in Python's API when the
``sunau`` module is removed in Python 3.13.

Sun audio is used because the format is very simple to implement to fulfill
the needs of this application.
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

# Information from https://en.wikipedia.org/wiki/Au_file_format
# For simplicity, we shall only support a few basic formats.
#
# Why Sun audio?
# 1. It's dead simple to implement.  Seriously, go have a look at the above
#    page, then compare it to https://en.wikipedia.org/wiki/WAV and tell me
#    which one you'd rather implement.
# 2. `slowrxd` uses it because of (1).
# 3. I wish to be consistent in the use of audio file formats.  Unless someone
#    has a burning desire to implement .wav format in `slowrxd`, I'd rather
#    just do this.

import struct
import enum
from collections import namedtuple


SUNAU_MAGIC_OFFSET_HEADER = struct.Struct(">LL")
SUNAU_REMAIN_HEADER = struct.Struct(">LLLL")
SUNAU_MAGIC = 0x2E736E64
SUNAU_SIZE_UNKNOWN = 0xFFFFFFFF
SUNAU_EMPTY_ANNOTATION = bytes([0, 0, 0, 0])


class SunAudioEncoding(enum.Enum):
    """
    The possible Sun audio encoding formats.  Not all here are supported by
    this module, we will focus on just the signed linear and floating-point
    types.
    """

    UNSPECIFIED = 0  # Unspecified
    G711U = 1  # 8-bit G.711 μ-law
    LINEAR_8BIT = 2  # 8-bit linear PCM
    LINEAR_16BIT = 3  # 16-bit linear PCM
    LINEAR_24BIT = 4  # 24-bit linear PCM
    LINEAR_32BIT = 5  # 32-bit linear PCM
    FLOAT_32BIT = 6  # 32-bit IEEE floating point
    FLOAT_64BIT = 7  # 64-bit IEEE floating point
    FRAGMENTED = 8  # Fragmented sample data
    NESTED = 9  # Nested (unclear format)
    DSP_PROGRAM = 10  # DSP program
    FIXED_8BIT = 11  # 8-bit fixed point
    FIXED_16BIT = 12  # 16-bit fixed point
    FIXED_24BIT = 13  # 24-bit fixed point
    FIXED_32BIT = 14  # 32-bit fixed point
    UNASSIGNED = 15  # (Unassigned)
    DISPLAY = 16  # non-audio display data
    ULAW_SQUELCH = 17  # μ-law Squelch format[6]
    LINEAR_16BIT_EMPH = 18  # 16-bit linear with emphasis
    LINEAR_16BIT_COMP = 19  # 16-bit linear compressed
    LINEAR_16BIT_EMPH_COMP = 20  # 16-bit linear with emphasis and compression
    MUSICKIT_DSP_COMMANDS = 21  # Music kit DSP commands
    MUSICKIT_DSP_COMMANDS_SAMPLES = 22  # Music kit DSP commands samples
    G721 = 23  # ITU-T G.721 4-bit ADPCM
    G722 = 24  # ITU-T G.722 SB-ADPCM
    G723_3BIT = 25  # ITU-T G.723 3-bit ADPCM
    G723_5BIT = 26  # ITU-T G.723 5-bit ADPCM
    G711A = 27  # 8-bit G.711 A-law


SunAudioEncodingSpec = namedtuple(
    "SunAudioEncodingSpec", ["pythontype", "bits", "encode", "decode"]
)


# Convenience structs for different sample types
_S8_STRUCT = struct.Struct(">b")
_S16_STRUCT = struct.Struct(">h")
_S24_STRUCT = struct.Struct(">bH")
_S32_STRUCT = struct.Struct(">l")
_F32_STRUCT = struct.Struct(">f")
_F64_STRUCT = struct.Struct(">d")


def _s24decode(data):
    """
    Decode a sample as signed 24-bit linear
    """
    # This is a nuisance as we don't have a 24-bit type, we must encode an
    # 8-bit and a 16-bit value
    (hi, lo) = _S24_STRUCT.unpack(data)
    return (hi << 16) | lo


# The audio encoding specifications.  This also defines the supported file
# encodings we will allow.
_ENCODING_SPECS = {
    # Linear PCM types
    SunAudioEncoding.LINEAR_8BIT: SunAudioEncodingSpec(
        pythontype=int,
        bits=8,
        encode=lambda sample: _S8_STRUCT.pack(sample),
        decode=lambda data: _S8_STRUCT.unpack(data)[0],
    ),
    SunAudioEncoding.LINEAR_16BIT: SunAudioEncodingSpec(
        pythontype=int,
        bits=16,
        encode=lambda sample: _S16_STRUCT.pack(sample),
        decode=lambda data: _S16_STRUCT.unpack(data)[0],
    ),
    SunAudioEncoding.LINEAR_24BIT: SunAudioEncodingSpec(
        pythontype=int,
        bits=24,
        encode=lambda sample: _S24_STRUCT.pack(sample >> 16, sample & 0xFFFF),
        decode=_s24decode,
    ),
    SunAudioEncoding.LINEAR_32BIT: SunAudioEncodingSpec(
        pythontype=int,
        bits=32,
        encode=lambda sample: _S32_STRUCT.pack(sample),
        decode=lambda data: _S32_STRUCT.unpack(data)[0],
    ),
    # Floating-point representations.
    SunAudioEncoding.FLOAT_32BIT: SunAudioEncodingSpec(
        pythontype=float,
        bits=32,
        encode=lambda sample: _F32_STRUCT.pack(sample),
        decode=lambda data: _F32_STRUCT.unpack(data)[0],
    ),
    SunAudioEncoding.FLOAT_64BIT: SunAudioEncodingSpec(
        pythontype=float,
        bits=64,
        encode=lambda sample: _F64_STRUCT.pack(sample),
        decode=lambda data: _F64_STRUCT.unpack(data)[0],
    ),
}


def get_spec(encoding):
    """
    Fetch the specification that corresponds to the encoding format.
    """
    encoding = SunAudioEncoding(encoding)

    try:
        return _ENCODING_SPECS[SunAudioEncoding(encoding)]
    except KeyError:
        raise ValueError("Encoding %s is not supported" % encoding.name)


class SunAudioDecoder(object):
    """
    Decode a Sun Audio file into a sequence of samples.
    """

    def __init__(
        self,
        inputstream,
        encoding=None,
        annotation_encoding="UTF-8",
        buffer_sz=4096,
    ):
        if isinstance(inputstream, str):
            inputstream = open(inputstream, "rb")

        self.header = SunAudioHeader.parse_from(
            inputstream, annotation_encoding=annotation_encoding
        )

        self._spec = get_spec(self.header.encoding)
        self._sample_sz = self._spec.bits // 8
        self._frame_sz = self._sample_sz * self.header.channels
        self._input = inputstream
        self._buffer_sz = buffer_sz

    def seek(self, frame_idx):
        """
        Seek to the specified frame.
        """
        self._input.seek(self.header.offset + (self._frame_sz * frame_idx))

    def read(self, frames, encoding=None):
        """
        Read the specified number of frames from the input stream.
        """
        if encoding is None:
            remain_sz = frames * self.header.channels * self._frame_sz
            while remain_sz > 0:
                # Clamp the read to the buffer size
                read_sz = min(remain_sz, self._buffer_sz)

                # Clamp to a whole frame
                read_sz -= read_sz % self._frame_sz

                data = self._input.read(read_sz)
                data_sz = len(data)

                for offset in range(0, data_sz, self._sample_sz):
                    yield self._spec.decode(
                        data[offset : offset + self._sample_sz]
                    )

                if data_sz < read_sz:
                    # We didn't get there, assume this is it.
                    return

                # Reduce the amount to read
                remain_sz -= data_sz
        else:
            encoding = SunAudioEncoding(encoding)
            yield from _transform_samples(
                self.read(frames, encoding=None),
                self.header.encoding,
                encoding,
            )


class SunAudioEncoder(object):
    """
    Encode a sequence of samples as a Sun Audio file to a file-like object.
    """

    def __init__(
        self,
        outputstream,
        sample_rate,
        channels,
        encoding,
        annotation=None,
        total_frames=None,
        annotation_encoding="UTF-8",
        buffer_sz=4096,
    ):

        if isinstance(outputstream, str):
            # This is a file name to be written
            outputstream = open(outputstream, "wb")

        self.header = SunAudioHeader(
            encoding=encoding,
            sample_rate=sample_rate,
            channels=channels,
            annotation=annotation,
            annotation_encoding=annotation_encoding,
        )
        self._remain = total_frames
        self._output = outputstream
        self._buffer = b""
        self._buffer_sz = buffer_sz
        self._header_written = False
        self._spec = get_spec(encoding)
        self._written = 0

        if total_frames is not None:
            # We can compute the total size here
            audio_sz = (total_frames * channels * self._spec.bits) // 8
            self.header.length = audio_sz + self.header.offset

    def __del__(self):
        return self.close()

    def flush(self):
        """
        Flush the currently buffered data to the output.
        """
        if self._buffer:
            self._output.write(self._buffer)
            self._output.flush()
            self._written += len(self._buffer)
            self._buffer = b""

    def close(self):
        """
        Finish writing the file.
        """
        if self._output is None:
            # Already closed
            return

        # Finish writing all audio data
        self.flush()

        if (self._remain is None) or (self._remain > 0):
            # Size in the file is not correct, can we fix it?
            try:
                seekable = self._output.seekable()
            except AttributeError:
                # Assume not
                seekable = False

            if seekable:
                # We can fix this.  Go back to the start of the file.
                self._output.seek(0)

                # Update the file length in the header
                self.header.length = self._written

                # Write the new header
                self._output.write(bytes(self.header))
                self._output.flush()

        # Close the file
        self._output.close()
        self._output = None

    def write_samples(self, samples, encoding=None):
        """
        Write the given samples, which are in the specified encoding.  The
        assumption will be made that the channel count is the same.  If the
        encoding is not given, it is assumed to match that of the header.
        """
        # Write out the header if we have not already done so
        if not self._header_written:
            self._write(bytes(self.header))
            self._header_written = True

        if encoding is None:
            encoding = self.header.encoding

        for sample in _transform_samples(
            samples, encoding, self.header.encoding
        ):
            data = self._spec.encode(sample)

            if self._remain is not None:
                data_sz = len(data)
                if self._remain < data_sz:
                    raise ValueError("File size exceeded")
                self._remain -= data_sz

            self._write(data)

        def _write(self, data):
            self._buffer += data
            if len(self._buffer) > self._buffer_sz:
                # Flush the current data
                self.flush()


def _transform_samples(samples, input_encoding, output_encoding):
    """
    Transform the input samples into the expected output encoding.
    """
    input_spec = get_spec(input_encoding)
    output_spec = get_spec(output_encoding)

    if output_spec.pythontype is not float:
        output_scale = 2 ** (output_spec.bits - 1)
        output_max = output_scale - 1
        output_min = -output_scale
    else:
        output_scale = 1.0
        output_max = 1.0
        output_min = -1.0

    # Apply clamping to the output
    output_cast = lambda sample: max(
        output_min, min(output_max, output_spec.pythontype(sample))
    )

    if input_spec.pythontype is float:
        # Input is float, if the output is float, we can pass it through
        # unchanged
        if output_spec.pythontype is float:
            # No change needed
            input_cast = lambda sample: sample
        else:
            # Oookay, we need to scale it!
            input_cast = lambda sample: float(sample) * output_scale
    else:
        # Input is signed linear PCM
        if output_spec.pythontype is float:
            # We need to scale it to a float
            input_scale = 2 ** (input_spec.bits - 1)
            input_cast = lambda sample: float(sample) / input_scale
        else:
            # Both input and output are signed linear
            if input_spec.bits > output_spec.bits:
                # Right shift the difference
                delta = input_spec.bits - output_spec.bits
                input_cast = lambda sample: sample >> delta
            elif input_spec.bits == output_spec.bits:
                # No change needed
                input_cast = lambda sample: sample
            else:
                # Left-shift the difference
                delta = output_spec.bits - input_spec.bits
                input_cast = lambda sample: sample << delta

    for sample in samples:
        yield output_cast(input_cast(sample))


class SunAudioHeader(object):
    """
    The header of a Sun audio file.  The length is the file size in bytes, not
    the number of samples.
    """

    @staticmethod
    def _pad_annotation(annotation):
        """
        Pad an annotation byte string to a whole number of 4 bytes, ensuring
        there is at least one null byte at the end.
        """
        # Figure out the padded annotation length as a multiple
        # of 4-byte words.
        annotation_words = (len(annotation) + 3) // 4

        # Pad the annotation out
        annotation = annotation.ljust(annotation_words * 4, bytes([0]))

        # If the last character is not a null, pad it out by another word
        if annotation[-1] != 0:
            annotation += SUNAU_EMPTY_ANNOTATION

        return annotation

    @staticmethod
    def detect_header_size(data):
        """
        Pick out the header size from the header data.  This requires at least
        8 bytes of header data to be read.
        """
        (magic, offset) = SUNAU_MAGIC_OFFSET_HEADER.unpack(
            data[0 : SUNAU_MAGIC_OFFSET_HEADER.size]
        )
        if magic != SUNAU_MAGIC:
            raise ValueError("This is not a Sun Audio file")

        return offset

    @classmethod
    def parse_from(cls, inputstream, annotation_encoding="UTF-8"):
        """
        Parse the Sun audio header from the file-like object given.
        """
        # Read enough to decode the magic and offset
        data = inputstream.read(SUNAU_MAGIC_OFFSET_HEADER.size)
        offset = cls.detect_header_size(data)

        # We now know where the data starts, read the rest of the header
        data = inputstream.read(offset - SUNAU_MAGIC_OFFSET_HEADER.size)
        return cls._parse_remainder(offset, data, annotation_encoding)

    @classmethod
    def parse(cls, data, annotation_encoding="UTF-8"):
        """
        Parse a raw Sun audio header from the byte string given.  The header
        is assumed to be complete.
        """
        offset = cls.detect_header_size(data)

        # Discard magic / offset, as these are decoded
        data = data[SUNAU_MAGIC_OFFSET_HEADER.size :]

        # Decode the rest
        return cls._parse_remainder(offset, data, annotation_encoding)

    @classmethod
    def _parse_remainder(cls, offset, data, annotation_encoding):
        (length, encoding, sample_rate, channels) = (
            SUNAU_REMAIN_HEADER.unpack(data[0 : SUNAU_REMAIN_HEADER.size])
        )

        annotation_bytes = data[SUNAU_REMAIN_HEADER.size :]
        return cls(
            encoding=SunAudioEncoding(encoding),
            sample_rate=sample_rate,
            channels=channels,
            annotation=annotation_bytes,
            annotation_encoding=annotation_encoding,
            length=length,
            offset=offset,
        )

    def __init__(
        self,
        encoding,
        sample_rate,
        channels,
        annotation=None,
        annotation_encoding="UTF-8",
        length=SUNAU_SIZE_UNKNOWN,
        offset=None,
    ):
        self._encoding = SunAudioEncoding(encoding)
        self._sample_rate = int(sample_rate)
        self._channels = int(channels)

        if isinstance(annotation, bytes):
            self._annotation_bytes = self._pad_annotation(annotation)
            self._annotation = None
        else:
            self._annotation = annotation
            self._annotation_bytes = None

        self._annotation_encoding = annotation_encoding
        self._length = length
        self._offset = offset

    @property
    def encoding(self):
        return self._encoding

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def channels(self):
        return self._channels

    @property
    def annotation(self):
        """
        Return the annotation, or decode it as a string from the
        byte-representation of it.
        """
        if (self._annotation is None) and (
            self._annotation_bytes is not None
        ):
            # Assume it's a string
            self._annotation = self._annotation_bytes.rstrip(
                bytes([0])
            ).decode(self.annotation_encoding)
        return self._annotation

    @property
    def annotation_bytes(self):
        """
        Return the byte-encoded annotation.
        """
        if self._annotation_bytes is None:
            if self._annotation is None:
                # Empty annotation
                annotation = SUNAU_EMPTY_ANNOTATION
            elif isinstance(self._annotation, str):
                # Encode as a string
                annotation = self._annotation.encode(self.annotation_encoding)
            else:
                # Assume it can be encoded as bytes
                annotation = bytes(self._annotation)

            # Pad the annotation out
            self._annotation_bytes = self._pad_annotation(annotation)

        return self._annotation_bytes

    @property
    def annotation_encoding(self):
        return self._annotation_encoding

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, new_length):
        self._length = new_length

    @property
    def offset(self):
        if self._offset is None:
            # Figure it out from the annotation size and the header size
            self._offset = (
                SUNAU_MAGIC_OFFSET_HEADER.size
                + SUNAU_REMAIN_HEADER.size
                + len(self.annotation_bytes)
            )

        return self._offset

    @property
    def is_known_length(self):
        return self._length != SUNAU_SIZE_UNKNOWN

    def __bytes__(self):
        # Encode the header bytes
        return (
            SUNAU_MAGIC_OFFSET_HEADER.pack(SUNAU_MAGIC, self.offset)
            + SUNAU_REMAIN_HEADER.pack(
                self.length,
                self.encoding.value,
                self.sample_rate,
                self.channels,
            )
            + self.annotation_bytes
        )
