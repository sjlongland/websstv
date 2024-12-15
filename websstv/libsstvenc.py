#!/usr/bin/env python3

"""
Python wrapper around libsstvenc.
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

import ctypes
import enum
import math
import os


def _errno_to_oserror(code):
    """
    Inspect the returned error code, and raise a suitable OSError if it's
    negative.
    """

    if code < 0:
        raise OSError(os.strerror(-code))


class _LibSSTVEncOscillator(ctypes.Structure):
    _fields_ = [
        ("amplitude", ctypes.c_double),
        ("offset", ctypes.c_double),
        ("output", ctypes.c_double),
        ("sample_rate", ctypes.c_uint32),
        ("phase", ctypes.c_uint32),
        ("phase_inc", ctypes.c_uint32),
    ]


class LibSSTVEncOscillator(object):
    """
    libsstvenc Oscillator object.  This is a convenience wrapper around the
    underlying C library that provides a simplified interface to the
    underlying library.

    The object behaves as an iterator, once initialised, iterating over it
    will yield successive oscillator samples.
    """

    def __init__(self, lib, osc):
        """
        Initialise an instance.  Don't call directly, use LibSSTVEnc.init_osc
        instead!
        """
        self._lib = lib
        self._osc = osc

    @property
    def amplitude(self):
        return self._osc.amplitude

    @amplitude.setter
    def amplitude(self, amplitude):
        self._osc.amplitude = float(amplitude)

    @property
    def offset(self):
        return self._osc.offset

    @offset.setter
    def offset(self, offset):
        self._osc.offset = float(offset)

    @property
    def output(self):
        return self._osc.output

    @property
    def frequency(self):
        return self._lib.sstvenc_osc_get_frequency(ctypes.byref(self._osc))

    @frequency.setter
    def frequency(self, frequency):
        self._lib.sstvenc_osc_set_frequency(
            ctypes.byref(self._osc), float(frequency)
        )

    def compute(self):
        self._lib.sstvenc_osc_compute(ctypes.byref(self._osc))

    def __iter__(self):
        return self

    def __next__(self):
        self.compute()
        return self.output


class LibSSTVEncPulseShaperPhase(enum.IntEnum):
    INIT = 0
    RISE = 1
    HOLD = 2
    FALL = 3
    DONE = 4


class _LibSSTVEncPulseShape(ctypes.Structure):
    _fields_ = [
        ("amplitude", ctypes.c_double),
        ("output", ctypes.c_double),
        ("sample_rate", ctypes.c_uint32),
        ("sample_idx", ctypes.c_uint32),
        ("hold_sz", ctypes.c_uint32),
        ("rise_sz", ctypes.c_uint16),
        ("fall_sz", ctypes.c_uint16),
        ("phase", ctypes.c_uint8),
    ]


class LibSSTVEncPulseShape(object):
    def __init__(self, lib, ps):
        self._lib = lib
        self._ps = ps

    @property
    def amplitude(self):
        return self._ps.amplitude

    @amplitude.setter
    def amplitude(self, amplitude):
        self._ps.amplitude = float(amplitude)

    @property
    def output(self):
        return self._ps.output

    @property
    def sample_idx(self):
        return self._ps.sample_idx

    @property
    def phase(self):
        return LibSSTVEncPulseShaperPhase(self._ps.phase)

    def compute(self):
        self._lib.sstvenc_ps_compute(ctypes.byref(self._ps))

    def __iter__(self):
        return self

    def __next__(self):
        self.compute()
        if self.state is not LibSSTVEncPulseShaperPhase.DONE:
            return self.output
        else:
            raise StopIteration


class LibSSTVEncCWModState(enum.IntEnum):
    INIT = 0
    NEXT_SYM = 1
    MARK = 2
    DITSPACE = 3
    DAHSPACE = 4
    DONE = 5


class _LibSSTVEncCWMod(ctypes.Structure):
    _fields_ = [
        ("output", ctypes.c_double),
        ("text_string", ctypes.c_char_p),
        ("symbol", ctypes.c_void_p),
        ("osc", _LibSSTVEncOscillator),
        ("ps", _LibSSTVEncPulseShape),
        ("dit_period", ctypes.c_uint16),
        ("state", ctypes.c_uint8),
        ("pos", ctypes.c_uint8),
    ]


class LibSSTVEncCWMod(object):
    def __init__(self, lib, mod):
        self._lib = lib
        self._mod = mod

    @property
    def amplitude(self):
        return self._mod.ps.amplitude

    @amplitude.setter
    def amplitude(self, amplitude):
        self._mod.ps.amplitude = float(amplitude)

    @property
    def output(self):
        return self._mod.output

    @property
    def state(self):
        return LibSSTVEncCWModState(self._mod.state)

    def compute(self):
        self._lib.sstvenc_cw_compute(ctypes.byref(self._mod))

    def __iter__(self):
        return self

    def __next__(self):
        self.compute()
        if self.state is not LibSSTVEncCWModState.DONE:
            return self.output
        else:
            raise StopIteration


class LibSSTVEncPulse(ctypes.Structure):
    _fields_ = [
        ("frequency", ctypes.c_uint32),
        ("duration_ns", ctypes.c_uint32),
    ]


class _LibSSTVEncMode(ctypes.Structure):
    _fields_ = [
        ("description", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
        ("initseq", ctypes.POINTER(LibSSTVEncPulse)),
        ("frontporch", ctypes.POINTER(LibSSTVEncPulse)),
        ("gap01", ctypes.POINTER(LibSSTVEncPulse)),
        ("gap12", ctypes.POINTER(LibSSTVEncPulse)),
        ("gap23", ctypes.POINTER(LibSSTVEncPulse)),
        ("backporch", ctypes.POINTER(LibSSTVEncPulse)),
        ("finalseq", ctypes.POINTER(LibSSTVEncPulse)),
        ("scanline_period_ns", ctypes.c_uint32 * 4),
        ("width", ctypes.c_uint16),
        ("height", ctypes.c_uint16),
        ("colour_space_order", ctypes.c_uint16),
        ("vis_code", ctypes.c_uint8),
    ]


class LibSSTVEncColourSpace(enum.IntEnum):
    MONO = 0
    RGB = 1
    YUV = 2
    YUV2 = 3


class LibSSTVEncChannel(enum.IntEnum):
    NONE = 0
    Y = 1
    U = 2
    V = 3
    R = 4
    G = 5
    B = 6
    Y2 = 7


class LibSSTVEncColourSpaceOrder(object):
    CSO_BIT_MODE = 12
    MAX_CHANNELS = 4

    @classmethod
    def decode(cls, value):
        cs = value >> CSO_BIT_MODE
        sources = [(value >> (ch * 3)) & 7 for ch in range(cls.MAX_CHANNELS)]
        return cls(cs, *sources)

    def __init__(self, cs, *sources):
        self._cs = LibSSTVEncColourSpace(cs)
        self._ch = [
            LibSSTVEncChannel(ch) for ch in sources[: self.MAX_CHANNELS]
        ]
        while len(self._ch) < self.MAX_CHANNELS:
            self._ch.append(LibSSTVEncChannel.NONE)

    @property
    def cs(self):
        return self._cs

    @cs.setter
    def cs(self, cs):
        self._cs = LibSSTVEncColourSpace(cs)

    def __len__(self):
        try:
            return self._ch.index(LibSSTVEncChannel.NONE)
        except ValueError:
            return self.MAX_CHANNELS

    def __getitem__(self, ch):
        return self._ch[ch]

    def __setitem__(self, ch, src):
        self._ch[ch] = LibSSTVEncChannel(src)

    def __iter__(self):
        for ch in self._ch:
            if ch is not LibSSTVEncChannel.NONE:
                yield ch

    def __int__(self):
        value = self.cs.value << self.CSO_BIT_MODE

        for num, ch in enumerate(self):
            value |= ch.value << (num * 3)

        return value


class LibSSTVPulseSequence(object):
    def __init__(self, seq):
        self._seq = seq
        if seq:
            # Length not known yet
            self._length = None
        else:
            # NULL sequence: length is zero
            self._length = 0

    def __len__(self):
        if self._length is None:
            self._length = len(list(self))
        return self._length

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(idx)

        return self._seq[idx].frequency, self._seq[idx].duration_ns

    def __iter__(self):
        idx = 0

        if self._seq:
            while self[idx].duration_ns > 0:
                yield self._seq[idx]
                idx += 1

        # End of the sequence, if we didn't know the length, we do now!
        if self._length is None:
            self._length = idx


class LibSSTVEncMode(object):
    def __init__(self, lib, mode):
        self._lib = lib
        self._mode = mode

    @property
    def description(self):
        return self._mode.content.description.decode("UTF-8")

    @property
    def name(self):
        return self._mode.content.name.decode("US-ASCII")

    @property
    def initseq(self):
        return LibSSTVPulseSequence(self._mode.content.initseq)

    @property
    def frontporch(self):
        return LibSSTVPulseSequence(self._mode.content.frontporch)

    @property
    def gap01(self):
        return LibSSTVPulseSequence(self._mode.content.gap01)

    @property
    def gap12(self):
        return LibSSTVPulseSequence(self._mode.content.gap12)

    @property
    def gap23(self):
        return LibSSTVPulseSequence(self._mode.content.gap23)

    @property
    def backporch(self):
        return LibSSTVPulseSequence(self._mode.content.backporch)

    @property
    def finalseq(self):
        return LibSSTVPulseSequence(self._mode.content.finalseq)

    @property
    def scanline_period_ns(self):
        return tuple(self._mode.content.scanline_period_ns)

    @property
    def width(self):
        return self._mode.content.width

    @property
    def height(self):
        return self._mode.content.height

    @property
    def colour_space_order(self):
        return LibSSTVEncColourSpaceOrder.decode(
            self._mode.content.colour_space_order
        )

    @property
    def fb_sz(self):
        return self._lib.sstvenc_mode_get_fb_sz(self._mode)

    def get_txtime(self, fsk_id=None):
        return self._lib.sstvenc_mode_get_txtime(
            self._mode, fsk_id.encode("US-ASCII") if fsk_id else None
        )

    def get_pixel_posn(self, x, y):
        return self._lib.sstvenc_get_pixel_posn(self._mode, x, y)

    def mkbuffer(self):
        return ctypes.c_uint8 * self.fb_sz()


class LibSSTVEncPhase(enum.IntEnum):
    INIT = 0
    VIS = 1
    INITSEQ = 2
    SCAN = 3
    FINALSEQ = 4
    FSK = 5
    DONE = 6


class _LibSSTVEncEncoderVisVars(ctypes.Structure):
    _fields_ = [
        ("bit", ctypes.c_uint8),
    ]


class _LibSSTVEncEncoderScanVars(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_uint16),
        ("y", ctypes.c_uint16),
        ("segment", ctypes.c_uint8),
    ]


class _LibSSTVEncEncoderFSKVars(ctypes.Structure):
    _fields_ = [
        ("segment", ctypes.c_uint8),
        ("seg_sz", ctypes.c_uint8),
        ("byte", ctypes.c_uint8),
        ("bv", ctypes.c_uint8),
        ("bit", ctypes.c_uint8),
    ]


class _LibSSTVEncEncoderVars(ctypes.Union):
    _fields_ = [
        ("vis", _LibSSTVEncEncoderVisVars),
        ("scan", _LibSSTVEncEncoderScanVars),
        ("fsk", _LibSSTVEncEncoderFSKVars),
    ]


class _LibSSTVEncEncoder(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.POINTER(_LibSSTVEncMode)),
        ("fsk_id", ctypes.c_char_p),
        ("framebuffer", ctypes.POINTER(ctypes.c_uint8)),
        ("seq", ctypes.c_void_p),
        ("seq_done_cb", ctypes.c_void_p),
        ("pulse", LibSSTVEncPulse),
        ("vars", _LibSSTVEncEncoderVars),
        ("phase", ctypes.c_uint8),
    ]


class LibSSTVEncEncoder(object):
    def __init__(self, lib, enc):
        self._lib = lib
        self._enc = enc

    @property
    def mode(self):
        return LibSSTVEncMode(self._lib, self._enc.mode)

    @property
    def fsk_id(self):
        return self._enc.mode.decode("UTF-8")

    @property
    def framebuffer(self):
        return self._enc.framebuffer

    @property
    def phase(self):
        return LibSSTVEncPhase(self._enc.phase)

    def __iter__(self):
        while self.phase is not LibSSTVEncPhase.DONE:
            pulseptr = self._lib.sstvenc_encoder_next_pulse(
                ctypes.byref(self._enc)
            )
            yield pulseptr.content.frequency, pulseptr.content.duration_ns


class LibSSTVEncSunAUFormat(enum.IntEnum):
    S8 = 0x02
    S16 = 0x03
    S32 = 0x05
    F32 = 0x06
    F64 = 0x07


class _LibSSTVEncSunAUEnc(ctypes.Structure):
    _fields_ = [
        ("fh", ctypes.c_void_p),
        ("written_sz", ctypes.c_uint32),
        ("sample_rate", ctypes.c_uint32),
        ("state", ctypes.c_uint16),
        ("encoding", ctypes.c_uint8),
        ("channels", ctypes.c_uint8),
    ]


class LibSSTVEncSunAuEnc(object):
    def __init__(self, lib, enc):
        self._lib = lib
        self._enc = enc

    def __del__(self):
        self.close()

    @property
    def written_sz(self):
        return self._enc.written_sz

    @property
    def sample_rate(self):
        return self._enc.sample_rate

    @property
    def encoding(self):
        return LibSSTVEncSunAUFormat(self._enc.encoding)

    @property
    def channels(self):
        return self._enc.channels

    def write(self, *samples):
        samples_ptr = ctypes.c_double * len(samples)
        for idx, sample in enumerate(samples):
            samples_ptr[idx] = sample

        _errno_to_oserror(
            self._lib.sstvenc_sunau_enc_write(
                ctypes.byref(self._enc), len(samples), samples_ptr
            )
        )

    def close(self):
        if self._enc is not None:
            _errno_to_oserror(
                self._lib.sstvenc_sunau_enc_close(ctypes.byref(self._enc))
            )
            self._enc = None


class LibSSTVEncTimescaleUnit(enum.IntEnum):
    SECONDS = 0
    MILLISECONDS = 1
    MICROSECONDS = 2
    NANOSECONDS = 3

    @property
    def unit_scale(self):
        return LibSSTVEnc.get_instance()._lib.sstvenc_ts_unit_scale(
            self.value
        )

    def unit_to_samples(self, time, sample_rate):
        return LibSSTVEnc.get_instance()._lib.sstvenc_ts_unit_to_samples(
            float(time), int(sample_rate), self.value
        )

    def samples_to_unit(self, samples, sample_rate):
        return LibSSTVEnc.get_instance()._lib.sstvenc_ts_samples_to_unit(
            int(samples), int(sample_rate), self.value
        )


class LibSSTVEnc(object):
    _INSTANCE = None

    @classmethod
    def get_instance(cls):
        if cls._INSTANCE is None:
            cls._INSTANCE = cls()
        return cls._INSTANCE

    def __init__(self):
        """
        Initialise an instance of the SSTV encoder library.  Don't call
        directly, call the ``get_instance()`` class method instead.
        """
        assert self._INSTANCE is None, "Already loaded"
        self._lib = ctypes.CDLL(
            name="libsstvenc.so.0",
            mode=0,
            handle=None,
            use_errno=True,
            use_last_error=False,
            winmode=None,
        )

        # Initialise API structure -- we'll do them all here since that makes
        # life easier utilising any of these functions later on.

        # cw.h
        self._lib.sstvenc_cw_init.restype = None
        self._lib.sstvenc_cw_init.argtypes = (
            ctypes.POINTER(_LibSSTVEncCWMod),  # cw
            ctypes.c_char_p,  # text
            ctypes.c_double,  # amplitude
            ctypes.c_double,  # frequency
            ctypes.c_double,  # dit_period
            ctypes.c_double,  # slope_period
            ctypes.c_uint32,  # sample_rate
            ctypes.c_uint8,  # time_unit
        )

        self._lib.sstvenc_cw_compute.restype = None
        self._lib.sstvenc_cw_compute.argtypes = (
            ctypes.POINTER(_LibSSTVEncCWMod),
        )

        # oscillator.h
        self._lib.sstvenc_osc_get_frequency.restype = ctypes.c_double
        self._lib.sstvenc_osc_get_frequency.argtypes = (
            ctypes.POINTER(_LibSSTVEncOscillator),
        )

        self._lib.sstvenc_osc_set_frequency.restype = None
        self._lib.sstvenc_osc_set_frequency.argtypes = (
            ctypes.POINTER(_LibSSTVEncOscillator),
            ctypes.c_double,
        )

        self._lib.sstvenc_osc_init.restype = None
        self._lib.sstvenc_osc_init.argtypes = (
            ctypes.POINTER(_LibSSTVEncOscillator),  # osc
            ctypes.c_double,  # amplitude
            ctypes.c_double,  # frequency
            ctypes.c_double,  # offset
            ctypes.c_uint32,  # sample_rate
        )

        self._lib.sstvenc_osc_compute.restype = None
        self._lib.sstvenc_osc_compute.argtypes = (
            ctypes.POINTER(_LibSSTVEncOscillator),
        )

        # pulseshape.h
        self._lib.sstvenc_ps_reset_samples.restype = None
        self._lib.sstvenc_ps_reset_samples.argtypes = (
            ctypes.POINTER(_LibSSTVEncPulseShape),
            ctypes.c_uint32,
        )

        self._lib.sstvenc_ps_reset.restype = None
        self._lib.sstvenc_ps_reset.argtypes = (
            ctypes.POINTER(_LibSSTVEncPulseShape),  # ps
            ctypes.c_double,  # hold_time
            ctypes.c_uint8,  # time_unit
        )

        self._lib.sstvenc_ps_init.restype = None
        self._lib.sstvenc_ps_init.argtypes = (
            ctypes.POINTER(_LibSSTVEncPulseShape),  # ps
            ctypes.c_double,  # amplitude
            ctypes.c_double,  # rise_time
            ctypes.c_double,  # hold_time
            ctypes.c_double,  # fall_time
            ctypes.c_uint32,  # sample_rate
            ctypes.c_uint8,  # time_unit
        )

        self._lib.sstvenc_ps_advance.restype = None
        self._lib.sstvenc_ps_advance.argtypes = (
            ctypes.POINTER(_LibSSTVEncPulseShape),
        )

        self._lib.sstvenc_ps_compute.restype = None
        self._lib.sstvenc_ps_compute.argtypes = (
            ctypes.POINTER(_LibSSTVEncPulseShape),
        )

        # sstv.h
        self._lib.sstvenc_encoder_init.restype = None
        self._lib.sstvenc_encoder_init.argtypes = (
            ctypes.POINTER(_LibSSTVEncEncoder),
            ctypes.POINTER(_LibSSTVEncMode),
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint8),
        )

        self._lib.sstvenc_encoder_next_pulse.restype = ctypes.POINTER(
            LibSSTVEncPulse
        )
        self._lib.sstvenc_encoder_next_pulse.argtypes = (
            ctypes.POINTER(_LibSSTVEncEncoder),
        )

        # sstvfreq.h
        self._lib.sstvenc_level_freq.restype = ctypes.c_int16
        self._lib.sstvenc_level_freq.argtypes = (ctypes.c_uint8,)

        # sstvmode.h
        self._lib.sstvenc_get_mode_by_idx.restype = ctypes.POINTER(
            _LibSSTVEncMode
        )
        self._lib.sstvenc_get_mode_by_idx.argtypes = (ctypes.c_uint8,)

        self._lib.sstvenc_get_mode_by_name.restype = ctypes.POINTER(
            _LibSSTVEncMode
        )
        self._lib.sstvenc_get_mode_by_name.argtypes = (ctypes.c_char_p,)

        self._lib.sstvenc_pulseseq_get_txtime.restype = ctypes.c_uint64
        self._lib.sstvenc_pulseseq_get_txtime.argtypes = (
            ctypes.POINTER(LibSSTVEncPulse),
        )

        self._lib.sstvenc_mode_get_txtime.restype = ctypes.c_uint64
        self._lib.sstvenc_mode_get_txtime.argtypes = (
            ctypes.POINTER(_LibSSTVEncMode),
            ctypes.c_char_p,
        )

        self._lib.sstvenc_mode_get_fb_sz.restype = ctypes.c_size_t
        self._lib.sstvenc_mode_get_fb_sz.argtypes = (
            ctypes.POINTER(_LibSSTVEncMode),
        )

        self._lib.sstvenc_get_pixel_posn.restype = ctypes.c_uint32
        self._lib.sstvenc_get_pixel_posn.argtypes = (
            ctypes.POINTER(_LibSSTVEncMode),
            ctypes.c_uint16,
            ctypes.c_uint16,
        )

        # sunau.h
        self._lib.sstvenc_sunau_enc_check.restype = ctypes.c_int
        self._lib.sstvenc_sunau_enc_check.argtypes = (
            ctypes.c_uint32,  # sample_rate
            ctypes.c_uint8,  # encoding
            ctypes.c_uint8,  # channels
        )

        # NB: we ignore sstvenc_sunau_enc_init_fh because ctypes does not
        # define a FILE* pointer type, and doing so would be awkward in
        # Python anyway.

        self._lib.sstvenc_sunau_enc_init.restype = ctypes.c_int
        self._lib.sstvenc_sunau_enc_init.argtypes = (
            ctypes.POINTER(_LibSSTVEncSunAUEnc),  # enc
            ctypes.c_char_p,  # path
            ctypes.c_uint32,  # sample_rate
            ctypes.c_uint8,  # encoding
            ctypes.c_uint8,  # channels
        )

        self._lib.sstvenc_sunau_enc_write.restype = ctypes.c_int
        self._lib.sstvenc_sunau_enc_write.argtypes = (
            ctypes.POINTER(_LibSSTVEncSunAUEnc),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
        )

        self._lib.sstvenc_sunau_enc_close.restype = ctypes.c_int
        self._lib.sstvenc_sunau_enc_write.argtypes = (
            ctypes.POINTER(_LibSSTVEncSunAUEnc),
        )

        # timescale.h
        self._lib.sstvenc_ts_unit_scale.restype = ctypes.c_uint64
        self._lib.sstvenc_ts_unit_scale.argtypes = (ctypes.c_uint8,)

        self._lib.sstvenc_ts_clamp_samples.restype = ctypes.c_uint32
        self._lib.sstvenc_ts_clamp_samples.argtypes = (ctypes.c_uint64,)

        self._lib.sstvenc_ts_unit_to_samples.restype = ctypes.c_uint32
        self._lib.sstvenc_ts_unit_to_samples.argtypes = (
            ctypes.c_double,  # time
            ctypes.c_uint32,  # sample_rate
            ctypes.c_uint8,  # unit
        )

        self._lib.sstvenc_ts_samples_to_unit.restype = ctypes.c_double
        self._lib.sstvenc_ts_samples_to_unit.argtypes = (
            ctypes.c_uint32,  # samples
            ctypes.c_uint32,  # sample_rate
            ctypes.c_uint8,  # unit
        )

        # yuv.h
        self._lib.sstvenc_yuv_calc_y.restype = ctypes.c_uint8
        self._lib.sstvenc_yuv_calc_y.argtypes = (
            ctypes.c_uint8,  # red
            ctypes.c_uint8,  # green
            ctypes.c_uint8,  # blue
        )

        self._lib.sstvenc_yuv_calc_u.restype = ctypes.c_uint8
        self._lib.sstvenc_yuv_calc_u.argtypes = (
            ctypes.c_uint8,  # red
            ctypes.c_uint8,  # green
            ctypes.c_uint8,  # blue
        )

        self._lib.sstvenc_yuv_calc_v.restype = ctypes.c_uint8
        self._lib.sstvenc_yuv_calc_v.argtypes = (
            ctypes.c_uint8,  # red
            ctypes.c_uint8,  # green
            ctypes.c_uint8,  # blue
        )

        self._lib.sstvenc_rgb_calc_r.restype = ctypes.c_uint8
        self._lib.sstvenc_rgb_calc_r.argtypes = (
            ctypes.c_uint8,  # Y
            ctypes.c_uint8,  # U
            ctypes.c_uint8,  # V
        )

        self._lib.sstvenc_rgb_calc_g.restype = ctypes.c_uint8
        self._lib.sstvenc_rgb_calc_g.argtypes = (
            ctypes.c_uint8,  # Y
            ctypes.c_uint8,  # U
            ctypes.c_uint8,  # V
        )

        self._lib.sstvenc_rgb_calc_b.restype = ctypes.c_uint8
        self._lib.sstvenc_rgb_calc_b.argtypes = (
            ctypes.c_uint8,  # Y
            ctypes.c_uint8,  # U
            ctypes.c_uint8,  # V
        )

        self._lib.sstvenc_rgb_to_yuv.restype = None
        self._lib.sstvenc_rgb_to_yuv.argtypes = (
            ctypes.POINTER(ctypes.c_uint8),  # dest
            ctypes.POINTER(ctypes.c_uint8),  # src
            ctypes.c_uint16,  # width
            ctypes.c_uint16,  # height
        )

        self._lib.sstvenc_rgb_to_mono.restype = None
        self._lib.sstvenc_rgb_to_mono.argtypes = (
            ctypes.POINTER(ctypes.c_uint8),  # dest
            ctypes.POINTER(ctypes.c_uint8),  # src
            ctypes.c_uint16,  # width
            ctypes.c_uint16,  # height
        )

        self._lib.sstvenc_yuv_to_rgb.restype = None
        self._lib.sstvenc_yuv_to_rgb.argtypes = (
            ctypes.POINTER(ctypes.c_uint8),  # dest
            ctypes.POINTER(ctypes.c_uint8),  # src
            ctypes.c_uint16,  # width
            ctypes.c_uint16,  # height
        )

        self._lib.sstvenc_yuv_to_mono.restype = None
        self._lib.sstvenc_yuv_to_mono.argtypes = (
            ctypes.POINTER(ctypes.c_uint8),  # dest
            ctypes.POINTER(ctypes.c_uint8),  # src
            ctypes.c_uint16,  # width
            ctypes.c_uint16,  # height
        )

        self._lib.sstvenc_mono_to_rgb.restype = None
        self._lib.sstvenc_mono_to_rgb.argtypes = (
            ctypes.POINTER(ctypes.c_uint8),  # dest
            ctypes.POINTER(ctypes.c_uint8),  # src
            ctypes.c_uint16,  # width
            ctypes.c_uint16,  # height
        )

        self._lib.sstvenc_mono_to_yuv.restype = None
        self._lib.sstvenc_mono_to_yuv.argtypes = (
            ctypes.POINTER(ctypes.c_uint8),  # dest
            ctypes.POINTER(ctypes.c_uint8),  # src
            ctypes.c_uint16,  # width
            ctypes.c_uint16,  # height
        )

        self._INSTANCE = self

    # cw.h

    def init_cw_mod(
        self,
        text,
        amplitude=1.0,
        frequency=800.0,
        dit_period=60.0,
        slope_period=None,
        sample_rate=48000,
        time_unit=LibSSTVEncTimescaleUnit.MILLISECONDS,
        mod=None,
    ):
        text = str(text).encode("UTF-8")
        amplitude = float(amplitude)
        frequency = float(frequency)
        dit_period = float(dit_period)

        if not slope_period:
            slope_period = dit_period / 5.0

        slope_period = float(slope_period)
        sample_rate = int(sample_rate)
        time_unit = LibSSTVEncTimescaleUnit(time_unit)

        if mod is None:
            mod = _LibSSTVEncCWMod()

        self._lib.sstvenc_cw_init(
            ctypes.byref(mod),
            text,
            amplitude,
            frequency,
            dit_period,
            slope_period,
            sample_rate,
            time_unit,
        )
        return LibSSTVEncCWMod(self._lib, mod)

    # oscillator.h

    def init_osc(
        self,
        amplitude=1.0,
        frequency=800.0,
        offset=0.0,
        sample_rate=48000,
        osc=None,
    ):
        amplitude = float(amplitude)
        frequency = float(frequency)
        offset = float(offset)
        sample_rate = int(sample_rate)

        if osc is None:
            osc = _LibSSTVEncOscillator()

        self._lib.sstvenc_osc_init(
            ctypes.byref(osc), amplitude, frequency, offset, sample_rate
        )
        return LibSSTVEncOscillator(self._lib, osc)

    # pulseshape.h

    def init_ps(
        self,
        rise_time,
        hold_time=None,
        fall_time=None,
        amplitude=1.0,
        sample_rate=48000,
        time_unit=LibSSTVEncTimescaleUnit.MILLISECONDS,
        ps=None,
    ):
        amplitude = float(amplitude)
        time_unit = LibSSTVEncTimescaleUnit(time_unit)
        rise_time = float(rise_time)

        if hold_time is None:
            hold_time = math.inf
        else:
            hold_time = float(hold_time)

        if fall_time is None:
            fall_time = rise_time
        else:
            fall_time = float(fall_time)

        sample_rate = int(sample_rate)

        if ps is None:
            ps = _LibSSTVEncPulseShape()

        self._lib.sstvenc_ps_init(
            ctypes.byref(ps),
            amplitude,
            rise_time,
            hold_time,
            fall_time,
            sample_rate,
            time_unit,
        )
        return LibSSTVEncPulseShape(self._lib, ps)

    # sstv.h

    def init_enc(self, mode, fsk_id=None, enc=None):
        framebuffer = mode.mkbuffer()

        if enc is None:
            enc = _LibSSTVEncEncoder()

        if fsk_id:
            fsk_id = str(fsk_id).encode("US-ASCII")
        else:
            fsk_id = None

        self._lib.sstvenc_encoder_init(
            ctypes.byref(enc), mode._mode, fsk_id, framebuffer
        )
        return LibSSTVEncEncoder(self._lib, enc)

    # sstvmode.h

    def get_sstv_mode_count(self):
        return self._lib.get_sstv_mode_count()

    def _return_sstv_mode(self, mode):
        if mode:
            return LibSSTVEncMode(self._lib, mode)
        else:
            return None

    def get_sstv_mode_by_idx(self, idx):
        return self._return_sstv_mode(self._lib.sstvenc_get_mode_by_idx(idx))

    def get_sstv_mode_by_name(self, name):
        return self._return_sstv_mode(
            self._lib.sstvenc_get_mode_by_name(name.encode("US-ASCII"))
        )

    # sunau.h

    def init_sunau_enc(
        self,
        path,
        sample_rate=48000,
        encoding=LibSSTVEncSunAUFormat.S16,
        channels=1,
        enc=None,
    ):

        sample_rate = int(sample_rate)
        encoding = LibSSTVEncSunAUFormat(encoding)
        channels = int(channels)

        if enc is None:
            enc = _LibSSTVEncSunAUEnc()

        _errno_to_oserror(
            self._lib.sstvenc_sunau_enc_init(
                ctypes.byref(enc),
                path.encode("UTF-8"),
                sample_rate,
                encoding.value,
                channels,
            )
        )

        return LibSSTVEncSunAuEnc(self._lib, enc)

    # yuv.h

    def convert_rgb_to_yuv(self, src, width, height, dest=None):
        if dest is None:
            dest = src

        self._lib.sstvenc_rgb_to_yuv(dest, src, width, height)

    def convert_rgb_to_mono(self, src, width, height, dest=None):
        if dest is None:
            dest = src

        self._lib.sstvenc_rgb_to_mono(dest, src, width, height)
