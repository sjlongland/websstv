#!/usr/bin/env python3

"""
Python wrapper around libsstvenc.
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

import array
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


class _Oscillator(ctypes.Structure):
    _fields_ = [
        ("amplitude", ctypes.c_double),
        ("offset", ctypes.c_double),
        ("output", ctypes.c_double),
        ("sample_rate", ctypes.c_uint32),
        ("phase", ctypes.c_uint32),
        ("phase_inc", ctypes.c_uint32),
    ]


class Oscillator(object):
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


class PulseShaperPhase(enum.IntEnum):
    INIT = 0
    RISE = 1
    HOLD = 2
    FALL = 3
    DONE = 4


class _PulseShape(ctypes.Structure):
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


class PulseShape(object):
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
        return PulseShaperPhase(self._ps.phase)

    def compute(self):
        self._lib.sstvenc_ps_compute(ctypes.byref(self._ps))

    def __iter__(self):
        return self

    def __next__(self):
        self.compute()
        if self.state is not PulseShaperPhase.DONE:
            return self.output
        else:
            raise StopIteration


class CWModState(enum.IntEnum):
    INIT = 0
    NEXT_SYM = 1
    MARK = 2
    DITSPACE = 3
    DAHSPACE = 4
    DONE = 5


class _CWMod(ctypes.Structure):
    _fields_ = [
        ("output", ctypes.c_double),
        ("text_string", ctypes.c_char_p),
        ("symbol", ctypes.c_void_p),
        ("osc", _Oscillator),
        ("ps", _PulseShape),
        ("dit_period", ctypes.c_uint16),
        ("state", ctypes.c_uint8),
        ("pos", ctypes.c_uint8),
    ]


class CWMod(object):
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
        return CWModState(self._mod.state)

    def compute(self):
        self._lib.sstvenc_cw_compute(ctypes.byref(self._mod))

    def __iter__(self):
        return self

    def __next__(self):
        self.compute()
        if self.state is not CWModState.DONE:
            return self.output
        else:
            raise StopIteration

    def read(self, n_samples):
        buffer = (ctypes.c_double * n_samples)()
        out_samples = self._lib.sstvenc_cw_fill_buffer(
            ctypes.byref(self._mod), buffer, n_samples
        )
        return array.array("d", iter(buffer[:out_samples]))


class Pulse(ctypes.Structure):
    _fields_ = [
        ("frequency", ctypes.c_uint32),
        ("duration_ns", ctypes.c_uint32),
    ]


class _Mode(ctypes.Structure):
    _fields_ = [
        ("description", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
        ("initseq", ctypes.POINTER(Pulse)),
        ("frontporch", ctypes.POINTER(Pulse)),
        ("gap01", ctypes.POINTER(Pulse)),
        ("gap12", ctypes.POINTER(Pulse)),
        ("gap23", ctypes.POINTER(Pulse)),
        ("backporch", ctypes.POINTER(Pulse)),
        ("finalseq", ctypes.POINTER(Pulse)),
        ("scanline_period_ns", ctypes.c_uint32 * 4),
        ("width", ctypes.c_uint16),
        ("height", ctypes.c_uint16),
        ("colour_space_order", ctypes.c_uint16),
        ("vis_code", ctypes.c_uint8),
    ]


class ColourSpace(enum.IntEnum):
    MONO = 0
    RGB = 1
    YUV = 2
    YUV2 = 3


class Channel(enum.IntEnum):
    NONE = 0
    Y = 1
    U = 2
    V = 3
    R = 4
    G = 5
    B = 6
    Y2 = 7


class ColourSpaceOrder(object):
    CSO_BIT_MODE = 12
    MAX_CHANNELS = 4

    @classmethod
    def decode(cls, value):
        cs = value >> CSO_BIT_MODE
        sources = [(value >> (ch * 3)) & 7 for ch in range(cls.MAX_CHANNELS)]
        return cls(cs, *sources)

    def __init__(self, cs, *sources):
        self._cs = ColourSpace(cs)
        self._ch = [Channel(ch) for ch in sources[: self.MAX_CHANNELS]]
        while len(self._ch) < self.MAX_CHANNELS:
            self._ch.append(Channel.NONE)

    @property
    def cs(self):
        return self._cs

    @cs.setter
    def cs(self, cs):
        self._cs = ColourSpace(cs)

    def __len__(self):
        try:
            return self._ch.index(Channel.NONE)
        except ValueError:
            return self.MAX_CHANNELS

    def __getitem__(self, ch):
        return self._ch[ch]

    def __setitem__(self, ch, src):
        self._ch[ch] = Channel(src)

    def __iter__(self):
        for ch in self._ch:
            if ch is not Channel.NONE:
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


class Mode(object):
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
        return ColourSpaceOrder.decode(self._mode.content.colour_space_order)

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


class EncoderPhase(enum.IntEnum):
    INIT = 0
    VIS = 1
    INITSEQ = 2
    SCAN = 3
    FINALSEQ = 4
    FSK = 5
    DONE = 6


class _EncoderVisVars(ctypes.Structure):
    _fields_ = [
        ("bit", ctypes.c_uint8),
    ]


class _EncoderScanVars(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_uint16),
        ("y", ctypes.c_uint16),
        ("segment", ctypes.c_uint8),
    ]


class _EncoderFSKVars(ctypes.Structure):
    _fields_ = [
        ("segment", ctypes.c_uint8),
        ("seg_sz", ctypes.c_uint8),
        ("byte", ctypes.c_uint8),
        ("bv", ctypes.c_uint8),
        ("bit", ctypes.c_uint8),
    ]


class _EncoderVars(ctypes.Union):
    _fields_ = [
        ("vis", _EncoderVisVars),
        ("scan", _EncoderScanVars),
        ("fsk", _EncoderFSKVars),
    ]


class _Encoder(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.POINTER(_Mode)),
        ("fsk_id", ctypes.c_char_p),
        ("framebuffer", ctypes.POINTER(ctypes.c_uint8)),
        ("seq", ctypes.c_void_p),
        ("seq_done_cb", ctypes.c_void_p),
        ("pulse", Pulse),
        ("vars", _EncoderVars),
        ("phase", ctypes.c_uint8),
    ]


class Encoder(object):
    def __init__(self, lib, enc):
        self._lib = lib
        self._enc = enc

    @property
    def mode(self):
        return Mode(self._lib, self._enc.mode)

    @property
    def fsk_id(self):
        return self._enc.mode.decode("UTF-8")

    @property
    def framebuffer(self):
        return self._enc.framebuffer

    @property
    def phase(self):
        return EncoderPhase(self._enc.phase)

    def __iter__(self):
        while self.phase is not EncoderPhase.DONE:
            pulseptr = self._lib.sstvenc_encoder_next_pulse(
                ctypes.byref(self._enc)
            )
            yield pulseptr.content.frequency, pulseptr.content.duration_ns


class _Modulator(ctypes.Structure):
    _fields_ = [
        ("enc", _Encoder),
        ("osc", _Oscillator),
        ("ps", _PulseShape),
        ("total_samples", ctypes.c_uint64),
        ("total_ns", ctypes.c_uint64),
        ("remaining", ctypes.c_uint32),
    ]


class Modulator(object):
    def __init__(self, lib, mod):
        self._lib = lib
        self._mod = mod
        self._enc = Encoder(lib, mod.enc)
        self._osc = Oscillator(lib, mod.osc)
        self._ps = PulseShape(lib, mod.ps)

    @property
    def encoder(self):
        return self._enc

    @property
    def oscillator(self):
        return self._osc

    @property
    def pulseshape(self):
        return self._ps

    def read(self, n_samples):
        buffer = (ctypes.c_double * n_samples)()
        out_samples = self._lib.sstvenc_modulator_fill_buffer(
            ctypes.byref(self._mod), buffer, n_samples
        )
        return array.array("d", iter(buffer[:out_samples]))


class SequencerStepType(enum.IntEnum):
    END = 0x00

    SET_TS_UNIT = 0x10

    SET_REGISTER = 0x20
    INC_REGISTER = 0x22
    DEC_REGISTER = 0x23
    MUL_REGISTER = 0x24
    DIV_REGISTER = 0x25
    IDEC_REGISTER = 0x2B
    IDIV_REGISTER = 0x2D

    EMIT_SILENCE = 0x30
    EMIT_TONE = 0x40
    EMIT_CW = 0x50
    EMIT_IMAGE = 0x60


class SequencerRegister(enum.IntEnum):
    AMPLITUDE = 0
    FREQUENCY = 1
    PHASE = 2
    PULSE_RISE = 3
    PULSE_FALL = 4
    DIT_PERIOD = 5


class SequencerStepToneSlopes(enum.IntEnum):
    NONE = 0
    RISING = 1
    FALLING = 2
    BOTH = 3


class _SequencerStepSetTSUnit(ctypes.Structure):
    _fields_ = [
        ("time_unit", ctypes.c_uint8),
        ("convert", ctypes.c_bool),
    ]


class _SequencerStepSetReg(ctypes.Structure):
    _fields_ = [
        ("value", ctypes.c_double),
        ("reg", ctypes.c_uint8),
    ]


class _SequencerStepDuration(ctypes.Structure):
    _fields_ = [
        ("duration", ctypes.c_double),
        ("slopes", ctypes.c_uint8),
    ]


class _SequencerStepCW(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
    ]


class _SequencerStepImage(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.POINTER(_Mode)),
        ("framebuffer", ctypes.POINTER(ctypes.c_uint8)),
        ("fsk_id", ctypes.c_char_p),
    ]


class _SequencerStepArgs(ctypes.Union):
    _fields_ = [
        ("ts", _SequencerStepSetTSUnit),
        ("reg", _SequencerStepSetReg),
        ("duration", _SequencerStepDuration),
        ("cw", _SequencerStepCW),
        ("image", _SequencerStepImage),
    ]


class _SequencerStep(ctypes.Structure):
    _fields_ = [
        ("args", _SequencerStepArgs),
        ("type", ctypes.c_uint8),
    ]


class _SequencerVarsSilence(ctypes.Structure):
    _fields_ = [
        ("remaining", ctypes.c_uint32),
    ]


class _SequencerVarsTone(ctypes.Structure):
    _fields_ = [
        ("osc", _Oscillator),
        ("ps", _PulseShape),
    ]


class _SequencerVars(ctypes.Union):
    _fields_ = [
        ("silence", _SequencerVarsSilence),
        ("tone", _SequencerVarsTone),
        ("cw", _CWMod),
        ("sstv", _Modulator),
    ]


class _Sequencer(ctypes.Structure):
    # Forward declaration for callback
    pass


SequencerEventCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(_Sequencer))

_Sequencer._fields_ = [
    ("steps", ctypes.POINTER(_SequencerStep)),
    ("event_cb", ctypes.POINTER(SequencerEventCallback)),
    ("event_cb_ctx", ctypes.c_void_p),
    ("output", ctypes.c_double),
    ("vars", _SequencerVars),
    ("regs", ctypes.c_double * 5),
    ("sample_rate", ctypes.c_uint32),
    ("step", ctypes.c_uint16),
    ("time_unit", ctypes.c_uint8),
    ("state", ctypes.c_uint8),
]


class SequencerState(enum.IntEnum):
    INIT = 0x00

    BEGIN_SILENCE = 0x10
    GEN_SILENCE = 0x17
    GEN_INF_SILENCE = 0x18
    END_SILENCE = 0x1F

    BEGIN_TONE = 0x20
    GEN_TONE = 0x27
    GEN_INF_TONE = 0x28
    END_TONE = 0x2F

    BEGIN_CW = 0x30
    GEN_CW = 0x37
    END_CW = 0x3F

    BEGIN_IMAGE = 0x40
    GEN_IMAGE = 0x47
    END_IMAGE = 0x4F

    DONE = 0xFF


class SunAUFormat(enum.IntEnum):
    S8 = 0x02
    S16 = 0x03
    S32 = 0x05
    F32 = 0x06
    F64 = 0x07


class _SunAUEnc(ctypes.Structure):
    _fields_ = [
        ("fh", ctypes.c_void_p),
        ("written_sz", ctypes.c_uint32),
        ("sample_rate", ctypes.c_uint32),
        ("state", ctypes.c_uint16),
        ("encoding", ctypes.c_uint8),
        ("channels", ctypes.c_uint8),
    ]


class SunAuEnc(object):
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
        return SunAUFormat(self._enc.encoding)

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


class TimescaleUnit(enum.IntEnum):
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


class SequencerStepBase(object):
    def __init__(self, lib):
        self._lib = lib
        self._step = None

    def _build(self, step):
        self._step = step


class SequencerStepSetTimescaleUnit(SequencerStepBase):
    def __init__(self, lib, time_unit, convert=False):
        super(SequencerStepSetTimescaleUnit, self).__init__(lib)
        self._unit = TimescaleUnit(time_unit)
        self._convert = bool(convert)

    @property
    def time_unit(self):
        return self._unit

    @property.setter
    def time_unit(self, time_unit):
        self._unit = TimescaleUnit(time_unit)
        if self._step:
            self._build(self._step)

    def _build(self, step):
        self._lib.sstvenc_sequencer_step_set_timescale(
            ctypes.byref(step), self._unit.value, self._convert
        )
        super(SequencerStepSetTimescaleUnit, self)._build(step)


class SequencerStepRegisterBase(SequencerStepBase):
    def __init__(self, lib, register, value):
        super(SequencerStepRegisterBase, self).__init__(lib)
        self._factory_fn = getattr(self._lib, self._FACTORY_FN)
        self._register = SequencerRegister(register)
        self._value = float(value)

    @property
    def register(self):
        return self._register

    @property.setter
    def register(self, register):
        self._register = TimescaleUnit(register)
        if self._step:
            self._build(self._step)

    @property
    def value(self):
        return self._value

    @property.setter
    def value(self, value):
        self._value = float(value)
        if self._step:
            self._build(self._step)

    def _build(self, step):
        self._factory_fn(
            ctypes.byref(step), self._register.value, self._value
        )
        super(SequencerStepRegisterBase, self)._build(step)


class SequencerStepSetRegister(SequencerStepRegisterBase):
    _FACTORY_FN = "sstvenc_sequencer_step_set_reg"


class SequencerStepIncRegister(SequencerStepRegisterBase):
    _FACTORY_FN = "sstvenc_sequencer_step_inc_reg"


class SequencerStepDecRegister(SequencerStepRegisterBase):
    _FACTORY_FN = "sstvenc_sequencer_step_dec_reg"


class SequencerStepMulRegister(SequencerStepRegisterBase):
    _FACTORY_FN = "sstvenc_sequencer_step_mul_reg"


class SequencerStepDivRegister(SequencerStepRegisterBase):
    _FACTORY_FN = "sstvenc_sequencer_step_div_reg"


class SequencerStepIDecRegister(SequencerStepRegisterBase):
    _FACTORY_FN = "sstvenc_sequencer_step_idec_reg"


class SequencerStepIDivRegister(SequencerStepRegisterBase):
    _FACTORY_FN = "sstvenc_sequencer_step_idiv_reg"


class SequencerStepDurationBase(SequencerStepBase):
    def __init__(self, lib, duration):
        super(SequencerStepDurationBase, self).__init__(lib)
        self._duration = float(duration)

    @property
    def duration(self):
        return self._duration

    @property.setter
    def duration(self, duration):
        self._duration = float(duration)
        if self._step:
            self._build(self._step)

    def _build(self, step):
        self._factory_fn(ctypes.byref(step), self._duration)
        super(SequencerStepDurationBase, self)._build(step)


class SequencerStepTone(SequencerStepDurationBase):
    def __init__(self, lib, duration, slopes=SequencerStepToneSlopes.BOTH):
        super(SequencerStepTone, self).__init__(lib, duration)
        self._slopes = SequencerStepToneSlopes(slopes)

    @property
    def slopes(self):
        return self._slopes

    @property.setter
    def slopes(self, duration):
        self._slopes = SequencerStepToneSlopes(slopes)
        if self._step:
            self._build(self._step)

    def _build(self, step):
        self._lib.sstvenc_sequencer_step_silence(
            ctypes.byref(step), self._duration, self._slopes.value
        )
        super(SequencerStepTone, self)._build(step)


class SequencerStepTone(SequencerStepDurationBase):
    def _build(self, step):
        self._lib.sstvenc_sequencer_step_tone(
            ctypes.byref(step), self._duration
        )
        super(SequencerStepTone, self)._build(step)


class SequencerStepCW(SequencerStepBase):
    def __init__(self, lib, text):
        super(SequencerStepCW, self).__init__(lib)
        self._text = str(text)

    @property
    def text(self):
        return self._text

    @property.setter
    def text(self, text):
        self._text = str(text)
        if self._step:
            self._build(self._step)

    def _build(self, step):
        self._lib.sstvenc_sequencer_step_cw(
            ctypes.byref(step), self._text.encode("UTF-8")
        )
        super(SequencerStepCW, self)._build(step)


class SequencerStepImage(SequencerStepBase):
    def __init__(self, lib, mode, framebuffer, fsk_id=None):
        super(SequencerStepImage, self).__init__(lib)
        self._mode = mode
        self._framebuffer = framebuffer
        if fsk_id:
            self._fsk_id = str(fsk_id)
        else:
            self._fsk_id = None

    @property
    def mode(self):
        return self._mode

    @property.setter
    def mode(self, mode):
        self._mode = mode
        if self._step:
            self._build(self._step)

    @property
    def framebuffer(self):
        return self._framebuffer

    @property.setter
    def framebuffer(self, framebuffer):
        self._framebuffer = framebuffer
        if self._step:
            self._build(self._step)

    @property
    def fsk_id(self):
        return self._fsk_id

    @property.setter
    def fsk_id(self, fsk_id):
        if fsk_id:
            self._fsk_id = fsk_id
        else:
            self._fsk_id = None

        if self._step:
            self._build(self._step)

    def _build(self, step):
        if self._fsk_id:
            fsk_id = self._fsk_id.encode("US-ASCII")
        else:
            fsk_id = None

        self._lib.sstvenc_sequencer_step_image(
            ctypes.byref(step), self._mode._mode, self._framebuffer, fsk_id
        )
        super(SequencerStepImage, self)._build(step)


class Sequencer(object):
    def __init__(self, lib, seq, steps):
        self._lib = lib
        self._seq = seq
        self._steps = steps

    @property
    def output(self):
        return self._seq.output

    @property
    def state(self):
        return SequencerState(self._seq.state)

    @property
    def sample_rate(self):
        return self._seq.sample_rate

    @property
    def time_unit(self):
        return TimescaleUnit(self._seq.time_unit)

    @property
    def event_cb(self):
        return self._event_cb

    @event_cb.setter
    def event_cb(self, event_cb):
        self._event_cb = SequencerEventCallback(event_cb)

    @property
    def event_cb_ctx(self):
        return self._event_cb_ctx

    @event_cb_ctx.setter
    def event_cb_ctx(self, event_cb_ctx):
        self._event_cb_ctx = event_cb_ctx

    def reset(self):
        self._lib.sstvenc_sequencer_reset(ctypes.byref(self._seq))

    def advance(self):
        self._lib.sstvenc_sequencer_advance(ctypes.byref(self._seq))

    def compute(self):
        self._lib.sstvenc_sequencer_compute(ctypes.byref(self._seq))

    def __iter__(self):
        return self

    def __next__(self):
        self.compute()
        if self.state is not SequencerState.DONE:
            return self.output
        else:
            raise StopIteration

    def read(self, n_samples):
        buffer = (ctypes.c_double * n_samples)()
        out_samples = self._lib.sstvenc_sequencer_fill_buffer(
            ctypes.byref(self._seq), buffer, n_samples
        )
        return array.array("d", iter(buffer[:out_samples]))


class SequencerBuilder(object):
    def __init__(
        self, lib, sample_rate=48000, event_cb=None, event_cb_ctx=None
    ):
        self._lib = lib
        self._steps = []
        self._steps_ptr = None
        self._sample_rate = sample_rate
        self._event_cb = event_cb
        self._event_cb_ctx = event_cb_ctx

    def set_timescale_unit(self, time_unit, convert=False):
        assert self._steps_ptr is None, "Already created"
        self._steps.append(
            SequencerStepSetTimescaleUnit(self._lib, time_unit, convert)
        )
        return self

    def set_reg(self, reg, value):
        assert self._steps_ptr is None, "Already created"
        self._steps.append(SequencerStepSetRegister(self._lib, reg, value))
        return self

    def inc_reg(self, reg, value):
        assert self._steps_ptr is None, "Already created"
        self._steps.append(SequencerStepIncRegister(self._lib, reg, value))
        return self

    def dec_reg(self, reg, value):
        assert self._steps_ptr is None, "Already created"
        self._steps.append(SequencerStepDecRegister(self._lib, reg, value))
        return self

    def mul_reg(self, reg, value):
        assert self._steps_ptr is None, "Already created"
        self._steps.append(SequencerStepMulRegister(self._lib, reg, value))
        return self

    def div_reg(self, reg, value):
        assert self._steps_ptr is None, "Already created"
        self._steps.append(SequencerStepDivRegister(self._lib, reg, value))
        return self

    def idec_reg(self, reg, value):
        assert self._steps_ptr is None, "Already created"
        self._steps.append(SequencerStepIDecRegister(self._lib, reg, value))
        return self

    def idiv_reg(self, reg, value):
        assert self._steps_ptr is None, "Already created"
        self._steps.append(SequencerStepIDivRegister(self._lib, reg, value))
        return self

    def silence(self, duration, time_unit=None):
        if time_unit is not None:
            self.set_timescale_unit(time_unit, True)

        assert self._steps_ptr is None, "Already created"
        self._steps.append(SequencerStepSilence(self._lib, duration))
        return self

    def tone(
        self,
        duration,
        slopes=SequencerStepToneSlopes.BOTH,
        frequency=None,
        amplitude=None,
        phase=None,
        pulse_rise=None,
        pulse_fall=None,
        time_unit=None,
    ):
        if time_unit is not None:
            self.set_timescale_unit(time_unit, True)

        if amplitude is not None:
            self.set_reg(SequencerRegister.AMPLITUDE, amplitude)

        if frequency is not None:
            self.set_reg(SequencerRegister.FREQUENCY, frequency)

        if phase is not None:
            self.set_reg(SequencerRegister.PHASE, phase)

        if pulse_rise is not None:
            self.set_reg(SequencerRegister.PULSE_RISE, pulse_rise)

        if pulse_fall is not None:
            self.set_reg(SequencerRegister.PULSE_FALL, pulse_fall)

        assert self._steps_ptr is None, "Already created"
        self._steps.append(SequencerStepTone(self._lib, duration))
        return self

    def cw(
        self,
        text,
        frequency=None,
        amplitude=None,
        pulse_rise=None,
        dit_period=None,
        time_unit=None,
    ):
        if time_unit is not None:
            self.set_timescale_unit(time_unit, True)

        if amplitude is not None:
            self.set_reg(SequencerRegister.AMPLITUDE, amplitude)

        if frequency is not None:
            self.set_reg(SequencerRegister.FREQUENCY, frequency)

        if pulse_rise is not None:
            self.set_reg(SequencerRegister.PULSE_RISE, pulse_rise)

        if dit_period is not None:
            self.set_reg(SequencerRegister.DIT_PERIOD, dit_period)

        assert self._steps_ptr is None, "Already created"
        self._steps.append(SequencerStepCW(self._lib, text))
        return self

    def image(
        self,
        mode,
        framebuffer,
        fsk_id=None,
        amplitude=None,
        pulse_rise=None,
        pulse_fall=None,
        time_unit=None,
    ):
        if time_unit is not None:
            self.set_timescale_unit(time_unit, True)

        if amplitude is not None:
            self.set_reg(SequencerRegister.AMPLITUDE, amplitude)

        if pulse_rise is not None:
            self.set_reg(SequencerRegister.PULSE_RISE, pulse_rise)

        if pulse_fall is not None:
            self.set_reg(SequencerRegister.PULSE_FALL, pulse_fall)

        assert self._steps_ptr is None, "Already created"
        self._steps.append(
            SequencerStepImage(self._lib, mode, framebuffer, fsk_id)
        )
        return self

    def build(self):
        # Build the steps.
        steps = (_SequencerStep * (len(self._steps) + 1))()
        for py_step, c_step in zip(self._steps, steps):
            py_step._build(c_step)

        # Put an END step as a final act.
        self._lib.sstvenc_sequencer_step_end(ctypes.byref(steps[-1]))

        # Wrap the event callback
        event_cb = None
        if self._event_cb:
            event_cb = SequencerEventCallback(event_cb)

        # Create the sequence
        seq = _Sequencer()
        self._lib.sstvenc_sequencer_init(
            ctypes.byref(seq),
            steps,
            event_cb,
            self._event_cb_ctx,
            self._sample_rate,
        )
        return Sequencer(self._lib, seq, self._steps)


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
            ctypes.POINTER(_CWMod),  # cw
            ctypes.c_char_p,  # text
            ctypes.c_double,  # amplitude
            ctypes.c_double,  # frequency
            ctypes.c_double,  # dit_period
            ctypes.c_double,  # slope_period
            ctypes.c_uint32,  # sample_rate
            ctypes.c_uint8,  # time_unit
        )

        self._lib.sstvenc_cw_compute.restype = None
        self._lib.sstvenc_cw_compute.argtypes = (ctypes.POINTER(_CWMod),)

        self._lib.sstvenc_cw_fill_buffer.restype = ctypes.c_size_t
        self._lib.sstvenc_cw_fill_buffer.argtypes = (
            ctypes.POINTER(_CWMod),  # cw
            ctypes.POINTER(ctypes.c_double),  # buffer
            ctypes.c_size_t,  # buffer_sz
        )

        # oscillator.h
        self._lib.sstvenc_osc_get_frequency.restype = ctypes.c_double
        self._lib.sstvenc_osc_get_frequency.argtypes = (
            ctypes.POINTER(_Oscillator),
        )

        self._lib.sstvenc_osc_set_frequency.restype = None
        self._lib.sstvenc_osc_set_frequency.argtypes = (
            ctypes.POINTER(_Oscillator),
            ctypes.c_double,
        )

        self._lib.sstvenc_osc_init.restype = None
        self._lib.sstvenc_osc_init.argtypes = (
            ctypes.POINTER(_Oscillator),  # osc
            ctypes.c_double,  # amplitude
            ctypes.c_double,  # frequency
            ctypes.c_double,  # offset
            ctypes.c_uint32,  # sample_rate
        )

        self._lib.sstvenc_osc_compute.restype = None
        self._lib.sstvenc_osc_compute.argtypes = (
            ctypes.POINTER(_Oscillator),
        )

        # pulseshape.h
        self._lib.sstvenc_ps_reset_samples.restype = None
        self._lib.sstvenc_ps_reset_samples.argtypes = (
            ctypes.POINTER(_PulseShape),
            ctypes.c_uint32,
        )

        self._lib.sstvenc_ps_reset.restype = None
        self._lib.sstvenc_ps_reset.argtypes = (
            ctypes.POINTER(_PulseShape),  # ps
            ctypes.c_double,  # hold_time
            ctypes.c_uint8,  # time_unit
        )

        self._lib.sstvenc_ps_init.restype = None
        self._lib.sstvenc_ps_init.argtypes = (
            ctypes.POINTER(_PulseShape),  # ps
            ctypes.c_double,  # amplitude
            ctypes.c_double,  # rise_time
            ctypes.c_double,  # hold_time
            ctypes.c_double,  # fall_time
            ctypes.c_uint32,  # sample_rate
            ctypes.c_uint8,  # time_unit
        )

        self._lib.sstvenc_ps_advance.restype = None
        self._lib.sstvenc_ps_advance.argtypes = (ctypes.POINTER(_PulseShape),)

        self._lib.sstvenc_ps_compute.restype = None
        self._lib.sstvenc_ps_compute.argtypes = (ctypes.POINTER(_PulseShape),)

        # sequence.h
        self._lib.sstvenc_sequencer_step_set_timescale.restype = None
        self._lib.sstvenc_sequencer_step_set_timescale.argtypes = (
            ctypes.POINTER(_SequencerStep),
            ctypes.c_uint8,
            ctypes.c_bool,
        )

        self._lib.sstvenc_sequencer_step_set_reg.restype = None
        self._lib.sstvenc_sequencer_step_set_reg.argtypes = (
            ctypes.POINTER(_SequencerStep),
            ctypes.c_uint8,
            ctypes.c_double,
        )

        self._lib.sstvenc_sequencer_step_inc_reg.restype = None
        self._lib.sstvenc_sequencer_step_inc_reg.argtypes = (
            ctypes.POINTER(_SequencerStep),
            ctypes.c_uint8,
            ctypes.c_double,
        )

        self._lib.sstvenc_sequencer_step_dec_reg.restype = None
        self._lib.sstvenc_sequencer_step_dec_reg.argtypes = (
            ctypes.POINTER(_SequencerStep),
            ctypes.c_uint8,
            ctypes.c_double,
        )

        self._lib.sstvenc_sequencer_step_mul_reg.restype = None
        self._lib.sstvenc_sequencer_step_mul_reg.argtypes = (
            ctypes.POINTER(_SequencerStep),
            ctypes.c_uint8,
            ctypes.c_double,
        )

        self._lib.sstvenc_sequencer_step_div_reg.restype = None
        self._lib.sstvenc_sequencer_step_div_reg.argtypes = (
            ctypes.POINTER(_SequencerStep),
            ctypes.c_uint8,
            ctypes.c_double,
        )

        self._lib.sstvenc_sequencer_step_idec_reg.restype = None
        self._lib.sstvenc_sequencer_step_idec_reg.argtypes = (
            ctypes.POINTER(_SequencerStep),
            ctypes.c_uint8,
            ctypes.c_double,
        )

        self._lib.sstvenc_sequencer_step_idiv_reg.restype = None
        self._lib.sstvenc_sequencer_step_idiv_reg.argtypes = (
            ctypes.POINTER(_SequencerStep),
            ctypes.c_uint8,
            ctypes.c_double,
        )

        self._lib.sstvenc_sequencer_step_silence.restype = None
        self._lib.sstvenc_sequencer_step_silence.argtypes = (
            ctypes.POINTER(_SequencerStep),
            ctypes.c_double,
        )

        self._lib.sstvenc_sequencer_step_tone.restype = None
        self._lib.sstvenc_sequencer_step_tone.argtypes = (
            ctypes.POINTER(_SequencerStep),
            ctypes.c_double,
        )

        self._lib.sstvenc_sequencer_step_cw.restype = None
        self._lib.sstvenc_sequencer_step_cw.argtypes = (
            ctypes.POINTER(_SequencerStep),
            ctypes.c_char_p,
        )

        self._lib.sstvenc_sequencer_step_image.restype = None
        self._lib.sstvenc_sequencer_step_image.argtypes = (
            ctypes.POINTER(_SequencerStep),
            ctypes.POINTER(_Mode),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_char_p,
        )

        self._lib.sstvenc_sequencer_step_end.restype = None
        self._lib.sstvenc_sequencer_step_end.argtypes = (
            ctypes.POINTER(_SequencerStep),
        )

        self._lib.sstvenc_sequencer_init.restype = None
        self._lib.sstvenc_sequencer_init.argtypes = (
            ctypes.POINTER(_Sequencer),
            ctypes.POINTER(_SequencerStep),
            ctypes.POINTER(SequencerEventCallback),
            ctypes.c_void_p,
        )

        self._lib.sstvenc_sequencer_reset.restype = None
        self._lib.sstvenc_sequencer_reset.argtypes = (
            ctypes.POINTER(_Sequencer),
        )

        self._lib.sstvenc_sequencer_advance.restype = None
        self._lib.sstvenc_sequencer_advance.argtypes = (
            ctypes.POINTER(_Sequencer),
        )

        self._lib.sstvenc_sequencer_compute.restype = None
        self._lib.sstvenc_sequencer_compute.argtypes = (
            ctypes.POINTER(_Sequencer),
        )

        self._lib.sstvenc_sequencer_fill_buffer.restype = None
        self._lib.sstvenc_sequencer_fill_buffer.argtypes = (
            ctypes.POINTER(_Sequencer),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        )

        # sstv.h
        self._lib.sstvenc_encoder_init.restype = None
        self._lib.sstvenc_encoder_init.argtypes = (
            ctypes.POINTER(_Encoder),
            ctypes.POINTER(_Mode),
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint8),
        )

        self._lib.sstvenc_encoder_next_pulse.restype = ctypes.POINTER(Pulse)
        self._lib.sstvenc_encoder_next_pulse.argtypes = (
            ctypes.POINTER(_Encoder),
        )

        # sstvfreq.h
        self._lib.sstvenc_level_freq.restype = ctypes.c_int16
        self._lib.sstvenc_level_freq.argtypes = (ctypes.c_uint8,)

        # sstvmod.h
        self._lib.sstvenc_modulator_init.restype = None
        self._lib.sstvenc_modulator_init.argtypes = (
            ctypes.POINTER(_Modulator),  # mod
            ctypes.POINTER(_Mode),  # mode
            ctypes.c_char_p,  # fsk_id
            ctypes.POINTER(ctypes.c_uint8),  # framebuffer
            ctypes.c_double,  # rise_time
            ctypes.c_double,  # fall_time
            ctypes.c_uint32,  # sample_rate
            ctypes.c_uint8,  # time_unit
        )

        self._lib.sstvenc_modulator_fill_buffer.restype = ctypes.c_size_t
        self._lib.sstvenc_modulator_fill_buffer.argtypes = (
            ctypes.POINTER(_Modulator),  # mod
            ctypes.POINTER(ctypes.c_double),  # buffer
            ctypes.c_size_t,  # buffer_sz
        )

        # sstvmode.h
        self._lib.sstvenc_get_mode_by_idx.restype = ctypes.POINTER(_Mode)
        self._lib.sstvenc_get_mode_by_idx.argtypes = (ctypes.c_uint8,)

        self._lib.sstvenc_get_mode_by_name.restype = ctypes.POINTER(_Mode)
        self._lib.sstvenc_get_mode_by_name.argtypes = (ctypes.c_char_p,)

        self._lib.sstvenc_pulseseq_get_txtime.restype = ctypes.c_uint64
        self._lib.sstvenc_pulseseq_get_txtime.argtypes = (
            ctypes.POINTER(Pulse),
        )

        self._lib.sstvenc_mode_get_txtime.restype = ctypes.c_uint64
        self._lib.sstvenc_mode_get_txtime.argtypes = (
            ctypes.POINTER(_Mode),
            ctypes.c_char_p,
        )

        self._lib.sstvenc_mode_get_fb_sz.restype = ctypes.c_size_t
        self._lib.sstvenc_mode_get_fb_sz.argtypes = (ctypes.POINTER(_Mode),)

        self._lib.sstvenc_get_pixel_posn.restype = ctypes.c_uint32
        self._lib.sstvenc_get_pixel_posn.argtypes = (
            ctypes.POINTER(_Mode),
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
            ctypes.POINTER(_SunAUEnc),  # enc
            ctypes.c_char_p,  # path
            ctypes.c_uint32,  # sample_rate
            ctypes.c_uint8,  # encoding
            ctypes.c_uint8,  # channels
        )

        self._lib.sstvenc_sunau_enc_write.restype = ctypes.c_int
        self._lib.sstvenc_sunau_enc_write.argtypes = (
            ctypes.POINTER(_SunAUEnc),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
        )

        self._lib.sstvenc_sunau_enc_close.restype = ctypes.c_int
        self._lib.sstvenc_sunau_enc_write.argtypes = (
            ctypes.POINTER(_SunAUEnc),
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
        time_unit=TimescaleUnit.MILLISECONDS,
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
        time_unit = TimescaleUnit(time_unit)

        if mod is None:
            mod = _CWMod()

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
        return CWMod(self._lib, mod)

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
            osc = _Oscillator()

        self._lib.sstvenc_osc_init(
            ctypes.byref(osc), amplitude, frequency, offset, sample_rate
        )
        return Oscillator(self._lib, osc)

    # pulseshape.h

    def init_ps(
        self,
        rise_time,
        hold_time=None,
        fall_time=None,
        amplitude=1.0,
        sample_rate=48000,
        time_unit=TimescaleUnit.MILLISECONDS,
        ps=None,
    ):
        amplitude = float(amplitude)
        time_unit = TimescaleUnit(time_unit)
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
            ps = _PulseShape()

        self._lib.sstvenc_ps_init(
            ctypes.byref(ps),
            amplitude,
            rise_time,
            hold_time,
            fall_time,
            sample_rate,
            time_unit.value,
        )
        return PulseShape(self._lib, ps)

    # sequence.h

    def build_sequence(
        self, sample_rate=48000, event_cb=None, event_cb_ctx=None
    ):
        return SequencerBuilder(self._lib, sample_rate, event_cb, event_cb_ctx)

    # sstv.h

    def init_enc(self, mode, fsk_id=None, framebuffer=None, enc=None):
        if framebuffer is None:
            framebuffer = mode.mkbuffer()

        if enc is None:
            enc = _Encoder()

        if fsk_id:
            fsk_id = str(fsk_id).encode("US-ASCII")
        else:
            fsk_id = None

        self._lib.sstvenc_encoder_init(
            ctypes.byref(enc), mode._mode, fsk_id, framebuffer
        )
        return Encoder(self._lib, enc)

    # sstvmod.h

    def init_mod(
        self,
        mode,
        fsk_id=None,
        framebuffer=None,
        sample_rate=48000,
        rise_time=10.0,
        fall_time=None,
        time_unit=TimescaleUnit.MILLISECONDS,
        mod=None,
    ):
        time_unit = TimescaleUnit(time_unit)
        rise_time = float(rise_time)

        if framebuffer is None:
            framebuffer = mode.mkbuffer()

        if fall_time is None:
            fall_time = rise_time
        else:
            fall_time = float(fall_time)

        if mod is None:
            mod = _Modulator()

        if fsk_id:
            fsk_id = str(fsk_id).encode("US-ASCII")
        else:
            fsk_id = None

        self._lib.sstvenc_modulator_init(
            ctypes.byref(mod),
            mode._mode,
            fsk_id,
            framebuffer,
            rise_time,
            fall_time,
            sample_rate,
            time_unit.value,
        )
        return Modulator(self._lib, mod)

    # sstvmode.h

    def get_sstv_mode_count(self):
        return self._lib.get_sstv_mode_count()

    def _return_sstv_mode(self, mode):
        if mode:
            return Mode(self._lib, mode)
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
        encoding=SunAUFormat.S16,
        channels=1,
        enc=None,
    ):

        sample_rate = int(sample_rate)
        encoding = SunAUFormat(encoding)
        channels = int(channels)

        if enc is None:
            enc = _SunAUEnc()

        _errno_to_oserror(
            self._lib.sstvenc_sunau_enc_init(
                ctypes.byref(enc),
                path.encode("UTF-8"),
                sample_rate,
                encoding.value,
                channels,
            )
        )

        return SunAuEnc(self._lib, enc)

    # yuv.h

    def convert_rgb_to_yuv(self, src, width, height, dest=None):
        if dest is None:
            dest = src

        self._lib.sstvenc_rgb_to_yuv(dest, src, width, height)

    def convert_rgb_to_mono(self, src, width, height, dest=None):
        if dest is None:
            dest = src

        self._lib.sstvenc_rgb_to_mono(dest, src, width, height)
