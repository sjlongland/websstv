#!/usr/bin/env python3

"""
Simple tone generator, for CW ID, repeater tones and tuning.
"""

from .sunaudio import get_spec
from math import cos, pi


class Oscillator(object):
    def __init__(self, sample_rate, encoding, amplitude=1.0, phase=0):
        self._sample_rate = sample_rate
        self._phase = phase
        self._amplitude = amplitude

        spec = get_spec(encoding)
        self._type = spec.pythontype
        if spec.pythontype is float:
            self._scale = 1.0
            self._min = -1.0
            self._max = 1.0
        else:
            self._scale = 2 ** (spec.bits - 1)
            self._min = -self._scale
            self._max = self._scale - 1

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, new_phase):
        self._phase = new_phase

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, new_amplitude):
        self._amplitude = max(0.0, min(1.0, new_amplitude))

    def samples(self, duration):
        """
        Return the number of samples needed to satisfy this duration.
        """
        return int((duration * self.sample_rate) + 0.5)

    def silence(self, duration):
        """
        Return the specified duration of silence.
        """
        for n in range(self.samples(duration)):
            yield self._type(0)

    def generate(self, frequency, duration):
        """
        Return the specified duration of tone, at the given frequency.
        """
        for n in range(self.samples(duration)):
            time = n / self.sample_rate
            phase = 2 * pi * frequency * time + self._phase
            sample = self.amplitude * cos(phase)

            yield self._type(
                max(self._min, min(self._max, sample * self._scale))
            )

        self._phase = phase % (2 * pi)
