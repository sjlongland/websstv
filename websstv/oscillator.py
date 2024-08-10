#!/usr/bin/env python3

"""
Simple tone generator, for CW ID, repeater tones and tuning.
"""

from .sunaudio import get_spec
from math import cos, pi


class SampledTimebase(object):
    def __init__(self, sample_rate):
        self._sample_rate = sample_rate
        self._indefinite = False

    @property
    def sample_rate(self):
        """
        Return the sample rate of this discrete timebase
        """
        return self._sample_rate

    def samples(self, duration):
        """
        Compute the number of samples for the given duration.
        """
        return int((duration * self.sample_rate) + 0.5)

    def timeof(self, sample):
        """
        Return the time in seconds since the start of the recording epoch
        for the given sample.
        """
        return float(sample) / float(self.sample_rate)

    def timerange(self, duration=None, start=0.0, tail=0.0):
        """
        Return the time samples for each timestamp over the specified
        duration.  ``start`` is assumed to be aligned with a sample.

        If ``tail`` is given, this specifies the time duration after receiving
        a stop signal to continue emitting samples for (for pulse shaping
        purposes).
        """
        if duration is None:
            self._indefinite = True
            tail_n = self.samples(tail)

        step = 1 / float(self.sample_rate)

        if duration is None:
            n = 0
            while (self._indefinite is True) or (tail_n > 0):
                yield start + (float(n) * step)
                n += 1
                tail_n -= 1
        else:
            for n in range(self.samples(duration)):
                yield start + (float(n) * step)

    def stop(self):
        """
        Stop an indefinite time series.
        """
        self._indefinite = False


class PulseShaper(SampledTimebase):
    @staticmethod
    def _sample(time):
        return (1 + cos(pi * time)) / 2

    def pulse(self, duration=None, risetime=0.01, falltime=None):
        # Set a default amplitude to handle the no-rise-time case
        amplitude = 1.0

        # Compute number of samples for the rise time
        rise_samples = self.samples(risetime)

        if falltime is None:
            falltime = risetime
            fall_samples = rise_samples
        else:
            fall_samples = self.samples(falltime)

        if duration is not None:
            total_samples = self.samples(duration)

            # Ensure we've got enough time for rise/fall
            slope_samples = rise_samples + fall_samples
            if total_samples < slope_samples:
                # There isn't enough time… so we won't reach full amplitude.
                # The difference in periods gives us the rise/fall time we have.
                diff = slope_samples - total_samples
                rise_crop = int((diff / 2) + 0.5)
                fall_crop = diff - rise_crop

                rise_skip = rise_samples - rise_crop
                fall_skip = fall_samples - fall_crop
            else:
                rise_skip = 0
                fall_crop = fall_samples
                fall_skip = 0
        else:
            # It's an indefinite pulse.
            rise_skip = 0
            fall_crop = fall_samples
            fall_skip = 0

        # Yield the rise samples
        for amplitude in self.rise(risetime, skip=rise_skip):
            yield amplitude
            if duration is not None:
                total_samples -= 1

        # Hold the amplitude at whatever the rise got to
        if duration is None:
            while self._indefinite:
                yield amplitude
        else:
            while total_samples > fall_crop:
                yield amplitude
                total_samples -= 1

        # Yield the fall samples
        for amplitude in self.fall(falltime, skip=fall_skip):
            yield amplitude
            if duration is not None:
                total_samples -= 1

    def fall(self, duration, skip=0):
        samples = self.samples(duration)
        if not samples:
            return

        step = 1 / float(samples)
        for n in range(samples):
            if skip > 0:
                skip -= 1
                continue

            yield self._sample(float(n) * step)

    def rise(self, duration, skip=0):
        samples = self.samples(duration)
        if not samples:
            return

        emit = samples - skip
        step = 1 / float(samples)
        for n in reversed(range(samples)):
            if emit > 0:
                yield self._sample(float(n) * step)
                emit -= 1


class Oscillator(object):
    DEFAULT_RISETIME = 0.01

    def __init__(self, sample_rate, encoding, amplitude=1.0, phase=0):
        self._shaper = PulseShaper(sample_rate)
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
        return self._shaper.sample_rate

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
        return self._shaper.samples(duration)

    def timerange(self, duration=None, start=0.0, tail=0.0):
        """
        Return the timestamps for every sample from the start point to the
        duration given.
        """
        return self._shaper.timerange(duration, start, tail)

    def silence(self, duration):
        """
        Return the specified duration of silence.
        """
        for n in range(self.samples(duration)):
            yield self._type(0)

    def generate(
        self,
        frequency,
        duration=None,
        risetime=None,
        falltime=None,
    ):
        """
        Return the specified duration of tone, at the given frequency.
        """
        if risetime is None:
            risetime = self.DEFAULT_RISETIME

        if falltime is None:
            tail = risetime
        else:
            tail = falltime

        for time, envelope in zip(
            self.timerange(duration, tail=tail),
            self._shaper.pulse(duration, risetime, falltime),
        ):
            phase = 2 * pi * frequency * time + self._phase
            sample = self.amplitude * envelope * cos(phase)

            yield self._type(
                max(self._min, min(self._max, sample * self._scale))
            )

        self._phase = phase % (2 * pi)

    def stop(self):
        """
        If generating an indefinitely-long tone, stop it now.
        """
        self._shaper.stop()
