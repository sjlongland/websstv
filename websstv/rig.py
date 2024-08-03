#!/usr/bin/env python3

"""
Rig control/status interface.  This is an abstraction layer atop the
underlying interfaces for PTT and rig status control.
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

from . import defaults
from .ptt import init_ptt
from .registry import Registry

try:
    import Hamlib
    from .hamlib import Rig, parse_enum

    HAVE_HAMLIB = True
except ImportError:
    HAVE_HAMLIB = False


_REGISTRY = Registry(defaults={"type": "basic", "ptt": None})


def init_rig(**kwargs):
    """
    Initialise a radio interface from the given parameters.
    """
    return _REGISTRY.init_instance(**kwargs)


def _scale_hz(freq):
    """
    Scale a frequency in Hz to some sensible unit.
    """
    # TODO: do people play with SSTV < 1MHz or >1GHz?
    if freq > 1e9:
        return "%.3f MHz" % (freq / 1e6)
    elif freq > 1e6:
        return "%.3f kHz" % (freq / 1e3)
    else:
        return "%d Hz" % freq


@_REGISTRY.register
class BasicRig(object):
    ALIAS = ("basic",)

    @classmethod
    def from_cfg(cls, loop=None, log=None, **config):
        loop = defaults.get_loop(loop)
        log = defaults.get_logger(log, cls.__module__)

        ptt_config = config.pop("ptt", None)
        if ptt_config is not None:
            ptt = init_ptt(**ptt_config, loop=loop, log=log)
        else:
            ptt = None
        return cls(ptt, loop=loop, log=log)

    def __init__(self, ptt=None, loop=None, log=None):
        self._loop = defaults.get_loop(loop)
        self._log = defaults.get_logger(log, self.__class__.__module__)
        self._ptt = ptt

    @property
    def ptt(self):
        return self._ptt

    async def get_freq_unit(self):
        """
        Return the VFO frequency scaled to sensible units.
        """
        raise NotImplementedError(
            "Not support by %s" % self.__class__.__name__
        )

    async def get_s_meter_pts(self):
        """
        Return a text-based representation of the S-meter reading
        as the number of S-points (e.g. "S3", "S9+40dB").
        """
        raise NotImplementedError(
            "Not support by %s" % self.__class__.__name__
        )


if HAVE_HAMLIB:

    @_REGISTRY.register
    class HamlibRig(BasicRig):
        ALIAS = ("hamlib",)

        @classmethod
        def from_cfg(cls, loop=None, log=None, **config):
            loop = defaults.get_loop(loop)
            log = defaults.get_logger(log, cls.__module__)

            ptt_config = config.pop("ptt", {"type": "hamlib"})
            rig = Rig(**config, loop=loop, log=log.getChild("hamlib"))

            if (ptt_config["type"] == "hamlib") and (
                "model" not in ptt_config
            ):
                ptt_config["rig"] = rig

            ptt = init_ptt(**ptt_config, loop=loop, log=log.getChild("ptt"))

            return cls(rig=rig, ptt=ptt, loop=loop, log=log)

        def __init__(
            self, rig, ptt, vfo=Hamlib.RIG_VFO_CURR, loop=None, log=None
        ):
            super().__init__(ptt=ptt, loop=loop, log=log)
            self._rig = rig
            self._vfo = parse_enum("RIG_VFO", vfo)

        @property
        def rig(self):
            return self._rig

        @property
        def vfo(self):
            return self._vfo

        async def get_freq_unit(self):
            """
            Return the VFO frequency scaled to sensible units.
            """
            return _scale_hz(await self._rig.get_freq(self._vfo))

        async def get_s_meter_pts(self):
            """
            Return a text-based representation of the S-meter reading
            as the number of S-points (e.g. "S3", "S9+40dB").
            """
            # The ideal S Meter scale is as follow[sic]: S0=-54, S1=-48, S2=-42,
            # S3=-36, S4=-30, S5=-24, S6=-18, S7=-12, S8=-6, S9=0, +10=10, +20=20,
            # +30=30, +40=40, +50=50 and +60=60.
            # -- https://hamlib.sourceforge.net/manuals/4.3/group__rig.html
            level = await self._get_s_meter_db()

            if level <= 0:
                # S9 is 0dB, S0 is -54dB; each S-point in between is 6dB
                pts = max(0, 54 + level) // 6
                return "S%d" % pts
            else:
                # dB above S9
                return "S9+%02ddB" % level

        async def _get_s_meter_db(self):
            """
            Return the S-meter reading in dB relative to S9.
            """
            return await self._rig.get_level_i(
                Hamlib.RIG_LEVEL_STRENGTH, self._vfo
            )
