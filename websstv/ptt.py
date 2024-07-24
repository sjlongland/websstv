#!/usr/bin/env python3

"""
PTT control interface.  There are a few ways that PTT is typically engaged:

- RTS or DTR pin on a RS-232 serial interface
- GPIO pin on a single-board computer
- hamlib rigctl interface

This interface provides a generic means of controlling the PTT via the
supported interfaces (RS-232 RTS/DTR, GPIO or hamlib).
"""

import enum

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

_INTERFACES = {}


def init_ptt(**kwargs):
    """
    Retrieve and initialise an instance of the PTT interface configured using
    the given configuration parameters.
    """
    ptt_type = kwargs.pop("type").lower()
    ptt_class = _INTERFACES[ptt_type]
    ptt = ptt_class.from_cfg(**kwargs)
    ptt.state = False
    return ptt


def _register(cls):
    """
    Add the class into a registry for later use.
    """
    for name in (cls.__name__,) + getattr(cls, "ALIASES", ()):
        name = name.lower()
        assert name not in _INTERFACES
        _INTERFACES[name] = cls


class PTTInterface(object):
    """
    Abstract class for defining a PTT control interface.
    """

    @classmethod
    def from_cfg(cls, **config):
        return cls(**config)

    def __init__(self, invert=False):
        if invert:
            self._hw_state = lambda state: not bool(state)
        else:
            self._hw_state = lambda state: bool(state)

    @property
    def state(self):
        return self._get_ptt_state()

    @state.setter
    def state(self, new_state):
        return self._set_ptt_state(new_state)


try:
    from gpio4 import SysfsGPIO

    @_register
    class GPIOPTT(PTTInterface):
        """
        Interface that uses the Sysfs GPIO interface.  Suitable for
        single-board computers such as the Raspberry Pi, BeagleBoard, etc.
        """

        ALIASES = ("gpio4",)

        def __init__(self, pin, invert=False):
            super().__init__(invert=invert)

            self._pin = SysfsGPIO(pin)
            self._pin.direction = b"out"

        def _get_ptt_state(self):
            return self._hw_state(self._pin.value)

        def _set_ptt_state(self, new_state):
            self._pin.value = 1 if self._hw_state(new_state) else 0

except ImportError:
    # No GPIO support
    pass


try:
    import serial

    @_register
    class SerialPTT(PTTInterface):
        """
        Serial port PTT interface using RS-232 DTR or RTS control lines.
        """

        ALIASES = (
            "serial",
            "rs232",
        )

        _PORT_DEFAULTS = dict(
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=None,
            write_timeout=None,
            inter_byte_timeout=None,
            exclusive=None,
        )

        class RS232Pin(enum.Enum):
            RTS = "rts"
            DTR = "dtr"

        @classmethod
        def from_cfg(cls, **config):
            # Pluck out the PTT-specific config settings
            ptt_config = dict(
                pin=cls.RS232Pin(config.pop("pin")),
                invert=config.pop("invert", False),
            )

            port_device = config.pop("port")
            if isinstance(port_device, str):
                port_config = cls._PORT_DEFAULTS.copy()
                port_config.update(config)

                port = serial.Serial(port_device, **port_config)
                port.open()
            else:
                # Assume we were given an already configured port
                port = port_config

            return cls(port, **ptt_config)

        def __init__(self, port, pin, invert=False):
            super().__init__(invert=invert)

            self._port = port
            self._pin = pin

        def _get_ptt_state(self):
            return self._hw_state(getattr(self._port, self._pin.value))

        def _set_ptt_state(self, new_state):
            setattr(self._port, self._pin.value, self._hw_state(new_state))

except ImportError:
    # No serial support
    pass


try:
    import Hamlib
    from . import hamlib

    @_register
    class HamlibPTT(PTTInterface):
        """
        Hamlib PTT interface.  The primary use case for this is talking to
        an already configured ``rigctld`` instance, but conceivably one can
        also talk to a radio directly this way.
        """

        @classmethod
        def from_cfg(cls, **config):
            # Pluck out the PTT-specific config settings
            ptt_config = dict(
                mode=hamlib.parse_enum(
                    "RIG_PTT", config.pop("mode", Hamlib.RIG_PTT_ON)
                ),
                vfo=hamlib.parse_enum(
                    "RIG_VFO", config.pop("vfo", Hamlib.RIG_VFO_CURR)
                ),
                invert=config.pop("invert", False),
            )

            try:
                # See if we were given a rig instance
                rig = config.pop("rig")
            except KeyError:
                # Nope, initialise it ourselves!
                model = config.pop("model", Hamlib.RIG_MODEL_NETRIGCTL)
                rig = hamlib.Rig(model, config)
                rig.start()

            return cls(rig, **ptt_config)

        def __init__(
            self,
            rig,
            mode=Hamlib.RIG_PTT_ON,
            vfo=Hamlib.RIG_VFO_CURR,
            invert=False,
        ):
            super().__init__(invert=invert)

            self._rig = rig
            self._mode = mode
            self._vfo = vfo

        def _get_ptt_state(self):
            return self._hw_state(
                self._rig.get_ptt(self._vfo) != Hamlib.RIG_PTT_OFF
            )

        def _set_ptt_state(self, new_state):
            self._rig.set_ptt(
                self._vfo,
                (
                    self._mode
                    if self._hw_mode(new_state)
                    else Hamlib.RIG_PTT_OFF
                ),
            )

except ImportError:
    # No serial support
    pass
