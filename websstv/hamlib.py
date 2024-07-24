#!/usr/bin/env python3

"""
Hamlib utility library.  This tries to simplify the set-up of Hamlib
interfaces.
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later
import Hamlib

from .subproc import ChildProcessWrapper

import logging
import asyncio
import uuid
import traceback
from enum import Enum
from multiprocessing import Process, Pipe
from sys import exc_info

# Defaults for all models
_ALL_MODEL_DEFAULTS = {}

# Sane defaults by radio model.
_MODEL_DEFAULTS = {
    Hamlib.RIG_MODEL_NETRIGCTL: {"rig_pathname": "localhost:4532"}
}


def parse_enum(name, value):
    """
    Parse an enumeration given by name into something meaningful to Hamlib.
    """
    if isinstance(value, str):
        value = getattr(Hamlib, "%s_%s" % (name, value.upper()))
    return value


class _RigMessage(Enum):
    LOG = "log"
    RPC_RQ = "rpc_rq"
    RPC_RES = "rpc_res"
    RPC_ERR = "rpc_err"
    ERROR = "error"
    EXIT = "exit"


class Rig(ChildProcessWrapper):
    """
    Asynchronous sub-process wrapper around the hamlib rig instance.
    """

    _Message = _RigMessage

    def __init__(
        self,
        model=Hamlib.RIG_MODEL_NETRIGCTL,
        rig_config=None,
        debug=Hamlib.RIG_DEBUG_NONE,
        poll_interval=0.1,
        loop=None,
        log=None,
    ):
        super().__init__(poll_interval=poll_interval, loop=loop, log=log)

        model = parse_enum("RIG_MODEL", model)
        debug = parse_enum("RIG_DEBUG", debug)

        config = {}

        config.update(_ALL_MODEL_DEFAULTS)
        try:
            config.update(_MODEL_DEFAULTS[model])
        except KeyError:
            pass
        if rig_config is not None:
            config.update(rig_config)

        self._model = model
        self._config = config
        self._debug = debug

    def get_freq(self, vfo=Hamlib.RIG_VFO_CURR):
        vfo = parse_enum("RIG_VFO", vfo)
        return self._call("get_freq", vfo)

    def set_freq(self, freq, vfo=Hamlib.RIG_VFO_CURR):
        vfo = parse_enum("RIG_VFO", vfo)
        return self._call("set_freq", vfo, freq)

    def get_level(self, level, vfo=Hamlib.RIG_VFO_CURR):
        vfo = parse_enum("RIG_VFO", vfo)
        level = parse_enum("RIG_LEVEL", level)
        return self._call("get_level", vfo, level)

    def get_level_i(self, level, vfo=Hamlib.RIG_VFO_CURR):
        vfo = parse_enum("RIG_VFO", vfo)
        level = parse_enum("RIG_LEVEL", level)
        return self._call("get_level_i", level, vfo)

    def get_ptt(self, vfo=Hamlib.RIG_VFO_CURR):
        vfo = parse_enum("RIG_VFO", vfo)
        return self._call("get_ptt", vfo)

    def set_ptt(self, state, vfo=Hamlib.RIG_VFO_CURR):
        vfo = parse_enum("RIG_VFO", vfo)
        state = parse_enum("RIG_PTT", state)
        return self._call("set_ptt", vfo, state)

    def _child_init(self, parent_pipe):
        Hamlib.rig_set_debug(self._debug)
        rig = Hamlib.Rig(self._model)

        for param, value in self._config.items():
            rig.set_conf(param, str(value))

        rig.open()
        return rig

    def _child_poll_tasks(self, parent_pipe, client):
        # Nothing to do
        pass


if __name__ == "__main__":
    import argparse

    async def main():
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "--model", type=str, default="netrigctl", help="hamlib model"
        )
        ap.add_argument(
            "--rig-config",
            default=[],
            action="append",
            nargs=2,
            help="rigctld model configuration argument",
        )
        ap.add_argument(
            "--rig-debug", default="none", help="hamlib debug level"
        )

        args = ap.parse_args()

        try:
            # Model ID?
            rig_model = int(args.model)
        except ValueError:
            # Must be a model name
            rig_model = args.model

        rig_config = {}
        for param, value in args.rig_config:
            try:
                value = int(value)
            except ValueError:
                pass

            try:
                if isinstance(value, str):
                    value = float(value)
            except ValueError:
                pass

            rig_config[param] = value

        logging.basicConfig(level=logging.DEBUG)

        client = Rig(
            model=rig_model, rig_config=rig_config, debug=args.rig_debug
        )
        client.start()

        try:
            print("Frequency: %r Hz" % await client.get_freq())
            print("Signal strength: %r" % await client.get_level_i("STRENGTH"))
            print("PTT: %r" % await client.get_ptt())
        finally:
            client.stop()

    asyncio.run(main())
