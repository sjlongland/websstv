#!/usr/bin/env python3

"""
Hamlib utility library.  This tries to simplify the set-up of Hamlib
interfaces.
"""

import Hamlib

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


def init_instance(model, **given_config):
    """
    Initialise an instance of a rig with the given model.  Apply sane defaults
    where possible.
    """
    model = parse_enum("RIG_MODEL", model)
    rig = Hamlib.Rig(model)

    config = {}

    config.update(_ALL_MODEL_DEFAULTS)
    config.update(_MODEL_DEFAULTS.get(model, {}))
    config.update(given_config)

    for param, value in config.items():
        rig.set_conf(param, str(value))

    rig.open()
    return rig


# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later
