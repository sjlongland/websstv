#!/usr/bin/env python3

"""
Implementation registry.  Used to allow instantiation of interface classes
from a `dict`-like configuration tree (e.g. loaded from YAML/JSON/TOML).
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later


class Registry(object):
    def __init__(self, defaults=None, typeprop="type", aliasprop="ALIASES"):
        self._typeprop = typeprop
        self._aliasprop = aliasprop
        self._subclasses = {}
        self._defaults = defaults

    def init_instance(**kwargs):
        """
        Retrieve and initialise an instance of a subclass using the given
        parameters.
        """
        if self._defaults is not None:
            defaults = self._defaults.copy()
            defaults.update(kwargs)
            kwargs = defaults

        subclass_name = kwargs.pop(self._typeprop).lower()
        subclass = self._subclasses[subclass_name]

        if hasattr(subclass, "from_cfg"):
            return subclass.from_cfg(**kwargs)
        else:
            return subclass(**kwargs)

    def register(self, subclass):
        """
        Add the class into a registry for later use.
        """
        for name in (subclass.__name__,) + getattr(
            subclass, self._aliasprop, ()
        ):
            name = name.lower()
            assert name not in self._subclasses
            self._subclasses[name] = subclass

        return subclass
