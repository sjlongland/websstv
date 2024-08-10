#!/usr/bin/env python3

"""
Path handler: determines or sets sane defaults for the paths used.
"""

import os
import os.path
from sys import argv


def get_runtime_dir():
    """
    This tries for the following environment variables, returning the first
    one that is set.

    - ``WEBSSTV_RUNTIME_DIR``
    - ``XDG_RUNTIME_DIR``
    - ``TMP``
    - ``TEMP``

    If none are present, we assume ``/tmp``.
    """
    for var in (
        "WEBSSTV_RUNTIME_DIR",
        "XDG_RUNTIME_DIR",
        "TMP",
        "TEMP",
    ):
        try:
            return os.path.expanduser(os.environ[var])
        except KeyError:
            pass

    return "/tmp"


def get_app_runtime_dir():
    """
    Determine a safe path beneath the directory returned by
    ``get_runtime_dir`` above.
    """
    return os.path.join(
        get_runtime_dir(), "%s-%d" % (os.path.basename(argv[0]), os.getpid())
    )


def get_home():
    """
    Determine the path for the user's home directory.
    """
    return os.path.expanduser("~")


def get_cache_dir():
    """
    Determine the path for cached files.  If the user does not set one with
    ``WEBSSTV_CACHE_DIR``, derive one from ``HOME`` or the current directory.
    """
    try:
        return os.path.expanduser(os.environ["WEBSSTV_CACHE_DIR"])
    except KeyError:
        pass

    return os.path.join(get_home(), ".local", "share", "websstv")


def get_config_dir():
    """
    Determine the path for config files.  If the user does not set one with
    ``WEBSSTV_CONFIG_DIR``, derive one from ``HOME`` or the current directory.
    """
    try:
        return os.path.expanduser(os.environ["WEBSSTV_CONFIG_DIR"])
    except KeyError:
        pass

    return os.path.join(get_home(), ".config", "websstv")
