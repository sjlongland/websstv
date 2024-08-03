#!/usr/bin/env python3

"""
Main application entrypoint
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

import logging
import logging.config
import inspect
import argparse
import asyncio
import yaml
import os
import os.path
import copy
from signal import SIGINT, SIGQUIT, SIGTERM, SIGHUP

from . import defaults
from .rig import init_rig
from .slowrxd import SlowRXDaemon
from .template import SVGTemplateDirectory
from .path import get_config_dir, get_cache_dir
from .webserver import Webserver

try:
    from .locator import GPSLocator

    GPS_ENABLED = True
except ImportError:
    GPS_ENABLED = False

LOG_FORMAT = (
    "%(asctime)s %(name)s[%(filename)s:%(lineno)4d] %(levelname)s %(message)s"
)
LOG_CONFIG = {
    "version": 1,
    "formatters": {"detail": {"format": LOG_FORMAT}},
    "handlers": {
        "console": {
            "formatter": "detail",
            "class": "logging.StreamHandler",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}


async def asyncmain(args, config, log):
    """
    Asynchronous core main function for the websstv application.
    """
    log.debug("Entering async loop")
    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()

    # ::: Instantiation :::
    # --- core settings, pluck these out for convenience ---
    audio_cfg = config.pop("audio", {})
    sstv_cfg = config.pop("sstv", {})
    fsk_id = sstv_cfg.get("fsk_id")

    # --- templates ---
    template_cfg = sstv_cfg.pop(
        "templates", {"dirname": os.path.join(get_config_dir(), "templates")}
    )
    log.info(
        "Loading templates from %r (recurse %r)",
        template_cfg["dirname"],
        template_cfg.get("recurse", False),
    )
    template_dir = SVGTemplateDirectory(**template_cfg)
    # If FSK ID is set, pass it through to the template
    if fsk_id is not None:
        if template_dir.defaults is None:
            template_dir.defaults = {"callsign": fsk_id}
        else:
            template_dir.defaults.setdefault("callsign", fsk_id)
    # Load the templates
    template_dir.reload()
    log.info(
        "Loaded %d templates from %r", len(template_dir), template_dir.dirname
    )

    # --- GPS locator ---
    locator = None
    if GPS_ENABLED:
        locator_cfg = config.pop("locator", None)
        locator_var = locator_cfg.pop("template_var", "grid")
        locator_nolock_msg = locator_cfg.pop("template_var_nolock", "N/A")
        if locator_cfg is not None:
            log.info("Setting up GPS locator")
            locator = GPSLocator(**locator_cfg, log=log.getChild("locator"))
        else:
            log.info("GPS feature disabled (no configuration provided)")

        if locator_var is not None:

            def _on_tpv(maidenhead, **kwargs):
                existing = template_dir.defaults.get(locator_var)
                if existing == maidenhead:
                    return

                log.debug(
                    "Proxying TPV locator → template: grid %s", maidenhead
                )
                if maidenhead is not None:
                    template_dir.defaults[locator_var] = maidenhead
                elif locator_nolock_msg is not None:
                    template_dir.defaults[locator_var] = locator_nolock_msg

            locator.tpv_received.connect(_on_tpv)
    else:
        log.info("GPS feature disabled (requisite modules not installed)")

    # --- rig control ---
    log.info("Initialising rig control interface")
    rigctl = init_rig(log=log.getChild("rig"), **config.pop("rig", {}))

    # --- slowrx daemon ---
    slowrxd_config = config.pop("slowrxd", {})
    slowrxd_delay = slowrxd_config.pop("startup_delay", 3)
    # Copy across relevant settings from sstv_cfg
    for src, dest in (("channel", "pcm_channel"),):
        if dest not in slowrxd_config:
            try:
                slowrxd_config[dest] = sstv_cfg[src]
            except KeyError:
                # Never mind
                pass

    # Copy audio settings across
    for src, dest in (
        ("sample_rate", "pcm_rate"),
        ("device", "pcm_device"),
    ):
        if dest not in slowrxd_config:
            try:
                slowrxd_config[dest] = audio_cfg[src]
            except KeyError:
                # Never mind
                pass

    # create the daemon handler (don't start yet)
    slowrxd = SlowRXDaemon(log=log.getChild("slowrxd"), **slowrxd_config)

    # create the web server instance
    webserver_cfg = config.pop("webserver", {})
    webserver = Webserver(
        log=log.getChild("webserver"),
        image_dir=slowrxd.image_dir,
        locator=locator,
        rigctl=rigctl,
        **webserver_cfg
    )

    # connect slowrxd and webserver
    slowrxd.slowrxd_event.connect(webserver.on_slowrxd_event)

    # --- hook up signal handlers ---
    loop.add_signal_handler(SIGHUP, template_dir.reload)
    loop.add_signal_handler(SIGINT, shutdown_event.set)
    loop.add_signal_handler(SIGTERM, shutdown_event.set)
    loop.add_signal_handler(SIGQUIT, shutdown_event.set)

    # --- showtime! ---
    log.info("Starting up")
    log.info("- slowrxd")
    await slowrxd.start()
    # Wait a moment and see if it dies
    await asyncio.sleep(slowrxd_delay)
    if slowrxd.pid is None:
        raise RuntimeError("slowrxd died prematurely!")
    else:
        log.info("  slowrxd is running at PID %d", slowrxd.pid)

    if locator is not None:
        log.info("- locator")
        await locator.start()
        log.info("  gpsd version: %s", locator.version)

    if rigctl.rig is not None:
        log.info("- hamlib")
        await rigctl.rig.start()
        log.info("  frequency: %s" % await rigctl.get_freq_unit())
        log.info("  S-meter:   %s" % await rigctl.get_s_meter_pts())
        log.info("  hamlib interface is sane")

    log.info("- ptt")
    if rigctl.ptt is not None:
        # Ensure we're not in transmit while we start up!
        await rigctl.ptt.set_ptt_state(False)
        log.info("  ptt interface is sane")

    log.info("- webserver")
    webserver.start()

    log.info("Waiting for events")
    await shutdown_event.wait()
    log.info("SIGINT/SIGQUIT/SIGTERM received, Shutting down.")

    for process in (rigctl.rig, locator, slowrxd):
        if process is not None:
            res = process.stop()
            if inspect.isawaitable(res):
                await res


def main():
    """
    Synchronous application entrypoint.  This parses command-line arguments,
    loads the configuration file and initialises logging before handing over
    to the asynchronous event loop.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default=os.path.join(get_config_dir(), "websstv.yaml")
    )

    args = ap.parse_args()

    print("Loading configuration from %r" % args.config)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f.read())

    logconfig = config.pop(
        "logging",
        LOG_CONFIG,
    )

    if "version" not in logconfig:
        # Assume simplified config
        level = logconfig.pop("level", "INFO")
        logfmt = logconfig.pop("format", LOG_FORMAT)
        print("Initialising logging using basic config at level %r" % level)

        logconfig = copy.deepcopy(LOG_CONFIG)
        logconfig["formatters"]["detail"]["format"] = logfmt
        logconfig["handlers"]["console"]["level"] = level
        logconfig["root"]["level"] = level

    logging.config.dictConfig(logconfig)
    log = logging.getLogger("websstv")
    log.info("Initialised logging")

    asyncio.run(asyncmain(args, config, log))


if __name__ == "__main__":
    main()
