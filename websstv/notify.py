#!/usr/bin/env python3

"""
websstv notification agent.  This is called by slowrxd via its ``-x`` hook
and reports what it receives by passing the data to a Unix domain socket.
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Ideas from:
# https://gist.github.com/Phaiax/ae7d1229e6f078457864dae712c51ae0

import socket
import os
import json
from sys import argv, exit


# Environment variable we use to define what the socket path is.
SOCKET_ENV_VAR = "WEBSSTV_SOCKET_PATH"


def notify():
    """
    Inspect the arguments given and the environment, then notify the websstv
    daemon of the event.
    """
    try:
        socket_path = os.environ[SOCKET_ENV_VAR]
    except KeyError:
        print("%s not set" % SOCKET_ENV_VAR)
        exit(1)

    # Construct the message, we just take the arguments we're given
    # and JSON-encode them
    msg = json.dumps(argv[1:])

    client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    client.sendto(msg.encode("utf-8"), socket_path)
    exit(0)


if __name__ == "__main__":
    notify()
