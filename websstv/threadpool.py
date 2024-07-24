#!/usr/bin/env python3

"""
Shared Thread pool
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor


class ThreadPool(object):

    _instance = None

    @classmethod
    def get_instance(cls, threads=None):
        if cls._instance is None:
            if threads is None:
                threads = cpu_count()

            cls._instance = ThreadPoolExecutor(threads)

        return cls._instance
