#!/usr/bin/env python3

"""
SSTV encoder with thread-based wrapper.
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

import ctypes

from PIL import Image

from .raster import (
    RasterDimensions,
    RasterHJustify,
    RasterVJustify,
    scale_image,
)


def read_image(
    imagefile,
    width,
    height,
    fill=False,
    hjust=RasterHJustify.CENTRE,
    vjust=RasterVJustify.CENTRE,
    resample=None,
    buffer=None,
):

    pixels = (
        scale_image(
            image=Image.open(imagefile),
            dimensions=RasterDimensions(width, height),
            fill=fill,
            hjust=hjust,
            vjust=vjust,
            resample=resample,
        )
        .convert("RGB")
        .load()
    )

    # Extract the RGB data
    buffer = (ctypes.c_uint8 * (width * height * 3))()
    ptr = 0
    for y in range(height):
        for x in range(width):
            # Read
            (r, g, b) = pixels[(x, y)]

            # Write
            buffer[ptr + 0] = r
            buffer[ptr + 1] = g
            buffer[ptr + 2] = b

            ptr += 3

    return buffer
