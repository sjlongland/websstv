#!/usr/bin/env python3

"""
SVG rasteriser interface.  This abstract interface allows for different
rasteriser engines to be potentially added (right now, Inkscape, but maybe in
the future using a headless browser or Batik to render SVG as a PNG).
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

from collections.abc import Sequence
from collections import namedtuple


RasterPosition = namedtuple("RasterPosition", ["x", "y"])


class RasterDimensions(Sequence):
    """
    Representation of a raster image's dimensions in pixels.  The Sequence
    interface is implemented to support tuple-like behaviour.
    """

    IDX_WIDTH = 0
    IDX_HEIGHT = 1
    LENGTH = 2

    # Justification options
    JUST_LEFT = -1
    JUST_TOP = -1
    JUST_CENTRE = 0
    JUST_RIGHT = 1
    JUST_BOTTOM = 1

    @classmethod
    def _justify(cls, just, obj_dim, cont_dim):
        """
        Justify the object dimension within the container dimension.  Positive
        values mean an offset into the container (from left or top), negative
        means the object must be cropped to fit the container by the number of
        pixels left/right or top/bottom.
        """
        if just < 0:
            # Left/Top justify… just return 0
            return 0

        pos = cont_dim - obj_dim

        if just == 0:
            # Centre justify: divide difference by two
            pos /= 2

        # Round to nearest pixel
        return int(pos + 0.5)

    def __init__(self, width, height):
        """
        Create a new raster dimensions object.
        """
        # Round inputs to the nearest integer
        self._width = int(width + 0.5)
        self._height = int(height + 0.5)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def ratio(self):
        return float(self.width) / float(self.height)

    def __repr__(self):
        return "%s(width=%r, height=%r)" % (
            self.__class__.__name__,
            self.width,
            self.height,
        )

    def scale(self, factor):
        """
        Scale the width and height by the given factor.
        """
        return self.__class__(
            width=self.width * factor, height=self.height * factor
        )

    def fill_width(self, width):
        """
        Scale the dimensions to fill the specified width.
        """
        return self.__class__(width=width, height=width / self.ratio)

    def fill_height(self, height):
        """
        Scale the dimensions to fill the specified height.
        """
        return self.__class__(width=height * self.ratio, height=height)

    def fit_container(self, container, hjust=JUST_CENTRE, vjust=JUST_CENTRE):
        """
        Determine the positioning of this object within the given container
        dimensions that will fit the space.  There will be empty border bars
        around the object.
        """
        rdiff = self.ratio - container.ratio

        if rdiff < 0:
            # Scale by container height
            obj = self.fill_height(container.height)
        elif rdiff > 0:
            # Scale by container width
            obj = self.fill_width(container.width)
        else:
            # Container and object are an exact fit
            return (container, RasterPosition(0, 0))

        x = self._justify(hjust, obj.width, container.width)
        y = self._justify(vjust, obj.height, container.height)

        return (obj, RasterPosition(x, y))

    def fill_container(self, container, hjust=JUST_CENTRE, vjust=JUST_CENTRE):
        """
        Determine the positioning of this object within the given container
        dimensions that will fill the space.  The object will be cropped.
        """
        rdiff = self.ratio - container.ratio

        if rdiff < 0:
            # Scale by container width
            obj = self.fill_width(container.width)
        elif rdiff > 0:
            # Scale by container height
            obj = self.fill_height(container.height)
        else:
            # Container and object are an exact fit
            return (container, RasterPosition(0, 0))

        x = self._justify(hjust, obj.width, container.width)
        y = self._justify(vjust, obj.height, container.height)

        return (obj, RasterPosition(x, y))

    # Sequence interface

    def __getitem__(self, idx):
        if idx == self.IDX_WIDTH:
            return self.width
        elif idx == self.IDX_HEIGHT:
            return self.height
        else:
            raise IndexError(idx)

    def __len__(self):
        return self.LENGTH

    def __iter__(self):
        yield self.width
        yield self.height
