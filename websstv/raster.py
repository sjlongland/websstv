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
import enum

from PIL import Image

from . import defaults
from .extproc import OneShotExternalProcess
from .registry import Registry

_REGISTRY = Registry(defaults={"type": "inkscape"})


def init_rasteriser(**kwargs):
    """
    Initialise a SVG rasteriser with the given parameters.
    """
    return _REGISTRY.init_instance(**kwargs)


RasterPosition = namedtuple("RasterPosition", ["x", "y"])


class RasterHJustify(enum.Enum):
    """
    Description of how to position an image horizontally within a canvas.
    """

    LEFT = -1
    CENTRE = 0
    RIGHT = 1

    @classmethod
    def from_string(cls, string):
        string = string.lower()
        if string.startswith("l"):
            return cls.LEFT
        elif string.startswith("c"):
            return cls.CENTRE
        elif string.startswith("r"):
            return cls.RIGHT
        else:
            raise ValueError("Unrecognised position: %r" % string)


class RasterVJustify(enum.Enum):
    """
    Description of how to position an image vertically within a canvas.
    """

    TOP = -1
    CENTRE = 0
    BOTTOM = 1

    @classmethod
    def from_string(cls, string):
        string = string.lower()
        if string.startswith("t"):
            return cls.TOP
        elif string.startswith("c"):
            return cls.CENTRE
        elif string.startswith("b"):
            return cls.BOTTOM
        else:
            raise ValueError("Unrecognised position: %r" % string)


RasterResample = enum.Enum(
    "_RasterResample",
    dict(
        (name, getattr(Image, name))
        for name in (
            # linear interpolation of contributing pixels
            "BILINEAR",
            # cubic interpolation of contributing pixels
            "BICUBIC",
            # (PIL >= 3.4.0) Each pixel of source image
            # contributes to one pixel of the destination
            # image with identical weights.
            "BOX",
            # Produces a sharper image than BILINEAR
            "HAMMING",
            # (PIL >= 1.1.3) Truncated sinc filter
            "LANCZOS",
            # Pick the nearest pixel, simplest and worst quality
            "NEAREST",
        )
        if hasattr(Image, name)
    ),
)


def pickresamplemethod(*preferences):
    for method in preferences:
        try:
            if isinstance(method, str):
                method = method.upper()

            return RasterResample[method]
        except KeyError:
            pass

    raise ValueError("None of the preferences listed are available")


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

    def fit_container(
        self,
        container,
        hjust=RasterHJustify.CENTRE,
        vjust=RasterVJustify.CENTRE,
    ):
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

        x = self._justify(hjust.value, obj.width, container.width)
        y = self._justify(vjust.value, obj.height, container.height)

        return (obj, RasterPosition(x, y))

    def fill_container(
        self,
        container,
        hjust=RasterHJustify.CENTRE,
        vjust=RasterVJustify.CENTRE,
    ):
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

        x = self._justify(hjust.value, obj.width, container.width)
        y = self._justify(vjust.value, obj.height, container.height)

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


def scale_image(
    image,
    dimensions,
    fill=False,
    hjust=RasterHJustify.CENTRE,
    vjust=RasterVJustify.CENTRE,
    resample=None,
):
    """
    Scale, crop and pad the given image to fit the dimensions given.
    """
    if resample is None:
        # Pick the first available in this order
        resample = pickresamplemethod(
            "HAMMING", "LANCZOS", "BOX", "BICUBIC", "BILINEAR", "NEAREST"
        )
    else:
        resample = RasterResample(resample)

    # Fetch dimensions
    orig_dims = RasterDimensions(width=image.width, height=image.height)

    # Figure out positioning and scaling
    if fill:
        (out_dims, out_pos) = orig_dims.fill_container(
            dimensions, hjust, vjust
        )
    else:
        (out_dims, out_pos) = orig_dims.fit_container(
            dimensions, hjust, vjust
        )

    # Perform scale
    image = image.resize(tuple(out_dims), resample.value)

    if (out_pos.x > 0) or (out_pos.y > 0):
        # Pad to new image size:
        #   - input image is shorter than output:
        #       x == 0
        #       y > 0  : d = out.y - in.y
        #     ⇒ vertically position image within canvas
        #       .--------. .--------.
        #       |--------| |        |
        #       |########| |--------|
        #       |--------| |########|
        #       '--------' '--------'
        #        y = d/2      y = d
        #
        #   - input image is narrower than output:
        #       x > 0  : d = out.x - in.x
        #       y == 0
        #     ⇒ horizontally position image within canvas
        #       .-.----.-. .---.----.
        #       | |####| | |   |####|
        #       | |####| | |   |####|
        #       | |####| | |   |####|
        #       '-'----'-' '---'----'
        #        x = d/2      x = d
        newimg = Image.new("RGB", tuple(dimensions))

        newimg.paste(image, tuple(out_pos))
        image = newimg
    elif (out_pos.x < 0) or (out_pos.y < 0):
        # Crop the image to fit the container
        #   - input image is taller than output:
        #       x == 0
        #       y < 0  : d = out.y - in.y
        #     ⇒ crop -y pixels off top and/or bottom
        #                    .----.
        #         .----.     |####|
        #       .-:----:-. .-:----:-.
        #       | |####| | | |####| |
        #       | |####| | | |####| |
        #       | |####| | | |####| |
        #       '-:----:-' '-'----'-'
        #         '----'
        #        y = -d/2    y = -d
        #
        #   - input image is wider than output:
        #       x < 0
        #       y == 0
        #     ⇒ crop -x pixels off left and/or right
        #       .--------.      .--------.
        #     .-|--------|-. .--|--------|
        #     |#|########|#| |##|########|
        #     '-|--------|-' '--|--------|
        #       '--------'      '--------'
        #        x = -d/2        x = -d
        image = image.crop(
            (
                # Left
                -out_pos.x,
                # Top
                -out_pos.y,
            )
            + tuple(dimensions)
        )

    # Final check, ensure the image will fit!
    if (image.width > dimensions.width) or (image.height > dimensions.height):
        # Force crop!
        image = image.crop(
            (
                0,
                0,
            )
            + tuple(dimensions)
        )

    return image


class Rasteriser(object):
    """
    This is a base class for a SVG rasteriser.  It takes a SVG file name
    and the name of a PNG file to emit, along with dimensions, and generates
    the resulting file.
    """

    async def render(self, inputsvg, outputpng, dimensions):
        raise NotImplementedError("Implement in %s" % self.__class__.__name__)


class SubprocessRasteriser(Rasteriser):
    """
    Sub-class that calls an external program.
    """

    def __init__(
        self,
        program,
        args=None,
        env=None,
        shell=False,
        inherit_env=True,
        loop=None,
        log=None,
    ):
        super().__init__()

        loop = defaults.get_loop(loop)
        log = defaults.get_logger(log, self.__class__.__module__)

        self._subproc = OneShotExternalProcess(
            proc_path=program,
            proc_args=args,
            proc_env=env,
            shell=shell,
            inherit_env=env,
            loop=loop,
            log=log.getChild("subproc"),
        )

        self._loop = loop
        self._log = log

    async def render(self, inputsvg, outputpng, dimensions):
        await self._subproc.run(
            extra_args=self._get_arguments(inputsvg, outputpng, dimensions),
            extra_env=self._get_environment(inputsvg, outputpng, dimensions),
        )

    def _get_arguments(self, inputsvg, outputpng, dimensions):
        """
        Return any command line arguments that must be given for the given
        input, output and dimensions.
        """
        raise NotImplementedError("Implement in %s" % self.__class__.__name__)

    def _get_environment(self, inputsvg, outputpng, dimensions):
        """
        Return any environment variables that must be set for the given
        input, output and dimensions.
        """
        return None


@_REGISTRY.register
class InkscapeRasteriser(Rasteriser):
    """
    Inkscape used as a command-line rasteriser.
    """

    ALIASES = ("inkscape",)

    def __init__(
        self,
        program="inkscape",
        args=None,
        env=None,
        shell=False,
        inherit_env=True,
        loop=None,
        log=None,
    ):
        super().__init__(
            self,
            program=program,
            args=args,
            env=env,
            shell=shell,
            inherit_env=inherit_env,
            loop=loop,
            log=log,
        )

    async def render(self, inputsvg, outputpng, dimensions):
        await self._subproc.run(
            extra_args=self._get_arguments(inputsvg, outputpng, dimensions),
            extra_env=self._get_environment(inputsvg, outputpng, dimensions),
        )

    def _get_arguments(self, inputsvg, outputpng, dimensions):
        return [
            "--export-width",
            dimensions.width,
            "--export-height",
            dimensions.height,
            "--export-filename",
            outputpng,
            inputsvg,
        ]
