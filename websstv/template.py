#!/usr/bin/env python3

"""
Crude templating engine for Inkscape-generated SVG files.

Usage:
    In your SVG file, use Inkscape's Selectors and CSS palette to
    create "data" classes in your SVG file.  In these CSS classes, you define
    special CSS properties to define how that CSS class is used.

    "Data classes" exist purely to declare some field value, at the moment the
    only real use case for this, is date/time fields timestamped at the
    current time.

    All properties here begin with whatever ``CSS_PROP_PREFIX`` is set to.
    Their values are given as JSON.

    More detail in the classes below, e.g. see ``SVGTemplateField`` and its
    sub-classes.  Once you have created the CSS classes, apply them as
    required to the objects you wish to change.

    Save the SVG file (standard Inkscape SVG is fine).

    In your code, use ``SVGTemplate.from_file("/path/to/template.svg")`` to
    load the template, then use the ``get_instance()`` method to create an
    instance of the template (you can call this as many times as you like).

    You can provide default values for the template fields in this call if you
    wish.  On the instance, use ``set_values()`` to update template fields,
    then finally call ``apply()`` to copy these into the template instance.

    Finally, use ``write()`` to dump the output SVG image.
"""

# © Stuart Longland VK4MSL
# SPDX-License-Identifier: GPL-2.0-or-later

from xml.etree import ElementTree
from collections.abc import Mapping
import argparse
import json
import os
import os.path
import css_parser
import datetime
import zoneinfo
import enum
import copy

from .defaults import get_logger

# Prefix used on CSS properties.  All other properties are ignored.
CSS_PROP_PREFIX = "-websstv-template-"


class SVGTemplateFieldType(enum.Enum):
    INTEGER = "int"
    UINTEGER = "uint"
    DATETIME = "datetime"
    DOUBLE = "double"
    STRING = "string"
    BOOLEAN = "boolean"
    URI = "uri"


class SVGTemplateField(object):
    """
    Base class for SVG template fields.  This class may be used to represent
    one of two possible types of field:

    - Data fields… which are used to fill in missing values in instances of a
      template.
    - Element fields… which fill in either tag text or attribute values.

    The CSS class defines the name of the field, while the CSS properties
    (that is, those prefixed with CSS_PROP_PREFIX, all others are ignored)
    define the specifics of this particular field.

    All field types have the following properties:

    - ``type``: the data type for the field, see SVGTemplateFieldType for
      possible values.  This chooses a sub-class of SVGTemplateField.
    - ``desc``: human-readable description of the field
    - ``required``: Boolean flag to indicate whether the field is mandatory or
      not.
    - ``path``: XPath expression describing what element beneath the tagged
      element is to be altered.
    - ``leaves``: Boolean flag indicating whether we should try to seek out
      the leaf nodes beneath the selected tag.  This is mainly useful for
      ``text`` SVG tags which often have one or two ``tspan`` tags that
      actually contain the text we want to change.
    - ``namespace``: If setting an attribute, this sets the attribute
      namespace.  Otherwise, a namespace will be chosen: if an attribute
      of that name exists, the first one seen will be updated, or if the
      attribute is new, the SVG namespace is used.
    - ``attribute``: If set, this defines what attribute we are changing.
      Otherwise we modify the text content for the tag.
    - ``data``: Boolean flag indicating this is a data field, if set to
      true, no DOM manipulation is performed, this defines a field's value
      in the template.
    - ``value``: Default value to assign to the field, if no value is given
      from other sources.
    """

    _FIELD_TYPES = {}

    @classmethod
    def from_properties(cls, template, name, properties, log):
        ftype = SVGTemplateFieldType(properties.pop("type"))
        fcls = cls._FIELD_TYPES[ftype]
        log.debug(
            "Initialising field %r with properties %r", name, properties
        )
        return fcls(template=template, name=name, log=log, **properties)

    @classmethod
    def _register_field_type(cls, fieldtype):
        assert fieldtype.TYPE not in cls._FIELD_TYPES, (
            "Duplicate field type %s" % fieldtype.TYPE
        )
        cls._FIELD_TYPES[fieldtype.TYPE] = fieldtype
        return fieldtype

    def __init__(
        self,
        template,
        name,
        desc,
        required=False,
        path=None,
        leaves=True,
        namespace=None,
        attribute=None,
        data=False,
        value=None,
        log=None,
        **kwargs
    ):
        self._log = get_logger(log, self.__class__.__module__)
        self._name = name
        self._template = template
        self._path = path
        self._leaves = leaves
        self._desc = desc
        self._required = required
        self._namespace = namespace
        self._attribute = attribute
        self._data = data
        self._value = value

    @property
    def spec(self):
        return {
            "type": self.TYPE.name,
            "desc": self.desc,
            "required": self.required,
        }

    @property
    def name(self):
        return self._name

    @property
    def desc(self):
        return self._desc

    @property
    def required(self):
        return self._required

    @property
    def namespace(self):
        return self._namespace

    @property
    def attribute(self):
        return self._attribute

    @property
    def path(self):
        return self._path

    @property
    def leaves(self):
        return self._leaves

    @property
    def data(self):
        return self._data

    def get_rawvalue(self, instancevalues):
        return instancevalues.get(self.name, self._value)

    def get_value(self, instancevalues):
        return self.get_rawvalue(instancevalues)

    def apply(self, value, element):
        """
        Apply the template parameter to the XML element.
        """
        if self.attribute is None:
            # Apply it to the element body text
            element.text = value
        else:
            # Do we have a namespace for this?
            ns = self.namespace
            attribute = None

            if ns is None:
                # No, look for an attribute with this name
                for attr in element.attrib.keys():
                    if attr.endswith("}%s" % self.attribute):
                        # This is probably it
                        attribute = attr
                        break

                if attribute is None:
                    # We still didn't find a match, assume the SVG namespace
                    ns = self._xmlns["svg"]
            elif ns in self._xmlns:
                # De-reference namespace
                ns = self._xmlns[ns]

            if attribute is None:
                # Combine namespace and attribute
                attribute = "{%s}%s" % (ns, self.attribute)

            # Finally apply the attribute value
            element.attrib[attribute] = value


@SVGTemplateField._register_field_type
class SVGStringTemplateField(SVGTemplateField):
    """
    String template field.  This is used to capture and store string data in a
    template.  It takes one additional property:

    - ``format``: Python ``%`` operator format string, defining a "format" for
      the string.  The values used are taken from the template instance field
      values.  See
      https://docs.python.org/3/library/stdtypes.html#printf-style-string-formatting
    """

    TYPE = SVGTemplateFieldType.STRING

    def __init__(self, template, desc, required=False, format=None, **kwargs):
        super().__init__(
            template=template, desc=desc, required=required, **kwargs
        )
        self._format = format

    @property
    def spec(self):
        spec = super().spec
        spec.update({"format": self.format})
        return spec

    @property
    def format(self):
        return self._format

    def get_value(self, instancevalues):
        if self.format is not None:
            return self.format % instancevalues
        else:
            return super().get_value(instancevalues)


@SVGTemplateField._register_field_type
class SVGURITemplateField(SVGStringTemplateField):
    """
    URI template field.  This behaves like a string field, but if the
    resulting path is a filesystem path, the real path will be found and that
    will be prefixed with ``file://`` to make it a valid URI.

    There are some additional parameters, useful for UIs but not used by the
    templating engine itself:
    - ``source_dir``: If set, defines a local directory where files suitable
      for this placeholder may be found.  This allows a UI to grab a list of
      possible options and display them for the user to choose from.
    - ``source_subdir``: If set to ``True``, we consider files in
      subdirectories of the given directory, not just top-level options.
    - ``source_extn``: If set, this is a set of permitted extensions we look
      for (e.g. to filter out non-image files).
    """

    TYPE = SVGTemplateFieldType.URI

    def __init__(
        self,
        template,
        desc,
        required=False,
        format=None,
        source_dir=None,
        source_subdir=False,
        source_extn=None,
        **kwargs
    ):
        super().__init__(
            template=template, desc=desc, required=required, **kwargs
        )

        self._log.debug(
            "Source directory: %r, Base: %r, Absolute: %r",
            source_dir,
            template.base_dir,
            source_dir and os.path.isabs(source_dir),
        )

        if (
            (source_dir is not None)
            and (template.base_dir is not None)
            and not os.path.isabs(source_dir)
        ):
            source_dir = os.path.realpath(
                os.path.join(template.base_dir, source_dir)
            )
            self._log.debug("Recording source as %r", source_dir)

        self._format = format
        self._source_dir = source_dir
        self._source_subdir = bool(source_subdir)

        if source_extn is not None:
            source_extn = set(source_extn.split(" "))
            self._log.debug("Expected extensions: %r", source_extn)

            def _is_extn(path):
                (_, extn) = os.path.splitext(path)
                return extn in source_extn

            self._is_extn = _is_extn
        else:
            self._is_extn = lambda path: True
            self._log.debug("Expected extensions: any")

    @property
    def spec(self):
        spec = super().spec
        spec.update({"options": [p for (_, p) in self.options]})
        return spec

    @property
    def source_dir(self):
        return self._source_dir

    @property
    def source_subdir(self):
        return self._source_subdir

    @property
    def source_extn(self):
        return self._source_extn

    @property
    def options(self):
        """
        Return the suggested options for this field.  This does a look-up of
        the path and enumerates possible options if not already done.  This
        will yield (as a generator) tuples of the form ``(label, fullpath)``.
        """
        if self.source_dir is None:
            # No source directory, nothing to do!
            self._log.debug("No directory set")
            return

        seen = set()
        for path in self._list_contents(self.source_dir, seen):
            yield (path, os.path.relpath(path, self.source_dir))

    def get_value(self, instancevalues):
        value = super().get_value(instancevalues)
        if value is None:
            return None

        scheme = value.split(":", 1)[0]

        if scheme in ("http", "https", "ftp", "data", "file"):
            # Make no change
            return value

        # We have a possibly bare file name.  Make it a URI
        # TODO: don't know how to do this for Windows UNC paths.
        # Maybe they can use MMSSTV instead?
        return "file://%s" % os.path.realpath(value)

    def _list_contents(self, path, seen):
        self._log.debug("Listing contents of %r", path)
        for name in os.listdir(path):
            fullpath = os.path.join(path, name)
            rpath = os.path.realpath(fullpath)
            if rpath in seen:
                continue
            else:
                seen.add(rpath)

            if os.path.isdir(fullpath):
                if self.source_subdir:
                    yield from self._list_contents(fullpath, seen)
            else:
                if self._is_extn(name):
                    yield fullpath


@SVGTemplateField._register_field_type
class SVGDateTimeTemplateField(SVGStringTemplateField):
    """
    Date/time field.  This takes an input that is ISO-8601 format, or the
    current time, and formats it according to the format string you provide.

    IANA time-zone support is provided.

    - ``format``: a date/time format string, according to ``strftime()`` on
      the host operating system.
    - ``tz``: IANA time-zone for the date/time
    - ``now``: Boolean flag, if set to ``true``, then any input value is
      ignored (and optional), and instead the current system time in the
      nominated time-zone is substituted.
    """

    TYPE = SVGTemplateFieldType.DATETIME

    def __init__(
        self,
        template,
        desc,
        required=False,
        format=None,
        tz=None,
        now=False,
        **kwargs
    ):
        super().__init__(
            template=template, desc=desc, required=required, **kwargs
        )

        if (tz is not None) and isinstance(tz, str):
            tz = zoneinfo.ZoneInfo(tz)

        self._format = format
        self._tz = tz
        self._now = now

    @property
    def spec(self):
        spec = super().spec
        spec.update(
            {
                "format": self.format,
                "now": self.now,
                "tz": str(self.tz) if self.tz else None,
            }
        )
        return spec

    @property
    def format(self):
        return self._format

    @property
    def now(self):
        return self._now

    @property
    def tz(self):
        return self._tz

    def get_rawvalue(self, instancevalues):
        if self.now:
            return datetime.datetime.now(tz=self.tz)
        else:
            v = super().get_rawvalue(instancevalues)
            if isinstance(v, str):
                v = datetime.datetime.fromisoformat(v)
            if self.tz is not None:
                v = v.astimezone(self.tz)
            return v

    def get_value(self, instancevalues):
        value = self.get_rawvalue(instancevalues)

        if self.format is not None:
            return value.strftime(self.format)
        else:
            return value.isoformat()


@SVGTemplateField._register_field_type
class SVGIntTemplateField(SVGTemplateField):
    """
    Signed Integer field.

    - ``digits``: the number of digits to pad the field out to
    - ``leadzero``: boolean flag, if true, pad with leading zeroes instead of
      spaces.
    """

    TYPE = SVGTemplateFieldType.INTEGER

    def __init__(
        self,
        template,
        desc,
        required=False,
        digits=None,
        leadzero=False,
        **kwargs
    ):
        super().__init__(
            template=template, desc=desc, required=required, **kwargs
        )

        self._formatstr = "%"

        if self._leadzero:
            self._formatstr += "0"

        if self._digits is not None:
            self._formatstr += "%d" % self._digits

        self._formatstr += "d"

    def get_value(self, instancevalues):
        return self._formatstr % self.get_rawvalue(instancevalues)


@SVGTemplateField._register_field_type
class SVGUIntTemplateField(SVGIntTemplateField):
    """
    Unsigned integer field, otherwise behaves like a signed field.
    """

    TYPE = SVGTemplateFieldType.UINTEGER


@SVGTemplateField._register_field_type
class SVGDoubleTemplateField(SVGIntTemplateField):
    """
    Double precision floating-point field.

    - ``digits``: the number of digits to pad the field out to
    - ``leadzero``: boolean flag, if true, pad with leading zeroes instead of
      spaces.
    - ``precision``: the number of decimal places to format
    - ``useint``: boolean flag, if the raw value _can_ be represented as an
      integer, do so (do not emit a decimal point).
    """

    TYPE = SVGTemplateFieldType.DOUBLE

    def __init__(
        self,
        template,
        desc,
        required=False,
        digits=None,
        precision=None,
        leadzero=False,
        useint=True,
        **kwargs
    ):
        super().__init__(
            template=template,
            desc=desc,
            required=required,
            digits=digits,
            leadzero=leadzero,
            **kwargs
        )

        self._useint = useint
        self._iformatstr = self._formatstr
        self._formatstr = "%"
        if self._leadzero:
            self._formatstr += "0"

        if self._digits is not None:
            self._formatstr += "%d" % self._digits

        if self._precision is not None:
            self._formatstr += ".%d" % self._precision

        self._formatstr += "f"

    def get_value(self, instancevalues):
        value = self.get_rawvalue(instancevalues)
        if self._useint:
            ivalue = int(value)
            if value == ivalue:
                return self._iformatstr % ivalue

        return self._formatstr % value


@SVGTemplateField._register_field_type
class SVGBoolTemplateField(SVGTemplateField):
    """
    Boolean field.  This basically allows for custom strings for the "true"
    and "false" values.

    - ``true-text``: The text to emit if the value is true
    - ``false-text``: The text to emit if the value is false
    """

    TYPE = SVGTemplateFieldType.BOOLEAN

    def __init__(
        self,
        template,
        desc,
        required=False,
        true_text="true",
        false_text="false",
        **kwargs
    ):
        super().__init__(
            template=template, desc=desc, required=required, **kwargs
        )

        self._true = true_text
        self._false = false_text

    def get_value(self, instancevalues):
        if self.get_rawvalue(instancevalues) is True:
            return self._true
        else:
            return self._false


class SVGTemplateDirectory(Mapping):
    """
    A collection of templates, all loaded from a filesystem path.
    """

    def __init__(self, dirname, subdirs=True, defaults=None, log=None):
        self._dirname = os.path.realpath(dirname)
        self._subdirs = subdirs
        self._templates = None
        self._defaults = defaults
        self._log = get_logger(log, self.__class__.__module__)

    @property
    def dirname(self):
        return self._dirname

    @property
    def defaults(self):
        return self._defaults

    @defaults.setter
    def defaults(self, defaults):
        self._defaults = defaults
        if self._templates is not None:
            for template in self._templates.values():
                template.defaults = defaults

    def __getitem__(self, name):
        if self._templates is None:
            self.reload()

        if self._templates is None:
            # Template directory nonexistant?
            raise KeyError(name)

        try:
            return self._templates[name]
        except KeyError:
            # Try a reload in case the file is new
            self.reload()

        return self._templates[name]

    def __iter__(self):
        if self._templates is None:
            self.reload()

        return iter(self._templates or {})

    def __len__(self):
        if self._templates is None:
            self.reload()

        return len(self._templates or {})

    def reload(self):
        """
        Re-scan the directories and re-load the templates.
        """
        if not os.path.isdir(self._dirname):
            # No directory, do nothing
            return

        seen = set()
        self._templates = dict(self._enumerate_dir(self._dirname, seen))

    def _enumerate_dir(self, dirname, seen):
        self._log.debug("Inspecting files in %r", dirname)
        existing = self._templates or {}

        for name in os.listdir(dirname):
            path = os.path.join(dirname, name)

            # Avoid symlink loops!
            rpath = os.path.realpath(path)
            if rpath in seen:
                self._log.debug("%r has been seen already", path)
                continue
            else:
                seen.add(rpath)

            # Recurse subdirectories
            if os.path.isdir(path):
                if not self._subdirs:
                    continue
                yield from self._enumerate_dir(path, seen)
            elif os.path.isfile(path):
                (name, ext) = os.path.splitext(
                    os.path.relpath(path, self._dirname)
                )
                if ext != ".svg":
                    # Ignore non-SVG files
                    continue

                mtime = os.stat(rpath).st_mtime
                template = existing.get(name)

                if (template is not None) and (template.mtime == mtime):
                    # No need to reload
                    yield (name, template)
                else:
                    # Replace
                    try:
                        yield (
                            name,
                            SVGTemplate.from_file(
                                filename=rpath,
                                base_dir=dirname,
                                mtime=mtime,
                                defaults=self._defaults,
                                log=self._log.getChild(name),
                            ),
                        )
                        self._log.debug("Loaded %r as %r", rpath, name)
                    except:
                        self._log.exception(
                            "Failed to load template file %r", rpath
                        )


class SVGTemplate(object):
    """
    SVG template class.  This class loads in a SVG file and parses out the
    template fields so that instances of it can be created.
    """

    @classmethod
    def from_file(
        cls, filename, defaults=None, base_dir=None, mtime=None, log=None
    ):
        if base_dir is None:
            base_dir = os.path.realpath(os.path.dirname(filename))

        if mtime is None:
            mtime = os.stat(filename).st_mtime

        return cls(
            ElementTree.parse(filename),
            base_dir=base_dir,
            mtime=mtime,
            defaults=defaults,
            log=log,
        )

    def __init__(
        self,
        svgdoc,
        css_prop_prefix=CSS_PROP_PREFIX,
        base_dir=None,
        mtime=None,
        defaults=None,
        log=None,
    ):
        self._mtime = mtime
        self._doc = svgdoc
        self._root = svgdoc.getroot()
        self._base_dir = base_dir
        self._defaults = defaults
        self._log = get_logger(log, self.__class__.__module__)

        if self._root.tag == "svg":
            self._xmlns = None
        else:
            assert self._root.tag.startswith("{") and self._root.tag.endswith(
                "}svg"
            ), "Root element is not a SVG tag"
            # Pick out the namespace URI
            self._xmlns = dict(svg=self._root.tag[1:-4])

        # Pick up all the style tags to identify classes
        self._log.debug("Parsing fields")
        self._domfields = {}
        self._datafields = {}
        for elem in svgdoc.iterfind(".//svg:style", namespaces=self._xmlns):
            css = css_parser.parseString(elem.text)
            for rule in css.cssRules:
                if not rule.selectorText.startswith("."):
                    continue

                fieldname = rule.selectorText[1:]
                properties = {}
                for prop in rule.style.getProperties():
                    if not prop.name.startswith(css_prop_prefix):
                        continue
                    p = prop.name[len(css_prop_prefix) :].replace("-", "_")
                    properties[p] = json.loads(prop.value)

                if properties:
                    field = SVGTemplateField.from_properties(
                        template=self,
                        name=fieldname,
                        properties=properties,
                        log=log.getChild(fieldname),
                    )
                    if field.data:
                        self._log.debug("Adding data field %r", fieldname)
                        self._datafields[fieldname] = field
                    else:
                        self._log.debug("Adding DOM field %r", fieldname)
                        self._domfields[fieldname] = field

    @property
    def defaults(self):
        return self._defaults

    @defaults.setter
    def defaults(self, defaults):
        self._defaults = defaults

    @property
    def mtime(self):
        """
        If the template was loaded from a file, this reports the modification
        time of the file at the time it was loaded.
        """
        return self._mtime

    @property
    def base_dir(self):
        """
        If the template was loaded from a file, this gives the path the file
        lives in for resolving relative URIs.
        """
        return self._base_dir

    @property
    def datafields(self):
        """
        Data fields for defining and declaring template field values.
        """
        return self._datafields.copy()

    @property
    def domfields(self):
        """
        DOM fields which manipulate properties and text values in the SVG DOM.
        """
        return self._domfields.copy()

    @property
    def fields(self):
        """
        All fields supported by the template.
        """
        return dict(
            list(self._datafields.items()) + list(self._domfields.items())
        )

    def get_instance(self, defaults=None):
        return SVGTemplateInstance(self, defaults=defaults)


class SVGTemplateInstance(object):
    def __init__(self, template, defaults=None):
        self._xmlns = template._xmlns
        self._doc = copy.deepcopy(template._doc)
        self._root = self._doc.getroot()
        self._datafields = template.datafields
        self._domfields = template.domfields
        self._values = copy.deepcopy(template.defaults)
        self._applied = False
        if template.defaults is not None:
            self._values.update(copy.deepcopy(template.defaults))
        if defaults is not None:
            self._values.update(defaults)

        # Figure out what the text and image tags will be called, they'll have
        # the same namespace as the `<svg>` tag.
        if self._xmlns:
            self._text_tag = "{%s}text" % self._xmlns["svg"]
            self._image_tag = "{%s}image" % self._xmlns["svg"]
        else:
            self._text_tag = "text"
            self._image_tag = "image"

        # Figure out all classes defined and the elements using them
        self._fields = []
        for elem in self._root.iterfind(
            ".//*[@class]", namespaces=self._xmlns
        ):
            elem_fields = []
            for cls in elem.attrib.get("class", "").split(" "):
                if not cls:
                    continue

                try:
                    elem_fields.append(self._domfields[cls])
                except KeyError:
                    # not a DOM field name class
                    continue

            if not elem_fields:
                continue

            self._fields.append(
                SVGTemplateInstanceField(self, elem, elem_fields)
            )

    def set_values(self, **values):
        if self._applied:
            raise ValueError("Values already applied")

        self._values.update(values)

    def apply(self):
        todo = list(self._datafields.values())
        while todo:
            changed = False
            failed = []
            for field in todo:
                try:
                    value = field.get_value(self._values)
                except KeyError:
                    # Can't compute this one yet!
                    failed.append(field)
                    continue

                self._values[field.name] = value
                changed = True

            if changed:
                todo = failed
            else:
                raise ValueError(
                    "Unable to compute values for %s"
                    ", ".join(f.name for f in failed)
                )

        for field in self._fields:
            field.apply()

        self._applied = True

    def write(self, path):
        if not self._applied:
            raise ValueError("Apply the values first")

        self._doc.write(path)


class SVGTemplateInstanceField(object):
    def __init__(self, instance, element, fields):
        self._instance = instance
        self._element = element
        self._fields = fields

    def apply(self):
        for field in self._fields:
            value = field.get_value(instance._values)

            if value is None:
                if field.required:
                    raise ValueError("Field %r is not set" % field.name)
                else:
                    value = ""

            elements = [self._element]

            if field.path:
                new_elements = []

                for e in elements:
                    new_elements.extend(e.findall(field.path))

                elements = new_elements

            if field.leaves:
                new_elements = []

                for e in elements:
                    new_elements.extend(self._leavesof(e))

                elements = new_elements

            for element in elements:
                field.apply(value, element)

    @staticmethod
    def _leavesof(elem):
        parent = False
        for child in elem:
            parent = True
            yield from SVGTemplateInstanceField._leavesof(child)

        if not parent:
            yield elem


if __name__ == "__main__":
    # Defaults
    defaults = {}

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--set-default",
        nargs=2,
        action="append",
        default=[],
        help="Set the default value for the named field",
    )
    ap.add_argument(
        "--set-value",
        nargs=2,
        action="append",
        default=[],
        help="Set the value for the named field",
    )
    ap.add_argument(
        "--default",
        nargs=1,
        help="Default value for unspecified template values",
    )
    ap.add_argument("template", help="Template SVG file")
    ap.add_argument("output", help="Output SVG file")
    ap.add_argument("inputs", nargs="*", help="Input JSON files")

    args = ap.parse_args()

    defaults.update((k, json.loads(v)) for k, v in args.set_default)
    values = defaults.copy()

    for jsonfile in args.inputs:
        print("Importing data from %r" % jsonfile)
        with open(jsonfile, "r") as f:
            values.update(json.loads(f.read()))

    print("Setting values from command line")
    values.update(dict((k, json.loads(v)) for k, v in args.set_value))

    print("Template input:\n%s" % json.dumps(values, indent=4))

    print("Importing template %r" % args.template)
    template = SVGTemplate.from_file(args.template)
    instance = template.get_instance(values)
    instance.apply()

    print("Writing out %r" % args.output)
    instance.write(args.output)
