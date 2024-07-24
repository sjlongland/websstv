#!/usr/bin/env python3

"""
CW ID generator
"""

# Yes, these are pronounced "dahs" and "dits", but we'll use - and . symbols
# to represent them in the international morse alphabet.

import enum

SYMBOLS = {
    # Letters
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    # Non-English
    "À": ".--.-",  # also Å
    "Ä": ".-.-",  # also Æ Ą
    "Ć": "-.-..",  # also Ĉ Ç
    "Ð": "..--.",
    "É": "..-..",  # also Ę
    "È": ".-..-",  # also Ł
    "Ĝ": "--.-.",
    "Ĥ": "----",  # also CH Š
    "Ĵ": ".---.",
    "Ń": "--.--",  # also Ñ
    "Ó": "---.",  # also Ö Ø
    "Ś": "...-...",
    "Ŝ": "...-.",
    "Þ": ".--..",
    "Ü": "..--",  # also Ŭ
    "Ź": "--..-.",
    "Ż": "--..-",
    # Digits
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    # Symbols
    ".": ".-.-.-",
    ",": "--..--",
    "?": "..--..",
    "'": ".----.",
    "!": "-.-.--",
    "/": "-..-.",
    "(": "-.--.",
    ")": "-.--.-",
    "&": ".-...",
    ":": "---...",
    "=": "-...-",
    "+": ".-.-.",
    "-": "-....-",
    "_": "..--.-",
    '"': ".-..-.",
    "$": "...-..-",
    "@": ".--.-.",
}

ALIASES = {
    "Å": "À",
    "Æ": "Ä",
    "Ą": "Ä",
    "Ĉ": "Ć",
    "Ç": "Ć",
    "Ę": "É",
    "Ł": "È",
    "CH": "Ĥ",
    "Š": "Ĥ",
    "Ñ": "Ń",
    "Ö": "Ó",
    "Ø": "Ó",
    "Ŭ": "Ü",
}

ALIASED_SYMBOLS = dict(
    list(SYMBOLS.items())
    + [(alias, SYMBOLS[sym]) for alias, sym in ALIASES.items()]
)


class Prosign(enum.Enum):
    END_OF_WORK = "...-.-"
    ERROR = "........"
    INVITATION = "-.-"
    START = "-.-.-"
    NEW_MESSAGE = ".-.-."
    VERIFIED = "...-."
    WAIT = ".-..."


class CWString(object):
    """
    CW String is a rough representation of a CW message encoding rough
    timing information with four symbols:

    - '.': dit
    - '-': dah (tone 3 "dits" long)
    - ' ': letter space (absence of tone three "dits" long)
    - '/': word space (absence of tone 7 "dits" long)
    """

    @classmethod
    def from_string(cls, s):
        """
        Encode a raw string of plain text to a CW string.
        """
        return cls(
            "/".join(
                " ".join(ALIASED_SYMBOLS[c] for c in w)
                for w in s.upper().split(" ")
            )
        )

    def __init__(self, cw):
        if not all(c in "-./ " for c in cw):
            raise ValueError(
                "CW strings may only consist of '-', '.', '/' and ' '."
            )
        self._cw = cw

    def __str__(self):
        return self._cw

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._cw)

    def __add__(self, other):
        """
        Implement concatenation with another entity.
        """
        if isinstance(other, Prosign):
            other = CWString(other.value)
        elif not isinstance(other, CWString):
            # Convert from string
            other = CWString.from_string(str(other))

        # Add a word separator if there isn't one there already
        if not (self._cw.endswith("/") or other._cw.endswith("/")):
            return CWString(self._cw + "/" + other._cw)
        else:
            return CWString(self._cw + other._cw)

    def modulate(self, oscillator, frequency, dit_period=0.120):
        """
        Modulate an oscillator with this CW string, at the given frequency
        and dit period timing.
        """
        # 0.120s is ~10 words per minute using the PARIS standard
        lastspace = True
        for sym in self._cw:
            if sym == ".":
                if not lastspace:
                    yield from oscillator.silence(dit_period)
                yield from oscillator.generate(frequency, dit_period)
                lastspace = False
            elif sym == "-":
                if not lastspace:
                    yield from oscillator.silence(dit_period)
                yield from oscillator.generate(frequency, 3 * dit_period)
                lastspace = False
            elif sym == " ":
                yield from oscillator.silence(3 * dit_period)
                lastspace = True
            elif sym == "/":
                yield from oscillator.silence(7 * dit_period)
                lastspace = True
