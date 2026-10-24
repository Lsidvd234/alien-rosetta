"""Writing systems: the same sounds written four different ways.

- latin          plain ASCII letters (baseline)
- latin_accent   Latin with precomposed accents (stress, long vowels, retroflex dots)
- abugida_tamil  real Tamil-block letters (consonant + vowel sign, virama for bare consonants)
- abugida_pua    same abugida structure, but invented glyphs from the Unicode
                 Private Use Area, which no tokenizer has ever learned

Changing the script never changes the language: only how it is written.
"""

from __future__ import annotations

from .lexicon import LONG_VOWELS, PHONOLOGIES, SHORT_VOWELS, is_vowel

SCRIPTS = ("latin", "latin_accent", "abugida_tamil", "abugida_pua")

_LATIN = {"T": "tx", "N": "nx", "L": "lx", "R": "rx", "zh": "zh", "ng": "ng", "ny": "ny", "c": "c"}
_ACCENT_CONS = {"T": "ṭ", "N": "ṇ", "L": "ḷ", "R": "ṟ", "zh": "ḻ", "ng": "ṅ", "ny": "ñ"}
_ACUTE = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú"}
_MACRON = {"a": "ā", "e": "ē", "i": "ī", "o": "ō", "u": "ū"}

# Tamil block. Preferred letter per sound; sounds Tamil lacks get leftover letters.
_TA_PREF = {"k": "க", "ng": "ங", "c": "ச", "ny": "ஞ", "T": "ட", "N": "ண", "t": "த", "n": "ந",
            "p": "ப", "m": "ம", "y": "ய", "r": "ர", "l": "ல", "v": "வ", "zh": "ழ", "L": "ள",
            "R": "ற"}
_TA_POOL = ["க", "ங", "ச", "ஞ", "ட", "ண", "த", "ந", "ப", "ம", "ய", "ர", "ல", "வ", "ழ", "ள",
            "ற", "ன", "ஜ", "ஷ", "ஸ", "ஹ"]
_TA_IND = {"a": "அ", "aa": "ஆ", "i": "இ", "ii": "ஈ", "u": "உ", "uu": "ஊ",
           "e": "எ", "ee": "ஏ", "o": "ஒ", "oo": "ஓ"}
_TA_SIGN = {"a": "", "aa": "ா", "i": "ி", "ii": "ீ", "u": "ு", "uu": "ூ",
            "e": "ெ", "ee": "ே", "o": "ொ", "oo": "ோ"}
_TA_VIRAMA = "்"

_VOWEL_ORDER = list(SHORT_VOWELS + LONG_VOWELS)


def _inventory(phonology: str) -> list[str]:
    # glides y/v can appear via sound changes even in the easy language
    cons = list(PHONOLOGIES[phonology].consonants)
    return cons + [g for g in ("y", "v") if g not in cons]


class Script:
    def __init__(self, name: str, phonology: str):
        if name not in SCRIPTS:
            raise ValueError(f"unknown script {name}; choose from {SCRIPTS}")
        self.name = name
        cons = _inventory(phonology)
        if name == "abugida_tamil":
            mapping, free = {}, [ch for ch in _TA_POOL if ch not in {_TA_PREF[c] for c in cons if c in _TA_PREF}]
            for c in cons:
                mapping[c] = _TA_PREF[c] if c in _TA_PREF else free.pop(0)
            self.cons, self.ind, self.sign, self.virama = mapping, _TA_IND, _TA_SIGN, _TA_VIRAMA
        elif name == "abugida_pua":
            self.cons = {c: chr(0xE000 + i) for i, c in enumerate(cons)}
            self.ind = {v: chr(0xE040 + i) for i, v in enumerate(_VOWEL_ORDER)}
            self.sign = {v: ("" if v == "a" else chr(0xE060 + i)) for i, v in enumerate(_VOWEL_ORDER)}
            self.virama = chr(0xE07F)

    def latin_of_glyphs(self) -> dict[str, str]:
        """For an abugida: each character -> the plain-Latin spelling of the sound it writes
        (consonant letter, vowel letter(s)); the virama (no sound) maps to ""."""
        out = {g: _LATIN.get(c, c) for c, g in self.cons.items()}
        out.update({g: v for v, g in self.ind.items()})
        out.update({g: v for v, g in self.sign.items() if g})
        out[self.virama] = ""
        return out

    def word(self, ph: tuple[str, ...]) -> str:
        if self.name == "latin":
            return "".join(_LATIN.get(p, p) for p in ph)
        if self.name == "latin_accent":
            return self._accent(ph)
        return self._abugida(ph)

    def _accent(self, ph: tuple[str, ...]) -> str:
        out, stressed = [], False
        for p in ph:
            if is_vowel(p):
                if len(p) == 2:
                    out.append(_MACRON[p[0]])          # long vowel
                elif not stressed:
                    out.append(_ACUTE[p])              # stress on the first vowel
                else:
                    out.append(p)
                stressed = True
            else:
                out.append(_ACCENT_CONS.get(p, p))
        return "".join(out)

    def _abugida(self, ph: tuple[str, ...]) -> str:
        out, i = [], 0
        while i < len(ph):
            p = ph[i]
            if is_vowel(p):
                out.append(self.ind[p]); i += 1
            elif i + 1 < len(ph) and is_vowel(ph[i + 1]):
                out.append(self.cons[p] + self.sign[ph[i + 1]]); i += 2
            else:
                out.append(self.cons[p] + self.virama); i += 1
        return "".join(out)


def render_sentence(words, script: Script) -> str:
    return " ".join(script.word(w.phonemes) for w in words)
