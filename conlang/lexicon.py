"""Sound inventories and seeded generation of roots and grammatical endings."""

from __future__ import annotations

import random
from dataclasses import dataclass

SHORT_VOWELS = ("a", "e", "i", "o", "u")
LONG_VOWELS = ("aa", "ee", "ii", "oo", "uu")
VOWELS = frozenset(SHORT_VOWELS + LONG_VOWELS)
STOPS = frozenset({"k", "c", "t", "p"})


def is_vowel(ph: str) -> bool:
    return ph in VOWELS


@dataclass(frozen=True)
class Phonology:
    name: str
    consonants: tuple[str, ...]
    codas: tuple[str, ...]           # consonants allowed at the end of a syllable
    long_vowel_rate: float           # chance a vowel is long (0 = no long vowels)
    root_syllables: tuple[int, ...]  # allowed root lengths in syllables
    coda_rate: float                 # chance a syllable ends in a consonant


EASY = Phonology(
    name="easy",
    consonants=tuple("p t k b d g m n s l r w h f v".split()),
    codas=tuple("n s t k m l r".split()),
    long_vowel_rate=0.0,
    root_syllables=(1, 2, 2, 2, 3),   # weighted: mostly 2 syllables, like common English words
    coda_rate=0.4,
)

# Tamil-like: retroflex series (T N L), alveolar R, zh, ng/ny, long vowels.
HARD = Phonology(
    name="hard",
    consonants=("k", "ng", "c", "ny", "T", "N", "t", "n", "p", "m",
                "y", "r", "l", "v", "zh", "L", "R"),
    codas=("m", "n", "N", "L", "l", "r", "y"),
    long_vowel_rate=0.3,
    root_syllables=(2, 3, 3),
    coda_rate=0.25,
)

PHONOLOGIES = {"easy": EASY, "hard": HARD}


class FormFactory:
    """Generates unique phoneme sequences (roots and endings) from a seed."""

    def __init__(self, phon: Phonology, rng: random.Random):
        self.phon = phon
        self.rng = rng
        self.used: set[tuple[str, ...]] = set()

    def _vowel(self) -> str:
        v = self.rng.choice(SHORT_VOWELS)
        if self.phon.long_vowel_rate and self.rng.random() < self.phon.long_vowel_rate:
            v = v + v
        return v

    def _syllable(self, final: bool) -> list[str]:
        syl = [self.rng.choice(self.phon.consonants), self._vowel()]
        if self.rng.random() < self.phon.coda_rate and (final or self.phon.name == "easy"):
            syl.append(self.rng.choice(self.phon.codas))
        return syl

    def _unique(self, make) -> tuple[str, ...]:
        for _ in range(1000):
            form = tuple(make())
            if form not in self.used:
                self.used.add(form)
                return form
        raise RuntimeError("could not generate a unique form; inventory too small")

    def root(self) -> tuple[str, ...]:
        def make():
            n = self.rng.choice(self.phon.root_syllables)
            out: list[str] = []
            for i in range(n):
                out += self._syllable(final=(i == n - 1))
            return out
        return self._unique(make)

    def ending(self) -> tuple[str, ...]:
        """Short grammatical ending. Hard languages get vowel-initial endings
        (like Tamil -ai, -il) so boundary sound changes actually fire."""
        def make():
            if self.phon.name == "easy":
                shape = self.rng.choice(["C", "VC", "CV"])
            else:
                shape = self.rng.choice(["V", "VC", "CV", "CVC", "VCV"])
            out: list[str] = []
            for i, ch in enumerate(shape):
                if ch == "C":
                    word_final = i == len(shape) - 1 and i > 0
                    pool = self.phon.codas if word_final else self.phon.consonants
                    out.append(self.rng.choice(pool))
                else:
                    out.append(self._vowel())
            return out
        return self._unique(make)
