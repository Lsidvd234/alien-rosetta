"""Meanings (sentence frames), the vocabulary, and English realization.

A frame is the language-independent meaning of one sentence. The English
side is rendered with real English grammar; the invented language renders
the same frame with its own rules (see grammar.py). The frame is the exact
answer key for every example.

Vocabulary: `Vocab.large()` loads conlang/data/vocab_en.json (2000+ words,
built by scripts/build_vocab.py from WordNet + wordfreq + lemminflect).
`Vocab.small()` is a tiny hand-written set used for fast tests.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

VOCAB_FILE = Path(__file__).resolve().parent / "data" / "vocab_en.json"

# key -> (English subject form, person, number, gender)
PRONOUNS = {
    "1sg": ("I", 1, "sg", None), "2sg": ("you", 2, "sg", None),
    "3sgm": ("he", 3, "sg", "m"), "3sgf": ("she", 3, "sg", "f"), "3sgn": ("it", 3, "sg", "n"),
    "1pl": ("we", 1, "pl", None), "2pl": ("you all", 2, "pl", None), "3pl": ("they", 3, "pl", None),
}
TENSES = ("present", "past", "future")

_SMALL = {
    "nouns": [
        {"lemma": l, "plural": p, "gender": g} for l, p, g in [
            ("man", "men", "m"), ("woman", "women", "f"), ("boy", "boys", "m"), ("girl", "girls", "f"),
            ("teacher", "teachers", "h"), ("child", "children", "h"),
            ("dog", "dogs", "n"), ("cat", "cats", "n"), ("horse", "horses", "n"), ("bird", "birds", "n"),
            ("fish", "fish", "n"), ("cow", "cows", "n"), ("book", "books", "n"), ("stone", "stones", "n"),
            ("cup", "cups", "n"), ("ball", "balls", "n"), ("box", "boxes", "n"), ("boat", "boats", "n"),
            ("apple", "apples", "n"), ("tree", "trees", "n")]],
    "locations": ["house", "river", "forest", "market", "garden", "field", "school", "village"],
    "transitive": [
        {"base": b, "third": t, "past": p} for b, t, p in [
            ("see", "sees", "saw"), ("like", "likes", "liked"), ("hold", "holds", "held"),
            ("carry", "carries", "carried"), ("find", "finds", "found"), ("help", "helps", "helped"),
            ("watch", "watches", "watched"), ("push", "pushes", "pushed"), ("take", "takes", "took"),
            ("catch", "catches", "caught"), ("build", "builds", "built"), ("bring", "brings", "brought")]],
    "intransitive": [
        {"base": b, "third": t, "past": p} for b, t, p in [
            ("sleep", "sleeps", "slept"), ("run", "runs", "ran"), ("walk", "walks", "walked"),
            ("sing", "sings", "sang"), ("swim", "swims", "swam"), ("wait", "waits", "waited")]],
    "adjectives": ["big", "small", "old", "young", "red", "black", "white", "happy", "tall", "quick"],
}


class Vocab:
    def __init__(self, d: dict, name: str):
        self.name = name
        self.nouns = [n["lemma"] for n in d["nouns"]]
        self.noun_info = {n["lemma"]: (n["plural"], n["gender"]) for n in d["nouns"]}
        self.locations = list(d["locations"])
        self.transitive = [v["base"] for v in d["transitive"]]
        self.intransitive = [v["base"] for v in d["intransitive"]]
        self.verb_info = {v["base"]: (v["base"], v["third"], v["past"])
                          for v in d["transitive"] + d["intransitive"]}
        self.adjectives = list(d["adjectives"])

    @classmethod
    def small(cls) -> "Vocab":
        return cls(_SMALL, "small")

    @classmethod
    @lru_cache(maxsize=1)
    def large(cls) -> "Vocab":
        return cls(json.loads(VOCAB_FILE.read_text(encoding="utf-8")), "large")

    def concepts(self) -> list[str]:
        """Every content concept, keyed by role so 'watch' (noun) and 'watch' (verb) differ."""
        return ([f"N:{w}" for w in self.nouns] + [f"L:{w}" for w in self.locations]
                + [f"V:{w}" for w in self.transitive + self.intransitive]
                + [f"A:{w}" for w in self.adjectives] + [f"PRON.{k}" for k in PRONOUNS])

    def size(self) -> int:
        return len(self.concepts()) - len(PRONOUNS)


@dataclass(frozen=True)
class NP:
    noun: str
    plural: bool
    adj: str | None = None


@dataclass(frozen=True)
class Frame:
    subj: NP | str            # NP, or a pronoun key like "3sgf"
    verb: str
    obj: NP | None            # None for intransitive verbs
    loc: str | None           # location noun, or None
    tense: str
    neg: bool

    def subject_png(self, vocab: Vocab) -> tuple[int, str, str | None]:
        """Person, number, gender of the subject."""
        if isinstance(self.subj, str):
            _, person, number, gender = PRONOUNS[self.subj]
            return person, number, gender
        number = "pl" if self.subj.plural else "sg"
        gender = vocab.noun_info[self.subj.noun][1] if number == "sg" else None
        return 3, number, gender

    def content_words(self) -> list[str]:
        """Role-keyed content concepts used in this frame (for coverage checks)."""
        out = []
        for np in (self.subj, self.obj):
            if isinstance(np, NP):
                out.append(f"N:{np.noun}")
                if np.adj:
                    out += [f"A:{a}" for a in np.adj.split()]   # several adjectives are space-separated
        out.append(f"V:{self.verb}")
        if self.loc:
            out.append(f"L:{self.loc}")
        return out


def sample_frame(rng: random.Random, vocab: Vocab, long: bool = False) -> Frame:
    """Random meaning. `long=True` forces the longest sentence type
    (adjectives on both noun phrases + a location)."""
    def np(force_adj: bool = False) -> NP:
        adj = rng.choice(vocab.adjectives) if (force_adj or rng.random() < 0.35) else None
        return NP(rng.choice(vocab.nouns), rng.random() < 0.4, adj)

    if long:
        return Frame(np(True), rng.choice(vocab.transitive), np(True), rng.choice(vocab.locations),
                     rng.choice(TENSES), rng.random() < 0.3)
    subj: NP | str = rng.choice(list(PRONOUNS)) if rng.random() < 0.3 else np()
    if rng.random() < 0.7:
        verb, obj = rng.choice(vocab.transitive), np()
    else:
        verb, obj = rng.choice(vocab.intransitive), None
    loc = rng.choice(vocab.locations) if rng.random() < 0.3 else None
    return Frame(subj, verb, obj, loc, rng.choice(TENSES), rng.random() < 0.3)


# --- English ------------------------------------------------------------

def _english_np(np: NP, vocab: Vocab) -> list[str]:
    noun = vocab.noun_info[np.noun][0] if np.plural else np.noun
    return ["the"] + ([np.adj] if np.adj else []) + [noun]


def english(frame: Frame, vocab: Vocab) -> str:
    person, number, _ = frame.subject_png(vocab)
    base, third, past = vocab.verb_info[frame.verb]
    third_sg = person == 3 and number == "sg"

    words = [PRONOUNS[frame.subj][0]] if isinstance(frame.subj, str) else _english_np(frame.subj, vocab)

    if frame.tense == "future":
        words += ["will", "not", base] if frame.neg else ["will", base]
    elif frame.tense == "past":
        words += ["did", "not", base] if frame.neg else [past]
    else:  # present
        if frame.neg:
            words += ["does" if third_sg else "do", "not", base]
        else:
            words += [third if third_sg else base]

    if frame.obj is not None:
        words += _english_np(frame.obj, vocab)
    if frame.loc is not None:
        words += ["in", "the", frame.loc]
    return " ".join(words)
