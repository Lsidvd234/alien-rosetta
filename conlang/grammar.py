"""Grammar dials, the two presets, and rendering a frame into the invented language."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, replace

from .frames import NP, PRONOUNS, Frame, Vocab
from .lexicon import PHONOLOGIES, STOPS, FormFactory, is_vowel


@dataclass(frozen=True)
class GrammarConfig:
    word_order: str = "SVO"        # SVO | SOV | VSO
    adj_order: str = "AN"          # AN (adjective before noun) | NA
    articles: bool = True          # separate word for "the"
    acc_marking: str = "none"      # object marking: none | suffix | particle
    loc_marking: str = "prep"      # "in the X": prep (word before) | post (word after) | suffix
    plural: str = "suffix"         # none | suffix
    tense: str = "english"         # english (past ending, future helper word) | suffix (every tense an ending)
    agreement: str = "3sg"         # verb agrees with subject: none | 3sg (English-like) | full (Tamil-like)
    negation: str = "particle"     # particle (separate word) | suffix
    sandhi: bool = False           # sound changes where endings join
    phonology: str = "easy"        # easy | hard (long vowels, retroflex consonants, longer roots)
    irregular_rate: float = 0.15   # share of verbs with an irregular past stem

    def as_dict(self) -> dict:
        return asdict(self)


ENGLISH_LIKE = GrammarConfig()
TAMIL_LIKE = GrammarConfig(
    word_order="SOV", articles=False, acc_marking="suffix", loc_marking="suffix",
    tense="suffix", agreement="full", negation="suffix", sandhi=True, phonology="hard",
)

DIALS = [name for name in ENGLISH_LIKE.as_dict() if getattr(ENGLISH_LIKE, name) != getattr(TAMIL_LIKE, name)]


def dial_variants() -> dict[str, GrammarConfig]:
    """English-like base, plus one variant per dial flipped to its Tamil-like value."""
    out = {"base": ENGLISH_LIKE}
    for name in DIALS:
        out[f"flip_{name}"] = replace(ENGLISH_LIKE, **{name: getattr(TAMIL_LIKE, name)})
    out["all_flipped"] = TAMIL_LIKE
    return out


@dataclass
class Word:
    phonemes: tuple[str, ...]
    gloss: str


@dataclass
class Realization:
    words: list[Word]
    boundaries: int = 0          # morpheme joins inside words
    sandhi_changes: int = 0      # joins where a sound change fired
    morphemes: list[int] = field(default_factory=list)  # morphemes per word


AGREEMENT_CELLS = ["1sg", "2sg", "3sgm", "3sgf", "3sgh", "3sgn", "1pl", "2pl", "3pl"]


class Language:
    """One concrete invented language: a grammar config plus a seeded lexicon."""

    def __init__(self, config: GrammarConfig, seed: int = 0, vocab: Vocab | None = None):
        self.config = config
        self.seed = seed
        self.vocab = vocab or Vocab.small()
        self.phon = PHONOLOGIES[config.phonology]
        rng = random.Random(seed)
        forms = FormFactory(self.phon, rng)

        # grammatical endings first (short forms), then roots
        names = ["ART", "PL", "ACC", "LOC", "NEG", "PRS", "PST", "FUT", "AGR.3sg"]
        names += [f"AGR.{k}" for k in AGREEMENT_CELLS]
        self.grams = {n: forms.ending() for n in names}
        self.roots = {c: forms.root() for c in self.vocab.concepts()}

        verbs = self.vocab.transitive + self.vocab.intransitive
        n_irr = round(config.irregular_rate * len(verbs))
        self.irregular_past = {v: forms.root() for v in rng.sample(verbs, n_irr)}
        self._rng, self._forms = rng, forms        # kept so conlang/v2.py can extend the language (no extra draws)

    # --- word building ---------------------------------------------------
    def _join(self, parts: list[tuple[str, ...]], real: Realization) -> tuple[str, ...]:
        out = list(parts[0])
        for nxt in parts[1:]:
            nxt = list(nxt)
            real.boundaries += 1
            if self.config.sandhi and out and nxt:
                left, right = out[-1], nxt[0]
                if is_vowel(left) and is_vowel(right):
                    real.sandhi_changes += 1
                    if left == "u" and len(out) > 2:
                        out.pop()                                    # enunciative -u drops
                    else:
                        out.append("v" if left in ("a", "o", "u", "aa", "oo", "uu") else "y")
                elif is_vowel(left) and len(left) == 1 and right in STOPS:
                    real.sandhi_changes += 1
                    out.append(right)                                # doubling: -kk-, -tt-
            out += nxt
        return tuple(out)

    def _word(self, parts: list[tuple[str, ...]], gloss: list[str], real: Realization) -> Word:
        real.morphemes.append(len(parts))
        return Word(self._join(parts, real), "-".join(gloss))

    def _np(self, np: NP, case: str | None, real: Realization) -> list[Word]:
        c = self.config
        parts, gloss = [self.roots[f"N:{np.noun}"]], [np.noun]
        if np.plural and c.plural == "suffix":
            parts.append(self.grams["PL"]); gloss.append("PL")
        after = []
        if case == "ACC":
            if c.acc_marking == "suffix":
                parts.append(self.grams["ACC"]); gloss.append("ACC")
            elif c.acc_marking == "particle":
                after.append(self._word([self.grams["ACC"]], ["ACC"], real))
        noun = self._word(parts, gloss, real)
        words: list[Word] = []
        if c.articles:
            words.append(self._word([self.grams["ART"]], ["the"], real))
        adjs = np.adj.split() if np.adj else []           # several adjectives: each its own word, English order
        if c.adj_order == "AN":
            words += [self._word([self.roots[f"A:{a}"]], [a], real) for a in adjs]
        words.append(noun)
        if c.adj_order == "NA":
            words += [self._word([self.roots[f"A:{a}"]], [a], real) for a in adjs]
        return words + after

    def _loc(self, loc: str, real: Realization) -> list[Word]:
        c = self.config
        words: list[Word] = []
        if c.articles:
            words.append(self._word([self.grams["ART"]], ["the"], real))
        if c.loc_marking == "suffix":
            words.append(self._word([self.roots[f"L:{loc}"], self.grams["LOC"]], [loc, "LOC"], real))
            return words
        words.append(self._word([self.roots[f"L:{loc}"]], [loc], real))
        particle = self._word([self.grams["LOC"]], ["in"], real)
        return [particle] + words if c.loc_marking == "prep" else words + [particle]

    def _verb(self, f: Frame, real: Realization) -> list[Word]:
        c = self.config
        person, number, gender = f.subject_png(self.vocab)
        irregular = f.tense == "past" and f.verb in self.irregular_past
        stem = self.irregular_past[f.verb] if irregular else self.roots[f"V:{f.verb}"]
        parts, gloss = [stem], [f.verb + (".PST" if irregular else "")]
        helpers: list[Word] = []

        if c.tense == "suffix":
            tag = {"present": "PRS", "past": "PST", "future": "FUT"}[f.tense]
            parts.append(self.grams[tag]); gloss.append(tag)
        else:  # english-like
            if f.tense == "past" and not irregular:
                parts.append(self.grams["PST"]); gloss.append("PST")
            elif f.tense == "future":
                helpers.append(self._word([self.grams["FUT"]], ["will"], real))

        if c.agreement == "full":
            key = f"{person}{number}" + (gender if (person == 3 and number == "sg") else "")
            parts.append(self.grams[f"AGR.{key}"]); gloss.append(f"AGR.{key}")
        elif c.agreement == "3sg" and person == 3 and number == "sg" and f.tense == "present":
            parts.append(self.grams["AGR.3sg"]); gloss.append("3SG")

        if f.neg:
            if c.negation == "suffix":
                parts.append(self.grams["NEG"]); gloss.append("NEG")
            else:
                helpers.append(self._word([self.grams["NEG"]], ["not"], real))

        return helpers + [self._word(parts, gloss, real)]

    # --- sentence ----------------------------------------------------------
    def realize(self, f: Frame) -> Realization:
        real = Realization(words=[])
        if isinstance(f.subj, str):
            subj = [self._word([self.roots[f"PRON.{f.subj}"]], [f.subj], real)]
        else:
            subj = self._np(f.subj, None, real)
        verb = self._verb(f, real)
        obj = self._np(f.obj, "ACC", real) if f.obj is not None else []
        loc = self._loc(f.loc, real) if f.loc is not None else []

        order = self.config.word_order
        if order == "SVO":
            real.words = subj + verb + obj + loc
        elif order == "SOV":
            real.words = subj + loc + obj + verb
        elif order == "VSO":
            real.words = verb + subj + obj + loc
        else:
            raise ValueError(f"unknown word order {order}")
        return real
