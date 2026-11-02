"""Language v2: 10,000+ root words, words built from pieces, a 'how' slot, partial inputs, and everyday grammar.

v2 extends the v1 language (same grammar dials, same seed) without changing it: the v1 language is built first with
exactly the v1 random draws, then v2 draws its new endings and roots after them. Every v1 sentence made only of v1
root words is therefore rendered exactly as before (checked by sealed/app/test_v2.py). The everyday additions (below)
are drawn after all of that, so every earlier v2 word keeps its form too.

New in v2
- Vocabulary (conlang/data/vocab_v2.json, scripts/build_vocab_v2.py): 11,000+ root words including abstract and
  mass nouns (fun, idea, water; a mass noun has no plural), plus a few plain adverbs (fast, well).
- Words built from pieces. A word key is a root plus endings, each ending one fixed morpheme of the language:
      fun>ADJ (funny)  hope>LESS (hopeless)  read>ABLE (readable)  run>AGENT (runner)  move>ACT (movement)
      happy>QUAL (happiness)  dark>VERB (darken)  quick>ADV (quickly)  happy>NOT (unhappy)
      big>MORE (bigger)  big>MOST (biggest)
  Endings stack: happy>NOT>QUAL (unhappiness). Compounds join two nouns: book_shop (bookshop), fun_ride (fun-ride).
  Glosses: fun-ADJ, book+shop.
- A 'how' slot (M): manner, time, place and degree words, placed right before the verb (Tamil-like).
- 'be' (X is Y): V=be with the tense and person endings; what X is comes right before it, as describing words
  (A=fun: "the rides were fun") or a noun phrase (P=teacher: "Mira is a teacher"), with no case ending.
- Partial inputs: every part of an event is optional, and whatever is given is translated:
      a noun phrase alone        -> the noun phrase (no case ending)
      a verb without a tense     -> its dictionary form: the bare stem (+ NEG if negated); also commands
      a verb with tense, no doer -> stem + tense (+ NEG), no person ending
      a place alone              -> place-LOC;  describing words alone (A) -> those words;  M alone -> that word

Everyday grammar (vocab additions in conlang/data/vocab_v2_extra.json, scripts/build_vocab_v2_extra.py)
- Pronouns anywhere: as the thing done to (me = PRON.1sg-ACC), after a relation word (with me), as owner (my).
- Owners (genitive): my book = PRON.1sg-GEN book; Mira's dog; the boy's dog; the book of the man. Written 'X^Y'.
- Relation words are case endings on the noun, like LOC (in/at): ON UNDER TO FROM WITH FOR NEAR BEHIND FRONT ABOUT
  THAN WITHOUT BEFORE AFTER THROUGH AROUND BETWEEN DURING UNTIL SINCE AGAINST. "to" also marks who receives
  something (give me the book = PRON.1sg-TO book-ACC give).
- Determiners are words of their own (Tamil has no articles, so 'the' and 'a' stay unwritten): this, that,
  numbers, all, some, many, ... (these dogs = this dog-PL).
- Joining nouns: the boy and the girl = boy AND girl (written boy&girl; 'or' is boy|girl).
- Verb endings after the stem, in this order: PROG (is running), PERF (has eaten), a modal CAN / MUST / SHOULD /
  MAY / WOULD, or LET (let's go), then tense, person, NEG, and Q for a yes/no question (are you happy?).
- Question words are words of their own and stay where the answer would be (Tamil-like): what, who, which, where,
  when, why, how ("what do you want" = you what want).
- Greetings and exclamations (I=hello), the person spoken to (Y=NAME.Mira), a joining word that starts an event
  (C=because: because, if, when, but, so, then, that, to, ...), and time phrases without a relation word
  (T=every+day, T=last+night).

Analysis format (the meaning of one event; events separated by ' ; '):
    S=fun>ADJ+old+dog.PL  V=run.PAST.NEG  M=quick>ADV  O=NAME.Mira  L=garden  A=happy  P=teacher
    C=because  I=hello  Y=NAME.Mira  T=every+day  ON=table  TO=PRON.1sg  WITH=PRON.1sg^friend.PL
All fields are optional (at least one). V without a tense (V=run) is the dictionary form.
A noun phrase is [owner^]words+noun[.PL]; owners nest (PRON.1sg^mother^book: my mother's book), PRON.1sg^ alone is
"mine"; nouns are joined with & (and) or | (or).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .dataset import PRESETS
from .frames import NP, PRONOUNS, Vocab
from .grammar import Language, Realization, Word
from .script import Script

VOCAB_V2 = Path(__file__).resolve().parent / "data" / "vocab_v2.json"
VOCAB_V2_EXTRA = Path(__file__).resolve().parent / "data" / "vocab_v2_extra.json"
AFFIXES = ["ADJ", "LESS", "ABLE", "AGENT", "ACT", "QUAL", "VERB", "ADV", "NOT", "MORE", "MOST"]
# which kind of word each ending attaches to (in order of preference), and what it makes
AFFIX_BASE = {"ADJ": "NVA", "LESS": "NV", "ABLE": "VN", "AGENT": "VN", "ACT": "VN", "QUAL": "AN", "VERB": "AN",
              "ADV": "A", "NOT": "AVN", "MORE": "ARD", "MOST": "ARD"}
AFFIX_OUT = {"ADJ": "A", "LESS": "A", "ABLE": "A", "AGENT": "N", "ACT": "N", "QUAL": "N", "VERB": "V", "ADV": "R",
             "MORE": "A", "MOST": "A"}
# where a word is looked up, by the slot it fills (N noun, L place, A describing, V verb, R how, I greeting,
# C joining word; D determiner and W question word are the grammar's own words)
ROLE = {"N": ("N", "L", "D", "W", "A", "R"), "L": ("L", "N", "W"), "A": ("A", "D", "W", "R"), "V": ("V",),
        "R": ("R", "W", "D", "A"), "D": ("D", "A", "W"), "I": ("I", "N", "A", "R", "D", "W", "V"), "C": ("C",)}
TENSE_TAG = {"present": "PRS", "past": "PST", "future": "FUT"}
TENSE_A = {"present": "PRES", "past": "PAST", "future": "FUT"}

# --- the grammar's own words (each a root of the language; drawn after everything else) -----------------------------
NUMBERS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
           "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
           "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion",
           "zero"]
DETERMINERS = ["this", "that", "all", "every", "each", "some", "any", "many", "much", "few", "little", "several",
               "both", "other", "another", "no", "more", "most", "less", "enough", "first", "second", "third",
               "half", "such", "own"] + NUMBERS
QUESTION_WORDS = ["what", "who", "which", "where", "when", "why", "how"]
INTERJECTIONS = ["hello", "hi", "hey", "bye", "goodbye", "thanks", "yes", "no", "okay", "please", "sorry", "wow",
                 "oh", "ah", "oops", "ouch", "hurray", "welcome", "congratulations", "cheers", "alas", "hmm",
                 "ofcourse"]
CONNECTORS = ["and", "or", "but", "so", "because", "if", "when", "while", "before", "after", "until", "since",
              "then", "that", "to", "although", "unless", "whether"]
# relation fields (case endings); L (in / at) is the v1 location ending LOC
CASE_FIELDS = ["ON", "UNDER", "TO", "FROM", "WITH", "FOR", "NEAR", "BEHIND", "FRONT", "ABOUT", "THAN", "WITHOUT",
               "BEFORE", "AFTER", "THROUGH", "AROUND", "BETWEEN", "DURING", "UNTIL", "SINCE", "AGAINST"]
MODALS = ["CAN", "MUST", "SHOULD", "MAY", "WOULD", "LET"]
VERB_MODS = ["PROG", "PERF"] + MODALS + ["Q"]                  # NEG and the tense are kept as before
NEW_GRAMS = ["GEN"] + [f"CASE.{c}" for c in CASE_FIELDS] + VERB_MODS
FIELDS = ["C", "I", "Y", "S", "T", "V", "M", "O", "L"] + CASE_FIELDS + ["A", "P"]
PLURALIA = {"people", "police", "cattle", "clothes", "staff"}   # nouns that mean many (they take PL, never 'a')
WHO_LIKE = {"who", "someone", "somebody", "everyone", "everybody", "anyone", "anybody", "nobody"}


class VocabV2(Vocab):
    """The v1 vocabulary plus v2's roots, derived words, compounds and adverbs, plus the everyday additions."""

    def __init__(self, d: dict, extra: dict | None = None):
        super().__init__(d, "v2")
        self.v1 = Vocab.large()
        self.adverbs = list(d["adverbs"])
        self.mass = {n["lemma"] for n in d["nouns"] if n["plural"] is None} - PLURALIA
        self.derived = {(x["word"], x["pos"]): x for x in d["derived"]}
        self.compounds = {c["word"]: c for c in d["compounds"]}
        self.v1_gender = {n["lemma"]: n["gender"] for n in d["nouns"]}
        # everyday additions: kept in their own lists (concepts() and so every earlier draw stay as they were)
        extra = extra or {"nouns": [], "verbs": [], "adjectives": [], "adverbs": []}
        self.x_nouns = [n["lemma"] for n in extra["nouns"]]
        self.x_verbs = [v["base"] for v in extra["verbs"]]
        self.x_adjectives = list(extra["adjectives"])
        self.x_adverbs = list(extra["adverbs"])
        self.participle = {v["base"]: v["participle"] for v in extra["verbs"]}
        for n in extra["nouns"]:
            self.noun_info[n["lemma"]] = (n["plural"], n["gender"])
            self.v1_gender.setdefault(n["lemma"], n["gender"])
            if n["plural"] is None and n["lemma"] not in PLURALIA:
                self.mass.add(n["lemma"])
        for v in extra["verbs"]:
            self.verb_info[v["base"]] = (v["base"], v["third"], v["past"])

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls) -> "VocabV2":
        extra = json.loads(VOCAB_V2_EXTRA.read_text(encoding="utf-8")) if VOCAB_V2_EXTRA.exists() else None
        return cls(json.loads(VOCAB_V2.read_text(encoding="utf-8")), extra)

    def concepts(self) -> list[str]:
        return super().concepts() + [f"R:{w}" for w in self.adverbs]

    def extra_concepts(self) -> list[str]:
        """Concepts of the everyday additions, in drawing order (after every v2 draw)."""
        return ([f"N:{w}" for w in self.x_nouns] + [f"V:{w}" for w in self.x_verbs]
                + [f"A:{w}" for w in self.x_adjectives] + [f"R:{w}" for w in self.x_adverbs]
                + [f"D:{w}" for w in DETERMINERS] + [f"W:{w}" for w in QUESTION_WORDS]
                + [f"I:{w}" for w in INTERJECTIONS] + [f"C:{w}" for w in CONNECTORS])

    def key(self, word: str, pos: str) -> str:
        """Canonical key of an English word: roots stay themselves; built words expand to root>ENDING..."""
        d = self.derived.get((word, pos))
        if d:
            return f"{self.key(d['base'], d['base_pos'])}>{d['affix']}"
        c = self.compounds.get(word) if pos == "N" else None
        if c:
            return "_".join(c["parts"])
        return word

    def gender(self, noun_key: str) -> str:
        """Gender of a noun key (for the verb's person ending): people nouns keep their v1 gender."""
        if noun_key in WHO_LIKE:
            return "h"
        for (w, pos), d in self.derived.items():
            if pos == "N" and self.key(w, "N") == noun_key and w in self.v1_gender:
                return self.v1_gender[w]
        if ">" in noun_key:
            return "h" if noun_key.endswith(">AGENT") else "n"
        if "_" in noun_key:
            return self.gender(noun_key.split("_")[-1])
        info = self.noun_info.get(noun_key)
        return info[1] if info else "n"


# --- meanings ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class NPX(NP):
    """A noun phrase with an owner (my book: poss='1sg'); noun '' = the owner alone (mine)."""
    poss: object = None                        # pronoun key, Name, NP/NPX or Coord


@dataclass(frozen=True)
class Name:
    name: str
    ph: tuple
    gender: str = "m"


@dataclass(frozen=True)
class Coord:
    items: tuple                               # nominals
    conj: str = "and"                          # and | or


@dataclass
class ClauseV2:
    subj: object = None                       # nominal: NP, a pronoun key ("3sgf"), Name, Coord; or None
    subj_name: str | None = None
    subj_ph: tuple | None = None
    subj_gender: str = "m"                    # for a name doer: m / f (it changes the verb's person ending)
    verb: str | None = None                   # verb key; None = no verb
    tense: str | None = None                  # present / past / future; None = dictionary form
    neg: bool = False
    obj: object = None                        # nominal
    obj_name: str | None = None
    obj_ph: tuple | None = None
    loc: str | None = None                    # a place key (v1: in the garden)
    adv: str | None = None                    # how/when/where words, '+'-joined (quick>ADV, very+much, today)
    adjs: tuple = field(default_factory=tuple)  # describing words: alone, or what X is (with V=be)
    pred: object = None                       # a noun phrase that X is (with V=be): "Mira is a teacher"
    obls: tuple = field(default_factory=tuple)  # ((field, nominal), ...): L and the relation fields, in order
    time: tuple = field(default_factory=tuple)  # time nominals without a relation word: every day
    intj: tuple = field(default_factory=tuple)  # greeting / exclamation keys
    voc: object = None                        # the person spoken to (nominal)
    link: str | None = None                   # joining word that starts this event (because, if, ...)
    mods: tuple = field(default_factory=tuple)  # verb endings from VERB_MODS


def nominal_number(x) -> tuple[int, str, str | None]:
    """Person, number and gender of a nominal (for the verb ending)."""
    if isinstance(x, str):
        _, p, n, g = PRONOUNS[x]
        return p, n, g
    if isinstance(x, Name):
        return 3, "sg", x.gender
    if isinstance(x, Coord):
        if x.conj == "or":
            return nominal_number(x.items[-1])
        ps = [(nominal_number(i) or (3,))[0] for i in x.items]
        return min(ps), "pl", None
    return None                                # an NP: decided by the language (needs the vocabulary)


LOAN_UNITS = ["aa", "ee", "ii", "oo", "uu", "ng", "ny", "a", "e", "i", "o", "u", "k", "c", "t", "n", "p", "m", "y", "r",
              "l", "v"]
LOAN_SPELL = [("tch", "c"), ("th", "t"), ("sh", "c"), ("ch", "c"), ("ph", "p"), ("ck", "k"), ("qu", "kv"), ("gh", ""),
              ("wh", "v"), ("ea", "ii"), ("ou", "av"), ("ow", "av"), ("oa", "oo"), ("ai", "ee"), ("ay", "ee"),
              ("b", "p"), ("d", "t"), ("f", "p"), ("g", "k"), ("h", ""), ("j", "c"), ("q", "k"), ("s", "c"), ("w", "v"),
              ("x", "kc"), ("z", "c")]


def loan_sounds(word: str) -> tuple[str, ...]:
    """An English word respelled with the language's sounds (pizza -> p i c a): letters the language lacks become
    the nearest sound (b->p, d->t, f->p, g->k, s/z->c, th->t, sh/ch->c, w->v, h dropped); doubled letters once."""
    s = re.sub(r"[^a-z]", "", word.lower())
    for a, b in LOAN_SPELL:
        s = s.replace(a, b.upper())
    s = re.sub(r"([^aeiou])\1+", r"\1", s.lower())                     # tt -> t, cc -> c
    s = re.sub(r"([aeiou])\1\1+", r"\1\1", s)
    out, i = [], 0
    while i < len(s):
        u = next(u for u in LOAN_UNITS if s.startswith(u, i))
        out.append(u)
        i += len(u)
    return tuple(out) or ("a",)


# Extension (2026-10-28, SOUND_RULE_LANGUAGE.md): every content root (nouns, verbs, describing and how words, places)
# is its English word respelled by the loanword rules (dog -> t o k); grammar, endings, pronouns, determiners,
# connectors, question words and interjections stay v2's. Off = the v2 language exactly (read when a LanguageV2 is made).
SOUND_RULE = {"on": False, "words": None}   # words: None = every content word (as before); a set = only these
                                            # English words are rule-spelled, the rest keep v2's roots (01 Nov)
CONTENT_ROLES = ("N", "V", "A", "R", "L")


class LanguageV2(Language):
    def __init__(self, config, seed: int, vocab: VocabV2):
        super().__init__(config, seed, vocab.v1)                   # exactly the v1 language, v1 random draws
        forms, rng = self._forms, self._rng
        for a in AFFIXES[:9]:
            self.grams[f"DER.{a}"] = forms.ending()
        for c in vocab.concepts():
            if c not in self.roots:
                self.roots[c] = forms.root()
        new_verbs = [v for v in vocab.transitive + vocab.intransitive if v not in vocab.v1.verb_info]
        self.irregular_past.update({v: forms.root() for v in
                                    rng.sample(new_verbs, round(config.irregular_rate * len(new_verbs)))})
        self.roots["V:be"] = forms.root()                          # 'be' (X is Y), drawn last: changes nothing else
        # everyday additions, drawn after all of the above (no earlier form changes)
        for g in NEW_GRAMS:
            self.grams[g] = forms.ending()
        for a in AFFIXES[9:]:
            self.grams[f"DER.{a}"] = forms.ending()
        have = {c.split(":", 1)[1] for c in self.roots if ":" in c}
        for c in vocab.extra_concepts():
            role, w = c.split(":", 1)
            if c in self.roots:
                continue
            if role == "D" and (f"A:{w}" in self.roots or f"R:{w}" in self.roots):
                continue                                            # already a describing / how word (many, much)
            if role == "I" and w in have:                           # already a word (hello, sorry): reuse it
                continue
            self.roots[c] = forms.root()
            have.add(w)
        self.grams["LOAN"] = forms.ending()                       # loanwords (LOAN.pizza), drawn last of all
        self.vocab = vocab
        self.v2 = vocab
        if SOUND_RULE["on"]:                                       # after every draw: endings, function words and
            sel = SOUND_RULE.get("words")                          # names stay exactly v2's (pronouns: PRON.1sg)
            for c in self.roots:
                role, _, w = c.partition(":")
                if w and role in CONTENT_ROLES and c != "V:be" and (sel is None or w in sel):
                    self.roots[c] = loan_sounds(w)
            if sel is None:
                self.irregular_past = {}                           # a random past form is not rule-built
            else:                                                  # partial rule: only the rule words lose theirs
                self.irregular_past = {v: f for v, f in self.irregular_past.items() if v not in sel}

    # --- words from keys ------------------------------------------------------------------------------------
    def root_of(self, word: str, prefs: str) -> tuple[tuple, str]:
        for p in prefs:
            for r in ROLE[p]:
                if f"{r}:{word}" in self.roots:
                    return self.roots[f"{r}:{word}"], r
        raise KeyError(f"'{word}' is not a word of the language")

    def stem(self, key: str, pos: str, real: Realization) -> tuple:
        if ">" in key:
            base, *affixes = key.split(">")
            for a in affixes:
                if a not in AFFIXES:
                    raise KeyError(f"unknown ending '{a}' (endings: {', '.join(AFFIXES)})")
            ph = self.stem(base, AFFIX_BASE[affixes[0]], real) if ("_" in base) else \
                self.root_of(base, AFFIX_BASE[affixes[0]])[0]
            return self._join([ph] + [self.grams[f"DER.{a}"] for a in affixes], real)
        if key.startswith("LOAN."):                                 # a borrowed English noun: its sounds + LOAN
            return self._join([loan_sounds(key[5:]), self.grams["LOAN"]], real)
        if "_" in key:
            return self._join([self.root_of(p, "NVA")[0] for p in key.split("_")], real)   # any root words
        return self.root_of(key, pos)[0]

    @staticmethod
    def gloss(key: str) -> str:
        return key.replace(">", "-").replace("_", "+")

    # --- parts ------------------------------------------------------------------------------------------------
    def case_parts(self, case: str | None) -> tuple[list, list[str], str]:
        """(suffix morphemes, their glosses, where a separate case word goes: '' | 'before' | 'after')."""
        c = self.config
        if case is None:
            return [], [], ""
        if case == "ACC":
            if c.acc_marking == "suffix":
                return [self.grams["ACC"]], ["ACC"], ""
            return [], [], ("after" if c.acc_marking == "particle" else "")
        if case == "GEN":
            return [self.grams["GEN"]], ["GEN"], ""
        gram = self.grams["LOC"] if case == "LOC" else self.grams[f"CASE.{case}"]
        if c.loc_marking == "suffix":
            return [gram], [case], ""
        return [], [], ("before" if c.loc_marking == "prep" else "after")

    def case_word(self, case: str, real: Realization) -> Word:
        gram = {"ACC": "ACC", "LOC": "LOC"}.get(case, f"CASE.{case}")
        return self._word([self.grams[gram]], [case if case != "LOC" else "in"], real)

    def np_words(self, np: NP, case: str | None, real: Realization, pos: str = "N") -> list[Word]:
        c = self.config
        owner = self.nominal_words(np.poss, "GEN", real) if isinstance(np, NPX) and np.poss is not None else []
        if not np.noun:                                              # the owner alone: mine, Mira's
            if case and owner:
                suf, gl, where = self.case_parts(case)
                last = owner[-1]
                owner[-1] = Word(self._join([last.phonemes] + suf, real), "-".join([last.gloss] + gl))
                if where:
                    w = self.case_word(case, real)
                    owner = [w] + owner if where == "before" else owner + [w]
            return owner
        parts, gloss = [self.stem(np.noun, pos, real)], [self.gloss(np.noun)]
        if np.plural and c.plural == "suffix":
            parts.append(self.grams["PL"]); gloss.append("PL")
        suf, gl, where = self.case_parts(case)
        parts += suf
        gloss += gl
        noun = self._word(parts, gloss, real)
        words: list[Word] = []
        if c.articles:
            words.append(self._word([self.grams["ART"]], ["the"], real))
        adjs = [self._word([self.stem(a, "A", real)], [self.gloss(a)], real) for a in (np.adj.split() if np.adj else [])]
        words += adjs if c.adj_order == "AN" else []
        words.append(noun)
        words += adjs if c.adj_order == "NA" else []
        words = owner + words
        if where == "before":
            words = [self.case_word(case, real)] + words
        elif where == "after":
            words = words + [self.case_word(case, real)]
        return words

    def nominal_words(self, x, case: str | None, real: Realization, pos: str = "N") -> list[Word]:
        """Words of any nominal (NP, pronoun key, Name, Coord) with a case ending (None = no ending)."""
        if isinstance(x, Coord):
            out: list[Word] = []
            for i, item in enumerate(x.items):
                if i:
                    out.append(self._word([self.roots[f"C:{x.conj}"]], [x.conj], real))
                out += self.nominal_words(item, case, real, pos)
            return out
        suf, gl, where = self.case_parts(case)
        if isinstance(x, str):
            words = [self._word([self.roots[f"PRON.{x}"]] + suf, [x] + gl, real)]
        elif isinstance(x, Name):
            words = [Word(self._join([x.ph] + suf, real) if suf else x.ph, "-".join([x.name] + gl))]
        else:
            return self.np_words(x, case, real, pos)
        if where:
            w = self.case_word(case, real)
            words = [w] + words if where == "before" else words + [w]
        return words

    def loc_words(self, loc: str, real: Realization) -> list[Word]:
        c = self.config
        words: list[Word] = []
        if c.articles:
            words.append(self._word([self.grams["ART"]], ["the"], real))
        stem = self.stem(loc, "L", real)
        if c.loc_marking == "suffix":
            return words + [self._word([stem, self.grams["LOC"]], [self.gloss(loc), "LOC"], real)]
        words.append(self._word([stem], [self.gloss(loc)], real))
        particle = self._word([self.grams["LOC"]], ["in"], real)
        return [particle] + words if c.loc_marking == "prep" else words + [particle]

    def obl_words(self, fld: str, x, real: Realization) -> list[Word]:
        if fld == "L":
            plain = isinstance(x, str) or (type(x) is NP and not x.adj and not x.plural)
            if plain and not (isinstance(x, str) and x in PRONOUNS):
                return self.loc_words(x if isinstance(x, str) else x.noun, real)     # exactly v1's place
            return self.nominal_words(x, "LOC", real, "L")
        return self.nominal_words(x, fld, real)

    def person(self, cl: ClauseV2) -> tuple[int, str, str | None] | None:
        if cl.subj_name:
            return 3, "sg", cl.subj_gender
        if cl.subj is None:
            return None
        s = cl.subj
        while isinstance(s, Coord) and s.conj == "or":             # tea or coffee: the verb agrees with the last
            s = s.items[-1]
        png = nominal_number(s)
        if png:
            return png
        if isinstance(s, NPX) and not s.noun:
            return 3, "sg", "n"
        n = "pl" if s.plural else "sg"
        return 3, n, (self.v2.gender(s.noun) if n == "sg" else None)

    def verb_words(self, cl: ClauseV2, real: Realization) -> list[Word]:
        c = self.config
        irregular = (cl.tense == "past" and cl.verb in self.irregular_past and ">" not in cl.verb
                     and not ({"PROG", "PERF"} | set(MODALS)) & set(cl.mods))
        stem = self.irregular_past[cl.verb] if irregular else self.stem(cl.verb, "V", real)
        parts, gloss = [stem], [self.gloss(cl.verb) + (".PST" if irregular else "")]
        helpers: list[Word] = []
        for m in VERB_MODS[:-1]:                                      # PROG, PERF, a modal or LET
            if m in cl.mods:
                parts.append(self.grams[m]); gloss.append(m)
        if cl.tense:
            if c.tense == "suffix":
                tag = TENSE_TAG[cl.tense]
                parts.append(self.grams[tag]); gloss.append(tag)
            elif cl.tense == "past" and not irregular:
                parts.append(self.grams["PST"]); gloss.append("PST")
            elif cl.tense == "future":
                helpers.append(self._word([self.grams["FUT"]], ["will"], real))
            png = self.person(cl)
            if png:
                person, number, gender = png
                if c.agreement == "full":
                    key = f"{person}{number}" + (gender if (person == 3 and number == "sg") else "")
                    parts.append(self.grams[f"AGR.{key}"]); gloss.append(f"AGR.{key}")
                elif c.agreement == "3sg" and person == 3 and number == "sg" and cl.tense == "present":
                    parts.append(self.grams["AGR.3sg"]); gloss.append("3SG")
        if cl.neg:
            if c.negation == "suffix":
                parts.append(self.grams["NEG"]); gloss.append("NEG")
            else:
                helpers.append(self._word([self.grams["NEG"]], ["not"], real))
        if "Q" in cl.mods:
            parts.append(self.grams["Q"]); gloss.append("Q")
        return helpers + [self._word(parts, gloss, real)]

    # --- one event --------------------------------------------------------------------------------------------
    def words_v2(self, cl: ClauseV2) -> list[Word]:
        real = Realization(words=[])
        if cl.subj_name:
            subj = [Word(cl.subj_ph, cl.subj_name)]
        elif isinstance(cl.subj, str) and cl.subj in PRONOUNS:
            subj = [self._word([self.roots[f"PRON.{cl.subj}"]], [cl.subj], real)]
        elif cl.subj is not None:
            subj = self.nominal_words(cl.subj, None, real)
        else:
            subj = []
        if cl.obj_name:
            obj = [Word(self._join([cl.obj_ph, self.grams["ACC"]], real), f"{cl.obj_name}-ACC")]
        elif cl.obj is not None:
            obj = self.nominal_words(cl.obj, "ACC", real)
        else:
            obj = []
        obls = ([("L", cl.loc)] if cl.loc else []) + list(cl.obls)
        loc = [w for fld, x in obls for w in self.obl_words(fld, x, real)]
        adv = [self._word([self.stem(a, "R" if ">" not in a else "A", real)], [self.gloss(a)], real)
               for a in (cl.adv.split("+") if cl.adv else [])]
        verb = self.verb_words(cl, real) if cl.verb else []
        adjs = [self._word([self.stem(a, "A", real)], [self.gloss(a)], real) for a in cl.adjs]
        pred = self.nominal_words(cl.pred, None, real) if cl.pred is not None else []
        link = [self._word([self.roots[f"C:{cl.link}"]], [cl.link], real)] if cl.link else []
        intj = [self._word([self.stem(k, "I", real)], [self.gloss(k)], real) for k in cl.intj]
        voc = self.nominal_words(cl.voc, None, real) if cl.voc is not None else []
        time = [w for t in cl.time for w in self.nominal_words(t, None, real)]
        head = link + intj + voc
        order = self.config.word_order
        if order == "SOV":
            return head + subj + time + adjs + pred + loc + obj + adv + verb
        if order == "SVO":
            return head + subj + verb + adv + adjs + pred + obj + loc + time
        return head + verb + adv + subj + adjs + pred + obj + loc + time            # VSO


def nominal_text(x) -> str:
    """A nominal in the analysis format."""
    if isinstance(x, str):
        return f"PRON.{x}"
    if isinstance(x, Name):
        return f"NAME.{x.name}"
    if isinstance(x, Coord):
        return ("&" if x.conj == "and" else "|").join(nominal_text(i) for i in x.items)
    owner = nominal_text(x.poss) + "^" if isinstance(x, NPX) and x.poss is not None else ""
    if not x.noun:
        return owner
    return owner + "+".join((x.adj.split() if x.adj else []) + [x.noun]) + (".PL" if x.plural else "")


class RepV2:
    """Renders v2 meanings into glyphs (the v2 counterpart of train/represent.py's Rep.target/analysis)."""

    def __init__(self, preset: str = "tamil_like", script: str = "abugida_pua", seed: int = 0):
        self.v = VocabV2.load()
        self.lang = LanguageV2(PRESETS[preset], seed, self.v)
        self.script = Script(script, PRESETS[preset].phonology)
        self.seed = seed

    def words(self, cl: ClauseV2) -> list[Word]:
        return self.lang.words_v2(cl)

    def target(self, cl: ClauseV2) -> str:
        return " ".join(self.script.word(w.phonemes) for w in self.words(cl))

    @staticmethod
    def analysis(cl: ClauseV2) -> str:
        out = []
        if cl.link:
            out.append(f"C={cl.link}")
        if cl.intj:
            out.append("I=" + "+".join(cl.intj))
        if cl.voc is not None:
            out.append(f"Y={nominal_text(cl.voc)}")
        if cl.subj_name:
            out.append(f"S=NAME.{cl.subj_name}")
        elif cl.subj is not None:
            out.append(f"S={nominal_text(cl.subj)}")
        for t in cl.time:
            out.append(f"T={nominal_text(t)}")
        if cl.verb:
            mods = [m for m in VERB_MODS[:-1] if m in cl.mods]
            out.append(f"V={cl.verb}" + (f".{TENSE_A[cl.tense]}" if cl.tense else "") + "".join(f".{m}" for m in mods)
                       + (".NEG" if cl.neg else "") + (".Q" if "Q" in cl.mods else ""))
        if cl.adv:
            out.append(f"M={cl.adv}")
        if cl.obj_name:
            out.append(f"O=NAME.{cl.obj_name}")
        elif cl.obj is not None:
            out.append(f"O={nominal_text(cl.obj)}")
        if cl.loc:
            out.append(f"L={cl.loc}")
        for fld, x in cl.obls:
            out.append(f"{fld}={nominal_text(x)}")
        if cl.adjs:
            out.append("A=" + "+".join(cl.adjs))
        if cl.pred is not None:
            out.append(f"P={nominal_text(cl.pred)}")
        return " ".join(out)

    def check(self, cl: ClauseV2) -> list[str]:
        """Problems with a meaning (unknown words, plural of a mass noun), [] if it can be rendered."""
        issues = []
        try:
            self.words(cl)
        except KeyError as e:
            issues.append(e.args[0])

        def nps(x):
            if isinstance(x, Coord):
                for i in x.items:
                    yield from nps(i)
            elif isinstance(x, NP):
                yield x
                if isinstance(x, NPX) and x.poss is not None:
                    yield from nps(x.poss)
        for top in [cl.subj, cl.obj, cl.pred, cl.voc] + [x for _, x in cl.obls] + list(cl.time):
            for np in nps(top):
                if np.plural and np.noun in self.v.mass:
                    issues.append(f"'{np.noun}' has no plural in the language (a mass noun like water or fun)")
        return issues


def split_top(s: str, seps: str) -> list[str]:
    return [p for p in re.split(f"[{re.escape(seps)}]", s)]


def nominal_from_text(s: str, spell_name, issues: list[str]):
    """Analysis text of a nominal -> nominal (pronoun key, Name, NP/NPX, Coord)."""
    for sep, conj in (("&", "and"), ("|", "or")):
        if sep in s:
            return Coord(tuple(nominal_from_text(p, spell_name, issues) for p in s.split(sep)), conj)
    if "^" in s:
        owner, head = s.rsplit("^", 1)
        poss = nominal_from_text(owner, spell_name, issues)
        if not head:
            return NPX("", False, None, poss)
        np = nominal_from_text(head, spell_name, issues)
        if not isinstance(np, NP):
            issues.append(f"'{head}' after '^' must be a noun phrase")
            return np
        return NPX(np.noun, np.plural, np.adj, poss)
    if s.startswith("NAME."):
        nm = s[5:]
        ph, why = spell_name(nm)
        if ph is None:
            issues.append(why)
        return Name(nm[:1].upper() + nm[1:].lower(), ph or ())
    if s.startswith("PRON."):
        if s[5:] not in PRONOUNS:
            issues.append(f"unknown pronoun '{s}' ({', '.join('PRON.' + k for k in PRONOUNS)})")
        return s[5:]
    plural = s.endswith(".PL")
    words = (s[:-3] if plural else s).split("+")
    return NP(words[-1], plural, " ".join(words[:-1]) or None)


def from_analysis(text: str, spell_name) -> tuple[list[ClauseV2], list[str]]:
    """Analysis text -> events. `spell_name(name) -> (sounds or None, reason)`."""
    out, issues = [], []
    tenses = {v: k for k, v in TENSE_A.items()}
    for part in [p.strip() for p in text.split(";") if p.strip()]:
        fields = re.findall(r"(?<![A-Za-z0-9_.>^+&|])([A-Z]+)=(\S+)", part)
        if not fields:
            issues.append(f"no field like S=, V=, O= in '{part}' (fields: {' '.join(FIELDS)})")
            continue
        cl = ClauseV2()
        obls, time = [], []
        for fld, val in fields:
            if fld not in FIELDS:
                issues.append(f"unknown field '{fld}=' (fields: {' '.join(FIELDS)})")
                continue
            if fld in ("S", "O", "P", "Y") or fld in CASE_FIELDS or fld in ("L", "T"):
                x = nominal_from_text(val, spell_name, issues)
                if fld == "S":
                    if isinstance(x, Name):
                        cl.subj_name, cl.subj_ph = x.name, x.ph
                    else:
                        cl.subj = x
                elif fld == "O":
                    if isinstance(x, Name):
                        cl.obj_name, cl.obj_ph = x.name, x.ph
                    else:
                        cl.obj = x
                elif fld == "P":
                    cl.pred = x
                elif fld == "Y":
                    cl.voc = x
                elif fld == "T":
                    time.append(x)
                elif fld == "L" and type(x) is NP and not x.adj and not x.plural and not cl.loc and not obls:
                    cl.loc = x.noun                                   # a plain place, as in v1
                else:
                    obls.append((fld, x))
            elif fld == "V":
                vp = val.split(".")
                cl.verb = vp[0]
                mods = []
                for t in vp[1:]:
                    if t == "NEG":
                        cl.neg = True
                    elif t in tenses:
                        cl.tense = tenses[t]
                    elif t in VERB_MODS:
                        mods.append(t)
                    else:
                        issues.append(f"'{t}' in V= is not PRES, PAST, FUT, NEG or one of {', '.join(VERB_MODS)}")
                cl.mods = tuple(mods)
            elif fld == "M":
                cl.adv = val
            elif fld == "A":
                cl.adjs = tuple(val.split("+"))
            elif fld == "I":
                cl.intj = tuple(val.split("+"))
            elif fld == "C":
                if val not in CONNECTORS:
                    issues.append(f"unknown joining word C={val} ({', '.join(CONNECTORS)})")
                cl.link = val
        cl.obls, cl.time = tuple(obls), tuple(time)
        out.append(cl)
    return out, issues
