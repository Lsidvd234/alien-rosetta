"""Representative English for the v2 language (the v2 counterpart of train/represent.py; test-agnostic training data).

Every example is built meaning first: an event (conlang/v2.py ClauseV2) is sampled, then English is written for it by
sampling, independently, one variant from each FAMILY below, so the analysis and the glyph answer are right by
construction. The families follow descriptive grammar of everyday English (kinds of utterance: statements,
questions, commands, fragments, greetings; voice; information structure; tense, aspect and modality; negation;
determiners and owners; describing words and degree; relation phrases; how / when / where words; joining events;
orthography; lexical choice), not any test set.

Vocabulary: content words are sampled by their English frequency, tempered (weight = 10^(0.5 * (zipf - 3))), so
common words come often and every word can still appear; coverage is reported by `coverage()`.

Honest measurement, as in E1: before any training, HELD_OUT is drawn at random (fixed seed) from the non-canonical
variants of the HOLDABLE families: those where every variant has the same meaning and only the English differs
(voice, word order, how the future / a question / a negation / an owner is written, ...). Variants that bring a word
or ending of their own (which relation word, which modal, which determiner, question words) are never held out: a
model cannot know the glyphs of something it never saw. 20% of the synonym pairs are held out too, and counts beyond
the training range (5-6 describing words, 5 events, 3 relation phrases) are measured as extrapolation.

E1b-style (--cot): the target is "<analysis> => <glyphs>", analysis in the v2 format (conlang/v2.py docstring).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from conlang.frames import NP, PRONOUNS
from conlang.v2 import NPX, PLURALIA, ClauseV2, Coord, Name, RepV2, VocabV2

from train.augment import IRREGULAR_PARTICIPLE, REFUSAL, SYN_TABLES, load_replay, refusal

FORMS = json.loads((Path(__file__).resolve().parents[1] / "conlang" / "data" / "english_forms.json")
                   .read_text(encoding="utf-8"))
HELD_OUT_SEED = 20261014

# family -> variants; the FIRST variant is canonical (never held out). HOLDABLE families: same meaning, other English.
FAMILIES: dict[str, list[str]] = {
    # what is said
    "kind": ["statement", "yes/no question", "wh question", "command", "fragment", "greeting", "let's"],
    "predicate": ["transitive", "intransitive", "give-like", "be + describing", "be + noun", "be + place",
                  "there is"],
    "aspect": ["simple", "progressive", "perfect"],
    "modal": ["none", "can", "could", "must", "have to", "should", "may", "might", "would"],
    "subject": ["noun", "pronoun", "name", "joined"],
    "object": ["noun", "pronoun", "name", "joined"],
    "det": ["the", "a/an", "this", "that", "number", "quantifier", "my", "owner", "none"],
    "degree": ["none", "very", "too", "so", "really", "-er / more", "-est / most"],
    "relation": ["L", "ON", "UNDER", "TO", "FROM", "WITH", "FOR", "NEAR", "BEHIND", "FRONT", "ABOUT", "WITHOUT",
                 "BEFORE", "AFTER", "THROUGH", "AROUND", "BETWEEN", "DURING", "UNTIL", "SINCE", "AGAINST"],
    "how": ["manner", "time", "place", "frequency", "focus", "degree + manner", "a lot"],
    "wh": ["what (object)", "who (object)", "who (doer)", "what (doer)", "which + noun", "how many", "where", "when",
           "why", "how", "who / what (is)", "how + describing", "whose", "who ... to (stranded)"],
    "greeting": ["hello", "thanks", "thank you", "please", "yes / no", "sorry", "bye", "exclamation"],
    "link": ["because", "if", "when", "while", "before", "after", "until", "although", "unless", "but", "so",
             "then", "or", "that", "to"],
    # how it is written (HOLDABLE)
    "voice": ["active", "passive", "passive without doer"],
    "order": ["canonical", "place fronted", "object topicalised", "subject cleft", "object cleft"],
    "future": ["will", "'ll", "be going to", "shall"],
    "present": ["simple", "do-emphatic"],
    "past": ["simple", "did-emphatic"],
    "negation": ["full", "contracted"],
    "question form": ["inverted", "statement + ?"],
    "owner form": ["'s", "of"],
    "give form": ["to + receiver", "receiver first"],
    "relation form": ["main word", "other word"],
    "how position": ["end", "start", "before the verb"],
    "vocative": ["none", "start", "end"],
    "adj_sep": ["space", "comma", "and"],
    "join": ["full stop", "comma and", "semicolon", "and"],
    "link order": ["between", "first"],
    "shared doer": ["repeated", "left out"],
    "case": ["lower", "sentence", "title", "upper"],
    "end": ["none", "full stop", "exclamation"],
    "spacing": ["single", "double"],
    "noise": ["none", "swap", "drop", "insert", "substitute"],
    "lexical": ["plain", "synonym"],
}
HOLDABLE = ["voice", "order", "future", "present", "past", "negation", "question form", "owner form", "give form",
            "relation form", "how position", "vocative", "adj_sep", "join", "link order", "shared doer", "case", "end",
            "spacing", "noise", "lexical"]
# sampling weights (canonical share for surface families; how often each kind of utterance comes)
CANONICAL_SHARE = {"case": 0.5, "end": 0.5, "spacing": 0.8, "noise": 0.7, "lexical": 0.8, "vocative": 0.8,
                   "aspect": 0.6, "modal": 0.6, "degree": 0.6, "det": 0.3, "voice": 0.6, "order": 0.6}
WEIGHTS = {"kind": [0.40, 0.12, 0.14, 0.13, 0.09, 0.09, 0.03],
           "predicate": [0.34, 0.20, 0.06, 0.14, 0.10, 0.10, 0.06]}
ADJ_COUNTS = {0: 0.45, 1: 0.30, 2: 0.15, 3: 0.07, 4: 0.03}      # describing words per noun phrase, in training
EVENT_COUNTS = {1: 0.60, 2: 0.25, 3: 0.10, 4: 0.05}               # events per input, in training
REL_COUNTS = {0: 0.55, 1: 0.35, 2: 0.10}                          # relation phrases per event, in training
HOW_COUNTS = {0: 0.60, 1: 0.32, 2: 0.08}                          # how-words per event, in training
P_LINK = 0.35                                                     # a later event starts with a joining word
P_TIME = 0.08                                                     # a time phrase (every day, last night)

REL_WORDS = {"L": ["in", "inside", "within", "at"], "ON": ["on", "onto", "upon", "above", "over"],
             "UNDER": ["under", "below", "beneath", "underneath"], "TO": ["to", "towards", "toward", "into"],
             "FROM": ["from", "out of", "away from"], "WITH": ["with"], "FOR": ["for", "instead of"],
             "NEAR": ["near", "beside", "next to", "close to"], "BEHIND": ["behind"], "FRONT": ["in front of"],
             "ABOUT": ["about"], "WITHOUT": ["without"], "BEFORE": ["before"], "AFTER": ["after"],
             "THROUGH": ["through", "across"], "AROUND": ["around"], "BETWEEN": ["between", "among"],
             "DURING": ["during"], "UNTIL": ["until", "till"], "SINCE": ["since"], "AGAINST": ["against"]}
SUBJ_EN = {"1sg": "I", "2sg": "you", "3sgm": "he", "3sgf": "she", "3sgn": "it", "1pl": "we", "2pl": "you all",
           "3pl": "they"}
OBJ_EN = {"1sg": "me", "2sg": "you", "3sgm": "him", "3sgf": "her", "3sgn": "it", "1pl": "us", "2pl": "you all",
          "3pl": "them"}
POSS_EN = {"1sg": "my", "2sg": "your", "3sgm": "his", "3sgf": "her", "3sgn": "its", "1pl": "our", "2pl": "your",
           "3pl": "their"}
MINE_EN = {"1sg": "mine", "2sg": "yours", "3sgm": "his", "3sgf": "hers", "1pl": "ours", "3pl": "theirs"}
NUMBERS = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twelve", "twenty", "hundred"]
QUANT_PL = ["many", "few", "several", "some", "all", "both", "no"]
QUANT_SG = ["every", "each", "another", "no"]
QUANT_MASS = ["some", "no", "a lot of"]
MODAL_EN = {"can": ("can", "CAN", "present"), "could": ("could", "CAN", "past"), "must": ("must", "MUST", "present"),
            "should": ("should", "SHOULD", "present"), "may": ("may", "MAY", "present"),
            "might": ("might", "MAY", "present"), "would": ("would", "WOULD", "present")}
MODAL_NEG = {"can": "can't", "could": "couldn't", "must": "mustn't", "should": "shouldn't", "would": "wouldn't"}
INTJ_EN = {"hello": ["hello", "hi", "hey"], "thanks": ["thanks"], "yes / no": ["yes", "no", "okay"],
           "sorry": ["sorry"], "bye": ["bye", "goodbye"], "exclamation": ["wow", "oh", "oops", "ah", "ouch", "hurray"],
           "please": ["please"]}
TIME_NOUNS = ["day", "night", "morning", "evening", "afternoon", "week", "month", "year", "weekend", "summer",
              "winter"]
TIME_DETS = ["every", "each", "last", "next", "this"]
HOW = {"time": ["today", "tomorrow", "yesterday", "tonight", "now", "later", "soon", "again"],
       "place": ["here", "there", "upstairs", "downstairs", "abroad", "away"],
       "frequency": ["always", "often", "sometimes", "usually", "never"],
       "focus": ["also", "only", "already", "still"]}
THINK_VERBS = ["think", "know", "say", "hope", "believe", "hear", "feel", "see"]
WANT_VERBS = ["want", "need", "try", "like", "love", "hope", "decide", "plan", "forget", "remember", "start"]
DITRANSITIVE = ["give", "send", "show", "tell", "bring", "teach", "buy", "pass", "offer", "lend", "sell", "pay",
                "write", "hand", "cook", "throw", "promise", "feed", "serve"]
FUNCTION_LIKE = (set(SUBJ_EN.values()) | set(OBJ_EN.values()) | set(POSS_EN.values()) | set(MINE_EN.values())
                 | {"the", "a", "an", "this", "that", "these", "those", "what", "who", "which", "where", "when", "why",
                    "how", "will", "shall", "can", "may", "must", "might", "would", "could", "should", "do", "does",
                    "did", "have", "has", "had", "be", "is", "am", "are", "was", "were", "not", "no", "yes", "and",
                    "or", "but", "so", "if", "then", "to", "of", "in", "on", "at", "by", "for", "with", "from",
                    "there", "here", "let", "please", "all", "some", "many", "much", "few", "one", "every", "each",
                    "also", "too", "very", "more", "most", "less", "up", "down", "out", "off", "over", "back",
                    "home", "away", "around", "about", "after", "before", "since", "until", "while", "because",
                    "that", "than", "like", "as", "right", "left", "near", "inside", "outside", "time", "today",
                    "lot", "lots", "plenty", "couple"}
                 | set(NUMBERS) | {"zero", "eleven", "thirteen", "hundred", "thousand", "million", "billion", "half",
                                   "second", "third", "first", "own", "such", "other", "another", "both", "several"})


def draw_held_out(frac: float = 0.2) -> set[tuple[str, str]]:
    pool = [(f, v) for f in HOLDABLE for v in FAMILIES[f][1:]]
    rng = random.Random(HELD_OUT_SEED)
    return set(rng.sample(pool, round(frac * len(pool))))


HELD_OUT = draw_held_out()


def set_held_out(frac: float) -> None:
    """Change the held-out share (0 = nothing held out, for a final model). Same seed, so the draw is fixed."""
    HELD_OUT.clear()
    HELD_OUT.update(draw_held_out(frac) if frac > 0 else set())


AUX_WORDS = {"will", "shall", "would", "can", "could", "must", "should", "may", "might", "do", "does", "did", "have",
             "has", "had", "am", "is", "are", "was", "were", "won't", "can't", "couldn't", "mustn't", "shouldn't",
             "wouldn't", "don't", "doesn't", "didn't", "haven't", "hasn't", "hadn't", "isn't", "aren't", "wasn't",
             "weren't"}


def plural_of(w: str) -> str:
    if w.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


def weight(z: float) -> float:
    return 10 ** (0.5 * (z - 3))


def ing_of(base: str) -> str:
    f = FORMS["ing"].get(base)
    if f:
        return f
    if base.endswith("ie"):
        return base[:-2] + "ying"
    if base.endswith("e") and not base.endswith(("ee", "ye", "oe")):
        return base[:-1] + "ing"
    return base + "ing"


class GenV2:
    """Samples v2 meanings and writes English for them."""

    def __init__(self, preset: str, script: str, seed: int):
        self.rep = RepV2(preset, script, seed)
        self.v: VocabV2 = self.rep.v
        self.seed = seed
        lang = self.rep.lang
        z = FORMS["zipf"]
        v = self.v
        ok = (lambda w: w.isalpha() and w not in FUNCTION_LIKE)
        # nouns: (english singular, english plural or None, key, mass)
        nouns = {}
        for n, (pl, _) in v.noun_info.items():
            if ok(n) and n != "people":
                k = v.key(n, "N")
                nouns[n] = (n, pl if pl and pl != n else None, k, k in v.mass)
        for (w, pos), d in v.derived.items():
            if pos == "N" and ok(w):
                nouns[w] = (w, d.get("plural"), v.key(w, "N"), False)
        for w, c in v.compounds.items():
            if ok(w):
                nouns[w] = (w, c.get("plural"), v.key(w, "N"), False)
        self.nouns = [x for x in nouns.values() if self._renders(x[2], "N")]
        self.noun_w = [weight(z.get(x[0], 2.0)) for x in self.nouns]
        self.people = [x for x in self.nouns if v.gender(x[2]) in ("m", "f", "h") and not x[3]]
        self.people_w = [weight(z.get(x[0], 2.0)) for x in self.people]
        self.places = [(w, FORMS.get("plural", {}).get(w) or plural_of(w), w, False) for w in v.locations if ok(w)]
        self.places_w = [weight(z.get(x[0], 2.0)) for x in self.places]
        adjs = {a: v.key(a, "A") for a in list(v.adjectives) + v.x_adjectives if ok(a)}
        for (w, pos), d in v.derived.items():
            if pos == "A" and ok(w):
                adjs[w] = v.key(w, "A")
        self.adjs = [(w, k) for w, k in adjs.items() if self._renders(k, "A")]
        self.adj_w = [weight(z.get(w, 2.0)) for w, _ in self.adjs]
        self.adj_words = set(adjs)
        verbs = {}                                                   # base -> (key, third, past, participle, trans)
        trans = set(v.transitive) | {"have", "say", "tell", "get"}
        for lemma, (base, third, past) in v.verb_info.items():
            if ok(base) and lemma not in ("do", "let", "rain", "snow", "become", "seem", "appear", "be"):
                part = (FORMS["participle"].get(base) or IRREGULAR_PARTICIPLE.get(lemma) or v.participle.get(lemma)
                        or past)
                verbs[base] = (v.key(lemma, "V"), third, past, part, lemma in trans)
        for (w, pos), d in v.derived.items():
            if pos == "V" and ok(w) and d.get("third") and d.get("past"):
                verbs[w] = (v.key(w, "V"), d["third"], d["past"], FORMS["participle"].get(w) or d["past"], True)
        self.verbs = [(b,) + x for b, x in verbs.items() if self._renders(x[0], "V")]
        self.verb_w = [weight(z.get(x[0], 2.0)) for x in self.verbs]
        self.tverbs = [x for x in self.verbs if x[5]]
        self.tverb_w = [weight(z.get(x[0], 2.0)) for x in self.tverbs]
        self.iverbs = [x for x in self.verbs if not x[5]]
        self.iverb_w = [weight(z.get(x[0], 2.0)) for x in self.iverbs]
        self.verb_by_base = {x[0]: x for x in self.verbs}
        roots_r = set(v.adverbs) | set(v.x_adverbs)               # a root adverb is read as itself (suddenly)
        manner = [(d["word"], v.key(w, pos)) for (w, pos), d in v.derived.items() if pos == "R"
                  and d["word"] not in roots_r]
        manner += [(w, w) for w in v.adverbs if w in ("fast", "well", "hard", "alone", "together", "slowly")]
        self.manner = [(w, k) for w, k in manner if self._renders(k, "R")]
        self.manner_w = [weight(z.get(w, 2.0)) for w, _ in self.manner]
        self.language_words = (set(nouns) | {x[1] for x in nouns.values() if x[1]} | set(v.locations) | set(adjs)
                               | set(verbs) | {x[1] for x in verbs.values()} | {x[2] for x in verbs.values()}
                               | set(v.adverbs) | set(v.x_adverbs) | {w for w, _ in manner})
        self.all_syn = {k: {**SYN_TABLES["train"][k], **SYN_TABLES["heldout"][k]} for k in "NLAV"}
        self.lexicon_sounds = {"".join(r) for r in lang.roots.values()}

    def _renders(self, key: str, pos: str) -> bool:
        try:
            self.rep.lang.stem(key, pos, __import__("conlang.grammar", fromlist=["Realization"]).Realization(words=[]))
            return True
        except KeyError:
            return False

    # --- sampling helpers -------------------------------------------------------------------------------------
    def pick(self, rng, family: str, force: dict | None = None, allow_held: bool = False) -> str:
        if force and family in force:
            return force[family]
        vs = FAMILIES[family]
        ws = WEIGHTS.get(family)
        if ws is None:
            share = CANONICAL_SHARE.get(family)
            ws = ([share] + [(1 - share) / (len(vs) - 1)] * (len(vs) - 1)) if share is not None else [1] * len(vs)
        pairs = [(v, w) for v, w in zip(vs, ws) if allow_held or (family, v) not in HELD_OUT]
        return rng.choices([p[0] for p in pairs], weights=[p[1] for p in pairs])[0]

    def name(self, rng) -> Name:
        cons = ["k", "c", "t", "n", "p", "m", "y", "r", "l", "v"]
        vowels = ["a", "e", "i", "o", "u", "a", "i", "u", "aa", "ii", "uu"]
        while True:
            ph = []
            for _ in range(rng.choice([2, 2, 3])):
                ph += [rng.choice(cons), rng.choice(vowels)]
            if rng.random() < 0.5:
                c = rng.choice(["n", "m", "l", "r", ""])
                if c:
                    ph.append(c)
            s = "".join(ph)
            if s not in self.lexicon_sounds:
                return Name(s.capitalize(), tuple(ph), rng.choice("mf"))

    # --- noun phrases: (nominal for the meaning, English, person/number for agreement) ---------------------------
    def np(self, rng, ch: dict, role: str, n_adj: int, people: bool = False, place: bool = False,
           allow_owner: bool = True, plural: bool | None = None, det: str | None = None) -> tuple:
        pool, w = (self.places, self.places_w) if place else ((self.people, self.people_w) if people
                                                                else (self.nouns, self.noun_w))
        sg, pl_form, key, mass = rng.choices(pool, weights=w)[0]
        if LOAN["rate"] and not (people or place) and rng.random() < LOAN["rate"]:
            sg, pl_form = rng.choice(LOAN[LOAN["pool"]])              # a borrowed English noun (LOAN.pizza)
            key, mass = f"LOAN.{sg}", False
        plural = (rng.random() < 0.4) if plural is None else plural
        plural = plural and pl_form is not None and not mass
        det = det or ch["det"]
        if sg in TIME_NOUNS + ["midnight", "noon", "today", "tonight"] and det in ("this", "that", "quantifier"):
            det = "the"                                              # 'this morning' is a time phrase
        if det == "owner" and not allow_owner:
            det = "the"
        if det == "my" and not allow_owner:
            det = "the"
        adj_keys, adj_en = [], []
        picked = rng.choices(self.adjs, weights=self.adj_w, k=n_adj) if n_adj else []
        picked = list(dict.fromkeys(picked))
        for i, (a, ak) in enumerate(picked):
            deg = ch["degree"] if i == 0 else "none"
            k, e = self.degree(rng, a, ak, deg)
            adj_keys += k
            adj_en.append(e)
        sep = ch["adj_sep"]
        if len(adj_en) >= 2 and sep == "comma":
            mod = ", ".join(adj_en)
        elif len(adj_en) >= 2 and sep == "and":
            mod = " ".join(adj_en[:-1]) + " and " + adj_en[-1]
        else:
            mod = " ".join(adj_en)
        noun_en = pl_form if plural else sg
        dkeys, det_en, poss, poss_en, of_en = [], "", None, "", ""
        if det == "a/an" and (plural or mass):
            det = "the"
        if det == "none" and not (plural or mass):
            det = "the"
        if det == "the":
            det_en = "the"
        elif det == "a/an":
            first = (mod or noun_en)[0]
            det_en = "an" if first in "aeiou" else "a"
        elif det in ("this", "that"):
            dkeys = [det]
            det_en = {"this": "these", "that": "those"}[det] if plural else det
        elif det == "number":
            if plural:
                n = rng.choice(NUMBERS)
                dkeys, det_en = [n], n
            elif not mass:
                dkeys, det_en = ["one"], "one"
            else:
                det_en = "the"
        elif det == "quantifier":
            q = rng.choice(QUANT_MASS if mass else (QUANT_PL if plural else QUANT_SG))
            if q == "a lot of":
                dkeys, det_en = ["much"], "a lot of"
            else:
                dkeys, det_en = [q], q
        elif det == "my":
            poss = rng.choice([k for k in POSS_EN if k != "2pl"])
            poss_en = POSS_EN[poss]
        elif det == "owner":
            if rng.random() < 0.5:
                o = self.name(rng)
                o_en = o.name
            else:
                on, o_en, _ = self.np(rng, dict(ch, degree="none", det="the"), "obj", 0, people=True,
                                      allow_owner=False, plural=False)
                o = on
            poss = o
            if ch["owner form"] == "of" and not isinstance(o, Name):
                of_en = " of " + o_en
                det_en = "the"
            else:
                poss_en = o_en + "'s"
        words = [x for x in (det_en, poss_en, mod, noun_en) if x]
        en = " ".join(words) + of_en
        adj = " ".join(dkeys + adj_keys) or None
        nominal = NPX(key, plural, adj, poss) if poss is not None else NP(key, plural, adj)
        if key in PLURALIA:
            nominal = NP(key, True, adj) if poss is None else NPX(key, True, adj, poss)
        num = "pl" if (plural or key in PLURALIA) else "sg"
        return nominal, en, (3, num)

    def degree(self, rng, a: str, ak: str, deg: str) -> tuple[list[str], str]:
        if deg in ("very", "too", "so", "really"):
            return [deg, ak], f"{deg} {a}"
        if deg == "-er / more":
            if a in FORMS["comparative"] and ">" not in ak:
                return [f"{ak}>MORE"], FORMS["comparative"][a]
            return ["more", ak], f"more {a}"
        if deg == "-est / most":
            if a in FORMS["superlative"] and ">" not in ak:
                return [f"{ak}>MOST"], FORMS["superlative"][a]
            return ["most", ak], f"most {a}"
        return [ak], a

    def adj_count(self, rng, counts) -> int:
        return rng.choices(list(counts), weights=list(counts.values()))[0]

    def nominal(self, rng, ch: dict, role: str, kind: str, counts, **kw) -> tuple:
        """A doer / thing done to: noun phrase, pronoun, name, or two joined. -> (nominal, English, (person, n))."""
        if kind == "pronoun":
            p = rng.choice(list(PRONOUNS))
            _, per, num, _ = PRONOUNS[p]
            return p, (SUBJ_EN if role == "subj" else OBJ_EN)[p], (per, num)
        if kind == "name":
            n = self.name(rng)
            return n, n.name, (3, "sg")
        if kind == "joined":
            a = self.nominal(rng, ch, role, rng.choice(["noun", "name", "pronoun"]), {0: 0.8, 1: 0.2}, **kw)
            b = self.nominal(rng, ch, role, rng.choice(["noun", "name"]), {0: 0.8, 1: 0.2}, **kw)
            conj = "and" if rng.random() < 0.85 else "or"
            if isinstance(a[0], Coord) or isinstance(b[0], Coord):
                return a
            per = min(a[2][0], b[2][0]) if conj == "and" else b[2][0]
            num = "pl" if conj == "and" else b[2][1]
            return Coord((a[0], b[0]), conj), f"{a[1]} {conj} {b[1]}", (per, num)
        return self.np(rng, ch, role, self.adj_count(rng, counts), **kw)

    # --- verbs ----------------------------------------------------------------------------------------------------
    def verb_group(self, ch: dict, vb: tuple, tense: str, aspect: str, modal: str, neg: bool, png, subj_en: str,
                   used: set) -> tuple[list[str], tuple, str] | None:
        """English verb group for a statement. -> (words, mods, tense)."""
        base, key, third, past, part, _ = vb
        per, num = png
        s3 = per == 3 and num == "sg"
        contr = ch["negation"] == "contracted"
        be_pres = "am" if (per == 1 and num == "sg") else ("is" if s3 else "are")
        aspect_mods = {"simple": (), "progressive": ("PROG",), "perfect": ("PERF",)}[aspect]
        tail = {"simple": [base], "progressive": ["be", ing_of(base)], "perfect": ["have", part]}[aspect]
        if modal != "none":
            if modal == "have to":
                if neg or aspect != "simple":
                    modal = "must"
                else:
                    have = {"present": "has" if s3 else "have", "past": "had", "future": None}[tense]
                    if have is None:
                        return ["will", "have", "to", base], ("MUST",), "future"
                    return [have, "to", base], ("MUST",), tense
            word, m, mt = MODAL_EN[modal]
            if neg:
                head = [MODAL_NEG[modal]] if (contr and modal in MODAL_NEG) else [word, "not"]
            else:
                head = [word]
            return head + tail, (m,) + aspect_mods, mt
        if tense == "future":
            form = ch["future"]
            if form == "be going to" and aspect == "simple":
                used.add(("future", form))
                return ([be_pres, "not", "going", "to", base] if neg else [be_pres, "going", "to", base]), (), "future"
            if form == "shall" and per == 1:
                used.add(("future", form))
                return (["shall", "not"] if neg else ["shall"]) + tail, aspect_mods, "future"
            if form == "'ll" and not neg and subj_en.lower() in ("i", "you", "he", "she", "it", "we", "they"):
                used.add(("future", form))
                return ["'ll"] + tail, aspect_mods, "future"
            if neg:
                return (["won't"] if contr else ["will", "not"]) + tail, aspect_mods, "future"
            return ["will"] + tail, aspect_mods, "future"
        if aspect == "progressive":
            be = be_pres if tense == "present" else ("was" if (s3 or (per == 1 and num == "sg")) else "were")
            neg_w = ([be + "n't"] if contr and be != "am" else [be, "not"]) if neg else [be]
            return neg_w + [ing_of(base)], ("PROG",), tense
        if aspect == "perfect":
            have = ("has" if s3 else "have") if tense == "present" else "had"
            neg_w = ([have + "n't"] if contr else [have, "not"]) if neg else [have]
            return neg_w + [part], ("PERF",), tense
        if tense == "past":
            if neg:
                return (["didn't", base] if contr else ["did", "not", base]), (), "past"
            if ch["past"] == "did-emphatic":
                used.add(("past", "did-emphatic"))
                return ["did", base], (), "past"
            return [past], (), "past"
        do = "does" if s3 else "do"
        if neg:
            return ([do + "n't", base] if contr else [do, "not", base]), (), "present"
        if ch["present"] == "do-emphatic":
            used.add(("present", "do-emphatic"))
            return [do, base], (), "present"
        return [third if s3 else base], (), "present"

    def be_group(self, ch: dict, tense: str, modal: str, aspect: str, neg: bool, png) -> tuple[list[str], tuple, str]:
        per, num = png
        s3 = per == 3 and num == "sg"
        contr = ch["negation"] == "contracted"
        if modal not in ("none", "have to"):
            word, m, mt = MODAL_EN[modal]
            head = ([MODAL_NEG[modal]] if (contr and modal in MODAL_NEG) else [word, "not"]) if neg else [word]
            return head + ["be"], (m,), mt
        if tense == "future":
            return ((["won't"] if contr else ["will", "not"]) if neg else ["will"]) + ["be"], (), "future"
        if aspect == "perfect":
            have = ("has" if s3 else "have") if tense == "present" else "had"
            return (([have + "n't"] if contr else [have, "not"]) if neg else [have]) + ["been"], ("PERF",), tense
        if tense == "present":
            be = "am" if (per == 1 and num == "sg") else ("is" if s3 else "are")
        else:
            be = "was" if (s3 or (per == 1 and num == "sg")) else "were"
        if neg:
            return ([be + "n't"] if contr and be != "am" else [be, "not"]), (), tense
        return [be], (), tense

    # --- relation phrases, how-words, time ----------------------------------------------------------------------
    def relation(self, rng, ch: dict, fld: str | None = None) -> tuple[str, object, str]:
        fld = fld or rng.choice([f for f in FAMILIES["relation"] if f != "THAN"])
        words = REL_WORDS[fld]
        word = words[0] if (ch["relation form"] == "main word" or len(words) == 1) else rng.choice(words[1:])
        r = rng.random()
        if fld == "L" and r < 0.7:
            x, en, _ = self.np(rng, dict(ch, det=rng.choice(["the", "the", "my", "this", "a/an"])), "obj",
                               self.adj_count(rng, {0: 0.8, 1: 0.2}), place=True)
        elif r < 0.15:
            x = rng.choice(list(PRONOUNS))
            en = OBJ_EN[x]
        elif r < 0.25:
            x = self.name(rng)
            en = x.name
        else:
            d = ch["det"] if ch["det"] != "none" else "the"
            x, en, _ = self.np(rng, dict(ch, det=d), "obj", self.adj_count(rng, {0: 0.8, 1: 0.2}))
        return fld, x, f"{word} {en}"

    def how(self, rng, ch: dict) -> tuple[list[str], str, str]:
        """-> (keys, English, class)."""
        cls = rng.choice(FAMILIES["how"])
        if cls in HOW:
            w = rng.choice(HOW[cls])
            return [w], w, cls
        if cls == "a lot":
            return ["much"], "a lot", cls
        w, k = rng.choices(self.manner, weights=self.manner_w)[0]
        if cls == "degree + manner":
            d = rng.choice(["very", "really", "quite"])
            return [d, k], f"{d} {w}", cls
        return [k], w, "manner"

    def time_phrase(self, rng) -> tuple[object, str]:
        n = rng.choice(TIME_NOUNS)
        d = rng.choice(TIME_DETS)
        if d in ("last", "next"):
            return NP(n, False, d), f"{d} {n}"
        return NP(n, False, d), f"{d} {n}"

    # --- one event ------------------------------------------------------------------------------------------------
    def event(self, rng, ch: dict, kind: str, counts, used: set, link_verb: str | None = None) -> tuple | None:
        """-> (ClauseV2, English words before/after, flags). English is one string; flags: dict."""
        pred = ch["predicate"]
        cl = ClauseV2()
        tense = rng.choice(["present", "past", "future"])
        neg = rng.random() < 0.25
        aspect, modal = ch["aspect"], ch["modal"]
        rels = [self.relation(rng, ch) for _ in range(rng.choices(list(REL_COUNTS), weights=list(REL_COUNTS.values()))[0])]
        hows = [self.how(rng, ch) for _ in range(rng.choices(list(HOW_COUNTS), weights=list(HOW_COUNTS.values()))[0])]
        hows = [h for i, h in enumerate(hows) if not any(h[0] == g[0] for g in hows[:i])]
        if any(h[2] == "frequency" and h[0] == ["never"] for h in hows):
            neg = False                                              # 'never' already says not
        times = [self.time_phrase(rng)] if rng.random() < P_TIME else []
        if kind == "fragment":
            return self.fragment(rng, ch, counts, used)
        if kind == "greeting":
            return self.greeting(rng, ch, counts, used)
        if kind == "command":
            return self.command(rng, ch, counts, used, rels, hows)
        if kind == "let's":
            vb = rng.choices(self.verbs, weights=self.verb_w)[0]
            obj = self.nominal(rng, ch, "obj", "noun", counts) if vb[5] else None
            cl = ClauseV2(subj="1pl", verb=vb[1], mods=("LET",), obj=obj[0] if obj else None)
            en = ["let's" if rng.random() < 0.7 else "let us", vb[0]] + ([obj[1]] if obj else [])
            return self.finish(rng, ch, cl, en, rels, hows, [], used, verb_at=1)
        if kind == "wh question":
            return self.wh(rng, ch, counts, used, tense, neg, aspect, modal)
        # statements and yes/no questions
        sk = ch["subject"]
        if link_verb:
            sk = rng.choice(["pronoun", "noun", "name"])
        subj, subj_en, png = self.nominal(rng, ch, "subj", sk, counts)
        q = kind == "yes/no question"
        if pred in ("be + describing", "be + noun", "be + place", "there is"):
            return self.copula(rng, ch, counts, used, pred, subj, subj_en, png, tense, neg, aspect, modal, q, rels,
                               hows, times)
        if link_verb:
            vb = self.verb_by_base.get(link_verb)
            if vb is None:
                return None
            trans = False
        elif pred == "give-like":
            cands = [self.verb_by_base[b] for b in DITRANSITIVE if b in self.verb_by_base]
            vb = rng.choice(cands)
            trans = True
        else:
            trans = pred == "transitive"
            vb = rng.choices(self.tverbs if trans else self.iverbs, weights=self.tverb_w if trans else self.iverb_w)[0]
        obj = obj_en = None
        if trans:
            ok_ = ch["object"]
            obj, obj_en, _ = self.nominal(rng, ch, "obj", ok_, counts)
        receiver = None
        if pred == "give-like":
            r = self.nominal(rng, ch, "obj", rng.choice(["pronoun", "name", "noun"]), {0: 0.9, 1: 0.1}, people=True)
            receiver = r
        # passive
        voice = ch["voice"]
        if voice == "passive without doer" and vb[4] in self.adj_words:
            voice = "active"                                         # 'are made' also reads as be + describing
        if voice != "active" and trans and receiver is None and not q and aspect != "progressive" \
                and not (modal == "have to"):
            res = self.passive(rng, ch, vb, subj, subj_en, png, obj, obj_en, tense, neg, aspect, modal, voice, used)
            if res:
                cl, en = res
                return self.finish(rng, ch, cl, en, rels, hows, times, used, verb_at=None)
        words, mods, tense2 = self.verb_group(ch, vb, tense, aspect, modal, neg, png, subj_en, used)
        cl = ClauseV2(verb=vb[1], tense=tense2, neg=neg, mods=mods)
        self.set_subj(cl, subj)
        if obj is not None:
            self.set_obj(cl, obj)
        extra_obls = []
        tail = [obj_en] if obj_en else []
        if receiver is not None:
            if ch["give form"] == "receiver first" and obj_en:
                used.add(("give form", "receiver first"))
                tail = [receiver[1], obj_en]
                extra_obls = [("TO", receiver[0])]
            else:
                rels = [("TO", receiver[0], "to " + receiver[1])] + rels
        if words[0] == "'ll":
            subj_en, words = subj_en + "'ll", words[1:]
        if q:
            en = self.invert(subj_en, words, tail, ch, used)
            if en is None:
                return None
            cl.mods = cl.mods + ("Q",)
            return self.finish(rng, ch, cl, en, rels, hows, times, used, verb_at=None, extra_obls=extra_obls,
                               question=True)
        order = ch["order"]
        if order in ("subject cleft", "object cleft") and tense2 in ("present", "past", "future") and not modal != "none" \
                and aspect == "simple" and not subj_en.endswith("'ll") and receiver is None:
            cop = {"present": "is", "past": "was", "future": "will be"}[tense2]
            if order == "subject cleft":
                used.add(("order", "subject cleft"))
                en = ["it", cop, subj_en, "that"] + words + tail
                return self.finish(rng, ch, cl, en, rels, hows, times, used, verb_at=None, no_front=True)
            if obj_en:
                used.add(("order", "object cleft"))
                en = ["it", cop, obj_en, "that", subj_en] + words
                return self.finish(rng, ch, cl, en, rels, hows, times, used, verb_at=None, no_front=True)
        if order == "object topicalised" and obj_en and receiver is None and not rels and not hows and not times:
            used.add(("order", "object topicalised"))
            en = [obj_en + ",", subj_en] + words
            return self.finish(rng, ch, cl, en, [], [], [], used, verb_at=None, no_front=True)
        en = [subj_en] + words + tail
        return self.finish(rng, ch, cl, en, rels, hows, times, used, verb_at=1, extra_obls=extra_obls)

    @staticmethod
    def set_subj(cl: ClauseV2, x) -> None:
        if isinstance(x, Name):
            cl.subj_name, cl.subj_ph, cl.subj_gender = x.name, x.ph, x.gender
        else:
            cl.subj = x

    @staticmethod
    def set_obj(cl: ClauseV2, x) -> None:
        if isinstance(x, Name):
            cl.obj_name, cl.obj_ph = x.name, x.ph
        else:
            cl.obj = x

    def invert(self, subj_en: str, words: list[str], tail: list[str], ch: dict, used: set) -> list[str] | None:
        """Yes/no question from a statement verb group: do you like tea / is she running / can you swim."""
        if ch["question form"] == "statement + ?" and ch.get("_last", True):
            used.add(("question form", "statement + ?"))
            return [subj_en] + words + tail
        if subj_en.endswith("'ll"):
            subj_en, words = subj_en[:-3], ["will"] + words
        w0 = words[0]
        aux_like = {"am", "is", "are", "was", "were", "will", "shall", "can", "could", "must", "should", "may",
                    "might", "would", "have", "has", "had", "do", "does", "did"}
        if w0 in aux_like or w0.endswith("n't"):
            rest = words[1:]
            if w0 in ("have", "has", "had") and rest[:1] == ["to"]:
                return None                                         # 'have to' questions: not used
            return [w0, subj_en] + rest + tail
        return None

    def passive(self, rng, ch, vb, subj, subj_en, png, obj, obj_en, tense, neg, aspect, modal, voice, used):
        base, key, third, past, part, _ = vb
        if isinstance(obj, Coord):
            return None
        o_png = (3, "pl") if (isinstance(obj, NP) and obj.plural) else (
            (PRONOUNS[obj][1], PRONOUNS[obj][2]) if isinstance(obj, str) else (3, "sg"))
        patient_en = SUBJ_EN[obj] if isinstance(obj, str) else obj_en
        words, mods, t = self.be_group(ch, tense, modal if modal != "have to" else "none",
                                       aspect if aspect == "perfect" else "simple", neg, o_png)
        cl = ClauseV2(verb=key, tense=t, neg=neg, mods=mods)
        self.set_obj(cl, obj)
        en = [patient_en] + words + [part]
        if voice == "passive":
            used.add(("voice", "passive"))
            self.set_subj(cl, subj)
            agent_en = OBJ_EN[subj] if isinstance(subj, str) else subj_en
            en += ["by", agent_en]
        else:
            used.add(("voice", "passive without doer"))
        return cl, en

    def copula(self, rng, ch, counts, used, pred, subj, subj_en, png, tense, neg, aspect, modal, q, rels, hows, times):
        aspect = "perfect" if aspect == "perfect" else "simple"
        hows = [h for h in hows if h[2] not in ("manner", "degree + manner", "a lot")]
        words, mods, t = self.be_group(ch, tense, modal, aspect, neg, png)
        cl = ClauseV2(verb="be", tense=t, neg=neg, mods=mods)
        tail = []
        if pred == "there is":
            if isinstance(subj, str):
                subj, subj_en, png = self.np(rng, ch, "subj", self.adj_count(rng, counts), allow_owner=False)
                words, mods, t = self.be_group(ch, tense, modal, aspect, neg, png)
                cl = ClauseV2(verb="be", tense=t, neg=neg, mods=mods)
            self.set_subj(cl, subj)
            if not rels:
                rels = [self.relation(rng, ch, "L")]
            if q:
                if words[0] not in ("is", "are", "was", "were", "isn't", "aren't", "wasn't", "weren't"):
                    return None
                cl.mods = cl.mods + ("Q",)
                en = [words[0], "there"] + words[1:] + [subj_en]
            else:
                en = ["there"] + words + [subj_en]
            return self.finish(rng, ch, cl, en, rels, hows, times, used, verb_at=None, question=q)
        self.set_subj(cl, subj)
        if pred == "be + describing":
            n = max(1, self.adj_count(rng, counts))
            picked = list(dict.fromkeys(rng.choices(self.adjs, weights=self.adj_w, k=n)))
            keys, ens = [], []
            for i, (a, ak) in enumerate(picked):
                k, e = self.degree(rng, a, ak, ch["degree"] if i == 0 else "none")
                keys += k
                ens.append(e)
            cl.adjs = tuple(keys)
            if len(ens) >= 2:
                tail = [", ".join(ens[:-1]) + " and " + ens[-1]] if ch["adj_sep"] == "and" or True else []
            else:
                tail = ens
            if any(k.endswith(">MORE") or k == "more" for k in keys) and rng.random() < 0.5:
                o = self.nominal(rng, ch, "obj", rng.choice(["noun", "pronoun", "name"]), {0: 0.9, 1: 0.1})
                rels = rels + [("THAN", o[0], "than " + o[1])]
        elif pred == "be + noun":
            pn, pen, _ = self.np(rng, dict(ch, det=rng.choice(["a/an", "a/an", "the", "my", "owner", "this"])), "obj",
                                 self.adj_count(rng, counts), people=rng.random() < 0.5,
                                 plural=png[1] == "pl")
            cl.pred = pn
            tail = [pen]
        else:                                                     # be + place
            if not rels:
                rels = [self.relation(rng, ch, rng.choice(["L", "L", "ON", "UNDER", "NEAR", "BEHIND", "FRONT",
                                                          "WITH"]))]
        if q:
            en = self.invert(subj_en, words, tail, ch, used)
            if en is None:
                return None
            cl.mods = cl.mods + ("Q",)
            return self.finish(rng, ch, cl, en, rels, hows, times, used, verb_at=None, question=True)
        return self.finish(rng, ch, cl, [subj_en] + words + tail, rels, hows, times, used, verb_at=None)

    def command(self, rng, ch, counts, used, rels, hows):
        neg = rng.random() < 0.25
        vb = rng.choices(self.verbs, weights=self.verb_w)[0]
        cl = ClauseV2(verb=vb[1], neg=neg)
        en = (["don't"] if ch["negation"] == "contracted" else ["do", "not"]) if neg else []
        en.append(vb[0])
        verb_at = len(en) - 1
        if vb[5]:
            if vb[0] in DITRANSITIVE and rng.random() < 0.4:
                r = self.nominal(rng, ch, "obj", "pronoun", {0: 1.0})
                o = self.nominal(rng, ch, "obj", "noun", counts)
                self.set_obj(cl, o[0])
                if ch["give form"] == "receiver first":
                    used.add(("give form", "receiver first"))
                    en += [r[1], o[1]]
                    return self.finish(rng, ch, cl, en, rels, hows, [], used, verb_at=verb_at,
                                       extra_obls=[("TO", r[0])])
                rels = [("TO", r[0], "to " + r[1])] + rels
                en.append(o[1])
            else:
                o = self.nominal(rng, ch, "obj", ch["object"], counts)
                self.set_obj(cl, o[0])
                en.append(o[1])
        please = rng.random() < 0.3
        if please:
            cl.intj = ("please",)
        out = self.finish(rng, ch, cl, en, rels, hows, [], used, verb_at=verb_at)
        if out and please:
            cl2, words = out
            words = (["please"] + words) if rng.random() < 0.5 else (words + ["please"])
            return cl2, words
        return out

    def fragment(self, rng, ch, counts, used):
        f = rng.choice(["noun phrase", "describing", "place", "how", "action", "to + action", "time"])
        if f == "noun phrase":
            x, en, _ = self.nominal(rng, ch, "subj", rng.choice(["noun", "noun", "name"]), counts)
            cl = ClauseV2()
            self.set_subj(cl, x)
            return cl, [en]
        if f == "describing":
            a, ak = rng.choices(self.adjs, weights=self.adj_w)[0]
            k, e = self.degree(rng, a, ak, ch["degree"])
            return ClauseV2(adjs=tuple(k)), [e]
        if f == "place":
            fld, x, en = self.relation(rng, ch, "L")
            return ClauseV2(obls=((fld, x),)), [en]
        if f == "how":
            k, en, _ = self.how(rng, ch)
            return ClauseV2(adv="+".join(k)), [en]
        if f == "time":
            x, en = self.time_phrase(rng)
            return ClauseV2(time=(x,)), [en]
        vb = rng.choices(self.verbs, weights=self.verb_w)[0]
        cl = ClauseV2(verb=vb[1])
        en = [vb[0]]
        if f == "to + action":
            cl.link = "to"
            en = ["to", vb[0]]
        if vb[5] and rng.random() < 0.5:
            o = self.nominal(rng, ch, "obj", "noun", counts)
            self.set_obj(cl, o[0])
            en.append(o[1])
        return cl, en

    def greeting(self, rng, ch, counts, used):
        g = rng.choice(FAMILIES["greeting"])
        cl = ClauseV2()
        if g == "thank you":
            cl.verb, cl.obj = "thank", "2sg"
            en = ["thank", "you"]
            r = rng.random()
            if r < 0.3:
                cl.adv = "very+much"
                en += ["very", "much"]
            elif r < 0.5:
                fld, x, e = self.relation(rng, ch, "FOR")
                cl.obls = ((fld, x),)
                en.append("for " + e.split(" ", 1)[1])
        else:
            words = INTJ_EN.get(g, [g])
            w = rng.choice(words)
            cl.intj = (w,)
            en = [w]
            if g == "exclamation" and w == "oh" and rng.random() < 0.3:
                cl.intj = ("oh", "no")
                en = ["oh", "no"]
        # the person spoken to: hello Mira, thanks, Mira
        if rng.random() < 0.4:
            if rng.random() < 0.6:
                p = self.name(rng)
                p_en = p.name
            else:
                p, p_en, _ = self.np(rng, dict(ch, det="none", degree="none"), "obj", 0, people=True, plural=True)
                if not p.plural:
                    return None
            cl.voc = p
            en = en + ([",", p_en] if g == "thank you" else [p_en])
        return cl, en

    def wh(self, rng, ch, counts, used, tense, neg, aspect, modal):
        w = rng.choice(FAMILIES["wh"])
        sk = ch["subject"] if ch["subject"] != "joined" else "pronoun"
        subj, subj_en, png = self.nominal(rng, ch, "subj", sk, counts)
        if w in ("where", "when", "why", "how"):
            vb = rng.choices(self.verbs, weights=self.verb_w)[0]
            obj = self.nominal(rng, ch, "obj", "noun", counts) if vb[5] else None
            words, mods, t = self.verb_group(dict(ch, present="do-emphatic", past="did-emphatic", future="will"),
                                             vb, tense, aspect, modal, neg, png, subj_en, set())
            cl = ClauseV2(verb=vb[1], tense=t, neg=neg, mods=mods, adv=w)
            self.set_subj(cl, subj)
            if obj:
                self.set_obj(cl, obj[0])
            inv = self.invert(subj_en, words, [obj[1]] if obj else [], dict(ch, **{"question form": "inverted"}),
                              set())
            if inv is None:
                return None
            return cl, [w] + inv
        if w in ("who / what (is)", "how + describing", "whose"):
            words, mods, t = self.be_group(ch, tense, "none", "simple", False, png)
            cl = ClauseV2(verb="be", tense=t, mods=mods)
            self.set_subj(cl, subj)
            if w == "who / what (is)":
                q = rng.choice(["who", "what"])
                cl.pred = NP(q, False)
                return cl, [q] + self.invert(subj_en, words, [], dict(ch, **{"question form": "inverted"}), set())
            if w == "how + describing":
                a, ak = rng.choices(self.adjs[:400], weights=self.adj_w[:400])[0]
                cl.adjs = ("how", ak)
                return cl, ["how", a] + self.invert(subj_en, words, [], dict(ch, **{"question form": "inverted"}), set())
            n, n_en, _ = self.np(rng, dict(ch, det="none", degree="none"), "obj", 0, plural=False)
            if not isinstance(subj, NP) or isinstance(subj, NPX) or subj.plural:
                subj = NP("this", False)
                subj_en = "this"
                words, mods, t = self.be_group(ch, tense, "none", "simple", False, (3, "sg"))
                cl = ClauseV2(verb="be", tense=t, mods=mods, subj=subj)
            noun_en = n_en.split()[-1]
            cl.pred = NPX(n.noun, False, None, NP("who", False))
            return cl, ["whose", noun_en] + self.invert(subj_en, words, [], dict(ch, **{"question form": "inverted"}),
                                                       set())
        vb = rng.choices(self.tverbs, weights=self.tverb_w)[0]
        if w in ("who (doer)", "what (doer)"):
            q = "who" if w.startswith("who") else "what"
            words, mods, t = self.verb_group(dict(ch, present="simple", past="simple"), vb, tense, aspect, modal, neg,
                                             (3, "sg"), q, set())
            if words[0] == "'ll":
                words = ["will"] + words[1:]
            obj, obj_en, _ = self.nominal(rng, ch, "obj", ch["object"], counts)
            cl = ClauseV2(subj=NP(q, False), verb=vb[1], tense=t, neg=neg, mods=mods)
            self.set_obj(cl, obj)
            return cl, [q] + words + [obj_en]
        dch = dict(ch, present="do-emphatic", past="did-emphatic", future="will")
        words, mods, t = self.verb_group(dch, vb, tense, aspect, modal, neg, png, subj_en, set())
        cl = ClauseV2(verb=vb[1], tense=t, neg=neg, mods=mods)
        self.set_subj(cl, subj)
        inv = self.invert(subj_en, words, [], dict(ch, **{"question form": "inverted"}), set())
        if inv is None:
            return None
        if w == "who ... to (stranded)":
            fld = rng.choice(["TO", "WITH", "ABOUT", "FOR"])
            cl.obls = ((fld, NP("who", False)),)
            return cl, ["who"] + inv + [REL_WORDS[fld][0]]
        if w == "which + noun" or w == "how many":
            n, n_en, _ = self.np(rng, dict(ch, det="none", degree="none"), "obj", 0, plural=(w == "how many"))
            if w == "how many" and not n.plural:
                return None
            head = n_en.split()[-1]
            if w == "which + noun":
                cl.obj = NP(n.noun, n.plural, "which")
                return cl, ["which", head] + inv
            cl.obj = NP(n.noun, True, "how many")
            return cl, ["how", "many", head] + inv
        q = "who" if w == "who (object)" else "what"
        cl.obj = NP(q, False)
        return cl, [q] + inv

    # --- relation phrases, how-words and time put into the English -------------------------------------------
    def finish(self, rng, ch, cl: ClauseV2, en: list[str], rels, hows, times, used, verb_at=None, extra_obls=(),
               question=False, no_front=False) -> tuple:
        """Adds relation phrases, how-words and time phrases to the English (and the meaning, in English order)."""
        front, back = [], []
        pos = ch["how position"]
        how_front, how_mid, how_back = [], [], []
        no_front = no_front or ch.get("_linked", False)
        for keys, h_en, cls in hows:
            if cls in ("frequency", "focus") and verb_at is not None:
                how_mid.append((keys, h_en))                          # she never sleeps, he will always win
            elif cls in ("frequency", "focus"):
                continue                                              # (not after 'be' in this generator)
            elif cls in ("manner", "degree + manner") and pos == "before the verb" and verb_at is not None:
                how_mid.append((keys, h_en))
                used.add(("how position", "before the verb"))
            elif pos == "start" and cls in ("time", "manner") and not question and not no_front:
                how_front.append((keys, h_en))
                used.add(("how position", "start"))
            else:
                how_back.append((keys, h_en))
        rel_front = []
        if (rels and ch["order"] == "place fronted" and not question and not no_front
                and rels[0][2].split()[0] not in ("since", "before", "after", "until", "till")):
            rel_front, rels = [rels[0]], rels[1:]
            used.add(("order", "place fronted"))
        t_front, t_back = [], []
        for t in times:
            (t_front if (rng.random() < 0.4 and not question and not no_front) else t_back).append(t)
        words = list(en)
        if how_mid:
            at = verb_at
            if verb_at + 1 < len(words) and words[verb_at] in AUX_WORDS:
                at = verb_at + 1 + (1 if words[verb_at + 1] == "not" else 0)
            words = words[:at] + [h for _, h in how_mid] + words[at:]
        # English order: [front time] [front how] [front relation ,] core [relations] [how] [time]
        pre = [t_en for _, t_en in t_front] + [h for _, h in how_front]
        if rel_front:
            pre += [rel_front[0][2] + ","]
        post = [r[2] for r in rels] + [h for _, h in how_back] + [t_en for _, t_en in t_back]
        # meaning, in English order
        adv = [k for keys, _ in how_front for k in keys] + [k for keys, _ in how_mid for k in keys] + \
              [k for keys, _ in how_back for k in keys]
        if cl.adv:
            adv = cl.adv.split("+") + adv
        cl.adv = "+".join(adv) or None
        obls = list(cl.obls) + [(f, x) for f, x, _ in rel_front] + [(f, x) for f, x, _ in rels] + list(extra_obls)
        cl.obls = tuple(obls)
        cl.time = tuple(list(cl.time) + [x for x, _ in t_front] + [x for x, _ in t_back])
        return cl, pre + words + post

    # --- one example ------------------------------------------------------------------------------------------------
    def example(self, rng, force: dict | None = None, n_events: int | None = None, adj_counts: dict | None = None,
                allow_held: bool = False, n_rel: int | None = None) -> dict | None:
        force = force or {}
        used: set = set()
        ch = {fam: self.pick(rng, fam, force, allow_held) for fam in FAMILIES}
        n = n_events or rng.choices(list(EVENT_COUNTS), weights=list(EVENT_COUNTS.values()))[0]
        counts = adj_counts or ADJ_COUNTS
        events = []
        for i in range(n):
            ech = dict(ch)
            if i:                                                # later events: their own kind / predicate / tense
                for fam in ("kind", "predicate", "aspect", "modal", "subject", "object", "det", "degree"):
                    ech[fam] = force.get(fam) or self.pick(rng, fam, None, allow_held)
            if n > 1 and ech["kind"] in ("greeting", "fragment") and i:
                ech["kind"] = "statement"
            link = None
            if i and rng.random() < P_LINK and ech["kind"] in ("statement", "command"):
                link = force.get("link") or self.pick(rng, "link")
                if link == "or" and ech["kind"] == "command":
                    link = None                                      # 'X or do Y' reads as joined nouns
            if link == "to" or link == "that":
                if not events:
                    link = None
            ech["_linked"] = bool(link)
            ech["_last"] = i == n - 1
            if link == "to":
                vb = rng.choices(self.verbs, weights=self.verb_w)[0]
                cl = ClauseV2(verb=vb[1], link="to")
                en = ["to", vb[0]]
                if vb[5]:
                    o = self.nominal(rng, ech, "obj", "noun", counts)
                    self.set_obj(cl, o[0])
                    en.append(o[1])
                ev = (cl, en)
            else:
                old_rels = REL_COUNTS
                ev = self.event(rng, ech, ech["kind"], counts, used)
            if ev is None:
                return None
            cl, en = ev
            if link and link != "to":
                cl.link = link
            events.append((cl, en, link))
        if n_rel is not None:
            return None
        # join the events in English
        parts, clauses = [], []
        for i, (cl, en, link) in enumerate(events):
            text = " ".join(x for x in en if x).replace(" ,", ",")
            if i == 0:
                parts.append(text)
            else:
                if link and link not in ("to",):
                    if ch["link order"] == "first" and link in ("because", "if", "when", "while", "before", "after",
                                                                "until", "although", "unless") and i == 1 and n == 2:
                        used.add(("link order", "first"))
                        parts = [f"{link} {text},", parts[0]]
                        clauses = [cl, events[0][0]]
                        continue
                    sep = ", " if link in ("but", "so", "then", "or") else " "
                    parts.append(sep.lstrip(" ") + f"{link} {text}" if sep == " " else f"{link} {text}")
                    if sep == ", ":
                        parts[-2] = parts[-2] + ","
                elif link == "to":
                    parts.append(text)
                else:
                    joiner = {"full stop": ". ", "comma and": ", and ", "semicolon": "; ", "and": " and "}[ch["join"]]
                    parts.append(joiner + text)
            clauses.append(cl)
        english = ""
        for i, p in enumerate(parts):
            if i == 0 or p.startswith((". ", ", and ", "; ", " and ")):
                english += p
            else:
                english += " " + p
        if "Q" in clauses[-1].mods and (ch["end"] != "none" or ("question form", "statement + ?") in used):
            ch = dict(ch, end="question")
        raw = english.strip()                              # before casing / typos / end mark (paraphrase run)
        english = self.surface(rng, raw, ch)
        if not english:
            return None
        return {"english": english, "clauses": clauses, "choices": ch, "used": used, "raw": raw}

    # --- surface (as in E1) -----------------------------------------------------------------------------------------
    def surface(self, rng, text: str, ch: dict) -> str:
        if ch["lexical"] == "synonym":
            text = self.synonym(rng, text)
        if ch["noise"] != "none":
            text = self.noise(rng, text, ch["noise"])
        end = {"full stop": ".", "exclamation": "!", "question": "?"}.get(ch["end"], "")
        text += end
        case = ch["case"]
        if case == "sentence":
            text = text[0].upper() + text[1:]
        elif case == "title":
            text = " ".join(w[:1].upper() + w[1:] for w in text.split(" "))
        elif case == "upper":
            text = text.upper()
        if ch["spacing"] == "double":
            text = text.replace(" ", "  ")
        return text

    def synonym(self, rng, text: str) -> str:
        words = text.split(" ")
        cands = []
        v1 = self.v.v1
        for i, w in enumerate(words):
            base = w.strip(",.;!?")
            for k in "NLAV":
                for lemma, forms in self.all_syn[k].items():
                    if k == "N" and base in (lemma, v1.noun_info.get(lemma, ("",))[0]):
                        new = forms[1] if base != lemma and len(forms) > 1 else forms[0]
                    elif k == "V" and lemma in v1.verb_info and base in v1.verb_info[lemma]:
                        new = forms[list(v1.verb_info[lemma]).index(base)] if len(forms) == 3 else None
                    elif k in "LA" and base == lemma:
                        new = forms[0]
                    else:
                        continue
                    if new and new not in self.language_words and (k, lemma) not in SYN_HELD:
                        cands.append((i, w.replace(base, new)))
        if cands:
            i, new = rng.choice(cands)
            words[i] = new
        return " ".join(words)

    @staticmethod
    def noise(rng, text: str, kind: str) -> str:
        words = text.split(" ")
        idx = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
        if not idx:
            return text
        i = rng.choice(idx)
        w = words[i]
        j = rng.randrange(1, len(w) - 1)
        if kind == "swap":
            w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        elif kind == "drop":
            w = w[:j] + w[j + 1:]
        elif kind == "insert":
            w = w[:j] + rng.choice("abcdefghijklmnopqrstuvwxyz") + w[j:]
        else:
            w = w[:j] + rng.choice("abcdefghijklmnopqrstuvwxyz") + w[j + 1:]
        words[i] = w
        return " ".join(words)

    # --- answers ----------------------------------------------------------------------------------------------------
    def item(self, ex: dict) -> dict | None:
        try:
            target = " ".join(self.rep.target(c) for c in ex["clauses"])
        except KeyError:
            return None
        if any(self.rep.check(c) for c in ex["clauses"]):
            return None
        return {"english": ex["english"], "target": target,
                "analysis": " ; ".join(self.rep.analysis(c) for c in ex["clauses"])}


SYN_HELD: set = set()


def _draw_syn_held() -> None:
    pairs = sorted((k, w) for k in "NLAV" for w in {**SYN_TABLES["train"][k], **SYN_TABLES["heldout"][k]})
    rng = random.Random(HELD_OUT_SEED + 1)
    SYN_HELD.update(rng.sample(pairs, round(0.2 * len(pairs))))


_draw_syn_held()


def with_cot(it: dict) -> dict:
    return {**it, "target": f"{it['analysis']} => {it['target']}"}


KEY_RE = __import__("re").compile(r"(?<![A-Z.])([a-z][a-z_]*(?:>[A-Z]+)*)")


def analysis_keys(analysis: str) -> list[str]:
    """Word keys in an analysis (fun>ADJ, book_shop, dog); grammar tags and names are not keys."""
    return KEY_RE.findall(analysis)


def pool_keys(gen: GenV2) -> set[str]:
    return ({x[2] for x in gen.nouns} | {x[2] for x in gen.places} | {k for _, k in gen.adjs}
            | {x[1] for x in gen.verbs} | {k for _, k in gen.manner})


def cover(gen: GenV2, rng, out: list[dict], seen: set, min_count: int) -> dict:
    """Data fix: every word of the language at least `min_count` times, same number of examples.
    1) count every word key; 2) add examples whose content words are drawn only from the words still below
    min_count (all other choices as usual) until each reaches it; 3) drop as many examples as were added,
    preferring examples made only of very frequent words, never taking any word below min_count."""
    import collections
    counts = collections.Counter(k for it in out for k in analysis_keys(it["analysis"]))
    keys = pool_keys(gen)
    full = (gen.nouns, gen.noun_w, gen.places, gen.places_w, gen.adjs, gen.adj_w, gen.verbs, gen.verb_w,
            gen.tverbs, gen.tverb_w, gen.iverbs, gen.iverb_w, gen.manner, gen.manner_w)
    before = {"below": sum(counts[k] < min_count for k in keys)}
    added, tries = [], 0
    while tries < 400000:
        low = {k for k in keys if counts[k] < min_count}
        if not low:
            break
        tries += 1

        def pick(pool, w, key):
            sel = [x for x in pool if key(x) in low]
            return (sel, [1.0] * len(sel)) if sel else (pool, w)
        gen.nouns, gen.noun_w = pick(full[0], full[1], lambda x: x[2])
        gen.places, gen.places_w = pick(full[2], full[3], lambda x: x[2])
        gen.adjs, gen.adj_w = pick(full[4], full[5], lambda x: x[1])
        gen.verbs, gen.verb_w = pick(full[6], full[7], lambda x: x[1])
        gen.tverbs, gen.tverb_w = pick(full[8], full[9], lambda x: x[1])
        gen.iverbs, gen.iverb_w = pick(full[10], full[11], lambda x: x[1])
        gen.manner, gen.manner_w = pick(full[12], full[13], lambda x: x[1])
        try:
            ex = gen.example(rng)
            it = gen.item(ex) if ex else None
        finally:
            (gen.nouns, gen.noun_w, gen.places, gen.places_w, gen.adjs, gen.adj_w, gen.verbs, gen.verb_w,
             gen.tverbs, gen.tverb_w, gen.iverbs, gen.iverb_w, gen.manner, gen.manner_w) = full
        if not it or it["english"] in seen:
            continue
        ks = analysis_keys(it["analysis"])
        if not any(k in low for k in ks):
            continue
        seen.add(it["english"])
        added.append({**it, "aug": "rep2"})
        counts.update(ks)
    # drop as many as were added: examples whose words are all far above min_count first (trims the top)
    order = sorted(range(len(out)), key=lambda i: -min((counts[k] for k in analysis_keys(out[i]["analysis"])
                                                        if k in keys), default=10 ** 6))
    drop = set()
    for i in order:
        if len(drop) >= len(added):
            break
        ks = [k for k in analysis_keys(out[i]["analysis"]) if k in keys]
        if all(counts[k] - ks.count(k) >= min_count for k in set(ks)):
            drop.add(i)
            counts.subtract(ks)
    kept = [it for i, it in enumerate(out) if i not in drop] + added
    after = {"below": sum(counts[k] < min_count for k in keys), "added": len(added), "dropped": len(drop),
             "examples": len(kept)}
    out[:] = kept
    return {"min_count": min_count, "words": len(keys), "below_before": before["below"], **after}


LOAN = {"rate": 0.0, "pool": "train", **json.loads((Path(__file__).resolve().parents[1] / "conlang" / "data" /
                                                   "loan_nouns.json").read_text(encoding="utf-8"))}


def build_training(preset: str, script: str, seed: int, n: int = 60000, n_refuse: int = 2000,
                   n_replay: int = 6000, cot: bool = False, held_frac: float = 0.2,
                   min_count: int = 0, paraphrase_file: str | None = None,
                   loan_rate: float = 0.0, real_file: str | None = None, held_only: str | None = None,
                   putback: int = 0) -> tuple[list[dict], dict]:
    """paraphrase_file: {original english: paraphrase} from scripts/build_paraphrases.py (each checked to read to
    exactly the same analysis); matched examples get the paraphrased English, labels unchanged.
    held_only "voice=passive": hold out only that variant (instead of held_frac), then put back `putback` examples
    of it (structure dose: how many sightings close an unseen-structure gap). Total stays n."""
    set_held_out(held_frac)
    if held_only:
        HELD_OUT.clear()
        HELD_OUT.add(tuple(held_only.split("=", 1)))
    LOAN.update(rate=loan_rate, pool="train")
    real = []
    if real_file:                                       # real English (e.g. Tatoeba) with reader labels: replaces the
        rf = Path(real_file)                            # same number of generated examples, total stays n
        if not rf.exists():
            rf = Path(__file__).resolve().parents[1] / real_file
        real = json.loads(rf.read_text(encoding="utf-8"))
        n = n - len(real)
    gen = GenV2(preset, script, seed)
    rng = random.Random(seed + 505)
    out, seen = [], set()
    if held_only and putback:                           # the dose: N examples of the held-out variant
        fam, var = next(iter(HELD_OUT))
        prng = random.Random(seed + 707)
        while len(out) < putback:
            ex = gen.example(prng, force={fam: var}, allow_held=True)
            it = gen.item(ex) if ex and (fam, var) in ex["used"] else None
            if it and it["english"] not in seen:
                seen.add(it["english"])
                ch = ex["choices"]
                clean = (ch.get("noise", "none") == "none" and ch.get("lexical", "plain") == "plain"
                         and ch.get("spacing", "single") == "single" and ch.get("case", "sentence") == "sentence")
                out.append({**it, "aug": "rep2", "clean": clean, "raw": ex["raw"], "putback": True,
                            "surf": {k: ch.get(k) for k in ("case", "end", "spacing", "noise", "lexical")}})
    while len(out) < n:
        ex = gen.example(rng)
        it = gen.item(ex) if ex else None
        if it and it["english"] not in seen:
            seen.add(it["english"])
            ch = ex["choices"]
            clean = (ch.get("noise", "none") == "none" and ch.get("lexical", "plain") == "plain"
                     and ch.get("spacing", "single") == "single" and ch.get("case", "sentence") == "sentence")
            out.append({**it, "aug": "rep2", "clean": clean, "raw": ex["raw"],
                        "surf": {k: ch.get(k) for k in ("case", "end", "spacing", "noise", "lexical")}})
    coverage_info = cover(gen, rng, out, seen, min_count) if min_count > 0 else None
    n_para = 0
    if paraphrase_file:
        pf = Path(paraphrase_file)
        if not pf.exists():
            pf = Path(__file__).resolve().parents[1] / paraphrase_file   # relative to the code folder (Kaggle)
        para = json.loads(pf.read_text(encoding="utf-8"))
        for it in out:
            if it["english"] in para:
                it["english"], it["paraphrased"] = para[it["english"]], True
                n_para += 1
    out += [{"english": x["english"], "target": x["target"], "analysis": x["analysis"], "aug": "real"} for x in real]
    for _ in range(n_refuse):
        kind, txt = refusal(rng, "train")
        out.append({"english": txt, "target": REFUSAL, "analysis": "NOT-ENGLISH", "aug": "refusal"})
    if cot:
        out = [with_cot(it) if it["aug"] != "refusal" else it for it in out]
    out += load_replay(n_replay, seed)
    return out, {"held_out": sorted(f"{f}={v}" for f, v in HELD_OUT), "synonym_pairs_held_out": len(SYN_HELD),
                 "n": len(out), "language": "v2", "coverage": coverage_info,
                 "paraphrased": n_para, "real": len(real),
                 **({"held_only": held_only, "putback": putback} if held_only else {})}


def eval_sets(preset: str, script: str, seed: int, train_en: set, per: int = 100,
              strict: bool = True) -> dict[str, list[dict]]:
    """In-distribution (overall and per kind of utterance), held-out English variants, held-out synonyms,
    count extrapolation, refusal."""
    gen = GenV2(preset, script, seed)
    rng = random.Random(seed + 606)
    sets: dict[str, list[dict]] = {}

    def fill(name, make, n=per):
        items, tries = [], 0
        while len(items) < n and tries < 300 * n:
            tries += 1
            ex = make()
            it = gen.item(ex) if ex else None
            if it and it["english"] not in train_en:
                items.append(it)
        sets[name] = items
    fill("in-distribution mix", lambda: gen.example(rng))
    for k in FAMILIES["kind"]:
        fill(f"in-distribution kind = {k}", lambda k=k: gen.example(rng, force={"kind": k}, n_events=1), per // 2)

    def forced(f, v):
        ex = gen.example(rng, force={f: v})
        if ex and strict and f in ("voice", "order", "future", "present", "past", "question form", "give form",
                                   "how position", "link order") and (f, v) not in ex["used"]:
            return None                                    # the variant could not apply: skip
        return ex
    for fam, var in sorted(HELD_OUT):
        fill(f"held-out {fam} = {var}", lambda f=fam, v=var: forced(f, v))
    fill("held-out synonym pairs", lambda: _syn_held_example(gen, rng))
    if LOAN["rate"]:                                   # loanwords: trained ones, and new ones never seen in training
        rate = LOAN["rate"]
        for pool in ("train", "held"):
            LOAN.update(rate=0.5, pool=pool)
            fill(f"loanwords [{'seen' if pool == 'train' else 'new'} loans]",
                 lambda: (lambda ex: ex if ex and "LOAN." in str(ex["clauses"]) else None)(gen.example(rng)))
        LOAN.update(rate=rate, pool="train")
    fill("extrapolation: 5-6 describing words on a noun",
         lambda: gen.example(rng, force={"kind": "statement", "predicate": "transitive"}, n_events=1,
                             adj_counts={5: 0.5, 6: 0.5}))
    fill("extrapolation: 5 events", lambda: gen.example(rng, n_events=5))
    for mode in ("train", "heldout"):
        sets[f"refusal [{mode}]"] = [{"english": refusal(rng, mode)[1], "target": REFUSAL, "analysis": "NOT-ENGLISH"}
                                     for _ in range(per)]
    return sets


def _syn_held_example(gen: GenV2, rng) -> dict | None:
    ex = gen.example(rng, force={"lexical": "plain", "noise": "none", "case": "lower", "kind": "statement"},
                     n_events=1)
    if not ex:
        return None
    words = ex["english"].split(" ")
    v1 = gen.v.v1
    for i, w in enumerate(words):
        base = w.strip(",.;!?").lower()
        for k, lemma in SYN_HELD:
            forms = gen.all_syn[k][lemma]
            new = None
            if k == "N" and base in (lemma, v1.noun_info.get(lemma, ("",))[0]):
                new = forms[1] if base != lemma and len(forms) > 1 else forms[0]
            elif k == "V" and lemma in v1.verb_info and base in v1.verb_info[lemma] and len(forms) == 3:
                new = forms[list(v1.verb_info[lemma]).index(base)]
            elif k in "LA" and base == lemma:
                new = forms[0]
            if new and new not in gen.language_words:
                words[i] = w.replace(w.strip(",.;!?"), new)
                return {**ex, "english": " ".join(words)}
    return None


def coverage(items: list[dict], gen: GenV2) -> dict:
    """How often the content words of the language appear in the training data (analysis keys)."""
    import collections
    import re
    cnt = collections.Counter()
    for it in items:
        if it.get("aug") == "rep2":
            for k in re.findall(r"[a-z][a-z_>A-Z]*", it["analysis"]):
                cnt[k.split(">")[0]] += 1
    roots = {x[2].split(">")[0] for x in gen.nouns} | {k.split(">")[0] for _, k in gen.adjs} | \
            {x[1].split(">")[0] for x in gen.verbs}
    seen = [cnt[r] for r in roots]
    return {"content roots": len(roots), "seen at least once": sum(c > 0 for c in seen),
            "seen 10+": sum(c >= 10 for c in seen), "seen 30+": sum(c >= 30 for c in seen)}
