"""Training-data augmentation for the retrain (R1): varied English, new structures, refusals, replay.

Everything has a TRAIN variant and a HELD-OUT variant. Held-out variants never appear in training; they are only
used by `eval_sets()` to test whether what was learned generalises.

- Rewrites (same meaning, so the target is unchanged): passive (verbs split in two halves), fronting (location /
  object), fillers, determiners, numerals, tense phrasing, synonyms (pairs split in two halves), typos, spacing,
  casing (consistent whole-sentence rules).
- New structures (new targets from the grammar): two adjectives on a noun (held out: three), a transitive verb
  without its object, names spelled by sound as subject (held out: as object), two sentences (held out: three).
- Refusal: inputs that are not English get the fixed output REFUSAL (held-out kinds test generalisation).
- Replay: plain-English passages (FineWeb-Edu sample, train/data/replay_en.jsonl) as language-modelling examples.

Same seed -> same data. The four standard test splits are never touched.
"""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from pathlib import Path

from conlang.dataset import PRESETS, split_of
from conlang.frames import NP, PRONOUNS, Frame, Vocab, english, sample_frame
from conlang.grammar import Language, Word
from conlang.script import Script, render_sentence

REFUSAL = "<not-english>"
DATA = Path(__file__).resolve().parent / "data"


def frame_of(d: dict) -> Frame:
    s = d["subj"]
    subj = s if isinstance(s, str) else NP(s["noun"], s["plural"], s.get("adj"))
    o = d["obj"]
    obj = None if o is None else NP(o["noun"], o["plural"], o.get("adj"))
    return Frame(subj, d["verb"], obj, d["loc"], d["tense"], d["neg"])


def half(kind: str, key: str) -> str:
    """Stable split of a word or pair into 'train' / 'heldout' halves."""
    return "train" if hashlib.md5(f"{kind}|{key}".encode()).digest()[0] % 2 == 0 else "heldout"


def clean_input(text: str) -> str:
    """Optional input clean-up (evaluation condition): NFKC + drop invisible format characters."""
    t = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in t if unicodedata.category(ch) != "Cf")


# --- English rewrites ----------------------------------------------------------------------------------

IRREGULAR_PARTICIPLE = {"see": "seen", "take": "taken", "find": "found", "hold": "held", "catch": "caught",
                        "build": "built", "bring": "brought", "eat": "eaten", "write": "written", "break": "broken",
                        "steal": "stolen", "choose": "chosen", "hide": "hidden", "throw": "thrown", "give": "given",
                        "drive": "driven", "forget": "forgotten", "beat": "beaten", "shake": "shaken", "know": "known"}
OBJ_PRONOUN = {"1sg": "me", "2sg": "you", "3sgm": "him", "3sgf": "her", "3sgn": "it", "1pl": "us",
               "2pl": "you all", "3pl": "them"}
BE = {"1sg": "am", "2sg": "are", "3sgm": "is", "3sgf": "is", "3sgn": "is", "1pl": "are", "2pl": "are", "3pl": "are"}
WILL = {"1sg": "I'll", "2sg": "you'll", "3sgm": "he'll", "3sgf": "she'll", "3sgn": "it'll", "1pl": "we'll",
        "3pl": "they'll"}
SYN = {  # word -> synonym forms; the synonym must not be a word of the language (checked at use)
    "N": {"dog": ("hound", "hounds"), "car": ("automobile", "automobiles"), "child": ("kid", "kids"),
          "stone": ("rock", "rocks"), "ship": ("vessel", "vessels"), "woman": ("lady", "ladies"),
          "cup": ("mug", "mugs"), "doctor": ("physician", "physicians"), "teacher": ("instructor", "instructors"),
          "student": ("pupil", "pupils"), "bucket": ("pail", "pails"), "sofa": ("couch", "couches"),
          "lawyer": ("attorney", "attorneys"), "photo": ("picture", "pictures"), "bag": ("sack", "sacks"),
          "plate": ("dish", "dishes"), "coat": ("jacket", "jackets"), "pig": ("hog", "hogs"),
          "rabbit": ("bunny", "bunnies"), "kitten": ("kitty", "kitties")},
    "L": {"house": ("home",), "road": ("street",), "shop": ("store",), "forest": ("woods",),
          "river": ("stream",), "jail": ("prison",), "hill": ("mound",), "field": ("meadow",)},
    "A": {"big": ("large",), "small": ("little",), "quick": ("fast",), "happy": ("glad",), "smart": ("clever",),
          "old": ("aged",), "angry": ("mad",), "rich": ("wealthy",), "sad": ("unhappy",), "huge": ("enormous",),
          "beautiful": ("pretty",), "quiet": ("silent",), "wet": ("damp",), "strong": ("powerful",),
          "brave": ("courageous",), "stupid": ("foolish",), "odd": ("strange",), "rude": ("impolite",),
          "loud": ("noisy",), "ill": ("sick",)},
    "V": {"see": ("notice", "notices", "noticed"), "help": ("assist", "assists", "assisted"),
          "buy": ("purchase", "purchases", "purchased"), "hate": ("despise", "despises", "despised"),
          "fight": ("battle", "battles", "battled"), "try": ("attempt", "attempts", "attempted"),
          "seek": ("search", "searches", "searched"), "shout": ("yell", "yells", "yelled"),
          "choose": ("select", "selects", "selected"), "repair": ("fix", "fixes", "fixed"),
          "destroy": ("ruin", "ruins", "ruined"), "gather": ("collect", "collects", "collected"),
          "grab": ("seize", "seizes", "seized"), "steal": ("rob", "robs", "robbed"),
          "speak": ("talk", "talks", "talked"), "hide": ("conceal", "conceals", "concealed")},
}


def synonym_tables() -> dict[str, dict]:
    """mode -> {"N"/"L"/"A"/"V": {word: forms}}. Held-out = the hand-checked pairs in the held-out half only;
    train = the other hand-checked pairs + WordNet pairs (train/data/synonyms.json), minus every held-out word."""
    held = {k: {w: f for w, f in d.items() if half("syn", f"{k}:{w}") == "heldout"} for k, d in SYN.items()}
    train = {k: {w: f for w, f in d.items() if half("syn", f"{k}:{w}") == "train"} for k, d in SYN.items()}
    path = DATA / "synonyms.json"
    if path.exists():
        for k, d in json.loads(path.read_text(encoding="utf-8")).items():
            for w, f in d.items():
                if w in held[k] or w in train[k]:
                    continue
                if k == "N" and (len(f) < 2 or len(f[1]) < len(f[0])):   # drop broken plurals (e.g. sis -> ses)
                    continue
                train[k][w] = tuple(f)
    return {"train": train, "heldout": held}


SYN_TABLES = synonym_tables()


def subj_key(f: Frame) -> str:
    return f.subj if isinstance(f.subj, str) else ("3pl" if f.subj.plural else "3sgn")


def np_text(np: NP, v: Vocab, det: str = "the") -> str:
    return " ".join([det] + ([np.adj] if np.adj else []) + [v.noun_info[np.noun][0] if np.plural else np.noun])


def rw_passive(f, v, rng, mode):
    if f.obj is None or half("passive", f.verb) != mode:
        return None
    base, _, past = v.verb_info[f.verb]
    part = IRREGULAR_PARTICIPLE.get(f.verb) or (past if past.endswith("ed") else None)
    if part is None:
        return None
    pl = f.obj.plural
    be = {"present": "are" if pl else "is", "past": "were" if pl else "was", "future": "will"}[f.tense]
    aux = [be, "not", "be"] if (f.tense == "future" and f.neg) else ([be, "be"] if f.tense == "future" else
                                                                      ([be, "not"] if f.neg else [be]))
    agent = OBJ_PRONOUN[f.subj] if isinstance(f.subj, str) else np_text(f.subj, v)
    out = f"{np_text(f.obj, v)} {' '.join(aux)} {part} by {agent}"
    return out + (f" in the {f.loc}" if f.loc else "")


def rw_fronting(f, v, rng, mode):
    if mode == "train":
        if not f.loc:
            return None
        return f"in the {f.loc}, " + english(Frame(f.subj, f.verb, f.obj, None, f.tense, f.neg), v)
    if f.obj is None or f.loc:
        return None
    en, obj = english(f, v), np_text(f.obj, v)
    return f"{obj}, {en[:len(en) - len(obj)].strip()}" if en.endswith(obj) else None


def rw_filler(f, v, rng, mode):
    pre = ["okay, ", "so, ", "please translate: "] if mode == "train" else ["well, ", "hey, "]
    return rng.choice(pre) + english(f, v)


def _replace_dets(f, v, choose) -> str | None:
    words, nps, k, changed = english(f, v).split(), [x for x in (f.subj, f.obj) if isinstance(x, NP)], 0, False
    for i, t in enumerate(words):
        if t == "the" and k < len(nps) and not (i >= 1 and words[i - 1] == "in"):
            new = choose(nps[k], words[i + 1])
            if new:
                words[i], changed = new, True
            k += 1
    return " ".join(words) if changed else None


def rw_determiner(f, v, rng, mode):
    if mode == "train":
        return _replace_dets(f, v, lambda np, nxt: None if np.plural else rng.choice(
            ["an" if nxt[0] in "aeiou" else "a", "this", "that"]))
    return _replace_dets(f, v, lambda np, nxt: rng.choice(["these", "those"]) if np.plural else None)


def rw_numeral(f, v, rng, mode):
    if mode == "train":
        return _replace_dets(f, v, lambda np, nxt: rng.choice(["two", "three"]) if np.plural else "one")
    return _replace_dets(f, v, lambda np, nxt: rng.choice(["four", "five"]) if np.plural else None)


def rw_tense(f, v, rng, mode):
    en = english(f, v)
    base, third, past = v.verb_info[f.verb]
    if mode == "heldout":                                   # "he'll sleep" (pronoun subjects, future)
        if f.tense != "future" or f.neg or f.subj not in WILL:
            return None
        return en.replace(f"{PRONOUNS[f.subj][0]} will ", WILL[f.subj] + " ", 1)
    if f.neg:
        out = (en.replace("does not", "doesn't").replace("do not", "don't").replace("did not", "didn't")
               .replace("will not", "won't"))
        return out if out != en else None
    w = en.split()
    if f.tense == "future":
        return en.replace(f" will {base}", f" {BE[subj_key(f)]} going to {base}", 1)
    if f.tense == "past" and past in w:
        return " ".join(w[:w.index(past)] + ["did", base] + w[w.index(past) + 1:])
    p, n, _ = f.subject_png(v)
    form = third if (p == 3 and n == "sg") else base
    if form in w:
        return " ".join(w[:w.index(form)] + ["does" if form == third else "do", base] + w[w.index(form) + 1:])
    return None


def rw_synonym(f, v, rng, mode, language_words: set):
    words = english(f, v).split()
    T = SYN_TABLES[mode]
    cands = []
    for np in (f.subj, f.obj):
        if isinstance(np, NP):
            if np.noun in T["N"]:
                s = T["N"][np.noun]
                cands.append(("N", np.noun, v.noun_info[np.noun][0] if np.plural else np.noun, s[1] if np.plural else s[0]))
            if np.adj in T["A"]:
                cands.append(("A", np.adj, np.adj, T["A"][np.adj][0]))
    if f.loc in T["L"]:
        cands.append(("L", f.loc, f.loc, T["L"][f.loc][0]))
    if f.verb in T["V"]:
        for old, new in zip(v.verb_info[f.verb], T["V"][f.verb]):
            cands.append(("V", f.verb, old, new))
    rng.shuffle(cands)
    for kind, key, old, new in cands:
        if old in words and new not in language_words:
            i = words.index(old)
            return " ".join(words[:i] + [new] + words[i + 1:])
    return None


KEYS = "qwertyuiop asdfghjkl zxcvbnm".split()


def rw_typo(f, v, rng, mode):
    words = english(f, v).split()
    cands = [i for i, w in enumerate(words) if len(w) >= 5]
    if not cands:
        return None
    i = rng.choice(cands)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    if mode == "train":                                     # swap two neighbouring letters
        new = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    elif rng.random() < 0.5:                                # held out: drop a letter
        new = w[:j] + w[j + 1:]
    else:                                                   # held out: keyboard-neighbour substitution
        row = next((r for r in KEYS if w[j] in r), None)
        if row is None:
            return None
        k = row.index(w[j])
        new = w[:j] + row[max(0, min(len(row) - 1, k + rng.choice([-1, 1])))] + w[j + 1:]
    if new == w:
        return None
    words[i] = new
    return " ".join(words)


def rw_spacing(f, v, rng, mode):
    en = english(f, v)
    return en.replace(" ", "  ") if mode == "train" else en.replace(" ", "   ")


def rw_casing(f, v, rng, mode):
    en = english(f, v)
    if mode == "train":
        rule = rng.choice(["sentence", "title", "upper"])
        if rule == "sentence":
            return en[0].upper() + en[1:]
        return en.title() if rule == "title" else en.upper()
    rule = rng.choice(["second", "alternate"])
    if rule == "second":                                    # consistent unusual rule: 2nd letter of every word
        return " ".join(w[0] + w[1].upper() + w[2:] if len(w) > 1 else w.upper() for w in en.split())
    return " ".join(w.upper() if i % 2 else w for i, w in enumerate(en.split()))   # every other word in capitals


REWRITES = {"passive": rw_passive, "fronting": rw_fronting, "filler": rw_filler, "determiner": rw_determiner,
            "numeral": rw_numeral, "tense": rw_tense, "synonym": rw_synonym, "typo": rw_typo,
            "spacing": rw_spacing, "casing": rw_casing}


# --- new structures ----------------------------------------------------------------------------------

class Factory:
    """The run's language and script, plus the new-structure generators."""

    def __init__(self, preset: str, script: str, seed: int):
        self.v = Vocab.large()
        self.lang = Language(PRESETS[preset], seed, self.v)
        self.script = Script(script, PRESETS[preset].phonology)
        self.seed = seed
        self.language_words = set(self.v.nouns) | set(self.v.locations) | set(self.v.transitive) \
            | set(self.v.intransitive) | set(self.v.adjectives)
        self.lexicon_sounds = {"".join(r) for r in self.lang.roots.values()}

    def target(self, f: Frame) -> str:
        return render_sentence(self.lang.realize(f).words, self.script)

    def ok(self, f: Frame) -> bool:
        return split_of(f, self.v, self.seed) is None       # never touches the held-out test structures

    def with_synonym_word(self, rng, mode: str) -> Frame:
        """A random new frame that contains a word from the given synonym half."""
        T = SYN_TABLES[mode]
        f = sample_frame(rng, self.v)
        kind = rng.choice([k for k in T if T[k]])
        w = rng.choice(sorted(T[kind]))
        subj = f.subj if isinstance(f.subj, NP) else NP(rng.choice(self.v.nouns), False)
        if kind == "N" and w in self.v.noun_info:
            return Frame(NP(w, subj.plural, subj.adj), f.verb, f.obj, f.loc, f.tense, f.neg)
        if kind == "A" and w in self.v.adjectives:
            return Frame(NP(subj.noun, subj.plural, w), f.verb, f.obj, f.loc, f.tense, f.neg)
        if kind == "L" and w in self.v.locations:
            return Frame(f.subj, f.verb, f.obj, w, f.tense, f.neg)
        if kind == "V" and w in self.v.verb_info:
            if w in self.v.transitive:
                return Frame(f.subj, w, f.obj or NP(rng.choice(self.v.nouns), False), f.loc, f.tense, f.neg)
            return Frame(f.subj, w, None, f.loc, f.tense, f.neg)
        return f

    def rewrite(self, kind: str, f: Frame, rng, mode: str) -> str | None:
        fn = REWRITES[kind]
        return fn(f, self.v, rng, mode, self.language_words) if kind == "synonym" else fn(f, self.v, rng, mode)

    def adjectives(self, rng, n_adj: int) -> dict | None:
        f = sample_frame(rng, self.v)
        nps = [r for r, x in (("subj", f.subj), ("obj", f.obj)) if isinstance(x, NP)]
        if not nps:
            return None
        role = rng.choice(nps)
        np = getattr(f, role)
        adj = " ".join(rng.sample(self.v.adjectives, n_adj))
        g = Frame(NP(np.noun, np.plural, adj), f.verb, f.obj, None, f.tense, f.neg) if role == "subj" else \
            Frame(f.subj, f.verb, NP(np.noun, np.plural, adj), None, f.tense, f.neg)
        if not self.ok(g):
            return None
        return {"english": english(g, self.v), "target": self.target(g)}

    def no_object(self, rng) -> dict | None:
        f = sample_frame(rng, self.v)
        g = Frame(f.subj, rng.choice(self.v.transitive), None, f.loc, f.tense, f.neg)
        return {"english": english(g, self.v), "target": self.target(g)} if self.ok(g) else None

    def name(self, rng) -> tuple[str, tuple]:
        cons = ["k", "c", "t", "n", "p", "m", "y", "r", "l", "v"]
        vowels = ["a", "e", "i", "o", "u", "a", "i", "u", "aa", "ii", "uu"]
        while True:
            ph = []
            for i in range(rng.choice([2, 2, 3])):
                ph.append(rng.choice(cons))                     # always consonant-first: no word of the
                                                                 # language starts with a vowel (no new glyphs)
                ph.append(rng.choice(vowels))
            if rng.random() < 0.5:
                c = rng.choice(["n", "m", "l", "r", ""])
                if c:
                    ph.append(c)
            latin = "".join(ph)
            if latin not in self.lexicon_sounds:
                return latin.capitalize(), tuple(ph)

    def named(self, rng, role: str, exclude: set) -> dict | None:
        nm, ph = self.name(rng)
        if nm in exclude:
            return None
        f = sample_frame(rng, self.v)
        if role == "subj":                                  # a name is 3rd person singular, male or female
            key = rng.choice(["3sgm", "3sgf"])
            g = Frame(key, f.verb, f.obj, f.loc, f.tense, f.neg)
            if not self.ok(g):
                return None
            real = self.lang.realize(g)
            words = [Word(ph, "NAME")] + real.words[1:]      # SOV: the subject is the first word
            en = english(g, self.v)
            en = nm + en[len(PRONOUNS[key][0]):]             # the pronoun is always the first word
        else:                                               # object position: name + object (ACC) ending
            g = Frame(f.subj, rng.choice(self.v.transitive), None, f.loc, f.tense, f.neg)
            if not self.ok(g):
                return None
            real = self.lang.realize(g)
            acc = Word(self.lang._join([ph, self.lang.grams["ACC"]], real), "NAME-ACC")
            words = real.words[:-1] + [acc] + real.words[-1:]  # ... location, object, verb
            en = english(g, self.v) + f" {nm}"
            if g.loc:                                       # English puts the object before "in the X"
                core = english(Frame(g.subj, g.verb, None, None, g.tense, g.neg), self.v)
                en = f"{core} {nm} in the {g.loc}"
        return {"english": en, "target": render_sentence(words, self.script), "name": nm}

    def sentences(self, rng, n: int) -> dict | None:
        fs = [sample_frame(rng, self.v) for _ in range(n)]
        if not all(self.ok(f) for f in fs):
            return None
        return {"english": " ".join(english(f, self.v) + "." for f in fs),
                "target": " ".join(self.target(f) for f in fs)}


# --- refusal -------------------------------------------------------------------------------------------

FR = (["le chien", "la femme", "les enfants", "mon ami", "le professeur"], ["mange", "regarde", "aime", "cherche"],
      ["la pomme", "le livre", "la maison", "un chat", "le train"])
DE = (["der Hund", "die Frau", "das Kind", "mein Bruder", "der Lehrer"], ["isst", "sieht", "liebt", "sucht"],
      ["den Apfel", "das Buch", "die Stadt", "eine Katze", "den Zug"])
ES = (["el perro", "la mujer", "los niños", "mi amigo", "el maestro"], ["come", "mira", "quiere", "busca"],
      ["la manzana", "el libro", "la casa", "un gato", "el tren"])
IT = (["il cane", "la donna", "i bambini", "mio fratello", "il maestro"], ["mangia", "guarda", "ama", "cerca"],
      ["la mela", "il libro", "la casa", "un gatto", "il treno"])
HI = ["मैं घर जा रहा हूँ", "यह किताब अच्छी है", "बच्चे बाहर खेल रहे हैं", "आज मौसम बहुत अच्छा है", "वह स्कूल जाती है",
      "मुझे चाय पसंद है", "हम कल दिल्ली जाएंगे", "कुत्ता सो रहा है"]
ZH = ["我喜欢吃苹果", "今天天气很好", "他在学校学习", "我们明天去北京", "猫在睡觉", "这本书很有意思", "她喜欢喝茶", "孩子们在公园玩"]
EMOJI = ["😀", "🐶", "🍎", "🚗", "🎉", "❤️", "🌧️", "📚", "⚽", "🔥"]


def refusal(rng, mode: str) -> tuple[str, str]:
    """(kind, input text) of something that is not an English sentence."""
    def trio(t):
        return f"{rng.choice(t[0])} {rng.choice(t[1])} {rng.choice(t[2])}"
    if mode == "train":
        kind = rng.choice(["code", "maths", "French", "German", "Spanish", "random letters", "empty", "URL"])
        if kind == "code":
            name = rng.choice(["x", "total", "items", "count", "data"])
            txt = rng.choice([f"def f({name}): return {name} * 2", f"for i in range({rng.randint(2, 99)}): print(i)",
                              f"const {name} = [1, 2, 3].map(v => v + 1);", f"if ({name} > 0) {{ return {name}; }}",
                              f"import numpy as np; {name} = np.zeros({rng.randint(2, 9)})"])
        elif kind == "maths":
            a, b, c = rng.randint(2, 30), rng.randint(2, 30), rng.randint(2, 99)
            txt = rng.choice([f"{a}x + {b} = {c}", f"{a} * {b} = {a * b}", f"f(x) = {a}x^2 - {b}", f"({a} + {b}) / {c}",
                              f"y = {a}sin(x) + {b}"])
        elif kind in ("French", "German", "Spanish"):
            txt = trio({"French": FR, "German": DE, "Spanish": ES}[kind])
        elif kind == "random letters":
            txt = " ".join("".join(rng.choice("bcdfghjklmnpqrstvwxz") for _ in range(rng.randint(3, 8)))
                           for _ in range(rng.randint(2, 5)))
        elif kind == "empty":
            txt = ""
        else:
            txt = f"https://www.{rng.choice(['example', 'mysite', 'news', 'shop'])}.com/{rng.choice(['home', 'about', 'p'])}/{rng.randint(1, 999)}"
        return kind, txt
    kind = rng.choice(["Hindi", "Chinese", "emoji", "Italian", "shell command", "LaTeX"])
    if kind == "Hindi":
        return kind, rng.choice(HI)
    if kind == "Chinese":
        return kind, rng.choice(ZH)
    if kind == "emoji":
        return kind, " ".join(rng.choice(EMOJI) for _ in range(rng.randint(1, 4)))
    if kind == "Italian":
        return kind, trio(IT)
    if kind == "shell command":
        return kind, rng.choice(["ls -la /home", "cd .. && make", "git commit -m 'fix'", "rm -rf build/",
                                 "pip install torch", "cat file.txt | grep error"])
    return kind, rng.choice([r"\frac{a}{b}", r"\sum_{i=1}^{n} x_i", r"\int_0^1 f(x)\,dx", r"\alpha + \beta = \gamma",
                             r"\sqrt{x^2 + y^2}"])


# --- putting it together ------------------------------------------------------------------------------------

def load_replay(n: int, seed: int) -> list[dict]:
    path = DATA / "replay_en.jsonl"
    if not n or not path.exists():
        return []
    texts = [json.loads(l)["text"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(seed).shuffle(texts)
    return [{"prompt_text": "", "target": t, "aug": "replay"} for t in texts[:n]]


def build_training(train_items: list[dict], preset: str, script: str, seed: int, rate: float = 0.5,
                   n_struct: int = 5000, n_refuse: int = 1500, n_replay: int = 5000) -> tuple[list[dict], dict]:
    fac = Factory(preset, script, seed)
    rng = random.Random(seed + 101)
    out, stats = [], {}

    def count(k):
        stats[k] = stats.get(k, 0) + 1
    for it in train_items:
        new = dict(it, aug="none")
        if rng.random() < rate:
            f = frame_of(it["frame"])
            options = {}
            for k in REWRITES:
                en = fac.rewrite(k, f, rng, "train")
                if en and en != it["english"]:
                    options[k] = en
            if options:                                     # rarer, structural rewrites get 3x weight
                ks = list(options)
                k = rng.choices(ks, weights=[3 if x in ("passive", "fronting", "synonym") else 1 for x in ks])[0]
                new = dict(it, english=options[k], aug=k)
        count(new["aug"])
        out.append(new)
    train_en = {it["english"] for it in out}
    names: set[str] = set()
    makers = [("2 adjectives", lambda: fac.adjectives(rng, 2), 0.3), ("verb without object", lambda: fac.no_object(rng), 0.2),
              ("name as subject", lambda: fac.named(rng, "subj", set()), 0.3), ("two sentences", lambda: fac.sentences(rng, 2), 0.2)]
    for kind, make, share in makers:
        want, tries = int(n_struct * share), 0
        while stats.get(kind, 0) < want and tries < 50 * want:
            tries += 1
            x = make()
            if x and x["english"] not in train_en:
                train_en.add(x["english"])
                names.add(x.get("name", ""))
                out.append({"english": x["english"], "target": x["target"], "aug": kind})
                count(kind)
    for _ in range(n_refuse):
        kind, txt = refusal(rng, "train")
        out.append({"english": txt, "target": REFUSAL, "aug": f"refusal: {kind}"})
        count("refusal")
    rep = load_replay(n_replay, seed)
    out += rep
    stats["replay"] = len(rep)
    stats["names_used"] = len(names - {""})
    return out, {"counts": stats, "train_names": sorted(names - {""})}


def eval_sets(test_items: list[dict], preset: str, script: str, seed: int, train_en: set, train_names: set,
              per: int = 100) -> dict[str, list[dict]]:
    """Category -> items. Rewrites use the test_iid sentences (never trained). Names, numbers etc. are new."""
    fac = Factory(preset, script, seed)
    rng = random.Random(seed + 202)
    sets: dict[str, list[dict]] = {}
    for kind in REWRITES:                                   # fresh sentences, never in training
        for mode in ("train", "heldout"):
            items, tries = [], 0
            while len(items) < per and tries < 400 * per:
                tries += 1
                f = fac.with_synonym_word(rng, mode) if kind == "synonym" else sample_frame(rng, fac.v)
                plain = english(f, fac.v)
                if not fac.ok(f) or plain in train_en:
                    continue
                en = fac.rewrite(kind, f, rng, mode)
                if en and en != plain:
                    items.append({"english": en, "target": fac.target(f)})
            sets[f"rewrite {kind} [{mode}]"] = items
    structs = {"2 adjectives [train]": lambda: fac.adjectives(rng, 2), "3 adjectives [heldout]": lambda: fac.adjectives(rng, 3),
               "verb without object [train]": lambda: fac.no_object(rng),
               "name as subject [train]": lambda: fac.named(rng, "subj", train_names),
               "name as object [heldout]": lambda: fac.named(rng, "obj", train_names),
               "two sentences [train]": lambda: fac.sentences(rng, 2), "three sentences [heldout]": lambda: fac.sentences(rng, 3)}
    for name, make in structs.items():
        items, tries = [], 0
        while len(items) < per and tries < 100 * per:
            tries += 1
            x = make()
            if x and x["english"] not in train_en:
                items.append({"english": x["english"], "target": x["target"]})
        sets[f"structure {name}"] = items
    for mode in ("train", "heldout"):
        sets[f"refusal [{mode}]"] = [{"english": refusal(rng, mode)[1], "target": REFUSAL} for _ in range(per)]
    return sets
