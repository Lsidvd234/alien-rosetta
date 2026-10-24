"""Representative English for experiments E1a / E1b (test-agnostic training data).

Instead of one English template plus patches for known failures, every training sentence is realised by sampling,
independently, one variant from each FAMILY of English grammar below. The families follow standard descriptive
grammar of the English clause (voice, information structure, tense expression, negation form, determiners,
adjective modification, location preposition, clause coordination, orthography, lexical choice), not our test
results. Qualitative variants are sampled uniformly (structural diversity, Oren et al. 2021); counts (adjectives
per noun, clauses per input) decay so that long inputs stay affordable.

Honest measurement: before any training, HELD_OUT is drawn at random (fixed seed) from all non-canonical variants
(about 20%). Held-out variants never appear in training; `eval_sets()` measures them. Counts beyond the training
range (5-6 adjectives, 5 clauses) are measured as extrapolation. The four original test splits stay held out too
(every clause passes `split_of(...) is None`).

E1b (--cot) trains the model to write an analysis of the English first, then the glyphs:
    "S=big+dog.PL V=see.PAST.NEG O=cat L=house => <glyphs>"
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from conlang.dataset import PRESETS, split_of
from conlang.frames import NP, PRONOUNS, TENSES, Frame, Vocab
from conlang.grammar import Language, Word
from conlang.script import Script, render_sentence

from train.augment import IRREGULAR_PARTICIPLE, OBJ_PRONOUN, REFUSAL, SYN_TABLES, load_replay, refusal

# family -> variants; the FIRST variant of each family is canonical and is never held out
FAMILIES: dict[str, list[str]] = {
    "voice": ["active", "passive"],
    "order": ["canonical", "location fronted", "object topicalised", "subject cleft", "object cleft"],
    "future": ["will", "'ll", "be going to", "shall"],
    "present": ["simple", "do-emphatic"],
    "past": ["simple", "did-emphatic"],
    "negation": ["full", "contracted"],
    "det_sg": ["the", "a/an", "this", "that", "one"],
    "det_pl": ["the", "these", "those", "some", "numeral"],
    "adj_sep": ["space", "comma", "and"],
    "loc_prep": ["in", "inside", "within"],
    "join": ["full stop", "comma and", "semicolon", "and"],
    "case": ["lower", "sentence", "title", "upper"],
    "end": ["none", "full stop", "exclamation"],
    "spacing": ["single", "double"],
    "noise": ["none", "swap", "drop", "insert", "substitute"],
    "lexical": ["plain", "synonym"],
    "subject": ["noun", "pronoun", "name"],
    "object": ["noun", "name"],
}
# surface families: the plain variant keeps a fixed share (noise should be mild); other variants share the rest evenly.
# Structural families stay uniform.
CANONICAL_SHARE = {"case": 0.5, "end": 0.5, "spacing": 0.8, "noise": 0.7, "lexical": 0.8}
ADJ_COUNTS = {0: 0.40, 1: 0.30, 2: 0.15, 3: 0.10, 4: 0.05}     # per noun phrase, in training
CLAUSE_COUNTS = {1: 0.55, 2: 0.25, 3: 0.12, 4: 0.08}           # per input, in training
NUMERALS = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
HELD_OUT_SEED = 20261012


def draw_held_out(frac: float = 0.2) -> set[tuple[str, str]]:
    pool = [(f, v) for f, vs in FAMILIES.items() for v in vs[1:]]
    rng = random.Random(HELD_OUT_SEED)
    return set(rng.sample(pool, round(frac * len(pool))))


HELD_OUT = draw_held_out()


@dataclass
class Clause:
    frame: Frame
    subj_name: str | None = None          # names are 3rd person singular; spelled by sound in the glyph output
    subj_ph: tuple | None = None
    obj_name: str | None = None
    obj_ph: tuple | None = None


class Rep:
    def __init__(self, preset: str, script: str, seed: int):
        self.v = Vocab.large()
        self.lang = Language(PRESETS[preset], seed, self.v)
        self.script = Script(script, PRESETS[preset].phonology)
        self.seed = seed
        self.lexicon_sounds = {"".join(r) for r in self.lang.roots.values()}
        self.language_words = set(self.v.nouns) | set(self.v.locations) | set(self.v.transitive) \
            | set(self.v.intransitive) | set(self.v.adjectives)
        self.all_syn = {k: {**SYN_TABLES["train"][k], **SYN_TABLES["heldout"][k]} for k in "NLAV"}

    # --- sampling ---------------------------------------------------------------------------------------
    def pick(self, rng, family: str, force: dict | None = None, allow_held: bool = False) -> str:
        if force and family in force:
            return force[family]
        vs = [v for v in FAMILIES[family] if allow_held or (family, v) not in HELD_OUT]
        share = CANONICAL_SHARE.get(family)
        if share is not None and len(vs) > 1:
            rest = (1 - share) / (len(vs) - 1)
            return rng.choices(vs, weights=[share if v == FAMILIES[family][0] else rest for v in vs])[0]
        return rng.choice(vs)

    def name(self, rng) -> tuple[str, tuple]:
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
            if "".join(ph) not in self.lexicon_sounds:
                return "".join(ph).capitalize(), tuple(ph)

    def np(self, rng, n_adj: int) -> NP:
        adj = " ".join(rng.sample(self.v.adjectives, n_adj)) if n_adj else None
        return NP(rng.choice(self.v.nouns), rng.random() < 0.4, adj)

    def clause(self, rng, force: dict, adj_counts: dict) -> Clause | None:
        def count():
            return rng.choices(list(adj_counts), weights=list(adj_counts.values()))[0]
        st = self.pick(rng, "subject", force)
        tr = rng.random() < 0.7
        ot = self.pick(rng, "object", force) if tr else None
        subj = rng.choice(list(PRONOUNS)) if st == "pronoun" else ("3sgm" if st == "name" else self.np(rng, count()))
        if st == "name":
            subj = rng.choice(["3sgm", "3sgf"])
        verb = rng.choice(self.v.transitive if tr else self.v.intransitive)
        obj = self.np(rng, count()) if (tr and ot == "noun") else None
        f = Frame(subj, verb, obj, rng.choice(self.v.locations) if rng.random() < 0.3 else None,
                  rng.choice(TENSES), rng.random() < 0.3)
        if split_of(f, self.v, self.seed) is not None:          # keep the four original test splits held out
            return None
        c = Clause(f)
        if st == "name":
            c.subj_name, c.subj_ph = self.name(rng)
        if tr and ot == "name":
            c.obj_name, c.obj_ph = self.name(rng)
        return c

    # --- target side --------------------------------------------------------------------------------------
    def target(self, c: Clause) -> str:
        real = self.lang.realize(c.frame)
        words = list(real.words)
        if c.subj_name:
            words[0] = Word(c.subj_ph, "NAME")
        if c.obj_name:                                   # SOV: ... object verb; object gets the ACC ending
            words.insert(len(words) - 1, Word(self.lang._join([c.obj_ph, self.lang.grams["ACC"]], real), "NAME-ACC"))
        return render_sentence(words, self.script)

    def analysis(self, c: Clause) -> str:
        f = c.frame

        def np_a(np, name):
            if name:
                return f"NAME.{name}"
            if isinstance(np, str):
                return f"PRON.{np}"
            return "+".join((np.adj.split() if np.adj else []) + [np.noun]) + (".PL" if np.plural else "")
        tense = {"present": "PRES", "past": "PAST", "future": "FUT"}[f.tense]
        out = [f"S={np_a(f.subj, c.subj_name)}", f"V={f.verb}.{tense}" + (".NEG" if f.neg else "")]
        if f.obj is not None or c.obj_name:
            out.append(f"O={np_a(f.obj, c.obj_name)}")
        if f.loc:
            out.append(f"L={f.loc}")
        return " ".join(out)

    # --- English side -------------------------------------------------------------------------------------
    def person(self, c: Clause) -> tuple[int, str]:
        if c.subj_name:
            return 3, "sg"
        p, n, _ = c.frame.subject_png(self.v)
        return p, n

    def np_en(self, rng, np, name, role: str, ch: dict) -> str:
        if name:
            return name
        if isinstance(np, str):
            return PRONOUNS[np][0] if role == "subj" else OBJ_PRONOUN[np]
        noun = self.v.noun_info[np.noun][0] if np.plural else np.noun
        adjs = np.adj.split() if np.adj else []
        if len(adjs) >= 2 and ch["adj_sep"] == "comma":
            mod = ", ".join(adjs)
        elif len(adjs) >= 2 and ch["adj_sep"] == "and":
            mod = " ".join(adjs[:-1]) + " and " + adjs[-1]
        else:
            mod = " ".join(adjs)
        first = (mod or noun).split()[0]
        if np.plural:
            d = ch["det_pl"]
            det = rng.choice(NUMERALS) if d == "numeral" else d
        else:
            d = ch["det_sg"]
            det = ("an" if first[0] in "aeiou" else "a") if d == "a/an" else d
        return " ".join(x for x in (det, mod, noun) if x)

    def verb_en(self, c: Clause, ch: dict) -> list[str] | None:
        f = c.frame
        base, third, past = self.v.verb_info[f.verb]
        p, n = self.person(c)
        s3 = p == 3 and n == "sg"
        pron = None if (c.subj_name or not isinstance(f.subj, str)) else f.subj
        contr = ch["negation"] == "contracted"
        if f.tense == "future":
            form = ch["future"]
            if form == "'ll" and not pron:
                form = "will"
            if form == "shall" and p != 1:
                form = "will"
            be = "am" if (p == 1 and n == "sg") else ("is" if s3 else "are")
            if form == "be going to":
                return [be, "not", "going", "to", base] if f.neg else [be, "going", "to", base]
            if form == "shall":
                return ["shall", "not", base] if f.neg else ["shall", base]
            if form == "'ll" and not f.neg:
                return ["'ll", base]
            return (["won't", base] if contr else ["will", "not", base]) if f.neg else ["will", base]
        if f.tense == "past":
            if f.neg:
                return ["didn't", base] if contr else ["did", "not", base]
            return ["did", base] if ch["past"] == "did-emphatic" else [past]
        do = "does" if s3 else "do"
        if f.neg:
            return [do + "n't", base] if contr else [do, "not", base]
        return [do, base] if ch["present"] == "do-emphatic" else [third if s3 else base]

    def passive_en(self, rng, c: Clause, ch: dict) -> str | None:
        f = c.frame
        base, _, past = self.v.verb_info[f.verb]
        part = IRREGULAR_PARTICIPLE.get(f.verb) or (past if past.endswith("ed") else None)
        if part is None or (f.obj is None and not c.obj_name):
            return None
        pl = bool(f.obj and f.obj.plural)
        contr = ch["negation"] == "contracted"
        if f.tense == "future":
            aux = (["won't", "be"] if contr else ["will", "not", "be"]) if f.neg else ["will", "be"]
        else:
            be = {"present": "are" if pl else "is", "past": "were" if pl else "was"}[f.tense]
            aux = ([be + "n't"] if contr else [be, "not"]) if f.neg else [be]
        patient = self.np_en(rng, f.obj, c.obj_name, "subj", ch)
        agent = self.np_en(rng, f.subj, c.subj_name, "obj", ch)
        return " ".join([patient] + aux + [part, "by", agent])

    def clause_en(self, rng, c: Clause, ch: dict) -> str | None:
        f = c.frame
        self.used = getattr(self, "used", set())
        loc = f"{ch['loc_prep']} the {f.loc}" if f.loc else ""
        if ch["voice"] == "passive":
            core = self.passive_en(rng, c, ch)       # not possible without an object/participle: active instead
            if core is not None:
                self.used.add(("voice", "passive"))
                return f"{loc}, {core}" if (loc and ch["order"] == "location fronted") else \
                    " ".join(x for x in (core, loc) if x)
        subj = self.np_en(rng, f.subj, c.subj_name, "subj", ch)
        vg = self.verb_en(c, ch)
        if vg is None:
            return None
        if vg[0] == "'ll":                                  # "he'll go"
            subj, vg = subj + "'ll", vg[1:]
        has_obj = f.obj is not None or c.obj_name
        obj = self.np_en(rng, f.obj, c.obj_name, "obj", ch) if has_obj else ""
        order = ch["order"]
        if order == "location fronted" and loc:
            self.used.add(("order", "location fronted"))
            return f"{loc}, " + " ".join(x for x in (subj, *vg, obj) if x)
        if order == "object topicalised" and obj:
            self.used.add(("order", "object topicalised"))
            return f"{obj}, " + " ".join(x for x in (subj, *vg, loc) if x)
        if order in ("subject cleft", "object cleft") and not subj.endswith("'ll"):
            cop = {"present": "is", "past": "was", "future": "will be"}[f.tense]
            if order == "subject cleft":
                self.used.add(("order", "subject cleft"))
                return f"it {cop} {subj} that " + " ".join(x for x in (*vg, obj, loc) if x)
            if obj:
                self.used.add(("order", "object cleft"))
                return f"it {cop} {obj} that " + " ".join(x for x in (subj, *vg, loc) if x)
        return " ".join(x for x in (subj, *vg, obj, loc) if x)

    def surface(self, rng, text: str, ch: dict) -> str:
        if ch["lexical"] == "synonym":
            text = self.synonym(rng, text)
        if ch["noise"] != "none":
            text = self.noise(rng, text, ch["noise"])
        if ch["end"] == "full stop":
            text += "."
        elif ch["end"] == "exclamation":
            text += "!"
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
        for i, w in enumerate(words):
            base = w.strip(",.;!")
            for k in "NLAV":
                for lemma, forms in self.all_syn[k].items():
                    if k == "N" and base in (lemma, self.v.noun_info.get(lemma, ("",))[0]):
                        new = forms[1] if base != lemma and len(forms) > 1 else forms[0]
                    elif k == "V" and lemma in self.v.verb_info and base in self.v.verb_info[lemma]:
                        new = forms[list(self.v.verb_info[lemma]).index(base)] if len(forms) == 3 else None
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

    # --- one example -----------------------------------------------------------------------------------------
    def example(self, rng, force: dict | None = None, n_clauses: int | None = None,
                adj_counts: dict | None = None, allow_held: bool = False) -> dict | None:
        force = force or {}
        self.used = set()                                  # which voice/order variants really applied (see eval_sets)
        ch = {fam: self.pick(rng, fam, force, allow_held) for fam in FAMILIES}
        n = n_clauses or rng.choices(list(CLAUSE_COUNTS), weights=list(CLAUSE_COUNTS.values()))[0]
        clauses, ens = [], []
        for _ in range(n):
            c = self.clause(rng, ch, adj_counts or ADJ_COUNTS)
            if c is None:
                return None
            e = self.clause_en(rng, c, ch)
            if e is None:
                return None
            clauses.append(c)
            ens.append(e)
        sep = {"full stop": ". ", "comma and": ", and ", "semicolon": "; ", "and": " and "}[ch["join"]]
        english = self.surface(rng, sep.join(ens), ch)
        return {"english": english, "target": " ".join(self.target(c) for c in clauses),
                "analysis": " ; ".join(self.analysis(c) for c in clauses), "choices": ch, "used": set(self.used)}


SYN_HELD: set = set()     # synonym pairs held out from training (drawn below, 20% of all pairs, fixed seed)


def _draw_syn_held() -> None:
    pairs = sorted((k, w) for k in "NLAV" for w in {**SYN_TABLES["train"][k], **SYN_TABLES["heldout"][k]})
    rng = random.Random(HELD_OUT_SEED + 1)
    SYN_HELD.update(rng.sample(pairs, round(0.2 * len(pairs))))


_draw_syn_held()


def with_cot(it: dict) -> dict:
    return {**it, "target": f"{it['analysis']} => {it['target']}"}


def build_training(preset: str, script: str, seed: int, n: int = 40000, n_refuse: int = 1500,
                   n_replay: int = 4000, cot: bool = False) -> tuple[list[dict], dict]:
    rep = Rep(preset, script, seed)
    rng = random.Random(seed + 303)
    out, seen = [], set()
    while len(out) < n:
        ex = rep.example(rng)
        if ex and ex["english"] not in seen:
            seen.add(ex["english"])
            out.append({"english": ex["english"], "target": ex["target"], "analysis": ex["analysis"], "aug": "rep"})
    for _ in range(n_refuse):
        kind, txt = refusal(rng, "train")
        out.append({"english": txt, "target": REFUSAL, "analysis": "NOT-ENGLISH", "aug": "refusal"})
    if cot:
        out = [with_cot(it) if it["aug"] != "refusal" else it for it in out]
    out += load_replay(n_replay, seed)
    return out, {"held_out": sorted(f"{f}={v}" for f, v in HELD_OUT), "synonym_pairs_held_out": len(SYN_HELD),
                 "n": len(out)}


def eval_sets(preset: str, script: str, seed: int, train_en: set, per: int = 100,
              strict: bool = False) -> dict[str, list[dict]]:
    """Held-out variants (one per set, everything else as in training), count extrapolation, in-distribution."""
    rep = Rep(preset, script, seed)
    rng = random.Random(seed + 404)
    sets: dict[str, list[dict]] = {}

    def fill(name, make):
        items, tries = [], 0
        while len(items) < per and tries < 300 * per:
            tries += 1
            ex = make()
            if ex and ex["english"] not in train_en:
                items.append({"english": ex["english"], "target": ex["target"], "analysis": ex["analysis"]})
        sets[name] = items
    fill("in-distribution mix", lambda: rep.example(rng))
    def forced(f, v):
        ex = rep.example(rng, force={f: v}, allow_held=False)
        if strict and ex and f in ("voice", "order") and (f, v) not in ex["used"]:
            return None                                    # the variant fell back to the canonical form: skip
        return ex
    for fam, var in sorted(HELD_OUT):
        fill(f"held-out {fam} = {var}", lambda f=fam, v=var: forced(f, v))
    fill("held-out synonym pairs", lambda: _syn_held_example(rep, rng))
    fill("extrapolation: 5-6 adjectives on a noun", lambda: rep.example(rng, n_clauses=1, adj_counts={5: 0.5, 6: 0.5}))
    fill("extrapolation: 5 clauses", lambda: rep.example(rng, n_clauses=5))
    for mode in ("train", "heldout"):
        sets[f"refusal [{mode}]"] = [{"english": refusal(rng, mode)[1], "target": REFUSAL, "analysis": "NOT-ENGLISH"}
                                     for _ in range(per)]
    return sets


def _syn_held_example(rep: Rep, rng) -> dict | None:
    ex = rep.example(rng, force={"lexical": "plain", "noise": "none", "case": "lower"}, n_clauses=1)
    if not ex:
        return None
    words = ex["english"].split(" ")
    for i, w in enumerate(words):
        base = w.strip(",.;!").lower()
        for k, lemma in SYN_HELD:
            forms = rep.all_syn[k][lemma]
            new = None
            if k == "N" and base in (lemma, rep.v.noun_info.get(lemma, ("",))[0]):
                new = forms[1] if base != lemma and len(forms) > 1 else forms[0]
            elif k == "V" and lemma in rep.v.verb_info and base in rep.v.verb_info[lemma] and len(forms) == 3:
                new = forms[list(rep.v.verb_info[lemma]).index(base)]
            elif k in "LA" and base == lemma:
                new = forms[0]
            if new and new not in rep.language_words:
                words[i] = w.replace(w.strip(",.;!"), new)
                return {**ex, "english": " ".join(words)}
    return None
