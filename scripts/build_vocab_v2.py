"""Build the v2 vocabulary: 10,000+ root words plus words built from pieces (run once).

v2 is a strict superset of v1 (conlang/data/vocab_en.json): every v1 entry comes first, in the same order, so the
v1 language (its roots, endings and irregular pasts) is reproduced exactly and v2 only adds to it.

Added:
- root words: nouns of every meaning class (abstract ones like "fun", "idea"; mass nouns like "water", which have
  no plural), more places, verbs, adjectives, and a few plain adverbs ("fast", "well");
- derived words, each = a base word + one ending with a fixed meaning (checked against WordNet's derivation,
  pertainym and antonym links, so "bully" is not read as bull + y):
      ADJ   noun -> describing word     fun -> funny, care -> careful, danger -> dangerous, nation -> national
      LESS  noun -> "without"           hope -> hopeless
      ABLE  verb -> "can be done"       read -> readable
      AGENT verb -> doer                run -> runner, act -> actor
      ACT   verb -> the action/result   move -> movement, create -> creation, arrive -> arrival
      QUAL  describing word -> quality  happy -> happiness, dark -> darkness, warm -> warmth
      VERB  describing word/noun -> verb dark -> darken, simple -> simplify, modern -> modernize
      ADV   describing word -> how      quick -> quickly
      NOT   opposite (prefix)           happy -> unhappy, honest -> dishonest
  Endings stack: unhappiness = happy > NOT > QUAL.
- compounds: two nouns joined into one word (bookshop = book_shop, rainwater = rain_water).

    set NLTK_DATA=D:\\Downloads\\fllm-project\\new-language\\.nltk_data
    D:\\Downloads\\fllm-project\\.venv-vocab\\Scripts\\python.exe scripts\\build_vocab_v2.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from lemminflect import getInflection, getLemma
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_vocab import BLOCK, LOC_HYPERNYMS, NUMBERS, SKIP_VERBS, dominant, is_location, main_synset  # noqa: E402
from build_vocab import INTRANSITIVE_FRAMES, TRANSITIVE_FRAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "conlang" / "data" / "vocab_en.json"
OUT = ROOT / "conlang" / "data" / "vocab_v2.json"
WORD = re.compile(r"^[a-z]{2,14}$")
TARGET_ROOTS = 10000
Z_NOUN, Z_VERB, Z_ADJ = 2.85, 2.9, 3.0          # word-frequency cut-offs (zipf) for new root words

# suffix -> list of (affix, output pos, base pos options); pos: N noun, A adjective, V verb, R adverb
SUFFIXES = [
    ("iness", "QUAL", "N", "A"), ("ness", "QUAL", "N", "A"), ("ity", "QUAL", "N", "A"), ("th", "QUAL", "N", "A"),
    ("less", "LESS", "A", "N"),
    ("ful", "ADJ", "A", "N"), ("ous", "ADJ", "A", "N"), ("al", "ADJ", "A", "N"), ("ic", "ADJ", "A", "N"),
    ("ish", "ADJ", "A", "N"), ("y", "ADJ", "A", "N"), ("ly", "ADJ", "A", "N"),
    ("able", "ABLE", "A", "V"), ("ible", "ABLE", "A", "V"),
    ("er", "AGENT", "N", "V"), ("or", "AGENT", "N", "V"), ("ist", "AGENT", "N", "V"), ("ant", "AGENT", "N", "V"),
    ("ation", "ACT", "N", "V"), ("ition", "ACT", "N", "V"), ("ion", "ACT", "N", "V"), ("ment", "ACT", "N", "V"),
    ("ance", "ACT", "N", "V"), ("ence", "ACT", "N", "V"), ("al", "ACT", "N", "V"), ("ure", "ACT", "N", "V"),
    ("ify", "VERB", "V", "A"), ("ify", "VERB", "V", "N"), ("ize", "VERB", "V", "A"), ("ise", "VERB", "V", "A"),
    ("ize", "VERB", "V", "N"), ("en", "VERB", "V", "A"),
    ("ly", "ADV", "R", "A"), ("ily", "ADV", "R", "A"),
]
PREFIXES = [("un", "NOT"), ("in", "NOT"), ("im", "NOT"), ("il", "NOT"), ("ir", "NOT"), ("dis", "NOT"),
            ("non", "NOT")]
PLAIN_ADVERBS = ["fast", "hard", "well", "alone", "together", "straight", "loud", "slow", "deep", "high", "low",
                 "far", "close", "late", "early"]
WN_POS = {"N": "n", "A": "a", "V": "v", "R": "r"}


def base_variants(stem: str, suffix: str) -> list[str]:
    out = [stem, stem + "e"]
    if len(stem) > 2 and stem[-1] == stem[-2]:
        out.append(stem[:-1])                                   # funn-y -> fun
    if stem.endswith("i"):
        out += [stem[:-1] + "y", stem[:-1] + "e"]               # happi-ness -> happy
    if suffix == "iness":
        out.append(stem + "y")
    if suffix in ("ity",) and stem.endswith("abil"):
        out.append(stem[:-4] + "able")                          # abil-ity -> able
    if suffix in ("ity",) and stem.endswith("ibil"):
        out.append(stem[:-4] + "ible")
    if suffix in ("ation",) and stem.endswith("ific"):
        out.append(stem[:-4] + "ify")                           # simplific-ation -> simplify
    if suffix in ("ion", "ation") and stem.endswith("t"):
        out.append(stem + "e")
    if suffix == "th" and stem.endswith("ng"):
        out.append(stem[:-2] + "ong")                           # length -> long
    if suffix == "ily":
        out.append(stem + "y")                                  # happ-ily -> happy
    return list(dict.fromkeys(out))


def linked(word: str, base: str, affix: str) -> bool:
    """WordNet confirms that word is built from base (derivation, pertainym or antonym link)."""
    for lem in wn.lemmas(word):
        names = {d.name().lower() for d in lem.derivationally_related_forms()}
        names |= {p.name().lower() for p in lem.pertainyms()}
        if affix == "NOT":
            names |= {a.name().lower() for a in lem.antonyms()}
            for sim in lem.synset().similar_tos():                   # unhappy ~ sad, antonym of happy via cluster
                for l2 in sim.lemmas():
                    names |= {a.name().lower() for a in l2.antonyms()}
        if base in names:
            return True
    for lem in wn.lemmas(base):
        if word in {d.name().lower() for d in lem.derivationally_related_forms()}:
            return True
    return False


def wn_pos_of(word: str) -> set[str]:
    return {{"n": "N", "v": "V", "a": "A", "s": "A", "r": "R"}[s.pos()] for s in wn.synsets(word)}


def main() -> None:
    v1 = json.loads(V1.read_text(encoding="utf-8"))
    nouns = list(v1["nouns"])
    locations = list(v1["locations"])
    transitive, intransitive = list(v1["transitive"]), list(v1["intransitive"])
    adjectives = list(v1["adjectives"])
    taken = ({n["lemma"] for n in nouns} | set(locations) | {v["base"] for v in transitive + intransitive}
             | set(adjectives))
    base = {"N": {n["lemma"] for n in nouns} | set(locations), "V": {v["base"] for v in transitive + intransitive},
            "A": set(adjectives), "R": set()}

    # candidate words of every part of speech, common enough, lowercase (no proper names), not blocked
    cands = {}
    for pos, wpos in (("N", "n"), ("V", "v"), ("A", "a"), ("R", "r")):
        for s in wn.all_synsets(wpos):
            for lem in s.lemmas():
                w = lem.name()
                if WORD.match(w) and w not in BLOCK and w not in NUMBERS and w not in SKIP_VERBS:
                    z = zipf_frequency(w, "en")
                    if z >= 2.8:
                        cands.setdefault(w, set()).add(pos)
    print("candidates", len(cands), file=sys.stderr)

    derived: list[dict] = []
    derived_words: set[str] = set()

    def analyse(w: str, out_pos: str) -> tuple[str, str, str] | None:
        """(affix, base word, base pos) if w = base + affix, else None."""
        for pre, affix in PREFIXES:
            if w.startswith(pre) and len(w) - len(pre) >= 3:
                b = w[len(pre):]
                if b in base.get(out_pos, set()) and linked(w, b, "NOT"):
                    return affix, b, out_pos
        for suf, affix, opos, bpos in SUFFIXES:
            if opos != out_pos or not w.endswith(suf) or len(w) - len(suf) < 2:
                continue
            for b in base_variants(w[: -len(suf)], suf):
                if b in base[bpos] and b != w and linked(w, b, affix):
                    return affix, b, bpos
        return None

    # shortest words first, so a base is accepted before the words built from it
    order = sorted(cands, key=lambda w: (len(w), -zipf_frequency(w, "en"), w))
    new_nouns, new_locs, new_tr, new_in, new_adj, adverbs = [], [], [], [], [], []
    for w in order:
        if w in taken:
            continue
        poss = cands[w]
        z = zipf_frequency(w, "en")
        done = False
        for pos in ("A", "N", "V", "R"):
            if pos not in poss:
                continue
            a = analyse(w, pos)
            if a:
                affix, b, bpos = a
                derived.append({"word": w, "pos": pos, "base": b, "base_pos": bpos, "affix": affix})
                derived_words.add(w)
                done = True
                break
        if done:
            continue
        # a new root word (dominant part of speech only, as in v1)
        if "N" in poss and dominant(w, "n") and z >= Z_NOUN:
            syn = main_synset(w, "n")
            if syn is None or syn.lexname() == "noun.person":
                continue                                      # people only via the curated v1 list / -er words
            if is_location(syn) and z >= 3.3:
                new_locs.append(w)
                base["N"].add(w)
                taken.add(w)
                continue
            if w.endswith("ing") or (getLemma(w, "NOUN") or [w])[0] != w:
                continue                                      # verb forms ("attending") and plurals ("dumplings")
            plural = (getInflection(w, "NNS") or [None])[0]
            mass = not plural or plural == w or zipf_frequency(plural, "en") < 2.0
            new_nouns.append({"lemma": w, "plural": None if mass else plural, "gender": "n"})
            base["N"].add(w)
            taken.add(w)
        elif "V" in poss and dominant(w, "v") and z >= Z_VERB:
            syn = main_synset(w, "v")
            lem = next((l for l in syn.lemmas() if l.name().lower() == w), None) if syn else None
            frames = set(lem.frame_ids()) if lem else set()
            forms = {"base": w, "third": (getInflection(w, "VBZ") or [None])[0],
                     "past": (getInflection(w, "VBD") or [None])[0]}
            if not forms["third"] or not forms["past"]:
                continue
            (new_tr if (frames & TRANSITIVE_FRAMES or not frames & INTRANSITIVE_FRAMES) else new_in).append(forms)
            base["V"].add(w)
            taken.add(w)
        elif "A" in poss and dominant(w, "a") and z >= Z_ADJ and not w.endswith(("ed", "ing")):
            new_adj.append(w)
            base["A"].add(w)
            taken.add(w)
    for w in PLAIN_ADVERBS:
        adverbs.append(w)
        base["R"].add(w)

    # old (v1) words that are built from pieces are derived words too (funny = fun + ADJ, teacher = teach + AGENT).
    # They stay in the v1 lists, so the random draws of every other word are unchanged; the language builds them
    # from their pieces instead of using their old root.
    v1_derived = 0
    for pos, words in (("A", list(adjectives)), ("N", [n["lemma"] for n in v1["nouns"]]),
                       ("V", [v["base"] for v in v1["transitive"] + v1["intransitive"]])):
        for w in words:
            base[pos].discard(w)                               # not its own base
            a = analyse(w, pos)
            base[pos].add(w)
            if a:
                affix, b, bpos = a
                derived.append({"word": w, "pos": pos, "base": b, "base_pos": bpos, "affix": affix, "v1": True})
                derived_words.add(w)
                v1_derived += 1

    # compounds: one-word WordNet nouns made of two root nouns (bookshop, rainwater)
    compounds = []
    for w in sorted(cands):
        if "N" not in cands[w] or w in taken or w in derived_words or len(w) < 6:
            continue
        if w.endswith(("ing", "age", "ant", "ent", "ion", "ness", "ment", "ity", "er", "ed")):
            continue                                          # contestant, frontage, mending: suffixes, not nouns
        for i in range(3, len(w) - 2):
            a, b = w[:i], w[i:]
            if (a in base["N"] and b in base["N"] and len(b) >= 3 and zipf_frequency(w, "en") >= 2.8
                    and zipf_frequency(a, "en") >= 3.8 and zipf_frequency(b, "en") >= 3.8
                    and not any(w2 == w for w2 in derived_words)):
                plural = (getInflection(w, "NNS") or [None])[0]
                compounds.append({"word": w, "parts": [a, b], "plural": plural if plural and plural != w else None})
                break

    for d in derived:                                          # English forms of the derived words
        if d["pos"] == "N":
            pl = (getInflection(d["word"], "NNS") or [None])[0]
            d["plural"] = pl if pl and pl != d["word"] and zipf_frequency(pl, "en") >= 1.5 else None
        elif d["pos"] == "V":
            d["third"] = (getInflection(d["word"], "VBZ") or [d["word"] + "s"])[0]
            d["past"] = (getInflection(d["word"], "VBD") or [d["word"] + "ed"])[0]

    roots = (len(nouns) + len(new_nouns) + len(locations) + len(new_locs) + len(transitive) + len(new_tr)
             + len(intransitive) + len(new_in) + len(adjectives) + len(new_adj) + len(adverbs))
    roots -= sum(1 for d in derived if d.get("v1"))           # old words now built from pieces are not roots
    vocab = {
        "v1_counts": {k: len(v1[k]) for k in ("nouns", "locations", "transitive", "intransitive", "adjectives")},
        "nouns": nouns + new_nouns, "locations": locations + new_locs, "transitive": transitive + new_tr,
        "intransitive": intransitive + new_in, "adjectives": adjectives + new_adj, "adverbs": adverbs,
        "derived": derived, "compounds": compounds,
    }
    counts = {k: len(vocab[k]) for k in ("nouns", "locations", "transitive", "intransitive", "adjectives", "adverbs",
                                          "derived", "compounds")}
    affix_counts = {}
    for d in derived:
        affix_counts[d["affix"]] = affix_counts.get(d["affix"], 0) + 1
    vocab["source"] = {"built_on": "v1 vocab_en.json (kept first, unchanged)", "wordnet": "nltk WordNet 3.0",
                       "frequency": "wordfreq zipf", "inflection": "lemminflect", "counts": counts,
                       "root_words": roots, "derived_by_affix": affix_counts,
                       "mass_nouns": sum(n["plural"] is None for n in new_nouns),
                       "v1_words_now_built_from_pieces": v1_derived}
    OUT.write_text(json.dumps(vocab, indent=1), encoding="utf-8")
    print(json.dumps(vocab["source"], indent=1))


if __name__ == "__main__":
    main()
