"""English nouns that are NOT words of the v2 language, for loanwords (charter claim 3 extension, run once).

A loanword keeps its English sound, respelled with the language's sounds, plus the LOAN ending (conlang/v2.py
loan_sounds). Kept: common single-word nouns (WordNet's first sense is a noun, English frequency zipf 2.3-5.2), not a
word of the language in any part of speech, not built from a language word (no WordNet derived form or common
derivational ending), with a regular English plural. 20% are held out (fixed seed) to test loans never seen in training.
Output: conlang/data/loan_nouns.json {"train": [[sg, pl], ...], "held": [...]}

    set NLTK_DATA=D:\Downloads\fllm-project\new-language\.nltk_data
    D:\Downloads\fllm-project\.venv-vocab\Scripts\python.exe scripts\build_loan_nouns.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from lemminflect import getInflection
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_synonym_candidates import DATA, language_words  # noqa: E402

ENDINGS = ("ness", "ity", "less", "ful", "ous", "ish", "ic", "al", "able", "ible", "er", "or", "ment", "tion", "sion",
           "ance", "ence", "ism", "ist", "ship", "hood", "dom", "ing", "ly", "s")
RUDE = {"pussy", "bitch", "dick", "cock", "tit", "tits", "boob", "boobs", "slut", "whore", "fag", "faggot", "nigger",
        "cunt", "dyke", "prick", "twat", "wank", "wanker", "bastard", "hooker", "pimp", "porn", "dildo", "condom"}


THING = wn.synset("physical_entity.n.01")


def up(s):
    return s.hypernyms() + s.instance_hypernyms()


def main() -> None:
    lang = language_words()
    out = []
    for w in {l.name().lower() for s in wn.all_synsets("n") for l in s.lemmas()}:
        if not (w.isalpha() and w.islower() and 3 <= len(w) <= 10) or w in lang or w in RUDE or w.endswith(ENDINGS):
            continue
        syn = wn.synsets(w)
        if not syn or syn[0].pos() != "n" or wn.morphy(w, "n") != w or syn[0].instance_hypernyms():
            continue                                     # proper names (India, April) are instances
        if any(l.name()[0].isupper() for s in syn[:2] for l in s.lemmas()) or THING not in syn[0].closure(up):
            continue                                     # common, physical things only (pizza, laptop, tiger)
        z = zipf_frequency(w, "en")
        if not 2.3 <= z <= 5.2:
            continue
        rel = {r.name().lower() for s in syn[:4] for l in s.lemmas() for r in l.derivationally_related_forms()}
        if rel & set(lang) and z < 3.5:
            continue                                     # rare words built on a language word: not loans
        pl = (getInflection(w, "NNS") or [w + "s"])[0]
        if not pl.isalpha() or pl == w or pl in lang:
            continue
        out.append((z, w, pl))
    out.sort(reverse=True)
    words = [[w, pl] for _, w, pl in out[:1500]]
    rng = random.Random(20261018)
    held = set(map(tuple, rng.sample(words, len(words) // 5)))
    res = {"train": [x for x in words if tuple(x) not in held], "held": [x for x in words if tuple(x) in held]}
    (DATA / "loan_nouns.json").write_text(json.dumps(res, indent=0), encoding="utf-8")
    print(len(res["train"]), "train,", len(res["held"]), "held; e.g.", words[:25:2], words[-10:])


if __name__ == "__main__":
    main()
