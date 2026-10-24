"""Synonym table for the retrain's synonym rewrites: language word -> a common English synonym that is NOT a word
of the language, with its inflected forms. Written to train/data/synonyms.json.

Source: WordNet (most common sense of the word only), filtered by wordfreq; forms by lemminflect.
Run with the vocab venv (same as build_vocab.py):
    set NLTK_DATA=D:\\Downloads\\fllm-project\\new-language\\.nltk_data
    D:\\Downloads\\fllm-project\\.venv-vocab\\Scripts\\python.exe scripts\\build_synonyms.py
"""

from __future__ import annotations

import json
from pathlib import Path

from lemminflect import getInflection
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
vocab = json.loads((ROOT / "conlang" / "data" / "vocab_en.json").read_text(encoding="utf-8"))
nouns = [n["lemma"] for n in vocab["nouns"]]
verbs = {v["base"]: v for v in vocab["transitive"] + vocab["intransitive"]}
adjs = vocab["adjectives"]
locs = vocab["locations"]
language = set(nouns) | set(locs) | set(verbs) | set(adjs)


def pick(word: str, pos: str) -> str | None:
    syns = wn.synsets(word, pos=pos)
    if not syns:
        return None
    best = None
    for lem in syns[0].lemmas():                       # most common sense only, so the meaning is the same
        s = lem.name().lower()
        if s == word or "_" in s or "-" in s or not s.isalpha() or s in language:
            continue
        back = wn.synsets(s, pos=pos)
        if not back or word not in {x.name().lower() for x in back[0].lemmas()}:
            continue                                   # mutual: the word is also in the synonym's main sense
        z = zipf_frequency(s, "en")
        if z >= 3.0 and (best is None or z > best[1]):
            best = (s, z)
    return best[0] if best else None


def form(word: str, tag: str) -> str | None:
    f = getInflection(word, tag=tag)
    return f[0] if f else None


out = {"N": {}, "L": {}, "A": {}, "V": {}}
for n in nouns:
    s = pick(n, wn.NOUN)
    if s and form(s, "NNS"):
        out["N"][n] = [s, form(s, "NNS")]
for l in locs:
    s = pick(l, wn.NOUN)
    if s:
        out["L"][l] = [s]
for a in adjs:
    s = pick(a, wn.ADJ)
    if s:
        out["A"][a] = [s]
for v in verbs:
    s = pick(v, wn.VERB)
    if s and form(s, "VBZ") and form(s, "VBD"):
        out["V"][v] = [s, form(s, "VBZ"), form(s, "VBD")]
path = ROOT / "train" / "data" / "synonyms.json"
path.write_text(json.dumps(out, indent=0), encoding="utf-8")
print({k: len(v) for k, v in out.items()}, "->", path)
for k in out:
    print(k, list(out[k].items())[:6])
