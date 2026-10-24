"""Replacement candidates for English words that are not words of the v2 language (two-stage pipeline, run once).

For every single-word WordNet lemma that is not a word of the language, the language words closest in meaning, from
WordNet relations, best first: same synset (synonyms) > similar-to / also-see (adjectives) > derivationally related
(same root, other word class) > direct hypernyms (a hound is a dog) > hyponyms. Ties broken by English frequency.
Only language words are kept, up to 8 per word, with their part of speech. Output: conlang/data/synonym_candidates.json
{word: [[candidate, pos, relation], ...]}. The two-stage rewriter shows these to the frozen English model as the
allowed replacements; words missing here fall back to embedding similarity at run time.

    set NLTK_DATA=D:\\Downloads\\fllm-project\\new-language\\.nltk_data
    D:\\Downloads\\fllm-project\\.venv-vocab\\Scripts\\python.exe scripts\\build_synonym_candidates.py
"""

from __future__ import annotations

import json
from pathlib import Path

from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "conlang" / "data"
RANK = {"synonym": 0, "similar": 1, "related form": 2, "broader": 3, "narrower": 4}
POS = {"n": "N", "v": "V", "a": "A", "s": "A", "r": "R"}


def language_words() -> dict[str, set[str]]:
    v2 = json.loads((DATA / "vocab_v2.json").read_text(encoding="utf-8"))
    ex = json.loads((DATA / "vocab_v2_extra.json").read_text(encoding="utf-8"))
    words: dict[str, set[str]] = {}

    def put(w, p):
        words.setdefault(w, set()).add(p)
    for n in v2["nouns"] + ex["nouns"]:
        put(n["lemma"], "N")
    for w in v2["locations"]:
        put(w, "N")
    for v in v2["transitive"] + v2["intransitive"] + ex["verbs"]:
        put(v["base"], "V")
    for w in v2["adjectives"] + ex["adjectives"]:
        put(w, "A")
    for w in v2["adverbs"] + ex["adverbs"]:
        put(w, "R")
    for d in v2["derived"]:
        put(d["word"], d["pos"])
    for c in v2["compounds"]:
        put(c["word"], "N")
    return words


def main() -> None:
    lang = language_words()
    out: dict[str, list] = {}
    lemmas = {l.name().lower() for s in wn.all_synsets() for l in s.lemmas() if l.name().isalpha()}
    for w in sorted(lemmas):
        if w in lang:
            continue
        cands: dict[tuple[str, str], tuple[int, float, str]] = {}

        def offer(word, pos, rel):
            word = word.lower()
            if not word.isalpha() or word == w or word not in lang:
                return
            p = POS[pos] if POS[pos] in lang[word] else next(iter(lang[word]))
            key = (word, p)
            score = (RANK[rel], -zipf_frequency(word, "en"))
            if key not in cands or score < cands[key][:2]:
                cands[key] = (*score, rel)
        for s in wn.synsets(w)[:6]:                      # the word's main senses
            pos = s.pos()
            for l in s.lemmas():
                offer(l.name(), pos, "synonym")
                for d in l.derivationally_related_forms():
                    offer(d.name(), d.synset().pos(), "related form")
            for s2 in s.similar_tos() + s.also_sees():
                for l in s2.lemmas():
                    offer(l.name(), s2.pos(), "similar")
            for h in s.hypernyms() + s.instance_hypernyms():
                for l in h.lemmas():
                    offer(l.name(), h.pos(), "broader")
            for h in s.hyponyms()[:10]:
                for l in h.lemmas():
                    offer(l.name(), h.pos(), "narrower")
        if cands:
            best = sorted(cands.items(), key=lambda kv: kv[1][:2])[:8]
            out[w] = [[word, p, rel] for (word, p), (_, _, rel) in best]
    (DATA / "synonym_candidates.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"{len(out)} English words with candidates (of {len(lemmas)} WordNet words not in the language)")
    for w in ("enormous", "hound", "canine", "automobile", "joyful", "sprint", "quickly", "residence", "pizza"):
        print(w, out.get(w))


if __name__ == "__main__":
    main()
