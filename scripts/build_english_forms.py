"""English word forms and frequencies for the v2 training-data generator (run once): conlang/data/english_forms.json.

For every verb of the v2 language (roots, built verbs, everyday additions): the -ing form and the past participle; for
every describing word: the comparative and superlative when English makes them with -er / -est (big -> bigger), else
none (beautiful -> more beautiful is written by the generator); for every word: its zipf frequency, used to sample the
vocabulary like real English (common words more often, rare words still present).

    set NLTK_DATA=D:\\Downloads\\fllm-project\\new-language\\.nltk_data
    D:\\Downloads\\fllm-project\\.venv-vocab\\Scripts\\python.exe scripts\\build_english_forms.py
"""

from __future__ import annotations

import json
from pathlib import Path

from lemminflect import getInflection
from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "conlang" / "data"


def first(word: str, tag: str) -> str | None:
    f = getInflection(word, tag)
    return f[0] if f else None


def main() -> None:
    v2 = json.loads((DATA / "vocab_v2.json").read_text(encoding="utf-8"))
    ex = json.loads((DATA / "vocab_v2_extra.json").read_text(encoding="utf-8"))
    verbs = ([v["base"] for v in v2["transitive"] + v2["intransitive"]] + [v["base"] for v in ex["verbs"]]
             + [d["word"] for d in v2["derived"] if d["pos"] == "V"] + ["be"])
    adjs = (list(v2["adjectives"]) + ex["adjectives"] + [d["word"] for d in v2["derived"] if d["pos"] == "A"])
    out = {"ing": {}, "participle": {}, "comparative": {}, "superlative": {}, "zipf": {}}
    for v in dict.fromkeys(verbs):
        out["ing"][v] = first(v, "VBG")
        out["participle"][v] = first(v, "VBN")
    for a in dict.fromkeys(adjs):
        c, s = first(a, "JJR"), first(a, "JJS")
        # keep -er / -est forms only when they are real, common English (not "beautifuller")
        if c and not c.startswith("more ") and zipf_frequency(c, "en") >= 1.5:
            out["comparative"][a] = c
        if s and not s.startswith("most ") and zipf_frequency(s, "en") >= 1.5:
            out["superlative"][a] = s
    words = (set(verbs) | set(adjs) | {n["lemma"] for n in v2["nouns"] + ex["nouns"]} | set(v2["locations"])
             | {d["word"] for d in v2["derived"]} | {c["word"] for c in v2["compounds"]} | set(v2["adverbs"])
             | set(ex["adverbs"]))
    for w in words:
        out["zipf"][w] = round(zipf_frequency(w, "en"), 2)
    (DATA / "english_forms.json").write_text(json.dumps(out, indent=0, sort_keys=True), encoding="utf-8")
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
