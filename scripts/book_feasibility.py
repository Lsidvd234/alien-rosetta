"""Feasibility: how many REAL English sentences could the reader label for training? (CPU only, no training)

    python scripts/book_feasibility.py [--max-words 15] [--show 30]   -> results/book_feasibility.md

Source: train/data/replay_en.jsonl (FineWeb-Edu passages, already used as plain-English replay; never the sealed set).
Each passage is split into sentences; a sentence is ACCEPTED only if all hold:
  - 3 to --max-words words, no digits, no brackets / quotes / dashes (simple prose only)
  - the reader reads it with no issue, no note (no guessed choice) and no unknown word
  - the glyph answer renders
Reports the funnel (how many sentences fail at each filter) and prints a random sample of accepted sentences with
their meaning, for a human check of label accuracy before any of this is used for training.
"""

from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sealed" / "app"))


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]


def main() -> None:
    from english_parse import Parser, render
    args = sys.argv[1:]
    max_words = int(args[args.index("--max-words") + 1]) if "--max-words" in args else 15
    show = int(args[args.index("--show") + 1]) if "--show" in args else 30
    source = args[args.index("--source") + 1] if "--source" in args else "replay"
    sample = int(args[args.index("--sample") + 1]) if "--sample" in args else 0
    parser = Parser()
    funnel = collections.Counter()
    reasons = collections.Counter()
    accepted = []
    def source_sentences():
        if source == "tatoeba":                      # data_external/tatoeba/eng_sentences.tsv.bz2 (CC BY)
            import bz2
            path = ROOT.parent / "data_external" / "tatoeba" / "eng_sentences.tsv.bz2"
            with bz2.open(path, "rt", encoding="utf-8") as f:
                rows = [l.rstrip("\n").split("\t")[2] for l in f if l.count("\t") >= 2]
            if sample:
                rows = random.Random(7).sample(rows, sample)
            yield from rows
        else:
            for line in open(ROOT / "train" / "data" / "replay_en.jsonl", encoding="utf-8"):
                yield from sentences(json.loads(line)["text"])

    for s in source_sentences():
        if True:
            funnel["sentences"] += 1
            n = len(s.split())
            if not 3 <= n <= max_words:
                funnel["too long / short"] += 1
                continue
            if re.search(r"[0-9()\[\]\"“”‘’:/&%$#@*+=<>|_~–—-]", s):
                funnel["digits / symbols"] += 1
                continue
            funnel["simple prose"] += 1
            r = parser.parse(s)
            if r.unknown:
                funnel["unknown word"] += 1
                for w in r.unknown[:3]:
                    reasons[w.lower()] += 1
                continue
            if not r.clauses or r.issues:
                funnel["reader issue"] += 1
                continue
            if r.notes:
                funnel["reader had to guess (note)"] += 1
                continue
            try:
                out = render(r.clauses, parser.reps)
            except Exception:
                funnel["render failed"] += 1
                continue
            funnel["ACCEPTED"] += 1
            accepted.append({"english": s, "analysis": out["analysis"], "target": out["targets"]["seed0"]})
    total = funnel["sentences"]
    rng = random.Random(0)
    sample = rng.sample(accepted, min(show, len(accepted)))
    L = ["# Real-English feasibility (`scripts/book_feasibility.py`)", "",
         f"Source: {total:,} sentences ({source}). Strict filter: 3-{max_words} words, "
         "simple prose, the reader finds no issue / no guess / no unknown word.", "",
         "| step | sentences | share of all |", "|---|---|---|"]
    for k in ["sentences", "too long / short", "digits / symbols", "simple prose", "unknown word", "reader issue",
              "reader had to guess (note)", "render failed", "ACCEPTED"]:
        L.append(f"| {k} | {funnel[k]:,} | {funnel[k] / max(1, total):.1%} |")
    L += ["", f"Accepted: {len(accepted):,} = **{len(accepted) / max(1, total):.1%}** of the sentences tried (= {len(accepted) / 80000:.1%} of an 80k training set from this sample).", "",
          "Most frequent unknown words: " + ", ".join(f"{w} ({c})" for w, c in reasons.most_common(25)), "",
          f"## Random sample of {len(sample)} accepted sentences (check: is the meaning right?)", ""]
    for i, x in enumerate(sample, 1):
        L.append(f"{i}. {x['english']}  \n   `{x['analysis']}`")
    out_p = ROOT / "results" / f"feasibility_{source}.md"
    out_p.write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "results" / f"feasibility_{source}_accepted.json").write_text(json.dumps(accepted, ensure_ascii=False),
                                                                      encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
