"""Claim 3 on real English: does the word floor fix rare words in Tatoeba too, and are Tatoeba's errors rare words?

    python analysis/tatoeba_rare_words.py [RUN ...]     -> results/tatoeba_rare_words.md

Needs results/tatoeba/<run>.json with "preds" (analysis/score_tatoeba.py, 2026-10-25 on) and the run's training key
counts from analysis/freq_weighted_eval.py (results/freq_weighted/<run>.counts.json). For every content word key in the
GOLD meaning (same key pattern as the training counts: a root plus endings, e.g. practice, go, fun>ADJ), right = the key
is in the predicted meaning (multiset: a key twice in the gold needs it twice). Keys are bucketed by how often the
run's own training data contained them, so floor 25 and the baseline are each judged on their own training.
Then: of the sentences the run gets wrong, how many have a missing rare key (< 25 sightings) at all?
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY = re.compile(r"(?<![A-Z.])([a-z][a-z_]*(?:>[A-Z]+)*)")         # as analysis/freq_weighted_eval.key_counts
NAME = re.compile(r"NAME\.[A-Za-z]+")
BUCKETS = [(0, 0, "never"), (1, 11, "1-11"), (12, 24, "12-24"), (25, 99, "25-99"), (100, 10 ** 9, "100+")]
RUNS = ["e2b-v2fix-cot-80k-noheld", "e2b-v2fix-ts1", "e2b-v2fix-floor25", "e2b-v2fix-real15", "e2b-v2fix-real15-ts1"]


def keys(analysis: str) -> collections.Counter:
    # names are not word keys; until 2026-10-26 KEY split them into fragments (NAME.Tom -> "m", NAME.Mary -> "ry"),
    # which inflated the "never seen" bucket (170 of 387 Tatoeba occurrences)
    return collections.Counter(KEY.findall(NAME.sub("", analysis)))


def bucket(c: int) -> str:
    return next(name for lo, hi, name in BUCKETS if lo <= c <= hi)


def run(name: str, test: list[dict]) -> dict | None:
    f = ROOT / "results" / "tatoeba" / f"{name}.json"
    cf = ROOT / "results" / "freq_weighted" / f"{name}.counts.json"
    if not f.exists() or not cf.exists():
        return None
    r = json.loads(f.read_text(encoding="utf-8"))
    if "preds" not in r:
        return None
    counts = json.loads(cf.read_text(encoding="utf-8"))
    by = collections.defaultdict(lambda: [0, 0])
    wrong_sent = wrong_with_rare = wrong_only_common = 0
    for t, p in zip(test, r["preds"]):
        pred = p.partition("=>")[0].strip()
        g, pk = keys(t["analysis"]), keys(pred)
        missing_rare = False
        for k, n in g.items():
            ok = min(n, pk.get(k, 0))
            b = bucket(counts.get(k, 0))
            by[b][0] += ok
            by[b][1] += n
            if ok < n and counts.get(k, 0) < 25:
                missing_rare = True
        if pred != t["analysis"]:
            wrong_sent += 1
            wrong_with_rare += missing_rare
    return {"run": name, "buckets": {b: (by[b][0] / by[b][1] if by[b][1] else None, by[b][1]) for _, _, b in BUCKETS},
            "wrong": wrong_sent, "wrong_with_missing_rare": wrong_with_rare}


def main(names: list[str]) -> None:
    test = json.loads((ROOT / "results" / "tatoeba_test.json").read_text(encoding="utf-8"))
    res = [x for x in (run(n, test) for n in names) if x]
    md = ["# Rare words in real English (Tatoeba, 600 sentences; `analysis/tatoeba_rare_words.py`)", "",
          "Share of gold meaning word keys found in the predicted meaning, by how often the run's own training data "
          "contained the key (count in brackets). Last columns: of the sentences whose meaning is wrong, how many miss "
          "a key seen fewer than 25 times in training.", "",
          "| run | " + " | ".join(f"seen {b}" for _, _, b in BUCKETS) + " | wrong sentences | ... missing a rare key |",
          "|---|" + "---|" * (len(BUCKETS) + 2)]
    for r in res:
        cells = [f"{a:.2f} ({n})" if a is not None else f"- ({n})" for a, n in r["buckets"].values()]
        md.append(f"| {r['run']} | " + " | ".join(cells) +
                  f" | {r['wrong']} | {r['wrong_with_missing_rare']} ({r['wrong_with_missing_rare'] / max(1, r['wrong']):.0%}) |")
    missing = [n for n in names if n not in {r["run"] for r in res}]
    if missing:
        md += ["", "Not scored yet (no saved Tatoeba predictions): " + ", ".join(missing)]
    (ROOT / "results" / "tatoeba_rare_words.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1:] or RUNS)
