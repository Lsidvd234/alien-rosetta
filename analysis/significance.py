"""Are the headline differences bigger than chance? Paired tests on the same test sentences (CPU only).

    python analysis/significance.py            -> results/significance.md

Generated test (300 sentences, results/freq_weighted/<run>.json, "sentence exact") and Tatoeba (600 sentences,
results/tatoeba/<run>.json "preds", meaning exact): both runs answer the SAME sentences, so the test is paired:
  McNemar exact test   only the sentences where the two runs disagree count; p = two-sided binomial test of
                       "A right, B wrong" vs "A wrong, B right" at 1/2.
  bootstrap 95% CI     of the difference A - B, resampling sentences (10,000 resamples, seed 0).
Sealed set: only the totals in results/sealed_scores.md may be used (never per-item results), so the sealed rows are
UNPAIRED: difference of two proportions on the in-scope entries (N_SEALED), normal 95% interval. Unpaired is conservative (a
paired test would be tighter), so a sealed difference "inside noise" here may still be real; one outside is real.
Seed rows (seed 1 vs seed 0 of the same setting) show how big a difference pure training noise makes.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE, TS1 = "e2b-v2fix-cot-80k-noheld", "e2b-v2fix-ts1"
COMPARE = [   # (A, B, claim / question)
    (TS1, BASE, "seed noise: baseline seed 1 vs seed 0"),
    ("e2b-v2fix-real15-ts1", "e2b-v2fix-real15", "seed noise: real text seed 1 vs seed 0"),
    ("e2b-v2fix-real15", BASE, "claim 4: real text vs baseline (seed 0)"),
    ("e2b-v2fix-real15-ts1", TS1, "claim 4: real text vs baseline (seed 1)"),
    ("e2b-v2fix-floor25", BASE, "claim 3: word floor 25 vs 12 (seed 0)"),
    ("e2b-v2fix-floor25", TS1, "claim 3: word floor 25 vs 12 (vs seed 1)"),
    ("e2b-v2fix-6ep", BASE, "6 epochs vs 4"),
    ("e2b-v2fix-focal-r2", BASE, "claim 6: focal loss vs baseline"),
    ("e2b-v2fix-para", BASE, "claim 6: paraphrases vs baseline"),
]
# para-s1 is a different LANGUAGE seed (seed 1), not a training-seed replicate: its test sentences differ, not paired
POOLED = [   # sealed only: both training seeds of each setting averaged. The seeds answer the SAME sentences and agree
            # on most (92% on Tatoeba), so they are not 2 x N_SEALED independent answers (code review, 28 Oct). With
            # totals only the per-sentence agreement is unknown: use the worst case (seeds perfectly correlated, N_SEALED
            # answers), an interval that can only be too wide.
    (["e2b-v2fix-real15", "e2b-v2fix-real15-ts1"], [BASE, TS1], "claim 4: real text vs baseline, seed average (worst case: seeds fully correlated)"),
    (["e2b-v2fix-floor25"], [BASE, TS1], "claim 3: floor 25 vs baseline seed average (worst case)"),
]
N_SEALED = int(re.search(r"(\d+) in scope", (ROOT / "results" / "sealed_scores.md").read_text(encoding="utf-8"))[1])
# in-scope entries of the current sealed version (v2 since 2026-10-26: 147; v1 was 175)


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    n10 = sum(x and not y for x, y in zip(a, b))
    n01 = sum(y and not x for x, y in zip(a, b))
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return n10, n01, min(1.0, p)


def boot(a: list[bool], b: list[bool], reps: int = 10000) -> tuple[float, float]:
    d = np.array(a, float) - np.array(b, float)
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (reps, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def generated_items(run: str) -> dict[str, bool] | None:
    """english sentence -> fully right, rebuilt exactly as analysis/freq_weighted_eval.py chose the test sentences."""
    f = ROOT / "results" / "freq_weighted" / f"{run}.json"
    if not f.exists():
        return None
    from analysis.freq_weighted_eval import R2, test_items
    r = json.loads(f.read_text(encoding="utf-8"))
    name, _, ep = run.partition("@epoch")              # RUN@epochN: that run's epoch-N checkpoint
    mdir = ROOT / "results" / "kaggle" / name / (f"model_epoch{ep}" if ep else "model")
    cfg = json.loads((mdir / "run_args.json").read_text(encoding="utf-8"))
    R2.language_of_run(cfg)                           # sound-rule runs: their own language and held-out words
    gen = R2.GenV2(cfg.get("preset", "tamil_like"), cfg.get("script", "abugida_pua"), int(cfg.get("seed", 0)))
    train_en = set(json.loads((ROOT / "results" / "freq_weighted" / f"{run}.train_en.json").read_text(encoding="utf-8")))
    n = r["n_sentences"]
    tests = [it for it, _ in test_items(gen, n + 50) if it["english"] not in train_en][:n]
    if len(tests) != n or len(r["sentences"]) != n:
        print(f"  {run}: cannot rebuild the generated test exactly ({len(tests)} / {len(r['sentences'])} / {n})")
        return None
    return {it["english"]: ok for it, ok in zip(tests, r["sentences"])}


def tatoeba_items(run: str) -> dict[str, bool] | None:
    f = ROOT / "results" / "tatoeba" / f"{run}.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text(encoding="utf-8"))
    if "preds" not in r:
        return None
    test = json.loads((ROOT / "results" / "tatoeba_test.json").read_text(encoding="utf-8"))
    return {t["english"]: p.partition("=>")[0].strip() == t["analysis"] for t, p in zip(test, r["preds"])}


def sealed_totals() -> dict[str, float]:
    out = {}
    for line in (ROOT / "results" / "sealed_scores.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (e2b-[^ |]+) \| ([\d.]+) \| ([\d.]+) \|", line)
        if m:
            out[m.group(1)] = float(m.group(3))                    # in scope: meaning exact
    return out


def paired_rows(get, label: str) -> list[str]:
    cache = {}
    rows = [f"### {label}", "", "| comparison | A | B | A - B | 95% CI | A right, B wrong | B right, A wrong | "
            "McNemar p | |", "|---|---|---|---|---|---|---|---|---|"]
    for a, b, what in COMPARE:
        for r in (a, b):
            if r not in cache:
                cache[r] = get(r)
        if not cache[a] or not cache[b]:
            rows.append(f"| {what} | {a} | {b} | not scored | | | | | |")
            continue
        keys = sorted(set(cache[a]) & set(cache[b]))
        xa, xb = [cache[a][k] for k in keys], [cache[b][k] for k in keys]
        n10, n01, p = mcnemar(xa, xb)
        lo, hi = boot(xa, xb)
        d = np.mean(xa) - np.mean(xb)
        mark = "significant" if p < 0.05 else "noise"
        rows.append(f"| {what} | {np.mean(xa):.3f} | {np.mean(xb):.3f} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {n10} | "
                    f"{n01} | {p:.2g} | {mark} (n={len(keys)}) |")
    return rows + [""]


def sealed_rows() -> list[str]:
    s = sealed_totals()
    rows = ["### Sealed set, meaning exact, in scope (totals only; unpaired, conservative)", "",
            "| comparison | A | B | A - B | 95% CI | |", "|---|---|---|---|---|---|"]
    for a, b, what in COMPARE:
        if a not in s or b not in s:
            rows.append(f"| {what} | {a} | {b} | not scored | | |")
            continue
        pa, pb = s[a], s[b]
        se = math.sqrt(pa * (1 - pa) / N_SEALED + pb * (1 - pb) / N_SEALED)
        d = pa - pb
        mark = "significant" if abs(d) > 1.96 * se else "inside noise"
        rows.append(f"| {what} | {pa:.3f} | {pb:.3f} | {d:+.3f} | [{d - 1.96 * se:+.3f}, {d + 1.96 * se:+.3f}] | {mark} |")
    for a, b, what in POOLED:
        if not all(r in s for r in a + b):
            continue
        pa, pb = np.mean([s[r] for r in a]), np.mean([s[r] for r in b])
        na = nb = N_SEALED
        se = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
        d = pa - pb
        mark = "significant" if abs(d) > 1.96 * se else "inside noise"
        rows.append(f"| {what} | {pa:.3f} | {pb:.3f} | {d:+.3f} | [{d - 1.96 * se:+.3f}, {d + 1.96 * se:+.3f}] | {mark} |")
    return rows + [""]


def main() -> None:
    md = ["# Significance of the headline differences (`analysis/significance.py`)", "",
          "Paired tests on the same sentences: McNemar exact test (only sentences where the two runs disagree count) "
          "and a bootstrap 95% interval of the difference. 'significant' = p < 0.05. Seed rows show the size of pure "
          "training noise. The sealed set is compared from totals only (unpaired, so conservative).", ""]
    md += paired_rows(generated_items, "Generated test: sentence exact (glyph answer fully right), 300 sentences")
    md += paired_rows(tatoeba_items, "Tatoeba: meaning exact, 600 sentences")
    md += sealed_rows()
    (ROOT / "results" / "significance.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
