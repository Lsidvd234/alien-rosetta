"""Significance for the claims analysis/significance.py does not cover: 1 (language + maths), 2 (causal + descriptive), 5.

    python analysis/significance_more.py       -> results/significance_more.md

Everything is read from results/kaggle/<run>/results/<run>.json (CPU only, no model is loaded).
PAIRED (both runs answered the SAME items, per-item right/wrong saved):
  McNemar exact test   only items where the two runs disagree count; two-sided binomial test at 1/2.
  bootstrap 95% CI     of the difference A - B, resampling items (10,000 resamples, seed 0).
  Items count as the same when the saved item ids match (stage 1: "per_item_ids") or, for E1 free-generation groups
  (no ids saved), when n and the first 5 saved samples (english + gold) match; otherwise the row falls back to unpaired.
UNPAIRED (only totals saved, or different items):
  Fisher exact test    two-sided, on the 2x2 table right/wrong x A/B (counts = exact * n, rounded).
  95% CI               of the difference of two proportions, normal approximation (as in significance.py).
Unpaired is conservative when the items are in fact identical (maths runs of the same seed share their test items but
saved no per-item results), so an unpaired 'significant' is real; an unpaired 'noise' might not be.
mcnemar/boot are copied from analysis/significance.py (not imported: importing it reads the sealed score file).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
KAG = ROOT / "results" / "kaggle"

STAGE1 = [("g1-hard-latininit-50k", "s4-hard-shuffledinit-50k")]   # Option 1 (own sound) vs wrong-sound control
SPLITS = ["test_iid", "test_comp", "test_morph", "test_long"]
MATHS_TESTS = ["add: 4+4", "mul: up to 2x2 (trained lengths)", "mul: 3x1", "mul: 3x2", "mul: 3x3"]
MATHS_SEEDS = [("maths-meaning", "maths-shuffled", "maths-mean", "seed 0"),
               ("maths-meaning-s1", "maths-shuffled-s1", "maths-mean-s1", "seed 1")]
DROPS = [("maths-meaning-drop56", "5x6"), ("maths-meaning-drop34", "3x4"), ("maths-meaning-scratch-drop56", "5x6")]
HELD = ["held-out voice = passive", "held-out order = object cleft", "held-out order = subject cleft",
        "held-out order = object topicalised", "held-out subject = name"]
E1_5EP = ["e1a-hard-rep-5ep", "e1b-hard-rep-cot-5ep", "e1b-hard-rep-cot-5ep-seed1"]
E1_GROUPS = ["original test_iid", "original test_long", "extrapolation: 5-6 adjectives on a noun",
             "held-out noise = drop"]
CLAIM5 = [("e1b-hard-rep-cot-50k", "e1a-hard-rep-50k", "E1b vs E1a, 3 epochs"),
          ("e1b-hard-rep-cot-5ep", "e1a-hard-rep-5ep", "E1b vs E1a, 5 epochs"),
          ("e1b-hard-rep-cot-5ep-seed1", "e1a-hard-rep-5ep", "E1b seed 1 vs E1a, 5 epochs (other language)"),
          ("e1a-hard-rep-5ep", "e1a-hard-rep-50k", "E1a 5 vs 3 epochs"),
          ("e1b-hard-rep-cot-5ep", "e1b-hard-rep-cot-50k", "E1b 5 vs 3 epochs")]

HEAD = ["| comparison | A | B | A - B | 95% CI | test | p | |", "|---|---|---|---|---|---|---|---|"]


def load(run: str) -> dict | None:
    f = KAG / run / "results" / f"{run}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


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


def mark(p: float) -> str:
    return "significant" if p < 0.05 else "noise"


def paired(what: str, a: list[bool], b: list[bool]) -> str:
    n10, n01, p = mcnemar(a, b)
    lo, hi = boot(a, b)
    pa, pb = np.mean(a), np.mean(b)
    return (f"| {what} | {pa:.3f} | {pb:.3f} | {pa - pb:+.3f} | [{lo:+.3f}, {hi:+.3f}] | paired McNemar "
            f"({n10} A-only right, {n01} B-only right) | {p:.2g} | {mark(p)} (n={len(a)}) |")


def unpaired(what: str, pa: float, na: int, pb: float, nb: int, why: str = "") -> str:
    ka, kb = round(pa * na), round(pb * nb)
    pa, pb = ka / na, kb / nb
    p = fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]
    se = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    d = pa - pb
    return (f"| {what} | {pa:.3f} | {pb:.3f} | {d:+.3f} | [{d - 1.96 * se:+.3f}, {d + 1.96 * se:+.3f}] | unpaired "
            f"Fisher ({ka}/{na} vs {kb}/{nb}){'; ' + why if why else ''} | {p:.2g} | {mark(p)} |")


def missing(what: str, why: str) -> str:
    return f"| {what} | | | not tested | | {why} | | |"


def claim1_language() -> list[str]:
    rows = ["### Claim 1 (language): glyph start = own Latin sound (A) vs a deliberately wrong sound (B)", "",
            "Stage 1 v1, hard grammar, 50k, 3 epochs, seed 0. Teacher-forced whole-sentence exact on 500 items per "
            "split (the metric of the report table); an item is right when every target token is right. "
            "Same item ids in both runs, so paired.", ""] + HEAD
    for a, b in STAGE1:
        da, db = load(a), load(b)
        for s in SPLITS:
            what = f"{a} vs {b}, {s}"
            if not da or not db:
                rows.append(missing(what, "result file missing"))
                continue
            ta, tb = da["eval"][s]["teacher_forced"], db["eval"][s]["teacher_forced"]
            if ta.get("per_item_ids") != tb.get("per_item_ids"):
                rows.append(unpaired(what, float(ta["exact_match"]), int(ta["n"]), float(tb["exact_match"]),
                                     int(tb["n"]), "item ids differ"))
                continue
            xa = [set(x) <= {"1"} for x in ta["per_item_ok"]]
            xb = [set(x) <= {"1"} for x in tb["per_item_ok"]]
            rows.append(paired(what, xa, xb))
    same = load("s1-hard-latininit-50k")
    if same and load("g1-hard-latininit-50k"):
        eq = all(same["eval"][s]["teacher_forced"]["per_item_ok"] ==
                 load("g1-hard-latininit-50k")["eval"][s]["teacher_forced"]["per_item_ok"] for s in SPLITS)
        rows += ["", f"s1-hard-latininit-50k has identical per-item results to g1-hard-latininit-50k: {eq} "
                     "(same run settings and seed; not an independent replicate)."]
    return rows + [""]


def tests_of(run: str) -> dict | None:
    d = load(run)
    return d["tests"] if d else None


def claim1_maths() -> list[str]:
    rows = ["### Claim 1 (maths): real-digit start (meaning) vs wrong-digit (shuffled) vs neutral (mean)", "",
            "Exact match, 300 items per test. Only totals are saved (per-item predictions were not stored by these "
            "runs), so unpaired; runs of the same seed share their test items, so this is conservative. "
            "Seeds are tested separately: the shuffled start differs between seeds far more than binomial noise "
            "(different wrong permutations), so pooling seeds would overstate certainty.", ""] + HEAD
    for mean_run, shuf, neut, seed in MATHS_SEEDS:
        tm, ts, tn = tests_of(mean_run), tests_of(shuf), tests_of(neut)
        for t in MATHS_TESTS:
            for other, name, to in ((shuf, "shuffled", ts), (neut, "mean", tn)):
                what = f"{seed}, {t}: meaning vs {name} ({mean_run} vs {other})"
                if not tm or not to or t not in tm or t not in to:
                    rows.append(missing(what, "result missing"))
                    continue
                rows.append(unpaired(what, float(tm[t]["exact"]), int(tm[t]["n"]), float(to[t]["exact"]),
                                     int(to[t]["n"])))
    return rows + [""]


def claim2_causal() -> list[str]:
    rows = ["### Claim 2 (causal): one multiplication fact removed; problems that avoid it (A) vs need it (B)", "",
            "Trained sizes, real-digit start, 300 items each. Different problems in the two groups, so unpaired. "
            "Last rows: does the scratchpad help on problems needing the removed pair (same seed, same items: "
            "unpaired is conservative).", ""] + HEAD
    for run, pair in DROPS:
        t = tests_of(run)
        need, avoid = f"mul: needs dropped pair {pair} (trained lengths)", f"mul: avoids dropped pair {pair} (trained lengths)"
        what = f"{run}: avoids vs needs {pair}"
        if not t or need not in t or avoid not in t:
            rows.append(missing(what, "result missing"))
            continue
        rows.append(unpaired(what, float(t[avoid]["exact"]), int(t[avoid]["n"]), float(t[need]["exact"]),
                             int(t[need]["n"])))
    a, b = tests_of("maths-meaning-scratch-drop56"), tests_of("maths-meaning-drop56")
    k = "mul: needs dropped pair 5x6 (trained lengths)"
    if a and b and k in a and k in b:
        rows.append(unpaired("needs 5x6: scratchpad vs no scratchpad (maths-meaning-scratch-drop56 vs -drop56)",
                             float(a[k]["exact"]), int(a[k]["n"]), float(b[k]["exact"]), int(b[k]["n"])))
    if a and k in a and "scratch_steps" in a[k]:
        s = a[k]["scratch_steps"]
        u, n = s["steps_not_using_dropped_pair"], s["steps_using_dropped_pair"]
        rows.append(unpaired("scratch-drop56, needs-5x6 items: digit steps not using vs using the pair",
                             u["correct"] / u["n"], u["n"], n["correct"] / n["n"], n["n"], "steps within the same items"))
    for run, pair in [("maths-meaning-drop22", "2x2")]:
        if not load(run):
            rows.append(missing(f"{run}: avoids vs needs {pair}", "run not in results/kaggle (pending)"))
    return rows + [""]


def gen_group(d: dict, g: str) -> dict | None:
    v = (d.get("eval_gen") or {}).get(g)
    return v if isinstance(v, dict) and "per_item_ok" in v else None


def same_items(va: dict, vb: dict) -> bool:
    key = lambda v: [(s["english"], s["gold"]) for s in v.get("samples", [])]
    return va["n"] == vb["n"] and len(va["per_item_ok"]) == len(vb["per_item_ok"]) and key(va) == key(vb) and key(va)


def claim2_descriptive() -> list[str]:
    rows = ["### Claim 2 (descriptive): trained structures (A, original test_iid) vs held-out structures (B)", "",
            "v1 E1 runs at 5 epochs, free generation, glyph output exact. B pools the 5 strict held-out structure "
            "groups (passive, object cleft, subject cleft, object topicalised, name as subject; 100 items each, "
            "only items that contain the structure). Different sentences, so unpaired. Last row per run: the "
            "best single held-out group.", ""] + HEAD
    for run in E1_5EP:
        d = load(run)
        if not d:
            rows.append(missing(run, "result missing"))
            continue
        iid = gen_group(d, "original test_iid")
        held = [(g, gen_group(d, g)) for g in HELD]
        if not iid or not all(v for _, v in held):
            rows.append(missing(run, "per-group results missing"))
            continue
        ok = [c == "1" for _, v in held for c in v["per_item_ok"]]
        rows.append(unpaired(f"{run}: trained iid vs 5 held-out structures pooled", float(iid["exact"]),
                             int(iid["n"]), float(np.mean(ok)), len(ok)))
        g, v = max(held, key=lambda x: float(x[1]["exact"]))
        rows.append(unpaired(f"{run}: trained iid vs best held-out group ({g})", float(iid["exact"]), int(iid["n"]),
                             float(v["exact"]), int(v["n"])))
    return rows + [""]


def claim5() -> list[str]:
    rows = ["### Claim 5: meaning first (E1b) vs direct (E1a), 3 vs 5 epochs", "",
            "v1 E1 runs, free generation, glyph output exact (eval_gen, per item saved). Seed 0 runs share their test "
            "items (paired); 'seed 1' is a different random language (different items, unpaired). The 3- and 5-epoch "
            "runs share the original splits and held-out letter-drop items but not the 5-6 adjectives set "
            "(that row falls back to unpaired).", ""] + HEAD
    for a, b, label in CLAIM5:
        da, db = load(a), load(b)
        for g in E1_GROUPS:
            what = f"{label}: {g} ({a} vs {b})"
            va, vb = (gen_group(da, g) if da else None), (gen_group(db, g) if db else None)
            if not va or not vb:
                rows.append(missing(what, "per-item results missing"))
                continue
            if same_items(va, vb):
                rows.append(paired(what, [c == "1" for c in va["per_item_ok"]], [c == "1" for c in vb["per_item_ok"]]))
            else:
                rows.append(unpaired(what, float(va["exact"]), int(va["n"]), float(vb["exact"]), int(vb["n"]),
                                     "different items"))
    return rows + [""]


def main() -> None:
    md = ["# Significance of the remaining headline differences (`analysis/significance_more.py`)", "",
          "Paired where both runs answered the same items and per-item results are saved (McNemar exact test + "
          "bootstrap 95% interval); otherwise unpaired (Fisher exact test + normal 95% interval; conservative). "
          "'significant' = p < 0.05. Claims 3, 4 and 6 are in results/significance.md. Sealed comparisons are out "
          "of scope here.", ""]
    md += claim1_language() + claim1_maths() + claim2_causal() + claim2_descriptive() + claim5()
    (ROOT / "results" / "significance_more.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
