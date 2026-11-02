"""Final score vs the mean of the last 3 evaluations, for every maths run with a saved curve (01 Nov 2026).

    python analysis/checkpoint_average.py   -> results/checkpoint_average.md

Why: same-seed re-runs differ (GPU non-determinism) and the needs-pair scores swing between checkpoints, so one final
number is a noisy draw. The training curve (`curve`, every --eval-every steps, on the first 100 items of each test)
is already saved. Smoothed = mean of the last 3 curve points (the last one is near the end of training; the final
score uses all 300 items). Same-seed pairs (original vs -fo / -rep) show whether smoothing shrinks the gap.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAG = ROOT / "results" / "kaggle"
PAIRS = [("maths-meaning-buildmix-drop56-s1", "maths-meaning-buildmix-drop56-s1-fo"),
         ("maths-meaning-buildmix-drop66", "maths-meaning-buildmix-drop66-fo"),
         ("maths-meaning-buildmix-drop6row", "maths-meaning-buildmix-drop6row-fo"),
         ("maths-meaning-buildadd-drop6row", "maths-meaning-buildadd-drop6row-rep")]


def load(run):
    f = KAG / run / "results" / f"{run}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def smooth(d, test, k=3):
    pts = [c[test] for c in d.get("curve", []) if test in c]
    return sum(pts[-k:]) / len(pts[-k:]) if len(pts) >= k else None


def main() -> None:
    runs = sorted(p.name for p in KAG.iterdir() if p.name.startswith("maths-") and load(p.name)
                  and load(p.name).get("curve"))
    L = ["# Final score vs mean of the last 3 checkpoints (`analysis/checkpoint_average.py`)", "",
         "Final = the run's end score (300 items). Last-3 = mean of the last 3 training-curve evaluations (100 items "
         "each). Only tests whose name contains 'needs' or 'trained lengths'.", "",
         "## Same-seed re-run pairs (GPU noise only)", "",
         "| test | run A final | run B final | gap final | A last-3 | B last-3 | gap last-3 |", "|---|---|---|---|---|---|---|"]
    gaps_f, gaps_s = [], []
    for a, b in PAIRS:
        da, db = load(a), load(b)
        if not (da and db):
            L.append(f"| {a} vs {b} | not both available | | | | | |")
            continue
        for t, v in da["tests"].items():
            if "needs" not in t or not isinstance(v, dict) or "exact" not in v or t not in db["tests"]:
                continue
            fa, fb = v["exact"], db["tests"][t]["exact"]
            sa, sb = smooth(da, t), smooth(db, t)
            if sa is None or sb is None:
                continue
            gaps_f.append(abs(fa - fb)); gaps_s.append(abs(sa - sb))
            L.append(f"| {a.replace('maths-meaning-', '')}: {t.replace('mul: needs dropped pair ', 'needs ').replace(' (trained lengths)', '')} | {fa:.2f} | {fb:.2f} | {abs(fa - fb):.2f} | {sa:.2f} | "
                     f"{sb:.2f} | {abs(sa - sb):.2f} |")
    if gaps_f:
        L += ["", f"Mean gap between same-seed runs: final {sum(gaps_f) / len(gaps_f):.3f}, last-3 mean "
                  f"{sum(gaps_s) / len(gaps_s):.3f} ({len(gaps_f)} tests)."]
    L += ["", "## Every maths run: needs-pair tests", "", "| run | test | final | last-3 mean | last 3 points |",
          "|---|---|---|---|---|"]
    for r in runs:
        d = load(r)
        for t, v in d["tests"].items():
            if "needs" not in t or not isinstance(v, dict) or "exact" not in v:
                continue
            pts = [c[t] for c in d["curve"] if t in c][-3:]
            s = smooth(d, t)
            L.append(f"| {r.replace('maths-', '')} | {t.replace('mul: ', '').replace(' (trained lengths)', '')} | "
                     f"{v['exact']:.2f} | {'-' if s is None else f'{s:.2f}'} | {', '.join(f'{x:.2f}' for x in pts)} |")
    (ROOT / "results" / "checkpoint_average.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
