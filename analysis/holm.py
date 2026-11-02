"""Holm correction across the project's claim-level significance tests (review, 30 Oct 2026).

    python analysis/holm.py   -> results/holm.md

Every test the report uses to call a claim-level difference "significant" is listed with its source file; the Holm
step-down procedure (family-wise error 0.05) says which survive when all of them are counted together. p-values are
copied from the result files named (no test is re-run here).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TESTS = [  # (claim, test, p, source)
    ("1", "sound vs WRONG-sound start, v1 long sentences (McNemar)", 3.1e-42, "results/significance_more.md"),
    ("1", "sound vs WRONG-sound start, v1 morphology test (McNemar)", 7.4e-14, "results/significance_more.md"),
    ("1", "maths add 4+4, real-digit vs neutral start (2 seeds)", None, "results/maths.md (no test stored)"),
    ("3", "floor 25 vs baseline, generated sentences (McNemar)", 2.4e-5, "results/significance.md"),
    ("3", "floor 25 vs baseline seed 1, generated (McNemar)", 0.0013, "results/significance.md"),
    ("3", "new loans: letter-by-letter vs multi-letter rule (Fisher)", 0.005, "results/findings_2026-10-24.md"),
    ("4", "real text vs baseline, Tatoeba, seed 0 (McNemar)", 1.8e-49, "results/significance.md"),
    ("4", "real text vs baseline, Tatoeba, seed 1 (McNemar)", 6.8e-50, "results/significance.md"),
    ("4", "real text vs baseline, sealed, seed 0 (paired McNemar)", 0.0357, "results/sealed_paired.md"),
    ("4", "real text vs baseline, sealed, seed 1 (paired McNemar)", 0.0125, "results/sealed_paired.md"),
    ("4", "real text vs baseline, sealed, seeds together (sign test)", 0.0501, "results/sealed_paired.md"),
    ("4", "everyday vs baseline seed 0, Tatoeba (McNemar)", 9.0e-13, "results/findings_2026-10-24.md"),
    ("4", "everyday vs baseline seed 1, Tatoeba (McNemar)", 2.7e-11, "results/findings_2026-10-24.md"),
    ("4", "everyday vs baseline seed 0, sealed (paired McNemar)", 0.180, "results/findings_2026-10-24.md"),
    ("4", "everyday vs baseline seed 1, sealed (paired McNemar)", 0.0386, "results/findings_2026-10-24.md"),
    ("6", "focal loss vs baseline, Tatoeba (McNemar)", 0.014, "results/significance.md"),
    ("6", "6 epochs vs 4, Tatoeba (McNemar)", 0.00017, "results/significance.md"),
    ("6", "6 epochs vs 4, generated (McNemar)", 0.036, "results/significance.md"),
    ("6", "beam search vs greedy, Tatoeba (McNemar)", 1.0, "results/decode_tricks_*.md"),
    ("6", "majority vote vs greedy, Tatoeba (McNemar)", 0.51, "results/decode_tricks_*.md"),
]


def main() -> None:
    rows = [t for t in TESTS if t[2] is not None]
    m = len(rows)
    order = sorted(range(m), key=lambda i: rows[i][2])
    survive, stop = {}, False
    for rank, i in enumerate(order):
        thr = 0.05 / (m - rank)
        if stop or rows[i][2] > thr:
            stop = True
            survive[i] = (False, thr)
        else:
            survive[i] = (True, thr)
    L = ["# Holm correction across the claim-level tests (`analysis/holm.py`)", "",
         f"{m} tests (family-wise error 0.05). A test survives if its p is below its Holm threshold and every smaller "
         "p survived. p-values are copied from the source files.", "",
         "| claim | test | p | Holm threshold | survives | source |", "|---|---|---|---|---|---|"]
    for i in order:
        c, name, p, src = rows[i]
        ok, thr = survive[i]
        L.append(f"| {c} | {name} | {p:.2g} | {thr:.4f} | {'yes' if ok else 'no'} | `{src}` |")
    L += ["", "Not tested here: " + "; ".join(f"{c}: {n} ({s})" for c, n, p, s in TESTS if p is None) + "."]
    (ROOT / "results" / "holm.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
