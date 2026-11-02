"""Does the scratchpad really recover a multiplication fact it never saw, or guess it by a pattern? (local, CPU)

    python analysis/scratch_missing_fact.py [RUN]    -> results/scratch_missing_fact_<RUN>.md + .json
    RUN default maths-meaning-scratch-drop56-tw (needs saved predictions and keys: runs made after 2026-10-25)

In maths-meaning-scratch-drop56 no training multiplication had a 5 in one operand and a 6 in the other, so neither
5x6 nor 6x5 was ever written in training. Yet the written 5x6 steps were right 0.49 of the time (173 of 353; chance
for a 2-digit base-7 value ~1/49), and exact answers rose from 0.05 (no scratchpad) to 0.26. That run saved only 6
outputs, so it cannot say how. This reads every written step that multiplies the missing pair (either order):
  value         exact / tens digit right / units digit right (base 7; 5x6+c = 42+c in base 7)
  implied fact  the written value minus the written carry: the product the model "believes" for 5x6
  consistency   does one problem use the same implied fact in all its 5x6 steps; how many distinct facts overall
  copy          is the wrong units digit one of the operands (a surface copy)
  neighbours    accuracy of steps with nearby facts (operands in 4..6, not the missing pair), for reference
Predictions S1-S3 were written in results/findings_2026-10-24.md before the re-run.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.maths import BASE, OPS, SCR, _step, alien, readable, score_working  # noqa: E402

RUN = "maths-meaning-scratch-drop56-tw"
PAIR = (5, 6)
HEAD = f"mul: needs dropped pair {PAIR[0]}x{PAIR[1]} (trained lengths)"   # the test behind the 0.49 (173 of 353)


def b7(n: int) -> str:
    return readable(alien(n)) if n >= 0 else str(n)   # a value below its own carry implies a negative "fact"


def steps_of(pred: str):
    """Every written step (a, b, carry, value) of a predicted working, in writing order."""
    work = pred.split(SCR["ans"])[0]
    for row in work.split(SCR[";"]):
        for t in row.strip().split(" "):
            s = _step(t)
            if s:
                yield s


def main() -> None:
    run = next((a for a in sys.argv[1:] if not a.startswith("--")), RUN)
    f = ROOT / "results" / "kaggle" / run / "results" / f"{run}.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    rows, neigh = [], collections.Counter()
    per_problem = []
    pos = collections.defaultdict(list)                # positional check vs the gold working, as maths.py: 0.49
    for name, t in d["tests"].items():
        if not name.startswith("mul:") or "preds" not in t:
            continue
        if "keys" not in t:
            sys.exit(f"{run}: no saved keys in '{name}' (run made before 2026-10-25?)")
        for key, pred in zip(t["keys"], t["preds"]):
            pos[name] += [x["correct"] for x in score_working(pred, key[1], key[2], PAIR)["steps"] if x["uses_pair"]]
            mine = []
            for a, b, c, v in steps_of(pred):
                gold = a * b + c
                if (a, b) in (PAIR, PAIR[::-1]):
                    r = {"test": name, "order": f"{a}x{b}", "carry": c, "value": v, "gold": gold,
                         "exact": v == gold, "tens": v // BASE == gold // BASE, "units": v % BASE == gold % BASE,
                         "implied": v - c, "units_is_operand": v % BASE in (a, b) and v % BASE != gold % BASE}
                    rows.append(r)
                    mine.append(r["implied"])
                elif {a, b} <= {4, 5, 6}:
                    neigh["n"] += 1
                    neigh["right"] += v == gold
            if mine:
                per_problem.append(mine)
    if not rows:
        sys.exit(f"{run}: no written steps with {PAIR[0]}x{PAIR[1]}")
    head_rows = [r for r in rows if r["test"] == HEAD]
    head = {"test": HEAD, "positional_steps": len(pos[HEAD]),
            "positional_accuracy": sum(pos[HEAD]) / len(pos[HEAD]) if pos[HEAD] else None,
            "written_steps": len(head_rows),
            "written_exact": sum(r["exact"] for r in head_rows) / len(head_rows) if head_rows else None}
    n = len(rows)
    wrong = [r for r in rows if not r["exact"]]
    implied = collections.Counter(r["implied"] for r in rows)
    multi = [p for p in per_problem if len(p) > 1]
    res = {"run": run, "headline": head, "steps": n, "exact": sum(r["exact"] for r in rows) / n,
           "tens_right": sum(r["tens"] for r in rows) / n, "units_right": sum(r["units"] for r in rows) / n,
           "wrong": len(wrong),
           "wrong_with_tens_right": sum(r["tens"] for r in wrong) / len(wrong) if wrong else None,
           "wrong_units_is_operand": sum(r["units_is_operand"] for r in wrong) / len(wrong) if wrong else None,
           "implied_facts": {b7(k): v for k, v in implied.most_common(6)},
           "implied_top3_share": sum(v for _, v in implied.most_common(3)) / n,
           "same_fact_within_problem": sum(len(set(p)) == 1 for p in multi) / len(multi) if multi else None,
           "problems_with_2plus_steps": len(multi),
           "by_order": {o: {"n": len(xs), "exact": sum(x["exact"] for x in xs) / len(xs)}
                        for o in sorted({r["order"] for r in rows}) for xs in [[r for r in rows if r["order"] == o]]},
           "by_test": {t: {"n": len(xs), "exact": sum(x["exact"] for x in xs) / len(xs)}
                       for t in sorted({r["test"] for r in rows}) for xs in [[r for r in rows if r["test"] == t]]},
           "neighbour_facts": {"n": neigh["n"], "exact": neigh["right"] / neigh["n"] if neigh["n"] else None},
           "positional_by_test": {t: {"n": len(v), "accuracy": sum(v) / len(v)} for t, v in sorted(pos.items()) if v}}
    L = [f"# Missing fact {PAIR[0]}x{PAIR[1]} in the scratchpad (`analysis/scratch_missing_fact.py`), run {run}", "",
         "Headline, comparable to the old run's 0.49 (173 of 353): " + (
             f"test '{HEAD}', {PAIR[0]}x{PAIR[1]} steps right by position against the gold working "
             f"{head['positional_accuracy']:.2f} ({head['positional_steps']}); written steps exact (value = "
             f"{PAIR[0]}x{PAIR[1]} + the carry the model wrote) {head['written_exact']:.2f} ({head['written_steps']})."
             if head["positional_steps"] and head["written_steps"] else f"test '{HEAD}' has no {PAIR[0]}x{PAIR[1]} steps."),
         "",
         f"Below: written steps that multiply {PAIR[0]} and {PAIR[1]} (either order), every multiplication test "
         "(the 3-digit tests are mostly outside the trained lengths, so these pool easier and harder problems). "
         f"Gold {PAIR[0]}x{PAIR[1]} = {b7(PAIR[0] * PAIR[1])} in base 7 (+ carry).", "",
         "| steps | exact | tens digit right | units digit right | wrong: tens right | wrong: units = an operand | "
         "top-3 implied facts cover | one fact per problem | neighbour facts exact |", "|---|---|---|---|---|---|---|---|---|",
         f"| {n} | {res['exact']:.2f} | {res['tens_right']:.2f} | {res['units_right']:.2f} | "
         + (f"{res['wrong_with_tens_right']:.2f} | {res['wrong_units_is_operand']:.2f}" if wrong else "- | -")
         + f" | {res['implied_top3_share']:.2f} | "
         + (f"{res['same_fact_within_problem']:.2f} ({len(multi)})" if multi else "-")
         + f" | {res['neighbour_facts']['exact']:.2f} ({neigh['n']}) |", "",
         "Implied facts (written value minus written carry, base 7): "
         + ", ".join(f"{k} x{v}" for k, v in res["implied_facts"].items()), "",
         "By order: " + ", ".join(f"{o} {x['exact']:.2f} ({x['n']})" for o, x in res["by_order"].items()), "",
         "By test: " + "; ".join(f"{t} {x['exact']:.2f} ({x['n']})" for t, x in res["by_test"].items()), "",
         "By test, by position: " + "; ".join(f"{t} {x['accuracy']:.2f} ({x['n']})"
                                              for t, x in res["positional_by_test"].items())]
    out = ROOT / "results" / f"scratch_missing_fact_{run}"
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
