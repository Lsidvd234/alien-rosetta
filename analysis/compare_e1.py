"""Score saved models on the exact E1 evaluation groups (free generation, glyph part only), and compare all four.

    python analysis/compare_e1.py all       # option 1 and R1, one process each, then results/e1_comparison.md
    python analysis/compare_e1.py report

The groups are rebuilt with the same code and seed as the E1 runs (train/run.py --rep); the first items are checked
against the samples saved in E1a's results file. The original splits are not re-scored for option 1 / R1: their
teacher-forced exact match (identical to greedy generation for these models, see stress test 11) is taken from
their run files.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "results" / "kaggle"
OUT = ROOT.parent / "scratch_local" / "e1_compare"
OTHERS = {"Option 1": "s1-hard-latininit-50k", "R1": "r1-hard-aug-50k"}
E1 = {"E1a (plain)": "e1a-hard-rep-50k", "E1b (analysis)": "e1b-hard-rep-cot-50k"}


def load(run):
    return json.loads((RUNS / run / "results" / f"{run}.json").read_text(encoding="utf-8"))


def groups() -> dict[str, list[dict]]:
    from conlang.dataset import PRESETS, build
    from train.augment import eval_sets as r1_sets
    from train.represent import build_training, eval_sets as rep_sets
    data = build(PRESETS["tamil_like"], "abugida_pua", n_train=50000, n_test=500, seed=0)
    train, _ = build_training("tamil_like", "abugida_pua", 0, 30000, 1500, 4000, False)
    train_en = {it["english"] for it in train if "english" in it}
    g = dict(rep_sets("tamil_like", "abugida_pua", 0, train_en, 100))
    g.update({f"R1 {k}": v[:50] for k, v in r1_sets(data["splits"]["test_iid"], "tamil_like", "abugida_pua", 0,
                                                    train_en, set(), 50).items()})
    ref = load(E1["E1a (plain)"])["eval_gen"]                  # check: same items as in the E1 runs
    for k, items in g.items():
        saved = [s["english"] for s in ref[k]["samples"]]
        assert [it["english"] for it in items[:len(saved)]] == saved, f"items differ from the E1 run: {k}"
    return g


def part(run: str) -> None:
    from train.load import load_trained
    from train.run import gen_eval
    g = groups()
    model, tok, _ = load_trained(RUNS / run / "model")
    dev = next(model.parameters()).device
    amp = lambda: torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda")   # noqa: E731
    res = {}
    for k, items in g.items():
        r = gen_eval(model, tok, items, 16, dev, amp, False)
        res[k] = {"n": r["n"], "exact": r["exact"]}
        print(f"{run} {k}: {r['exact']:.2f}", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{run}.json").write_text(json.dumps(res), encoding="utf-8")   # saved as it goes


def report() -> None:
    a = load(E1["E1a (plain)"])
    held = set(a["rep"]["held_out"])
    cols = {}
    for label, run in OTHERS.items():
        r = load(run)
        p = OUT / f"{run}.json"
        loc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        cols[label] = {**{f"original {k}": v["teacher_forced"]["exact_match"] for k, v in r["eval"].items()},
                       **{k: v["exact"] for k, v in loc.items()}}
    for label, run in E1.items():
        cols[label] = {k: v["exact"] for k, v in load(run)["eval_gen"].items()}
    L = ["# Option 1 vs R1 vs E1a vs E1b on the same test items", "",
         "Exact match of the glyph output (free generation; for the original splits of Option 1 and R1, teacher-forced "
         "exact match, which equals greedy generation for these models). All models: Qwen3-0.6B-Base + LoRA, seed 0.", "",
         f"E1's randomly held-out variants (never in E1 training): {', '.join(sorted(held))}. "
         "Rows starting 'R1' are R1's categories; their [train]/[heldout] labels refer to R1's training, not E1's.", "",
         "| group | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    names = list(a["eval_gen"])
    for k in names:
        n = a["eval_gen"][k]["n"]
        L.append(f"| {k} | {n} | " + " | ".join(f"{cols[c][k]:.2f}" if k in cols[c] else "-" for c in cols) + " |")
    (ROOT / "results" / "e1_comparison.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "part":
        part(sys.argv[2])
    elif sys.argv[1] == "all":
        for run in OTHERS.values():
            subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "part", run])
        report()
    else:
        report()
