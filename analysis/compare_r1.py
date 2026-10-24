"""Score saved models on R1's augmented evaluation categories (exactly the items R1 was scored on in its run).

    python analysis/compare_r1.py part <run-folder-name>     # one model per process
    python analysis/compare_r1.py report                     # writes results/r1_vs_option1.md

The items are rebuilt deterministically (same seed, same augmentation code) and scored teacher-forced, as in
train/run.py. Re-scoring R1 itself checks that the downloaded weights reproduce the numbers from Kaggle.
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
OUT = ROOT.parent / "scratch_local" / "r1_compare"
MODELS = {"Option 1 (before)": "s1-hard-latininit-50k", "R1 (retrained)": "r1-hard-aug-50k"}


def items():
    from conlang.dataset import PRESETS, build
    from train.augment import build_training, eval_sets
    d = build(PRESETS["tamil_like"], "abugida_pua", n_train=50000, n_test=500, seed=0)
    tr, info = build_training(d["splits"]["train"], "tamil_like", "abugida_pua", 0)
    return eval_sets(d["splits"]["test_iid"], "tamil_like", "abugida_pua", 0,
                     {it["english"] for it in tr if "english" in it}, set(info["train_names"]), 100)


def part(run: str) -> None:
    from train.load import load_trained
    from train.run import encode, teacher_forced
    sets = items()
    model, tok, _ = load_trained(RUNS / run / "model")
    dev = next(model.parameters()).device
    amp = lambda: torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda")   # noqa: E731
    res = {}
    for name, its in sets.items():
        tf = teacher_forced(model, tok, [encode(tok, it) for it in its], 16, dev, amp)
        res[name] = {"n": tf["n"], "exact_match": tf["exact_match"]}
        print(f"{run} {name}: {tf['exact_match']:.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{run}.json").write_text(json.dumps(res), encoding="utf-8")


def report() -> None:
    def splits(run):
        r = json.loads((RUNS / run / "results" / f"{run}.json").read_text(encoding="utf-8"))
        return {k: v["teacher_forced"]["exact_match"] for k, v in r["eval"].items()}, r
    s1, _ = splits(MODELS["Option 1 (before)"])
    sr, rr = splits(MODELS["R1 (retrained)"])
    local = {l: json.loads((OUT / f"{run}.json").read_text(encoding="utf-8")) for l, run in MODELS.items()}
    kag = rr["eval_aug"]
    L = ["# R1 (retrained) vs Option 1", "",
         "Both: Qwen3-0.6B-Base + LoRA r=16, 3 epochs, glyphs as new tokens starting at their Latin sounds, seed 0. "
         "R1 adds varied English, new structures, refusals and plain-English replay (train/augment.py). "
         "Teacher-forced exact match. [train] = this kind of variant was in R1's training data; [heldout] = it never "
         "was (tests generalisation). Option 1 never saw any of them.", "",
         "## The four original test splits", "", "| split | Option 1 | R1 |", "|---|---|---|"]
    for k in s1:
        L.append(f"| {k} | {s1[k]:.3f} | {sr[k]:.3f} |")
    L += ["", f"Training time: Option 1 {json.loads((RUNS / MODELS['Option 1 (before)'] / 'results' / (MODELS['Option 1 (before)'] + '.json')).read_text(encoding='utf-8'))['training']['seconds'] / 60:.1f} min, "
          f"R1 {rr['training']['seconds'] / 60:.1f} min.", "",
          "## New categories (100 items each)", "",
          "| category | Option 1 | R1 | R1 on Kaggle (check) |", "|---|---|---|---|"]
    for k in kag:
        o, r = local["Option 1 (before)"].get(k), local["R1 (retrained)"].get(k)
        L.append(f"| {k} | {o['exact_match']:.2f} | {r['exact_match']:.2f} | {kag[k]['exact_match']:.2f} |")
    L += ["", "## Training data used by R1", "", "| kind | examples |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in rr["data"]["aug"].items()]
    (ROOT / "results" / "r1_vs_option1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "part":
        part(sys.argv[2])
    elif sys.argv[1] == "all":
        for run in MODELS.values():
            if not (OUT / f"{run}.json").exists():
                subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "part", run])
        report()
    else:
        report()
