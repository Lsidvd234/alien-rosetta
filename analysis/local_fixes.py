"""Inference-time fixes, no retraining (laptop):

L2  split the input into sentences/clauses, translate each part, join the outputs
L3  lower-case the input before translating

    python analysis/local_fixes.py all        # Option 1, R1, E1a, E1b (one process each), then results/local_fixes.md
    python analysis/local_fixes.py report

Items are E1's evaluation groups (rebuilt exactly, see compare_e1.py); 50 per group. Split rule: break at '.', ';',
'!' and ', and ' (not at a bare ' and ', which also joins adjectives). Lower-casing also lower-cases names.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RUNS = ROOT / "results" / "kaggle"
OUT = ROOT.parent / "scratch_local" / "local_fixes"
MODELS = {"Option 1": "s1-hard-latininit-50k", "R1": "r1-hard-aug-50k", "E1a (plain)": "e1a-hard-rep-50k",
          "E1b (analysis)": "e1b-hard-rep-cot-50k"}
COT = {"e1b-hard-rep-cot-50k"}
SPLIT_GROUPS = ["R1 structure two sentences [train]", "R1 structure three sentences [heldout]",
                "extrapolation: 5 clauses", "in-distribution mix"]
CASE_GROUPS = ["R1 rewrite casing [train]", "R1 rewrite casing [heldout]", "held-out case = sentence",
               "in-distribution mix"]


def split_parts(text: str) -> list[str]:
    parts = re.split(r"\s*(?:[.;!]|,\s+and)\s+", text.strip())
    return [p.strip(" .;!") for p in parts if p.strip(" .;!")]


@torch.no_grad()
def translate(model, tok, texts: list[str], cot: bool, bsz: int = 16) -> list[str]:
    from train.run import prompt
    dev = next(model.parameters()).device
    tok.padding_side = "left"
    out = []
    for s in range(0, len(texts), bsz):
        enc = tok([prompt(t) for t in texts[s:s + bsz]], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            g = model.generate(**enc, max_new_tokens=260 if cot else 140, do_sample=False,
                               pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row in g[:, enc["input_ids"].shape[1]:]:
            p = tok.decode(row, skip_special_tokens=True).split("\n")[0].strip()
            out.append(p.split("=>", 1)[1].strip() if (cot and "=>" in p) else p)
    tok.padding_side = "right"
    return out


def part(run: str) -> None:
    from compare_e1 import groups
    from train.load import load_trained
    g = groups()
    model, tok, _ = load_trained(RUNS / run / "model")
    cot = run in COT
    res = {}
    for name in SPLIT_GROUPS:
        items = g[name][:50]
        plain = translate(model, tok, [it["english"] for it in items], cot)
        flat, idx = [], []
        for i, it in enumerate(items):
            ps = split_parts(it["english"])
            idx.append((len(flat), len(flat) + len(ps)))
            flat += ps
        outs = translate(model, tok, flat, cot)
        joined = [" ".join(outs[a:b]) for a, b in idx]
        res[f"split | {name}"] = {"n": len(items),
                                  "as is": sum(p == it["target"] for p, it in zip(plain, items)) / len(items),
                                  "fixed": sum(p == it["target"] for p, it in zip(joined, items)) / len(items)}
        print(run, name, res[f"split | {name}"], flush=True)
    for name in CASE_GROUPS:
        items = g[name][:50]
        a = translate(model, tok, [it["english"] for it in items], cot)
        b = translate(model, tok, [it["english"].lower() for it in items], cot)
        res[f"lowercase | {name}"] = {"n": len(items),
                                      "as is": sum(p == it["target"] for p, it in zip(a, items)) / len(items),
                                      "fixed": sum(p == it["target"] for p, it in zip(b, items)) / len(items)}
        print(run, name, res[f"lowercase | {name}"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{run}.json").write_text(json.dumps(res), encoding="utf-8")


def report() -> None:
    cols = {l: json.loads((OUT / f"{r}.json").read_text(encoding="utf-8")) for l, r in MODELS.items()
            if (OUT / f"{r}.json").exists()}
    keys = list(next(iter(cols.values())))
    L = ["# Inference-time fixes without retraining (L2 sentence splitting, L3 lower-casing)", "",
         "50 items per group from E1's evaluation groups; exact match of the glyph output. Cells: as given -> with the fix. "
         "Split rule: break at '.', ';', '!' and ', and '; translate each part; join. Lower-casing also lower-cases "
         "names (none in these groups' casing rows except the in-distribution mix).", "",
         "| fix | group | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for k in keys:
        fix, grp = k.split(" | ", 1)
        L.append(f"| {fix} | {grp} | " + " | ".join(f"{c[k]['as is']:.2f} -> {c[k]['fixed']:.2f}" for c in cols.values()) + " |")
    (ROOT / "results" / "local_fixes.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "part":
        part(sys.argv[2])
    elif sys.argv[1] == "all":
        for run in MODELS.values():
            if not (OUT / f"{run}.json").exists():
                r = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "part", run])
                print("part", run, "exit", r.returncode, flush=True)
        report()
    else:
        report()
