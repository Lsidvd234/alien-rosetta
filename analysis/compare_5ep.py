"""E1 with 5 epochs (RunPod V100): E1a, E1b and an E1b replicate, next to the 3-epoch E1 runs.

    python analysis/compare_5ep.py        # -> results/e1_5ep.md

Scores are eval_gen glyph exact match from each run's JSON. The 5-epoch runs used strict held-out groups (only items
that really contain the held-out variant); the 3-epoch runs used the pooled groups, so their held-out structure rows
are not shown here (corrected values: results/e1_heldout_corrected.md). Seed 1 is a new random language instance, so its
test items differ from seed 0's.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "kaggle"
COLS = [("E1a 3ep", "e1a-hard-rep-50k"), ("E1b 3ep", "e1b-hard-rep-cot-50k"), ("E1a 5ep", "e1a-hard-rep-5ep"),
        ("E1b 5ep", "e1b-hard-rep-cot-5ep"), ("E1b 5ep seed 1", "e1b-hard-rep-cot-5ep-seed1")]
STRICT_ONLY = ("held-out order", "held-out voice")       # pooled in the 3-epoch runs: not comparable


def load(run: str) -> tuple[dict, dict]:
    d = json.loads((RUNS / run / "results" / f"{run}.json").read_text(encoding="utf-8"))
    return d["eval_gen"], d


def main() -> None:
    data = {lab: load(run) for lab, run in COLS}
    rows = list(data["E1b 5ep"][0])
    L = ["# E1 with 5 epochs: E1a, E1b and an E1b replicate", "",
         "Free generation, exact match of the glyph output. 5-epoch runs: RunPod V100 (fp16, torch 2.6, transformers "
         "5.16.1), otherwise identical settings to the 3-epoch Kaggle runs. Held-out voice/order groups of the 5-epoch "
         "runs are strict (only items that contain the variant); the 3-epoch cells for those rows are left blank "
         "(see e1_heldout_corrected.md). Seed 1 = a new random language instance (different test items). "
         "n = 500 per original split, 100 per E1 group, 50 per R1 group.", "",
         "| group | n | " + " | ".join(l for l, _ in COLS) + " |", "|---|---|" + "---|" * len(COLS)]
    for name in rows:
        if name.startswith("R1 rewrite"):
            continue
        cells = []
        for lab, _ in COLS:
            g = data[lab][0].get(name)
            blank = "3ep" in lab and name.startswith(STRICT_ONLY)
            cells.append("" if (g is None or blank) else f"{g['exact']:.2f}")
        L.append(f"| {name} | {data['E1b 5ep'][0][name]['n']} | " + " | ".join(cells) + " |")
    L += ["", "| run | training minutes | target tokens/s |", "|---|---|---|"]
    for lab, run in COLS:
        tr = data[lab][1]["training"]
        L.append(f"| {lab} | {tr['seconds'] / 60:.0f} | {tr['target_tokens_per_second']:.0f} |")
    (ROOT / "results" / "e1_5ep.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
