"""Are our trained translation models in (or near) softmax collapse? (local; train/collapse.py has the measure)

    python analysis/softmax_collapse.py RUN [RUN ...] [--n 300]   -> results/collapse/<run>.json
    python analysis/softmax_collapse.py --table                   -> results/softmax_collapse.md

Uses each run's own TRAINING examples (the first n, rebuilt exactly from its settings, in its training format: meaning
first for the meaning-first runs), because collapse is about tokens the model has already fitted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "collapse"


def run_one(run: str, n: int) -> None:
    import train.represent_v2 as R2
    from train.collapse import collapse_stats
    from train.load import load_trained
    from train.run import encode
    mdir = ROOT / "results" / "kaggle" / run / "model"
    cfg = json.loads((mdir / "run_args.json").read_text(encoding="utf-8"))
    cot = str(cfg.get("cot")) == "True"
    para = cfg.get("rep_paraphrases")
    if para and para != "None":
        para = str(next((ROOT / "results" / "kaggle" / run).glob("paraphrases_seed*.json"), "")) or None
    else:
        para = None
    items, _ = R2.build_training(cfg.get("preset", "tamil_like"), cfg.get("script", "abugida_pua"),
                                 int(cfg.get("seed", 0)), n=n, n_refuse=0, n_replay=0, cot=cot,
                                 held_frac=float(cfg.get("rep_held_frac", 0.2)), min_count=0,
                                 paraphrase_file=para, loan_rate=float(cfg.get("rep_loan_rate", 0) or 0))
    model, tok, _ = load_trained(mdir)
    device = next(model.parameters()).device
    amp = lambda: torch.autocast("cuda", dtype=torch.float16)  # noqa: E731  (as trained on T4 / P100)
    tok.padding_side = "right"
    ex = [encode(tok, it) for it in items]
    st = collapse_stats(model, tok, ex, device, amp, bsz=4)
    st.update({"run": run, "examples": len(ex), "cot": cot})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{run}.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    print(run, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()}), flush=True)
    del model
    torch.cuda.empty_cache()


def table() -> None:
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(OUT.glob("*.json"))]
    L = ["# Softmax collapse check (`analysis/softmax_collapse.py`, measure in `train/collapse.py`)", "",
         "Each model's own training examples (first 300, answer given), every answer token. **Zero-gradient share** = "
         "tokens whose softmax gives exactly 1.0 for the right token (loss and gradient exactly 0). float32 is what "
         "the training loss used; bf16 / fp16 show how close lower precision would be. Gap = right logit minus the best "
         "wrong logit (float32 collapse needs ~16+).", "",
         "| run | answer tokens | mean p(right) | mean loss | gap median | gap 90th pct | zero-grad fp32 | near fp32 (gap>15) "
         "| zero-grad fp16 | zero-grad bf16 | logit norm | hidden norm |", "|---" * 12 + "|"]
    for r in rows:
        L.append(f"| {r['run']} | {r['tokens']} | {r['p_mean']:.4f} | {r['loss_mean']:.4f} | {r['gap_median']:.1f} | "
                 f"{r['gap_p90']:.1f} | **{r['zero_grad_fp32']:.1%}** | {r['near_fp32']:.1%} | {r['zero_grad_fp16']:.1%} | "
                 f"{r['zero_grad_bf16']:.1%} | {r['logit_norm']:.0f} | {r['hidden_norm']:.0f} |")
    (ROOT / "results" / "softmax_collapse.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--table" in args:
        table()
    else:
        n = int(args[args.index("--n") + 1]) if "--n" in args else 300
        for r in [x for i, x in enumerate(args) if x != "--n" and (i == 0 or args[i - 1] != "--n")]:
            run_one(r, n)
