"""Was each run still learning when it stopped? Evaluate every saved epoch checkpoint (local, generated tests only).

    python analysis/epoch_curve.py RUN [RUN ...] [--n 100]   -> results/epochs/<run>.json
    python analysis/epoch_curve.py --table                   -> results/epoch_curves.md + results/fig_epoch_curves.png

For epochs 1, 2, 3 and the final model (epoch 4): exact match of the glyphs and of the meaning, and word accuracy, on
the same n generated test sentences (the frequency-weighted test's sentences, none in the run's training data), plus
the loss and the zero-gradient share (train/collapse.py) on 150 of the run's own training examples. If test accuracy
still rises from epoch 3 to 4, the run was still learning when it stopped (the learning rate had decayed to 0, so the
end of the loss curve is flat by construction and cannot answer this).
"""

from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

OUT = ROOT / "results" / "epochs"
def ckpts(cfg: dict) -> list[tuple[str, str]]:
    """(epoch, folder): model_epochK for every saved epoch, then `model` = the LAST epoch. Fixed 2026-10-25: this was a
    fixed list ending in ("4", "model"), so the 6-epoch run's final model was scored as "epoch 4" and its
    model_epoch4/5 were never scored."""
    e = int(float(cfg.get("epochs", 4)))
    return [(str(k), f"model_epoch{k}") for k in range(1, e)] + [(str(e), "model")]


def word_acc(gold: str, pred: str) -> float:
    g, p = gold.split(), pred.split()
    sm = difflib.SequenceMatcher(a=g, b=p, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / max(1, len(g))


def run_one(run: str, n: int) -> None:
    import train.represent_v2 as R2
    from freq_weighted_eval import test_items
    from subgraph_check import training_pieces
    from train.collapse import collapse_stats
    from train.load import load_trained, translate
    from train.run import encode
    base = ROOT / "results" / "kaggle" / run
    cfg = json.loads((base / "model" / "run_args.json").read_text(encoding="utf-8"))
    cot = str(cfg.get("cot")) == "True"
    _, train_en, _, _ = training_pieces(cfg, ROOT / "results" / "subgraph" / f"{run}.pieces.json")
    R2.LOAN.update(rate=0.0, pool="train")
    gen = R2.GenV2(cfg.get("preset", "tamil_like"), cfg.get("script", "abugida_pua"), int(cfg.get("seed", 0)))
    tests = [it for it, _ in test_items(gen, n + 50) if it["english"] not in train_en][:n]
    items, _ = R2.build_training(cfg.get("preset", "tamil_like"), cfg.get("script", "abugida_pua"),
                                 int(cfg.get("seed", 0)), n=150, n_refuse=0, n_replay=0, cot=cot,
                                 held_frac=float(cfg.get("rep_held_frac", 0.2)), min_count=0,
                                 loan_rate=float(cfg.get("rep_loan_rate", 0) or 0),
                                 everyday=str(cfg.get("rep_everyday")) == "True")
    R2.LOAN.update(rate=0.0, pool="train")
    # each finished epoch is saved at once, so a stopped run resumes at the next epoch (2026-10-26, low-memory stop);
    # named .partial, not .json, so table() never reads it
    OUT.mkdir(parents=True, exist_ok=True)
    part = OUT / f"{run}_n{n}.partial"
    rows = json.loads(part.read_text(encoding="utf-8")) if part.exists() else []
    for ep, d in ckpts(cfg):
        if not (base / d).exists() or int(ep) in {r["epoch"] for r in rows}:
            continue
        model, tok, _ = load_trained(base / d)
        preds = []
        for i in range(0, len(tests), 16):
            preds += translate(model, tok, [t["english"] for t in tests[i:i + 16]], max_new_tokens=700)
        gl = [(p.partition("=>")[2].strip() if cot else p) for p in preds]
        row = {"epoch": int(ep), "glyph_exact": sum(g == t["target"] for g, t in zip(gl, tests)) / len(tests),
               "word_acc": sum(word_acc(t["target"], g) for g, t in zip(gl, tests)) / len(tests)}
        if cot:
            row["meaning_exact"] = sum(p.partition("=>")[0].strip() == t["analysis"] for p, t in zip(preds, tests)) / len(tests)
        tok.padding_side = "right"
        cs = collapse_stats(model, tok, [encode(tok, it) for it in items], next(model.parameters()).device,
                            lambda: torch.autocast("cuda", dtype=torch.float16), bsz=4)
        row.update({"train_loss": cs["loss_mean"], "zero_grad_fp32": cs["zero_grad_fp32"], "gap_median": cs["gap_median"]})
        rows.append(row)
        part.write_text(json.dumps(rows), encoding="utf-8")
        print(run, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        del model
        torch.cuda.empty_cache()
    (OUT / f"{run}.json").write_text(json.dumps({"run": run, "n": len(tests), "cot": cot, "epochs": rows}, indent=1),
                                     encoding="utf-8")
    part.unlink(missing_ok=True)


def table() -> None:
    res = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(OUT.glob("*.json"))]
    L = ["# Was each run still learning? (`analysis/epoch_curve.py`)", "",
         "Each epoch checkpoint on the same generated test sentences (none trained on) and on 150 of the run's own "
         "training examples. The learning rate decays linearly to 0 over each run's epochs (4; 6 for e2b-v2fix-6ep).", "",
         "| run | epoch | glyphs exact | meaning exact | word acc | train loss | zero-grad (fp32) |", "|---|---|---|---|---|---|---|"]
    for r in res:
        for e in r["epochs"]:
            L.append(f"| {r['run']} | {e['epoch']} | {e['glyph_exact']:.2f} | {e.get('meaning_exact', float('nan')):.2f} | "
                     f"{e['word_acc']:.3f} | {e['train_loss']:.4f} | {e['zero_grad_fp32']:.1%} |")
    L += ["", "| run | glyphs exact, epoch 3 → 4 | word acc, epoch 3 → 4 | train loss, epoch 3 → 4 |", "|---|---|---|---|"]
    for r in res:
        e = {x["epoch"]: x for x in r["epochs"]}
        if 3 in e and 4 in e:
            L.append(f"| {r['run']} | {e[3]['glyph_exact']:.2f} → {e[4]['glyph_exact']:.2f} | "
                     f"{e[3]['word_acc']:.3f} → {e[4]['word_acc']:.3f} | {e[3]['train_loss']:.4f} → {e[4]['train_loss']:.4f} |")
    (ROOT / "results" / "epoch_curves.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for r in res:
        ep = [e["epoch"] for e in r["epochs"]]
        ax[0].plot(ep, [e["glyph_exact"] for e in r["epochs"]], marker="o", label=r["run"])
        ax[1].plot(ep, [e["word_acc"] for e in r["epochs"]], marker="o", label=r["run"])
        ax[2].plot(ep, [e["train_loss"] for e in r["epochs"]], marker="o", label=r["run"])
    for a, t in zip(ax, ["test: glyphs exact", "test: word accuracy", "training examples: loss (log)"]):
        a.set(xlabel="epoch", title=t, xticks=range(1, max(e["epoch"] for r in res for e in r["epochs"]) + 1))
        a.grid(alpha=0.3)
    ax[2].set_yscale("log")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    try:
        fig.savefig(ROOT / "results" / "fig_epoch_curves.png", dpi=140)
    except OSError as e:
        print("figure not saved:", e)
    print("\n".join(L))


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--table" in args:
        table()
    else:
        n = int(args[args.index("--n") + 1]) if "--n" in args else 100
        for r in [x for i, x in enumerate(args) if x != "--n" and (i == 0 or args[i - 1] != "--n")]:
            run_one(r, n)
