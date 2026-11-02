"""Second real-English test set: 600 held-out Tatoeba sentences (results/tatoeba_test.json), never trained on.

    python analysis/score_tatoeba.py RUN [RUN ...]     -> results/tatoeba/<run>.json
                                                 (RUN@epoch2 = that run's saved epoch-2 checkpoint)
    python analysis/score_tatoeba.py --from-run RUN    -> the same, from the predictions the run made on Kaggle right
                                                 after training (train/run.py --extra-translate; e.g. Qwen3-1.7B,
                                                 too big for the 4 GB laptop GPU)
    python analysis/score_tatoeba.py --table           -> results/tatoeba_scores.md

Gold = the sealed-set reader's labels (hand-checked precision of the label source ~88%, so even a perfect model would
score about 0.88 on meaning; compare models with each other, not with 1.0). Short, everyday human sentences: a much
larger real-English benchmark than the sealed set. Seed-0 language only.
"""

from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "tatoeba"


def word_acc(gold: str, pred: str) -> float:
    g = gold.split()
    sm = difflib.SequenceMatcher(a=g, b=pred.split(), autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / max(1, len(g))


def score(run: str) -> None:
    import torch
    from train.load import load_trained, translate
    test = json.loads((ROOT / "results" / "tatoeba_test.json").read_text(encoding="utf-8"))
    name, _, ep = run.partition("@epoch")                 # RUN@epoch2: that run's saved epoch-2 checkpoint
    mdir = ROOT / "results" / "kaggle" / name / (f"model_epoch{ep}" if ep else "model")
    cfg = json.loads((mdir / "run_args.json").read_text(encoding="utf-8"))
    cot = str(cfg.get("cot")) == "True"
    if str(cfg.get("rep_sound_rule")) == "True":          # sound-rule language: glyph golds re-rendered from the gold
        from train.represent_v2 import render_targets                    # meanings (meaning golds unchanged)
        gold = render_targets([t["analysis"] for t in test], [t["target"] for t in test])
        test = [{**t, "target": g if g is not None else "\0"} for t, g in zip(test, gold)]   # \0: never matches
    if "--from-run" in sys.argv:                          # translated on Kaggle right after training
        assert not ep, "--from-run holds the final model's translations, not an epoch checkpoint's"
        r = json.loads((ROOT / "results" / "kaggle" / name / "results" / f"{name}.json").read_text(encoding="utf-8"))
        x = r["extra_translate"]
        assert x["english"] == [t["english"] for t in test], "the run translated a different Tatoeba file"
        preds, model = x["preds"], None
    else:
        model, tok, _ = load_trained(mdir)
        preds = []
        for i in range(0, len(test), 16):
            preds += translate(model, tok, [t["english"] for t in test[i:i + 16]], max_new_tokens=500)
    gl = [(p.partition("=>")[2].strip() if cot else p) for p in preds]
    res = {"run": run, "n": len(test), "cot": cot,
           "glyph_exact": sum(g == t["target"] for g, t in zip(gl, test)) / len(test),
           "word_acc": sum(word_acc(t["target"], g) for g, t in zip(gl, test)) / len(test),
           "meaning_exact": (sum(p.partition("=>")[0].strip() == t["analysis"] for p, t in zip(preds, test)) / len(test))
           if cot else None,
           "preds": preds}                       # every raw output, for paired tests (analysis/significance.py)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{run}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(run, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != "preds"}, flush=True)
    del model
    torch.cuda.empty_cache()


def table() -> None:
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(OUT.glob("*.json"))]
    L = ["# Real-English test 2: 600 held-out Tatoeba sentences (`analysis/score_tatoeba.py`)", "",
         "Gold labels from the sealed-set reader (label precision ~88% by hand check), so ~0.88 meaning is the ceiling.",
         "", "| run | glyphs exact | meaning exact | glyph word accuracy |", "|---|---|---|---|"]
    for r in rows:
        m = f"{r['meaning_exact']:.3f}" if r["meaning_exact"] is not None else "-"
        L.append(f"| {r['run']} | {r['glyph_exact']:.3f} | {m} | {r['word_acc']:.3f} |")
    (ROOT / "results" / "tatoeba_scores.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if "--table" in sys.argv:
        table()
    else:
        for r in [x for x in sys.argv[1:] if not x.startswith("--")]:
            score(r)
