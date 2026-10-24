"""Score models on the sealed set (written by the user in sealed/app; never used to design training data).

    python analysis/score_sealed.py RUN [RUN ...]      -> results/sealed/<run>.json (per-item results)
    python analysis/score_sealed.py --table            -> results/sealed_scores.md (totals only)

Gold answers are the ones stored in sealed/sealed_draft.json (the reader's meaning, checked by the user; for a name
doer, the he/she the user chose). Only totals are printed and written to the table: which sentences fail is not
looked at, so the sealed set cannot shape the training data. In scope: glyph answer exact (seed-0 language), meaning
(analysis) exact for models that write one, and glyph word accuracy (difflib alignment). Out of scope: share answered
with the refusal output. While the set is still growing, scores are PROVISIONAL (the table says how many entries).
"""

from __future__ import annotations

import difflib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DRAFT = ROOT / "sealed" / "sealed_frozen.json"         # frozen 2026-10-22 (sealed/FROZEN.md); was sealed_draft.json
OUT = ROOT / "results" / "sealed"


def word_acc(pred: str, gold: str) -> float:
    g, p = gold.split(), pred.split()
    sm = difflib.SequenceMatcher(a=g, b=p, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / max(1, len(g))


def score(run: str, rewriter: str | None = None) -> None:
    from train.load import load_trained, translate
    from train.augment import REFUSAL
    mdir = ROOT / "results" / "kaggle" / run / "model"
    cfg = json.loads((mdir / "run_args.json").read_text(encoding="utf-8"))
    if int(cfg.get("seed", 0)) != 0:
        print(f"{run}: seed {cfg.get('seed')} language differs from the sealed answers (seed 0); skipped")
        return
    cot = str(cfg.get("cot")) == "True"
    d = json.loads(DRAFT.read_text(encoding="utf-8"))
    entries = [e for e in d["entries"] if e.get("targets", {}).get("seed0")]
    inputs = [e["english"] for e in entries]
    actions = None
    if rewriter:                                    # two-stage: frozen rewriter first (then unloaded), then translator
        import gc
        import torch
        from two_stage.pipeline import Rewriter
        rw = Rewriter(rewriter)
        rws = rw.rewrite(inputs)
        rw.close()
        del rw
        gc.collect()
        torch.cuda.empty_cache()
        inputs = [r.text for r in rws]
        actions = dict(__import__("collections").Counter(r.action for r in rws))
    model, tok, _ = load_trained(mdir)
    t0 = time.time()
    preds = []
    for i in range(0, len(entries), 8):
        preds += translate(model, tok, inputs[i:i + 8], max_new_tokens=700)   # analysis + glyphs can be long
    items = []
    for e, p in zip(entries, preds):
        glyph = p.partition("=>")[2].strip() if (cot and "=>" in p) else p.strip()
        ana = p.partition("=>")[0].strip() if (cot and "=>" in p) else ""
        gold = e["targets"]["seed0"]
        if e["scope"] == "out":
            items.append({"id": e["id"], "scope": "out", "refused": REFUSAL in p})
        else:
            items.append({"id": e["id"], "scope": "in", "exact": glyph == gold,
                          "meaning": (ana == e.get("analysis")) if cot else None,
                          "word_acc": word_acc(glyph, gold)})
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"two-stage({rewriter.split('/')[-1]})+{run}" if rewriter else run
    (OUT / f"{name}.json").write_text(json.dumps({"run": name, "cot": cot, "n_entries": len(entries),
                                                  "draft_saved": d.get("saved"), "actions": actions, "items": items}),
                                      encoding="utf-8")
    print(f"{run}: scored {len(entries)} entries in {time.time() - t0:.0f} s")


def table() -> None:
    d = json.loads(DRAFT.read_text(encoding="utf-8"))
    n_in = sum(e["scope"] == "in" for e in d["entries"])
    n_out = len(d["entries"]) - n_in
    out = ["# Sealed set scores (PROVISIONAL while the set is still growing)", "",
           f"Sealed set as of {d.get('saved')}: {n_in} in scope, {n_out} out of scope. Totals only "
           "(`analysis/score_sealed.py`).", "",
           "| model | in scope: glyphs exact | meaning exact | glyph word accuracy | out of scope: refused |",
           "|---|---|---|---|---|"]
    for f in sorted(OUT.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        ins = [x for x in r["items"] if x["scope"] == "in"]
        outs = [x for x in r["items"] if x["scope"] == "out"]
        ex = sum(x["exact"] for x in ins) / max(1, len(ins))
        me = (f"{sum(bool(x['meaning']) for x in ins) / max(1, len(ins)):.3f}" if r["cot"] else "-")
        wa = sum(x["word_acc"] for x in ins) / max(1, len(ins))
        rf = sum(x["refused"] for x in outs) / max(1, len(outs))
        out.append(f"| {r['run']} | {ex:.3f} | {me} | {wa:.3f} | {rf:.3f} |")
    (ROOT / "results" / "sealed_scores.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[5:]))


if __name__ == "__main__":
    if "--table" in sys.argv:
        table()
    elif "--two-stage" in sys.argv:                # score_sealed.py --two-stage REWRITER RUN
        i = sys.argv.index("--two-stage")
        score(sys.argv[i + 2], rewriter=sys.argv[i + 1])
    else:
        for r in sys.argv[1:]:
            score(r)
