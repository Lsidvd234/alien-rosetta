"""Reader or writer: which step is the bottleneck? Two-model pipeline vs the one-model translator. (local)

    python analysis/bottleneck.py [READER WRITER SINGLE]   -> results/bottleneck.md + .json
    defaults: e2b-v2fix-passive0-meaning (English -> meaning), e2b-v2fix-passive0-writer (meaning -> glyphs),
              e2b-v2fix-passive0 (one model, meaning then glyphs). All three share passive0's data and settings.

For every test sentence:
  reading      the reader's meaning line is exactly the gold meaning
  writing      the writer, given the GOLD meaning, writes exactly the gold glyphs (writing alone, perfect input)
  pipeline     the writer, given the READER's meaning, writes exactly the gold glyphs
  one model    the one-model translator's glyphs are exactly right (its own meaning line also scored)
Pipeline failures are split into: reader wrong (the meaning was wrong) / reader right but writer wrong.
Tests: Tatoeba 600 (gold from the reader's labels), generated "in-distribution mix" and "held-out voice = passive"
(300 each, not in training), and the sealed set (laptop only, totals only: no sentence is stored or printed).
Predictions in results/findings_2026-10-24.md (BN1-BN3), written before the runs returned.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
DEFAULT = ["e2b-v2fix-passive0-meaning", "e2b-v2fix-passive0-writer", "e2b-v2fix-passive0"]


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new: int, bsz: int = 16) -> list[str]:
    dev = next(model.parameters()).device
    out = []
    tok.padding_side = "left"
    for i in range(0, len(prompts), bsz):
        enc = tok(prompts[i:i + bsz], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                               eos_token_id=tok.eos_token_id)
        out += [tok.decode(r, skip_special_tokens=True).split("\n")[0].strip() for r in g[:, enc["input_ids"].shape[1]:]]
    tok.padding_side = "right"
    return out


def load(run: str):
    from train.load import load_trained
    model, tok, _ = load_trained(ROOT / "results" / "kaggle" / run / "model")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model.eval(), tok


def tests(single: str) -> dict[str, list[dict]]:
    """{name: [{english, analysis, target}]}: Tatoeba, two generated sets (not in passive0's training), sealed."""
    import train.represent_v2 as R2
    from score_sealed import DRAFT
    from subgraph_check import training_pieces
    cfg = json.loads((ROOT / "results" / "kaggle" / single / "model" / "run_args.json").read_text(encoding="utf-8"))
    _, train_en, _, _ = training_pieces(cfg, ROOT / "results" / "subgraph" / f"{single}.pieces.json")
    R2.set_held_out(0)
    R2.HELD_OUT.add(("voice", "passive"))                       # passive0's held-out structure
    sets = R2.eval_sets("tamil_like", "abugida_pua", 0, train_en, per=300)
    out = {"Tatoeba 600": json.loads((ROOT / "results" / "tatoeba_test.json").read_text(encoding="utf-8")),
           "generated: in-distribution mix": sets["in-distribution mix"],
           "generated: held-out voice = passive": sets["held-out voice = passive"]}
    d = json.loads(DRAFT.read_text(encoding="utf-8"))
    out["sealed (totals only)"] = [{"english": e["english"], "analysis": e.get("analysis") or "",
                                    "target": e["targets"]["seed0"]} for e in d["entries"]
                                   if e["scope"] == "in" and e.get("targets", {}).get("seed0")]
    return out


def main() -> None:
    from train.run import prompt, writer_prompt
    reader, writer, single = (sys.argv[1:4] if len(sys.argv) >= 4 else DEFAULT)
    T = tests(single)
    pred = {k: {} for k in T}
    for role, run in (("reader", reader), ("single", single)):
        model, tok = load(run)
        for k, items in T.items():
            pred[k][role] = generate(model, tok, [prompt(it["english"]) for it in items], 700 if role == "single" else 300)
        del model
        torch.cuda.empty_cache()
    model, tok = load(writer)
    for k, items in T.items():
        reader_meaning = [p.partition("=>")[0].strip() for p in pred[k]["reader"]]
        pred[k]["writer_gold"] = generate(model, tok, [writer_prompt(it["analysis"]) for it in items], 400)
        pred[k]["writer_pipe"] = generate(model, tok, [writer_prompt(m) for m in reader_meaning], 400)
    del model
    torch.cuda.empty_cache()

    res = {}
    for k, items in T.items():
        P, n = pred[k], len(items)
        read = [p.partition("=>")[0].strip() == it["analysis"] for p, it in zip(P["reader"], items)]
        write = [g == it["target"] for g, it in zip(P["writer_gold"], items)]
        pipe = [g == it["target"] for g, it in zip(P["writer_pipe"], items)]
        one = [p.partition("=>")[2].strip() == it["target"] for p, it in zip(P["single"], items)]
        one_m = [p.partition("=>")[0].strip() == it["analysis"] for p, it in zip(P["single"], items)]
        fails = [i for i in range(n) if not pipe[i]]
        res[k] = {"n": n, "reading": sum(read) / n, "writing_gold_meaning": sum(write) / n,
                  "pipeline": sum(pipe) / n, "one_model_glyphs": sum(one) / n, "one_model_meaning": sum(one_m) / n,
                  "pipeline_failures": len(fails),
                  "failures_reader_wrong": sum(not read[i] for i in fails),
                  "failures_reader_right_writer_wrong": sum(read[i] for i in fails)}
    L = ["# Reader or writer? Two-model pipeline vs one model (`analysis/bottleneck.py`)", "",
         f"reader = {reader} (English -> meaning), writer = {writer} (meaning -> glyphs), one model = {single}. "
         "Exact-match shares; the sealed row is totals only.", "",
         "| test | n | reading | writing (gold meaning) | pipeline | one model: glyphs | one model: meaning | "
         "pipeline failures: reader wrong / writer wrong |", "|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        L.append(f"| {k} | {r['n']} | {r['reading']:.3f} | {r['writing_gold_meaning']:.3f} | {r['pipeline']:.3f} | "
                 f"{r['one_model_glyphs']:.3f} | {r['one_model_meaning']:.3f} | {r['failures_reader_wrong']} / "
                 f"{r['failures_reader_right_writer_wrong']} |")
    (ROOT / "results" / "bottleneck.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "results" / "bottleneck.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
