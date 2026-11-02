"""Claim 6 at answer time: do decoding tricks fix real English? (local GPU; Tatoeba 600; never the sealed set)

    python analysis/decode_tricks.py [RUN] [--n 600]     -> results/decode_tricks_<RUN>.md + .json

Every earlier claim-6 trick changed training (focal loss, rewriter, paraphrases, more epochs). These change only how
the trained model answers, on the baseline (default e2b-v2fix-cot-80k-noheld):
  greedy      as evaluated everywhere else (reference; must reproduce results/tatoeba/<run>.json)
  beam 4      beam search, 4 beams, best-scoring answer
  vote 5      5 samples at temperature 0.7; the most common meaning line wins (its first glyph line kept); ties -> the
              earliest sample (self-consistency)
Scored like analysis/score_tatoeba.py: meaning exact (analysis before '=>') and glyphs exact; paired McNemar test
against greedy on the same 600 sentences (analysis/significance.py helpers).
Predictions were written in results/findings_2026-10-24.md before this ran.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.load import load_trained  # noqa: E402
from train.run import prompt  # noqa: E402

RUN = "e2b-v2fix-cot-80k-noheld"


@torch.no_grad()
def generate(model, tok, sentences, **kw) -> list[list[str]]:
    """Returns, per sentence, the decoded answers (num_return_sequences of them)."""
    device = next(model.parameters()).device
    tok.padding_side = "left"
    enc = tok([prompt(s) for s in sentences], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    k = kw.get("num_return_sequences", 1)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        out = model.generate(**enc, max_new_tokens=700, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                             **kw)
    tok.padding_side = "right"
    dec = [tok.decode(r, skip_special_tokens=True).split("\n")[0].strip() for r in out[:, enc["input_ids"].shape[1]:]]
    return [dec[i * k:(i + 1) * k] for i in range(len(sentences))]


def vote(answers: list[str]) -> str:
    c = collections.Counter(a.partition("=>")[0].strip() for a in answers)
    top = max(c.values())
    return next(a for a in answers if c[a.partition("=>")[0].strip()] == top)


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    from scipy.stats import binomtest
    x = sum(p and not q for p, q in zip(a, b))
    y = sum(q and not p for p, q in zip(a, b))
    return x, y, (binomtest(x, x + y, 0.5).pvalue if x + y else 1.0)


def main() -> None:
    args = [a for i, a in enumerate(sys.argv[1:]) if not a.startswith("--") and (i == 0 or sys.argv[i] != "--n")]
    run = args[0] if args else RUN
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 600
    test = json.loads((ROOT / "results" / "tatoeba_test.json").read_text(encoding="utf-8"))[:n]
    model, tok, _ = load_trained(ROOT / "results" / "kaggle" / run / "model")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    settings = {"greedy": (8, dict(do_sample=False)),
                "beam 4": (4, dict(do_sample=False, num_beams=4)),
                "vote 5": (4, dict(do_sample=True, temperature=0.7, top_p=1.0, num_return_sequences=5))}
    # partial results are saved after every batch, so a killed run resumes where it stopped
    part = ROOT / "results" / f"decode_tricks_{run}.partial.json"
    preds = json.loads(part.read_text(encoding="utf-8")) if part.exists() else {}
    for name, (bs, kw) in settings.items():
        out = preds.setdefault(name, [])
        for i in range(len(out), len(test), bs):
            torch.manual_seed(i)  # seeded per batch so sampling does not depend on where a resume started
            out += [vote(a) if len(a) > 1 else a[0] for a in generate(model, tok, [t["english"] for t in test[i:i + bs]], **kw)]
            part.write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")
            print(f"  {name} {len(out)}/{len(test)}", flush=True)
        preds[name] = out[:len(test)]
    meaning = {k: [p.partition("=>")[0].strip() == t["analysis"] for p, t in zip(v, test)] for k, v in preds.items()}
    glyphs = {k: [p.partition("=>")[2].strip() == t["target"] for p, t in zip(v, test)] for k, v in preds.items()}
    md = [f"# Decoding tricks on Tatoeba (`analysis/decode_tricks.py`), run {run}, {len(test)} sentences", "",
          "| decoding | meaning exact | glyphs exact | vs greedy (meaning): fixes / breaks | McNemar p |",
          "|---|---|---|---|---|"]
    for k in preds:
        x, y, p = mcnemar(meaning[k], meaning["greedy"])
        md.append(f"| {k} | {sum(meaning[k]) / len(test):.3f} | {sum(glyphs[k]) / len(test):.3f} | "
                  + ("-" if k == "greedy" else f"{x} / {y} | {p:.2g}") + (" | - |" if k == "greedy" else " |"))
    ref = ROOT / "results" / "tatoeba" / f"{run}.json"
    if ref.exists():
        old = json.loads(ref.read_text(encoding="utf-8")).get("preds", [])[:len(test)]
        md += ["", f"Greedy reproduces the saved Tatoeba predictions on {sum(a == b for a, b in zip(old, preds['greedy']))}"
                   f" of {len(old)} sentences."]
    (ROOT / "results" / f"decode_tricks_{run}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (ROOT / "results" / f"decode_tricks_{run}.json").write_text(json.dumps({"run": run, "preds": preds}, ensure_ascii=False),
                                                                 encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
