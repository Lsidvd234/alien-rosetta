"""Per-loanword accuracy of the loanword run (local; generated sentences only, never the sealed set).

    python analysis/loan_words.py [--n 150]   -> results/loanwords.md

Sentences with borrowed English nouns (half of the noun phrases become loans), once with loans SEEN in training and
once with loans held out of training (new). For every loan noun in a gold answer:
  meaning right : the model's meaning contains the same LOAN.word (it recognised the loan and spelled its key)
  glyphs right  : the loan's glyph word (sound spelling + LOAN ending + case/plural) is in the model's glyphs, aligned
The whole-sentence scores of the run's own tests understate this (long sentences with several loans).
"""

from __future__ import annotations

import collections
import difflib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

RUN = "e2b-v2fix-loan-r2"


def main() -> None:
    import train.represent_v2 as R2
    from subgraph_check import training_pieces
    from train.load import load_trained, translate
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 150
    mdir = ROOT / "results" / "kaggle" / RUN / "model"
    cfg = json.loads((mdir / "run_args.json").read_text(encoding="utf-8"))
    _, train_en, _, _ = training_pieces(cfg, ROOT / "results" / "subgraph" / f"{RUN}.pieces.json")
    R2.set_held_out(0)
    gen = R2.GenV2("tamil_like", "abugida_pua", 0)
    sets = {}
    for pool in ("train", "held"):
        R2.LOAN.update(rate=0.5, pool=pool)
        rng = random.Random(31 if pool == "train" else 32)
        items = []
        while len(items) < n:
            ex = gen.example(rng)
            it = gen.item(ex) if ex else None
            if it and "LOAN." in it["analysis"] and it["english"] not in train_en:
                words = [(w.gloss, gen.rep.script.word(w.phonemes)) for cl in ex["clauses"] for w in gen.rep.words(cl)]
                items.append({**it, "words": words})
        sets["seen loans" if pool == "train" else "new loans (never trained)"] = items
    R2.LOAN.update(rate=0.0, pool="train")
    model, tok, _ = load_trained(mdir)
    L = ["# Per-loanword accuracy (`analysis/loan_words.py`)", "",
         f"Run {RUN}. {n} generated sentences per set; every loan noun in the gold answer is scored.", "",
         "| set | loan nouns | meaning right (LOAN.word) | glyph word right | sentences fully right |", "|---|---|---|---|---|"]
    examples = []
    for name, items in sets.items():
        preds = []
        for i in range(0, len(items), 8):
            preds += translate(model, tok, [x["english"] for x in items[i:i + 8]], max_new_tokens=700)
            print(f"  {name} {len(preds)}/{len(items)}", flush=True)
        tot = mean_ok = gl_ok = 0
        sent_ok = 0
        for it, p in zip(items, preds):
            ana, _, gly = p.partition("=>")
            ana, gly = ana.strip(), gly.strip()
            sent_ok += gly == it["target"]
            pred_loans = collections.Counter(re.findall(r"LOAN\.[a-z]+", ana))
            gold_glyphs = [g for _, g in it["words"]]
            ok = [False] * len(gold_glyphs)
            for blk in difflib.SequenceMatcher(a=gold_glyphs, b=gly.split(), autojunk=False).get_matching_blocks():
                for j in range(blk.size):
                    ok[blk.a + j] = True
            for k, (gloss, _) in enumerate(it["words"]):
                m = re.match(r"(LOAN\.[a-z]+)", gloss)
                if not m:
                    continue
                tot += 1
                if pred_loans[m.group(1)] > 0:
                    mean_ok += 1
                    pred_loans[m.group(1)] -= 1
                gl_ok += ok[k]
            if len(examples) < 12 and name.startswith("new") and "LOAN." in it["analysis"]:
                examples.append((it["english"], it["analysis"], ana))
        L.append(f"| {name} | {tot} | **{mean_ok / tot:.2f}** | **{gl_ok / tot:.2f}** | {sent_ok / len(items):.2f} |")
    L += ["", "## Examples (new loans): gold meaning vs the model's", ""]
    for e, g, a in examples:
        L += [f"- {e}  \n  gold: `{g}`  \n  model: `{a}`"]
    (ROOT / "results" / "loanwords.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:8]))


if __name__ == "__main__":
    main()
