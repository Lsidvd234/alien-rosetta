"""Evaluate the two-stage pipeline against the translator alone, on generated sentences only (never the sealed set).

    python two_stage/evaluate.py [--rewriter Qwen/Qwen3-0.6B] [--translator e2b-v2fix-cot-80k-noheld] [--n 100]

Test sets (meaning and glyph answers are known exactly):
  unknown-word  generated statements where one word is replaced by an English synonym that is NOT a word of the
                language (from the WordNet candidate table, relation 'synonym', inverted): same meaning, same answer
  held-out synonyms  the v2 generator's held-out synonym pairs
  control       ordinary generated sentences inside the language (the pipeline must not make them worse)
Stage 1 runs for all sets first (rewriter loaded once), then stage 2 (translator loaded once): the 4 GB GPU holds one
model at a time. Writes results/two_stage/<rewriter>__<translator>.json and prints a table.
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train.represent_v2 as R2  # noqa: E402
from two_stage.pipeline import CANDIDATES, Rewriter  # noqa: E402
from english_parse import FUNCTION  # noqa: E402

CLEAN = {"noise": "none", "lexical": "plain", "spacing": "single", "case": "sentence", "end": "full stop"}


def unknown_word_set(gen: R2.GenV2, rng, n: int) -> list[dict]:
    """Statements with one language word swapped for an English synonym outside the language."""
    inv: dict[str, list[str]] = collections.defaultdict(list)
    for u, cands in CANDIDATES.items():
        for c, pos, rel in cands[:3]:
            if rel == "synonym" and u.isalpha() and len(u) > 2:
                inv[c].append(u)
    out, tries = [], 0
    while len(out) < n and tries < 20000:
        tries += 1
        ex = gen.example(rng, force=dict(CLEAN, kind="statement"), n_events=1)
        it = gen.item(ex) if ex else None
        if not it:
            continue
        words = it["english"].rstrip(".").split(" ")
        slots = [i for i, w in enumerate(words) if w.lower() in inv and i > 0 and w.lower() not in FUNCTION]
        if not slots:
            continue
        i = rng.choice(slots)
        new = rng.choice(inv[words[i].lower()])
        words[i] = new
        out.append({**it, "english": " ".join(words) + ".", "swapped": new})
    return out


def sets(n: int) -> dict[str, list[dict]]:
    R2.set_held_out(0)
    gen = R2.GenV2("tamil_like", "abugida_pua", 0)
    rng = random.Random(777)
    held = []
    while len(held) < n:
        ex = R2._syn_held_example(gen, rng)
        it = gen.item(ex) if ex else None
        if it:
            held.append(it)
    ctrl = []
    while len(ctrl) < n:
        ex = gen.example(rng, force=dict(CLEAN), n_events=1)
        it = gen.item(ex) if ex else None
        if it:
            ctrl.append(it)
    return {"unknown-word": unknown_word_set(gen, rng, n), "held-out synonyms": held, "control": ctrl}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewriter", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--translator", default="e2b-v2fix-cot-80k-noheld")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    data = sets(a.n)
    t0 = time.time()
    rw = Rewriter(a.rewriter)
    rewrites = {k: rw.rewrite([it["english"] for it in v]) for k, v in data.items()}
    rw.close()
    del rw
    gc.collect()
    torch.cuda.empty_cache()
    t1 = time.time()
    from train.load import load_trained, translate
    model, tok, cfg = load_trained(ROOT / "results" / "kaggle" / a.translator / "model")
    cot = str(cfg.get("cot")) == "True"

    def run(texts):
        preds = []
        for i in range(0, len(texts), 8):
            preds += translate(model, tok, texts[i:i + 8], max_new_tokens=700)
        return preds
    report = {"rewriter": a.rewriter, "translator": a.translator, "sets": {}}
    lines = [f"rewriter {a.rewriter} | translator {a.translator} | stage 1 {t1 - t0:.0f} s"]
    lines.append(f"{'set':18} {'n':>4} {'kept':>5} {'subst':>5} {'rewr':>5} {'fallb':>5} | {'alone: glyph':>12} {'meaning':>7}"
                 f" | {'2-stage: glyph':>14} {'meaning':>7}")
    for k, items in data.items():
        rws = rewrites[k]
        alone = run([it["english"] for it in items])
        two = run([r.text for r in rws])

        def score(preds):
            g = sum((p.partition("=>")[2].strip() if cot else p) == it["target"] for p, it in zip(preds, items))
            m = sum(p.partition("=>")[0].strip() == it["analysis"] for p, it in zip(preds, items)) if cot else 0
            return g / len(items), m / len(items)
        ga, ma = score(alone)
        gt, mt = score(two)
        acts = collections.Counter(r.action for r in rws)
        lines.append(f"{k:18} {len(items):4} {acts['kept']:5} {acts['substituted']:5} {acts['rewritten']:5} "
                     f"{acts['fallback']:5} | "
                     f"{ga:12.3f} {ma:7.3f} | {gt:14.3f} {mt:7.3f}")
        report["sets"][k] = {"n": len(items), "actions": dict(acts), "alone": {"glyph": ga, "meaning": ma},
                             "two_stage": {"glyph": gt, "meaning": mt},
                             "examples": [{"english": it["english"], "rewrite": r.text, "action": r.action,
                                           "tries": r.tries, "subs": r.subs, "gold": it["analysis"],
                                           "alone": pa.partition("=>")[0].strip(),
                                           "two_stage": pt.partition("=>")[0].strip()}
                                          for it, r, pa, pt in zip(items, rws, alone, two)][:40]}
    out = ROOT / "results" / "two_stage"
    out.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9.-]+", "_", f"{a.rewriter}__{a.translator}")
    (out / f"{name}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
