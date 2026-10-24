"""Paraphrased training English for the 'English variety' run (charter claim 4). Run locally (frozen model, no training).

    python scripts/build_paraphrases.py --seed 0 [--share 0.3] [--model Qwen/Qwen2.5-1.5B-Instruct] [--limit N]

Rebuilds exactly the training examples the run will use (train/represent_v2.build_training with the run's settings:
80k, data fix min_count 12, nothing held out), picks a share of them (fixed seed), and asks a frozen instruction-tuned
model to say each sentence another natural way. A paraphrase is ACCEPTED only if the sealed-set reader reads it to
exactly the original meaning (same analysis), so every label stays right; otherwise the example keeps its original
English. Output: train/data/paraphrases_seed{S}.json {original english: paraphrase}, used by build_training(...,
paraphrase_file=...). Generated English only; never the sealed set.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sealed" / "app"))

from english_parse import Parser, render  # noqa: E402

import train.represent_v2 as R2  # noqa: E402

SYSTEM = ("You rewrite English sentences in a different grammatical form WITHOUT changing any word's meaning. "
          "Keep EVERY noun, verb, describing word, name, number and helper verb (can, could, might, must, will...) "
          "exactly as given - never replace a word with a synonym, even if the sentence sounds odd. You may only: "
          "move a place or time phrase to the front, use contractions (do not -> don't, will -> 'll), use 'there is/"
          "are', turn an active sentence with an object into a passive with 'by' (or back), use 'it was X that...', "
          "move 'not', change 'the' / 'a', change punctuation. If no such change fits, add a contraction. "
          "Answer with the new sentence only.")
SHOTS = [("The old dog did not bite the man in the garden.", "In the garden, the man wasn't bitten by the old dog."),
         ("Mira will give the book to the children.", "Mira'll give the book to the children."),
         ("Where might Kavi be brushing the emeralds?", "Where might the emeralds be being brushed by Kavi?"),
         ("A tall woman is in the kitchen.", "There is a tall woman in the kitchen."),
         ("The weird fighters will not be these ants.", "These ants won't be the weird fighters.")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--share", type=float, default=0.3)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--limit", type=int, default=0, help="only try this many (speed test)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cands", type=int, default=4, help="sampled candidates per sentence (first accepted is kept)")
    a = ap.parse_args()
    items, _ = R2.build_training("tamil_like", "abugida_pua", a.seed, n=80000, n_refuse=2500, n_replay=6000,
                                 cot=False, held_frac=0.0, min_count=12)
    rep = [it for it in items if it.get("aug") == "rep2" and it.get("clean")]   # typos/case/spacing stay as made
    rng = random.Random(1000 + a.seed)
    pick = rng.sample(rep, round(a.share * len(rep)))
    if a.limit:
        pick = pick[:a.limit]
    parser = Parser()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to("cuda").eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to("cuda").eval()

    def prompt(s):
        msgs = [{"role": "system", "content": SYSTEM}]
        for u, v in SHOTS:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": v}]
        msgs.append({"role": "user", "content": s})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    out_path = a.out or ROOT / "train" / "data" / f"paraphrases_seed{a.seed}.json"
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() and not a.limit else {}
    tried = accepted = 0
    t0 = time.time()
    todo = [it for it in pick if it["english"] not in done]
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        enc = tok([prompt(it["english"]) for it in chunk], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.95,
                               num_return_sequences=a.cands, pad_token_id=tok.eos_token_id)
        texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, it in enumerate(chunk):
            tried += 1
            for t in texts[a.cands * j: a.cands * (j + 1)]:
                t = t.strip().splitlines()[0].strip().strip('"') if t.strip() else ""
                if not t or t == it["english"] or not t[0].isalnum():
                    continue
                r = parser.parse(t)
                if r.clauses and not r.issues and render(r.clauses, parser.reps)["analysis"] == it["analysis"]:
                    done[it["english"]] = t
                    accepted += 1
                    break
        if not a.limit and (i // a.batch) % 25 == 0:
            out_path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
            el = time.time() - t0
            print(f"{tried}/{len(todo)} tried, {accepted} accepted ({accepted / max(1, tried):.0%}), {el / 60:.1f} min, "
                  f"eta {el / max(1, tried) * (len(todo) - tried) / 60:.0f} min", flush=True)
    if not a.limit:
        out_path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    el = time.time() - t0
    print(f"done: tried {tried}, accepted {accepted} ({accepted / max(1, tried):.0%}), {el:.0f} s "
          f"({el / max(1, tried):.2f} s per sentence); total paraphrases {len(done)}")
    if a.limit:
        for k, v in list(done.items())[:12]:
            print(f"  {k}\n   -> {v}")


if __name__ == "__main__":
    main()
