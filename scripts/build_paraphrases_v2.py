"""Paraphrase run v2: reword 10-20% of the training English (v1 reached only 1.4%). Frozen model, no training.

    python scripts/build_paraphrases_v2.py --seed 0 --target 0.15 [--model Qwen/Qwen2.5-1.5B-Instruct] [--cands 8]
           [--limit N] [--out FILE] [--engine auto|vllm|hf]

v1 paraphrased only "clean" examples (no typos / odd casing / double spaces), 6% of the data. v2 paraphrases the CLEAN
form of every example (the generator's text before casing, typos and end mark), then puts the same kind of surface
back: the same end mark, casing style, spacing and kind of typo (new random typo positions). Only examples whose
English uses a synonym (lexical = synonym, ~20%) stay out: the reader does not know those words.

A paraphrase is accepted only if the sealed-set reader reads it (with the example's end mark, sentence casing, no
typos) to EXACTLY the example's analysis, so every label stays right. Examples are tried in a fixed random order and
the run stops when --target x 80k paraphrases are accepted. Output {original english: paraphrased english}, used by
build_training(paraphrase_file=...). Generated English only; never the sealed set.

Speed: vLLM (T4 or newer; prefix caching makes the long shared prompt cheap) with a plain transformers fallback;
the reader checks run in parallel processes (~7 ms each).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sealed" / "app"))

SYSTEM = ("You rewrite English sentences in a different grammatical form WITHOUT changing any word's meaning. "
          "Keep EVERY noun, verb, describing word, name, number and helper verb (can, could, might, must, will...) "
          "exactly as given - never replace a word with a synonym, even if the sentence sounds odd. You may only: "
          "move a place or time phrase to the front (with a comma), use contractions (do not -> don't, will -> 'll, "
          "they are -> they're), turn an active sentence with an object into a passive with 'by' (or back), use "
          "'there is/are', move 'not', change punctuation. Answer with the new sentence only.")
SHOTS = [("the old dog did not bite the man in the garden", "In the garden, the man wasn't bitten by the old dog."),
         ("Mira will give the book to the children", "Mira'll give the book to the children."),
         ("where might Kavi be brushing the emeralds", "Where might the emeralds be being brushed by Kavi?"),
         ("a tall woman is in the kitchen", "There is a tall woman in the kitchen."),
         ("they are not with another pet tonight", "Tonight, they're not with another pet."),
         ("the weird fighters will not be these ants", "These ants won't be the weird fighters."),
         ("who repaired the bridge yesterday", "Yesterday, who repaired the bridge?")]

_W: dict = {}


def _init():
    from english_parse import Parser
    import train.represent_v2 as R2
    _W["parser"] = Parser()
    _W["gen"] = R2.GenV2("tamil_like", "abugida_pua", 0)


def body_of(cand: str, names: set) -> str:
    t = cand.strip().strip('"').strip()
    t = re.sub(r"[\s,;:.!?]+$", "", t).strip()
    if not t:
        return ""
    first = t.split(" ")[0]
    if first.strip(",'") not in names and first != "I" and not first.startswith("I'"):
        t = t[0].lower() + t[1:]
    return t


def check(job):
    """(index, candidates, raw, analysis, surf, english) -> (index, final english or None)."""
    from english_parse import render
    i, cands, raw, analysis, surf, english = job
    parser, gen = _W["parser"], _W["gen"]
    names = set(re.findall(r"NAME\.(\w+)", analysis))
    names |= {n[:1].upper() + n[1:].lower() for n in names}
    for c in cands:
        b = body_of(c, names)
        if not b or b.lower() == raw.lower() or len(b) > 3 * len(raw) + 20:
            continue
        if re.match(r"(?i)\s*(how|what|who|whom|whose|where|when|which|why)\b", raw) and not re.match(
                r"(?i)\s*((how|what|who|whom|whose|where|when|which|why)\b|[^,]*,)", b):
            continue                                     # 'Noci swung what': echo questions read as odd English
        probe = gen.surface(random.Random(0), b, dict(surf, noise="none", spacing="single", case="sentence",
                                                       lexical="plain"))
        r = parser.parse(probe)
        if not r.clauses or r.issues:
            continue
        if render(r.clauses, parser.reps)["analysis"] != analysis:
            continue
        final = gen.surface(random.Random(zlib.crc32((english + '|' + b).encode())), b, dict(surf, lexical="plain"))
        if final and final != english:
            return i, final
    return i, None


class Generator:
    def __init__(self, model: str, engine: str, cands: int):
        self.cands = cands
        self.engine = engine
        if engine in ("auto", "vllm"):
            try:
                from vllm import LLM, SamplingParams
                self.llm = LLM(model=model, dtype="half", max_model_len=1024, gpu_memory_utilization=0.85,
                               enable_prefix_caching=True)
                self.sp = SamplingParams(n=cands, temperature=0.8, top_p=0.95, max_tokens=70)
                self.tok = self.llm.get_tokenizer()
                self.engine = "vllm"
                return
            except Exception as e:                       # noqa: BLE001  (fall back to transformers)
                if engine == "vllm":
                    raise
                print("vLLM unavailable, using transformers:", repr(e)[:200], flush=True)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float16).to("cuda").eval()
        self.engine = "hf"

    def prompt(self, s: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}]
        for u, v in SHOTS:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": v}]
        msgs.append({"role": "user", "content": s})
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def __call__(self, texts: list[str]) -> list[list[str]]:
        ps = [self.prompt(t) for t in texts]
        if self.engine == "vllm":
            outs = self.llm.generate(ps, self.sp, use_tqdm=False)
            return [[o.text.strip().split("\n")[0] for o in r.outputs] for r in outs]
        torch = self.torch
        out = []
        for s0 in range(0, len(ps), 24):                    # small batches: 24 sentences x cands sequences
            enc = self.tok(ps[s0:s0 + 24], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.95,
                                        num_return_sequences=self.cands, pad_token_id=self.tok.eos_token_id)
            dec = self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out += [[x.strip().split("\n")[0] for x in dec[k * self.cands:(k + 1) * self.cands]]
                    for k in range(len(dec) // self.cands)]
        return out


def main() -> None:
    import train.represent_v2 as R2
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=float, default=0.15, help="stop at this share of the 80k examples")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--cands", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=512, help="sentences per generation call")
    ap.add_argument("--limit", type=int, default=0, help="only try this many (test)")
    ap.add_argument("--engine", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--shard", type=int, default=0, help="this process's share of the pool (one per GPU)")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=0, help="stop and keep what is done after this long")
    ap.add_argument("--skip", default="", help="continue a sharded run: tried counts per old shard, e.g. 12480,13680")
    ap.add_argument("--merge", type=Path, default=None, help="existing paraphrase file to keep and add to")
    a = ap.parse_args()
    items, _ = R2.build_training("tamil_like", "abugida_pua", a.seed, n=80000, n_refuse=2500, n_replay=6000,
                                 cot=False, held_frac=0.0, min_count=12)
    pool = [it for it in items if it.get("aug") == "rep2" and it.get("raw") and it.get("surf")
            and it["surf"].get("lexical") == "plain"]
    n_rep = sum(1 for it in items if it.get("aug") == "rep2")
    goal = round(a.target * n_rep)
    rng = random.Random(2000 + a.seed)
    rng.shuffle(pool)
    if a.skip:                                           # the rest of an earlier run's shards, in the same order
        done_n = [int(x) for x in a.skip.split(",")]
        k = len(done_n)
        rest = [pool[g::k][done_n[g]:] for g in range(k)]
        pool = [x for grp in zip(*rest) for x in grp] + [x for r in rest for x in r[min(map(len, rest)):]]
    pool = pool[a.shard::a.nshards]
    goal = -(-goal // a.nshards)
    if a.limit:
        pool = pool[:a.limit]
    out_path = a.out or ROOT / "train" / "data" / f"paraphrases_v2_seed{a.seed}.json"
    print(f"{n_rep} translation examples; pool {len(pool)} (synonym examples left out); goal {goal} "
          f"({a.target:.0%})", flush=True)
    gen = Generator(a.model, a.engine, a.cands)
    print("engine:", gen.engine, flush=True)
    done: dict[str, str] = json.loads(a.merge.read_text(encoding="utf-8")) if a.merge else {}
    if a.merge:
        print(f"keeping {len(done)} paraphrases from {a.merge}", flush=True)
    t0, tried = time.time(), 0
    with Pool(a.workers, initializer=_init) as workers:
        for s in range(0, len(pool), a.chunk):
            chunk = pool[s:s + a.chunk]
            cands = gen([it["raw"] for it in chunk])
            jobs = [(k, c, it["raw"], it["analysis"], it["surf"], it["english"]) for k, (it, c) in
                    enumerate(zip(chunk, cands))]
            for k, final in workers.imap_unordered(check, jobs, chunksize=8):
                if final:
                    done[chunk[k]["english"]] = final
            tried += len(chunk)
            el = time.time() - t0
            print(f"{tried}/{len(pool)} tried, {len(done)} accepted ({len(done) / tried:.0%}; "
                  f"{len(done) / n_rep:.1%} of training), {el / 60:.1f} min", flush=True)
            out_path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
            if not a.limit and len(done) >= goal:
                break
            if a.max_minutes and el / 60 > a.max_minutes:
                print("time limit reached", flush=True)
                break
    print(f"done: {len(done)} paraphrases = {len(done) / n_rep:.1%} of the translation examples, tried {tried}, "
          f"{(time.time() - t0) / 60:.1f} min -> {out_path}", flush=True)
    for k, v in list(done.items())[:15]:
        print(f"  {k}\n   -> {v}")


if __name__ == "__main__":
    main()
