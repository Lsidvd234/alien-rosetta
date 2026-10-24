"""Grokking runs: does generalisation arrive LONG after the training set is fitted? One chunk per call (resumable).

    python train/grok.py --task maths|translate [--init mean|meaning|shuffled] --steps-total 30000 --chunk-steps 10000
           [--resume auto|PATH] --save DIR --out results/grok_x.json

Grokking needs (Power et al. 2022; Kumar et al. 2024): a small, finite task the model can memorise, training far past
100% train accuracy, weight decay, a constant learning rate, and starting features that do not fit the task. So:
  maths      the full table of base-7 addition and multiplication of 0..48 (1-2 alien digits): 2 x 49 x 49 = 4,802
             problems; a fixed random 40% trains, the other 60% tests (same sizes, new pairs); extra test: 3-digit
             problems (longer than any trained).
  translate  a small finite version of the v2 language: the generator's word pools cut to the most common words
             (~120 nouns, 75 verbs, 60 describing words, 30 places, 20 how-words), clean English (no typos / casing
             noise), 4,000 training sentences, direct answers (no meaning step). Test: new sentences whose every word
             appeared >= 3 times in training and whose clause structures all appeared (only new combinations of known
             pieces); extra test: sentences with a clause structure never trained.
Glyph tokens start with no meaning (mean) by default. AdamW with weight decay (also on the glyph table), constant
learning rate after warmup. Every --eval-every steps: exact match on a training sample, the test set and the extra
test. The whole training state (trainable weights, optimiser, step, curve) is saved at the end of the chunk, so the
next chunk continues exactly (the data order is a function of the global step).
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.collapse import collapse_stats  # noqa: E402
from train.maths import DIGITS, MEANING, OPS, alien, evaluate  # noqa: E402
from train.run import (DTYPES, add_glyph_tokens, batches_by_length, collate, log, pick_precision,  # noqa: E402
                       prompt)

FIELD_RE = re.compile(r"(?<![A-Za-z0-9_.>^+&|])([A-Z]+)=(\S+)")
KEY_RE = re.compile(r"(?<![A-Z.])([a-z][a-z_]*(?:>[A-Z]+)*)")
CLEAN = {"noise": "none", "lexical": "plain", "spacing": "single", "case": "sentence", "end": "full stop"}


# --- data ------------------------------------------------------------------------------------------------------
def maths_data(seed: int, frac: float) -> tuple[list[dict], dict[str, list[dict]], list[str], list[str]]:
    rng = random.Random(seed)
    table = []
    for op in ("+", "x"):
        for a in range(49):
            for b in range(49):
                val = a + b if op == "+" else a * b
                table.append({"prompt_text": f"Alien: {alien(a)} {OPS[op]} {alien(b)}\nAlien:\n", "target": alien(val)})
    rng.shuffle(table)
    k = round(frac * len(table))
    train, test = table[:k], table[k:]
    longer = []
    for _ in range(200):
        op = rng.choice("+x")
        a, b = rng.randrange(49, 343), rng.randrange(49, 343) if op == "+" else rng.randrange(1, 7)
        val = a + b if op == "+" else a * b
        longer.append({"prompt_text": f"Alien: {alien(a)} {OPS[op]} {alien(b)}\nAlien:\n", "target": alien(val)})
    tests = {"train (sample)": rng.sample(train, 300), "test: new pairs, same sizes": rng.sample(test, 300),
             "extra: 3-digit (longer)": longer}
    added = DIGITS + list(OPS.values())
    return train, tests, added, [MEANING[g] for g in added]


def restrict(gen, sizes: dict) -> None:
    """Cut the generator's word pools to the most common words (finite task)."""
    def top(items, weights, n):
        order = sorted(range(len(items)), key=lambda i: -weights[i])[:n]
        return [items[i] for i in order], [weights[i] for i in order]
    gen.nouns, gen.noun_w = top(gen.nouns, gen.noun_w, sizes["nouns"])
    keep = {x[2] for x in gen.nouns}
    pp = [(x, w) for x, w in zip(gen.people, gen.people_w) if x[2] in keep] or list(zip(gen.people, gen.people_w))
    gen.people, gen.people_w = top([x for x, _ in pp], [w for _, w in pp], sizes["people"])
    gen.places, gen.places_w = top(gen.places, gen.places_w, sizes["places"])
    gen.adjs, gen.adj_w = top(gen.adjs, gen.adj_w, sizes["adjs"])
    gen.tverbs, gen.tverb_w = top(gen.tverbs, gen.tverb_w, sizes["tverbs"])
    gen.iverbs, gen.iverb_w = top(gen.iverbs, gen.iverb_w, sizes["iverbs"])
    gen.verbs, gen.verb_w = gen.tverbs + gen.iverbs, gen.tverb_w + gen.iverb_w
    gen.verb_by_base = {x[0]: x for x in gen.verbs}
    gen.manner, gen.manner_w = top(gen.manner, gen.manner_w, sizes["manner"])


def templates(analysis: str) -> list[str]:
    return [" ".join(f for f, _ in FIELD_RE.findall(c)) for c in analysis.split(";") if c.strip()]


def keys_of(analysis: str) -> list[str]:
    return KEY_RE.findall(analysis) + [f"PRON.{p}" for p in re.findall(r"PRON\.(\w+)", analysis)]


def translate_data(seed: int, n_train: int, n_test: int, n_extra: int, preset: str, script: str):
    import collections
    import train.represent_v2 as R2
    from conlang.script import Script
    from conlang.dataset import PRESETS
    R2.set_held_out(0)
    gen = R2.GenV2(preset, script, seed)
    restrict(gen, {"nouns": 120, "people": 30, "places": 30, "adjs": 60, "tverbs": 50, "iverbs": 25, "manner": 20})
    rng = random.Random(seed + 505)
    train, seen = [], set()
    while len(train) < n_train:
        ex = gen.example(rng, force=dict(CLEAN))
        it = gen.item(ex) if ex else None
        if it and it["english"] not in seen and len(it["target"]) < 400:
            seen.add(it["english"])
            train.append({"english": it["english"], "target": it["target"], "analysis": it["analysis"],
                          "prompt_text": prompt(it["english"])})
    counts = collections.Counter(k for it in train for k in keys_of(it["analysis"]))
    tmpl = {t for it in train for t in templates(it["analysis"])}
    rng2 = random.Random(seed + 606)
    test, extra, tries = [], [], 0
    while (len(test) < n_test or len(extra) < n_extra) and tries < 200000:
        tries += 1
        ex = gen.example(rng2, force=dict(CLEAN))
        it = gen.item(ex) if ex else None
        if not it or it["english"] in seen or len(it["target"]) >= 400:
            continue
        if any(counts[k] < 3 for k in keys_of(it["analysis"])):
            continue                                        # a word seen < 3 times: not a fair test
        rec = {"english": it["english"], "target": it["target"], "analysis": it["analysis"],
               "prompt_text": prompt(it["english"])}
        if all(t in tmpl for t in templates(it["analysis"])):
            if len(test) < n_test:
                test.append(rec)
        elif len(extra) < n_extra:
            extra.append(rec)
        seen.add(it["english"])
    tests = {"train (sample)": random.Random(seed).sample(train, min(150, len(train))),
             "test: known words, new combinations": test, "extra: new clause structure": extra}
    chars = sorted({c for it in train + test + extra for c in it["target"]})
    sounds = Script(script, PRESETS[preset].phonology).latin_of_glyphs()
    info = {"vocab_keys_in_training": len(counts), "templates_in_training": len(tmpl), "test_tries": tries}
    return train, tests, chars, [sounds.get(c, "") for c in chars], info


# --- training ------------------------------------------------------------------------------------------------
def stablemax_loss(model, batch: dict) -> torch.Tensor:
    """StableMax cross-entropy (Prieto et al., ICLR 2025): s(x) = x + 1 for x >= 0, 1 / (1 - x) for x < 0, then
    normalise. It grows linearly instead of exponentially, so the right token's probability does not round to exactly
    1.0 and the gradient does not vanish (no softmax collapse). Same ranking as softmax, so generation is unchanged.
    Computed in float64 on the answer tokens, as in the paper."""
    labels = batch["labels"]
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1]
    tgt = labels[:, 1:]
    m = tgt != -100
    x = logits[m].double()
    # each branch only sees its own range: torch.where computes BOTH, and 1/(1-x) at x = 1 would give inf -> NaN grads
    s = torch.where(x >= 0, x.clamp(min=0) + 1, 1 / (1 - x.clamp(max=0)))
    p = s.gather(1, tgt[m][:, None]).squeeze(1) / s.sum(1)
    return (-torch.log(p)).mean().float()


# --- training ------------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["maths", "translate"], required=True)
    ap.add_argument("--init", choices=["mean", "meaning", "shuffled"], default="mean")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--train-frac", type=float, default=0.4, help="maths: share of the table used for training")
    ap.add_argument("--n-train", type=int, default=4000, help="translate: training sentences")
    ap.add_argument("--steps-total", type=int, default=30000)
    ap.add_argument("--chunk-steps", type=int, default=0, help="stop this call after this many steps (0 = to the end)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--loss", choices=["ce", "stablemax"], default="ce",
                    help="stablemax: Prieto et al. 2025 StableMax cross-entropy (no softmax collapse)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-batch-tokens", type=int, default=1600, help="split bigger batches (GPU memory)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="", help="'auto' = find grok_state.pt under /kaggle/input, or a path")
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/grok.json"))
    ap.add_argument("--preset", default="tamil_like")
    ap.add_argument("--script", default="abugida_pua")
    a = ap.parse_args(argv)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prec = pick_precision("auto")
    amp = (lambda: torch.autocast("cuda", dtype=DTYPES[prec])) if (device == "cuda" and prec != "fp32") \
        else (lambda: torch.autocast("cpu", enabled=False))
    t0 = time.time()
    info = {}
    if a.task == "maths":
        train, tests, added, meaning = maths_data(a.seed, a.train_frac)
    else:
        train, tests, added, meaning, info = translate_data(a.seed, a.n_train, 300, 120, a.preset, a.script)
    log(f"data: {len(train)} train, tests {{{', '.join(f'{k}: {len(v)}' for k, v in tests.items())}}} "
        f"({time.time() - t0:.0f} s) {info}")
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    n_old = len(tok)
    added = [c for c in added if a.task == "maths" or len(tok(c, add_special_tokens=False)["input_ids"]) > 1]
    if a.task == "translate":
        meaning = None                                    # recomputed below for the kept characters
    tok.add_tokens(added)
    load_dtype = DTYPES[prec] if device == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=load_dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=load_dtype)
    model.to(device)
    model.config.use_cache = False
    if a.init == "mean":
        delta = add_glyph_tokens(model, n_old, len(added), a.seed, "mean")
    else:
        if a.task == "translate":
            from conlang.script import Script
            from conlang.dataset import PRESETS
            snd = Script(a.script, PRESETS[a.preset].phonology).latin_of_glyphs()
            meaning = [snd.get(c, "") for c in added]
        start = list(meaning)
        if a.init == "shuffled":
            rng = random.Random(a.seed + 1)
            idx = [i for i, s in enumerate(start) if s]
            perm = idx[:]
            while any(start[i] == start[j] for i, j in zip(idx, perm)):
                rng.shuffle(perm)
            start = list(start)
            for i, j in zip(idx, perm):
                start[i] = meaning[j]
        donors = [tok(s, add_special_tokens=False)["input_ids"] if s else [] for s in start]
        delta = add_glyph_tokens(model, n_old, len(added), a.seed, "latin", donors)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    delta.requires_grad_(True)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW(list(params.values()), lr=a.lr, weight_decay=a.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(prec == "fp16"))
    examples = []
    for it in train:
        p_ids = tok(it["prompt_text"], add_special_tokens=False)["input_ids"]
        t_ids = tok(it["target"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        examples.append({"ids": p_ids + t_ids, "labels": [-100] * len(p_ids) + t_ids})

    collapse_ex = random.Random(a.seed + 9).sample(examples, min(200, len(examples)))
    # resume
    step, curve = 0, []
    path = a.resume
    if path == "auto":
        hits = glob.glob("/kaggle/input/**/grok_state.pt", recursive=True)
        path = hits[0] if hits else ""
    if path:
        st = torch.load(path, map_location="cpu", weights_only=False)
        for n, v in st["params"].items():
            params[n].data.copy_(v.to(params[n].device))
        opt.load_state_dict(st["opt"])
        for g in opt.param_groups:                       # the optimiser state restores the OLD weight decay:
            g["weight_decay"] = a.weight_decay            # use this call's value (e.g. continue with stronger decay)
        step, curve = st["step"], st["curve"]
        log(f"resumed from {path} at step {step}")
    stop_at = min(a.steps_total, step + a.chunk_steps) if a.chunk_steps else a.steps_total
    per_epoch = len(batches_by_length(examples, a.batch, random.Random(0)))   # batches per epoch (fixed count)
    log(f"{a.task} init={a.init}: {len(examples)} examples ({per_epoch} steps/epoch), steps {step} -> {stop_at} "
        f"of {a.steps_total}, wd={a.weight_decay}, lr={a.lr} constant, precision {prec}")

    def evaluate_all():
        row = {"step": step, "epoch": round(step / per_epoch, 1)}
        cfg = model.get_base_model().config
        cfg.use_cache = True                              # fast generation; training runs without the cache
        for name, items in tests.items():
            r = evaluate(model, tok, items, device, amp, a.eval_batch)
            row[name] = r["exact"]
        cfg.use_cache = False
        cs = collapse_stats(model, tok, collapse_ex, device, amp, bsz=8)     # softmax collapse on fitted tokens
        row.update({"train_loss": round(cs["loss_mean"], 5), "zero_grad_fp32": round(cs["zero_grad_fp32"], 4),
                    "gap_median": round(cs["gap_median"], 2), "logit_norm": round(cs["logit_norm"], 1)})
        curve.append(row)
        log("curve " + json.dumps(row))
        return row

    if step == 0:
        evaluate_all()
    model.train()
    hist, t1 = [], time.time()
    while step < stop_at:
        epoch = step // per_epoch
        batches = batches_by_length(examples, a.batch, random.Random(a.seed * 100003 + epoch))
        for b in batches[step % per_epoch:]:
            for g in opt.param_groups:
                g["lr"] = a.lr * min(1.0, (step + 1) / a.warmup)
            # a batch bigger than the token budget runs in pieces (gradients add up, token-weighted as one batch)
            T = max(len(e["ids"]) for e in b)
            k = max(1, -(-len(b) * T // a.max_batch_tokens))
            pieces = [b[i::k] for i in range(k)] if k > 1 else [b]
            n_lab = sum(sum(x != -100 for x in e["labels"]) for e in b)
            loss, bad = 0.0, False
            for piece in pieces:
                batch = collate(piece, tok.pad_token_id, device)
                share = int((batch["labels"] != -100).sum()) / max(1, n_lab)
                with amp():
                    if a.loss == "stablemax":
                        pl = stablemax_loss(model, batch) * share
                    else:
                        pl = model(**batch).loss * share
                if not torch.isfinite(pl):
                    bad = True
                    break
                scaler.scale(pl).backward()
                loss += pl.item()
            if bad:
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            loss = torch.tensor(loss)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            hist.append(loss.item())
            if step % 100 == 0:
                log(f"step {step} loss {sum(hist[-100:]) / len(hist[-100:]):.4f} ({(time.time() - t1) / 60:.1f} min)")
            if step % a.eval_every == 0:
                evaluate_all()
            if step >= stop_at:
                break
    if not curve or curve[-1]["step"] != step:
        evaluate_all()
    if a.save:
        a.save.mkdir(parents=True, exist_ok=True)
        torch.save({"params": {n: p.detach().cpu() for n, p in params.items()}, "opt": opt.state_dict(),
                    "step": step, "curve": curve, "args": vars(a) | {"save": str(a.save), "out": str(a.out)}},
                   a.save / "grok_state.pt")
        log(f"saved state at step {step} -> {a.save / 'grok_state.pt'}")
    res = {"task": a.task, "init": a.init, "step": step, "steps_total": a.steps_total, "done": step >= a.steps_total,
           "steps_per_epoch": per_epoch, "curve": curve, "data": info, "minutes": (time.time() - t0) / 60,
           "args": {k: str(v) for k, v in vars(a).items()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
