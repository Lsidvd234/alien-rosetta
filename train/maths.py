"""Maths extension (charter claim 1 extension): alien number notation, base 7, with its own digit and operator glyphs.

    python train/maths.py --init meaning|shuffled|mean [--n 60000] [--epochs 2] [--out results/maths_meaning.json]

The notation: seven new digit glyphs (base 7, most significant digit first) and new glyphs for plus, times, minus.
Tasks (one fine-tune, mixed):
  to    decimal number -> alien numeral                 Number: 342 / Alien: <glyphs>
  from  alien numeral -> decimal number
  add   alien a + alien b -> alien sum                   (base-7 carrying)
  mul   alien a x alien b -> alien product               (the Faith and Fate multiplication task, in a new notation)
Training uses short numbers only (to/from <= 4 alien digits, add <= 3 digits each, mul <= 2 x 2 digits). Tests: the
same lengths on unseen pairs (in-distribution) and longer ones (length generalisation: 5 digits, 4+4 add, 3x2, 3x3).
The glyph tokens start as (--init) meaning: the embedding of what they mean (digit glyph 3 = token '3', plus = '+'),
shuffled: the embedding of a WRONG meaning (digits deranged), mean: no meaning (average embedding). Question: does
starting a new symbol at its meaning help a model compute with it, and does any start help beyond the trained lengths?
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.run import DTYPES, add_glyph_tokens, batches_by_length, collate, log, pick_precision  # noqa: E402

BASE = 7
DIGITS = [chr(0xF0100 + i) for i in range(BASE)]           # supplementary private use area: no clash with the script
OPS = {"+": chr(0xF0110), "x": chr(0xF0111), "-": chr(0xF0112)}
MEANING = {**{g: str(i) for i, g in enumerate(DIGITS)}, OPS["+"]: " plus", OPS["x"]: " times", OPS["-"]: " minus"}


def alien(n: int) -> str:
    s = ""
    while True:
        n, d = divmod(n, BASE)
        s = DIGITS[d] + s
        if n == 0:
            return s


def rand_len(rng, k: int) -> int:
    """A number with exactly k alien digits."""
    return rng.randrange(BASE ** (k - 1) if k > 1 else 0, BASE ** k)


def example(rng, task: str, la: int, lb: int = 0) -> dict:
    a = rand_len(rng, la)
    if task == "to":
        return {"prompt_text": f"Number: {a}\nAlien:\n", "target": alien(a), "key": ("to", a)}
    if task == "from":
        return {"prompt_text": f"Alien: {alien(a)}\nNumber:\n", "target": str(a), "key": ("from", a)}
    b = rand_len(rng, lb)
    op, val = ("+", a + b) if task == "add" else ("x", a * b)
    return {"prompt_text": f"Alien: {alien(a)} {OPS[op]} {alien(b)}\nAlien:\n", "target": alien(val),
            "key": (task, a, b)}


TRAIN_SPEC = [("to", lambda r: (r.randint(1, 4), 0)), ("from", lambda r: (r.randint(1, 4), 0)),
              ("add", lambda r: (r.randint(1, 3), r.randint(1, 3))), ("mul", lambda r: (r.randint(1, 2), r.randint(1, 2)))]
TEST_SPEC = {"to: 1-4 digits (trained lengths)": ("to", lambda r: (r.randint(1, 4), 0)),
             "to: 5 digits": ("to", lambda r: (5, 0)),
             "from: 1-4 digits (trained lengths)": ("from", lambda r: (r.randint(1, 4), 0)),
             "from: 5 digits": ("from", lambda r: (5, 0)),
             "add: up to 3+3 (trained lengths)": ("add", lambda r: (r.randint(1, 3), r.randint(1, 3))),
             "add: 4+4": ("add", lambda r: (4, 4)),
             "add: 5+5": ("add", lambda r: (5, 5)),
             "mul: up to 2x2 (trained lengths)": ("mul", lambda r: (r.randint(1, 2), r.randint(1, 2))),
             "mul: 3x1": ("mul", lambda r: (3, 1)),
             "mul: 3x2": ("mul", lambda r: (3, 2)),
             "mul: 3x3": ("mul", lambda r: (3, 3))}


def digits7(n: int) -> set[int]:
    s = set()
    while True:
        n, d = divmod(n, BASE)
        s.add(d)
        if n == 0:
            return s


def needs_pair(it: dict, pair: tuple[int, int] | None) -> bool:
    """A multiplication whose long-hand working needs the single-digit product pair[0] x pair[1] (either order)."""
    if pair is None or it["key"][0] != "mul":
        return False
    da, db = digits7(it["key"][1]), digits7(it["key"][2])
    p, q = pair
    return (p in da and q in db) or (q in da and p in db)


def build(n: int, per_test: int, seed: int, train_spec=None, extra_tests=None,
          drop: tuple[int, int] | None = None) -> tuple[list[dict], dict[str, list[dict]]]:
    """train_spec / extra_tests: longer training lengths (--add-max, --mul-max); drop: data removal (--drop-pair).
    With the defaults the data is exactly as before."""
    train_spec = train_spec or TRAIN_SPEC
    rng = random.Random(seed)
    tests, used = {}, set()
    for name, (task, lens) in {**TEST_SPEC, **(extra_tests or {})}.items():   # tests first: never trained
        items = []
        while len(items) < per_test:
            it = example(rng, task, *lens(rng))
            if it["key"] not in used:
                used.add(it["key"])
                items.append(it)
        tests[name] = items
    if drop is not None:                                 # the causal test: same sizes, with / without the dropped fact
        for name, want in ((f"mul: needs dropped pair {drop[0]}x{drop[1]} (trained lengths)", True),
                           (f"mul: avoids dropped pair {drop[0]}x{drop[1]} (trained lengths)", False)):
            items = []
            while len(items) < per_test:
                it = example(rng, "mul", *train_spec[3][1](rng))
                if it["key"] not in used and needs_pair(it, drop) == want:
                    used.add(it["key"])
                    items.append(it)
            tests[name] = items
    train = []
    while len(train) < n:
        task, lens = train_spec[len(train) % len(train_spec)]
        it = example(rng, task, *lens(rng))
        if it["key"] in used or needs_pair(it, drop):
            continue                                     # never a test question; repeats allowed (only 7^4
                                                         # numbers have 1-4 digits, fewer than the to/from share)
        train.append(it)
    rng.shuffle(train)
    return train, tests


def encode(tok, it: dict) -> dict:
    p = tok(it["prompt_text"], add_special_tokens=False)["input_ids"]
    t = tok(it["target"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return {"ids": p + t, "labels": [-100] * len(p) + t}


@torch.no_grad()
def evaluate(model, tok, items: list[dict], device, amp, bsz: int = 64) -> dict:
    model.eval()
    tok.padding_side = "left"
    preds = []
    for s in range(0, len(items), bsz):
        batch = items[s:s + bsz]
        enc = tok([it["prompt_text"] for it in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        max_new = max(len(tok(it["target"], add_special_tokens=False)["input_ids"]) for it in batch) + 6
        with amp():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        preds += [tok.decode(r, skip_special_tokens=True).split("\n")[0].strip()
                  for r in out[:, enc["input_ids"].shape[1]:]]
    tok.padding_side = "right"
    model.train()
    ok = [p == it["target"] for p, it in zip(preds, items)]
    # digit accuracy: share of answer digits right, aligned from the least significant end (where carrying starts)
    dig = [sum(a == b for a, b in zip(p[::-1], it["target"][::-1])) / max(len(it["target"]), len(p), 1)
           for p, it in zip(preds, items)]
    return {"n": len(items), "exact": sum(ok) / len(items), "digit_accuracy": sum(dig) / len(items),
            "samples": [{"q": it["prompt_text"].split("\n")[0], "gold": it["target"], "pred": p}
                        for it, p in list(zip(items, preds))[:6]]}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["meaning", "shuffled", "mean", "digits_only", "ops_only"], default="meaning",
                    help="digits_only: digit glyphs at their meaning, operators at the mean; ops_only: the reverse")
    ap.add_argument("--add-max", type=int, default=3, help="training: add up to N+N digits (tests unchanged)")
    ap.add_argument("--mul-max", default="2x2", help="training: mul up to AxB digits, e.g. 3x2 (tests unchanged)")
    ap.add_argument("--drop-pair", default="", help="data removal, e.g. 5,6: no training multiplication whose operands "
                                                    "contain digits 5 and 6 (one each); adds needs/avoids test sets")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--per-test", type=int, default=300)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0, help="also evaluate every N steps (learning curves)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/maths.json"))
    a = ap.parse_args(argv)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prec = pick_precision("auto")
    amp = (lambda: torch.autocast("cuda", dtype=DTYPES[prec])) if (device == "cuda" and prec != "fp32") \
        else (lambda: torch.autocast("cpu", enabled=False))
    ma, mb = (int(x) for x in a.mul_max.split("x"))
    spec, extra = None, None
    if (a.add_max, ma, mb) != (3, 2, 2):                 # longer training lengths: 4+4 / 3x2 become trained sizes,
        spec = [TRAIN_SPEC[0], TRAIN_SPEC[1],              # so add tests further out
                ("add", lambda r: (r.randint(1, a.add_max), r.randint(1, a.add_max))),
                ("mul", lambda r: (r.randint(1, ma), r.randint(1, mb)))]
        extra = {f"add: {a.add_max + 2}+{a.add_max + 2}": ("add", lambda r: (a.add_max + 2, a.add_max + 2)),
                 f"mul: {ma + 1}x{mb}": ("mul", lambda r: (ma + 1, mb)),
                 f"mul: {ma + 1}x{mb + 1}": ("mul", lambda r: (ma + 1, mb + 1))}
    drop = tuple(int(x) for x in a.drop_pair.split(",")) if a.drop_pair else None
    train, tests = build(a.n, a.per_test, a.seed, spec, extra, drop)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    n_old = len(tok)
    added = DIGITS + list(OPS.values())
    tok.add_tokens(added)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=DTYPES[prec] if device == "cuda" else torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model,
                                                     torch_dtype=DTYPES[prec] if device == "cuda" else torch.float32)
    model.to(device)
    model.config.use_cache = False
    if a.init == "mean":
        delta = add_glyph_tokens(model, n_old, len(added), a.seed, "mean")
        start = ["(mean)"] * len(added)
    else:
        start = [MEANING[g] for g in added]
        if a.init == "shuffled":                         # every digit starts as a WRONG digit, operators swapped
            rng = random.Random(a.seed + 1)
            while True:
                perm = list(range(BASE))
                rng.shuffle(perm)
                if all(i != j for i, j in enumerate(perm)):
                    break
            start = [str(perm[i]) for i in range(BASE)] + [" times", " minus", " plus"]
        elif a.init == "digits_only":                    # "" = no donor = neutral (mean) start
            start = start[:BASE] + [""] * len(OPS)
        elif a.init == "ops_only":
            start = [""] * BASE + start[BASE:]
        donors = [tok(s, add_special_tokens=False)["input_ids"] for s in start]
        delta = add_glyph_tokens(model, n_old, len(added), a.seed, "latin", donors)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    delta.requires_grad_(True)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    examples = [encode(tok, it) for it in train]
    steps = a.max_steps or math.ceil(len(examples) / a.batch * a.epochs)
    warm = max(1, int(0.03 * steps))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) *
                                              max(0.0, (steps - s) / max(1, steps - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=(prec == "fp16"))
    result = {"args": {k: str(v) for k, v in vars(a).items()}, "start_meaning": dict(zip(added, start)),
              "n_train": len(train), "steps": steps, "curve": []}
    log(f"maths init={a.init}: {len(train)} examples, {steps} steps, precision {prec}")
    model.train()
    step, t0, rng, hist = 0, time.time(), random.Random(a.seed + 2), []
    curve_sets = {k: v[:100] for k, v in tests.items()}
    while step < steps:
        for batch in batches_by_length(examples, a.batch, rng):
            b = collate(batch, tok.pad_token_id, device)
            with amp():
                loss = model(**b).loss
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            hist.append(loss.item())
            if step % 100 == 0:
                log(f"step {step}/{steps} loss {sum(hist[-100:]) / len(hist[-100:]):.4f} "
                    f"{(time.time() - t0) / 60:.1f} min")
            if a.eval_every and step % a.eval_every == 0 and step < steps:
                pt = {"step": step, **{k: evaluate(model, tok, v, device, amp)["exact"] for k, v in curve_sets.items()}}
                result["curve"].append(pt)
                log(f"curve {pt}")
            if step >= steps:
                break
    result["train_minutes"] = (time.time() - t0) / 60
    result["final_loss"] = sum(hist[-100:]) / max(1, len(hist[-100:]))
    result["tests"] = {}
    for name, items in tests.items():
        r = evaluate(model, tok, items, device, amp)
        result["tests"][name] = r
        log(f"{name:36} exact {r['exact']:.3f} digits {r['digit_accuracy']:.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {a.out}")
    return result


if __name__ == "__main__":
    main()
