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
import os
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
# --scratch only: three more glyphs for the written-out working (equals, next row, answer). Without --scratch they are
# never added, so the vocabulary and the data stay exactly as before.
SCR = {"=": chr(0xF0113), ";": chr(0xF0114), "ans": chr(0xF0115)}
MEANING.update({SCR["="]: " equals", SCR[";"]: " then", SCR["ans"]: " answer"})
GLYPH_DIGIT = {g: i for i, g in enumerate(DIGITS)}


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


def pairs_of(drop) -> list[tuple[int, int]]:
    """--drop-pair: one pair (5,6) or several (6,4;6,5;6,6, 2026-10-29) -> list of pairs."""
    if drop is None:
        return []
    return [tuple(drop)] if isinstance(drop[0], int) else [tuple(x) for x in drop]


def needs_ordered(it: dict, pr: tuple[int, int] | None) -> bool:
    """--drop-ordered 6,5: the working has a step pr[0] x pr[1] in that order (top digit pr[0], bottom digit pr[1])."""
    if pr is None or it["key"][0] != "mul":
        return False
    return pr[0] in digits7(it["key"][1]) and pr[1] in digits7(it["key"][2])


def needs_any(it: dict, drop) -> bool:
    return any(needs_pair(it, pr) for pr in pairs_of(drop))


def build(n: int, per_test: int, seed: int, train_spec=None, extra_tests=None,
          drop=None, ordered=None, skip_small: bool = False) -> tuple[list[dict], dict[str, list[dict]]]:
    """train_spec / extra_tests: longer training lengths (--add-max, --mul-max); drop: data removal (--drop-pair),
    one pair or a list. With the defaults (and with one pair) the data is exactly as before."""
    train_spec = train_spec or TRAIN_SPEC
    rng = random.Random(seed)
    tests, used = {}, set()
    for name, (task, lens) in {**TEST_SPEC, **(extra_tests or {})}.items():   # tests first: never trained
        items = []
        while len(items) < per_test:
            it = example(rng, task, *lens(rng))
            if skip_small and task in ("to", "from") and it["key"][1] < BASE * BASE:   # --tests-skip-small (01 Nov)
                continue
            if it["key"] not in used:
                used.add(it["key"])
                items.append(it)
        tests[name] = items
    if drop is not None:                                 # the causal test: same sizes, with / without the dropped fact
        prs = pairs_of(drop)
        sets = [(f"mul: needs dropped pair {a}x{b} (trained lengths)", (lambda it, pr=(a, b): needs_pair(it, pr)))
                for a, b in prs]
        sets.append((f"mul: avoids dropped pair {prs[0][0]}x{prs[0][1]} (trained lengths)" if len(prs) == 1 else
                     "mul: avoids every dropped pair (trained lengths)", (lambda it: not needs_any(it, drop))))
        for name, want in sets:
            items, tries = [], 0
            while len(items) < per_test and tries < 200 * per_test:   # few problems may exist: 2x2 has only 169
                tries += 1                                            # at these sizes; take them all, never loop forever
                it = example(rng, "mul", *train_spec[3][1](rng))
                if it["key"] not in used and want(it):
                    used.add(it["key"])
                    items.append(it)
            tests[name] = items
    if ordered is not None:                              # one order of one fact only (6x5 gone, 5x6 kept)
        a0, b0 = ordered
        # few problems use one order at these sizes (~150 for 5x6 at <= 2x2): the "other order" test takes at most
        # 40 of them, so the kept order stays in training
        for name, want, cap in ((f"mul: needs ordered {a0}x{b0} (trained lengths)", True, per_test),
                                (f"mul: needs the other order {b0}x{a0} only (trained lengths)", "rev", 40),
                                (f"mul: avoids both orders of {a0}x{b0} (trained lengths)", False, per_test)):
            items, tries = [], 0
            while len(items) < cap and tries < 200 * per_test:
                tries += 1
                it = example(rng, "mul", *train_spec[3][1](rng))
                ok = (needs_ordered(it, ordered) and not needs_ordered(it, (b0, a0))) if want is True else \
                     (needs_ordered(it, (b0, a0)) and not needs_ordered(it, ordered)) if want == "rev" else \
                     not (needs_ordered(it, ordered) or needs_ordered(it, (b0, a0)))
                if it["key"] not in used and ok:
                    used.add(it["key"])
                    items.append(it)
            tests[name] = items
    train = []
    while len(train) < n:
        task, lens = train_spec[len(train) % len(train_spec)]
        it = example(rng, task, *lens(rng))
        if it["key"] in used or needs_any(it, drop) or needs_ordered(it, ordered):
            continue                                     # never a test question; repeats allowed (only 7^4
                                                         # numbers have 1-4 digits, fewer than the to/from share)
        train.append(it)
    rng.shuffle(train)
    return train, tests


def lsd(n: int) -> list[int]:
    """Base-7 digits, least significant first."""
    out = []
    while True:
        n, d = divmod(n, BASE)
        out.append(d)
        if n == 0:
            return out


def working(a: int, b: int) -> dict:
    """Long multiplication a x b in base 7, as done by hand: one row per digit of b (least significant first); each
    step = digit of a x digit of b + carry; the row's partial product shifted by the row; then the partials added."""
    rows = []
    for j, bj in enumerate(lsd(b)):
        carry, steps = 0, []
        for ai in lsd(a):
            v = ai * bj + carry
            steps.append((ai, bj, carry, v))
            carry = v // BASE
        rows.append({"steps": steps, "partial": a * bj * BASE ** j})
    return {"rows": rows, "total": a * b}


def built_fact(ai: int, bj: int) -> list[int]:
    """--scratch-build: the single-digit fact ai x bj built from a smaller one, ai x (bj-2) + ai + ai (values in
    order); bj < 2 is written directly. Never uses 5x6/6x5 to build another fact (bj-2 <= 4)."""
    if bj < 2:
        return [ai * bj]
    p = ai * (bj - 2)
    return [p, p + ai, p + 2 * ai]


MIX = {"on": False, "avoid": set(), "add": False, "rule": ""}   # --scratch-build-mix/-add/-RULE; avoid = dropped


def add_chain(ai: int, bj: int, seed: int) -> str:
    """--scratch-build-add: a x b as pure repeated addition, a + a + ... (b times) or b + b + ... (a times), chosen
    at random per step (fixed seed); every running sum is written: 6x5 = 5+5=13+5=21+5=26+5=34+5=42 (base 7)."""
    x, k = random.Random(seed).choice([(ai, bj), (bj, ai)])
    if k == 0:
        return alien(0)
    out = DIGITS[x]
    for i in range(2, k + 1):
        out += f"{OPS['+']}{DIGITS[x]}{SCR['=']}{alien(i * x)}"
    return out


def mixed_chain(ai: int, bj: int, seed: int) -> str | None:
    """--scratch-build-mix: one of three rules, chosen at random per step (fixed seed), never using a dropped fact as
    the block: a x (b-1) + a, a x (b-2) + a + a, (a-1) x b + b. So one fact appears as a block in many workings."""
    X, P, E = OPS["x"], OPS["+"], SCR["="]
    rules = []
    if bj >= 2 and (ai, bj - 1) not in MIX["avoid"]:
        rules.append(("b-1", ai, bj - 1, [ai]))
    if bj >= 2 and (ai, bj - 2) not in MIX["avoid"]:
        rules.append(("b-2", ai, bj - 2, [ai, ai]))
    if ai >= 2 and (ai - 1, bj) not in MIX["avoid"]:
        rules.append(("a-1", ai - 1, bj, [bj]))
    if not rules:
        return None
    _, x, y, adds = random.Random(seed).choice(rules)
    val = x * y
    out = f"{DIGITS[x]}{X}{DIGITS[y]}{E}{alien(val)}"
    for d in adds:
        val += d
        out += f"{P}{DIGITS[d]}{E}{alien(val)}"
    return out


def trick_chain(ai: int, bj: int, seed: int) -> str:
    """--scratch-build-trick: base 7 has 6 = 7 - 1, so x times 6 = x0 - x (shift one place, subtract x), like x9 =
    x0 - x in decimal: 5x6 = 50-5 = 42. Facts without a 6 are written directly. No other fact is needed."""
    if 6 not in (ai, bj):
        return alien(ai * bj)
    x = bj if ai == 6 else ai
    return f"{alien(x * BASE)}{OPS['-']}{DIGITS[x]}{SCR['=']}{alien(6 * x)}"


def split_chain(ai: int, bj: int, seed: int) -> str:
    """--scratch-build-split: a x b = a x p + a x q with p + q = b (or the same on a), the split chosen at random
    among those whose two blocks are allowed: 6x6 = 6x2=15 + 6x4=33 = 51."""
    X, P, E = OPS["x"], OPS["+"], SCR["="]
    opts = [("b", q) for q in range(1, bj) if (ai, q) not in MIX["avoid"] and (ai, bj - q) not in MIX["avoid"]]
    opts += [("a", q) for q in range(1, ai) if (q, bj) not in MIX["avoid"] and (ai - q, bj) not in MIX["avoid"]]
    if not opts:
        return alien(ai * bj)
    side, q = random.Random(seed).choice(opts)
    (x1, y1), (x2, y2) = (((ai, q), (ai, bj - q)) if side == "b" else ((q, bj), (ai - q, bj)))
    return (f"{DIGITS[x1]}{X}{DIGITS[y1]}{E}{alien(x1 * y1)}{P}{DIGITS[x2]}{X}{DIGITS[y2]}{E}{alien(x2 * y2)}"
            f"{E}{alien(ai * bj)}")


def double_chain(ai: int, bj: int, seed: int) -> str:
    """--scratch-build-double: halve an even factor and double the result (6x6 = 6x3=24, 24+24 = 51), or for an odd
    factor take one off and add it back (6x5 = 6x4=33, +6 = 42); chosen at random among the allowed ones."""
    X, P, E = OPS["x"], OPS["+"], SCR["="]
    opts = []
    for side, f, g in (("b", ai, bj), ("a", bj, ai)):    # f stays, g is halved or reduced
        blk = (lambda h, side=side, f=f: (f, h) if side == "b" else (h, f))
        if g >= 2 and g % 2 == 0 and blk(g // 2) not in MIX["avoid"]:
            opts.append(("double", blk(g // 2), None))
        if g >= 3 and g % 2 == 1 and blk(g - 1) not in MIX["avoid"]:
            opts.append(("plus", blk(g - 1), f))
    if not opts:
        return alien(ai * bj)
    kind, (x, y), f = random.Random(seed).choice(opts)
    v = x * y
    head = f"{DIGITS[x]}{X}{DIGITS[y]}{E}{alien(v)}"
    if kind == "double":
        return f"{head}{P}{alien(v)}{E}{alien(2 * v)}"
    return f"{head}{P}{DIGITS[f]}{E}{alien(v + f)}"


def swap_chain(ai: int, bj: int, seed: int) -> str:
    """--scratch-build-swap: a fact with the bigger number first is rewritten with the smaller first and then
    recalled (6x4 = 4x6 = 33); smaller-first facts are written directly. With --drop-ordered 6,5 the model never sees
    6x5 but sees 5x6: does it answer 6x5 by swapping?"""
    if ai > bj:
        return f"{DIGITS[bj]}{OPS['x']}{DIGITS[ai]}{SCR['=']}{alien(ai * bj)}"
    return alien(ai * bj)


RULES = {"trick": trick_chain, "split": split_chain, "double": double_chain, "swap": swap_chain}


ROLE = {"heads": set()}   # --role-dose: in the chosen workings these facts use the a x (b-1) + a rule instead


def step_text(ai: int, bj: int, c: int, v: int, build: bool, seed: int = 0) -> str:
    """One step: 'a x b + c = v'; built: 'a x b = a x (b-2) = P + a = Q + a = R + c = v' (all in glyphs)."""
    X, P, E = OPS["x"], OPS["+"], SCR["="]
    if not build:
        return f"{DIGITS[ai]}{X}{DIGITS[bj]}{P}{DIGITS[c]}{E}{alien(v)}"
    if MIX["add"]:
        return f"{DIGITS[ai]}{X}{DIGITS[bj]}{E}{add_chain(ai, bj, seed)}{P}{DIGITS[c]}{E}{alien(v)}"
    if MIX.get("rule"):
        return f"{DIGITS[ai]}{X}{DIGITS[bj]}{E}{RULES[MIX['rule']](ai, bj, seed)}{P}{DIGITS[c]}{E}{alien(v)}"
    if MIX["on"]:
        chain = mixed_chain(ai, bj, seed)
        if chain is None:                                  # small facts (b < 2 and a < 2): written directly
            chain = alien(ai * bj)
        return f"{DIGITS[ai]}{X}{DIGITS[bj]}{E}{chain}{P}{DIGITS[c]}{E}{alien(v)}"
    if (ai, bj) in ROLE["heads"]:                        # --role-dose: 5x5 = 5x4=26 + 5 = 34 (5x4 as a block)
        chain = f"{DIGITS[ai]}{X}{DIGITS[bj - 1]}{E}{alien(ai * (bj - 1))}{P}{DIGITS[ai]}{E}{alien(ai * bj)}"
        return f"{DIGITS[ai]}{X}{DIGITS[bj]}{E}{chain}{P}{DIGITS[c]}{E}{alien(v)}"
    vals = built_fact(ai, bj)
    if len(vals) == 1:
        chain = alien(vals[0])
    else:
        chain = (f"{DIGITS[ai]}{X}{DIGITS[bj - 2]}{E}{alien(vals[0])}{P}{DIGITS[ai]}{E}{alien(vals[1])}"
                 f"{P}{DIGITS[ai]}{E}{alien(vals[2])}")
    return f"{DIGITS[ai]}{X}{DIGITS[bj]}{E}{chain}{P}{DIGITS[c]}{E}{alien(v)}"


def scratch_text(a: int, b: int, build: bool = False) -> str:
    """The working in glyphs, e.g. (readable) 5x6+0=42 2x6+4=22 =222 ; 5x3+0=21 2x3+2=11 =1110 ; 222+1110=1332 ans 1332.
    Steps are separated by spaces, rows by the ';' glyph, and the answer follows the 'ans' glyph.
    build (--scratch-build, 2026-10-29): every single-digit fact is built in the step (step_text), not recalled."""
    w = working(a, b)
    rows = [" ".join(step_text(ai, bj, c, v, build, seed=a * 7919 + b * 104729 + j * 31 + k)
                     for k, (ai, bj, c, v) in enumerate(r["steps"])) + f" {SCR['=']}{alien(r['partial'])}"
            for j, r in enumerate(w["rows"])]
    total = OPS["+"].join(alien(r["partial"]) for r in w["rows"]) + SCR["="] + alien(w["total"])
    return SCR[";"].join(rows + [total]) + SCR["ans"] + alien(w["total"])


def scratchify(items: list[dict], build: bool = False) -> list[dict]:
    """--scratch: every multiplication target becomes working + answer; the answer alone is kept for scoring."""
    out = []
    for it in items:
        if it["key"][0] == "mul":
            it = {**it, "answer": it["target"], "target": scratch_text(it["key"][1], it["key"][2], build)}
        out.append(it)
    return out


def readable(s: str) -> str:
    """Glyphs -> plain characters, for logs and checks."""
    m = {**{g: str(i) for i, g in enumerate(DIGITS)}, OPS["+"]: "+", OPS["x"]: "x", OPS["-"]: "-",
         SCR["="]: "=", SCR[";"]: " ; ", SCR["ans"]: " ans "}
    return "".join(m.get(ch, ch) for ch in s)


def _num(s: str) -> int | None:
    if not s or any(ch not in GLYPH_DIGIT for ch in s):
        return None
    v = 0
    for ch in s:
        v = v * BASE + GLYPH_DIGIT[ch]
    return v


def _step(tokn: str):
    """'d x d + d = v' in glyphs -> (a, b, carry, value), or None if it does not have that shape. Built steps
    ('d x d = ... = R + c = v') are read by their first part (a x b) and their last two parts (+ carry, value)."""
    if tokn.count(SCR["="]) > 1:
        parts = tokn.split(SCR["="])
        head, last2 = parts[0], parts[-2]
        if len(head) != 3 or head[1] != OPS["x"] or OPS["+"] not in last2:
            return None
        vals = (_num(head[0]), _num(head[2]), _num(last2.rsplit(OPS["+"], 1)[1]), _num(parts[-1]))
        return None if None in vals else vals
    if len(tokn) < 7 or tokn[1] != OPS["x"] or tokn[3] != OPS["+"] or tokn[5] != SCR["="]:
        return None
    parts = (_num(tokn[0]), _num(tokn[2]), _num(tokn[4]), _num(tokn[6:]))
    return None if None in parts else parts


def score_working(pred: str, a: int, b: int, drop: tuple[int, int] | None) -> dict:
    """Step-by-step check of a predicted working against the gold one. first_wrong: the first step type that is wrong
    (in writing order): wrong step count (a digit skipped or added in a row), digit product, carry, partial product,
    final addition, answer copy, or none. Positional comparison is only meaningful when a row has the right number of
    steps, so each written step is also checked on its own arithmetic (a x b + carry = value, base 7): `arith_ok`.
    Found 2026-10-25: a scratchpad model trained on <=2-digit operands writes exactly 2 steps per row for 3-digit
    numbers (every written step correct, one digit silently skipped)."""
    gold = working(a, b)
    dropped = {pr for x, y in pairs_of(drop) for pr in ((x, y), (y, x))}
    work = pred.split(SCR["ans"])[0]
    parts = work.split(SCR[";"])
    rows_p, sum_p = (parts[:-1], parts[-1]) if len(parts) > 1 else (parts, "")
    first, steps, written, counts = None, [], [], []
    for j, g in enumerate(gold["rows"]):
        toks = rows_p[j].strip().split(" ") if j < len(rows_p) else []
        partial = _num(toks[-1][1:]) if toks and toks[-1].startswith(SCR["="]) else None
        ps = [_step(t) for t in (toks[:-1] if partial is not None else toks)]
        counts.append((len(g["steps"]), len(ps)))
        for x in ps:                                         # each written step, judged on its own arithmetic
            if x:
                written.append({"uses_pair": bool(drop) and (x[0], x[1]) in dropped,
                                "arith_ok": x[0] * x[1] + x[2] == x[3]})
        if len(ps) != len(g["steps"]) and first is None:
            first = "wrong step count"
        for k, gs in enumerate(g["steps"]):
            p = ps[k] if k < len(ps) else None
            ok = p == gs
            steps.append({"uses_pair": bool(drop) and (gs[0], gs[1]) in dropped, "correct": ok})
            if not ok and first is None:
                first = "carry" if p and p[:2] == gs[:2] and p[2] != gs[2] else "digit product"
        if partial != g["partial"] and first is None:
            first = "partial product"
    total = _num(sum_p.split(SCR["="])[-1]) if SCR["="] in sum_p else None
    if total != gold["total"] and first is None:
        first = "final addition"
    ans = pred.split(SCR["ans"])[-1].strip() if SCR["ans"] in pred else ""
    if _num(ans) != gold["total"] and first is None:
        first = "answer copy"
    return {"first_wrong": first or "none", "all_digit_steps_right": all(s["correct"] for s in steps), "steps": steps,
            "written": written, "step_counts": counts}


def encode(tok, it: dict) -> dict:
    p = tok(it["prompt_text"], add_special_tokens=False)["input_ids"]
    t = tok(it["target"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return {"ids": p + t, "labels": [-100] * len(p) + t}


def twin(key: tuple) -> tuple:
    """The same fact the other way round: b op a for a op b; 'from n' for 'to n' and back."""
    if key[0] in ("add", "mul"):
        return (key[0], key[2], key[1])
    return ("from" if key[0] == "to" else "to", key[1])


@torch.no_grad()                                        # was on twin() by mistake until 2026-10-26
def evaluate(model, tok, items: list[dict], device, amp, bsz: int = 64, drop=None, train_keys=None) -> dict:
    was_training = model.training                       # restored at the end (was: always back to train mode)
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
    model.train(was_training)
    full = preds
    if any("answer" in it for it in items):              # --scratch: score the final answer only (after the 'ans' glyph)
        preds = [(p.split(SCR["ans"])[-1].strip() if SCR["ans"] in p else "") if "answer" in it else p
                 for p, it in zip(full, items)]
    gold = [it.get("answer", it["target"]) for it in items]
    ok = [p == g for p, g in zip(preds, gold)]
    # digit accuracy: share of answer digits right, aligned from the least significant end (where carrying starts)
    dig = [sum(a == b for a, b in zip(p[::-1], g[::-1])) / max(len(g), len(p), 1) for p, g in zip(preds, gold)]
    res = {"n": len(items), "exact": sum(ok) / len(items), "digit_accuracy": sum(dig) / len(items),
           "samples": [{"q": it["prompt_text"].split("\n")[0], "gold": it["target"], "pred": p}
                       for it, p in list(zip(items, full))[:6]]}
    if train_keys is not None:
        # Added 2026-10-25: test sets exclude only the exact problem, so the swapped twin (6x5 for 5x6) may be in
        # training; exact accuracy split by that separates computing from recalling the reversed fact.
        tw = [twin(it["key"]) in train_keys and twin(it["key"]) != it["key"] for it in items]
        for name, want in (("twin_in_training", True), ("no_twin_in_training", False)):
            xs = [o for o, t in zip(ok, tw) if t == want]
            res[name] = {"n": len(xs), "exact": sum(xs) / len(xs) if xs else None}
        res["keys"] = [list(it["key"]) for it in items]
        res["preds_all"] = full
    mul = [(p, it) for p, it in zip(full, items) if "answer" in it]
    if mul:                                              # step-by-step check of the written working
        sc = [score_working(p, it["key"][1], it["key"][2], drop) for p, it in mul]
        firsts: dict[str, int] = {}
        for s in sc:
            firsts[s["first_wrong"]] = firsts.get(s["first_wrong"], 0) + 1
        st = [x for s in sc for x in s["steps"]]

        def acc(xs):
            return {"n": len(xs), "correct": sum(x["correct"] for x in xs),
                    "accuracy": sum(x["correct"] for x in xs) / len(xs) if xs else None}
        wr = [x for s in sc for x in s["written"]]
        right_count = sum(all(g == p for g, p in s["step_counts"]) for s in sc)
        res["scratch_steps"] = {"all_digit_steps_right": sum(s["all_digit_steps_right"] for s in sc) / len(sc),
                                "digit_step_accuracy": acc(st)["accuracy"], "first_wrong_step": firsts,
                                "right_step_count": right_count / len(sc),
                                "written_steps_arith_ok": (sum(x["arith_ok"] for x in wr) / len(wr)) if wr else None}
        if drop:
            res["scratch_steps"]["steps_using_dropped_pair"] = acc([x for x in st if x["uses_pair"]])
            res["scratch_steps"]["steps_not_using_dropped_pair"] = acc([x for x in st if not x["uses_pair"]])
            for name, keep in (("written_using_dropped_pair", True), ("written_not_using_dropped_pair", False)):
                xs = [x for x in wr if x["uses_pair"] == keep]
                res["scratch_steps"][name] = {"n": len(xs), "arith_ok": (sum(x["arith_ok"] for x in xs) / len(xs))
                                              if xs else None}
        res["preds"] = full                              # every prediction, so the working can be re-scored later
    return res


@torch.no_grad()
def vote_eval(model, tok, items: list[dict], device, amp, k: int, seed: int, bsz: int = 16) -> dict:
    """--vote: k sampled workings per problem (temperature 0.7, top-k 50), the final answer by majority (ties: the
    earliest sample); also the mean single-sample accuracy and the answers, for the laptop."""
    torch.manual_seed(seed + 99)
    model.eval()
    tok.padding_side = "left"
    answers = []
    for s in range(0, len(items), bsz):
        batch = items[s:s + bsz]
        enc = tok([it["prompt_text"] for it in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        max_new = max(len(tok(it["target"], add_special_tokens=False)["input_ids"]) for it in batch) + 6
        with amp():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_k=50,
                                 num_return_sequences=k, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        dec = [tok.decode(r, skip_special_tokens=True).split("\n")[0].strip()
               for r in out[:, enc["input_ids"].shape[1]:]]
        for i in range(len(batch)):
            answers.append([(p.split(SCR["ans"])[-1].strip() if SCR["ans"] in p else "") for p in dec[i * k:(i + 1) * k]])
    tok.padding_side = "right"
    gold = [it.get("answer", it["target"]) for it in items]
    major = [max(a, key=lambda x: (a.count(x), -a.index(x))) for a in answers]
    return {"k": k, "exact_majority": sum(m == g for m, g in zip(major, gold)) / len(items),
            "exact_single_mean": sum(sum(x == g for x in a) / k for a, g in zip(answers, gold)) / len(items),
            "answers": answers}


@torch.no_grad()
def bridge_eval(model, tok, device, amp, train_keys: set, pairs: set, bsz: int = 16) -> dict:
    """--bridge-tests (30 Oct): does the model have the two pieces that could replace a dropped base-7 fact?
      A  the decimal fact, asked in ordinary digits (3 decimal examples first; scored: the first number written)
      B  the conversion of that product to the alien notation, in the trained 'to' format (Number: 30 / Alien:)
    for every single-digit pair a x b (0..6), marked dropped or not, and whether the product's conversion was a
    training item. C (combining them) is the run's own 'needs dropped pair' test."""
    shots = "3 x 4 = 12\n2 x 8 = 16\n7 x 9 = 63\n"
    facts = [(a_, b_) for a_ in range(BASE) for b_ in range(BASE)]
    model.eval()
    tok.padding_side = "left"

    def gen(prompts, n):
        out = []
        for s0 in range(0, len(prompts), bsz):
            enc = tok(prompts[s0:s0 + bsz], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with amp():
                g = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id,
                                   eos_token_id=tok.eos_token_id)
            out += [tok.decode(r, skip_special_tokens=True) for r in g[:, enc["input_ids"].shape[1]:]]
        return out
    dec = gen([f"{shots}{a_} x {b_} =" for a_, b_ in facts], 6)
    conv = gen([f"Number: {a_ * b_}\nAlien:\n" for a_, b_ in facts], 8)
    tok.padding_side = "right"
    rows = []
    for (a_, b_), d, c in zip(facts, dec, conv):
        num = __import__("re").match(r"\s*(\d+)", d)
        rows.append({"fact": f"{a_}x{b_}", "dropped": (a_, b_) in pairs,
                     "decimal_right": bool(num) and int(num.group(1)) == a_ * b_,
                     "decimal_written": d[:16], "conversion_written": readable(c.split("\n")[0][:12]),   # 01 Nov
                     "conversion_right": c.split("\n")[0].strip() == alien(a_ * b_),
                     "conversion_trained": ("to", a_ * b_) in train_keys})

    def share(xs, k):
        return sum(x[k] for x in xs) / len(xs) if xs else None
    drop_rows = [x for x in rows if x["dropped"]]
    keep_rows = [x for x in rows if not x["dropped"]]
    return {"dropped": {"n": len(drop_rows), "decimal_right": share(drop_rows, "decimal_right"),
                        "conversion_right": share(drop_rows, "conversion_right")},
            "kept": {"n": len(keep_rows), "decimal_right": share(keep_rows, "decimal_right"),
                     "conversion_right": share(keep_rows, "conversion_right")},
            "rows": rows}


@torch.no_grad()
def forced_route_eval(model, tok, items: list[dict], device, amp, pairs: set, bsz: int = 16) -> dict:
    """--force-routes (review, 30 Oct): the route is chosen by us, not the model. For every problem that needs a
    dropped fact, the gold working is given up to the first step with that fact, then the step's head and a FORCED
    first block (rule b-1: a x (b-1), b-2: a x (b-2), a-1: (a-1) x b); the model writes the rest of the step. Scored:
    the step's final value right (a x b + carry), and the block's own value right. Every allowed route is forced on
    every problem, so route and confidence are no longer confounded."""
    X, E = OPS["x"], SCR["="]
    jobs = []                                            # (route, block, prefix, (a, b))
    for it in items:
        if "answer" not in it:
            continue
        toks = it["target"].split(SCR["ans"])[0].replace(SCR[";"], " ").split(" ")
        hit = next((t for t in toks if (st := _step(t)) and (st[0], st[1]) in pairs), None)
        if hit is None:
            continue
        a_, b_ = _step(hit)[:2]
        head = it["prompt_text"] + it["target"][:it["target"].index(hit)] + f"{DIGITS[a_]}{X}{DIGITS[b_]}{E}"
        for route, (x, y) in (("b-1", (a_, b_ - 1)), ("b-2", (a_, b_ - 2)), ("a-1", (a_ - 1, b_))):
            if min(x, y) >= 0 and (x, y) not in pairs:
                jobs.append((route, (x, y), head + f"{DIGITS[x]}{X}{DIGITS[y]}{E}", (a_, b_)))
    was_training = model.training
    model.eval()
    tok.padding_side = "left"
    res = {}
    for s0 in range(0, len(jobs), bsz):
        batch = jobs[s0:s0 + bsz]
        enc = tok([j[2] for j in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with amp():
            out = model.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        for (route, (x, y), prefix, (a_, b_)), r in zip(batch, out[:, enc["input_ids"].shape[1]:]):
            rest = tok.decode(r, skip_special_tokens=True).replace(SCR[";"], " ").split(" ")[0].split("\n")[0]
            step = _step(f"{DIGITS[a_]}{X}{DIGITS[b_]}{E}{DIGITS[x]}{X}{DIGITS[y]}{E}{rest}")
            blk = _num(rest.split(OPS["+"])[0]) if rest else None
            key = f"{a_}x{b_} via {x}x{y} ({route})"
            d = res.setdefault(key, {"n": 0, "step_right": 0, "block_right": 0, "written": {}})
            d["n"] += 1
            w = readable(rest.split(OPS["+"])[0])[:24] if rest else "(nothing)"   # 31 Oct: what fills the block slot
            d["written"][w] = d["written"].get(w, 0) + 1
            d["step_right"] += bool(step and step[0] * step[1] + step[2] == step[3])
            d["block_right"] += blk == x * y
    tok.padding_side = "right"
    model.train(was_training)
    for d in res.values():
        d["step_right"] /= d["n"]
        d["block_right"] /= d["n"]
        d["written"] = dict(sorted(d["written"].items(), key=lambda kv: -kv[1])[:12])
    return res



def set_deterministic() -> None:
    """--deterministic (01 Nov): same seed + same GPU type + same library versions -> the same model. GPU kernels that
    add in a varying order are replaced by fixed-order ones (slower, ~10-30%); ops with no deterministic version only
    warn (warn_only) and are listed in the log, so a re-run can still differ if one of them is used."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before the first cuBLAS call
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # DT1 (01 Nov) found the one op that stayed random: memory-efficient attention's backward pass. Turn it off so
    # scaled_dot_product_attention uses the plain (math) kernel, which adds in a fixed order (flash needs sm80+).
    torch.backends.cuda.enable_mem_efficient_sdp(False)

def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["meaning", "shuffled", "mean", "digits_only", "ops_only"], default="meaning",
                    help="digits_only: digit glyphs at their meaning, operators at the mean; ops_only: the reverse")
    ap.add_argument("--shuffle-perm", default="",
                    help="with --init shuffled: the wrong digit each glyph starts at, e.g. 1,0,3,4,2,6,5 (default: "
                         "drawn from --seed, as before). Separates the permutation from the data seed (2026-10-26: "
                         "seed 0 drew a scrambled permutation and scored 0.08 on add 4+4, seed 1 a near-identity one, 0.60)")
    ap.add_argument("--add-max", type=int, default=3, help="training: add up to N+N digits (tests unchanged)")
    ap.add_argument("--mul-max", default="2x2", help="training: mul up to AxB digits, e.g. 3x2 (tests unchanged)")
    ap.add_argument("--drop-pair", default="", help="data removal, e.g. 5,6: no training multiplication whose operands "
                                                    "contain digits 5 and 6 (one each); adds needs/avoids test sets")
    ap.add_argument("--scratch", action="store_true",
                    help="multiplication answers are preceded by the written-out long multiplication (Faith and Fate "
                         "scratchpad); adds 3 glyphs (equals, next row, answer); scored on the final answer + per step")
    ap.add_argument("--scratch-build", action="store_true",
                    help="with --scratch: every single-digit fact in the working is BUILT from a smaller one "
                         "(a x b = a x (b-2) + a + a), not recalled: can the model build a fact it never saw?")
    ap.add_argument("--scratch-build-mix", action="store_true",
                    help="with --scratch-build: each step uses one of three building rules at random (a x (b-1) + a, "
                         "a x (b-2) + a + a, (a-1) x b + b), so a fact appears as a building block in many workings")
    ap.add_argument("--scratch-build-add", action="store_true",
                    help="with --scratch: every single-digit fact built by pure repeated addition, a+a+... (b times) "
                         "or b+b+... (a times) at random per step: can a counting chain be carried further than seen?")
    ap.add_argument("--role-dose", type=int, default=0,
                    help="with --scratch-build --drop-pair 5,6: in N training workings that have a 5x5 step, build 5x5 "
                         "as 5x4 + 5, so 5x4 is seen N+ times AS A BLOCK (it is otherwise only a block inside 5x6); "
                         "everything else unchanged. Role dose curve (2026-10-30)")
    ap.add_argument("--scratch-build-rule", choices=["", "trick", "split", "double", "swap"], default="",
                    help="with --scratch: build every fact by one rule: trick (x6 = x0 - x), split (a x (p+q)), "
                         "double (halve and double / one off), swap (bigger x smaller -> smaller x bigger)")
    ap.add_argument("--drop-ordered", default="",
                    help="e.g. 6,5: no training working has the step 6x5 in that order (5x6 is kept)")
    ap.add_argument("--tests-skip-small", action="store_true",
                    help="01 Nov: the to/from tests never take numbers below 49, so every number 0-48 can be a "
                         "training conversion (by default the tests, drawn first, use up all of them: the bridge "
                         "test's conversion part then asked only untrained numbers). Changes the data; default off")
    ap.add_argument("--bridge-tests", action="store_true",
                    help="with --drop-pair: also test the two pieces that could replace the dropped fact: the decimal "
                         "fact (5 x 6 = 30) and the conversion of its product to the notation (30 -> alien)")
    ap.add_argument("--force-routes", action="store_true",
                    help="with --scratch-build-mix and a drop: also score every building route forced (review, 30 Oct)")
    ap.add_argument("--vote", type=int, default=0,
                    help="also score the multiplication tests by majority vote over N sampled workings "
                         "(temperature 0.7): test-time reasoning without retraining")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--per-test", type=int, default=300)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="split each batch into this many parts (same updates, less GPU memory; 1 = as before)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0, help="also evaluate every N steps (learning curves)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true",
                    help="fixed-order GPU kernels: a same-seed re-run gives the same model (01 Nov; default off)")
    ap.add_argument("--out", type=Path, default=Path("results/maths.json"))
    a = ap.parse_args(argv)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if a.deterministic:
        set_deterministic()
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
    drop = None
    if a.drop_pair:                                      # "5,6" (one pair, as before) or "6,4;6,5;6,6"
        prs = [tuple(int(x) for x in q.split(",")) for q in a.drop_pair.split(";")]
        drop = prs[0] if len(prs) == 1 else prs
    ordered = tuple(int(x) for x in a.drop_ordered.split(",")) if a.drop_ordered else None
    train, tests = build(a.n, a.per_test, a.seed, spec, extra, drop, ordered, a.tests_skip_small)
    if a.scratch_build_rule:                             # one building rule; the mixed-rule path does the rest
        a.scratch_build_mix = True
        MIX["rule"] = a.scratch_build_rule
    if a.scratch_build_add:                              # pure addition: the mixed-rule path, with add chains
        a.scratch_build_mix = True
        MIX["add"] = True
    if a.scratch_build_mix:
        a.scratch_build = True
        MIX["on"] = True
        if a.drop_pair:
            MIX["avoid"] = {pr for q in a.drop_pair.split(";")
                            for d0, d1 in [tuple(int(x) for x in q.split(","))] for pr in ((d0, d1), (d1, d0))}
    if a.scratch_build:
        a.scratch = True
    if a.scratch:                                        # applied after building: same problems as without --scratch
        train = scratchify(train, a.scratch_build)
        tests = {k: scratchify(v, a.scratch_build) for k, v in tests.items()}
    if a.role_dose:                                      # the role dose: re-write N chosen workings only
        assert a.scratch_build and not a.scratch_build_mix and drop == (5, 6), "--role-dose: --scratch-build, 5,6"
        cand = [i for i, it in enumerate(train) if it["key"][0] == "mul" and
                any((ai, bj) == (5, 5) for r in working(it["key"][1], it["key"][2])["rows"] for ai, bj, _, _ in r["steps"])]
        pick = random.Random(a.seed + 404).sample(cand, min(a.role_dose, len(cand)))
        ROLE["heads"] = {(5, 5)}
        for i in pick:
            it = train[i]
            train[i] = {**it, "target": scratch_text(it["key"][1], it["key"][2], True)}
        ROLE["heads"] = set()
        blocks = sum(train[i]["target"].count(DIGITS[5] + OPS["x"] + DIGITS[5] + SCR["="] + DIGITS[5] + OPS["x"]
                                              + DIGITS[4]) for i in pick)
        log(f"role dose: {len(pick)} workings re-written ({len(cand)} have a 5x5 step), 5x4 as a block {blocks} times")
    if a.scratch_build_mix and drop:                     # text check: the dropped fact nowhere in a training working
        bad = {DIGITS[x] + OPS["x"] + DIGITS[y] for x, y in MIX["avoid"]}
        leaks = [it for it in train if it["key"][0] == "mul" and any(x in it["target"] for x in bad)]
        if leaks:
            raise ValueError(f"--scratch-build-mix: {len(leaks)} training workings contain the dropped fact")
    elif a.scratch_build and drop:                       # no training working may contain the dropped fact anywhere,
        pair = {pr for x, y in pairs_of(drop) for pr in ((x, y), (y, x))}   # also not as a block for another one
        leaks = [it for it in train if it["key"][0] == "mul" and any(
            (ai, bj) in pair or (bj >= 2 and (ai, bj - 2) in pair)
            for r in working(it["key"][1], it["key"][2])["rows"] for ai, bj, _, _ in r["steps"])]
        if leaks:
            raise ValueError(f"--scratch-build --drop-pair {a.drop_pair}: {len(leaks)} training workings use the "
                             f"dropped fact to build another (the build rule cannot hide this pair); not supported")
    if ordered and a.scratch:                            # the removed order nowhere in a training working
        bad1 = DIGITS[ordered[0]] + OPS["x"] + DIGITS[ordered[1]]
        leaks = [it for it in train if it["key"][0] == "mul" and bad1 in it["target"]]
        if leaks:
            raise ValueError(f"--drop-ordered: {len(leaks)} training workings contain {a.drop_ordered}")
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    n_old = len(tok)
    added = DIGITS + list(OPS.values()) + (list(SCR.values()) if a.scratch else [])
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
            if a.shuffle_perm:                           # given explicitly: must be a derangement of 0..BASE-1
                perm = [int(x) for x in a.shuffle_perm.split(",")]
                if sorted(perm) != list(range(BASE)) or any(i == j for i, j in enumerate(perm)):
                    raise ValueError(f"--shuffle-perm {a.shuffle_perm}: not a derangement of 0..{BASE - 1}")
            start = [str(perm[i]) for i in range(BASE)] + [" times", " minus", " plus"]
            if a.scratch:                                # the working glyphs get wrong meanings too (rotated)
                start += [" then", " answer", " equals"]
        elif a.init == "digits_only":                    # "" = no donor = neutral (mean) start
            start = start[:BASE] + [""] * (len(added) - BASE)
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
        micro, parts = 0, []
        for batch in batches_by_length(examples, a.batch // a.grad_accum, rng):
            b = collate(batch, tok.pad_token_id, device)
            with amp():
                loss = model(**b).loss
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                micro, parts = 0, []
                continue
            scaler.scale(loss / a.grad_accum).backward()
            micro += 1
            parts.append(loss.item())
            if micro < a.grad_accum:
                continue
            loss = torch.tensor(sum(parts) / len(parts))
            micro, parts = 0, []
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
        r = evaluate(model, tok, items, device, amp, drop=drop, train_keys={it["key"] for it in train})
        if a.force_routes and drop and name.startswith("mul: needs dropped pair"):
            prs = {pr for x, y in pairs_of(drop) for pr in ((x, y), (y, x))}
            r["forced_routes"] = forced_route_eval(model, tok, items, device, amp, prs)
            for k, v in r["forced_routes"].items():
                log(f"{name:36} forced {k}: n {v['n']}, step right {v['step_right']:.2f}, block right {v['block_right']:.2f}"
                    f"; written: {list(v['written'].items())[:4]}")
        if a.vote and name.startswith("mul:") and "trained lengths" in name:
            r["vote"] = vote_eval(model, tok, items, device, amp, a.vote, a.seed)
            log(f"{name:36} vote x{a.vote}: majority exact {r['vote']['exact_majority']:.3f}, "
                f"single sample {r['vote']['exact_single_mean']:.3f}")
        result["tests"][name] = r
        log(f"{name:36} exact {r['exact']:.3f} digits {r['digit_accuracy']:.3f}"
            + f" | twin trained {r['twin_in_training']['exact']} (n={r['twin_in_training']['n']}),"
              f" no twin {r['no_twin_in_training']['exact']} (n={r['no_twin_in_training']['n']})"
            + (f" | all digit steps right {r['scratch_steps']['all_digit_steps_right']:.3f}"
               if "scratch_steps" in r else ""))
    if a.bridge_tests and drop:
        prs = {pr for x, y in pairs_of(drop) for pr in ((x, y), (y, x))}
        result["bridge"] = bridge_eval(model, tok, device, amp, {it["key"] for it in train}, prs)
        for k in ("dropped", "kept"):
            b = result["bridge"][k]
            log(f"bridge {k} facts (n {b['n']}): decimal right {b['decimal_right']:.2f}, "
                f"conversion of the product right {b['conversion_right']:.2f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {a.out}")
    return result


if __name__ == "__main__":
    main()
