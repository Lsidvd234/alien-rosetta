"""Second batch of stress tests (7-12) for the saved hard-language glyph models. All local.

    python analysis/stress_tests2.py roles      # 7. who did what
    python analysis/stress_tests2.py format     # 8. input-format brittleness
    python analysis/stress_tests2.py forms      # 9. irregular verbs, sound changes at morpheme joins
    python analysis/stress_tests2.py unusual    # 10. two sentences at once, an unknown English word
    python analysis/stress_tests2.py output     # 11. free generation vs teacher-forced, reverse direction
    python analysis/stress_tests2.py confidence # 12. does the model know when it is wrong?

Writes results/stress/test<N>_*.md. Set HF_HOME to the cache holding Qwen3-0.6B-Base.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stress_tests import (MODELS, RUNS, Ctx, example, exact_rate, is_heldout_shape, passive,  # noqa: E402
                          run_models, synonym_english, write)
from conlang.dataset import heldout_pair  # noqa: E402
from conlang.frames import NP, PRONOUNS, TENSES, Frame, english  # noqa: E402
from conlang.script import render_sentence  # noqa: E402


def saved_models() -> dict:
    return {l: r for l, r in MODELS.items() if (RUNS / r / "model" / "glyph_tokens.pt").exists()}


def each_model():
    """Yield (label, model, tok) one at a time (frees GPU memory between models)."""
    from train.load import load_trained
    for label, run in saved_models().items():
        model, tok, _ = load_trained(RUNS / run / "model")
        yield label, model, tok
        del model
        import gc
        gc.collect()                 # release the memory-mapped weights now, not at some later collection
        torch.cuda.empty_cache()


def valid(ctx: Ctx, f: Frame) -> bool:
    """A new sentence: not in training, no held-out verb+object pair, not a held-out shape."""
    if english(f, ctx.vocab) in ctx.train_en or is_heldout_shape(f, ctx.vocab):
        return False
    return not (f.obj is not None and heldout_pair(ctx.seed, f.verb, f.obj.noun))


def random_frame(ctx: Ctx, rng: random.Random, transitive: bool | None = None) -> Frame:
    def np():
        return NP(rng.choice(ctx.nouns), rng.random() < 0.4, rng.choice(ctx.adjs) if rng.random() < 0.3 else None)
    subj = rng.choice(list(PRONOUNS)) if rng.random() < 0.3 else np()
    tr = (rng.random() < 0.7) if transitive is None else transitive
    verb, obj = (rng.choice(ctx.trans), np()) if tr else (rng.choice(ctx.intr), None)
    return Frame(subj, verb, obj, rng.choice(ctx.locs) if rng.random() < 0.3 else None,
                 rng.choice(TENSES), rng.random() < 0.3)


def table(title: str, intro: list[str], groups: dict[str, list[int]], items: list[dict], res: dict,
          paired: bool = False) -> list[str]:
    labels = list(res)
    head = " (normal -> changed)" if paired else ""
    lines = [f"# {title}", ""] + intro + ["", "| case | n | example | " + " | ".join(l + head for l in labels) + " |",
                                          "|---|---|---|" + "---|" * len(labels)]
    for name, idx in groups.items():
        if not idx:
            continue
        if paired:
            cells = [f"{exact_rate(res[l], idx):.1%} -> {exact_rate(res[l], [i + 1 for i in idx]):.1%}" for l in labels]
            ex = example(items, [i + 1 for i in idx])
        else:
            cells = [f"{exact_rate(res[l], idx):.1%}" for l in labels]
            ex = example(items, idx)
        lines.append(f"| {name} | {len(idx)} | {ex} | " + " | ".join(cells) + " |")
    return lines


# --- 7. who did what ---------------------------------------------------------------------

def test_roles(n: int = 300) -> None:
    ctx = Ctx()
    rng = random.Random(21)
    items, groups = [], {"subject and object of a training sentence swapped": [],
                         "verb that always had an object, used without one": [],
                         "same noun as subject and object": [],
                         "control: ordinary new sentences": []}

    def add(group: str, f: Frame) -> None:
        if valid(ctx, f):
            groups[group].append(len(items))
            items.append(ctx.item(f))

    train_tr = [it for it in ctx.train if it["frame"]["obj"] and isinstance(it["frame"]["subj"], dict)]
    while len(groups["subject and object of a training sentence swapped"]) < n:
        fr = rng.choice(train_tr)["frame"]
        s, o = fr["subj"], fr["obj"]
        add("subject and object of a training sentence swapped",
            Frame(NP(o["noun"], o["plural"], o["adj"]), fr["verb"], NP(s["noun"], s["plural"], s["adj"]),
                  fr["loc"], fr["tense"], fr["neg"]))
    while len(groups["verb that always had an object, used without one"]) < n:
        f = random_frame(ctx, rng, transitive=True)
        add("verb that always had an object, used without one", Frame(f.subj, f.verb, None, f.loc, f.tense, f.neg))
    while len(groups["same noun as subject and object"]) < n:
        f = random_frame(ctx, rng, transitive=True)
        if isinstance(f.subj, NP):
            add("same noun as subject and object",
                Frame(f.subj, f.verb, NP(f.subj.noun, rng.random() < 0.4, None), f.loc, f.tense, f.neg))
    while len(groups["control: ordinary new sentences"]) < n:
        add("control: ordinary new sentences", random_frame(ctx, rng))
    res = run_models(items)
    write("test7_roles", table("Test 7: who did what", [
        f"{n} new sentences per case, only words seen in training, none in the training set.",
        "Training never used a transitive verb without an object (the generator always gave it one)."],
        groups, items, res))


# --- 8. input-format brittleness -----------------------------------------------------------

def typo(text: str, rng: random.Random) -> str | None:
    words = text.split()
    cands = [i for i, w in enumerate(words) if len(w) >= 5 and w not in ("the", "will", "not", "does")]
    if not cands:
        return None
    i = rng.choice(cands)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def test_format(n: int = 250) -> None:
    ctx = Ctx()
    rng = random.Random(33)
    variants = {
        "double spaces between words": lambda en: {"english": en.replace(" ", "  ")},
        "trailing space": lambda en: {"english": en + " "},
        "leading space": lambda en: {"english": " " + en},
        "prompt label 'Input/Output' instead of 'English/Target'": lambda en: {"prompt_text": f"Input: {en}\nOutput:\n"},
        "no newline after 'Target:'": lambda en: {"prompt_text": f"English: {en}\nTarget: "},
        "typo (two letters swapped in a word)": lambda en: ({"english": t} if (t := typo(en, rng)) else None),
        "'the the' (doubled word)": lambda en: ({"english": en.replace("the ", "the the ", 1)} if "the " in en else None),
        "'Translate:' instruction before the prompt": lambda en: {"prompt_text": f"Translate:\nEnglish: {en}\nTarget:\n"},
    }
    items, groups = [], {k: [] for k in variants}
    while any(len(g) < n for g in groups.values()):
        f = random_frame(ctx, rng)
        if not valid(ctx, f):
            continue
        base = ctx.item(f)
        for name, make in variants.items():
            if len(groups[name]) >= n:
                continue
            ch = make(base["english"])
            if ch is None:
                continue
            groups[name].append(len(items))
            items.append(dict(base))
            items.append({**base, **ch, "english": ch.get("english", base["english"]) if "prompt_text" not in ch
                          else ch["prompt_text"].replace("\n", " / ")})
            break
    res = run_models(items)
    write("test8_format", table("Test 8: input-format brittleness", [
        f"{n} new sentences per case, each scored as normally written and changed; the correct translation is the same."],
        groups, items, res, paired=True))


# --- 9. hard word forms ----------------------------------------------------------------------

def test_forms(n: int = 300) -> None:
    ctx = Ctx()
    rng = random.Random(44)
    irregular = [v for v in ctx.lang.irregular_past if v in ctx.trans + ctx.intr]
    items, groups = [], {"past tense, irregular verb": [], "past tense, regular verb": [],
                         "no sound change at morpheme joins": [], "1 sound change": [], "2+ sound changes": []}
    while len(groups["past tense, irregular verb"]) < n and irregular:
        v = rng.choice(irregular)
        f = random_frame(ctx, rng, transitive=v in ctx.trans)
        f = Frame(f.subj, v, f.obj, f.loc, "past", f.neg)
        if valid(ctx, f):
            groups["past tense, irregular verb"].append(len(items))
            items.append(ctx.item(f))
    while len(groups["past tense, regular verb"]) < n:
        f = random_frame(ctx, rng)
        if f.verb in ctx.lang.irregular_past:
            continue
        f = Frame(f.subj, f.verb, f.obj, f.loc, "past", f.neg)
        if valid(ctx, f):
            groups["past tense, regular verb"].append(len(items))
            items.append(ctx.item(f))
    tries = 0
    while any(len(groups[k]) < n for k in ("no sound change at morpheme joins", "1 sound change", "2+ sound changes")) \
            and tries < 200000:
        tries += 1
        f = random_frame(ctx, rng)
        if not valid(ctx, f):
            continue
        k = ctx.lang.realize(f).sandhi_changes
        g = ("no sound change at morpheme joins", "1 sound change", "2+ sound changes")[min(k, 2)]
        if len(groups[g]) < n:
            groups[g].append(len(items))
            items.append(ctx.item(f))
    res = run_models(items)
    write("test9_word_forms", table("Test 9: hard word forms", [
        f"Up to {n} new sentences per case. Irregular verbs have their own past stem (rarer in training).",
        "Sound changes = places where joining a stem and an ending changes a sound (sandhi)."],
        groups, items, res))


# --- 10. unusual input ---------------------------------------------------------------------

NONCE = ["blicket", "dax", "wug", "toma", "fep", "zorb", "kiki", "lorp", "mipple", "snerg", "tufa", "vopple"]


@torch.no_grad()
def test_unusual(n: int = 150) -> None:
    from train.load import translate
    ctx = Ctx()
    rng = random.Random(55)
    two, nonce = [], []
    while len(two) < n:
        a, b = random_frame(ctx, rng), random_frame(ctx, rng)
        if valid(ctx, a) and valid(ctx, b):
            ia, ib = ctx.item(a), ctx.item(b)
            two.append({"english": f"{ia['english']}. {ib['english']}.", "t1": ia["target"], "t2": ib["target"]})
    while len(nonce) < n:
        f = random_frame(ctx, rng, transitive=True)
        if not valid(ctx, f):
            continue
        it = ctx.item(f)
        word = rng.choice(NONCE)
        en = english(f, ctx.vocab).split()
        noun = ctx.vocab.noun_info[f.obj.noun][0] if f.obj.plural else f.obj.noun
        i = len(en) - 1 - en[::-1].index(noun)
        en[i] = word + ("s" if f.obj.plural else "")
        gold = it["target"].split()
        obj_i = [k for k, g in enumerate(it["gloss"].split()) if g.split("-")[0] == f.obj.noun and "ACC" in g][0]
        nonce.append({"english": " ".join(en), "gold": gold, "obj_i": obj_i})
    lines = ["# Test 10: unusual input", "",
             f"{n} cases each, free generation (greedy).", "",
             "**Two sentences in one input** ('S1. S2.'): which did the output contain?", "",
             "| model | both, in order | only the first | other |", "|---|---|---|---|"]
    lines2 = ["", "**An unknown English word as the object** (e.g. 'the dog saw the blicket'):", "",
              "| model | rest of the sentence right | same number of words as expected |", "|---|---|---|"]
    for label, model, tok in each_model():
        out = translate(model, tok, [t["english"] for t in two], max_new_tokens=160)
        both = sum(o.strip() == f"{t['t1']} {t['t2']}" for o, t in zip(out, two))
        first = sum(o.strip() == t["t1"] for o, t in zip(out, two))
        lines.append(f"| {label} | {both / n:.1%} | {first / n:.1%} | {(n - both - first) / n:.1%} |")
        out = translate(model, tok, [t["english"] for t in nonce])
        rest = shape = 0
        for o, t in zip(out, nonce):
            p = o.split()
            if len(p) == len(t["gold"]):
                shape += 1
                rest += all(p[k] == t["gold"][k] for k in range(len(p)) if k != t["obj_i"])
        lines2.append(f"| {label} | {rest / n:.1%} | {shape / n:.1%} |")
    write("test10_unusual_input", lines + lines2)


# --- 11. output behaviour --------------------------------------------------------------------

@torch.no_grad()
def test_output(n: int = 300) -> None:
    from conlang.dataset import PRESETS, build
    from train.load import translate
    from train.run import encode, teacher_forced
    ctx = Ctx()
    data = build(PRESETS[ctx.cfg["preset"]], ctx.cfg["script"], n_train=50000, n_test=500, seed=ctx.seed)
    items = data["splits"]["test_iid"][:n]
    lines = ["# Test 11: output behaviour", "",
             f"{n} test_iid sentences. Teacher-forced exact = every next token right given the correct prefix;",
             "free generation = what the model actually writes (greedy).", "",
             "| model | teacher-forced exact | free generation exact | output too long | output too short | "
             "reverse direction (glyphs -> English) exact |", "|---|---|---|---|---|---|"]
    for label, model, tok in each_model():
        device = next(model.parameters()).device
        amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if device.type == "cuda" \
            else (lambda: torch.autocast("cpu", enabled=False))
        tf = teacher_forced(model, tok, [encode(tok, it) for it in items], 16, device, amp)
        out = translate(model, tok, [it["english"] for it in items])
        ex = sum(o == it["target"] for o, it in zip(out, items)) / n
        long_ = sum(len(o.split()) > len(it["target"].split()) for o, it in zip(out, items)) / n
        short = sum(len(o.split()) < len(it["target"].split()) for o, it in zip(out, items)) / n
        # reverse: never trained; prompt in the same style
        enc_prompts = [f"Target: {it['target']}\nEnglish:\n" for it in items]
        tok.padding_side = "left"
        enc = tok(enc_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            g = model.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id,
                               eos_token_id=tok.eos_token_id)
        tok.padding_side = "right"
        rev = [tok.decode(r, skip_special_tokens=True).split("\n")[0].strip() for r in g[:, enc["input_ids"].shape[1]:]]
        rex = sum(r == it["english"] for r, it in zip(rev, items)) / n
        lines.append(f"| {label} | {tf['exact_match']:.1%} | {ex:.1%} | {long_:.1%} | {short:.1%} | {rex:.1%} |")
        print(label, "reverse samples:", rev[:3], flush=True)
    write("test11_output", lines)


# --- 12. does the model know when it is wrong? ------------------------------------------------

@torch.no_grad()
def token_probs(model, tok, items: list[dict]) -> list[tuple[bool, float, float]]:
    """(exact, lowest target-token probability, mean log-probability) per item, teacher-forced."""
    from train.run import collate, encode
    device = next(model.parameters()).device
    ex = [encode(tok, it) for it in items]
    out = []
    for s in range(0, len(ex), 16):
        batch = ex[s:s + 16]
        b = collate(batch, tok.pad_token_id, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.float()
        lp = logits[:, :-1].log_softmax(-1)
        gold = b["input_ids"][:, 1:]
        pred = lp.argmax(-1)
        for i, e in enumerate(batch):
            lo, hi = e["n_prompt"] - 1, len(e["ids"]) - 1
            g = lp[i, lo:hi].gather(-1, gold[i, lo:hi, None])[:, 0]
            out.append((bool((pred[i, lo:hi] == gold[i, lo:hi]).all()), g.min().exp().item(), g.mean().item()))
        del logits, lp
    return out


def auroc(scores: list[float], pos: list[bool]) -> float:
    """Probability that a random correct item scores higher than a random wrong one."""
    p = [s for s, y in zip(scores, pos) if y]
    q = [s for s, y in zip(scores, pos) if not y]
    if not p or not q:
        return float("nan")
    q = sorted(q)
    import bisect
    return sum(bisect.bisect_left(q, x) + 0.5 * (bisect.bisect_right(q, x) - bisect.bisect_left(q, x))
               for x in p) / (len(p) * len(q))


def test_confidence(n: int = 200) -> None:
    ctx = Ctx()
    rng = random.Random(66)
    from stress_tests import SYN  # noqa: F401  (synonym_english uses it)
    lang_words = (set(ctx.vocab.nouns) | set(ctx.vocab.locations) | set(ctx.vocab.transitive)
                  | set(ctx.vocab.intransitive) | set(ctx.vocab.adjectives))
    pools: dict[str, list[dict]] = {"normal new sentences": [], "passive voice": [], "synonym": [],
                                    "UPPER CASE": [], "2 adjectives on a noun": []}
    while any(len(v) < n for v in pools.values()):
        f = random_frame(ctx, rng)
        if not valid(ctx, f):
            continue
        it = ctx.item(f)
        if len(pools["normal new sentences"]) < n:
            pools["normal new sentences"].append(it)
        p = passive(f, ctx.vocab)
        if p and len(pools["passive voice"]) < n:
            pools["passive voice"].append({**it, "english": p})
        s = synonym_english(f, ctx.vocab, lang_words)
        if s and len(pools["synonym"]) < n:
            pools["synonym"].append({**it, "english": s[0]})
        if len(pools["UPPER CASE"]) < n:
            pools["UPPER CASE"].append({**it, "english": it["english"].upper()})
        if isinstance(f.subj, NP) and len(pools["2 adjectives on a noun"]) < n:
            a, b = rng.sample(ctx.adjs, 2)
            g = Frame(NP(f.subj.noun, f.subj.plural, None), f.verb, f.obj, f.loc, f.tense, f.neg)
            if valid(ctx, g):
                real = ctx.lang.realize(g)
                adj = [ctx.lang._word([ctx.lang.roots[f"A:{x}"]], [x], real) for x in (a, b)]
                en = english(Frame(NP(f.subj.noun, f.subj.plural, f"{a} {b}"), f.verb, f.obj, f.loc, f.tense, f.neg),
                             ctx.vocab)
                pools["2 adjectives on a noun"].append({"english": en,
                                                         "target": render_sentence(adj + real.words, ctx.script)})
    names = list(pools)
    flat = [it for k in names for it in pools[k]]
    lines = ["# Test 12: does the model know when it is wrong?", "",
             f"{n} sentences per kind. Confidence = the model's lowest probability on any token of the correct "
             "translation (teacher-forced).",
             "**Separation** = chance that a correct sentence gets higher confidence than a wrong one "
             "(0.5 = no idea, 1.0 = perfect warning signal).", "",
             "| model | separation (all kinds pooled) | " + " | ".join(f"{k}: right / mean confidence" for k in names) + " |",
             "|---|---|" + "---|" * len(names)]
    for label, model, tok in each_model():
        r = token_probs(model, tok, flat)
        sep = auroc([x[1] for x in r], [x[0] for x in r])
        cells, k0 = [], 0
        for k in names:
            part = r[k0:k0 + len(pools[k])]
            k0 += len(pools[k])
            cells.append(f"{sum(x[0] for x in part) / len(part):.0%} / {sum(x[1] for x in part) / len(part):.2f}")
        lines.append(f"| {label} | {sep:.2f} | " + " | ".join(cells) + " |")
    write("test12_confidence", lines)


if __name__ == "__main__":
    {"roles": test_roles, "format": test_format, "forms": test_forms, "unusual": test_unusual,
     "output": test_output, "confidence": test_confidence}[sys.argv[1]]()
