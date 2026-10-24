"""Fourth batch of stress tests (22-26): follow-ups that check claims made from tests 1-21.

    python analysis/stress_tests4.py all        # run every model (one process each), then write all reports
    python analysis/stress_tests4.py report     # only rebuild the reports from saved parts

22. localise    typo / synonym / UPPER CASE: is only the changed word wrong, or the whole sentence?
23. passive     does a passive sentence get translated as "first noun = doer" (subject and object swapped)?
24. confidence  is own-output confidence useful WITHIN one input type (not just "this input looks odd")?
25. twoadj      2 adjectives on a noun: which candidate translation does the model prefer, and what does it write?
26. forgetting  English damage: stolen probability (glyph tokens) vs a changed English distribution; glyph rows vs LoRA

Each saved model runs in its own process (loading several models in one process exhausted Windows
memory before). Per-model results go to scratch_local/st4_parts/model<i>.json; reports to results/stress/.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stress_tests import RUNS, Ctx, passive, synonym_english, write  # noqa: E402
from stress_tests2 import random_frame, saved_models, typo, valid  # noqa: E402
from stress_tests3 import (ENGLISH_TEXT, auroc, failure_pools, generate, glyph_vocab, item_of, pct,  # noqa: E402
                           score, set_lora_scale)
from conlang.frames import NP, Frame, english  # noqa: E402
from conlang.script import render_sentence  # noqa: E402

SEED = 4040
PARTS = Path(__file__).resolve().parents[2] / "scratch_local" / "st4_parts"
TESTS = ["localise", "passive", "confidence", "twoadj", "forgetting"]


# --- helpers -----------------------------------------------------------------------------------

def lemma_of_gloss(g: str) -> str:
    return g.split("-")[0].split(".")[0]


def surface_lemmas(f: Frame, vocab) -> dict[str, tuple[str, str]]:
    """English surface word -> (lemma, word type) for the content words of a frame."""
    d = {}
    for np_ in (f.subj, f.obj):
        if isinstance(np_, NP):
            d[vocab.noun_info[np_.noun][0] if np_.plural else np_.noun] = (np_.noun, "noun")
            if np_.adj:
                d[np_.adj] = (np_.adj, "adjective")
    for s in vocab.verb_info[f.verb]:
        d[s] = (f.verb, "verb")
    if f.loc:
        d[f.loc] = (f.loc, "location")
    return d


def word_types(it: dict, f: Frame, vocab) -> list[str]:
    lem = {v[0]: v[1] for v in surface_lemmas(f, vocab).values()}
    return [lem.get(lemma_of_gloss(g), "pronoun/other") for g in it["glosses"]]


# --- 22. localisation ----------------------------------------------------------------------------

def pools_localise(ctx: Ctx, per: int = 300) -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    lw = set(ctx.vocab.nouns) | set(ctx.vocab.locations) | set(ctx.vocab.transitive) | set(ctx.vocab.intransitive) \
        | set(ctx.vocab.adjectives)
    pools: dict[str, list[dict]] = defaultdict(list)
    tries = 0
    while min(len(pools[k]) for k in ("typo", "synonym", "UPPER CASE")) < per and tries < 200000:
        tries += 1
        f = random_frame(ctx, rng)
        if not valid(ctx, f):
            continue
        it = item_of(ctx, f)
        sl = surface_lemmas(f, ctx.vocab)
        types = word_types(it, f, ctx.vocab)
        base = {**it, "types": types}
        words = it["english"].split()
        if len(pools["typo"]) < per:
            t = typo(it["english"], rng)
            if t and t != it["english"]:          # swapping two equal letters changes nothing
                i = next(k for k, (a, b) in enumerate(zip(words, t.split())) if a != b)
                if words[i] in sl:
                    lemma = sl[words[i]][0]
                    pos = [k for k, g in enumerate(it["glosses"]) if lemma_of_gloss(g) == lemma]
                    if len(pos) == 1:
                        pools["typo"].append({**base, "english": t, "changed": pos[0]})
        if len(pools["synonym"]) < per:
            s = synonym_english(f, ctx.vocab, lw)
            if s:
                old = s[1].split("->")[0]
                if old in sl:
                    lemma = sl[old][0]
                    pos = [k for k, g in enumerate(it["glosses"]) if lemma_of_gloss(g) == lemma]
                    if len(pos) == 1:
                        pools["synonym"].append({**base, "english": s[0], "changed": pos[0]})
        if len(pools["UPPER CASE"]) < per:
            pools["UPPER CASE"].append({**base, "english": it["english"].upper(), "changed": None})
    return dict(pools)


def run_localise(ctx, model, tok) -> dict:
    out = {}
    for kind, items in pools_localise(ctx).items():
        preds = generate(model, tok, items)
        n = len(items)
        right = sum(p == it["target"] for p, it in zip(preds, items))
        same_len = [(p.split(), it) for p, it in zip(preds, items) if len(p.split()) == len(it["words"])]
        ch_ok = ch_n = oth_ok = oth_n = only_changed = wrong_same = 0
        by_type = defaultdict(lambda: [0, 0])
        n_wrong_words = Counter()
        for pw, it in same_len:
            ok = [a == b for a, b in zip(pw, it["words"])]
            for o, t in zip(ok, it["types"]):
                by_type[t][0] += o
                by_type[t][1] += 1
            if not all(ok):
                n_wrong_words[min(sum(not o for o in ok), 3)] += 1
            c = it["changed"]
            if c is not None:
                ch_ok += ok[c]
                ch_n += 1
                oth = [o for k, o in enumerate(ok) if k != c]
                oth_ok += sum(oth)
                oth_n += len(oth)
                if not all(ok):
                    wrong_same += 1
                    only_changed += (not ok[c]) and all(oth)
        out[kind] = {"n": n, "right": right / n, "same_len": len(same_len) / n,
                     "changed_ok": ch_ok / ch_n if ch_n else None, "other_ok": oth_ok / oth_n if oth_n else None,
                     "only_changed_wrong": only_changed / wrong_same if wrong_same else None,
                     "by_type": {t: v[0] / v[1] for t, v in by_type.items() if v[1]},
                     "n_wrong_words": {str(k): v for k, v in sorted(n_wrong_words.items())},
                     "example": [items[0]["english"], preds[0] == items[0]["target"]]}
    return out


# --- 23. passive: first noun = doer? -------------------------------------------------------------

def pools_passive(ctx: Ctx, per: int = 300) -> list[dict]:
    rng = random.Random(SEED + 1)
    items, tries = [], 0
    while len(items) < per and tries < 200000:
        tries += 1
        f = random_frame(ctx, rng, transitive=True)
        if not valid(ctx, f) or not isinstance(f.subj, NP) or f.subj.noun == f.obj.noun:
            continue
        en = passive(f, ctx.vocab)
        if not en:
            continue
        sw = Frame(f.obj, f.verb, f.subj, f.loc, f.tense, f.neg)
        it = item_of(ctx, f, en=en)
        items.append({**it, "swapped": item_of(ctx, sw)["target"]})
    return items


def run_passive(ctx, model, tok) -> dict:
    items = pools_passive(ctx)
    preds = generate(model, tok, items)
    s_ok = score(model, tok, items)
    s_sw = score(model, tok, items, target_key="swapped")

    def overlap(p, t):
        p, t = p.split(), t.split()
        return sum(a == b for a, b in zip(p, t)) / max(len(p), len(t))
    n = len(items)
    return {"n": n,
            "eq_correct": sum(p == it["target"] for p, it in zip(preds, items)) / n,
            "eq_swapped": sum(p == it["swapped"] for p, it in zip(preds, items)) / n,
            "overlap_correct": sum(overlap(p, it["target"]) for p, it in zip(preds, items)) / n,
            "overlap_swapped": sum(overlap(p, it["swapped"]) for p, it in zip(preds, items)) / n,
            "prefers_swapped": sum(b["logp"] > a["logp"] for a, b in zip(s_ok, s_sw)) / n,
            "example": items[0]["english"]}


# --- 24. confidence within each input type -------------------------------------------------------

def run_confidence(ctx, model, tok) -> dict:
    pools = failure_pools(ctx, 150, SEED + 2)
    flat = [(k, it) for k, v in pools.items() for it in v]
    items = [it for _, it in flat]
    preds = generate(model, tok, items)
    so = score(model, tok, [{**it, "target": p} for it, p in zip(items, preds)])
    right = [p == it["target"] for p, it in zip(preds, items)]
    cmin = [min(s["tok_logp"]) if s["tok_logp"] else -99.0 for s in so]
    cmean = [sum(s["tok_logp"]) / len(s["tok_logp"]) if s["tok_logp"] else -99.0 for s in so]
    norm_ok = sorted(c for (k, _), c, r in zip(flat, cmin, right) if k == "normal" and r)
    thr = norm_ok[int(0.05 * len(norm_ok))] if norm_ok else -99.0     # keeps 95% of correct normal outputs
    per = {}
    for k in pools:
        idx = [i for i, (kk, _) in enumerate(flat) if kk == k]
        r = [right[i] for i in idx]
        flagged = [cmin[i] < thr for i in idx]
        kept = [i for i, fl in zip(idx, flagged) if not fl]
        wrong = [i for i in idx if not right[i]]
        per[k] = {"n": len(idx), "acc": sum(r) / len(r),
                  "auroc_min": auroc([cmin[i] for i in idx], r), "auroc_mean": auroc([cmean[i] for i in idx], r),
                  "flagged": sum(flagged) / len(idx),
                  "wrong_flagged": sum(cmin[i] < thr for i in wrong) / len(wrong) if wrong else None,
                  "acc_kept": sum(right[i] for i in kept) / len(kept) if kept else None}
    return {"pooled_min": auroc(cmin, right), "pooled_mean": auroc(cmean, right), "threshold_p": float(torch.tensor(thr).exp()),
            "per": per}


# --- 25. two adjectives: candidates ---------------------------------------------------------------

CANDS = ["both adjectives, English order (assumed gold)", "both adjectives, reversed order",
         "first adjective only", "second adjective only", "no adjective", "both adjectives, noun dropped"]


def pools_twoadj(ctx: Ctx, per: int = 200) -> list[dict]:
    rng = random.Random(SEED + 3)
    items, tries = [], 0
    while len(items) < per and tries < 200000:
        tries += 1
        f = random_frame(ctx, rng)
        if not isinstance(f.subj, NP):
            continue
        g = Frame(NP(f.subj.noun, f.subj.plural, None), f.verb, f.obj, f.loc, f.tense, f.neg)
        if not valid(ctx, g):
            continue
        a, b = rng.sample(ctx.adjs, 2)
        real = ctx.lang.realize(g)
        A = {x: render_sentence([ctx.lang._word([ctx.lang.roots[f"A:{x}"]], [x], real)], ctx.script) for x in (a, b)}
        W = [render_sentence([w], ctx.script) for w in real.words]
        c = [[A[a], A[b]] + W, [A[b], A[a]] + W, [A[a]] + W, [A[b]] + W, W, [A[a], A[b]] + W[1:]]
        en = english(Frame(NP(f.subj.noun, f.subj.plural, f"{a} {b}"), f.verb, f.obj, f.loc, f.tense, f.neg), ctx.vocab)
        items.append({"english": en, "target": " ".join(c[0]), "cands": [" ".join(x) for x in c]})
    return items


def run_twoadj(ctx, model, tok) -> dict:
    items = pools_twoadj(ctx)
    preds = generate(model, tok, items)
    scores = [score(model, tok, [{**it, "target": it["cands"][j]} for it in items]) for j in range(len(CANDS))]
    pref, wrote = Counter(), Counter()
    for i, it in enumerate(items):
        lp = [scores[j][i]["logp"] for j in range(len(CANDS))]
        pref[CANDS[max(range(len(CANDS)), key=lambda j: lp[j])]] += 1
        hit = [CANDS[j] for j, c in enumerate(it["cands"]) if preds[i] == c]
        wrote[hit[0] if hit else "none of these"] += 1
    n = len(items)
    return {"n": n, "prefers": {k: v / n for k, v in pref.items()}, "wrote": {k: v / n for k, v in wrote.items()}}


# --- 26. forgetting decomposition ---------------------------------------------------------------

@torch.no_grad()
def english_ppl2(model, tok, glyph_ids: list[int], mask: bool) -> tuple[float, float]:
    """English perplexity; mask=True removes the glyph tokens from the softmax (renormalises over the rest)."""
    device = next(model.parameters()).device
    gid = torch.tensor(glyph_ids, device=device)
    nll = n = 0.0
    gm = []
    for s in ENGLISH_TEXT:
        ids = tok(s, return_tensors="pt").input_ids.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            lg = model(input_ids=ids).logits[0, :-1].float()
        gm.append(lg.log_softmax(-1)[:, gid].exp().sum(-1).mean().item())
        if mask:
            lg[:, gid] = float("-inf")
        lp = lg.log_softmax(-1)
        nll -= lp.gather(-1, ids[0, 1:, None]).sum().item()
        n += ids.shape[1] - 1
    return float(torch.exp(torch.tensor(nll / n))), sum(gm) / len(gm)


def run_forgetting(ctx, model, tok, run: str) -> dict:
    from train.run import AddedHead
    delta = next(m.delta for m in model.modules() if isinstance(m, AddedHead))
    g = torch.load(RUNS / run / "model" / "glyph_tokens.pt", map_location="cpu", weights_only=True)
    trained = delta.detach().clone()
    start = g.get("delta_init")
    gids = sorted(glyph_vocab(tok).values())
    rng = random.Random(SEED + 4)
    probe = []
    while len(probe) < 100:
        f = random_frame(ctx, rng)
        if valid(ctx, f):
            probe.append(item_of(ctx, f))
    res, saved = {}, {}
    for rows in (["trained", "start"] if start is not None else ["trained"]):
        with torch.no_grad():
            delta.copy_(trained if rows == "trained" else start.to(delta.device, delta.dtype))
        for lam in (1.0, 0.0):
            set_lora_scale(model, lam, saved)
            ppl, gm = english_ppl2(model, tok, gids, mask=False)
            ppl_m, _ = english_ppl2(model, tok, gids, mask=True)
            acc = None
            if lam == 1.0:
                preds = generate(model, tok, probe)
                acc = sum(p == it["target"] for p, it in zip(preds, probe)) / len(probe)
            res[f"{rows}|{lam}"] = {"ppl": ppl, "ppl_masked": ppl_m, "glyph_mass": gm, "acc": acc}
        set_lora_scale(model, 1.0, saved)
    with torch.no_grad():
        delta.copy_(trained)
    return res


# --- per-model process and reports ---------------------------------------------------------------

def part(idx: str) -> None:
    from train.load import load_trained
    ctx = Ctx()
    label, run = list(saved_models().items())[int(idx)]
    PARTS.mkdir(parents=True, exist_ok=True)
    path = PARTS / f"model{idx}.json"
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"label": label}
    model, tok, _ = load_trained(RUNS / run / "model")
    for t in TESTS:
        if t in done:
            continue
        print(f"[{label}] {t}", flush=True)
        done[t] = run_forgetting(ctx, model, tok, run) if t == "forgetting" else \
            {"localise": run_localise, "passive": run_passive, "confidence": run_confidence,
             "twoadj": run_twoadj}[t](ctx, model, tok)
        path.write_text(json.dumps(done), encoding="utf-8")       # save after every test


def f1(x) -> str:
    return "-" if x is None or x != x else f"{x:.2f}"


def p1(x) -> str:
    return "-" if x is None else pct(x)


def report() -> None:
    parts = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(PARTS.glob("model*.json"))]
    labels = [p["label"] for p in parts]
    hdr = "| " + " | ".join(labels) + " |"
    sep = "---|" * len(labels)

    if all("localise" in p for p in parts):
        L = ["# Test 22: is only the changed word wrong?", "",
             "300 new sentences per kind; free generation. 'Changed word' = the target word for the English word that "
             "was misspelt / replaced by a synonym (UPPER CASE changes every word).", ""]
        for kind in ("typo", "synonym", "UPPER CASE"):
            L += [f"## {kind}", "", "| measure | " + hdr[2:], f"|---|{sep}"]
            rows = [("sentence right", lambda d: p1(d["right"])),
                    ("output has the right number of words", lambda d: p1(d["same_len"])),
                    ("changed word right", lambda d: p1(d["changed_ok"])),
                    ("other words right", lambda d: p1(d["other_ok"])),
                    ("wrong sentences where ONLY the changed word is wrong", lambda d: p1(d["only_changed_wrong"]))]
            if kind == "UPPER CASE":
                rows = rows[:2]
            for name, fn in rows:
                L.append(f"| {name} | " + " | ".join(fn(p["localise"][kind]) for p in parts) + " |")
            types = sorted({t for p in parts for t in p["localise"][kind]["by_type"]})
            for t in types:
                L.append(f"| word right: {t} | " + " | ".join(p1(p["localise"][kind]["by_type"].get(t)) for p in parts) + " |")
            for k, nm in (("1", "1 wrong word"), ("2", "2 wrong words"), ("3", "3+ wrong words")):
                L.append(f"| sentences with {nm} (same length) | " + " | ".join(
                    str(p["localise"][kind]["n_wrong_words"].get(k, 0)) for p in parts) + " |")
            L.append("")
        write("test22_error_localisation", L)

    if all("passive" in p for p in parts):
        L = ["# Test 23: passive voice - does the model take the first noun as the doer?", "",
             f"{parts[0]['passive']['n']} passive sentences (noun subject, transitive verb), e.g. "
             f"'{parts[0]['passive']['example']}'. 'Swapped' = the translation of the ACTIVE sentence with subject and "
             "object exchanged (what a first-noun-is-the-doer reading gives).", "",
             "| measure | " + hdr[2:], f"|---|{sep}"]
        for name, key, fn in (("output = correct translation", "eq_correct", p1),
                              ("output = SWAPPED translation", "eq_swapped", p1),
                              ("word overlap with correct", "overlap_correct", p1),
                              ("word overlap with swapped", "overlap_swapped", p1),
                              ("scores swapped above correct", "prefers_swapped", p1)):
            L.append(f"| {name} | " + " | ".join(fn(p["passive"][key]) for p in parts) + " |")
        write("test23_passive_heuristic", L)

    if all("confidence" in p for p in parts):
        L = ["# Test 24: own-output confidence within each input type", "",
             "Confidence = lowest token probability of what the model wrote. AUROC = chance a right output gets higher "
             "confidence than a wrong one (0.5 useless, 1 perfect; '-' = all right or all wrong). Threshold = the level "
             "that keeps 95% of the correct NORMAL outputs.", "",
             "| | " + hdr[2:], f"|---|{sep}",
             "| pooled AUROC (min / mean token) | " + " | ".join(
                 f"{f1(p['confidence']['pooled_min'])} / {f1(p['confidence']['pooled_mean'])}" for p in parts) + " |",
             "| threshold (token probability) | " + " | ".join(f1(p['confidence']['threshold_p']) for p in parts) + " |",
             "", "Cells: accuracy / within-type AUROC (min token) / share of WRONG outputs flagged / accuracy of the "
             "outputs NOT flagged.", "", "| input kind | " + hdr[2:], f"|---|{sep}"]
        for k in parts[0]["confidence"]["per"]:
            L.append(f"| {k} | " + " | ".join(
                "{} / {} / {} / {}".format(p1(d["acc"]), f1(d["auroc_min"]), p1(d["wrong_flagged"]), p1(d["acc_kept"]))
                for d in (p["confidence"]["per"][k] for p in parts)) + " |")
        write("test24_confidence_within_type", L)

    if all("twoadj" in p for p in parts):
        L = ["# Test 25: two adjectives on a noun - which translation does the model prefer?", "",
             f"{parts[0]['twoadj']['n']} sentences; the subject has 2 adjectives (never in training). Each candidate "
             "is scored (teacher-forced log-probability) and the greedy output is matched against the candidates.", "",
             "## Highest-scoring candidate", "", "| candidate | " + hdr[2:], f"|---|{sep}"]
        for c in CANDS:
            L.append(f"| {c} | " + " | ".join(p1(p["twoadj"]["prefers"].get(c, 0.0)) for p in parts) + " |")
        L += ["", "## What the model actually wrote", "", "| output equals | " + hdr[2:], f"|---|{sep}"]
        for c in CANDS + ["none of these"]:
            L.append(f"| {c} | " + " | ".join(p1(p["twoadj"]["wrote"].get(c, 0.0)) for p in parts) + " |")
        write("test25_two_adjectives", L)

    if all("forgetting" in p for p in parts):
        L = ["# Test 26: where does the English damage come from?", "",
             "English perplexity on 30 everyday sentences (base Qwen3-0.6B-Base: 20.3). 'Masked' = glyph tokens removed "
             "from the softmax: if masked perplexity is near 20, the damage is just probability STOLEN by glyphs; if it "
             "stays high, the adapter changed the English predictions themselves. Rows: trained glyph rows vs their "
             "starting values; LoRA on (1) or off (0). Translation accuracy on 100 new sentences (LoRA on only).", "",
             "Cells: perplexity / masked perplexity / probability on glyphs / translation accuracy.", "",
             "| glyph rows, LoRA | " + hdr[2:], f"|---|{sep}"]
        for key in ("trained|1.0", "trained|0.0", "start|1.0", "start|0.0"):
            cells = []
            for p in parts:
                d = p["forgetting"].get(key)
                cells.append("-" if d is None else f"{d['ppl']:.1f} / {d['ppl_masked']:.1f} / {d['glyph_mass']:.3f} / "
                                                   f"{p1(d['acc'])}")
            L.append(f"| {key.replace('|', ', LoRA ')} | " + " | ".join(cells) + " |")
        write("test26_forgetting_decomposition", L)


def run_all() -> None:
    for i in range(len(saved_models())):
        p = PARTS / f"model{i}.json"
        have = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if all(t in have for t in TESTS):
            continue
        r = subprocess.run([sys.executable, "-X", "utf8", "-X", "faulthandler", str(Path(__file__).resolve()), "part", str(i)])
        print(f"model {i} exit {r.returncode}", flush=True)
        report()                                                    # partial reports after every model
    report()


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"all": run_all, "report": report, "part": lambda: part(sys.argv[2])}[cmd]()
