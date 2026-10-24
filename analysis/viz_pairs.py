"""Minimal-pair test: change ONE thing in the English, check the model changes exactly the matching glyph words.

    python analysis/viz_pairs.py build                 -> results/viz/pairs_items.json (no GPU)
    python analysis/viz_pairs.py run MODEL [MODEL...]  -> results/viz/pairs.json (adds each model's answers)

A model that composes changes the one word the edit touches (dog -> cat changes the dog word; past -> future changes
the verb). A model that matches whole remembered chunks changes several words, or none. Base sentences are clean
generated statements; each edit is made in the English and checked with the sealed-set reader (english_parse): the
reader must read the base exactly as the generator meant it, and the edited sentence to the base meaning with only the
intended change. The gold glyphs of both come from the language. Generated sentences only; never the sealed set.

Edits: noun swap (another noun of the language), tense (is/are <-> was/were <-> will be), negation (add / remove
"not"), plural object, added describing word.
"""

from __future__ import annotations

import difflib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sealed" / "app"))

import train.represent_v2 as R2  # noqa: E402

OUT = ROOT / "results" / "viz"
CLEAN = {"noise": "none", "lexical": "plain", "spacing": "single", "case": "sentence", "end": "full stop"}
TENSE_SWAP = [(" is ", " was "), (" are ", " were "), (" was ", " is "), (" were ", " are "), (" is ", " will be "),
              (" are ", " will be ")]
AUX = ["is", "are", "was", "were", "will", "can", "could", "should", "must", "might", "may", "would", "has", "have",
       "had", "did", "does", "do"]


def read(parser, text: str, genders: dict):
    from english_parse import render
    r = parser.parse(text)
    if not r.clauses or r.issues:
        return None
    out = render(r.clauses, parser.reps, genders)
    return out["analysis"], out["targets"]["seed0"]


def changed(a: list[str], b: list[str]) -> list[list]:
    """Where b differs from a: [op, a-positions, b-words]."""
    return [[op, list(range(a1, a2)), b[b1:b2]] for op, a1, a2, b1, b2 in
            difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op != "equal"]


def build(n_base: int = 70) -> None:
    from english_parse import Parser
    parser = Parser()
    R2.set_held_out(0)
    gen = R2.GenV2("tamil_like", "abugida_pua", 0)
    rng = random.Random(9090)
    nouns = [(sg, pl, k) for sg, pl, k, mass in gen.nouns if pl and not mass and sg.isalpha() and len(sg) > 2]
    adjs = [a for a, k in gen.adjs if a.isalpha() and ">" not in k and len(a) > 2][:400]
    items, tries = [], 0
    while len(items) < n_base * 5 and tries < 20000:
        tries += 1
        ex = gen.example(rng, force=dict(CLEAN, kind="statement"), n_events=1)
        it = gen.item(ex) if ex else None
        if not it or len(it["english"].split()) > 12:
            continue
        cl = ex["clauses"][0]
        genders = {cl.subj_name: cl.subj_gender} if cl.subj_name else {}
        base = read(parser, it["english"], genders)
        if not base or base != (it["analysis"], it["target"]):
            continue                                   # the reader must agree with the generator on the base
        en = it["english"]
        words = en.rstrip(".").split(" ")
        cands = []
        # noun swap / plural / adjective: on a plain singular noun preceded by 'the' or 'a'
        for i, w in enumerate(words):
            hit = next((x for x in nouns if x[0] == w.lower()), None)
            if hit and i > 0 and words[i - 1].lower() in ("the", "a", "an"):
                other = rng.choice([x for x in nouns if x[0] != hit[0]])
                cands.append(("noun swap", " ".join(words[:i] + [other[0]] + words[i + 1:]) + "."))
                if words[i - 1].lower() == "the":
                    cands.append(("plural noun", " ".join(words[:i] + [hit[1]] + words[i + 1:]) + "."))
                adj = rng.choice(adjs)
                cands.append(("add describing word", " ".join(words[:i] + [adj] + words[i:]) + "."))
                break
        s = " " + en + " "
        for a, b in rng.sample(TENSE_SWAP, len(TENSE_SWAP)):
            if a in s:
                cands.append(("tense", (s.replace(a, b, 1)).strip()))
                break
        for i, w in enumerate(words):
            if w.lower() in AUX:
                if i + 1 < len(words) and words[i + 1] == "not":
                    cands.append(("negation", " ".join(words[:i + 1] + words[i + 2:]) + "."))
                else:
                    cands.append(("negation", " ".join(words[:i + 1] + ["not"] + words[i + 1:]) + "."))
                break
        for kind, new in cands:
            new = re.sub(r"\b([Aa])n? (\w)", lambda m: m.group(1) + ("n " if m.group(2).lower() in "aeiou" else " ")
                         + m.group(2), new)                  # a / an agree with the new next word
            got = read(parser, new, genders)
            if not got or got[0] == base[0]:
                continue
            ga, gb = base[1].split(), got[1].split()
            ch = changed(ga, gb)
            if not ch or sum(len(c[1]) + len(c[2]) for c in ch) > 6:
                continue                              # keep clean one-place edits
            items.append({"kind": kind, "base": en, "edit": new, "base_analysis": base[0], "edit_analysis": got[0],
                          "base_glyphs": base[1], "edit_glyphs": got[1], "gold_change": ch})
        if sum(1 for x in items if x["base"] == en) == 0:
            continue
    by = {}
    for x in items:
        by.setdefault(x["kind"], []).append(x)
    keep = [x for k in by for x in by[k][:60]]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pairs_items.json").write_text(json.dumps(keep, ensure_ascii=False, indent=0), encoding="utf-8")
    print({k: min(60, len(v)) for k, v in by.items()}, "pairs from", tries, "tries")


def outcome(item: dict, pb: str, pe: str) -> str:
    gb, ge = item["base_glyphs"], item["edit_glyphs"]
    if pb == gb and pe == ge:
        return "composed (both right)"
    if pb == gb:
        mch = changed(pb.split(), pe.split())
        if not mch:
            return "ignored the change"
        return "changed, but wrong" if len(mch) == len(item["gold_change"]) else "changed other words too"
    mch = changed(pb.split(), pe.split())
    if [c[1] for c in mch] == [c[1] for c in item["gold_change"]]:
        return "base wrong, change in the right place"
    return "base wrong, change elsewhere"


def run(models: list[str]) -> None:
    from train.load import load_trained, translate
    import gc
    import torch
    items = json.loads((OUT / "pairs_items.json").read_text(encoding="utf-8"))
    res_path = OUT / "pairs.json"
    res = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {"items": items, "models": {}}
    res["items"] = items
    texts = sorted({x["base"] for x in items} | {x["edit"] for x in items})
    for m in models:
        mdir = ROOT / "results" / "kaggle" / m / "model"
        cot = str(json.loads((mdir / "run_args.json").read_text(encoding="utf-8")).get("cot")) == "True"
        model, tok, _ = load_trained(mdir)
        preds = []
        for i in range(0, len(texts), 16):
            preds += translate(model, tok, texts[i:i + 16], max_new_tokens=500)
            print(f"  {m} {len(preds)}/{len(texts)}", flush=True)
        g = {t: (p.partition("=>")[2].strip() if cot else p) for t, p in zip(texts, preds)}
        res["models"][m] = [{"pred_base": g[x["base"]], "pred_edit": g[x["edit"]],
                             "outcome": outcome(x, g[x["base"]], g[x["edit"]])} for x in items]
        del model
        gc.collect()
        torch.cuda.empty_cache()
        res_path.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        import collections
        print(m, collections.Counter(r["outcome"] for r in res["models"][m]))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        run(sys.argv[2:])
