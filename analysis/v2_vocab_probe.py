"""Does a model trained on the v1 language struggle with the v2 vocabulary? (local, small)

    python analysis/v2_vocab_probe.py [model_dir] [n_per_set]     one model -> results/v2_vocab_probe/<run>.json
    python analysis/v2_vocab_probe.py --table                      all models -> results/v2_vocab_probe.md

Four sets of the simplest sentence shape E1 trained on ("the [describing] noun VERB the [describing] noun", one
tense, optional negation, lower case, no extras), identical in everything except where the words come from:
  1. v1 words      - roots the model was trained on (control)
  2. new roots     - v2 root words that are not in v1 (the model has never seen them)
  3. built words   - v2 words built from pieces (fun>ADJ, run>AGENT, book_shop) that are not v1 words
  4. everyday adds - the everyday additions (mom, morning, have, say, tired, ...)
Answers come from conlang/v2.py, which renders v1 words exactly as the model was trained (test_v2: 0 differences).
Scores: glyph answer exact, analysis exact (E1b writes an analysis first), and word accuracy (glyph words in place).
Writes results/v2_vocab_probe.md. Uses generated English only.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train.represent_v2 as R2  # noqa: E402
from train.load import load_trained, translate  # noqa: E402

SHAPE = {"kind": "statement", "predicate": "transitive", "aspect": "simple", "modal": "none", "voice": "active",
         "order": "canonical", "det": "the", "degree": "none", "subject": "noun", "object": "noun", "noise": "none",
         "lexical": "plain", "case": "lower", "end": "none", "spacing": "single", "present": "simple",
         "past": "simple", "future": "will", "negation": "full", "adj_sep": "space"}


def pools(gen: R2.GenV2) -> dict:
    v = gen.v
    v1n, v1a, v1v = set(v.v1.nouns), set(v.v1.adjectives), set(v.v1.verb_info)
    xn, xa, xv = set(v.x_nouns), set(v.x_adjectives), set(v.x_verbs)
    built = (lambda k: ">" in k or "_" in k)
    return {
        "1. v1 words (control)": (
            [x for x in gen.nouns if x[0] in v1n and not built(x[2])],
            [a for a in gen.adjs if a[0] in v1a and not built(a[1])],
            [x for x in gen.tverbs if x[0] in v1v and not built(x[1])]),
        "2. new root words": (
            [x for x in gen.nouns if x[0] not in v1n | xn and not built(x[2])],
            [a for a in gen.adjs if a[0] not in v1a | xa and not built(a[1])],
            [x for x in gen.tverbs if x[0] not in v1v | xv and not built(x[1])]),
        "3. built words": (
            [x for x in gen.nouns if x[0] not in v1n and built(x[2])],
            [a for a in gen.adjs if a[0] not in v1a and built(a[1])],
            [x for x in gen.tverbs if x[0] not in v1v and built(x[1])]),
        "4. everyday additions": (
            [x for x in gen.nouns if x[0] in xn],
            [a for a in gen.adjs if a[0] in xa],
            [x for x in gen.tverbs if x[0] in xv]),
    }


def make_sets(n: int, seed: int = 0) -> dict:
    R2.REL_COUNTS.clear(); R2.REL_COUNTS.update({0: 1.0})
    R2.HOW_COUNTS.clear(); R2.HOW_COUNTS.update({0: 1.0})
    R2.P_TIME = 0.0
    R2.set_held_out(0)
    gen = R2.GenV2("tamil_like", "abugida_pua", seed)
    full = (gen.nouns, gen.noun_w, gen.adjs, gen.adj_w, gen.tverbs, gen.tverb_w)
    sets = {}
    for name, (ns, ads, vs) in pools(gen).items():
        gen.nouns, gen.noun_w = ns, [1.0] * len(ns)                  # uniform within the set
        gen.adjs, gen.adj_w = (ads or full[2]), [1.0] * len(ads or full[2])
        gen.tverbs, gen.tverb_w = vs, [1.0] * len(vs)
        rng = random.Random(7)
        items, seen = [], set()
        while len(items) < n:
            ex = gen.example(rng, force=dict(SHAPE), n_events=1, adj_counts={0: 0.6, 1: 0.4})
            it = gen.item(ex) if ex else None
            if it and it["english"] not in seen:
                seen.add(it["english"])
                items.append(it)
        sets[name] = {"items": items, "pool": (len(ns), len(ads), len(vs))}
    gen.nouns, gen.noun_w, gen.adjs, gen.adj_w, gen.tverbs, gen.tverb_w = full
    return sets


def word_acc(pred: str, gold: str) -> float:
    p, g = pred.split(), gold.split()
    return sum(a == b for a, b in zip(p, g)) / max(len(g), 1)


def main() -> None:
    if "--table" in sys.argv:
        return table()
    mdir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/kaggle/e1b-hard-rep-cot-5ep/model"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    cfg0 = json.loads((mdir / "run_args.json").read_text(encoding="utf-8"))
    sets = make_sets(n, int(cfg0.get("seed", 0)))
    model, tok, cfg = load_trained(mdir)
    cot = str(cfg.get("cot")) == "True"
    rows, examples = [], {}
    for name, s in sets.items():
        items = s["items"]
        t0 = time.time()
        preds = []
        for i in range(0, len(items), 16):
            preds += translate(model, tok, [it["english"] for it in items[i:i + 16]], max_new_tokens=220)
        glyph = [p.split("=>", 1)[1].strip() if (cot and "=>" in p) else p for p in preds]
        ana = [p.split("=>", 1)[0].strip() if (cot and "=>" in p) else "" for p in preds]
        ex = sum(g == it["target"] for g, it in zip(glyph, items)) / len(items)
        an = sum(a == it["analysis"] for a, it in zip(ana, items)) / len(items) if cot else float("nan")
        wa = sum(word_acc(g, it["target"]) for g, it in zip(glyph, items)) / len(items)
        rows.append((name, len(items), s["pool"], ex, an, wa, time.time() - t0))
        examples[name] = [{"english": it["english"], "want": it["analysis"], "got": a}
                          for it, a, g in zip(items, ana, glyph) if g != it["target"]][:5]
        print(f"{name}: n={len(items)} glyph exact {ex:.2f}  analysis exact {an:.2f}  word acc {wa:.2f}", flush=True)
    run = mdir.parent.name
    outd = ROOT / "results" / "v2_vocab_probe"
    outd.mkdir(exist_ok=True)
    json.dump({"run": run, "cot": cot, "seed": cfg0.get("seed"), "rows": rows, "examples": examples},
              open(outd / f"{run}.json", "w", encoding="utf-8"), indent=1, default=str)
    print(f"wrote {outd / (run + '.json')}")


def table() -> None:
    order = ["s1-hard-latininit-50k", "s2-hard-translit-50k", "s3-hard-plaintext-50k", "s4-hard-shuffledinit-50k",
             "r1-hard-aug-50k", "e1a-hard-rep-50k", "e1b-hard-rep-cot-50k", "e1a-hard-rep-5ep",
             "e1b-hard-rep-cot-5ep", "e1b-hard-rep-cot-5ep-seed1"]
    label = {"s1-hard-latininit-50k": "Option 1 (sound start)", "s2-hard-translit-50k": "Option 2 (transliteration)",
             "s3-hard-plaintext-50k": "Option 3a (plain text first)", "s4-hard-shuffledinit-50k": "control (wrong sounds)",
             "r1-hard-aug-50k": "R1", "e1a-hard-rep-50k": "E1a 3ep", "e1b-hard-rep-cot-50k": "E1b 3ep",
             "e1a-hard-rep-5ep": "E1a 5ep", "e1b-hard-rep-cot-5ep": "E1b 5ep",
             "e1b-hard-rep-cot-5ep-seed1": "E1b 5ep seed 1"}
    res = {}
    for f in (ROOT / "results" / "v2_vocab_probe").glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        res[d["run"]] = d
    sets = ["1. v1 words (control)", "2. new root words", "3. built words", "4. everyday additions"]
    out = ["# Do the v1-trained models struggle with the v2 vocabulary?", "",
           "Every saved model, 50 sentences per set, same sentence shape in every set (the [describing] noun VERB "
           "the [describing] noun; one tense; optional 'not'; lower case); only where the words come from changes. "
           "Answers from conlang/v2.py for each model's own language seed. Cell = glyph answer exact "
           "(analysis exact for models that write one). `analysis/v2_vocab_probe.py`.", "",
           "| model | " + " | ".join(sets) + " |", "|---|" + "---|" * len(sets)]
    for run in order + sorted(set(res) - set(order)):
        if run not in res:
            continue
        d = res[run]
        cells = []
        for sname in sets:
            row = next((r for r in d["rows"] if r[0] == sname), None)
            if row is None:
                cells.append("-")
                continue
            ex, an = row[3], row[4]
            cells.append(f"{ex:.2f}" + (f" ({an:.2f})" if d["cot"] else ""))
        out.append(f"| {label.get(run, run)} | " + " | ".join(cells) + " |")
    out += ["", "## Wrong examples, E1b 5ep (first 5 per set): analysis wanted vs written", ""]
    ex = res.get("e1b-hard-rep-cot-5ep", {}).get("examples", {})
    for name, exs in ex.items():
        out.append(f"**{name}**")
        out += [f"- `{e['english']}`  \n  want `{e['want']}`  \n  got  `{e['got']}`" for e in exs]
        out.append("")
    (ROOT / "results" / "v2_vocab_probe.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote results/v2_vocab_probe.md")




if __name__ == "__main__":
    main()
