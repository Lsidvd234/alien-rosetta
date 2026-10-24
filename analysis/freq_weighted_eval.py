"""Frequency-weighted word accuracy for the v2 models (local).

    python analysis/freq_weighted_eval.py RUN [n_items]     score one run   -> results/freq_weighted/<run>.json
    python analysis/freq_weighted_eval.py --table           all scored runs -> results/freq_weighted.md

Why: a wrong glyph word for a word that is common in English AND common in the model's training data (the dog, I,
go) is a serious error; a wrong word that is rare in English and never or hardly seen in training (anvil) matters
little. So every glyph word of the gold answer is scored right / wrong and weighted by how common it is:

    L = clip((zipf - 1) / 6, 0, 1)                    English frequency of the word (wordfreq zipf, 1..7)
    T = log(1 + count) / log(1 + C99)                 how often its key appears in the run's own training data
                                                      (rebuilt exactly from the run's seed and settings; C99 = 99th
                                                      percentile of the counts; capped at 1)
    w = exp(lambda * (L + T) / 2)                     lambda = 0 is plain word accuracy
    weighted accuracy = sum(w * right) / sum(w)

Pronouns and the grammar's own words (this, two, what, very, ...) count as very common (zipf 6). Names are left
out (they are invented per sentence: no frequency). Gold and predicted glyph words are aligned (difflib), so one
missing word does not count every later word wrong. Test sentences are generated (same 300 for every run), never
in that run's training data; it never looks at the sealed set.
"""

from __future__ import annotations

import collections
import difflib
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train.represent_v2 as R2  # noqa: E402
from conlang.frames import PRONOUNS  # noqa: E402
from conlang.v2 import CONNECTORS, DETERMINERS, INTERJECTIONS, QUESTION_WORDS  # noqa: E402

OUT = ROOT / "results" / "freq_weighted"
LAMBDAS = (0, 1, 2, 3)                                      # columns of the main table
HEADLINE = 3      # fixed in advance (2026-10-15, before scoring e2b held / e2a): a common, often-trained word counts
                  # up to e^3 ~ 20x more than a rare, unseen one. Not chosen from the results.
SWEEP = (0, 0.1, 0.3, 0.5, 0.7, 0.9) + tuple(range(1, 31))  # lambda sweep (user: 0.1-1 by 0.2, 1-30 by 1)
GRAMMAR = set(DETERMINERS) | set(QUESTION_WORDS) | set(INTERJECTIONS) | set(CONNECTORS) | {
    "very", "too", "so", "really", "more", "most", "much", "also", "be"}


def key_counts(items: list[dict]) -> collections.Counter:
    """How often each word key appears in the training analyses (a key = root plus endings: fun>ADJ, book_shop)."""
    c = collections.Counter()
    for it in items:
        if it.get("aug") != "rep2":
            continue
        a = it["analysis"]
        for k in re.findall(r"(?<![A-Z.])([a-z][a-z_]*(?:>[A-Z]+)*)", a):
            c[k] += 1
        for p in re.findall(r"PRON\.(\w+)", a):
            c[f"PRON.{p}"] += 1
    return c


def english_of(gen: R2.GenV2) -> dict:
    """word key -> English word (for its zipf frequency)."""
    m = {}
    for sg, _, k, _ in gen.nouns + gen.places:
        m.setdefault(k, sg)
    for w, k in gen.adjs + gen.manner:
        m.setdefault(k, w)
    for x in gen.verbs:
        m.setdefault(x[1], x[0])
    return m


def word_keys(rep, cl) -> list[tuple[str, str]]:
    """(kind, key) of every glyph word of one event, in order: kind = word | name | pron | grammar."""
    out = []
    for w in rep.words(cl):
        g = w.gloss
        head = g.split("-")[0].replace(".PST", "")
        if head[:1].isupper() and head.isalpha() and head not in ("NAME",):
            out.append(("name", head))
        elif head in PRONOUNS:
            out.append(("pron", f"PRON.{head}"))
        elif head in GRAMMAR or head in ("ACC", "in", "LOC", "will", "not", "the") or head.isupper():
            out.append(("grammar", head))
        else:
            # rebuild the key: gloss pieces after the head that are word-building endings belong to the key
            parts = g.split("-")
            key = parts[0].replace("+", "_").replace(".PST", "")
            for p in parts[1:]:
                if p in R2.FORMS.get("_affixes", ["ADJ", "LESS", "ABLE", "AGENT", "ACT", "QUAL", "VERB", "ADV",
                                                  "NOT", "MORE", "MOST"]):
                    key += f">{p}"
                else:
                    break
            out.append(("word", key))
    return out


def test_items(gen: R2.GenV2, n: int, seed: int = 4242) -> list[tuple[dict, dict]]:
    R2.set_held_out(0)
    rng = random.Random(seed)
    items = []
    while len(items) < n:
        ex = gen.example(rng)
        it = gen.item(ex) if ex else None
        if it:
            items.append((it, ex))
    return items


def score_run(run: str, n: int) -> None:
    from train.load import load_trained, translate
    mdir = ROOT / "results" / "kaggle" / run / "model"
    cfg = json.loads((mdir / "run_args.json").read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 0))
    held = float(cfg.get("rep_held_frac", 0.2))
    rep_n = int(cfg.get("rep_n", 80000))
    cot = str(cfg.get("cot")) == "True"
    OUT.mkdir(parents=True, exist_ok=True)
    cpath = OUT / f"{run}.counts.json"
    if cpath.exists():
        counts = collections.Counter(json.loads(cpath.read_text(encoding="utf-8")))
        train_en = set(json.loads((OUT / f"{run}.train_en.json").read_text(encoding="utf-8")))
    else:
        items, _ = R2.build_training(cfg.get("preset", "tamil_like"), cfg.get("script", "abugida_pua"), seed,
                                     n=rep_n, n_refuse=int(cfg.get("rep_refuse", 2500)),
                                     n_replay=int(cfg.get("rep_replay", 6000)), cot=False, held_frac=held,
                                     min_count=int(cfg.get("rep_min_count", 0)),   # data-fixed runs
                                     loan_rate=float(cfg.get("rep_loan_rate", 0) or 0),   # loanword runs
                                     held_only=None if cfg.get("rep_held_only") in (None, "None", "") else
                                     cfg["rep_held_only"],                                 # structure-dose runs
                                     putback=int(cfg.get("rep_held_putback", 0) or 0))
        R2.LOAN.update(rate=0.0, pool="train")     # build_training set the loan rate: tests use the standard mix
        counts = key_counts(items)
        train_en = {it["english"] for it in items if "english" in it}
        cpath.write_text(json.dumps(counts), encoding="utf-8")
        (OUT / f"{run}.train_en.json").write_text(json.dumps(sorted(train_en)), encoding="utf-8")
    gen = R2.GenV2(cfg.get("preset", "tamil_like"), cfg.get("script", "abugida_pua"), seed)
    eng = english_of(gen)
    tests = [(it, ex) for it, ex in test_items(gen, n + 50) if it["english"] not in train_en][:n]
    model, tok, _ = load_trained(mdir)
    preds = []
    for i in range(0, len(tests), 16):
        preds += translate(model, tok, [it["english"] for it, _ in tests[i:i + 16]], max_new_tokens=700)   # analysis + glyphs can exceed 300 tokens (4% of tests did)
    rows = []
    sentences = []                                      # per test sentence: fully right? (glyph answer exact)
    for si, ((it, ex), p) in enumerate(zip(tests, preds)):
        glyph = p.partition("=>")[2].strip() if cot else p
        gold = it["target"].split()
        kinds = [k for cl in ex["clauses"] for k in word_keys(gen.rep, cl)]
        if len(kinds) != len(gold):
            continue                                   # (should not happen; keep the scoring honest)
        ok = [False] * len(gold)
        sm = difflib.SequenceMatcher(a=gold, b=glyph.split(), autojunk=False)
        for blk in sm.get_matching_blocks():
            for j in range(blk.size):
                ok[blk.a + j] = True
        for (kind, key), right in zip(kinds, ok):
            if kind == "name":
                continue
            if kind in ("pron", "grammar"):
                z = 6.0
            else:
                base = key.split(">")[0].split("_")[0]
                z = R2.FORMS["zipf"].get(eng.get(key, ""), R2.FORMS["zipf"].get(base, 2.0))
            rows.append({"key": key, "kind": kind, "zipf": z, "count": counts.get(key, 0), "right": right, "s": si})
        sentences.append(glyph == it["target"])
        rows_sent = None
    exact = sum((p.partition("=>")[2].strip() if cot else p) == it["target"] for (it, _), p in zip(tests, preds))
    (OUT / f"{run}.json").write_text(json.dumps({"run": run, "cot": cot, "n_sentences": len(tests),
                                                 "sentence_exact": exact / len(tests), "words": rows,
                                                 "sentences": sentences}),
                                     encoding="utf-8")
    print(f"{run}: {len(tests)} sentences, {len(rows)} scored words, sentence exact {exact / len(tests):.3f}")


def weighted(rows: list[dict], lam: float, c99: float) -> float:
    num = den = 0.0
    for r in rows:
        L = min(max((r["zipf"] - 1) / 6, 0.0), 1.0)
        T = min(math.log1p(r["count"]) / math.log1p(c99), 1.0) if c99 > 0 else 0.0
        w = math.exp(lam * (L + T) / 2)
        num += w * r["right"]
        den += w
    return num / den if den else float("nan")


def table() -> None:
    res = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(OUT.glob("*.json"))
           if not f.name.endswith((".counts.json", ".train_en.json"))]
    out = ["# Frequency-weighted word accuracy (v2 models)", "",
           "Every glyph word of the gold answer is right or wrong (aligned with difflib), weighted by "
           "w = exp(lambda * (L + T) / 2): L = English frequency (zipf, scaled 0-1), T = how often the word appears in "
           "that run's own training data (log scale, 0-1). Common, often-trained words weigh up to e^lambda times more "
           "than rare, unseen ones. lambda = 0 is plain word accuracy. Names left out; pronouns and grammar words count "
           "as very common. 300 generated test sentences, the same for every run, none in its training data. "
           "`analysis/freq_weighted_eval.py`.", "",
           "| run | sentences exact | " + " | ".join(f"word acc, lambda={l}" for l in LAMBDAS) + " |",
           "|---|---|" + "---|" * len(LAMBDAS)]
    for r in res:
        counts = sorted(x["count"] for x in r["words"])
        c99 = counts[int(0.99 * (len(counts) - 1))] if counts else 0
        out.append(f"| {r['run']} | {r['sentence_exact']:.3f} | "
                   + " | ".join(f"{weighted(r['words'], l, c99):.3f}" for l in LAMBDAS) + " |")
    out += ["", "## Lambda sweep (weighted word accuracy)", "",
            "| lambda | " + " | ".join(r["run"] for r in res) + " |", "|---|" + "---|" * len(res)]
    c99s = []
    for r in res:
        counts = sorted(x["count"] for x in r["words"])
        c99s.append(counts[int(0.99 * (len(counts) - 1))] if counts else 0)
    for lam in SWEEP:
        out.append(f"| {lam:g} | " + " | ".join(f"{weighted(r['words'], lam, c):.4f}" for r, c in zip(res, c99s))
                   + " |")
    if len(res) > 1:
        out += ["", "Ranking of the runs at every lambda (best first):", ""]
        orders = {}
        for lam in SWEEP:
            order = tuple(r["run"] for _, r in sorted(((weighted(r["words"], lam, c), r) for r, c in zip(res, c99s)),
                                                       key=lambda x: -x[0]))
            orders.setdefault(order, []).append(lam)
        for order, lams in orders.items():
            out.append(f"- lambda {min(lams):g} to {max(lams):g}: " + " > ".join(order))
    out += ["", f"## Summary: plain and weighted (lambda = {HEADLINE}, fixed in advance) accuracy", "",
            "| run | word, plain | word, weighted | sentence, plain | sentence, weighted |", "|---|---|---|---|---|"]
    for r, c in zip(res, c99s):
        sw = f"{weighted_sentences(r, HEADLINE, c):.3f}" if r.get("sentences") else "(rerun pending)"
        out.append(f"| {r['run']} | {weighted(r['words'], 0, c):.3f} | {weighted(r['words'], HEADLINE, c):.3f} | "
                   f"{r['sentence_exact']:.3f} | {sw} |")
    sweep_plot(res, c99s)
    sentence_plot(res)
    with_s = [(r, c) for r, c in zip(res, c99s) if r.get("sentences")]
    if with_s:
        out += ["", "## Lambda sweep, full-sentence accuracy (sentences weighted by how common their words are)", "",
                "| lambda | " + " | ".join(r["run"] for r, _ in with_s) + " |", "|---|" + "---|" * len(with_s)]
        for lam in SWEEP:
            out.append(f"| {lam:g} | " + " | ".join(f"{weighted_sentences(r, lam, c):.4f}" for r, c in with_s) + " |")
        sweep_plot([r for r, _ in with_s], [c for _, c in with_s], sentences=True)
    for r in res:
        out += ["", f"## {r['run']}: word accuracy by frequency (number of words in brackets)", "",
                "| seen in training | rare in English (zipf < 3) | medium (3-4.5) | common (> 4.5) | all |",
                "|---|---|---|---|---|"]
        for lo, hi, name in ((0, 0, "never"), (1, 9, "1-9 times"), (10, 29, "10-29 times"), (30, 10 ** 9, "30+ times")):
            cells = []
            for zl, zh in ((0, 3), (3, 4.5), (4.5, 99)):
                sel = [w for w in r["words"] if lo <= w["count"] <= hi and zl <= w["zipf"] < zh]
                cells.append(f"{sum(w['right'] for w in sel) / len(sel):.2f} ({len(sel)})" if sel else "-")
            sel = [w for w in r["words"] if lo <= w["count"] <= hi]
            cells.append(f"{sum(w['right'] for w in sel) / len(sel):.2f} ({len(sel)})" if sel else "-")
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        wrong = collections.Counter(w["key"] for w in r["words"] if not w["right"] and w["count"] >= 30
                                    and w["zipf"] >= 4.5)
        if wrong:
            out += ["", "Common, often-trained words it still got wrong: "
                    + ", ".join(f"{k} ({n})" for k, n in wrong.most_common(15))]
    (ROOT / "results" / "freq_weighted.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote results/freq_weighted.md")


def word_score(r: dict, c99: float) -> float:
    L = min(max((r["zipf"] - 1) / 6, 0.0), 1.0)
    T = min(math.log1p(r["count"]) / math.log1p(c99), 1.0) if c99 > 0 else 0.0
    return (L + T) / 2


def weighted_sentences(run: dict, lam: float, c99: float) -> float:
    """Full-sentence accuracy, each sentence weighted by exp(lambda * mean word score of its words)."""
    by_s: dict = {}
    for w in run["words"]:
        by_s.setdefault(w["s"], []).append(word_score(w, c99))
    num = den = 0.0
    for si, ok in enumerate(run["sentences"]):
        sc = by_s.get(si)
        if not sc:
            continue
        wgt = math.exp(lam * sum(sc) / len(sc))
        num += wgt * ok
        den += wgt
    return num / den if den else float("nan")


def sweep_plot(res: list[dict], c99s: list[float], sentences: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r, c in zip(res, c99s):
        f = (lambda lam: weighted_sentences(r, lam, c)) if sentences else (lambda lam: weighted(r["words"], lam, c))
        ax.plot(SWEEP, [f(lam) for lam in SWEEP], marker="o", ms=3, lw=1.8, label=r["run"])
    ax.axvline(HEADLINE, color="#888", ls="--", lw=1)
    ax.text(HEADLINE, ax.get_ylim()[0], f" lambda = {HEADLINE} (fixed in advance)", fontsize=9, color="#555",
            va="bottom")
    ax.set_xlabel("lambda (weight on common, often-trained words)")
    ax.set_ylabel("weighted full-sentence accuracy" if sentences else "weighted word accuracy")
    ax.set_title(("Frequency-weighted full-sentence accuracy" if sentences else "Frequency-weighted word accuracy")
                 + " vs lambda")
    if len(res) > 1:
        ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    try:
        fig.savefig(ROOT / "results" / "figures" / ("fig_lambda_sweep_sentences.png" if sentences else
                                                     "fig_lambda_sweep.png"), dpi=150)
    except OSError:
        print("could not replace a lambda-sweep figure (open in another program?)")
    plt.close(fig)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def sentence_plot(res: list[dict]) -> None:
    """Full-sentence accuracy (whole glyph answer exactly right) per run, with 95% Wilson intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    res = sorted(res, key=lambda r: r["sentence_exact"])
    names = [r["run"] for r in res]
    vals = [r["sentence_exact"] for r in res]
    ci = [wilson(round(r["sentence_exact"] * r["n_sentences"]), r["n_sentences"]) for r in res]
    err = [[v - lo for v, (lo, hi) in zip(vals, ci)], [hi - v for v, (lo, hi) in zip(vals, ci)]]
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(res) + 1.6))
    ax.barh(names, vals, xerr=err, color="#1f4e79", ecolor="#555", capsize=3, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + err[1][i] + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, min(1, max(hi for _, hi in ci) + 0.12))
    ax.set_xlabel(f"share of {res[0]['n_sentences']} test sentences fully right")
    ax.set_title("Full-sentence accuracy (95% intervals)")
    fig.tight_layout()
    try:
        fig.savefig(ROOT / "results" / "figures" / "fig_sentence_accuracy.png", dpi=150)
    except OSError:
        print("could not replace fig_sentence_accuracy.png (open in another program?)")
    plt.close(fig)


if __name__ == "__main__":
    if "--table" in sys.argv:
        table()
    else:
        score_run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300)
