"""How often every word of the v2 language appears in a run's training data, and which words are left behind.

    python analysis/plot_word_coverage.py [run]        (default: e2b-v2-cot-80k-noheld)

Needs the run's word counts from analysis/freq_weighted_eval.py (results/freq_weighted/<run>.counts.json: the exact
training data rebuilt from the run's seed and settings). Writes
    results/figures/fig_word_coverage.png       four panels (see the titles)
    results/word_coverage_left_behind.md        words never or rarely seen, most common English words first
"""

from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import train.represent_v2 as R2  # noqa: E402

BUCKETS = [(0, 0, "never", "#c0392b"), (1, 9, "1-9", "#e67e22"), (10, 29, "10-29", "#f1c40f"),
           (30, 10 ** 9, "30+", "#27ae60")]


def words(gen: R2.GenV2) -> list[dict]:
    v = gen.v
    xn, xa, xv = set(v.x_nouns), set(v.x_adjectives), set(v.x_verbs)
    out = {}

    def put(en, key, cat):
        if key not in out:
            out[key] = {"en": en, "key": key, "cat": cat}
    built = (lambda k: ">" in k or "_" in k)
    for sg, _, k, _ in gen.nouns:
        put(sg, k, "everyday addition" if sg in xn else ("noun, built" if built(k) else "noun"))
    for sg, _, k, _ in gen.places:
        put(sg, k, "place")
    for x in gen.verbs:
        put(x[0], x[1], "everyday addition" if x[0] in xv else ("verb, built" if built(x[1]) else "verb"))
    for w, k in gen.adjs:
        put(w, k, "everyday addition" if w in xa else ("describing, built" if built(k) else "describing"))
    for w, k in gen.manner:
        put(w, k, "how-word")
    return list(out.values())


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "e2b-v2-cot-80k-noheld"
    counts = json.loads((ROOT / "results" / "freq_weighted" / f"{run}.counts.json").read_text(encoding="utf-8"))
    R2.set_held_out(0)
    gen = R2.GenV2("tamil_like", "abugida_pua", 0)
    ws = words(gen)
    for w in ws:
        w["count"] = counts.get(w["key"], 0)
        w["zipf"] = R2.FORMS["zipf"].get(w["en"], 2.0)
    n = len(ws)
    bucket = (lambda c: next(b for b in BUCKETS if b[0] <= c <= b[1]))
    tally = collections.Counter(bucket(w["count"])[2] for w in ws)

    fig, ax = plt.subplots(2, 2, figsize=(15, 10.5))
    # (a) every word, most to least frequent
    cs = sorted((w["count"] for w in ws), reverse=True)
    a = ax[0, 0]
    a.plot(range(n), [c + 1 for c in cs], color="#2c3e50", lw=1.2)
    a.set_yscale("log")
    for lo, hi, name, col in BUCKETS:
        idx = [i for i, c in enumerate(cs) if lo <= c <= hi]
        if idx:
            a.axvspan(idx[0], idx[-1] + 1, color=col, alpha=0.18, lw=0)
    a.set_title(f"(a) Every word of the language ({n:,}), most to least seen in training")
    a.set_xlabel("words, ranked by how often they appear in the training data")
    a.set_ylabel("appearances + 1 (log scale)")
    a.text(0.98, 0.95, "\n".join(f"{name}: {tally[name]:,} words ({tally[name] / n:.0%})" for _, _, name, _ in BUCKETS),
           transform=a.transAxes, ha="right", va="top", fontsize=10,
           bbox=dict(facecolor="white", edgecolor="#bbb", boxstyle="round"))
    # (b) by kind of word
    cats = ["noun", "noun, built", "place", "verb", "verb, built", "describing", "describing, built", "how-word",
            "everyday addition"]
    b = ax[0, 1]
    left = [0.0] * len(cats)
    for lo, hi, name, col in BUCKETS:
        share = []
        for c in cats:
            sel = [w for w in ws if w["cat"] == c]
            share.append(sum(lo <= w["count"] <= hi for w in sel) / max(1, len(sel)))
        b.barh(cats, share, left=left, color=col, label=f"seen {name}")
        left = [l + s for l, s in zip(left, share)]
    b.set_xlim(0, 1)
    b.invert_yaxis()
    b.set_yticks(range(len(cats)), [f"{c} ({sum(w['cat'] == c for w in ws):,})" for c in cats])
    b.set_title("(b) Share of each kind of word by how often it was seen")
    b.set_xlabel("share of the words of that kind")
    b.legend(loc="lower right", fontsize=9)
    # (c) English frequency vs training appearances
    c_ = ax[1, 0]
    for lo, hi, name, col in BUCKETS:
        sel = [w for w in ws if lo <= w["count"] <= hi]
        c_.scatter([w["zipf"] for w in sel], [w["count"] + 1 for w in sel], s=5, alpha=0.5, color=col,
                   label=f"seen {name}")
    c_.set_yscale("log")
    c_.set_xlabel("how common the word is in English (wordfreq zipf; 3 = rare, 5 = common, 6+ = very common)")
    c_.set_ylabel("appearances in training + 1 (log)")
    c_.set_title("(c) Common English words are seen more often (sampling follows English frequency)")
    c_.legend(fontsize=9, markerscale=3)
    # (d) histogram of appearances
    d = ax[1, 1]
    edges = [0, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 5000, 100000]
    hist = [sum(edges[i] <= w["count"] < edges[i + 1] for w in ws) for i in range(len(edges) - 1)]
    labels = [f"{edges[i]}" if edges[i + 1] - edges[i] == 1 else f"{edges[i]}-{edges[i + 1] - 1}"
              for i in range(len(edges) - 1)]
    labels[-1] = f"{edges[-2]}+"
    cols = [bucket(edges[i])[3] for i in range(len(edges) - 1)]
    d.bar(range(len(hist)), hist, color=cols)
    d.set_xticks(range(len(hist)), labels, rotation=45, ha="right")
    for i, h in enumerate(hist):
        d.text(i, h, f"{h:,}", ha="center", va="bottom", fontsize=8)
    d.set_title("(d) How many words appear how often")
    d.set_xlabel("appearances in the training data")
    d.set_ylabel("number of words")
    fig.suptitle(f"Word coverage of the training data: {run} (80k examples; each word's key counted in the "
                 f"analyses)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = ROOT / "results" / "figures" / "fig_word_coverage.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    # left-behind list: most common English words first
    lines = [f"# Words left behind in {run}'s training data", "",
             f"{n:,} words the generator can use; " + ", ".join(f"seen {name}: {tally[name]:,}" for _, _, name, _ in
                                                               BUCKETS) + ".", "",
             "## Never seen (most common in English first)", ""]
    never = sorted((w for w in ws if w["count"] == 0), key=lambda w: -w["zipf"])
    lines += [", ".join(f"{w['en']} ({w['zipf']:.1f})" for w in never[:300]) or "(none)", "",
              "## Seen 1-9 times, but common in English (zipf >= 4)", ""]
    rare_common = sorted((w for w in ws if 1 <= w["count"] <= 9 and w["zipf"] >= 4), key=lambda w: -w["zipf"])
    lines += [", ".join(f"{w['en']} ({w['count']}x, zipf {w['zipf']:.1f})" for w in rare_common[:300]) or "(none)", "",
              "## Counts by kind of word", "", "| kind | words | never | 1-9 | 10-29 | 30+ |", "|---|---|---|---|---|---|"]
    for c in cats:
        sel = [w for w in ws if w["cat"] == c]
        lines.append(f"| {c} | {len(sel):,} | " + " | ".join(
            str(sum(lo <= w["count"] <= hi for w in sel)) for lo, hi, _, _ in BUCKETS) + " |")
    (ROOT / "results" / "word_coverage_left_behind.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    print({name: tally[name] for _, _, name, _ in BUCKETS}, "of", n)


if __name__ == "__main__":
    main()
