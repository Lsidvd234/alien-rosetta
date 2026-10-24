"""Word frequency in the v2 training data, and its correlation with English frequency (computed from the data).

    python analysis/plot_word_freq.py [run]

Inputs (both computed locally, nothing typed in by hand):
  - results/freq_weighted/<run>.counts.json: how often each word key appears in the run's training data (the exact
    data, rebuilt from the run's seed and settings by analysis/freq_weighted_eval.py)
  - conlang/data/english_forms.json "zipf": English frequency of each word from wordfreq (Speer et al.; Wikipedia,
    subtitles, news, books, web, Twitter, Reddit), made by scripts/build_english_forms.py. zipf = log10(frequency per
    billion words), so English probability = 10^(zipf - 9).
Outputs:
  results/figures/fig_train_word_prob.png     words sorted by training frequency (least left, most right), grouped
                                              into equal-size groups; y = probability of the word in training
  results/figures/fig_train_vs_english.png    English probability vs training probability, one dot per word,
                                              with the binned median and the correlation (Pearson on logs, Spearman)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import train.represent_v2 as R2  # noqa: E402
from analysis.plot_word_coverage import words  # noqa: E402

GROUPS = 200


def save(fig, path: Path) -> None:
    try:
        fig.savefig(path, dpi=150)
    except OSError:                                   # e.g. the file is open in an image viewer on Windows
        print(f"could not replace {path.name} (open in another program?); kept the old file")
    plt.close(fig)


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), for Spearman's correlation."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    vals, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / cnt)[inv]


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "e2b-v2-cot-80k-noheld"
    counts = json.loads((ROOT / "results" / "freq_weighted" / f"{run}.counts.json").read_text(encoding="utf-8"))
    R2.set_held_out(0)
    ws = words(R2.GenV2("tamil_like", "abugida_pua", 0))
    zipf = R2.FORMS["zipf"]
    ws = [w for w in ws if w["en"] in zipf]                    # words with a wordfreq value
    c = np.array([counts.get(w["key"], 0) for w in ws], dtype=float)
    p_train = c / c.sum()
    p_en = np.array([10 ** (zipf[w["en"]] - 9) for w in ws])
    out_dir = ROOT / "results" / "figures"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    # 1. training probability, words sorted from least to most frequent, grouped into equal-size groups
    s = np.sort(p_train)
    groups = np.array_split(s, GROUPS)
    x = np.array([(i + 0.5) / GROUPS * 100 for i in range(GROUPS)])
    y = np.array([g.mean() for g in groups])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, color="#1f4e79", lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("words, sorted by training frequency (percentile; least frequent left)")
    ax.set_ylabel("probability in training data")
    ax.set_title(f"Word probability in the training data ({len(ws):,} words, {GROUPS} groups)")
    fig.tight_layout()
    save(fig, out_dir / "fig_train_word_prob.png")

    # 2. English probability vs training probability
    seen = c > 0
    lx, ly = np.log10(p_en[seen]), np.log10(p_train[seen])
    pearson = float(np.corrcoef(lx, ly)[0, 1])
    spearman = float(np.corrcoef(ranks(p_en), ranks(p_train))[0, 1])
    bins = np.quantile(lx, np.linspace(0, 1, 21))
    idx = np.clip(np.digitize(lx, bins[1:-1]), 0, 19)
    bx = np.array([np.median(lx[idx == i]) for i in range(20)])
    by = np.array([np.median(ly[idx == i]) for i in range(20)])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(10 ** lx, 10 ** ly, s=3, color="#9aa5b1", alpha=0.35, lw=0)
    ax.plot(10 ** bx, 10 ** by, color="#1f4e79", lw=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("probability in English (wordfreq)")
    ax.set_ylabel("probability in training data")
    ax.set_title("English frequency vs training frequency")
    ax.text(0.03, 0.97, f"Pearson r (log-log) = {pearson:.2f}\nSpearman rho = {spearman:.2f}\nn = {len(ws):,} words",
            transform=ax.transAxes, va="top", fontsize=10)
    fig.tight_layout()
    save(fig, out_dir / "fig_train_vs_english.png")
    # 3. histogram: share of words by how often they appear (log-spaced bins; words never seen cannot sit on a log
    #    axis and are reported in the title)
    # bin edges on whole numbers (counts are integers: a log bin narrower than 1 would hold no count or two)
    edges = np.unique(np.floor(np.logspace(0, np.log10(c.max() + 1), 40))) - 0.5
    edges = np.append(edges, c.max() + 0.5)
    h, _ = np.histogram(c[seen], bins=edges)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dens = h / len(c) / np.diff(np.log10(edges))            # share of words per unit of log10: bins of different
    ax.bar(edges[:-1], dens, width=np.diff(edges), align="edge", color="#1f4e79", edgecolor="white", lw=0.5)
    ax.set_xscale("log")                                      # widths become comparable (area = share of words)
    ax.set_xlabel("appearances in the training data (log scale)")
    ax.set_ylabel("density (share of words per log10 unit)")
    ax.set_title(f"How often words appear in training ({len(c):,} words; {int((~seen).sum())} never seen)")
    fig.tight_layout()
    save(fig, out_dir / "fig_train_word_hist.png")
    lc = np.log10(c[seen])
    print(f"log10(appearances): mean {lc.mean():.2f}  sd {lc.std():.2f}  median count {np.median(c):.0f}")
    print(f"words {len(ws)}  never seen {int((~seen).sum())}  pearson(log) {pearson:.3f}  spearman {spearman:.3f}")
    print("training probability, groups 1 / 50 / 100 / 150 / 200:",
          ", ".join(f"{y[i]:.2e}" for i in (0, 49, 99, 149, 199)))


if __name__ == "__main__":
    main()
