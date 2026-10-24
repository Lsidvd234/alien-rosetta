"""Report figures. Reads the run results (results/kaggle/<run>/results/<run>.json) and the
content-word scores already computed in results/final_20k.md. Writes PNG + PDF to results/figures/.

    python analysis/plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "kaggle"
OUT = ROOT / "results" / "figures"
SPLITS = ("test_iid", "test_comp", "test_morph", "test_long")
SPLIT_NAMES = ("Normal", "New verb+object\npairs", "New ending\ncombination", "Long (new\nstructure)")

COL = {"Latin": "#1f77b4", "Tamil script": "#d62728", "Glyphs, 1 token each": "#2ca02c",
       "Glyphs, 3 byte tokens each": "#9467bd", "Glyphs, 1 token each, sound start": "#000000"}
# (script label) -> {training size in k: run name}, per language
CURVES = {
    "easy": {"Latin": {5: "c-easy-latin-5k", 10: "c-easy-latin-10k", 20: "c-easy-latin-20k", 50: "l06-easy-latin-50k3"},
             "Tamil script": {5: "c-easy-tamil-5k", 10: "c-easy-tamil-10k", 20: "c-easy-tamil-20k", 50: "g-easy-tamil-50k"},
             "Glyphs, 3 byte tokens each": {20: "p-easy-pua-20k", 50: "p-easy-pua-50k", 80: "p-easy-pua-80k"},
             "Glyphs, 1 token each": {20: "p-easy-puatok-20k", 50: "p-easy-puatok-50k"}},
    "hard": {"Latin": {5: "c-hard-latin-5k", 10: "c-hard-latin-10k", 20: "c-hard-latin-20k", 50: "c-hard-latin-50k"},
             "Tamil script": {5: "c-hard-tamil-5k", 10: "c-hard-tamil-10k", 20: "c-hard-tamil-20k", 50: "g-hard-tamil-50k"},
             "Glyphs, 3 byte tokens each": {20: "p-hard-pua-20k", 35: "p-hard-pua-35k", 50: "p-hard-pua-50k",
                                            80: "p-hard-pua-80k"},
             "Glyphs, 1 token each": {20: "p-hard-puatok-20k", 50: "p-hard-puatok-50k"},
             "Glyphs, 1 token each, sound start": {50: "g1-hard-latininit-50k"}},
}
LANG_TITLE = {"easy": "Easy language (English-like grammar)", "hard": "Hard language (Tamil-like grammar)"}


def style(label: str) -> dict:
    if "sound" in label:
        return {"marker": "*", "ms": 16, "ls": "none"}
    return {"marker": "s", "ls": "--"} if "1 token" in label else {"marker": "o", "ls": "-"}


def load(name: str) -> dict:
    return json.loads((RUNS / name / "results" / f"{name}.json").read_text(encoding="utf-8"))


def exact(r: dict, split: str = "test_iid") -> float:
    return 100 * r["eval"][split]["teacher_forced"]["exact_match"]


def md_rows(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and not line.startswith("|---"):
            rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_learning_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, lang in zip(axes, ("easy", "hard")):
        for label, runs in CURVES[lang].items():
            xs = sorted(runs)
            ys = [exact(load(runs[n])) for n in xs]
            ax.plot(xs, ys, color=COL[label], label=label, lw=2, **style(label))
        ax.set_title(LANG_TITLE[lang])
        ax.set_xlabel("Training sentences (thousands)")
        ax.set_xticks([5, 10, 20, 35, 50, 80])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Sentences fully correct (%)")
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    extra = [(h, l + " (hard only)") for h, l in zip(h1, l1) if l not in l0]
    fig.legend(h0 + [h for h, _ in extra], l0 + [l for _, l in extra], loc="lower center", ncol=5,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Learning curves: every script is learned eventually, unfamiliar glyphs need far more data",
                 fontsize=11)
    save(fig, "fig1_learning_curves")


LABEL_OFFSETS = {("hard", "Accented Latin, 1 char/token"): (8, 12),
                 ("hard", "Invented glyphs, 1 new token each"): (-8, -14),
                 ("easy", "Invented glyphs, 1 new token each"): (8, -14),
                 ("hard", "Invented glyphs (PUA)"): (-8, 10),
                 ("easy", "Invented glyphs (PUA)"): (-8, -14)}


def fig_tokens_vs_accuracy() -> None:
    rows = [r for r in md_rows(ROOT / "results" / "final_20k.md") if len(r) == 6 and r[0] in ("easy", "hard")]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for lang, marker, color in (("easy", "o", "#1f77b4"), ("hard", "s", "#d62728")):
        pts = [(float(r[2]), 100 * float(r[3]), r[1]) for r in rows if r[0] == lang]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=marker, s=60, color=color,
                   label=f"{lang} language", zorder=3)
        for x, y, lab in pts:
            short = (lab.replace("Invented glyphs (PUA)", "Glyphs, 3 byte tokens each")
                     .replace("Invented glyphs, 1 new token each", "Glyphs, 1 token each")
                     .replace(", 1 char/token", " (1 char/token)"))
            dx, dy = LABEL_OFFSETS.get((lang, lab), (6, 4))
            ax.annotate(short, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=7.5,
                        ha="right" if dx < 0 else "left")
    ax.set_xlabel("Tokens per sentence")
    ax.set_ylabel("Content words correct (%)")
    ax.set_title("At 20k sentences, token count does not predict accuracy; unfamiliar characters do")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "fig2_tokens_vs_accuracy_20k")


def fig_generalisation() -> None:
    groups = {"easy": [("Latin", "l06-easy-latin-50k3", "50k"), ("Tamil script", "g-easy-tamil-50k", "50k"),
                       ("Glyphs, 1 token each", "p-easy-puatok-50k", "50k"),
                       ("Glyphs, 3 byte tokens each", "p-easy-pua-50k", "50k"),
                       ("Glyphs, 3 byte tokens each", "p-easy-pua-80k", "80k")],
              "hard": [("Latin", "c-hard-latin-50k", "50k"), ("Tamil script", "g-hard-tamil-50k", "50k"),
                       ("Glyphs, 1 token each", "p-hard-puatok-50k", "50k"),
                       ("Glyphs, 3 byte tokens each", "p-hard-pua-50k", "50k"),
                       ("Glyphs, 3 byte tokens each", "p-hard-pua-80k", "80k")]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    w = 0.16
    for ax, lang in zip(axes, ("easy", "hard")):
        for i, (label, run, n) in enumerate(groups[lang]):
            r = load(run)
            ys = [exact(r, s) for s in SPLITS]
            xs = [j + (i - 2) * w for j in range(len(SPLITS))]
            ax.bar(xs, ys, w, color=COL[label], alpha=1.0 if n == "50k" else 0.5,
                   hatch=None if n == "50k" else "//", edgecolor="white", label=f"{label} ({n})")
        ax.set_xticks(range(len(SPLITS)))
        ax.set_xticklabels(SPLIT_NAMES, fontsize=9)
        ax.set_title(LANG_TITLE[lang])
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Sentences fully correct (%)")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Generalisation after full training: only hard grammar + unfamiliar glyphs fails on new "
                 "combinations and structures", fontsize=11)
    save(fig, "fig3_generalisation")


def fig_compute() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, lang in zip(axes, ("easy", "hard")):
        for label, runs in CURVES[lang].items():
            xs = sorted(runs)
            rs = [load(runs[n]) for n in xs]
            ax.plot([r["training"]["seconds"] / 60 for r in rs], [exact(r) for r in rs],
                    color=COL[label], label=label, lw=2, **style(label))
        ax.set_title(LANG_TITLE[lang])
        ax.set_xlabel("Training time on one T4 GPU (minutes)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Sentences fully correct (%)")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Compute cost: byte-split glyphs need ~3-4x the GPU time of Latin; 1 token per glyph halves it",
                 fontsize=11)
    save(fig, "fig4_compute")


FIX_RUNS = [("Glyphs, random-word start (before)", "p-hard-puatok-50k", "#9e9e9e"),
            ("Option 1: Latin-sound start", "g1-hard-latininit-50k", "#2ca02c"),
            ("Control: shuffled-sound start", "s4-hard-shuffledinit-50k", "#8c564b"),
            ("Option 2: + transliteration", "g2-hard-translit-50k", "#17becf"),
            ("Option 3a: plain text first", "g3-hard-plaintext-50k", "#ff7f0e"),
            ("Latin script (target)", "c-hard-latin-50k", "#1f77b4"),
            ("Tamil script (target)", "g-hard-tamil-50k", "#d62728")]


def fig_fixes() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.12
    for i, (label, run, color) in enumerate(FIX_RUNS):
        r = load(run)
        xs = [j + (i - (len(FIX_RUNS) - 1) / 2) * w for j in range(len(SPLITS))]
        target = "target" in label
        ys = [exact(r, s) for s in SPLITS]
        ax.bar(xs, ys, w, color=color, alpha=0.45 if target else 1.0,
               hatch="//" if target else None, edgecolor="white", label=label)
        for x, y in zip(xs, ys):
            if y < 3:                                   # near-zero bars are invisible otherwise
                ax.annotate(f"{y:.0f}%", (x, y), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=7, color=color)
    ax.set_xticks(range(len(SPLITS)))
    ax.set_xticklabels(SPLIT_NAMES, fontsize=9)
    ax.set_ylabel("Sentences fully correct (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8.5)
    ax.set_title("Hard language, invented glyphs, 50k sentences: starting each glyph at its sound's "
                 "embedding fixes generalisation", fontsize=11)
    save(fig, "fig6_generalisation_fixes")


def fig_seeds() -> None:
    rows = {r[0]: r for r in md_rows(ROOT / "results" / "final_20k.md") if len(r) == 7 and "-" in r[0]}
    cells = [("easy-latin", "Latin"), ("easy-tamil", "Tamil script"), ("hard-latin", "Latin"),
             ("hard-tamil", "Tamil script")]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (cell, script) in enumerate(cells):
        r = rows[cell]
        vals = [100 * float(v) for v in r[1].split("/")]
        ax.bar(i, 100 * float(r[2]), yerr=100 * float(r[3]), color=COL[script], capsize=5, alpha=0.85)
        ax.scatter([i] * len(vals), vals, color="black", s=12, zorder=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Easy\nLatin", "Easy\nTamil script", "Hard\nLatin", "Hard\nTamil script"])
    ax.set_ylabel("Content words correct (%)")
    ax.set_title("20k sentences, 3 random languages per cell: grammar dominates")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig5_grammar_vs_script_20k")


if __name__ == "__main__":
    fig_learning_curves()
    fig_tokens_vs_accuracy()
    fig_generalisation()
    fig_compute()
    fig_seeds()
    fig_fixes()
