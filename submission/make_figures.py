"""Result figures for the ICLR-style report and the slides.

Every number below is copied from the project's result files (see the comment on each figure);
nothing is recomputed here. Sealed-set numbers are totals only.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, LIGHTGRAY = "#8f8d88", "#d6d4cf"
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": LIGHTGRAY, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe8",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "text.color": INK,
    "savefig.dpi": 220, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def label_bars(ax, bars, fmt="{:.2f}", dy=0.015, size=8.5):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom",
                fontsize=size, color=INK2)


# 1. Claim A: passive dose (report sec. claimA, results/score_passive.log)
fig, ax = plt.subplots(figsize=(4.6, 3.2))
doses = [0, 50, 300]
ax.plot(doses, [0.00, 0.53, 0.75], "-o", color=BLUE, lw=2, ms=7, label="meaning exact (seed 0)")
ax.plot(doses, [0.00, 0.31, 0.53], "-o", color=ORANGE, lw=2, ms=7, label="glyphs exact (seed 0)")
ax.plot([0, 50], [0.00, 0.66], "D", color=BLUE, mfc="white", mew=2, ms=7, label="meaning, seed 1")
ax.plot([0, 50], [0.00, 0.47], "D", color=ORANGE, mfc="white", mew=2, ms=7, label="glyphs, seed 1")
ax.axhline(0.86, color=GRAY, ls="--", lw=1)
ax.text(300, 0.875, "ordinary sentences (meaning, dose 300)", ha="right", fontsize=8, color=INK2)
ax.set_xticks(doses)
ax.set_xlabel("plain passives put back into ~88k training examples")
ax.set_ylabel("held-out passive test (100)")
ax.set_ylim(-0.03, 1.0)
ax.set_title("Fifty passives move reading from 0 to over half")
ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
save(fig, "fig_passive_dose")

# 2. Claim B: rule vs list (results/loanwords_*.md, soundrule scores)
fig, ax = plt.subplots(figsize=(5.6, 3.2))
groups = ["loanwords\n(4% of nouns)", "loanwords x3\n(12% of nouns)", "sound-rule language\n(all content words)"]
seen = [0.51, 0.82, 0.96]
new = [0.19, 0.27, 0.67]
rec = [0.34, 0.46, 0.87]
x = range(3)
w = 0.26
b1 = ax.bar([i - w for i in x], seen, w * 0.92, color=BLUE, label="seen words: glyph word right")
b2 = ax.bar(list(x), new, w * 0.92, color=ORANGE, label="never-trained words: glyph word right")
b3 = ax.bar([i + w for i in x], rec, w * 0.92, color=AQUA, label="never-trained words: recognised in meaning")
for bs in (b1, b2, b3):
    label_bars(ax, bs, size=7.5)
ax.set_xticks(list(x))
ax.set_xticklabels(groups, fontsize=8.5)
ax.set_ylim(0, 1.3)
ax.set_ylabel("share right (1 run each)")
ax.set_title("A spelling rule transfers only when it covers the vocabulary")
ax.legend(fontsize=7.5, loc="upper left", ncol=1)
save(fig, "fig_rule_vs_list")

# 3. Claim C: English variety (report claim C tables; Tatoeba meaning exact, sealed meaning exact)
runs = [("baseline (seed 0)", 0.617, 0.476, GRAY), ("focal loss", 0.588, 0.442, GRAY),
        ("paraphrases 1.4%", 0.615, 0.449, GRAY), ("paraphrases 8%", 0.620, 0.469, GRAY),
        ("6 epochs", 0.563, 0.449, GRAY), ("word floor 25", 0.603, 0.503, GRAY),
        ("everyday English words", 0.725, 0.517, AQUA), ("15% real Tatoeba text", 0.923, 0.558, BLUE)]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
names = [r[0] for r in runs]
for ax, col, title in ((axes[0], 1, "Tatoeba 600: meaning exact"), (axes[1], 2, "Sealed set 147: meaning exact")):
    vals = [r[col] for r in runs]
    bars = ax.barh(range(len(runs)), vals, color=[r[3] for r in runs], height=0.7)
    for b, v in zip(bars, vals):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.axvline(runs[0][col], color=INK2, lw=0.8, ls=":")
    ax.set_xlim(0, 1.08)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(runs)))
axes[0].set_yticklabels(names, fontsize=8.5)
axes[0].invert_yaxis()
fig.suptitle("Only more varied English moves real-English reading (gray = model or training tricks)",
             fontsize=10, y=1.02)
save(fig, "fig_english_variety")

# 4. Claim D: reading vs writing (report claim D verdict; results/bottleneck_items.md)
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.3})
ax = axes[0]
tests = ["generated", "Tatoeba", "sealed"]
ax.plot(tests, [0.21, 0.38, 0.52], "-o", color=BLUE, lw=2, ms=7, label="reading errors")
ax.plot(tests, [0.24, 0.31, 0.26], "-o", color=ORANGE, lw=2, ms=7, label="writing errors")
ax.set_ylim(0, 0.65)
ax.set_ylabel("share of sentences (baseline)")
ax.set_title("Reading errors climb")
ax.legend(fontsize=8, loc="upper left")
ax = axes[1]
cats = ["every word\ntrained >=25x\n(n 312)", "rarest word\n1-24x (n 94)", "a word never\ntrained (n 194)"]
wt = [0.84, 0.82, 0.20]
rd = [0.79, 0.70, 0.25]
x = range(3)
b1 = ax.bar([i - 0.19 for i in x], rd, 0.36, color=BLUE, label="reader right")
b2 = ax.bar([i + 0.19 for i in x], wt, 0.36, color=ORANGE, label="writer from gold meaning right")
label_bars(ax, b1, size=7.5)
label_bars(ax, b2, size=7.5)
ax.set_xticks(list(x))
ax.set_xticklabels(cats, fontsize=8)
ax.set_ylim(0, 1.2)
ax.set_title("Tatoeba, by least-trained word")
ax.legend(fontsize=7.5, loc="upper right")
save(fig, "fig_reading_writing")

# 5. Claim E: steering (report claim E tables, seed 0 and seed 1 baseline)
pairs = ["PRES\n>PAST", "PAST\n>PRES", "PRES\n>FUT", "FUT\n>PRES", "NEG\nadd", "NEG\nremove", "Q\nadd", "Q\nremove"]
s0 = [1.00, 0.98, 0.98, 0.98, 0.93, 0.99, 0.29, 0.81]
s1 = [1.00, 0.99, 1.00, 0.97, 0.91, 0.98, 0.95, 0.89]
rnd = [0.02, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.02]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
x = range(len(pairs))
b1 = ax.bar([i - 0.27 for i in x], s0, 0.26, color=BLUE, label="steered, seed 0")
b2 = ax.bar(list(x), s1, 0.26, color=AQUA, label="steered, seed 1")
b3 = ax.bar([i + 0.27 for i in x], rnd, 0.26, color=GRAY, label="random direction (seed 0)")
ax.set_xticks(list(x))
ax.set_xticklabels(pairs, fontsize=8.5)
ax.set_ylim(0, 1.12)
ax.set_ylabel("share of outputs flipped")
ax.set_title("Directions from the English hidden states steer the invented-language output (baseline)")
ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24))
save(fig, "fig_steering")

# 6. Claim A in maths: one removed fact (report claim A table; maths results)
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.25})
ax = axes[0]
facts = ["5x6", "3x4", "2x2"]
need = [0.047, 0.103, 0.41]
avoid = [0.923, 0.957, 0.91]
x = range(3)
b1 = ax.bar([i - 0.19 for i in x], need, 0.36, color=ORANGE, label="problems that need it")
b2 = ax.bar([i + 0.19 for i in x], avoid, 0.36, color=BLUE, label="problems that avoid it")
label_bars(ax, b1, size=7.5)
label_bars(ax, b2, size=7.5)
ax.set_xticks(list(x))
ax.set_xticklabels([f"{f} removed" for f in facts], fontsize=8.5)
ax.set_ylim(0, 1.35)
ax.set_ylabel("exact answer (base-7 glyph digits)")
ax.set_title("A: a missing fact is not inferred")
ax.legend(fontsize=7.5, loc="upper left", ncol=2)
ax = axes[1]
sp = ["recall\nscratchpad", "build, one\nrule", "build, three\nrules mixed"]
ans = [0.26, 0.08, 0.77]
steps = [0.49, 0.07, 0.83]
b1 = ax.bar([i - 0.19 for i in x], ans, 0.36, color=ORANGE, label="needs 5x6: answer right")
b2 = ax.bar([i + 0.19 for i in x], steps, 0.36, color=AQUA, label="5x6 steps right")
label_bars(ax, b1, size=7.5)
label_bars(ax, b2, size=7.5)
ax.set_xticks(list(x))
ax.set_xticklabels(sp, fontsize=8)
ax.set_ylim(0, 1.35)
ax.set_title("F: building the fact (1 seed)")
ax.legend(fontsize=7.5, loc="upper left", ncol=2)
save(fig, "fig_maths")

# 7. Error split, baseline vs real text (results/error_split.md, % of items)
fig, ax = plt.subplots(figsize=(6.8, 2.7))
rows = [("baseline, generated", 60, 19, 10, 10), ("real text, generated", 58, 21, 9, 11),
        ("baseline, Tatoeba", 46, 16, 14, 24), ("real text, Tatoeba", 89, 3, 5, 3)]
cols = [(BLUE, "both right"), (ORANGE, "translation error"), (AQUA, "understanding error"),
        (GRAY, "other (incl. word not in language)")]
for i, r in enumerate(rows):
    left = 0
    for v, (c, lab) in zip(r[1:], cols):
        ax.barh(i, v - 0.4, left=left, color=c, height=0.62, label=lab if i == 0 else None)
        if v >= 6:
            ax.text(left + v / 2, i, f"{v}", ha="center", va="center", fontsize=8, color="white")
        left += v
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("% of test items")
ax.grid(axis="y", visible=False)
ax.set_title("Every output split into understanding vs translation errors")
ax.legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.25))
save(fig, "fig_error_split")

# 8. Held-out structures (report claim A, v1 E1 table, E1b 5 epochs seed 0)
fig, ax = plt.subplots(figsize=(6.6, 3.1))
groups = ["trained\nstructures", "5-6\nadjectives", "held-out\nsynonyms", "subject\ncleft",
          "object\ncleft", "passive\nvoice", "topicalised\nobject"]
vals = [0.99, 0.71, 0.10, 0.09, 0.02, 0.01, 0.00]
colors = [BLUE, AQUA] + [ORANGE] * 5
bars = ax.bar(range(len(vals)), vals, color=colors, width=0.7)
label_bars(ax, bars, size=7.5)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(groups, fontsize=8)
ax.set_ylim(0, 1.12)
ax.set_ylabel("exact (strict held-out sets)")
ax.set_title("Structures held out of training are not translated")
save(fig, "fig_heldout_structures")

print("figures written to", OUT)
