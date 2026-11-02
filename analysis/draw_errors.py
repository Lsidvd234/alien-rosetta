"""Why does each wrong-sound start fail on a different test? (31 Oct 2026, claim 1 follow-up)

    python analysis/draw_errors.py          -> results/draw_errors.md

Runs: own sound (g1-hard-latininit-50k) and three wrong-sound draws (s4-hard-shuffledinit-50k = shuffle seed 0,
-p1, -p2). Same language, data, seed and test items. Uses only saved files (CPU):
  - per_item_ok (teacher-forced right/wrong per target token) -> errors per glyph, per position in the word, and
    where in a long sentence the first error falls;
  - generation samples (gold vs predicted, 100 items per test) -> which glyph is written in place of which;
  - glyph_tokens.pt (each glyph's start and final embedding) -> how close the confused glyphs started and ended.
"""

from __future__ import annotations

import collections
import difflib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from conlang.dataset import PRESETS, build  # noqa: E402

KAG = ROOT / "results" / "kaggle"
RUNS = [("own", "g1-hard-latininit-50k"), ("draw0", "s4-hard-shuffledinit-50k"),
        ("p1", "s4-hard-shuffledinit-50k-p1"), ("p2", "s4-hard-shuffledinit-50k-p2"),
        # 31 Oct follow-up runs (DV1-DV4)
        ("own-vswap", "s4-own-vswap"), ("p2-vfix", "s4-p2-vfix"), ("p1-ts1", "s4-p1-ts1"), ("p1-ts2", "s4-p1-ts2"),
        ("own-ts1", "g1-own-ts1"), ("own-ts2", "g1-own-ts2")]
SPLITS = ["test_iid", "test_comp", "test_morph", "test_long"]
TOKDIR = KAG / "s4-hard-shuffledinit-50k-p1" / "model"


def load(run):
    return json.loads((KAG / run / "results" / f"{run}.json").read_text(encoding="utf-8"))


def glyphs(run):
    f = KAG / run / "model" / "glyph_tokens.pt"
    return torch.load(f, weights_only=False) if f.exists() else None


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOKDIR)
    data = build(PRESETS["tamil_like"], "abugida_pua", n_train=50000, n_test=500, seed=0, n_extra=0)
    res = {k: load(r) for k, r in RUNS}
    gt = glyphs(RUNS[2][1])
    chars, sounds = gt["chars"], gt["sounds"]
    name = {c: (s or "virama") for c, s in zip(chars, sounds)}            # glyph -> its real sound (label)
    start = {k: dict(zip(chars, res[k]["data"]["donor_tokens"])) for k, _ in RUNS}
    L = ["# Why each wrong-sound start fails on a different test (`analysis/draw_errors.py`)", "",
         "Own sound vs three wrong-sound draws (same language, data, seed, test items). Glyphs are named by their "
         "REAL sound; 'start' = the sound the run's embedding started from. Teacher-forced token errors unless "
         "stated.", ""]

    # ---- 1. errors per glyph and per position in the word, from per_item_ok
    per = {}      # (run, split) -> glyph -> [n, err]
    pos = {}      # (run, split) -> position-from-word-end bucket -> [n, err]
    first = {}    # (run, split) -> list of (first error index / length) for failed items
    errs_at = {}  # (run, split) -> token index bucket -> [n, err]
    for k, run in RUNS:
        for s in SPLITS:
            tf = res[k]["eval"][s]["teacher_forced"]
            items = data["splits"][s]
            assert [it["id"] for it in items] == tf["per_item_ids"]
            g, p, ea = (collections.defaultdict(lambda: [0, 0]) for _ in range(3))
            f = []
            for it, ok in zip(items, tf["per_item_ok"]):
                enc = tok(it["target"], add_special_tokens=False, return_offsets_mapping=True)
                offs = enc["offset_mapping"]
                assert len(offs) + 1 == len(ok), (len(offs), len(ok))
                tgt = it["target"]
                # position of each char from the end of its word (0 = last glyph of the word)
                from_end = {}
                st = 0
                for w in tgt.split(" "):
                    for i in range(len(w)):
                        from_end[st + i] = len(w) - 1 - i
                    st += len(w) + 1
                for ti, ((a, b), o) in enumerate(zip(offs, ok)):
                    bad = o == "0"
                    ea[min(ti // 10, 9)][0] += 1
                    ea[min(ti // 10, 9)][1] += bad
                    for ci in range(a, b):
                        ch = tgt[ci]
                        if ch == " ":
                            continue
                        g[ch][0] += 1
                        g[ch][1] += bad
                        pb = min(from_end.get(ci, 0), 3)
                        p[pb][0] += 1
                        p[pb][1] += bad
                if "0" in ok:
                    f.append(ok.index("0") / len(ok))
            per[k, s], pos[k, s], first[k, s], errs_at[k, s] = g, p, f, ea

    vch = next(c for c in chars if name[c] == "v")
    L += ["## 0. Summary per run (teacher-forced exact; glide = error rate on glyph v in test_morph)", "",
          "| run | start of v | iid | comp | morph | long | v error (morph) |", "|---|---|---|---|---|---|---|"]
    for k, _ in RUNS:
        ex = [res[k]["eval"][s]["teacher_forced"]["exact_match"] for s in SPLITS]
        d = per[k, "test_morph"][vch]
        L.append(f"| {k} | {start[k][vch] or '-'} | " + " | ".join(f"{x:.3f}" for x in ex) +
                 f" | {d[1] / d[0]:.3f} |")
    L.append("")
    L += ["## 1. Error rate by position in the word (teacher-forced, per glyph token)", "",
          "0 = last glyph of a word (where case/tense endings sit), 3 = fourth from the end or earlier.", ""]
    for s in ("test_morph", "test_long"):
        L += [f"**{s}**", "", "| run | last | 2nd last | 3rd last | earlier |", "|---|---|---|---|---|"]
        for k, _ in RUNS:
            d = pos[k, s]
            L.append(f"| {k} | " + " | ".join(f"{d[b][1] / d[b][0]:.3f}" if d[b][0] else "-" for b in range(4)) + " |")
        L.append("")

    L += ["## 2. Where in a long sentence the first error falls (test_long)", "",
          "Error rate by token index (10-token bins), and the first error's relative position in failed items.", "",
          "| run | " + " | ".join(f"{10 * b}-{10 * b + 9}" if b < 9 else "90+" for b in range(10)) +
          " | failed items | median first error at |", "|---|" + "---|" * 12]
    for k, _ in RUNS:
        d, f = errs_at[k, "test_long"], sorted(first[k, "test_long"])
        L.append(f"| {k} | " + " | ".join(f"{d[b][1] / d[b][0]:.3f}" if d[b][0] else "-" for b in range(10)) +
                 f" | {len(f)} | {f[len(f) // 2]:.2f} |" if f else f" | 0 | - |")
    L.append("")

    L += ["## 3. Error rate per glyph (teacher-forced)", "",
          "Glyphs with an error rate of at least 0.02 in some wrong draw on test_morph or test_long; the "
          "start sound each draw gave that glyph in brackets.", ""]
    for s in ("test_morph", "test_long"):
        rows = []
        for c in chars:
            rates = {k: (per[k, s][c][1] / per[k, s][c][0] if per[k, s][c][0] else None) for k, _ in RUNS}
            if any(v is not None and v >= 0.02 for kk, v in rates.items() if kk != "own"):
                rows.append((max(v or 0 for v in rates.values()), c, rates))
        L += [f"**{s}**", "", "| glyph (real sound) | uses | " + " | ".join(k for k, _ in RUNS) + " |", "|---|---|" + "---|" * len(RUNS)]
        for _, c, rates in sorted(rows, reverse=True):
            cells = []
            for k, _ in RUNS:
                v = rates[k]
                lab = "" if k == "own" else f" [{start[k][c] or '-'}]"
                cells.append("-" if v is None else f"{v:.3f}{lab}")
            L.append(f"| {name[c]} | {per['own', s][c][0]} | " + " | ".join(cells) + " |")
        L.append("")

    # ---- 4. confusions in generation (gold vs pred, aligned)
    L += ["## 4. Which glyph is written instead of which (generation, 100 items per test)", "",
          "Character substitutions from aligning gold and predicted text (difflib); the most frequent per run.", ""]
    conf = {}
    for k, _ in RUNS:
        for s in ("test_morph", "test_long"):
            c = collections.Counter()
            drops = collections.Counter()
            for smp in res[k]["eval"][s]["generation"]["samples"]:
                a, b = smp["gold"], smp["pred"]
                for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
                    if op == "replace" and i2 - i1 == j2 - j1:
                        for x, y in zip(a[i1:i2], b[j1:j2]):
                            c[(x, y)] += 1
                    elif op == "delete":
                        for x in a[i1:i2]:
                            drops[x] += 1
            conf[k, s] = (c, drops)
    for s in ("test_morph", "test_long"):
        L += [f"**{s}**", "", "| run | top substitutions (gold -> written, count) | top dropped glyphs |", "|---|---|---|"]
        for k, _ in RUNS:
            c, drops = conf[k, s]
            sub = ", ".join(f"{name.get(x, repr(x))}->{name.get(y, repr(y))} {n}" for (x, y), n in c.most_common(6))
            dr = ", ".join(f"{name.get(x, repr(x))} {n}" for x, n in drops.most_common(4))
            L.append(f"| {k} | {sub or '-'} | {dr or '-'} |")
        L.append("")

    # ---- 5. embeddings: were the confused glyphs close at the start, and at the end?
    L += ["## 5. Start and final embeddings of the glyphs", "",
          "Cosine similarity between glyph embeddings. 'start' = delta_init, 'final' = trained (saved "
          "glyph_tokens.pt). For each run: the most similar pairs of glyphs at the start and at the end.", ""]
    for k, run in RUNS:
        gk = glyphs(run)
        if gk is None:
            L.append(f"- {k}: no glyph_tokens.pt")
            continue
        for when in ("delta_init", "delta"):
            e = torch.nn.functional.normalize(gk[when].float(), dim=1)
            sim = e @ e.T
            pairs = [(float(sim[i, j]), i, j) for i in range(len(chars)) for j in range(i + 1, len(chars))]
            top = sorted(pairs, reverse=True)[:5]
            L.append(f"- {k}, {'start' if when == 'delta_init' else 'final'}: " +
                     ", ".join(f"{name[chars[i]]}~{name[chars[j]]} {v:.2f}" for v, i, j in top))
        # do the top generation confusions involve pairs that ended close?
        e = torch.nn.functional.normalize(gk["delta"].float(), dim=1)
        e0 = torch.nn.functional.normalize(gk["delta_init"].float(), dim=1)
        idx = {c: i for i, c in enumerate(chars)}
        for s in ("test_morph", "test_long"):
            c, _ = conf[k, s]
            row = []
            for (x, y), n in c.most_common(4):
                if x in idx and y in idx:
                    row.append(f"{name[x]}->{name[y]} ({n}): start {float(e0[idx[x]] @ e0[idx[y]]):.2f}, "
                               f"final {float(e[idx[x]] @ e[idx[y]]):.2f}")
            if row:
                L.append(f"  - {s} confusions: " + "; ".join(row))
    L.append("")
    out = ROOT / "results" / "draw_errors.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
