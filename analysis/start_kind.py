"""Are the word-piece starts a different KIND of vector from the letter starts? (01 Nov 2026, user's check)

    python analysis/start_kind.py   -> results/start_kind.md

Uses only the base model's input embedding table (Qwen3-0.6B-Base, CPU) and the saved glyph starts (delta_init) of
s1-hard-latininit-50k (own letters), s4-hard-shuffledinit-50k-p1 (wrong letters), s4-randtok (random word pieces)
and s4-random (random vectors). All vectors centred on the mean token embedding.

1. Similarity within and between the start sets (mean pairwise cosine).
2. Nearest tokens of each start in the whole vocabulary.
3. A "letter vs word" probe: logistic regression trained on base-vocabulary tokens that are short spelling pieces
   (1-2 ASCII letters, with or without a leading space) vs whole words (5+ ASCII letters), excluding every token used
   as a start; applied to each start set -> mean probability "spelling piece".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
KAG = ROOT / "results" / "kaggle"
BASE = Path("D:/Downloads/wm-feasibility/hf_cache/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/"
            "da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
SETS = [("own letters", "s1-hard-latininit-50k"), ("wrong letters (p1)", "s4-hard-shuffledinit-50k-p1"),
        ("random word pieces", "s4-randtok"), ("random vectors", "s4-random")]
N_OLD = 151669


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(BASE))
    with safe_open(str(BASE / "model.safetensors"), "pt") as f:
        key = next(k for k in f.keys() if k.endswith("embed_tokens.weight"))
        W = f.get_tensor(key).float()[:N_OLD].numpy()
    mu = W.mean(0)
    Wc = W - mu
    Wu = unit(Wc)
    starts, names = {}, None
    for lab, run in SETS:
        g = torch.load(KAG / run / "model" / "glyph_tokens.pt", weights_only=False)
        names = [s or "virama" for s in g["sounds"]]
        starts[lab] = g["delta_init"].numpy() - mu
    # donor token ids (to exclude from the probe's training set)
    text = [tok.decode([i]) for i in range(N_OLD)]
    used = set()
    for lab, run in SETS[:3]:
        import json
        d = json.loads((KAG / run / "results" / f"{run}.json").read_text(encoding="utf-8"))["data"]["donor_tokens"]
        for s in d:
            if s:
                used.update(tok(s, add_special_tokens=False)["input_ids"])
    L = ["# Are word-piece starts a different kind of vector from letter starts? (`analysis/start_kind.py`)", "",
         "Base model input embeddings (Qwen3-0.6B-Base), centred on the mean token. Starts = each run's saved "
         "glyph start vectors (27 glyphs).", ""]

    # 1. within / between
    labs = [l for l, _ in SETS]
    U = {l: unit(starts[l]) for l in labs}
    L += ["## 1. Mean cosine between start vectors", "",
          "Diagonal: two different glyphs of the same run. Off-diagonal: any glyph of one run vs any glyph of the "
          "other.", "", "| | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs)]
    for a in labs:
        row = []
        for b in labs:
            C = U[a] @ U[b].T
            row.append(C[~np.eye(27, dtype=bool)].mean() if a == b else C.mean())
        L.append(f"| {a} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    L.append("")

    # 2. nearest vocabulary tokens
    L += ["## 2. Nearest base-vocabulary tokens of each start (cosine)", ""]
    for lab in labs:
        L += [f"**{lab}**", ""]
        sims = U[lab] @ Wu.T
        for gi in [names.index(s) for s in ("k", "aa", "t", "i", "m", "zh")]:
            top = np.argsort(-sims[gi])[:6]
            L.append(f"- glyph {names[gi]}: " + ", ".join(f"{text[j]!r} {sims[gi, j]:.2f}" for j in top))
        L.append("")

    # 3. probe: short spelling piece vs whole word
    def kind(i):
        s = text[i]
        core = s[1:] if s.startswith(" ") else s
        if not core.isascii() or not core.isalpha() or i in used:
            return None
        if 1 <= len(core) <= 2:
            return 1
        if len(core) >= 5:
            return 0
        return None
    ks = [(i, kind(i)) for i in range(N_OLD)]
    pos = [i for i, k in ks if k == 1]
    neg = [i for i, k in ks if k == 0]
    rng = np.random.default_rng(0)
    neg = list(rng.choice(neg, size=min(len(neg), 4 * len(pos)), replace=False))
    X = Wu[pos + neg]
    y = np.array([1] * len(pos) + [0] * len(neg))
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    acc = cross_val_score(clf, X, y, cv=5, scoring="balanced_accuracy").mean()
    clf.fit(X, y)
    L += ["## 3. Letter-vs-word probe", "",
          f"Logistic regression on base-vocabulary tokens: {len(pos)} spelling pieces (1-2 letters) vs {len(neg)} "
          f"whole words (5+ letters), start tokens excluded. Held-out balanced accuracy {acc:.3f}.", "",
          "| start set | mean P(spelling piece) | glyphs judged spelling piece (P > 0.5) |", "|---|---|---|"]
    for lab in labs:
        p = clf.predict_proba(U[lab])[:, 1]
        L.append(f"| {lab} | {p.mean():.3f} | {int((p > 0.5).sum())} / 27 |")
    L.append("")
    out = ROOT / "results" / "start_kind.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
