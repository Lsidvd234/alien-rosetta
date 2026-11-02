"""Does adding a word's letter vectors give the word's vector? (01 Nov 2026, user's check)

    python analysis/letters_sum_to_word.py   -> results/letters_sum_to_word.md

Words = the 27 word pieces s4-randtok started its glyphs from (base-model token embeddings, e.g. 'Extra').
Each word is spelled in letters (lower case, leading space dropped) and its letter vectors are added:
  - English reference: the base model's single-letter tokens ('e', 'x', 't', 'r', 'a');
  - own-sound model (s1-hard-latininit-50k): the glyph whose sound is that letter (start and trained vector);
  - wrong-sound model (p1): the glyph that STARTED as that letter (start and trained vector).
The glyph inventory has 17 consonant and 9 vowel sounds; letters with no glyph (x, s, g, ...) are left out of the glyph
sums, and the share of letters covered is reported. Two-letter sounds (aa, ny, ...) are matched first.

Score: cosine(letter sum, word vector), all vectors centred on the base mean token embedding. Chance level: the same
letter sum against the OTHER 26 words' vectors (mean). Also: is the word's own vector the nearest of the 27?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
KAG = ROOT / "results" / "kaggle"
BASE = Path("D:/Downloads/wm-feasibility/hf_cache/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/"
            "da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
N_OLD = 151669


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def spell(word, inventory):
    """Split into the glyph sounds (longest match first); letters with no glyph -> None."""
    w, out, i = word.lower(), [], 0
    while i < len(w):
        two = w[i:i + 2]
        if len(two) == 2 and two in inventory:
            out.append(two); i += 2
        elif w[i] in inventory:
            out.append(w[i]); i += 1
        else:
            out.append(None); i += 1
    return out


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(BASE))
    with safe_open(str(BASE / "model.safetensors"), "pt") as f:
        key = next(k for k in f.keys() if k.endswith("embed_tokens.weight"))
        W = f.get_tensor(key).float()[:N_OLD].numpy()
    mu = W.mean(0)
    rt = torch.load(KAG / "s4-randtok" / "model" / "glyph_tokens.pt", weights_only=False)
    words = json.loads((KAG / "s4-randtok" / "results" / "s4-randtok.json").read_text(encoding="utf-8"))["data"][
        "donor_tokens"]
    wordvec = rt["delta_init"].numpy() - mu                     # = the base embedding of each word piece
    sounds = [s or "" for s in rt["sounds"]]
    inv = {s for s in sounds if s}

    def glyph_table(run):
        g = torch.load(KAG / run / "model" / "glyph_tokens.pt", weights_only=False)
        d = json.loads((KAG / run / "results" / f"{run}.json").read_text(encoding="utf-8"))["data"]
        start_letter = d.get("donor_tokens") or sounds          # latin runs store each glyph's start sound
        by_letter = {}
        for i, s in enumerate(start_letter):
            if s:
                by_letter[s] = i
        return by_letter, g["delta_init"].numpy() - mu, g["delta"].numpy() - mu

    own_map, own_s, own_f = glyph_table("s1-hard-latininit-50k")
    p1_map, p1_s, p1_f = glyph_table("s4-hard-shuffledinit-50k-p1")

    def eng_letter(c):
        ids = tok(c, add_special_tokens=False)["input_ids"]
        return W[ids].mean(0) - mu

    rows, sums = [], {k: [] for k in ("English letters", "own start", "own trained", "p1 start", "p1 trained")}
    for wi, word in enumerate(words):
        core = word.strip()
        letters = [c.lower() for c in core if c.isalpha()]
        parts = spell(core, inv)
        cover = sum(p is not None for p in parts) / max(1, len(parts))
        sums["English letters"].append(np.sum([eng_letter(c) for c in letters], 0) if letters else np.zeros_like(mu))
        for lab, mp, S, F in (("own", own_map, own_s, own_f), ("p1", p1_map, p1_s, p1_f)):
            idx = [mp[p] for p in parts if p is not None and p in mp]
            sums[f"{lab} start"].append(S[idx].sum(0) if idx else np.zeros_like(mu))
            sums[f"{lab} trained"].append(F[idx].sum(0) if idx else np.zeros_like(mu))
        rows.append((word, "-".join(p or "_" for p in parts), cover))

    Wv = unit(wordvec)
    L = ["# Does adding a word's letter vectors give the word's vector? (`analysis/letters_sum_to_word.py`)", "",
         "Words: the 27 word pieces s4-randtok started its glyphs from. Letter sum vs the word's base embedding, "
         "cosine (centred). `_` in the spelling = a letter with no glyph (left out of the glyph sums).", "",
         "## Summary", "",
         "| letters taken from | mean cos(letter sum, its word) | chance (vs the other 26 words) | "
         "own word nearest of 27 |", "|---|---|---|---|"]
    per = {}
    for k, v in sums.items():
        Sv = np.stack(v)
        ok = np.linalg.norm(Sv, axis=1) > 0
        C = np.zeros((27, 27))
        C[ok] = unit(Sv[ok]) @ Wv.T
        own = np.diag(C)
        other = (C.sum(1) - own) / 26
        hit = sum(int(np.argmax(C[i]) == i) for i in range(27) if ok[i])
        per[k] = own
        L.append(f"| {k} | {own[ok].mean():.3f} | {other[ok].mean():.3f} | {hit} / {int(ok.sum())} |")
    L += ["", "## Per word", "",
          "| word | glyph spelling | letters covered | English | own start | own trained | p1 start | "
          "p1 trained |", "|---|---|---|---|---|---|---|---|"]
    for i, (w, sp, cov) in enumerate(rows):
        L.append(f"| {w!r} | {sp} | {cov:.0%} | " + " | ".join(f"{per[k][i]:.2f}" for k in sums) + " |")
    L.append("")
    out = ROOT / "results" / "letters_sum_to_word.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
