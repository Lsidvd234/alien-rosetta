"""Inside the network: are glyphs combined into words (spelling), or does each glyph stay its own unit? (01 Nov 2026)

    python analysis/hidden_word_assembly.py [N_PER_STEM]   -> results/hidden_word_assembly.md (+ .png)

Models (same setting, only the glyph start differs): own letters (s1-hard-latininit-50k), wrong letters
(s4-hard-shuffledinit-50k-p1), random word pieces (s4-randtok), random vectors (s4-random). CPU, fp32.

Input: training sentences that contain one of 40 frequent content stems, given with an EMPTY English line
("English: \\nTarget:\\n" + glyph text), so whatever the model knows about a word must come from its glyphs.

1. Word assembly. At the LAST glyph of the word (an ending glyph, shared by many words) take the hidden state at each
   layer and train a linear probe (5-fold) to name the stem (40 classes). Layer 0 = the glyph's own embedding (chance:
   the last glyph alone). If accuracy rises through the layers, the model is putting the word together from its glyphs.
2. Within-word attention. For each glyph that is not first in its word: attention (mean over heads) per glyph on the
   earlier glyphs of the SAME word, divided by attention per glyph on the glyphs of the PREVIOUS word. > 1 = the model
   reads glyphs together with their own word (spelling); about 1 = no preference for its own word.
3. English-word mode (logit lens). At each glyph, the hidden state after the final norm, scored against the base
   vocabulary only (glyph tokens excluded): share of the top 10 tokens that are whole words (3+ ASCII letters) vs
   letter pieces (1-2 letters). Tells what kind of English token the hidden state looks like, layer by layer.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from conlang.dataset import PRESETS, build  # noqa: E402
from train.load import load_trained  # noqa: E402

KAG = ROOT / "results" / "kaggle"
BASE = "D:/Downloads/wm-feasibility/hf_cache/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/" \
       "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
MODELS = [("own letters", "s1-hard-latininit-50k"), ("wrong letters", "s4-hard-shuffledinit-50k-p1"),
          ("random word pieces", "s4-randtok"), ("random vectors", "s4-random")]
PRONOUNS = {"1sg", "2sg", "3sgm", "3sgf", "3sgn", "1pl", "2pl", "3pl"}
N_STEMS = 40
PREFIX = "English: \nTarget:\n"
LENS_LAYERS = {0, 1, 2, 4, 6, 8, 12, 16, 20, 24}   # logit lens only here (+ last layer): it is the slow part


def pick_items(n_per):
    data = build(PRESETS["tamil_like"], "abugida_pua", n_train=50000, n_test=500, seed=0, n_extra=0)
    tr = data["splits"]["train"]
    c = collections.Counter(w.split("-")[0] for it in tr for w in it["gloss"].split())
    stems = [s for s, _ in c.most_common() if s not in PRONOUNS][:N_STEMS]
    sid = {s: i for i, s in enumerate(stems)}
    got = collections.Counter()
    items = []                                        # (target, word index in sentence, stem id)
    for it in tr:
        g = it["gloss"].split()
        for wi, w in enumerate(g):
            s = w.split("-")[0]
            if s in sid and got[s] < n_per:
                items.append((it["target"], wi, sid[s]))
                got[s] += 1
                break
    return items, stems


def word_kind(s):
    core = s[1:] if s.startswith(" ") else s
    if not core.isascii() or not core.isalpha():
        return None
    return "letter" if len(core) <= 2 else "word"


@torch.no_grad()
def run_model(run, items):
    model, tok, _ = load_trained(KAG / run / "model", device="cpu", base_model=BASE)
    inner = model.base_model.model
    inner.config._attn_implementation = "eager"
    for m in inner.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"
    norm = inner.model.norm
    W = inner.lm_head.base.weight                      # base vocabulary rows (tied)
    n_old = inner.lm_head.start
    kinds = [word_kind(tok.decode([i])) for i in range(n_old)]
    is_word = torch.tensor([k == "word" for k in kinds])
    is_letter = torch.tensor([k == "letter" for k in kinds])
    pre = tok(PREFIX, add_special_tokens=False)["input_ids"]
    feats = None
    ys = []
    att_in, att_prev = None, None
    lens_word, lens_letter, lens_n = None, None, 0
    for target, wi, y in items:
        enc = tok(target, add_special_tokens=False, return_offsets_mapping=True)
        ids = pre + enc["input_ids"]
        # word index of every target token (spaces -> None)
        starts = [0]
        for i, ch in enumerate(target):
            if ch == " ":
                starts.append(i + 1)
        def widx(a):
            return sum(1 for s in starts if s <= a) - 1
        wof = [None if target[a:b] == " " else widx(a) for a, b in enc["offset_mapping"]]
        pos = [len(pre) + k for k, w in enumerate(wof) if w is not None]
        wpos = {w: [len(pre) + k for k, ww in enumerate(wof) if ww == w] for w in set(x for x in wof if x is not None)}
        out = model(input_ids=torch.tensor([ids]), output_hidden_states=True, output_attentions=True)
        hs = out.hidden_states                          # L+1 x [1, T, d]
        nl = len(hs)
        if feats is None:
            feats = [[] for _ in range(nl)]
            att_in, att_prev = np.zeros(nl - 1), np.zeros(nl - 1)
            lens_word, lens_letter = np.zeros(nl), np.zeros(nl)
        last = wpos[wi][-1]
        for l in range(nl):
            feats[l].append(hs[l][0, last].float().numpy())
        ys.append(y)
        # 2. attention: within the word vs the previous word, per glyph
        att = torch.stack(out.attentions)[:, 0].mean(1)  # [layers, T, T], mean over heads
        for w, ps in wpos.items():
            prev = wpos.get(w - 1)
            if not prev:
                continue
            for j in range(1, len(ps)):
                q = ps[j]
                att_in += att[:, q, ps[:j]].mean(-1).numpy()
                att_prev += att[:, q, prev].mean(-1).numpy()
        # 3. logit lens on every glyph position
        for l in range(nl):
            if l not in LENS_LAYERS and l != nl - 1:
                continue
            h = norm(hs[l][0, pos]) if l < nl - 1 else hs[l][0, pos]
            top = (h @ W[:n_old].T).topk(10, dim=-1).indices
            lens_word[l] += float(is_word[top].float().mean(-1).sum())
            lens_letter[l] += float(is_letter[top].float().mean(-1).sum())
        lens_n += len(pos)
    ys = np.array(ys)
    acc = []
    for l in range(nl):
        X = np.stack(feats[l])
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        clf = LogisticRegression(max_iter=2000, C=0.1)
        acc.append(cross_val_score(clf, X, ys, cv=5).mean())
    del model
    return np.array(acc), att_in / np.maximum(att_prev, 1e-12), lens_word / lens_n, lens_letter / lens_n


def main() -> None:
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    items, stems = pick_items(n_per)
    res = {}
    for lab, run in MODELS:
        print(f"{lab} ({run}): {len(items)} sentences", flush=True)
        res[lab] = run_model(run, items)
    nl = len(res[MODELS[0][0]][0])
    show = sorted(set([0, 1, 2, 4, 6, 8, 12, 16, 20, 24, nl - 1]))
    L = ["# Inside the network: are glyphs put together into words? (`analysis/hidden_word_assembly.py`)", "",
         f"{len(items)} training sentences, each with one of {N_STEMS} frequent content stems ({n_per} per stem), "
         "fed with an EMPTY English line, so word identity can only come from the glyphs. CPU, fp32.", "",
         f"## 1. Can the stem be read from the word's LAST glyph? (linear probe, {N_STEMS} classes, chance "
         f"{1 / N_STEMS:.3f})", "",
         "Layer 0 = the last glyph's own embedding (an ending glyph shared by many words).", "",
         "| model | " + " | ".join(f"L{l}" for l in show) + " |", "|---|" + "---|" * len(show)]
    for lab, _ in MODELS:
        L.append(f"| {lab} | " + " | ".join(f"{res[lab][0][l]:.2f}" for l in show) + " |")
    sa = [l for l in show if l < nl - 1]
    L += ["", "## 2. Attention to the glyph's own word vs the previous word (ratio per glyph, mean over heads)", "",
          "| model | " + " | ".join(f"L{l}" for l in sa) + " | mean all layers |", "|---|" + "---|" * (len(sa) + 1)]
    for lab, _ in MODELS:
        r = res[lab][1]
        L.append(f"| {lab} | " + " | ".join(f"{r[l]:.2f}" for l in sa) + f" | {r.mean():.2f} |")
    L += ["", "## 3. Logit lens: what kind of English token does a glyph's hidden state look like?", "",
          "Share of the top-10 base-vocabulary tokens that are whole words (3+ letters) / letter pieces (1-2).", "",
          "| model | " + " | ".join(f"L{l}" for l in show) + " |", "|---|" + "---|" * len(show)]
    for lab, _ in MODELS:
        w, le = res[lab][2], res[lab][3]
        L.append(f"| {lab} | " + " | ".join(f"{w[l]:.2f}/{le[l]:.2f}" for l in show) + " |")
    L.append("")
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    for lab, _ in MODELS:
        axs[0].plot(res[lab][0], label=lab)
        axs[1].plot(res[lab][1], label=lab)
        axs[2].plot(res[lab][2], label=lab)
    axs[0].axhline(1 / N_STEMS, color="grey", ls=":")
    axs[0].set_title("1. stem readable from the last glyph (probe accuracy)")
    axs[1].axhline(1, color="grey", ls=":")
    axs[1].set_title("2. attention: own word / previous word")
    axs[2].set_title("3. logit lens: share of whole-word tokens in top 10")
    for ax in axs:
        ax.set_xlabel("layer"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "results" / "hidden_word_assembly.png", dpi=130); plt.close(fig)
    L.append("Plot: `results/hidden_word_assembly.png`.")
    (ROOT / "results" / "hidden_word_assembly.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
