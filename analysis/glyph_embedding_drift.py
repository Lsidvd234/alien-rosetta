"""How far do the glyph embeddings move in training, and do different starts end in the same place? (31 Oct 2026)

    python analysis/glyph_embedding_drift.py   -> results/glyph_embedding_drift.md + results/glyph_embeddings/*.png

The 27 glyph tokens each have one trainable embedding row (1024 numbers, Qwen3-0.6B-Base). glyph_tokens.pt keeps the
start (delta_init) and the trained value (delta). All runs here share model, language (seed 0), data and the 50k / 3
epoch hard setting; they differ only in where the glyphs start (own sound, wrong sound, copied token, random) and the
training seed. (The model has no learned position embedding (RoPE); these are the glyph TOKEN embeddings.)

Vectors are centred on the base model's mean token embedding before any cosine, so the shared offset every token has
does not inflate the similarities.

Measures:
  1. movement: per glyph cos(start, final) and |final - start| / |start|;
  2. same place? per glyph cos(final in run A, final in run B), averaged over glyphs -> run x run;
  3. same shape? correlation of the 27x27 glyph-glyph similarity matrices of two runs (ignores where the cloud
     sits, only how the glyphs are arranged relative to each other) -> run x run;
  4. for wrong-sound runs: is a glyph's final value nearer its wrong start or its OWN sound's start?
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KAG = ROOT / "results" / "kaggle"
OUT = ROOT / "results" / "glyph_embeddings"
BASE = Path("D:/Downloads/wm-feasibility/hf_cache/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/"
            "da87bfb608c14b7cf20ba1ce41287e8de496c0cd/model.safetensors")
# (label, run, kind)
RUNS = [("own", "s1-hard-latininit-50k", "own sound"),
        ("own-ts1", "g1-own-ts1", "own sound"),
        ("own-ts2", "g1-own-ts2", "own sound"),
        ("own-vswap", "s4-own-vswap", "own, v<->r"),
        ("draw0", "s4-hard-shuffledinit-50k", "wrong sound"),
        ("p1", "s4-hard-shuffledinit-50k-p1", "wrong sound"),
        ("p1-ts1", "s4-p1-ts1", "wrong sound"),
        ("p1-ts2", "s4-p1-ts2", "wrong sound"),
        ("p2", "s4-hard-shuffledinit-50k-p2", "wrong sound"),
        ("p2-vfix", "s4-p2-vfix", "wrong, v fixed"),
        ("copy-tr", "s2-hard-translit-50k", "copied token"),
        ("copy-pl", "s3-hard-plaintext-50k", "copied token"),
        ("random", "s4-random", "random"),
        ("randtok", "s4-randtok", "copied token")]   # 31 Oct: random real word-piece start, s4 setting


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with safe_open(str(BASE), "pt") as f:
        key = next(k for k in f.keys() if k.endswith("embed_tokens.weight"))
        mu = f.get_tensor(key).float()[:151669].mean(0).numpy()
    runs, S, F = [], {}, {}
    chars = sounds = None
    for lab, run, kind in RUNS:
        p = KAG / run / "model" / "glyph_tokens.pt"
        if not p.exists():
            print(f"skip {lab}: no {p}")
            continue
        g = torch.load(p, weights_only=False)
        if chars is None:
            chars, sounds = g["chars"], g["sounds"]
        assert g["chars"] == chars
        S[lab] = g["delta_init"].numpy() - mu
        F[lab] = g["delta"].numpy() - mu
        runs.append((lab, run, kind))
    labs = [r[0] for r in runs]
    TAG = "_with_random_randtok" if "randtok" in labs else ("_with_random" if "random" in labs else "")
    names = [s or "virama" for s in sounds]
    n = len(labs)
    L = ["# Glyph embeddings: start vs after training, and across runs (`analysis/glyph_embedding_drift.py`)", "",
         "27 glyph token embeddings (1024-d), centred on the base model's mean token embedding. Same model, language, "
         "data and setting in every run; only the start and the training seed differ.", ""]

    # 1. movement
    cos_sf = {l: (unit(S[l]) * unit(F[l])).sum(1) for l in labs}
    rel = {l: np.linalg.norm(F[l] - S[l], axis=1) / np.linalg.norm(S[l], axis=1) for l in labs}
    L += ["## 1. How much each run's glyphs moved", "",
          "cos(start, final): 1 = did not turn at all, 0 = ended unrelated to where it started. Change = "
          "|final - start| / |start|. Mean over the 27 glyphs (min-max in brackets).", "",
          "| run | start | cos(start, final) | change |", "|---|---|---|---|"]
    for lab, run, kind in runs:
        c, r = cos_sf[lab], rel[lab]
        L.append(f"| {lab} | {kind} | {c.mean():.3f} ({c.min():.2f}-{c.max():.2f}) | {r.mean():.3f} "
                 f"({r.min():.2f}-{r.max():.2f}) |")
    L.append("")
    fig, ax = plt.subplots(figsize=(13, 0.42 * n + 1.5))
    im = ax.imshow(np.stack([cos_sf[l] for l in labs]), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(27), names, rotation=90); ax.set_yticks(range(n), labs)
    ax.set_title("cos(start, final) per glyph: 1 = stayed where it started")
    fig.colorbar(im, ax=ax, fraction=0.02); fig.tight_layout(); fig.savefig(OUT / f"1_movement{TAG}.png", dpi=130)
    plt.close(fig)

    # 2. same place: per glyph final-final cosine, averaged; and the same for starts
    def pairmean(D):
        M = np.zeros((n, n))
        for i, a in enumerate(labs):
            for j, b in enumerate(labs):
                M[i, j] = (unit(D[a]) * unit(D[b])).sum(1).mean()
        return M
    Ms, Mf = pairmean(S), pairmean(F)
    # 3. same shape: correlation of glyph-glyph similarity matrices (off-diagonal)
    iu = np.triu_indices(27, 1)
    G = {l: (unit(F[l]) @ unit(F[l]).T)[iu] for l in labs}
    G0 = {l: (unit(S[l]) @ unit(S[l]).T)[iu] for l in labs}
    Mg = np.array([[np.corrcoef(G[a], G[b])[0, 1] for b in labs] for a in labs])
    Mg0 = np.array([[np.corrcoef(G0[a], G0[b])[0, 1] for b in labs] for a in labs])
    fig, axs = plt.subplots(2, 2, figsize=(15, 13))
    for ax, M, t in [(axs[0, 0], Ms, "SAME PLACE at the start\n(mean over glyphs of cos(glyph in A, same glyph in B))"),
                     (axs[0, 1], Mf, "SAME PLACE after training"),
                     (axs[1, 0], Mg0, "SAME SHAPE at the start\n(correlation of glyph-glyph similarity matrices)"),
                     (axs[1, 1], Mg, "SAME SHAPE after training")]:
        im = ax.imshow(M, vmin=-0.2, vmax=1, cmap="magma")
        ax.set_xticks(range(n), labs, rotation=90); ax.set_yticks(range(n), labs); ax.set_title(t)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] < 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(OUT / f"2_same_place_same_shape{TAG}.png", dpi=130); plt.close(fig)

    def table(M, title, note):
        out = [f"## {title}", "", note, "", "| | " + " | ".join(labs) + " |", "|---|" + "---|" * n]
        for i, a in enumerate(labs):
            out.append(f"| {a} | " + " | ".join(f"{M[i, j]:.2f}" for j in range(n)) + " |")
        return out + [""]
    L += table(Mf, "2. Same place after training?",
               "Mean over glyphs of cos(glyph's final vector in run A, same glyph's final vector in run B). "
               "1 = identical; runs that started at the same place sit high just because they started there, "
               "so compare with the start matrix (in the PNG).")
    L += table(Mg, "3. Same shape after training?",
               "Correlation between two runs' 27x27 glyph-to-glyph similarity matrices: 1 = the glyphs are arranged "
               "the same way relative to each other, wherever the cloud sits.")

    # summary by group pair
    grp = {l: k for l, _, k in runs}
    def gmean(M, A, B):
        v = [M[i, j] for i, a in enumerate(labs) for j, b in enumerate(labs)
             if i < j and {grp[a], grp[b]} == {A, B} or (i < j and A == B and grp[a] == grp[b] == A)]
        return (np.mean(v), len(v)) if v else (np.nan, 0)
    kinds = ["own sound", "wrong sound", "copied token", "random"]
    L += ["## Summary: average over run pairs (start -> after training)", "",
          "| pair of runs | n pairs | same place: start -> final | same shape: start -> final |",
          "|---|---|---|---|"]
    for a_i, A in enumerate(kinds):
        for B in kinds[a_i:]:
            (s0, k), (f0, _), (g0, _), (g1, _) = (gmean(Ms, A, B), gmean(Mf, A, B), gmean(Mg0, A, B),
                                                  gmean(Mg, A, B))
            if k:
                L.append(f"| {A} vs {B} | {k} | {s0:.2f} -> {f0:.2f} | {g0:.2f} -> {g1:.2f} |")
    L.append("")

    # 4. wrong-sound runs: nearer own sound's start or own wrong start?
    if "own" in S:
        L += ["## 4. Wrong-sound runs: does a glyph drift toward its REAL sound?", "",
              "Per glyph: cos(final, the start it was given) vs cos(final, its own sound's start "
              "(= the own run's start)). Glyphs whose given start already equals the own start are left out.", "",
              "| run | glyphs | cos to given (wrong) start | cos to own-sound start | glyphs nearer own sound |",
              "|---|---|---|---|---|"]
        for lab, run, kind in runs:
            diff = ~np.isclose(S[lab], S["own"]).all(1)
            if lab == "own" or diff.sum() == 0:
                continue
            fu = unit(F[lab][diff])
            cg = (fu * unit(S[lab][diff])).sum(1)
            co = (fu * unit(S["own"][diff])).sum(1)
            L.append(f"| {lab} | {int(diff.sum())} | {cg.mean():.3f} | {co.mean():.3f} | "
                     f"{int((co > cg).sum())} / {int(diff.sum())} |")
        L.append("")

    # 5. picture: PCA of every start and final vector, arrows start -> final, for a few glyphs
    show = ["v", "r", "k", "aa", "uu", "n"]
    idx = [names.index(s) for s in show if s in names]
    X = np.concatenate([unit(S[l][idx]) for l in labs] + [unit(F[l][idx]) for l in labs])
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T
    m = len(idx)
    Ps, Pf = P[: n * m].reshape(n, m, 2), P[n * m:].reshape(n, m, 2)
    fig, ax = plt.subplots(figsize=(11, 9))
    cols = plt.cm.tab10(np.arange(m))
    mk = {"own sound": "o", "own, v<->r": "o", "wrong sound": "s", "wrong, v fixed": "s", "copied token": "^",
          "random": "*"}
    for r_i, (lab, _, kind) in enumerate(runs):
        for g_i in range(m):
            a, b = Ps[r_i, g_i], Pf[r_i, g_i]
            ax.annotate("", b, a, arrowprops=dict(arrowstyle="->", color=cols[g_i], alpha=0.35, lw=0.8))
            ax.scatter(*b, color=cols[g_i], marker=mk[kind], s=40, edgecolor="k", lw=0.4)
    for g_i, s in enumerate(show[:m]):
        ax.scatter([], [], color=cols[g_i], label=f"glyph '{s}' (final)")
    for k_, mm in [("own", "o"), ("wrong", "s"), ("copied", "^"), ("random", "*")]:
        ax.scatter([], [], color="grey", marker=mm, label=f"{k_} start")
    ax.legend(fontsize=8); ax.set_title("6 glyphs in every run: arrow = start -> final (2 main directions of all "
                                        "start+final vectors)")
    fig.tight_layout(); fig.savefig(OUT / f"3_start_to_final_pca{TAG}.png", dpi=130); plt.close(fig)
    L += ["## Pictures", "", f"- `results/glyph_embeddings/1_movement{TAG}.png`: cos(start, final), run x glyph.",
          f"- `results/glyph_embeddings/2_same_place_same_shape{TAG}.png`: the four run x run matrices.",
          f"- `results/glyph_embeddings/3_start_to_final_pca{TAG}.png`: start -> final arrows for 6 glyphs, all runs.", ""]
    (ROOT / "results" / "glyph_embedding_drift.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
