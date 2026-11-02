"""Claim 4 from inside the model, phase 2 redesigned: does the model USE the meaning directions found by the probes?
(local; generated sentences + the held-out Tatoeba test set; never the sealed set)

    python analysis/inside_claim4_steer2.py RUN [--n 100] [--quick]      -> results/inside/steer2_<run>.json + .md

Why a redesign (first version, analysis/inside_claim4_steer.py, 2026-10-24): one layer, pushes too small (tense flipped
12% at the largest, negation 0%), only the English-word positions, and the real-failure test looked at one error
direction only (none of the 13 real tense errors went that way). This version:

  Direction   difference of the mean hidden states of generated sentences with value B and with value A (per layer),
              scaled to unit length; the push is alpha x (the mean hidden-state norm at that layer) x direction, so
              alpha is a share of the layer's own scale and comparable across layers.
  Stage 1     sweep on generated sentences for one pair (tense PRES -> PAST): 6 layers x 3 positions (the English words,
              the last prompt token, every prompt token) x alphas 0.005-0.1. Reports the flip rate (output now says B)
              and the damage rate (output no longer has a readable analysis). Picks the setting with the most flips
              at damage <= 10%.
  Stage 2     at the picked setting (and its neighbours), every pair in both directions: tense PRES<->PAST,
              PRES<->FUT, NEG no<->yes, Q no<->yes. Control: a random direction of the same length. Collateral: share
              of the OTHER features left unchanged.
  Stage 3     the claim-4 test: every real Tatoeba sentence where the unsteered output gets a feature wrong is pushed
              from the value it wrote toward the gold value, at the picked setting; fixed rate vs the random control.
              High fixed rate with a low random rate = the model reads this direction, and the real-English failure is a
              too-weak/misplaced signal that a nudge repairs (a mapping gap, claim 4). Near zero = not a simple
              read-out failure.
A probe finding information is not proof of use; a push that changes the output, beyond a random push of the same
size and without breaking the rest, is.
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.inside_claim4 import OUT, english_span, features, generated, real  # noqa: E402
from train.load import load_trained  # noqa: E402
from train.run import prompt  # noqa: E402

PAIRS = {"tense:PRES>PAST": ("tense", "PRES", "PAST"), "tense:PAST>PRES": ("tense", "PAST", "PRES"),
         "tense:PRES>FUT": ("tense", "PRES", "FUT"), "tense:FUT>PRES": ("tense", "FUT", "PRES"),
         "NEG:no>yes": ("NEG", "False", "True"), "NEG:yes>no": ("NEG", "True", "False"),
         "Q:no>yes": ("Q", "False", "True"), "Q:yes>no": ("Q", "True", "False")}
POSITIONS = ("english", "last", "all")
ALPHAS = (0.005, 0.01, 0.02, 0.035, 0.05, 0.1)   # smoke test: 0.05 already flips tense, 0.1 breaks outputs


def layers_of(model):
    m = model
    for attr in ("base_model", "model", "model"):
        m = getattr(m, attr, m)
    return m.layers


def _enc(tok, sents):
    """Prompt token ids and the [a, b) span of the English words (english_span; first word kept since 2026-10-25)."""
    return [tok(prompt(x), add_special_tokens=False)["input_ids"] for x in sents], [english_span(tok, x) for x in sents]


@torch.no_grad()
def layer_stats(model, tok, sents, layers, bsz=16):
    """{layer: (mean state over the English words per sentence [n, d], mean norm of all prompt states)}."""
    device = next(model.parameters()).device
    enc, spans = _enc(tok, sents)
    per = {L: [] for L in layers}
    norms = {L: [] for L in layers}
    for s in range(0, len(enc), bsz):
        batch = enc[s:s + bsz]
        n = max(map(len, batch))
        ids = torch.full((len(batch), n), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, e in enumerate(batch):
            ids[i, :len(e)], att[i, :len(e)] = torch.tensor(e), 1
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            hs = model(input_ids=ids.to(device), attention_mask=att.to(device), output_hidden_states=True).hidden_states
        for L in layers:
            h = hs[L].float()
            for i, e in enumerate(batch):
                per[L].append(h[i, spans[s + i][0]:spans[s + i][1]].mean(0).cpu())
                norms[L].append(h[i, :len(e)].norm(dim=-1).mean().item())
    return {L: (torch.stack(per[L]), float(np.mean(norms[L]))) for L in layers}


@torch.no_grad()
def outputs(model, tok, sents, layer=None, vec=None, position="english", bsz=16, max_new=40):
    """Greedy analyses; with layer/vec, vec is added at `position` of the prompt at hidden_states[layer]."""
    device = next(model.parameters()).device
    enc, spans = _enc(tok, sents)
    block = layers_of(model)[layer - 1] if layer else None
    res = []
    for s in range(0, len(enc), bsz):
        batch = enc[s:s + bsz]
        n = max(map(len, batch))
        ids = torch.full((len(batch), n), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        mask = torch.zeros((len(batch), n, 1))
        for i, e in enumerate(batch):                                # left padding, as in translate()
            o = n - len(e)
            ids[i, o:], att[i, o:] = torch.tensor(e), 1
            if position == "english":
                a, b = spans[s + i]
                mask[i, o + a:o + b] = 1
            elif position == "last":
                mask[i, n - 1] = 1
            else:
                mask[i, o:] = 1
        handle = None
        if block is not None:
            add = (mask * vec.view(1, 1, -1)).to(device)

            def hook(_m, _inp, output):
                h = output[0] if isinstance(output, tuple) else output
                if h.shape[1] == n:                                  # only while the prompt is read
                    h = h + add.to(h.dtype)
                    return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
                return output
            handle = block.register_forward_hook(hook)
        try:
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                g = model.generate(input_ids=ids.to(device), attention_mask=att.to(device), max_new_tokens=max_new,
                                   do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        finally:
            if handle:
                handle.remove()
        res += [tok.decode(r, skip_special_tokens=True).split("\n")[0].split("=>")[0].strip() for r in g[:, n:]]
    return res


class Checkpoint:
    """Every generation's outputs saved to <path> as soon as they exist, so a stopped run resumes where it was cut
    (2026-10-26: the overnight queue was stopped for low memory and lost ~8 h of subject steering). The file is
    removed when the run finishes; delete it by hand to start a run over."""

    def __init__(self, path: Path):
        self.path = path
        self.d = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if self.d:
            print(f"  resuming from {path.name}: {len(self.d)} saved generations", flush=True)

    def get(self, key: str, fn):
        if key not in self.d:
            self.d[key] = fn()
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.d), encoding="utf-8")
            tmp.replace(self.path)
        return self.d[key]

    def done(self):
        self.path.unlink(missing_ok=True)


def random_unit(shape, key: str, salt: str = "random") -> torch.Tensor:
    """A random direction seeded by its own key (not one shared stream), so a resumed run draws the same controls."""
    r = torch.randn(shape, generator=torch.Generator().manual_seed(zlib.crc32(f"{salt}:{key}".encode())))
    return r / r.norm()


def rates(outs, feat, target, before):
    """(share whose feature now equals target, share unreadable, share with the other features unchanged)."""
    f = [features(o) for o in outs]
    flip = np.mean([bool(x) and str(x[feat]) == target for x in f])
    broken = np.mean([not x for x in f])
    keep = [all(str(x[k]) == str(b[k]) for k in x if k != feat) for x, b in zip(f, before) if x and b]
    return float(flip), float(broken), float(np.mean(keep)) if keep else None


def run(name: str, n: int, quick: bool, heldout: bool = False) -> dict:
    """heldout (review, 30 Oct): the setting is picked in stage 1 on OTHER sentences than the ones stage 2 reports on
    (before, stage 1 and the PRES>PAST row of stage 2 used the same 100 sentences)."""
    model, tok, _ = load_trained(ROOT / "results" / "kaggle" / name / "model")
    n_layers = len(layers_of(model))
    layers = sorted({max(1, round(n_layers * f)) for f in ((0.5, 0.75) if quick else (0.2, 0.35, 0.5, 0.65, 0.8, 0.95))})
    alphas = ALPHAS[:2] if quick else ALPHAS
    gen_items = generated(400 if quick else 3000)
    by_val = {}
    for it in gen_items:
        f = features(it["analysis"])
        for feat in ("tense", "NEG", "Q"):
            by_val.setdefault((feat, str(f[feat])), []).append(it["english"])
    fit = {k: v[:min(len(v), 300)] for k, v in by_val.items()}
    test = {k: v[300:300 + n] if len(v) > 300 + n // 2 else v[-n:] for k, v in by_val.items()}
    select = {k: v[300 + n:300 + 2 * n] for k, v in by_val.items() if len(v) >= 300 + 2 * n}   # never in test
    stats = {}
    for key, sents in fit.items():
        if key[1] in ("PRES", "PAST", "FUT", "True", "False"):
            stats[key] = layer_stats(model, tok, sents, layers)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = ("_quick" if quick else "") + ("_heldout" if heldout else "")
    ck = Checkpoint(OUT / f"steer2_{name}{tag}_n{n}.partial")

    def gen(key, sents, *push):
        return ck.get(key, lambda: outputs(model, tok, sents, *push))

    def direction(feat, a, b, L):
        d = stats[(feat, b)][L][0].mean(0) - stats[(feat, a)][L][0].mean(0)
        scale = (stats[(feat, a)][L][1] + stats[(feat, b)][L][1]) / 2
        return d / d.norm(), scale

    out = {"run": name, "layers": layers, "alphas": alphas, "positions": POSITIONS, "stage1": {}, "stage2": {},
           "stage3": {}}
    # stage 1: sweep on one pair
    feat, a, b = PAIRS["tense:PRES>PAST"]
    sents = select[(feat, a)] if heldout else test[(feat, a)]
    assert not heldout or not set(sents) & set(test[(feat, a)]), "selection and test sentences overlap"
    before = [features(o) for o in gen("s1:before", sents)]
    best = None
    for L in layers:
        d, scale = direction(feat, a, b, L)
        r = random_unit(d.shape, f"s1:L{L}")
        for pos in POSITIONS:
            for al in alphas:
                k = f"L{L}:{pos}:{al}"
                fl, br, kp = rates(gen(f"s1:{k}", sents, L, al * scale * d, pos), feat, b, before)
                rf, rb, _ = rates(gen(f"s1:{k}:random", sents, L, al * scale * r, pos), feat, b, before)
                out["stage1"][f"L{L}:{pos}:{al}"] = {"flip": fl, "broken": br, "others_kept": kp,
                                                    "random_flip": rf, "random_broken": rb}
                print(f"  stage1 L{L} {pos:7s} a={al}: flip {fl:.2f} (random {rf:.2f}) broken {br:.2f}", flush=True)
                if br <= 0.10 and (best is None or fl - rf > best[0]):
                    best = (fl - rf, L, pos, al)
    if best is None:
        best = (0, layers[len(layers) // 2], "english", alphas[0])
    _, L, pos, al = best
    out["picked"] = {"layer": L, "position": pos, "alpha": al}
    print(f"  picked layer {L}, {pos}, alpha {al}", flush=True)
    # stage 2: every pair, both directions, at the picked setting
    for pname, (feat, a, b) in PAIRS.items():
        if (feat, a) not in test or (feat, b) not in stats and (feat, b) not in fit:
            continue
        sents = test[(feat, a)]
        before = [features(o) for o in gen(f"s2:{pname}:before", sents)]
        d, scale = direction(feat, a, b, L)
        fl, br, kp = rates(gen(f"s2:{pname}", sents, L, al * scale * d, pos), feat, b, before)
        rf, rb, _ = rates(gen(f"s2:{pname}:random", sents, L, al * scale * random_unit(d.shape, f"s2:{pname}"), pos),
                          feat, b, before)
        out["stage2"][pname] = {"n": len(sents), "flip": fl, "broken": br, "others_kept": kp, "random_flip": rf}
        print(f"  stage2 {pname:16s}: flip {fl:.2f} (random {rf:.2f}) broken {br:.2f} others kept {kp}", flush=True)
    # stage 3: real failures, pushed from the value written toward the gold value
    reals = real()
    base = gen("s3:base", [it["english"] for it in reals])
    for feat in ("tense", "NEG", "Q"):
        cases = []
        for it, o in zip(reals, base):
            g, p = features(it["analysis"]), features(o)
            if g and p and str(g[feat]) != str(p[feat]) and (feat, str(p[feat])) in stats and \
                    (feat, str(g[feat])) in stats:
                cases.append((it["english"], str(p[feat]), str(g[feat])))
        fixed = rfixed = 0
        for eng, wrote, gold in cases:
            d, scale = direction(feat, wrote, gold, L)
            k = f"s3:{feat}:{eng}"
            o1 = features(gen(k, [eng], L, al * scale * d, pos)[0])
            o2 = features(gen(k + ":random", [eng], L, al * scale * random_unit(d.shape, k), pos)[0])
            fixed += bool(o1) and str(o1[feat]) == gold
            rfixed += bool(o2) and str(o2[feat]) == gold
        out["stage3"][feat] = {"n_failures": len(cases), "fixed": fixed, "random_fixed": rfixed}
        print(f"  stage3 {feat}: {fixed}/{len(cases)} real failures fixed (random {rfixed})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out["heldout_selection"] = heldout
    (OUT / f"steer2_{name}{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = [f"# Steering test (phase 2, redesigned): {name}", "",
          f"Picked setting: layer {L}, position {pos}, alpha {al} (share of the layer's hidden-state norm)"
          + (", picked on 100 OTHER sentences than the ones reported below." if heldout else ".") + "", "",
          "| pair | n | flip | random flip | unreadable | other features kept |", "|---|---|---|---|---|---|"]
    md += [f"| {k} | {v['n']} | {v['flip']:.2f} | {v['random_flip']:.2f} | {v['broken']:.2f} | "
           f"{v['others_kept'] if v['others_kept'] is None else round(v['others_kept'], 2)} |"
           for k, v in out["stage2"].items()]
    md += ["", "| feature | real failures | fixed by the push | fixed by a random push |", "|---|---|---|---|"]
    md += [f"| {k} | {v['n_failures']} | {v['fixed']} | {v['random_fixed']} |" for k, v in out["stage3"].items()]
    (OUT / f"steer2_{name}{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    ck.done()
    return out


if __name__ == "__main__":
    if (ROOT / "results" / "STOP_steer2").exists():     # 30 Oct: user unqueued the held-out steering re-run
        print("steering skipped: results/STOP_steer2 exists (delete it to run)", flush=True)
        sys.exit(0)
    a = sys.argv[1:]
    n = int(a[a.index("--n") + 1]) if "--n" in a else 100
    quick = "--quick" in a
    for r in [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] != "--n")]:
        run(r, n, quick, heldout="--heldout" in a)
