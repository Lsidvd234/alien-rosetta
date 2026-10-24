"""Claim 4 from inside the model, phase 2 (use): does the model READ the meaning direction found in phase 1?
(local; generated sentences + held-out Tatoeba test set; never the sealed set)

    python analysis/inside_claim4_steer.py RUN [--layer L] [--features tense,NEG]   -> results/inside/steer_<run>.json

For a feature value pair (PRES -> PAST, not-NEG -> NEG), the direction at layer L is the difference of the mean hidden
states of generated sentences with each value (English word positions). Adding alpha x direction to the English
positions at layer L while the prompt is read, then generating, tests use:
  1. generated sentences with value A: share whose OUTPUT flips to B as alpha grows (dose-response), vs a random
     direction of the same length (control). A flip means the model uses this direction to decide the feature.
  2. real Tatoeba sentences whose gold value is B but whose output says A (the claim-4 failures): share fixed by
     the same push. If small pushes fix them, the meaning was there and only too weak / misread for real English.
The layer defaults to the best mean-representation layer from phase 1 (results/inside/claim4_<run>.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.inside_claim4 import OUT, features, generated, real  # noqa: E402
from train.load import load_trained  # noqa: E402
from train.run import prompt  # noqa: E402

PAIRS = {"tense": ("PRES", "PAST"), "NEG": ("False", "True")}
ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0)


def layers_of(model):
    m = model
    for attr in ("base_model", "model", "model"):
        m = getattr(m, attr, m)
    return m.layers


@torch.no_grad()
def states(model, tok, sents, layer, bsz=16):
    """Mean hidden state over the English words at `layer` (hidden_states index), per sentence."""
    device = next(model.parameters()).device
    head = len(tok("English: ", add_special_tokens=False)["input_ids"])
    tail = len(tok("\nTarget:\n", add_special_tokens=False)["input_ids"])
    out = []
    for s in range(0, len(sents), bsz):
        enc = [tok(prompt(x), add_special_tokens=False)["input_ids"] for x in sents[s:s + bsz]]
        L = max(map(len, enc))
        ids = torch.full((len(enc), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, e in enumerate(enc):
            ids[i, :len(e)], att[i, :len(e)] = torch.tensor(e), 1
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            hs = model(input_ids=ids.to(device), attention_mask=att.to(device), output_hidden_states=True).hidden_states
        h = hs[layer].float()
        for i, e in enumerate(enc):
            out.append(h[i, head:max(head + 1, len(e) - tail)].mean(0).cpu())
    return torch.stack(out)


@torch.no_grad()
def steered_outputs(model, tok, sents, layer, vec, alpha, bsz=16, max_new=40):
    """Generate with alpha*vec added to the English positions of the prompt at `layer` (output of block layer-1)."""
    device = next(model.parameters()).device
    head = len(tok("English: ", add_special_tokens=False)["input_ids"])
    tail = len(tok("\nTarget:\n", add_special_tokens=False)["input_ids"])
    block = layers_of(model)[layer - 1]
    outs = []
    for s in range(0, len(sents), bsz):
        batch = sents[s:s + bsz]
        enc = [tok(prompt(x), add_special_tokens=False)["input_ids"] for x in batch]
        L = max(map(len, enc))
        ids = torch.full((len(enc), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        mask = torch.zeros((len(enc), L, 1))
        for i, e in enumerate(enc):                                  # left padding, as in translate()
            ids[i, L - len(e):], att[i, L - len(e):] = torch.tensor(e), 1
            mask[i, L - len(e) + head:max(L - len(e) + head + 1, L - tail)] = 1
        add = (alpha * mask * vec.view(1, 1, -1)).to(device)

        def hook(_m, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] == L:                                      # only while the prompt is read
                h = h + add.to(h.dtype)
                return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
            return output
        handle = block.register_forward_hook(hook)
        try:
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                gen = model.generate(input_ids=ids.to(device), attention_mask=att.to(device), max_new_tokens=max_new,
                                     do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        finally:
            handle.remove()
        outs += [tok.decode(r, skip_special_tokens=True).split("\n")[0].split("=>")[0].strip()
                 for r in gen[:, L:]]
    return outs


def run(name: str, layer: int | None, feats: list[str], n_gen: int = 2400) -> dict:
    model, tok, _ = load_trained(ROOT / "results" / "kaggle" / name / "model")
    if layer is None:
        p1 = json.loads((OUT / f"claim4_{name}.json").read_text(encoding="utf-8"))
        layer = int(p1["features"]["tense"]["best_layer"].split(":")[1])
    gen_items, real_items = generated(n_gen), real()
    res = {"run": name, "layer": layer, "alphas": ALPHAS, "features": {}}
    rng = torch.Generator().manual_seed(0)
    for f in feats:
        a, b = PAIRS[f]
        ga = [it["english"] for it in gen_items if str(features(it["analysis"])[f]) == a]
        gb = [it["english"] for it in gen_items if str(features(it["analysis"])[f]) == b]
        fit_a, test_a = ga[:300], ga[300:500]
        vec = states(model, tok, gb[:300], layer).mean(0) - states(model, tok, fit_a, layer).mean(0)
        rand = torch.randn(vec.shape, generator=rng)
        rand = rand / rand.norm() * vec.norm()
        # the claim-4 failures: real sentences with gold b whose unsteered output says a
        rb = [it["english"] for it in real_items if str(features(it["analysis"])[f]) == b]
        base_out = steered_outputs(model, tok, rb, layer, vec, 0.0)
        fails = [s for s, o in zip(rb, base_out) if features(o) and str(features(o)[f]) == a]
        row = {"pair": [a, b], "n_test_gen": len(test_a), "n_real_fail": len(fails), "gen_flip": {}, "gen_flip_random": {},
               "real_fixed": {}, "real_fixed_random": {}}
        for al in ALPHAS:
            for key, v in (("", vec), ("_random", rand)):
                o = steered_outputs(model, tok, test_a, layer, v, al)
                row["gen_flip" + key][al] = round(float(np.mean([bool(features(x)) and str(features(x)[f]) == b
                                                                  for x in o])), 4)
                if fails:
                    o = steered_outputs(model, tok, fails, layer, v, al)
                    row["real_fixed" + key][al] = round(float(np.mean([bool(features(x)) and str(features(x)[f]) == b
                                                                       for x in o])), 4)
            print(f"  {f} alpha {al}: generated flip {row['gen_flip'][al]} (random {row['gen_flip_random'][al]}); "
                  f"real failures fixed {row['real_fixed'].get(al)} (random {row['real_fixed_random'].get(al)}) "
                  f"of {len(fails)}", flush=True)
        res["features"][f] = row
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"steer_{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    layer = int(a[a.index("--layer") + 1]) if "--layer" in a else None
    feats = a[a.index("--features") + 1].split(",") if "--features" in a else ["tense", "NEG"]
    for r in [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--layer", "--features"))]:
        run(r, layer, feats)
