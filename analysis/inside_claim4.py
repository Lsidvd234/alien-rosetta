"""Claim 4 from inside the model: when a real English sentence is translated wrong, is its meaning still in the model?
(local; generated sentences + the held-out Tatoeba test set; never the sealed set)

    python analysis/inside_claim4.py RUN [RUN ...] [--n-gen 2400]    -> results/inside/claim4_<run>.json
    python analysis/inside_claim4.py --table                          -> results/inside/claim4.md + figure
    RUN = a run folder under results/kaggle (its model/), or "base" (untrained Qwen3-0.6B-Base, no output scoring)

Method (phase 1, read-out):
  1. Meaning features of the main verb and subject, read from an analysis string: tense (PAST/PRES/FUT), NEG, Q,
     PROG, PERF, modal (CAN/MUST/SHOULD/MAY/WOULD/...), subject (1sg, 2sg, 3sgm, 3sgf, 1pl, 3pl, name, noun, none).
  2. Hidden states of the prompt "English: ...\nTarget:\n" at every layer: the last prompt token (what the answer is
     generated from) and the mean over the English words.
  3. A linear probe per feature and layer, trained ONLY on generated sentences (the training distribution). The layer
     is chosen on held-out generated sentences, never on real ones. Then the probe reads the real Tatoeba sentences.
  4. The model's own answer for each real sentence gives the same features (output accuracy).
Key number: probe accuracy on the real items whose OUTPUT got that feature wrong. High = the meaning was in the model
and only the mapping to the output failed (a coverage gap of the trained mapping, claim 4). Near chance = the model
did not represent it either. A probe can find information the model does not use: phase 2 (steering along the probe
direction) tests use; this file is the read-out half.
"""

from __future__ import annotations

import json
import random
from collections import Counter
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.load import load_trained, translate  # noqa: E402
from train.run import prompt  # noqa: E402

OUT = ROOT / "results" / "inside"
MODALS = ("CAN", "MUST", "SHOULD", "MAY", "WOULD", "COULD", "MIGHT", "LET", "WILL")
FEATURES = ("tense", "NEG", "Q", "PROG", "PERF", "modal", "subject")


def english_span(tok, sentence: str) -> tuple[int, int]:
    """Token positions [a, b) of the English sentence inside tok(prompt(sentence)), from character offsets.
    Fixed 2026-10-25: the old span started at len(tok("English: ")), but that ends in a space token which in context
    merges into the first word (" I"), so the first word was left out (the subject, for subject probes/steering)."""
    text = prompt(sentence)
    s0 = text.index(sentence)
    s1 = s0 + len(sentence)
    offs = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    idx = [i for i, (x, y) in enumerate(offs) if y > s0 and x < s1]
    return idx[0], idx[-1] + 1


def features(analysis: str) -> dict | None:
    """Features of the FIRST clause's main verb and subject; None if there is no verb. A possessive subject
    (S=PRON.1sg^dog, "my dog") is the possessed noun: "noun" (fixed 2026-10-25; it was labelled as the possessor)."""
    first = analysis.split(" ; ")[0]
    v = re.search(r"(?:^| )V=([^ ]+)", first)
    if not v:
        return None
    marks = v.group(1).split(".")[1:]
    tense = next((t for t in ("PAST", "PRES", "FUT") if t in marks), "none")
    s = re.search(r"(?:^| )S=([^ ]+)", first)
    if not s:
        subj = "none"
    elif "^" in s.group(1):
        subj = "noun"
    else:
        head = s.group(1).split("&")[0]
        if "&" in s.group(1):
            subj = "group"
        elif head.startswith("PRON."):
            subj = re.split(r"[\^|]", head.split(".")[1])[0]
        elif head.startswith("NAME."):
            subj = "name"
        else:
            subj = "noun"
    return {"tense": tense, "NEG": "NEG" in marks, "Q": "Q" in marks, "PROG": "PROG" in marks,
            "PERF": "PERF" in marks, "modal": any(m in marks for m in MODALS), "subject": subj}


def generated(n: int, seed: int = 777) -> list[dict]:
    """Single-clause generated sentences of the run's language family (seed-0 language, own rng)."""
    from train.represent_v2 import GenV2
    gen = GenV2("tamil_like", "abugida_pua", 0)
    rng = random.Random(seed)
    items, seen = [], set()
    while len(items) < n:
        ex = gen.example(rng, n_events=1)
        it = gen.item(ex) if ex else None
        if it and it["english"] not in seen and features(it["analysis"]):
            seen.add(it["english"])
            items.append(it)
    return items


def real() -> list[dict]:
    items = json.loads((ROOT / "results" / "tatoeba_test.json").read_text(encoding="utf-8"))
    return [it for it in items if features(it["analysis"])]


@torch.no_grad()
def hidden(model, tok, sentences: list[str], bsz: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """(last prompt token, mean over the English words): arrays [n, layers+1, d] in float16."""
    device = next(model.parameters()).device
    last, mean = [], []
    for s in range(0, len(sentences), bsz):
        batch = sentences[s:s + bsz]
        texts = [prompt(x) for x in batch]
        enc = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        spans = [english_span(tok, x) for x in batch]
        L = max(map(len, enc))
        ids = torch.full((len(enc), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(enc), L), dtype=torch.long)
        for i, e in enumerate(enc):                                 # right padding: positions stay comparable
            ids[i, :len(e)] = torch.tensor(e)
            att[i, :len(e)] = 1
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            hs = model(input_ids=ids.to(device), attention_mask=att.to(device), output_hidden_states=True).hidden_states
        hs = torch.stack(hs, 1).float()                             # [b, layers+1, L, d]
        for i, e in enumerate(enc):
            n = len(e)
            last.append(hs[i, :, n - 1].cpu().half().numpy())
            a, b = spans[i]                                         # the English words
            mean.append(hs[i, :, a:b].mean(1).cpu().half().numpy())
    return np.stack(last), np.stack(mean)


def probe(xtr, ytr, xva, yva, xte):
    """Logistic regression on standardised states; returns (validation accuracy, test predictions)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(xtr)
    clf = LogisticRegression(max_iter=2000, C=0.05).fit(sc.transform(xtr), ytr)
    return float((clf.predict(sc.transform(xva)) == yva).mean()), clf.predict(sc.transform(xte))


def analyse(run: str, n_gen: int) -> dict:
    if run == "base":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base",
                                                     dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    else:
        model, tok, _ = load_trained(ROOT / "results" / "kaggle" / run / "model")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    gen_items, real_items = generated(n_gen), real()
    print(f"{run}: {len(gen_items)} generated, {len(real_items)} real sentences", flush=True)
    g_last, g_mean = hidden(model, tok, [it["english"] for it in gen_items])
    r_last, r_mean = hidden(model, tok, [it["english"] for it in real_items])
    out_feats = None
    if run != "base":                                               # the model's own answers for the real sentences
        outs = []
        for s in range(0, len(real_items), 32):
            outs += translate(model, tok, [it["english"] for it in real_items[s:s + 32]], max_new_tokens=96)
        out_feats = [features(o.split("=>")[0].strip()) for o in outs]
    gold_g = [features(it["analysis"]) for it in gen_items]
    gold_r = [features(it["analysis"]) for it in real_items]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(gen_items))
    tr, va = idx[: int(0.8 * len(idx))], idx[int(0.8 * len(idx)):]
    res = {"run": run, "n_gen": len(gen_items), "n_real": len(real_items), "features": {},
           "english": [it["english"] for it in real_items]}
    for f in FEATURES:
        yg = np.array([str(g[f]) for g in gold_g])
        yr = np.array([str(g[f]) for g in gold_r])
        row = {"majority_real": float(max((yr == v).mean() for v in set(yr))), "layers": {}}
        best = None
        for rep, (G, R) in {"last": (g_last, r_last), "mean": (g_mean, r_mean)}.items():
            for layer in range(G.shape[1]):
                if len(set(yg[tr])) < 2:
                    continue
                acc_va, pred_r = probe(G[tr, layer], yg[tr], G[va, layer], yg[va], R[:, layer])
                row["layers"][f"{rep}:{layer}"] = {"gen_val": round(acc_va, 4),
                                                    "real": round(float((pred_r == yr).mean()), 4)}
                if best is None or acc_va > best[0]:
                    best = (acc_va, rep, layer, pred_r)
        acc_va, rep, layer, pred_r = best                          # chosen on generated sentences only
        row.update(best_layer=f"{rep}:{layer}", probe_gen_val=round(acc_va, 4),
                   probe_real=round(float((pred_r == yr).mean()), 4))
        # per sentence (added 2026-10-26), in the order of res["english"]: gold value, the probe's reading at the
        # best layer, and the model's own output ("?" = unreadable), so agreement can be checked item by item
        row["items"] = {"gold": yr.tolist(), "probe": [str(x) for x in pred_r]}
        if out_feats is not None:
            # Fixed 2026-10-25: an output with no readable analysis used to count as "wrong" for every feature at
            # once (all 10 NEG and 10 Q "errors" of the baseline were such outputs). Now "wrong" = a readable output
            # with the wrong value; unreadable ones are reported apart, with the majority-class rate in the wrong
            # subset as the baseline the probe has to beat there.
            yo = np.array([str(o[f]) if o else "?" for o in out_feats])
            readable = np.array([bool(o) for o in out_feats])
            ok = yo == yr
            wrong = readable & ~ok
            maj = Counter(yr[wrong].tolist()).most_common(1)[0][1] / int(wrong.sum()) if wrong.any() else None
            row["items"]["output"] = yo.tolist()
            # where the output is wrong: does the probe read the SAME wrong value the model wrote?
            row["probe_agrees_with_wrong_output"] = (round(float((pred_r[wrong] == yo[wrong]).mean()), 4)
                                                     if wrong.any() else None)
            row.update(n_output_unreadable=int((~readable).sum()),
                       majority_where_output_wrong=round(float(maj), 4) if maj is not None else None,
                       probe_real_where_unreadable=round(float((pred_r[~readable] == yr[~readable]).mean()), 4)
                       if (~readable).any() else None)
            row.update(output_real=round(float(ok.mean()), 4), n_output_wrong=int(wrong.sum()),
                       probe_real_where_output_wrong=round(float((pred_r[wrong] == yr[wrong]).mean()), 4)
                       if wrong.any() else None,
                       probe_real_where_output_right=round(float((pred_r[ok] == yr[ok]).mean()), 4)
                       if ok.any() else None)
        res["features"][f] = row
        print(f"  {f:8s} probe gen {row['probe_gen_val']:.3f} real {row['probe_real']:.3f} ({row['best_layer']})"
              + (f" | output real {row['output_real']:.3f}, {row['n_output_wrong']} wrong, probe right on "
                 f"{row['probe_real_where_output_wrong']}" if out_feats is not None else ""), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"claim4_{run}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return res


def table() -> None:
    def v(x, k):
        return "-" if x.get(k) is None else x[k]
    rows = ["| run | feature | majority (real) | probe on real | model output on real | output wrong (n, readable) | "
            "majority where wrong | probe right where output wrong | output unreadable (n) | probe right where "
            "unreadable | probe right where output right | probe = the wrong output | layer |", "|" + "---|" * 13]
    for f in sorted(OUT.glob("claim4_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        for feat, x in r["features"].items():
            rows.append(f"| {r['run']} | {feat} | {x['majority_real']:.2f} | {x['probe_real']:.2f} | "
                        f"{x.get('output_real', float('nan')):.2f} | {v(x, 'n_output_wrong')} | "
                        f"{v(x, 'majority_where_output_wrong')} | {v(x, 'probe_real_where_output_wrong')} | "
                        f"{v(x, 'n_output_unreadable')} | {v(x, 'probe_real_where_unreadable')} | "
                        f"{x.get('probe_real_where_output_right') if x.get('probe_real_where_output_right') is not None else '-'} | "
                        f"{v(x, 'probe_agrees_with_wrong_output')} | "
                        f"{x['best_layer']} |")
    (OUT / "claim4.md").write_text("# Claim 4 from inside the model (read-out)\n\n" + "\n".join(rows) + "\n",
                                   encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--table"]:
        table()
    else:
        n = int(args[args.index("--n-gen") + 1]) if "--n-gen" in args else 2400
        for r in [a for a in args if not a.startswith("--") and not a.isdigit()]:
            analyse(r, n)
        table()
