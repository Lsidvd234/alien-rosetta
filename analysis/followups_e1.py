"""Follow-ups after R1 / E1 (laptop, one model per process):

    python analysis/followups_e1.py all        # every part below, then results/followups_e1.md
    python analysis/followups_e1.py report

5. forget  English forgetting: WikiText-2 test perplexity (never trained on), the 30 everyday sentences of test 21,
           both also with glyph tokens removed from the prediction, and glyphs leaking into plain-English
           continuations. Models: base Qwen, Option 1, R1, E1a, E1b.
6. scale   E1b with its LoRA scaled to 0.25 / 0.5 / 0.75 / 1.0 (glyph rows unchanged): English perplexity and
           translation on the key test groups.
7. epochs  per-epoch checkpoints (after epochs 1 and 2) of R1, E1a, E1b on the key test groups
           (epoch 3 = the final model, already scored in results/e1_comparison.md and the run files).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RUNS = ROOT / "results" / "kaggle"
OUT = ROOT.parent / "scratch_local" / "followups_e1"
WIKI = ROOT / "train" / "data" / "wikitext2_test.txt"
FORGET = {"Base Qwen3-0.6B-Base": None, "Option 1": "s1-hard-latininit-50k", "R1": "r1-hard-aug-50k",
          "E1a (plain)": "e1a-hard-rep-50k", "E1b (analysis)": "e1b-hard-rep-cot-50k"}
COT = {"e1b-hard-rep-cot-50k"}
KEY = ["in-distribution mix", "held-out voice = passive", "held-out order = object topicalised",
       "held-out case = sentence", "held-out noise = drop", "held-out synonym pairs",
       "extrapolation: 5-6 adjectives on a noun", "extrapolation: 5 clauses", "refusal [heldout]",
       "R1 rewrite synonym [train]", "R1 rewrite typo [train]", "R1 structure 3 adjectives [heldout]",
       "R1 structure three sentences [heldout]"]


def amp_for(dev):
    return lambda: torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda")


def key_groups() -> dict[str, list[dict]]:
    from compare_e1 import groups
    from conlang.dataset import PRESETS, TEST_SPLITS, build
    from train.augment import frame_of
    from train.represent import Clause, Rep
    g = {k: v for k, v in groups().items() if k in KEY}
    data = build(PRESETS["tamil_like"], "abugida_pua", n_train=50000, n_test=500, seed=0)
    rp = Rep("tamil_like", "abugida_pua", 0)
    for k in TEST_SPLITS:                                  # first 200 of each original split
        g[f"original {k} (first 200)"] = [{**it, "analysis": rp.analysis(Clause(frame_of(it["frame"])))}
                                          for it in data["splits"][k][:200]]
    return g


@torch.no_grad()
def wiki_ppl(model, tok, gids: list[int], mask: bool, n_chunks: int = 40, size: int = 256) -> float:
    dev = next(model.parameters()).device
    ids = tok(WIKI.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"][:n_chunks * size]
    g = torch.tensor(gids, device=dev) if gids else None
    nll = n = 0.0
    for s in range(0, len(ids) - 1, size):
        x = torch.tensor([ids[s:s + size]], device=dev)
        if x.shape[1] < 2:
            break
        with amp_for(dev)():
            lg = model(input_ids=x).logits[0, :-1].float()
        if mask and g is not None:
            lg[:, g] = float("-inf")
        nll -= lg.log_softmax(-1).gather(-1, x[0, 1:, None]).sum().item()
        n += x.shape[1] - 1
    return float(torch.exp(torch.tensor(nll / n)))


@torch.no_grad()
def leakage(model, tok) -> float:
    from stress_tests3 import ENGLISH_TEXT
    dev = next(model.parameters()).device
    tok.padding_side = "left"
    enc = tok([s.split(",")[0] for s in ENGLISH_TEXT], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    with amp_for(dev)():
        g = model.generate(**enc, max_new_tokens=25, do_sample=False, pad_token_id=tok.pad_token_id)
    tok.padding_side = "right"
    conts = [tok.decode(r, skip_special_tokens=True) for r in g[:, enc["input_ids"].shape[1]:]]
    return sum(any("" <= c <= "" for c in s) for s in conts) / len(conts)


def load(run):
    from train.load import load_trained
    if run is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
        try:
            m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", dtype=torch.float16)
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", torch_dtype=torch.float16)
        return m.to("cuda" if torch.cuda.is_available() else "cpu").eval(), tok
    m, tok, _ = load_trained(RUNS / run)
    return m, tok


def save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")


def part_forget(label: str) -> None:
    from stress_tests3 import glyph_vocab
    from stress_tests4 import english_ppl2
    run = FORGET[label]
    model, tok = load(None if run is None else f"{run}/model")
    gids = sorted(glyph_vocab(tok).values())
    res = {"wiki": wiki_ppl(model, tok, gids, False), "wiki_masked": wiki_ppl(model, tok, gids, True)}
    if gids:
        res["everyday"], res["glyph_mass"] = english_ppl2(model, tok, gids, False)
        res["everyday_masked"], _ = english_ppl2(model, tok, gids, True)
        res["leak"] = leakage(model, tok)
    else:
        from stress_tests3 import english_ppl
        res["everyday"], _ = english_ppl(model, tok, set())
        res["everyday_masked"], res["glyph_mass"], res["leak"] = res["everyday"], 0.0, 0.0
    save(f"forget_{label.split()[0]}_{label.split()[-1].strip('()')}", {"label": label, **res})
    print(label, res, flush=True)


def part_scale() -> None:
    from stress_tests3 import glyph_vocab, set_lora_scale
    from train.run import gen_eval
    from stress_tests4 import english_ppl2
    run = "e1b-hard-rep-cot-50k"
    model, tok = load(f"{run}/model")
    dev = next(model.parameters()).device
    gids = sorted(glyph_vocab(tok).values())
    keep = ["in-distribution mix", "held-out synonym pairs", "R1 rewrite synonym [train]", "held-out voice = passive",
            "R1 rewrite typo [train]", "original test_iid (first 200)"]
    groups = {k: v[:50] for k, v in key_groups().items() if k in keep}     # 50 items each: 4 strengths x CoT
    prev = OUT / "scale_e1b.json"
    out, saved = (json.loads(prev.read_text(encoding="utf-8")) if prev.exists() else {}), {}
    for lam in (0.25, 0.5, 0.75, 1.0):
        if str(lam) in out:                                     # resume: strengths already measured are kept
            continue
        set_lora_scale(model, lam, saved)
        r = {"wiki": wiki_ppl(model, tok, gids, False, n_chunks=20), "everyday": english_ppl2(model, tok, gids, False)[0]}
        for k, items in groups.items():
            r[k] = gen_eval(model, tok, items, 16, dev, amp_for(dev), True)["exact"]
        out[str(lam)] = r
        save("scale_e1b", out)
        print(lam, r, flush=True)


def part_epochs(run: str, ckpt: str) -> None:
    from train.run import gen_eval
    model, tok = load(f"{run}/{ckpt}")
    dev = next(model.parameters()).device
    res = {}
    groups = {k: (v[:100] if k.startswith("original") else v[:30]) for k, v in key_groups().items()}   # laptop time
    for k, items in groups.items():
        res[k] = gen_eval(model, tok, items, 16, dev, amp_for(dev), run in COT)["exact"]
        save(f"epochs_{run}_{ckpt}", res)
    print(run, ckpt, res, flush=True)


def final_scores(run: str) -> dict:
    """Epoch-3 (final model) scores on the key groups, from the run file or the option1/R1 comparison."""
    if run in ("e1a-hard-rep-50k", "e1b-hard-rep-cot-50k"):    # same items as the epoch subsets, from per-item results
        g = json.loads((RUNS / run / "results" / f"{run}.json").read_text(encoding="utf-8"))["eval_gen"]
        out = {}
        for k, v in g.items():
            if k.startswith("original "):
                ok = v["per_item_ok"][:100]
                out[f"{k} (first 200)"] = ok.count("1") / len(ok)
            else:
                ok = v["per_item_ok"][:30]
                out[k] = ok.count("1") / len(ok)
        return out
    q = OUT / f"epochs_{run}_model.json"                       # R1: final model scored on the same subsets
    return json.loads(q.read_text(encoding="utf-8")) if q.exists() else {}


def report() -> None:
    L = ["# Follow-ups 5-7 (laptop)", ""]
    # 5
    rows = []
    for label in FORGET:
        p = OUT / f"forget_{label.split()[0]}_{label.split()[-1].strip('()')}.json"
        if p.exists():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    L += ["## 5. English forgetting", "",
          "WikiText-2 test: 40 chunks x 256 tokens (never trained on). Everyday: the 30 sentences of test 21. 'Masked' = "
          "glyph tokens removed from the prediction. Leak = plain-English continuations that contain glyphs.", "",
          "| model | WikiText-2 ppl | masked | everyday ppl | masked | probability on glyphs | leak |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['label']} | {r['wiki']:.1f} | {r['wiki_masked']:.1f} | {r['everyday']:.1f} | "
                 f"{r['everyday_masked']:.1f} | {r['glyph_mass']:.3f} | {r['leak']:.0%} |")
    # 6
    p = OUT / "scale_e1b.json"
    if p.exists():
        s = json.loads(p.read_text(encoding="utf-8"))
        lams = list(s)
        L += ["", "## 6. E1b with the LoRA adapter scaled at inference", "",
              "| measure | " + " | ".join(f"strength {x}" for x in lams) + " |", "|---|" + "---|" * len(lams)]
        for k in s[lams[0]]:
            fmt = (lambda v: f"{v:.1f}") if k in ("wiki", "everyday") else (lambda v: f"{v:.2f}")
            name = {"wiki": "WikiText-2 ppl (20 chunks)", "everyday": "everyday ppl"}.get(k, k)
            L.append(f"| {name} | " + " | ".join(fmt(s[x][k]) for x in lams) + " |")
    # 7
    runs = {"R1": "r1-hard-aug-50k", "E1a (plain)": "e1a-hard-rep-50k", "E1b (analysis)": "e1b-hard-rep-cot-50k"}
    cols = {}
    for label, run in runs.items():
        for ck, name in (("model_epoch1", "ep1"), ("model_epoch2", "ep2")):
            q = OUT / f"epochs_{run}_{ck}.json"
            if q.exists():
                cols[f"{label} {name}"] = json.loads(q.read_text(encoding="utf-8"))
        cols[f"{label} ep3"] = final_scores(run)
    if cols:
        keys = list(next(c for c in cols.values() if c))
        L += ["", "## 7. Per-epoch checkpoints (exact match of the glyph output)", "",
              "Subsets for laptop time: first 100 items of each original split (label says 200: name kept for the "
              "lookup), first 30 of every other group, identical for every checkpoint (about +/-9 points per cell).", "",
              "| group | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for k in keys:
            L.append(f"| {k} | " + " | ".join(f"{c[k]:.2f}" if k in c else "-" for c in cols.values()) + " |")
    (ROOT / "results" / "followups_e1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def sub(*args: str) -> None:
    r = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), *args])
    print("part", args, "exit", r.returncode, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "forget":
        part_forget(sys.argv[2])
    elif cmd == "scale":
        part_scale()
    elif cmd == "epochs":
        part_epochs(sys.argv[2], sys.argv[3])
    elif cmd == "all":
        for label in FORGET:
            sub("forget", label)
        sub("scale")
        for run in ("r1-hard-aug-50k", "e1a-hard-rep-50k", "e1b-hard-rep-cot-50k"):
            for ck in ("model_epoch1", "model_epoch2") + (("model",) if run == "r1-hard-aug-50k" else ()):
                sub("epochs", run, ck)
        report()
    else:
        report()
