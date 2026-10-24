"""Fifth batch (27-30), from results/stress/more_test_ideas.md, all on one shared pool of inputs.

    python analysis/stress_tests5.py all       # one process per saved model, then all reports
    python analysis/stress_tests5.py report    # rebuild the reports from the saved parts

27. unseen      does an input fail exactly when it contains a local English structure never seen in training?
                (unseen word / word bigram / word-class bigram / word-class trigram; Bogin et al. 2022)
28. ood         input-side "this English is strange" detector: Mahalanobis distance of the prompt's hidden states
                from the training inputs (Lee et al. 2018; Ren et al. 2023)
29. calib       calibration: ECE, reliability, Platt scaling, risk-coverage / coverage at 99% accuracy
30. unicode     invisible Unicode changes (homoglyphs, zero-width, no-break space, full-width, soft hyphen) and the
                NFKC fix (Boucher et al. 2022)

Per-model records: scratch_local/st5_parts/model<i>.json (one record per input).
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stress_tests import RUNS, Ctx, write  # noqa: E402
from stress_tests2 import random_frame, saved_models, valid  # noqa: E402
from stress_tests3 import (ODD_INPUTS, auroc, english_variants, failure_pools, generate, item_of, pct,  # noqa: E402
                           score)
from conlang.dataset import PRESETS, build  # noqa: E402
from conlang.frames import PRONOUNS  # noqa: E402

SEED = 5050
PARTS = Path(__file__).resolve().parents[2] / "scratch_local" / "st5_parts"
ID_KINDS = {"normal", "test_iid", "test_comp", "test_morph", "test_long"}


# --- the shared pool ------------------------------------------------------------------------------

HOMO = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у",
        "i": "і"}


def perturb(s: str, kind: str, rng: random.Random) -> str | None:
    w = s.split(" ")
    long_w = [i for i, x in enumerate(w) if len(x) >= 3]
    if kind == "homoglyph (Cyrillic look-alike letter)":
        c = [(i, j) for i in long_w for j, ch in enumerate(w[i]) if ch in HOMO]
        if not c:
            return None
        i, j = rng.choice(c)
        w[i] = w[i][:j] + HOMO[w[i][j]] + w[i][j + 1:]
    elif kind in ("zero-width space inside a word", "soft hyphen inside a word"):
        if not long_w:
            return None
        i = rng.choice(long_w)
        j = rng.randrange(1, len(w[i]))
        w[i] = w[i][:j] + ("​" if kind.startswith("zero") else "­") + w[i][j:]
    elif kind == "no-break space between words":
        sp = [i for i, ch in enumerate(s) if ch == " "]
        if not sp:
            return None
        i = rng.choice(sp)
        return s[:i] + " " + s[i + 1:]
    elif kind == "full-width letters in one word":
        i = rng.choice(range(len(w)))
        w[i] = "".join(chr(ord(ch) + 0xFEE0) if "!" <= ch <= "~" else ch for ch in w[i])
    return " ".join(w)


UNI = ["homoglyph (Cyrillic look-alike letter)", "zero-width space inside a word", "no-break space between words",
       "full-width letters in one word", "soft hyphen inside a word"]


def strip_invisible(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Cf")


def build_pool(ctx: Ctx) -> list[dict]:
    rng = random.Random(SEED)
    pool = []
    for k, items in failure_pools(ctx, 100, SEED).items():
        if k != "'Input/Output' prompt labels":          # a prompt-format change, not an English input
            pool += [{**it, "kind": k} for it in items]
    data = build(PRESETS[ctx.cfg["preset"]], ctx.cfg["script"], n_train=50000, n_test=500, seed=ctx.seed)
    for k in ("test_iid", "test_comp", "test_morph", "test_long"):
        pool += [{"english": it["english"], "target": it["target"], "kind": k} for it in data["splits"][k][:100]]
    var = defaultdict(list)
    uni = defaultdict(list)
    tries = 0
    while tries < 50000 and (min((len(v) for v in var.values()), default=0) < 60 or len(var) < 8
                             or min((len(uni[k]) for k in UNI + ["3 mixed invisible changes"]), default=0) < 100):
        tries += 1
        f = random_frame(ctx, rng)
        if not valid(ctx, f):
            continue
        it = item_of(ctx, f)
        for k, en in english_variants(ctx, f).items():
            if en and len(var[k]) < 60:
                var[k].append({"english": en, "target": it["target"], "kind": k})
        for k in UNI:
            if len(uni[k]) < 100:
                p = perturb(it["english"], k, rng)
                if p:
                    uni[k].append({"english": p, "target": it["target"], "kind": k, "clean": it["english"]})
        k = "3 mixed invisible changes"
        if len(uni[k]) < 100:
            p = it["english"]
            for kk in rng.sample(UNI, 3):
                p = perturb(p, kk, rng) or p
            uni[k].append({"english": p, "target": it["target"], "kind": k, "clean": it["english"]})
    for v in var.values():
        pool += v
    for k, v in uni.items():
        pool += v
        pool += [{**x, "english": unicodedata.normalize("NFKC", x["english"]), "kind": f"FIX NFKC: {k}"} for x in v]
        pool += [{**x, "english": strip_invisible(x["english"]), "kind": f"FIX NFKC + drop invisible: {k}"} for x in v]
    pool += [{"english": s, "target": None, "kind": "odd inputs"} for s in ODD_INPUTS]
    return pool


# --- hidden-state features for the OOD detector ----------------------------------------------------

@torch.no_grad()
def prompt_features(model, tok, texts: list[str], bsz: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean hidden state over the prompt tokens, at the middle and the last layer."""
    from train.run import prompt
    device = next(model.parameters()).device
    tok.padding_side = "right"
    mids, lasts = [], []
    for s in range(0, len(texts), bsz):
        enc = tok([prompt(t) for t in texts[s:s + bsz]], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float()
        for layer, dst in ((len(hs) // 2, mids), (len(hs) - 1, lasts)):
            h = hs[layer].float()
            dst.append(((h * m).sum(1) / m.sum(1)).cpu())
        del hs
    return torch.cat(mids), torch.cat(lasts)


def mahalanobis(train: torch.Tensor, x: torch.Tensor) -> list[float]:
    mu = train.mean(0)
    c = train - mu
    cov = c.T @ c / len(train)
    cov += 1e-3 * cov.diagonal().mean() * torch.eye(cov.shape[0])          # shrinkage: 2,000 points, 1,024 dims
    prec = torch.linalg.inv(cov.double()).float()
    d = x - mu
    return ((d @ prec) * d).sum(1).tolist()


# --- per-model process ---------------------------------------------------------------------------------

def part(idx: str) -> None:
    from train.load import load_trained
    ctx = Ctx()
    label, run = list(saved_models().items())[int(idx)]
    pool = build_pool(ctx)
    model, tok, _ = load_trained(RUNS / run / "model")
    texts = sorted({it["english"] for it in pool})
    print(f"[{label}] generate {len(texts)} distinct inputs", flush=True)
    outs = dict(zip(texts, generate(model, tok, [{"english": t, "target": ""} for t in texts])))
    preds = [outs[it["english"]] for it in pool]
    print(f"[{label}] score own outputs", flush=True)
    own = score(model, tok, [{"english": it["english"], "target": p} for it, p in zip(pool, preds)])
    print(f"[{label}] hidden-state features", flush=True)
    rng = random.Random(SEED + 1)
    fit = [it["english"] for it in rng.sample(ctx.train, 2000)]
    fm, fl = prompt_features(model, tok, fit)
    xm, xl = prompt_features(model, tok, texts)
    dm = dict(zip(texts, mahalanobis(fm, xm)))
    dl = dict(zip(texts, mahalanobis(fl, xl)))
    recs = []
    for it, p, s in zip(pool, preds, own):
        tl = s["tok_logp"] or [-99.0]
        recs.append({"kind": it["kind"], "english": it["english"],
                     "right": None if it["target"] is None else p == it["target"],
                     "logp": sum(tl), "cmin": min(tl), "cmean": sum(tl) / len(tl),
                     "maha_mid": dm[it["english"]], "maha_last": dl[it["english"]]})
    PARTS.mkdir(parents=True, exist_ok=True)
    (PARTS / f"model{idx}.json").write_text(json.dumps({"label": label, "recs": recs}), encoding="utf-8")


# --- 27. unseen local structure --------------------------------------------------------------------------

def word_classes(ctx: Ctx) -> dict[str, str]:
    v = ctx.vocab
    cls = {}
    for n, info in v.noun_info.items():
        cls[n] = cls[info[0]] = "NOUN"
    for n in v.locations:
        cls[n] = "NOUN"
    for a in v.adjectives:
        cls[a] = "ADJ"
    for vb, forms in v.verb_info.items():
        for x in forms:
            cls[x] = "VERB"
    for p in PRONOUNS.values():
        cls[str(p[0]).lower()] = "PRON"
    return cls


def flags_for(ctx: Ctx):
    cls = word_classes(ctx)
    tr_w, tr_b, tr_c2, tr_c3 = set(), set(), set(), set()

    def classes(ws):
        return [cls.get(w, w if w in tr_w_func else "UNK") for w in ws]
    tr_w_func = {w for it in ctx.train for w in it["english"].split() if w not in cls}
    for it in ctx.train:
        ws = it["english"].split()
        tr_w.update(ws)
        tr_b.update(zip(ws, ws[1:]))
        c = classes(ws)
        tr_c2.update(zip(c, c[1:]))
        tr_c3.update(zip(c, c[1:], c[2:]))

    def flags(s: str) -> dict[str, bool]:
        ws = s.split()
        c = classes(ws)
        return {"unseen word": any(w not in tr_w for w in ws),
                "unseen word pair": any(b not in tr_b for b in zip(ws, ws[1:])),
                "unseen word-class pair": any(b not in tr_c2 for b in zip(c, c[1:])),
                "unseen word-class triple": any(b not in tr_c3 for b in zip(c, c[1:], c[2:]))}
    return flags


FLAGS = ["unseen word", "unseen word pair", "unseen word-class pair", "unseen word-class triple"]


# --- reports ------------------------------------------------------------------------------------------

def ece(conf: list[float], right: list[bool], bins: int = 10) -> tuple[float, list]:
    b = defaultdict(list)
    for c, r in zip(conf, right):
        b[min(int(c * bins), bins - 1)].append((c, r))
    e, table = 0.0, []
    for k in range(bins):
        if b[k]:
            mc = sum(c for c, _ in b[k]) / len(b[k])
            ac = sum(r for _, r in b[k]) / len(b[k])
            e += len(b[k]) / len(conf) * abs(mc - ac)
            table.append((k / bins, (k + 1) / bins, len(b[k]), mc, ac))
    return e, table


def platt(x: list[float], y: list[bool]) -> tuple[float, float]:
    a, c = 1.0, 0.0
    for _ in range(200):                                     # Newton steps for logistic regression on one feature
        ga = gc = haa = hac = hcc = 0.0
        for xi, yi in zip(x, y):
            p = 1 / (1 + math.exp(-max(-30, min(30, a * xi + c))))
            g = p - yi
            w = max(p * (1 - p), 1e-6)
            ga += g * xi; gc += g; haa += w * xi * xi; hac += w * xi; hcc += w
        det = haa * hcc - hac * hac
        if abs(det) < 1e-12:
            break
        a -= (hcc * ga - hac * gc) / det
        c -= (haa * gc - hac * ga) / det
    return a, c


def coverage_at(conf: list[float], right: list[bool], target: float) -> float:
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    best, ok = 0.0, 0
    for n, i in enumerate(order, 1):
        ok += right[i]
        if ok / n >= target:
            best = n / len(conf)
    return best


def aurc(conf: list[float], right: list[bool]) -> float:
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    ok, tot = 0, 0.0
    for n, i in enumerate(order, 1):
        ok += right[i]
        tot += 1 - ok / n
    return tot / len(conf)


def report() -> None:
    parts = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(PARTS.glob("model*.json"))]
    if not parts:
        return
    labels = [p["label"] for p in parts]
    hdr = " | ".join(labels)
    sep = "---|" * len(labels)
    ctx = Ctx()
    flags = flags_for(ctx)

    # 27
    L = ["# Test 27: does an input fail exactly when it contains something never seen in the training English?", "",
         "Pool: normal sentences, the 4 test splits, every failure kind of tests 17-19, and the invisible-Unicode inputs "
         "(the FIX rows and odd inputs excluded). Word classes: NOUN / VERB / ADJ / PRON from the lexicon, function words "
         "kept as themselves, anything else UNK (Bogin et al. 2022, 'unobserved local structures').", "",
         "Cells: share of inputs flagged / accuracy when flagged / accuracy when NOT flagged / share of all failures "
         "that are flagged (recall) / share of flagged inputs that fail (precision).", "",
         f"| flag | {hdr} |", f"|---|{sep}"]
    for fl in FLAGS:
        cells = []
        for p in parts:
            rs = [r for r in p["recs"] if r["right"] is not None and not r["kind"].startswith("FIX")]
            f = [flags(r["english"])[fl] for r in rs]
            y = [r["right"] for r in rs]
            nf = sum(f)
            acc_f = sum(yy for ff, yy in zip(f, y) if ff) / nf if nf else float("nan")
            acc_n = sum(yy for ff, yy in zip(f, y) if not ff) / (len(f) - nf) if len(f) > nf else float("nan")
            fails = [not yy for yy in y]
            rec = sum(ff and fa for ff, fa in zip(f, fails)) / sum(fails) if sum(fails) else float("nan")
            prec = sum(ff and fa for ff, fa in zip(f, fails)) / nf if nf else float("nan")
            cells.append(f"{pct(nf / len(f))} / {pct(acc_f)} / {pct(acc_n)} / {pct(rec)} / {pct(prec)}")
        L.append(f"| {fl} | " + " | ".join(cells) + " |")
    p0 = parts[0]
    kinds = list(dict.fromkeys(r["kind"] for r in p0["recs"] if r["right"] is not None and not r["kind"].startswith("FIX")))
    L += ["", f"## Per input kind ({labels[0]})", "",
          "| input kind | n | accuracy | " + " | ".join(f"flag: {fl}" for fl in FLAGS) + " |", "|---|---|---|" + "---|" * len(FLAGS)]
    for k in kinds:
        rs = [r for r in p0["recs"] if r["kind"] == k]
        fs = [flags(r["english"]) for r in rs]
        L.append(f"| {k} | {len(rs)} | {pct(sum(r['right'] for r in rs) / len(rs))} | " +
                 " | ".join(pct(sum(f[fl] for f in fs) / len(fs)) for fl in FLAGS) + " |")
    write("test27_unseen_structure", L)

    # 28
    L = ["# Test 28: can the INPUT alone tell that the English is off-template? (Mahalanobis distance)", "",
         "Hidden states of the prompt (mean over its tokens) at the middle and the last layer; a Gaussian is fitted on "
         "2,000 training inputs. AUROC = chance an input of this kind is further from the training inputs than an "
         "in-distribution one (normal + the 4 test splits). 1.0 = perfectly separated, 0.5 = invisible.", "",
         f"| input kind | {hdr} |", f"|---|{sep}"]
    kinds_all = list(dict.fromkeys(r["kind"] for r in p0["recs"] if not r["kind"].startswith("FIX")))
    for k in kinds_all:
        if k in ID_KINDS:
            continue
        cells = []
        for p in parts:
            idd = [r for r in p["recs"] if r["kind"] in ID_KINDS]
            oo = [r for r in p["recs"] if r["kind"] == k]
            a_m = auroc([r["maha_mid"] for r in oo + idd], [True] * len(oo) + [False] * len(idd))
            a_l = auroc([r["maha_last"] for r in oo + idd], [True] * len(oo) + [False] * len(idd))
            acc = [r["right"] for r in oo if r["right"] is not None]
            cells.append(f"{a_m:.2f} / {a_l:.2f}" + (f" (acc {pct(sum(acc) / len(acc))})" if acc else ""))
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += ["", "Cells: AUROC middle layer / last layer (model accuracy on that kind).", "",
          "## Predicting a WRONG output (all inputs with a known answer, FIX rows excluded)", "",
          f"| signal | {hdr} |", f"|---|{sep}"]
    for name, fn in (("input distance, middle layer", lambda r: r["maha_mid"]),
                     ("input distance, last layer", lambda r: r["maha_last"]),
                     ("output confidence (lowest token prob), inverted", lambda r: -r["cmin"])):
        cells = []
        for p in parts:
            rs = [r for r in p["recs"] if r["right"] is not None and not r["kind"].startswith("FIX")]
            cells.append(f"{auroc([fn(r) for r in rs], [not r['right'] for r in rs]):.2f}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    cells = []
    for p in parts:
        rs = [r for r in p["recs"] if r["right"] is not None and not r["kind"].startswith("FIX")]

        def rank(v):
            o = sorted(range(len(v)), key=lambda i: v[i])
            rk = [0.0] * len(v)
            for j, i in enumerate(o):
                rk[i] = j / len(v)
            return rk
        comb = [a + b for a, b in zip(rank([r["maha_mid"] for r in rs]), rank([-r["cmin"] for r in rs]))]
        cells.append(f"{auroc(comb, [not r['right'] for r in rs]):.2f}")
    L.append("| input distance (middle) + output confidence, rank average | " + " | ".join(cells) + " |")
    write("test28_input_ood_detector", L)

    # 29
    L = ["# Test 29: calibration and selective translation", "",
         "Inputs with a known answer (normal, test splits, failure kinds, rewordings, invisible Unicode; FIX rows "
         "excluded). Confidence = probability of the whole output (product of token probabilities) or its lowest token "
         "probability. Platt scaling is fitted on even-numbered inputs and evaluated on odd-numbered ones.", "",
         f"| measure | {hdr} |", f"|---|{sep}"]
    res = {}
    for p in parts:
        rs = [r for r in p["recs"] if r["right"] is not None and not r["kind"].startswith("FIX")]
        y = [r["right"] for r in rs]
        seq = [math.exp(r["logp"]) for r in rs]
        mn = [math.exp(r["cmin"]) for r in rs]
        dev, ev = rs[0::2], rs[1::2]
        a, c = platt([r["logp"] for r in dev], [r["right"] for r in dev])
        cal = [1 / (1 + math.exp(-max(-30, min(30, a * r["logp"] + c)))) for r in ev]
        idd = [r for r in rs if r["kind"] in ID_KINDS]
        res[p["label"]] = {
            "accuracy": pct(sum(y) / len(y)),
            "ECE, whole-output probability": f"{ece(seq, y)[0]:.3f}",
            "ECE, lowest-token probability": f"{ece(mn, y)[0]:.3f}",
            "ECE after Platt scaling (held-out half)": f"{ece(cal, [r['right'] for r in ev])[0]:.3f}",
            "coverage at 99% accuracy (rank by lowest token)": pct(coverage_at([r["cmin"] for r in rs], y, 0.99)),
            "coverage at 95% accuracy (rank by lowest token)": pct(coverage_at([r["cmin"] for r in rs], y, 0.95)),
            "AURC (lower = better; lowest token)": f"{aurc([r['cmin'] for r in rs], y):.3f}",
            "in-distribution only: accuracy": pct(sum(r["right"] for r in idd) / len(idd)),
            "in-distribution only: coverage at 99% accuracy": pct(coverage_at([r["cmin"] for r in idd],
                                                                              [r["right"] for r in idd], 0.99)),
            "in-distribution only: ECE (whole output)": f"{ece([math.exp(r['logp']) for r in idd], [r['right'] for r in idd])[0]:.3f}",
        }
    for m in next(iter(res.values())):
        L.append(f"| {m} | " + " | ".join(res[l][m] for l in labels) + " |")
    rs = [r for r in p0["recs"] if r["right"] is not None and not r["kind"].startswith("FIX")]
    _, tab = ece([math.exp(r["logp"]) for r in rs], [r["right"] for r in rs])
    L += ["", f"## Reliability table ({labels[0]}, whole-output probability)", "",
          "| confidence bin | n | mean confidence | accuracy |", "|---|---|---|---|"]
    L += [f"| {lo:.1f}-{hi:.1f} | {n} | {mc:.2f} | {pct(ac)} |" for lo, hi, n, mc, ac in tab]
    write("test29_calibration", L)

    # 30
    L = ["# Test 30: invisible Unicode changes (look the same on screen)", "",
         "100 new sentences per kind, one change each (or 3 mixed). FIX rows: the same inputs after NFKC normalisation, "
         "and after NFKC + removing invisible format characters. Cells: accuracy (share of inputs restored to the exact "
         "original text by the fix).", "", f"| change | {hdr} |", f"|---|{sep}"]
    for k in UNI + ["3 mixed invisible changes"]:
        for kk in (k, f"FIX NFKC: {k}", f"FIX NFKC + drop invisible: {k}"):
            cells = []
            for p in parts:
                rs = [r for r in p["recs"] if r["kind"] == kk]
                cells.append(pct(sum(r["right"] for r in rs) / len(rs)) if rs else "-")
            restored = ""
            if kk.startswith("FIX"):
                base = [r for r in parts[0]["recs"] if r["kind"] == k]
                fixed = [r for r in parts[0]["recs"] if r["kind"] == kk]
                cl = build_clean_map()
                restored = f" (restored {pct(sum(cl.get((k, i)) == f['english'] for i, f in enumerate(fixed)) / len(fixed))})" \
                    if fixed else ""
            L.append(f"| {'   ' if kk.startswith('FIX') else ''}{kk}{restored} | " + " | ".join(cells) + " |")
    write("test30_invisible_unicode", L)


_CLEAN = None


def build_clean_map():
    """(kind, index) -> the clean English of that perturbed input (rebuilt from the deterministic pool)."""
    global _CLEAN
    if _CLEAN is None:
        pool = build_pool(Ctx())
        _CLEAN, cnt = {}, defaultdict(int)
        for it in pool:
            if "clean" in it and not it["kind"].startswith("FIX"):
                _CLEAN[(it["kind"], cnt[it["kind"]])] = it["clean"]
                cnt[it["kind"]] += 1
    return _CLEAN


def run_all() -> None:
    for i in range(len(saved_models())):
        if (PARTS / f"model{i}.json").exists():
            continue
        r = subprocess.run([sys.executable, "-X", "utf8", "-X", "faulthandler", str(Path(__file__).resolve()), "part", str(i)])
        print(f"model {i} exit {r.returncode}", flush=True)
    report()


if __name__ == "__main__":
    {"all": run_all, "report": report, "part": lambda: part(sys.argv[2])}[sys.argv[1]]()
