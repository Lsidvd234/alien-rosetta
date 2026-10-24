"""Data for the model viewer (analysis/viewer.html): step graphs, model maps, coverage heatmaps, glyph embedding maps.

    python analysis/viz_export.py            -> results/viz/viewer_data.js  (open analysis/viewer.html)

Uses the cached predictions and rebuilt training pieces of analysis/subgraph_check.py (run that first for every model
to show), the minimal-pair results of analysis/viz_pairs.py if present, and the saved glyph-token tables. Generated
test sentences only; never the sealed set.

Step graph of one sentence (three layers): English words -> meaning fields (S=, V=, O= ... of the analysis) -> glyph
words. Gold structure from the generator; the model's analysis and glyphs are compared field by field and word by word.
"Seen in training" flags: field shape, clause template, verb ending stack, exact glyph word form. English -> field
links are matched by word (approximate for synonyms, contractions); field -> glyph word links come from the language
itself (exact).
"""

from __future__ import annotations

import collections
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "sealed" / "app"))

import train.represent_v2 as R2  # noqa: E402
from freq_weighted_eval import english_of, test_items, word_keys  # noqa: E402
from subgraph_check import OUT as SG, clause_parts, training_pieces  # noqa: E402

OUT = ROOT / "results" / "viz"
FIELD_RE = re.compile(r"(?<![A-Za-z0-9_.>^+&|])([A-Z]+)=(\S+)")
LABEL = {"e2b-v2fix-cot-80k-noheld": "baseline (meaning first, data fix)",
         "e2b-v2fix-cot-80k": "held-out (some English forms never trained)",
         "e2b-v2fix-focal-r2": "focal loss", "e2b-v2fix-loan-r2": "loanwords", "e2b-v2fix-para": "paraphrase",
         "e2b-v2-cot-80k-noheld": "before data fix", "e2a-v2-80k": "direct (no meaning step)"}
PRON_EN = {"1sg": "i me my mine myself", "1pl": "we us our ours ourselves", "2sg": "you your yours yourself",
           "2pl": "you your yours yourselves", "3sgm": "he him his himself", "3sgf": "she her hers herself",
           "3sgn": "it its itself", "3sgh": "he she him her", "3pl": "they them their theirs themselves"}
VMOD_EN = {"PAST": "did was were had", "FUT": "will 'll shall going", "PERF": "have has had having",
           "PROG": "is are am was were be been being", "NEG": "not n't no never", "CAN": "can could",
           "MUST": "must", "SHOULD": "should", "MAY": "may might", "WOULD": "would 'd", "LET": "let let's",
           "Q": "do does did", "PRES": "is are am do does"}
CASE_EN = {"L": "in at on inside", "TO": "to into", "FRONT": "front", "THAN": "than"}


def translit(word: str, lat: dict, cons: set, signs: set, virama: str) -> str:
    out, i = [], 0
    while i < len(word):
        ch = word[i]
        if ch in cons:
            nxt = word[i + 1] if i + 1 < len(word) else ""
            if nxt in signs:
                out.append(lat[ch] + lat[nxt]); i += 2
            elif nxt == virama:
                out.append(lat[ch]); i += 2
            else:
                out.append(lat[ch] + "a"); i += 1
        else:
            out.append(lat.get(ch, "?")); i += 1
    return "".join(out)


def tokens_of(value: str) -> list[str]:
    toks = []
    for t in re.split(r"[\^&|+]", value):
        t = re.sub(r"\.(PL)$", "", t)
        if t.startswith(("PRON.", "NAME.", "LOAN.")):
            toks.append(t.split(".", 1)[1].lower())
        else:
            toks.append(t.split(".")[0].lower())
    return [t for t in toks if t]


def english_candidates(field: str, value: str, eng: dict, gen) -> set[str]:
    """English words that can express this field (for the approximate English -> field links)."""
    out = set()
    if field == "V":
        parts = value.split(".")
        out |= verb_forms(parts[0], gen)
        for m in parts[1:]:
            out |= set(VMOD_EN.get(m, "").split())
        return out
    out |= set(CASE_EN.get(field, field.lower()).split()) if field not in "SOPMAITYC" else set()
    if "&" in value:
        out.add("and")
    if "|" in value:
        out.add("or")
    for t in re.split(r"[\^&|+]", value):
        base = re.sub(r"\.PL$", "", t)
        if base.startswith("PRON."):
            out |= set(PRON_EN.get(base[5:], "").split())
        elif base.startswith(("NAME.", "LOAN.")):
            w = base.split(".", 1)[1].lower()
            out |= {w, w + "s", w + "es"}
        else:
            k = base.lower()
            w = eng.get(k, k.split(">")[0].replace("_", " "))
            for x in w.split():
                out |= {x, x + "s", x + "es", x + "er", x + "est", x + "ly"}
                out |= {R2.FORMS["comparative"].get(x, ""), R2.FORMS["superlative"].get(x, "")}
            out |= noun_plural(k, gen)
    return {x for x in out if x}


_NOUN_PL: dict = {}
_VERB: dict = {}


def noun_plural(key: str, gen) -> set:
    if not _NOUN_PL:
        for sg, pl, k, _ in gen.nouns + gen.places:
            _NOUN_PL.setdefault(k, set()).update(x for x in (sg, pl) if x)
    return _NOUN_PL.get(key, set())


def verb_forms(key: str, gen) -> set:
    if not _VERB:
        for v in gen.verbs:
            _VERB.setdefault(v[1], set()).update(x for x in v[:5] if isinstance(x, str))
            _VERB[v[1]].add(R2.FORMS["ing"].get(v[0], v[0] + "ing"))
    return _VERB.get(key, {key, key + "s", key + "ed", key + "ing"})


# --- one test sentence: gold structure -----------------------------------------------------------------------
def gold_structure(it: dict, ex: dict, gen, eng: dict, parts: dict, forms, keys, lat_word) -> dict:
    clauses = []
    words = []
    for ci, (cl, ana) in enumerate(zip(ex["clauses"], [c.strip() for c in it["analysis"].split(";")])):
        fields = FIELD_RE.findall(ana)
        cparts = clause_parts(ana)
        tmpl = next(p for k, p in cparts if k == "clause template")
        shapes = [p for k, p in cparts if k == "slot shape"]
        fl = []
        si = 0
        for f, v in fields:
            seen = True
            if f == "V":
                vend = ".".join(v.split(".")[1:]) or "-"
                seen = vend in parts.get("verb endings", set())
            else:
                seen = shapes[si] in parts.get("slot shape", set()) if si < len(shapes) else True
                si += 1
            fl.append({"f": f, "v": v, "seen": seen, "tok": tokens_of(v)})
        clauses.append({"fields": fl, "template": tmpl, "template_seen": tmpl in parts.get("clause template", set())})
        # glyph words of this clause -> fields
        cw = gen.rep.words(cl)
        kinds = word_keys(gen.rep, cl)
        last = None
        pending = []
        for w, (kind, key) in zip(cw, kinds):
            g = gen.rep.script.word(w.phonemes)
            head = w.gloss.split("-")[0].replace(".PST", "")
            hl = head.lower().replace("+", "_")
            fi = None
            for j, fd in enumerate(fl):
                if hl in fd["tok"] or key.lower() in fd["tok"] or (kind == "pron" and f"pron.{head}".lower() in
                                                                    [("pron." + t) for t in fd["tok"]]):
                    fi = j
                    if j == last:
                        break
            if fi is None and head.isupper():
                fi = next((j for j, fd in enumerate(fl) if fd["f"] == head or (head == "LOC" and fd["f"] == "L")),
                          None)
            if fi is None and head in ("in", "LOC"):
                fi = next((j for j, fd in enumerate(fl) if fd["f"] == "L"), None)
            if fi is None and (head in ("will", "not") or head in VMOD_EN):
                fi = next((j for j, fd in enumerate(fl) if fd["f"] == "V"), None)
            rec = {"latin": lat_word(g), "glyph": g, "gloss": w.gloss, "kind": kind, "key": key,
                   "count": keys.get(key, 0) if kind == "word" else None, "form_seen": forms[g] > 0,
                   "clause": ci, "field": fi}
            if fi is None:
                pending.append(rec)                          # article etc.: joins the next word's field
            else:
                for p in pending:
                    p["field"] = fi
                pending = []
                last = fi
            words.append(rec)
        for p in pending:
            p["field"] = last
    # English tokens -> fields (approximate)
    toks = re.findall(r"[A-Za-z]+(?:'[a-z]+)?|[^\sA-Za-z]", it["english"])
    links = []
    for ti, t in enumerate(toks):
        tl = t.lower()
        for ci, c in enumerate(clauses):
            hit = next((j for j, fd in enumerate(c["fields"])
                        if tl in english_candidates(fd["f"], fd["v"], eng, gen)), None)
            if hit is not None:
                links.append([ti, ci, hit])
                break
    return {"english": it["english"], "tokens": toks, "links": links, "clauses": clauses, "words": words,
            "analysis": it["analysis"], "glyphs": it["target"]}


# --- one model's answer ------------------------------------------------------------------------------------
def model_answer(gold: dict, pred: str, cot: bool, lat_word) -> dict:
    ana_p, _, gly_p = pred.partition("=>") if cot else ("", "", pred)
    ana_p, gly_p = ana_p.strip(), gly_p.strip()
    res = {"analysis": ana_p if cot else None, "glyph_exact": gly_p == gold["glyphs"],
           "analysis_exact": (ana_p == gold["analysis"]) if cot else None}
    # fields
    pcl = [FIELD_RE.findall(c) for c in ana_p.split(";")] if cot else []
    fstat = []
    for ci, c in enumerate(gold["clauses"]):
        pf = dict(pcl[ci]) if ci < len(pcl) else {}
        row = []
        for fd in c["fields"]:
            pv = pf.get(fd["f"])
            row.append("right" if pv == fd["v"] else ("missing" if pv is None else f"wrong:{pv}"))
        extra = [f"{f}={v}" for f, v in (pcl[ci] if ci < len(pcl) else []) if f not in {x["f"] for x in c["fields"]}]
        fstat.append({"fields": row, "extra": extra})
    res["fields"] = fstat if cot else None
    if cot and len(pcl) > len(gold["clauses"]):
        res["extra_clauses"] = len(pcl) - len(gold["clauses"])
    # glyph words (aligned)
    gw = [w["glyph"] for w in gold["words"]]
    pw = gly_p.split()
    wst = [None] * len(gw)
    extra_words = []
    for op, a1, a2, b1, b2 in difflib.SequenceMatcher(a=gw, b=pw, autojunk=False).get_opcodes():
        if op == "equal":
            for i in range(a1, a2):
                wst[i] = {"s": "right"}
        elif op == "replace":
            for k, i in enumerate(range(a1, a2)):
                j = b1 + k
                wst[i] = {"s": "wrong", "pred": lat_word(pw[j])} if j < b2 else {"s": "missing"}
            extra_words += [lat_word(x) for x in pw[b1 + (a2 - a1):b2]]
        elif op == "delete":
            for i in range(a1, a2):
                wst[i] = {"s": "missing"}
        else:
            extra_words += [lat_word(x) for x in pw[b1:b2]]
    res["words"] = wst
    res["extra_words"] = extra_words
    if cot:
        if res["analysis_exact"]:
            res["error"] = "correct" if res["glyph_exact"] else "local"
        elif res["glyph_exact"]:
            res["error"] = "restoration"
        else:
            res["error"] = "meaning wrong"           # propagation / both wrong: refined from subgraph_check below
    else:
        res["error"] = "correct" if res["glyph_exact"] else "glyphs wrong"
    return res


# --- model map (step types) ----------------------------------------------------------------------------------
STEP_ORDER = ["S", "V: verb", "V: tense", "V: helpers", "O", "P", "L", "M", "A", "T", "case fields", "other fields",
              "word: noun/verb/adj", "word: built (endings)", "word: compound", "word: pronoun", "word: name",
              "word: grammar", "word form: seen", "word form: new"]


def step_stats(sents: list[dict], model: str) -> dict:
    acc = collections.defaultdict(lambda: [0, 0])
    for s in sents:
        a = s["models"].get(model)
        if not a:
            continue
        if a["fields"]:
            for c, fs in zip(s["clauses"], a["fields"]):
                for fd, st in zip(c["fields"], fs["fields"]):
                    f = fd["f"]
                    if f == "V":
                        pv = st[6:] if st.startswith("wrong:") else (fd["v"] if st == "right" else "")
                        gp, pp = fd["v"].split("."), pv.split(".") if pv else []
                        _add(acc, "V: verb", bool(pp) and pp[0] == gp[0])
                        tense = [x for x in gp[1:] if x in ("PRES", "PAST", "FUT")]
                        ptense = [x for x in pp[1:] if x in ("PRES", "PAST", "FUT")]
                        if tense:
                            _add(acc, "V: tense", tense == ptense)
                        helpers = sorted(x for x in gp[1:] if x not in ("PRES", "PAST", "FUT"))
                        if helpers:
                            _add(acc, "V: helpers", helpers == sorted(x for x in pp[1:] if x not in
                                                                       ("PRES", "PAST", "FUT")))
                    else:
                        name = f if f in ("S", "O", "P", "L", "M", "A", "T") else (
                            "case fields" if f in R2_CASES else "other fields")
                        _add(acc, name, st == "right")
        for w, st in zip(s["words"], a["words"]):
            ok = st["s"] == "right"
            k = w["kind"]
            if k == "word":
                name = "word: compound" if "_" in w["key"] else ("word: built (endings)" if ">" in w["key"]
                                                                  else "word: noun/verb/adj")
                _add(acc, "word form: seen" if w["form_seen"] else "word form: new", ok)
            else:
                name = {"pron": "word: pronoun", "name": "word: name"}.get(k, "word: grammar")
            _add(acc, name, ok)
    return {k: {"right": v[0], "n": v[1]} for k, v in acc.items()}


def _add(acc, k, ok):
    acc[k][0] += int(ok)
    acc[k][1] += 1


R2_CASES = {"ON", "UNDER", "TO", "FROM", "WITH", "FOR", "NEAR", "BEHIND", "FRONT", "ABOUT", "THAN", "WITHOUT",
            "BEFORE", "AFTER", "THROUGH", "AROUND", "BETWEEN", "DURING", "UNTIL", "SINCE", "AGAINST"}


def train_step_counts(cache: Path) -> dict:
    """How often each step type occurs in the baseline's training data (same generator for every v2 run)."""
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    items, _ = R2.build_training("tamil_like", "abugida_pua", 0, n=80000, n_refuse=2500, n_replay=0, cot=False,
                                 held_frac=0.0, min_count=12)
    c = collections.Counter()
    for it in items:
        if it.get("aug") != "rep2":
            continue
        for cl in it["analysis"].split(";"):
            for f, v in FIELD_RE.findall(cl):
                if f == "V":
                    gp = v.split(".")
                    c["V: verb"] += 1
                    c["V: tense"] += any(x in ("PRES", "PAST", "FUT") for x in gp[1:])
                    c["V: helpers"] += any(x not in ("PRES", "PAST", "FUT") for x in gp[1:])
                else:
                    c[f if f in ("S", "O", "P", "L", "M", "A", "T") else
                      ("case fields" if f in R2_CASES else "other fields")] += 1
    cache.write_text(json.dumps(c), encoding="utf-8")
    return dict(c)


# --- glyph embeddings ------------------------------------------------------------------------------------------
EMB_RUNS = {"s1-hard-latininit-50k": "starts at its sound", "s4-hard-shuffledinit-50k": "starts at a WRONG sound",
            "s3-hard-plaintext-50k": "starts as a random word piece (no meaning)",
            "e2b-v2fix-cot-80k-noheld": "v2 baseline (starts at its sound)"}


def embeddings() -> dict:
    import numpy as np
    import torch
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    path = hf_hub_download("Qwen/Qwen3-0.6B-Base", "model.safetensors")
    with safe_open(path, "pt") as f:
        name = next(k for k in f.keys() if k.endswith("embed_tokens.weight"))
        E = f.get_tensor(name).float().numpy()
    out = {}
    for run, label in EMB_RUNS.items():
        p = ROOT / "results" / "kaggle" / run / "model" / "glyph_tokens.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        sounds = d.get("sounds") or ["?"] * len(d["chars"])
        fin, ini = d["delta"].float().numpy(), d["delta_init"].float().numpy()
        snd = sorted({s for s in sounds if s and s != "?"})
        S = np.stack([E[tok(s, add_special_tokens=False)["input_ids"]].mean(0) for s in snd])
        Sn = S / np.linalg.norm(S, axis=1, keepdims=True)

        def ranks(X):
            Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
            sim = Xn @ Sn.T
            r = []
            for i, s in enumerate(sounds):
                if s in snd:
                    order = np.argsort(-sim[i])
                    r.append(int(np.where(order == snd.index(s))[0][0]) + 1)
            return r
        both = np.concatenate([ini, fin])
        mu = both.mean(0)
        u, sv, vt = np.linalg.svd(both - mu, full_matrices=False)
        P = (both - mu) @ vt[:2].T
        n = len(sounds)
        rf, ri = ranks(fin), ranks(ini)

        def nearest(X):
            Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
            return (Xn @ Sn.T).argmax(1)
        stay = float((nearest(fin) == nearest(ini)).mean())       # keeps the sound it started at (right or wrong)
        out[run] = {"label": label, "sounds": sounds, "start": P[:n].round(4).tolist(), "end": P[n:].round(4).tolist(),
                    "rank_end": rf, "rank_start": ri, "n_sounds": len(snd), "stay": stay,
                    "top1_end": sum(x == 1 for x in rf) / max(1, len(rf)),
                    "top1_start": sum(x == 1 for x in ri) / max(1, len(ri)),
                    "moved": float(np.linalg.norm(fin - ini, axis=1).mean() / np.linalg.norm(ini, axis=1).mean())}
    return out


# --- main ----------------------------------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    models = [m for m in LABEL if (SG / f"{m}.preds.json").exists() and (SG / f"{m}.pieces.json").exists()]
    print("models:", models)
    gen = R2.GenV2("tamil_like", "abugida_pua", 0)
    eng = english_of(gen)
    lat = gen.rep.script.latin_of_glyphs()
    sc = gen.rep.script
    cons, signs = set(sc.cons.values()), {g for g in sc.sign.values() if g}
    lat_word = lambda w: translit(w, lat, cons, signs, sc.virama)  # noqa: E731
    base = "e2b-v2fix-cot-80k-noheld"
    parts, _, _, (forms, keys) = training_pieces({}, SG / f"{base}.pieces.json")
    sents: dict[str, dict] = {}
    pool = test_items(gen, 400)
    for m in models:
        cfg = json.loads((ROOT / "results" / "kaggle" / m / "model" / "run_args.json").read_text(encoding="utf-8"))
        cot = str(cfg.get("cot")) == "True"
        _, train_en, _, _ = training_pieces(cfg, SG / f"{m}.pieces.json")
        tests = [(it, ex) for it, ex in pool if it["english"] not in train_en][:300]
        preds = json.loads((SG / f"{m}.preds.json").read_text(encoding="utf-8"))
        sg = json.loads((SG / f"{m}.json").read_text(encoding="utf-8")) if (SG / f"{m}.json").exists() else None
        errs = None
        if sg and cot:                     # exact error type (propagation vs both wrong) from subgraph_check order
            errs = recompute_errors(tests, preds, gen)
        for i, ((it, ex), p) in enumerate(zip(tests, preds)):
            if it["english"] not in sents:
                sents[it["english"]] = {**gold_structure(it, ex, gen, eng, parts, forms, keys, lat_word), "models": {}}
            a = model_answer(sents[it["english"]], p, cot, lat_word)
            if errs:
                a["error"] = errs[i]
            sents[it["english"]]["models"][m] = a
        print(f"  {m}: {len(tests)} sentences")
    S = list(sents.values())
    steps = {m: step_stats(S, m) for m in models}
    counts = train_step_counts(OUT / "train_step_counts.json")
    coverage = {}
    for m in models:
        sg = json.loads((SG / f"{m}.json").read_text(encoding="utf-8"))
        grid = {}
        for lo, hi, name in ((0, 0, "never"), (1, 9, "1-9"), (10, 29, "10-29"), (30, 99, "30-99"),
                             (100, 10 ** 9, "100+")):
            for seen in (True, False):
                b = [w for w in sg["words"] if w["kind"] == "word" and lo <= w["key_count"] <= hi
                     and w["form_seen"] == seen]
                grid[f"{name}|{'seen form' if seen else 'new form'}"] = [sum(w["right"] for w in b), len(b)]
        struct = {}
        for k in ("clause template", "English variant", "word form"):
            a = [s for s in sg["sentences"] if k not in s["new_kinds"]]
            b = [s for s in sg["sentences"] if k in s["new_kinds"]]
            struct[k] = [[sum(s["right"] for s in a), len(a)], [sum(s["right"] for s in b), len(b)]]
        coverage[m] = {"grid": grid, "struct": struct}
    pairs = json.loads((OUT / "pairs.json").read_text(encoding="utf-8")) if (OUT / "pairs.json").exists() else None
    try:
        emb = embeddings()
    except Exception as e:                                       # keep the rest of the viewer usable
        print("embeddings skipped:", e)
        emb = {}
    data = {"labels": {m: LABEL[m] for m in models}, "models": models, "sentences": S, "steps": steps,
            "step_order": STEP_ORDER, "train_counts": counts, "coverage": coverage, "pairs": pairs, "emb": emb,
            "script": {"lat": lat, "cons": sorted(cons), "signs": sorted(signs), "virama": sc.virama}}
    (OUT / "viewer_data.js").write_text("window.VIZ = " + json.dumps(data, ensure_ascii=False) + ";\n",
                                        encoding="utf-8")
    print(f"wrote {OUT / 'viewer_data.js'} ({(OUT / 'viewer_data.js').stat().st_size / 1e6:.1f} MB), "
          f"{len(S)} sentences")


def recompute_errors(tests, preds, gen) -> list[str]:
    import itertools
    from conlang.v2 import from_analysis
    from english_parse import spell_name, with_gender
    out = []
    for (it, ex), p in zip(tests, preds):
        ana, _, gly = p.partition("=>")
        ana, gly = ana.strip(), gly.strip()
        if ana == it["analysis"]:
            out.append("correct" if gly == it["target"] else "local")
        elif gly == it["target"]:
            out.append("restoration")
        else:
            own = set()
            try:
                cls, _ = from_analysis(ana, spell_name)
                for gs in itertools.product("mf", repeat=min(len(cls), 6)):
                    own.add(" ".join(gen.rep.target(with_gender(c, g)) for c, g in zip(cls, gs + ("m",) * 9)))
            except Exception:
                pass
            out.append("propagation" if gly in own else ("both wrong" if own else "unreadable meaning"))
    return out


if __name__ == "__main__":
    main()
