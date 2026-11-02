"""Does the translator compose, or match pieces it has seen? (Faith and Fate's "linearized subgraph matching", applied to
Alien Rosetta; local, generated sentences only, never the sealed set.)

    python analysis/subgraph_check.py RUN [n_items]      predictions (cached) + analysis -> results/subgraph/<run>.json
    python analysis/subgraph_check.py --table            all analysed runs -> results/subgraph.md + figure

Translation here is a computation with known steps: English -> meaning analysis (fields S V O ..., verb endings) ->
one glyph word per analysed word (root + endings). We rebuild the run's training data exactly (its seed and settings)
and ask, for every piece of a test answer, whether that exact piece occurred in training:

  1. word forms     each gold glyph word: was this exact FORM in a training answer? A new form can still be built
                    from known pieces (known root + known endings, never together). Accuracy of seen vs new forms is
                    compared WITHIN bins of how often the root was trained, so frequency does not explain it away.
  2. sentence parts clause template (which fields, in order), verb ending stack (PAST.PERF.NEG.CAN), slot shapes (a
                    field's form, e.g. O=w+w^w), English variants held out of training (held-out runs only).
                    Sentence accuracy when every part was seen vs when at least one part of a kind was new.
  3. coverage       sentence accuracy against the share of its pieces seen in training (the paper's seen-subgraph plot).
  4. error types    (meaning-first runs) the analysis is an intermediate step we can check: render the model's OWN
                    analysis with the language and compare with its glyphs:
                      local        analysis right, glyphs wrong                (step 2 failed)
                      propagation  analysis wrong, glyphs = its own analysis   (faithful to a wrong step 1)
                      restoration  analysis wrong, glyphs right anyway         (shortcut / memorised answer)
                      both wrong   analysis wrong, glyphs match neither
If the model matched seen subgraphs only, new forms and new parts would be ~0 and errors mostly propagation/
restoration; if it composes, new forms built from known pieces are right about as often as seen ones.
"""

from __future__ import annotations

import collections
import difflib
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "sealed" / "app"))

import train.represent_v2 as R2  # noqa: E402
from freq_weighted_eval import test_items, word_keys  # noqa: E402

OUT = ROOT / "results" / "subgraph"
COUNT_BINS = ((1, 9, "1-9"), (10, 29, "10-29"), (30, 99, "30-99"), (100, 10 ** 9, "100+"))


# --- pieces of an answer ----------------------------------------------------------------------------------------
def clause_parts(analysis: str) -> list[tuple[str, str]]:
    """(kind, piece) for every clause of an analysis: template, verb endings, slot shapes."""
    out = []
    for cl in [c.strip() for c in analysis.split(";") if c.strip()]:
        fields = re.findall(r"(?<![A-Za-z0-9_.>^+&|])([A-Z]+)=(\S+)", cl)
        out.append(("clause template", " ".join(f for f, _ in fields)))
        for f, v in fields:
            if f == "V":
                out.append(("verb endings", ".".join(p for p in v.split(".")[1:]) or "-"))
            else:
                shape = re.sub(r"NAME\.\w+", "N", v)
                shape = re.sub(r"PRON\.\w+", "P", shape)
                shape = re.sub(r"LOAN\.[a-z]+", "L", shape)
                shape = re.sub(r"[a-z][a-z_]*(?:>[A-Z]+)*", "w", shape)
                out.append(("slot shape", f"{f}={shape}"))
    return out


def training_pieces(cfg: dict, cache: Path | None = None) -> tuple[dict, set, set, tuple]:
    """Pieces of the run's rebuilt training data (cached: rebuilding 80k examples takes minutes)."""
    if cache and cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        return ({k: set(v) for k, v in d["parts"].items()}, set(d["train_en"]), set(d["held"]),
                (collections.Counter(d["forms"]), collections.Counter(d["keys"])))
    res = _training_pieces(cfg)
    if cache:
        parts, train_en, held, (forms, keys) = res
        cache.write_text(json.dumps({"parts": {k: sorted(v) for k, v in parts.items()}, "train_en": sorted(train_en),
                                     "held": sorted(held), "forms": forms, "keys": keys}, ensure_ascii=False),
                         encoding="utf-8")
    return res


def _training_pieces(cfg: dict) -> tuple[dict, set, set, tuple]:
    seed = int(cfg.get("seed", 0))
    items, _ = R2.build_training(cfg.get("preset", "tamil_like"), cfg.get("script", "abugida_pua"), seed,
                                 n=int(cfg.get("rep_n", 80000)), n_refuse=int(cfg.get("rep_refuse", 2500)),
                                 n_replay=int(cfg.get("rep_replay", 6000)), cot=False,
                                 held_frac=float(cfg.get("rep_held_frac", 0.2)),
                                 min_count=int(cfg.get("rep_min_count", 0)),
                                 loan_rate=float(cfg.get("rep_loan_rate", 0) or 0),
                                 held_only=None if cfg.get("rep_held_only") in (None, "None", "") else
                                 cfg["rep_held_only"],                                     # structure-dose runs
                                 putback=int(cfg.get("rep_held_putback", 0) or 0),
                                 real_file=None if cfg.get("rep_real") in (None, "None", "") else
                                 str(ROOT / cfg["rep_real"]),
                                 everyday=str(cfg.get("rep_everyday")) == "True",    # real-text runs; --rep-everyday runs
                                 exact_n=str(cfg.get("rep_exact_n")) == "True",
                                 protect_putback=str(cfg.get("rep_protect_putback")) == "True")   # as trained
    R2.LOAN.update(rate=0.0, pool="train")     # build_training set the loan rate: tests use the standard mix
    rep = [it for it in items if it.get("aug") == "rep2"]
    forms = collections.Counter(w for it in rep for w in it["target"].split())
    parts = {}
    for it in rep:
        for k, p in clause_parts(it["analysis"]):
            parts.setdefault(k, set()).add(p)
    from freq_weighted_eval import key_counts
    held = {f"{f}={v}" for f, v in R2.HELD_OUT}
    return parts, {it["english"] for it in items if "english" in it}, held, (forms, key_counts(items))


# --- one run ----------------------------------------------------------------------------------------------------
def analyse(run: str, n: int) -> None:
    from conlang.v2 import from_analysis
    from english_parse import spell_name, with_gender
    name, _, ep = run.partition("@epoch")              # RUN@epochN: that run's epoch-N checkpoint (as score_tatoeba)
    mdir = ROOT / "results" / "kaggle" / name / (f"model_epoch{ep}" if ep else "model")
    cfg = json.loads((mdir / "run_args.json").read_text(encoding="utf-8"))
    R2.language_of_run(cfg)                           # sound-rule runs: their own language and held-out words
    cot = str(cfg.get("cot")) == "True"
    OUT.mkdir(parents=True, exist_ok=True)
    parts, train_en, held, (forms, keys) = training_pieces(cfg, OUT / f"{run}.pieces.json")
    gen = R2.GenV2(cfg.get("preset", "tamil_like"), cfg.get("script", "abugida_pua"), int(cfg.get("seed", 0)))
    tests = [(it, ex) for it, ex in test_items(gen, n + 50) if it["english"] not in train_en][:n]
    ppath = OUT / f"{run}.preds.json"
    if ppath.exists():
        preds = json.loads(ppath.read_text(encoding="utf-8"))
        if isinstance(preds, dict):                    # newer caches keep the English to check the alignment
            assert preds["english"] == [it["english"] for it, _ in tests], f"{ppath.name}: test list changed"
            preds = preds["preds"]
        else:
            assert len(preds) == len(tests), f"{ppath.name}: {len(preds)} saved predictions, {len(tests)} tests"
    else:
        from train.load import load_trained, translate
        model, tok, _ = load_trained(mdir)
        preds = []
        for i in range(0, len(tests), 16):
            preds += translate(model, tok, [it["english"] for it, _ in tests[i:i + 16]], max_new_tokens=700)
            print(f"  {len(preds)}/{len(tests)}", flush=True)
        ppath.write_text(json.dumps({"english": [it["english"] for it, _ in tests], "preds": preds},
                                    ensure_ascii=False), encoding="utf-8")
        del model

    words, sents, errors = [], [], collections.Counter()
    for (it, ex), p in zip(tests, preds):
        ana_p, _, gly_p = p.partition("=>") if cot else ("", "", p)
        ana_p, gly_p = ana_p.strip(), gly_p.strip()
        gold = it["target"].split()
        kinds = [k for cl in ex["clauses"] for k in word_keys(gen.rep, cl)]
        ok = [False] * len(gold)
        for blk in difflib.SequenceMatcher(a=gold, b=gly_p.split(), autojunk=False).get_matching_blocks():
            for j in range(blk.size):
                ok[blk.a + j] = True
        pieces = []
        for g, (kind, key), right in zip(gold, kinds, ok):
            seen = forms[g] > 0
            words.append({"kind": kind, "key": key, "form_seen": seen, "key_count": keys.get(key, 0), "right": right})
            if kind != "name":                                   # names are invented per sentence: always new
                pieces.append(("word form", seen))
        for k, piece in clause_parts(it["analysis"]):
            pieces.append((k, piece in parts.get(k, set())))
        for f, v in ex.get("used", []):
            if held:
                pieces.append(("English variant", f"{f}={v}" not in held))
        glyph_ok = gly_p == it["target"]
        ana_ok = ana_p == it["analysis"] if cot else None
        sents.append({"right": glyph_ok, "analysis_right": ana_ok, "coverage": sum(s for _, s in pieces) / len(pieces),
                      "new_kinds": sorted({k for k, s in pieces if not s})})
        if cot:
            if ana_ok:
                errors["correct" if glyph_ok else "local"] += 1
            elif glyph_ok:
                errors["restoration"] += 1
            else:
                own = set()
                try:
                    cls, issues = from_analysis(ana_p, spell_name)
                    if cls:                                # a named doer's gender is not in the analysis: try all
                        for gs in itertools.product("mf", repeat=min(len(cls), 6)):
                            own.add(" ".join(gen.rep.target(with_gender(c, g)) for c, g in zip(cls, gs + ("m",) * 9)))
                except Exception:                                 # unreadable analysis
                    pass
                errors["propagation" if gly_p in own else ("both wrong" if own else "unreadable analysis")] += 1
    res = {"run": run, "cot": cot, "n": len(tests), "words": words, "sentences": sents, "errors": dict(errors)}
    (OUT / f"{run}.json").write_text(json.dumps(res), encoding="utf-8")
    print(f"{run}: {len(tests)} sentences, errors {dict(errors)}")


# --- report -----------------------------------------------------------------------------------------------------
def acc(xs: list, key: str = "right") -> str:
    return f"{sum(x[key] for x in xs) / len(xs):.2f} ({len(xs)})" if xs else "-"


def table() -> None:
    res = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(OUT.glob("*.json"))
           if not f.name.endswith((".preds.json", ".pieces.json"))]
    out = ["# Compose or match? Seen vs new pieces (Faith and Fate's subgraph-matching test on Alien Rosetta)", "",
           "Generated test sentences (the same 300 as the frequency-weighted test, none in training); the run's "
           "training data rebuilt exactly. Accuracy (number in brackets). `analysis/subgraph_check.py`.", ""]
    for r in res:
        w = [x for x in r["words"] if x["kind"] == "word"]
        out += [f"## {r['run']}", "", "**1. Glyph words: exact form seen in training vs new form (root trained this "
                "many times)**", "", "| root trained | form seen | new form, known root |", "|---|---|---|"]
        for lo, hi, name in COUNT_BINS:
            b = [x for x in w if lo <= x["key_count"] <= hi]
            out.append(f"| {name} | {acc([x for x in b if x['form_seen']])} | "
                       f"{acc([x for x in b if not x['form_seen']])} |")
        out.append(f"| root never trained | - | {acc([x for x in w if x['key_count'] == 0])} |")
        g = [x for x in r["words"] if x["kind"] in ("pron", "grammar")]
        out += [f"| (pronouns and grammar words) | {acc([x for x in g if x['form_seen']])} | "
                f"{acc([x for x in g if not x['form_seen']])} |", "",
                "**2. Sentences: every piece of a kind seen vs at least one new**", "",
                "| piece | all seen | at least one new |", "|---|---|---|"]
        s = r["sentences"]
        kinds = sorted({k for x in s for k in x["new_kinds"]} | {"word form", "clause template", "verb endings",
                                                                  "slot shape"})
        for k in kinds:
            out.append(f"| {k} | {acc([x for x in s if k not in x['new_kinds']])} | "
                       f"{acc([x for x in s if k in x['new_kinds']])} |")
        out.append(f"| (everything seen) | {acc([x for x in s if not x['new_kinds']])} | - |")
        out += ["", "**3. Sentence accuracy by share of its pieces seen in training**", "",
                "| pieces seen | sentences right |", "|---|---|"]
        for lo, hi in ((1.0, 1.01), (0.95, 1.0), (0.9, 0.95), (0.8, 0.9), (0.0, 0.8)):
            out.append(f"| {lo:.0%}{'' if lo == 1.0 else f'-{hi:.0%}'} | "
                       f"{acc([x for x in s if lo <= x['coverage'] < hi])} |")
        if r["cot"] and r["errors"]:
            e, tot = r["errors"], sum(r["errors"].values())
            out += ["", "**4. Where the steps went wrong (meaning first, then glyphs)**", "",
                    "Check of the check: re-rendering the GOLD analysis gives the gold glyphs for 198/200 generated "
                    "sentences, so about 1% of true propagation errors may be counted as 'both wrong'.", "",
                    "| outcome | sentences | share |", "|---|---|---|"]
            for k in ("correct", "local", "propagation", "restoration", "both wrong", "unreadable analysis"):
                out.append(f"| {k} | {e.get(k, 0)} | {e.get(k, 0) / tot:.2f} |")
        out.append("")
    (ROOT / "results" / "subgraph.md").write_text("\n".join(out), encoding="utf-8")
    plot(res)
    print("wrote results/subgraph.md")


def plot(res: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for r in res:
        w = [x for x in r["words"] if x["kind"] == "word"]
        for seen, ls in ((True, "-"), (False, "--")):
            xs, ys = [], []
            for lo, hi, name in COUNT_BINS:
                b = [x for x in w if lo <= x["key_count"] <= hi and x["form_seen"] == seen]
                if len(b) >= 10:
                    xs.append(name)
                    ys.append(sum(x["right"] for x in b) / len(b))
            ax[0].plot(xs, ys, ls, marker="o", label=f"{r['run']} ({'seen form' if seen else 'new form'})")
        bins = ((0.0, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0), (1.0, 1.01))
        xs, ys = [], []
        for lo, hi in bins:
            b = [x for x in r["sentences"] if lo <= x["coverage"] < hi]
            if len(b) >= 5:
                xs.append("100%" if lo == 1.0 else f"{lo:.0%}-{hi:.0%}")
                ys.append(sum(x["right"] for x in b) / len(b))
        ax[1].plot(xs, ys, marker="o", label=r["run"])
    ax[0].set(xlabel="root seen in training (times)", ylabel="glyph word right", ylim=(0, 1.02),
              title="Word forms: seen vs new")
    ax[1].set(xlabel="share of the sentence's pieces seen in training", ylabel="sentence right", ylim=(0, 1.02),
              title="Sentences by coverage")
    for a in ax:
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    fig.tight_layout()
    try:
        fig.savefig(ROOT / "results" / "fig_subgraph.png", dpi=150)
    except OSError as e:
        print("figure not saved (file open?):", e)


if __name__ == "__main__":
    if "--table" in sys.argv:
        table()
    else:
        analyse(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300)
