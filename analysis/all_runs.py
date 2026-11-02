"""Results of every Alien Rosetta run in one document.

    python analysis/all_runs.py        -> results/ALL_RUNS.md (rewritten each time; pending runs fill in by themselves)

Reads results/kaggle/<run>/ (each run's result JSON and arguments), the scored tables results/sealed_scores.md
(totals only), results/tatoeba_scores.md and results/freq_weighted.md, and the grokking records results/grok/*.json.
Never reads the sealed set itself, per-item sealed results or the reader audit.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
KAG = RES / "kaggle"
OUT = RES / "ALL_RUNS.md"


def f3(x) -> str:
    return "-" if x is None or x == "" else (f"{x:.3f}" if isinstance(x, (int, float)) and not isinstance(x, bool)
                                            else str(x))


def load(run: str):
    """(result dict, error) for results/kaggle/<run>; error is None, 'missing' or 'unreadable: ...'."""
    files = sorted(glob.glob(str(KAG / run / "results" / "*.json"))) or sorted(glob.glob(str(KAG / run / "*.json")))
    files = [f for f in files if Path(f).stem == run] or files
    if not files:
        return None, "missing"
    try:
        return json.loads(Path(files[0]).read_text(encoding="utf-8")), None
    except Exception as e:                                          # one bad file must not stop the document
        return None, f"unreadable: {type(e).__name__}"


def runs_matching(pattern: str) -> list[str]:
    return sorted(p.name for p in KAG.iterdir() if p.is_dir() and re.fullmatch(pattern, p.name))


def md_table(path: Path, key_col: int = 0) -> dict[str, list[str]]:
    """First markdown table in a file whose header starts with '| run' or '| model': {run: cells}."""
    if not path.exists():
        return {}
    rows, on = {}, False
    for line in path.read_text(encoding="utf-8").splitlines():
        if re.match(r"\|\s*(run|model)\s*\|", line):
            on = True
            continue
        if on:
            if not line.startswith("|"):
                if rows:
                    break
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set(cells[0]) <= set("-: "):
                continue
            rows.setdefault(cells[key_col], cells[1:])
    return rows


SEALED = md_table(RES / "sealed_scores.md")          # glyphs exact, meaning exact, word acc, refused (totals)
TATOEBA = md_table(RES / "tatoeba_scores.md")        # glyphs exact, meaning exact, word acc
FREQ = md_table(RES / "freq_weighted.md")            # sentence exact, word acc lambda 0..3


def train_done(d: dict) -> str:
    t = (d or {}).get("training") or {}
    if "steps" in t and "planned_steps" in t:
        return f"{t['steps']}/{t['planned_steps']}" + (" (time limit)" if t.get("stopped_by_time") else "")
    if d and "steps" in d:
        return str(d["steps"])
    return "-"


def args_of(d: dict) -> dict:
    a = (d or {}).get("args") or {}
    return a if isinstance(a, dict) else {}


def settings(d: dict, keys: list[str]) -> str:
    a, out = args_of(d), []
    for k in keys:
        v = a.get(k)
        if v in (None, "", "None", "False", False, "0", 0, "0.0") or (k == "loss" and v == "ce") or                 (k == "focal_gamma" and a.get("loss") != "focal"):         # defaults that change nothing
            continue
        out.append(k if v in ("True", True) else f"{k}={v}")
    return ", ".join(out) or "-"


def table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


# ---------------------------------------------------------------- stage-1 style runs (translation, 4 test splits)
SPLITS = ("test_iid", "test_comp", "test_morph", "test_long")
S1_KEYS = ["model", "preset", "script", "n_train", "epochs", "target_tok", "add_tokens", "new_tok_init",
           "aux_translit", "mono_n"]


def stage1_rows(runs: list[str]) -> list[list[str]]:
    rows = []
    for r in runs:
        d, err = load(r)
        if err:
            rows.append([r, err] + ["-"] * 7)
            continue
        ev = d.get("eval") or {}
        tf = [((ev.get(s) or {}).get("teacher_forced") or {}).get("exact_match") for s in SPLITS]
        gen = (ev.get("test_iid") or {}).get("generation") or {}
        rows.append([r, settings(d, S1_KEYS), train_done(d)] + [f3(x) for x in tf] +
                    [f3(gen.get("word_accuracy")), f3((d.get("training") or {}).get("final_loss"))])
    return rows


S1_HEAD = ["run", "settings", "steps", "exact iid", "exact comp", "exact morph", "exact long",
           "word acc iid (generated, 100)", "final loss"]

# ---------------------------------------------------------------- E1 / R1 (v1 language, generated test families)


def e1_rows(runs: list[str]) -> list[list[str]]:
    rows = []
    for r in runs:
        d, err = load(r)
        if err:
            rows.append([r, err] + ["-"] * 7)
            continue
        g = d.get("eval_gen") or {}

        def ex(k, f="exact"):
            return (g.get(k) or {}).get(f)
        held = [v.get("exact") for k, v in g.items() if k.startswith("held-out") and "synonym" not in k
                and isinstance(v, dict) and v.get("exact") is not None]
        rows.append([r, settings(d, ["epochs", "cot", "aug", "rep", "seed"]), train_done(d),
                     f3(ex("original test_iid")), f3(ex("original test_iid", "analysis_exact")),
                     f3(ex("in-distribution mix")), f3(sum(held) / len(held) if held else None),
                     f3(ex("held-out voice = passive")), ])
    return rows


E1_HEAD = ["run", "settings", "steps", "exact orig iid", "analysis exact orig iid", "exact in-dist mix",
           "held-out structures, mean exact", "held-out passive exact"]

# ---------------------------------------------------------------- v2 language (E2 + extensions)
V2_KEYS = ["epochs", "cot", "rep_n", "rep_held_frac", "rep_min_count", "loss", "focal_gamma", "rep_loan_rate",
           "rep_paraphrases", "rep_real", "seed", "train_seed"]


def v2_rows(runs: list[str]) -> list[list[str]]:
    rows = []
    for r in runs:
        d, err = load(r)
        s, t, fq = SEALED.get(r, []), TATOEBA.get(r, []), FREQ.get(r, [])
        if err:
            rows.append([r, "pending" if err == "missing" else err] + ["-"] * 10)
            continue
        g = (d.get("eval_gen") or {}).get("in-distribution mix") or {}
        cell = lambda lst, i: lst[i] if len(lst) > i else "-"
        rows.append([r, settings(d, V2_KEYS), train_done(d), f3(g.get("exact")), f3(g.get("analysis_exact")),
                     cell(fq, 0), cell(t, 1), cell(t, 0), cell(s, 0), cell(s, 1), cell(s, 2), cell(s, 3)])
    return rows


V2_HEAD = ["run", "settings", "steps", "gen exact (100)", "gen analysis exact (100)", "gen sentence exact (300)",
           "Tatoeba meaning", "Tatoeba glyphs", "sealed glyphs", "sealed meaning", "sealed word acc",
           "sealed refused (out of scope)"]

# ---------------------------------------------------------------- maths
MATHS_TESTS = ["add: up to 3+3 (trained lengths)", "add: 4+4", "add: 5+5", "mul: up to 2x2 (trained lengths)",
               "mul: 3x1", "mul: 3x2", "mul: 3x3"]


def maths_rows(runs: list[str]) -> list[list[str]]:
    rows = []
    for r in runs:
        d, err = load(r)
        if err:
            rows.append([r, "pending" if err == "missing" else err] + ["-"] * (len(MATHS_TESTS) + 3))
            continue
        tests = d.get("tests") or {}
        ex = lambda k: (tests.get(k) or {}).get("exact") if isinstance(tests.get(k), dict) else tests.get(k)
        short = lambda k: re.sub(r"\s*\(trained lengths\)", "", k.split(": ", 1)[1]).replace("dropped pair ", "pair ")
        extra = [f"{short(k)} {ex(k):.3f}" for k in tests
                 if ("dropped" in k or k in ("add: 6+6", "mul: 4x2", "mul: 4x3")) and isinstance(ex(k), (int, float))]
        rows.append([r, settings(d, ["init", "seed", "epochs", "add_max", "mul_max", "drop_pair", "scratch"]),
                     f3(d.get("final_loss"))] + [f3(ex(k)) for k in MATHS_TESTS] + ["; ".join(extra) or "-"])
    return rows


MATHS_HEAD = ["run", "settings", "final loss", "add <=3+3", "add 4+4", "add 5+5", "mul <=2x2", "mul 3x1",
              "mul 3x2", "mul 3x3", "other tests"]

# ---------------------------------------------------------------- grokking


def grok_rows() -> list[list[str]]:
    rows = []
    for f in sorted((RES / "grok").glob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            rows.append([f.stem, f"unreadable: {type(e).__name__}"] + ["-"] * 6)
            continue
        curve = rec.get("curve") or []
        last = curve[-1] if curve else {}
        test_k = next((k for k in last if k.startswith("test")), None)
        extra_k = next((k for k in last if k.startswith("extra")), None)
        note = "; ".join(f"{k}: {v[:60]}" for k, v in (rec.get("stale") or {}).items())
        if f.stem == "grok-translate-wd1-r8k":
            note = "step 8000-12000 rerun chunk; merged into grok-translate-wd1"
        rows.append([f.stem, ", ".join(rec.get("chunks") or []) or "-",
                     f"{rec.get('step', '-')}/{rec.get('steps_total', '-')}", f3(last.get("train (sample)")),
                     f3(last.get(test_k)) + (f" ({test_k.split(': ', 1)[-1]})" if test_k else ""),
                     f3(last.get(extra_k)) + (f" ({extra_k.split(': ', 1)[-1]})" if extra_k else ""),
                     f3(last.get("logit_norm")), note or "-"])
    return rows


GROK_HEAD = ["chain", "chunks", "steps done/total", "train", "test", "extra", "logit norm", "note"]

# ---------------------------------------------------------------- Faith and Fate: real models, prompted


def ff_rows() -> list[list[str]]:
    rows = []
    for run in runs_matching(r"ff-.*"):
        for mdir in sorted((KAG / run / "ff_answers").glob("*")):
            accs = []
            for size in ("1_by_1", "2_by_1", "2_by_2", "3_by_1", "3_by_2", "3_by_3"):
                f = mdir / f"scratchpad_{size}_50prompts.json"
                try:
                    items = json.loads(f.read_text(encoding="utf-8"))
                    ok = 0
                    for it in items:
                        a, b = map(int, re.findall(r"\d+", it["question"])[:2])
                        nums = re.findall(r"\d[\d,]*", str(it.get("GPT3 answer", "")))
                        ok += bool(nums) and int(nums[-1].replace(",", "")) == a * b
                    accs.append(f"{ok / len(items):.2f}")
                except FileNotFoundError:
                    accs.append("-")
                except Exception as e:
                    accs.append(f"unreadable ({type(e).__name__})")
            rows.append([run, mdir.name] + accs)
    return rows


FF_HEAD = ["run", "model", "1x1", "2x1", "2x2", "3x1", "3x2", "3x3"]

# ---------------------------------------------------------------- document

FAMILIES = [
    ("1. Stage 1: model and size pick", r"l06-.*|l17-.*", "stage1",
     "Which base model to use: Qwen3-0.6B-Base (l06) vs Qwen3-1.7B-Base (l17) on the easy (Latin) and hard (Tamil "
     "script) languages. 1.7B was better but still failing at 20k x 2 epochs; 0.6B was kept, since capacity was not "
     "the bottleneck (PLAN.md, stage 1)."),
    ("2. Script x grammar grid (2x2, sizes and seeds)", r"c-.*|s[12]-(easy|hard)-(latin|tamil)-20k|g-(easy|hard)-tamil-50k",
     "stage1",
     "{easy, hard} grammar x {Latin, Tamil script} at 5k/10k/20k/50k sentences; s1/s2 = seeds 1/2 at 20k (a new seed "
     "is a new random language with the same grammar). At 20k, content-word accuracy: easy 0.82-0.86, hard Latin 0.04, "
     "hard Tamil 0.24 (results/final_20k.md). Exact here is teacher-forced whole-sentence match on 500 items."),
    ("3. Cause tests: tokenisation and invented glyphs", r"m[23]-.*|p-.*", "stage1",
     "Is it token count or familiar characters? m2/m3 = one character per token (Latin / accented Latin); p-*-pua = "
     "invented glyphs split into byte pieces (collapse at 20k, recover slowly with more data); p-*-puatok = each glyph "
     "one new token. Byte pieces, not token count, cause the collapse; whole-glyph tokens fix the easy language fully "
     "but the hard one only partly (results/final_20k.md, PLAN.md)."),
    ("4. Glyph starting points (claim 1, stage 1)", r"g[123]-.*|s1-hard-latininit-50k|s2-hard-translit-50k|"
     r"s3-hard-plaintext-50k|s4-hard-shuffledinit-50k", "stage1",
     "Hard language, glyphs as one token each, 50k, baseline = p-hard-puatok-50k (exact 0.570/0.506/0.420/0.000). "
     "g1 starts each glyph at the embedding of its sound (fixes it fully), g2 adds transliteration examples, g3 "
     "trains on plain glyph text first (worse); s1-s3 are the same runs re-trained with saved weights, s4 starts "
     "every glyph at a WRONG sound (control)."),
    ("5. R1 / E1: robustness and the reasoning step (v1 language)", r"r1-.*|e1[ab]-.*", "e1",
     "R1 = varied English for robustness; E1a = direct translation, E1b = writes the meaning analysis first (the "
     "reasoning step). At 3 epochs E1b looked far better (0.96 vs 0.26 exact) but at 5 epochs both reach 0.97-0.99: "
     "the analysis step speeds learning, it does not generalise further. Unseen structures stay near 0 for all "
     "(passive 0.00-0.02, clefts 0.01-0.09) (results/e1_5ep.md, e1_comparison.md)."),
    ("6. v2 language: E2a / E2b", r"e2a-v2-80k|e2b-v2-cot-80k(-noheld)?|e2b-v2fix-cot-80k(-noheld)?", "v2",
     "The full language v2 (11k roots, built words, new sentence kinds), 80k examples. E2a direct, E2b analysis "
     "first; 'noheld' = no English variants held out; 'v2fix' = every word seen at least 12 times (word floor). "
     "e2b-v2fix-cot-80k-noheld is the baseline for all extensions."),
    ("7. Extensions on the v2 baseline (claims 3, 4, 6)", r"e2b-v2fix-(?!cot-80k).*", "v2",
     "One change each vs the baseline: focal loss, loanwords (0.04), paraphrased English (para, para8), real Tatoeba "
     "sentences (real15), training-seed replicates (ts1, real15-ts1), 6 epochs, word floor 25. para-s1 (--seed 1 = a different "
     "language) is excluded: results/excluded/README.md. Real text is the clear win; focal, "
     "loans, paraphrases and 6 epochs do not help on real English (results/findings_2026-10-24.md)."),
    ("8. Maths in glyph notation (claims 1 and 2)", r"maths-.*", "maths",
     "Base-7 arithmetic in glyph digits and operators; trained lengths add <= 3+3, mul <= 2x2. Starts: meaning (real "
     "digits), shuffled (wrong digits), mean (none); partial starts; longer training lengths; dropped digit pairs "
     "(no training multiplication uses that single-digit product); scratchpad (long-multiplication working before the "
     "answer). Exact match on 300 items per test (full tables: results/maths.md)."),
]
PENDING = ["e2b-v2fix-floor25", "maths-meaning-drop22", "maths-meaning-scratch", "maths-meaning-scratch-drop56"]

CLAIMS = [
    ("1. New symbols are learnable when they start from what they stand for (sound / real digit)",
     "g1-hard-latininit-50k exact 0.990/0.986/0.972/0.974 vs p-hard-puatok-50k 0.570/0.506/0.420/0.000; maths add 4+4 "
     "two-seed means: meaning 0.86 (0.887/0.840), shuffled 0.34 (0.083/0.600: unstable), mean 0.09 (0.020/0.153); digits-only start 0.87, operators-only 0.06"),
    ("2. Generalisation stops at training coverage of structure",
     "E1 at 5 epochs: ~0.97 on trained structures, 0.00-0.09 on held-out ones; maths dropped pairs: problems needing "
     "the removed fact 0.05 (5x6) / 0.10 (3x4) vs 0.92 / 0.96 avoiding it"),
    ("3. ... of vocabulary", "rare words fail below ~10 sightings; word floor 25 vs floor 12 baseline seeds: generated sentence exact "
     "0.720 vs 0.607/0.627 (p < 0.002; but floor 25 also trained on ~20% more examples: 13,304 vs 11,064 steps), but Tatoeba meaning 0.603 vs 0.617/0.625 and sealed (v2, 147) meaning 0.503 vs 0.476/0.463 (no significant gain "
     "on real English: about half of Tatoeba's wrong sentences (54%) miss a word the generated data never had, 217 "
     "occurrences at 0.41, mostly common words/uses the generator does not produce, which no floor reaches; 30% have every word right)"),
    ("4. ... of English variety",
     "real15 vs baseline seeds: Tatoeba meaning 0.923 vs 0.617/0.625; sealed (v2, 147) glyphs 0.558 vs 0.422/0.435, sealed "
     "meaning 0.558 vs 0.476/0.463; 6 epochs: generated 0.660 (above baseline 0.607) but Tatoeba meaning 0.563 (lowest) and sealed meaning 0.449. "
     "Caveat: Tatoeba gold and real15's training labels come from the same automatic reader, so the sealed gap "
     "(sealed v2 +0.089; paired McNemar per seed p = 0.036 and 0.013, significant; unpaired totals not) is the fairer size"),
    ("5. Writing the meaning first speeds learning, not generalisation",
     "E1b vs E1a: 0.96 vs 0.26 exact at 3 epochs, both 0.97-0.99 at 5 epochs"),
    ("6. Model tricks are not the fix",
     "sealed v2 glyphs: focal-r2 0.388, para 0.429, two-stage rewriter 0.422 vs baseline 0.422: no gain on real English; "
     "Tatoeba (paired): focal -0.028 (p=0.014) and 6 epochs -0.053 (p=0.0002) significantly worse, paraphrases 0.00"),
    ("Inside the model (claim 4)",
     "probes (corrected 2026-10-25): where the output is wrong the probe is mostly wrong too (0.41 vs majority 0.79, 107 items, both seeds), so errors are mostly in the representation, not a lost read-out; steering at layer 6 flips tense 0.98-1.00 "
     "and negation 0.93-0.99 vs random <=0.02 (directions are used, but entangled)"),
    ("Side study: grokking",
     "no grokking: translation 12k steps (test 0.80), maths 40% 60k steps (test 0.73; wd 0.1 for the first 16k, wd 1.0 after), maths with 60% of the table "
     "60k steps (test rising slowly 0.74 -> 0.84, no jump); logit norm rises instead of falling"),
]


def main() -> None:
    covered, unreadable = {}, []
    parts = [
        "# Alien Rosetta: results of every run", "",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `analysis/all_runs.py` (refresh: `python analysis/all_runs.py`). "
        "Sealed-set numbers are totals from `results/sealed_scores.md`; '-' = metric not measured for that run.", "",
        "**Research question:** what does it take for a small LLM to learn a language it has never seen, and where "
        "does that learning stop?", "",
        "**Claims:** (1) new symbols are learnable when they start from meaning; generalisation stops at training "
        "coverage of (2) sentence structure, (3) vocabulary and (4) English variety; (5) writing the meaning first "
        "speeds learning, not generalisation; (6) model tricks are not the fix.", ""]
    for title, pattern, kind, blurb in FAMILIES:
        runs = runs_matching(pattern)
        if kind == "v2":
            runs += [p for p in PENDING if re.fullmatch(pattern, p) and p not in runs]
        if kind == "maths":
            runs += [p for p in PENDING if p.startswith("maths") and p not in runs]
        covered[title] = len(runs)
        rows = {"stage1": stage1_rows, "e1": e1_rows, "v2": v2_rows, "maths": maths_rows}[kind](runs)
        head = {"stage1": S1_HEAD, "e1": E1_HEAD, "v2": V2_HEAD, "maths": MATHS_HEAD}[kind]
        unreadable += [f"{r[0]} ({r[1]})" for r in rows if str(r[1]).startswith("unreadable")]
        parts += [f"## {title}", "", blurb, "", table(head, rows)]
    g = grok_rows()
    covered["9. Grokking"] = len(g)
    parts += ["## 9. Grokking (side study)", "",
              "Long training with weight decay 1.0 to see whether test accuracy jumps late. One row per chain (chunks "
              "merged). Translation chunk c2 resumed from a stale checkpoint (step 4000) and was replaced by the "
              "r8k rerun from step 8000. Curves: results/grok/*.png.", "", table(GROK_HEAD, g)]
    ff = ff_rows()
    covered["10. Faith and Fate, real models"] = len(ff)
    parts += ["## 10. Faith and Fate replication on real models (presentation)", "",
              "Qwen2.5-1.5B/3B-Instruct prompted with the paper's step-by-step multiplication scratchpad, 50 problems "
              "per size; accuracy = last number in the answer equals the product.", "", table(FF_HEAD, ff)]
    other = runs_matching(r"paragen-.*|smoke")
    covered["11. Data generation and smoke test"] = len(other)
    prow = []
    for r in other:
        if r == "smoke":
            prow.append([r, "smoke test of the pipeline, not a result"])
            continue
        f = KAG / r / "paraphrases_v2_seed0.json"
        try:
            n = len(json.loads(f.read_text(encoding="utf-8"))) if f.exists() else None
            prow.append([r, "paraphrase generation for the para runs (data, not a model)" +
                         (f"; {n} records" if n is not None else "; output merged elsewhere")])
        except Exception as e:
            prow.append([r, f"unreadable: {type(e).__name__}"])
    parts += ["## 11. Data generation and smoke test", "", table(["run", "what it is"], prow)]
    parts += ["## Headline numbers per claim", "", table(["claim", "runs and numbers"], [list(c) for c in CLAIMS])]
    pend = []
    for p in PENDING:
        d, err = load(p)
        pend.append([p, "done (in the tables above)" if not err else ("pending" if err == "missing" else err)])
    g_rec = RES / "grok" / "grok-maths-frac06.json"
    try:
        rec = json.loads(g_rec.read_text(encoding="utf-8"))
        pend.append(["grok-maths-frac06 (last chunk)", "done" if rec.get("step", 0) >= rec.get("steps_total", 1)
                     else f"pending (at step {rec.get('step')} of {rec.get('steps_total')})"])
    except Exception:
        pend.append(["grok-maths-frac06 (last chunk)", "pending"])
    parts += ["## Runs still running or queued", "", table(["run", "status"], pend)]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", OUT)
    for k, v in covered.items():
        print(f"  {v:3d}  {k}")
    print("unreadable:", unreadable or "none")


if __name__ == "__main__":
    main()
