"""Per-loanword accuracy of a loanword run (local; generated sentences only, never the sealed set).

    python analysis/loan_words.py [RUN] [--n 150]   -> results/loanwords_<RUN>.md + .json   (RUN: e2b-v2fix-loan-r2)

Sentences with borrowed English nouns (half of the noun phrases become loans), once with loans SEEN in training and
once with loans held out of training (new). For every loan noun in a gold answer:
  meaning right : the model's meaning contains the same LOAN.word (it recognised the loan and spelled its key)
  glyphs right  : the loan's glyph word (sound spelling + LOAN ending + case/plural) is in the model's glyphs, aligned
  closeness     : for a loan whose glyph word is wrong, the best character similarity (difflib ratio) between the gold
                  glyph word and any word the model wrote; near miss = closeness >= 0.75
The whole-sentence scores of the run's own tests understate this (long sentences with several loans).

Rule or list (added 2026-10-27 for e2b-v2fix-loan12 vs e2b-v2fix-loan-r2): a loan's spelling is computed from its
English LETTERS by fixed rules (conlang/v2.py loan_sounds: b->p, d->t, s->c, ...; multi-letter rules th->t, sh->c,
ea->ii, ...), so a model that learned the rule spells new loans almost as well as seen ones, its errors are near
misses, and it handles a word whatever rules it needs; a model that learned a list does not. Loans are split by
  rules       letter by letter only (one-letter rules alone give loan_sounds' spelling), or needs a multi-letter
              rule (ck, ea, ou, qu..., doubled letters written once, ee/oo/ng/ny read as one sound). Fixed 28 Oct
              (code review): the first version looked only at the digraph list and put ~22% of words that need the
              other rules (referee, hubby, prong) in "letter by letter".
  length      sounds in the loan spelling: <=4, 5-7, >=8
Predictions for loan12 were written in results/findings_2026-10-24.md before it ran.

Recognition (added 2026-10-27, after loan-r2's examples showed O=tomboy for O=LOAN.tomboy): for every loan noun, what
the model's MEANING did with it: LOAN.word right / marker missing (the bare English word, written as if it were a word
of the language) / other (misspelled key, another word, dropped). Plus false loans: LOAN.x on a word that is not a
loan (misspelled loan keys also count here). Every prediction is saved to results/loanwords_<RUN>.preds.json; --rescore recomputes the tables from it with
no model (n is taken from the saved file).
"""

from __future__ import annotations

import collections
import difflib
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from conlang.v2 import LOAN_SPELL, loan_sounds  # noqa: E402

RUN = "e2b-v2fix-loan-r2"
SETS = ("seen loans", "new loans (never trained)")


ONE_LETTER = {a: b for a, b in LOAN_SPELL if len(a) == 1}


def letter_by_letter(word: str) -> bool:
    """Does mapping each English letter on its own (one-letter rules only, one sound per letter) give the loan's
    spelling? False when it needs a multi-letter rule, a doubled letter written once, or a two-letter sound."""
    s = re.sub(r"[^a-z]", "", word.lower())
    return tuple("".join(ONE_LETTER.get(c, c) for c in s)) == loan_sounds(word)


def groups(word: str) -> list[str]:
    n = len(loan_sounds(word))
    return ["letter by letter only" if letter_by_letter(word) else "needs a multi-letter rule",
            "<=4 sounds" if n <= 4 else ("5-7 sounds" if n <= 7 else ">=8 sounds")]


RECOG = ["LOAN.word right", "marker missing (bare word)", "other"]
GROUPS = ["letter by letter only", "needs a multi-letter rule", "<=4 sounds", "5-7 sounds", ">=8 sounds"]


def main() -> None:
    import train.represent_v2 as R2
    from subgraph_check import training_pieces
    from train.load import load_trained, translate
    args = [a for i, a in enumerate(sys.argv[1:]) if not a.startswith("--") and (i == 0 or sys.argv[i] != "--n")]
    run = args[0] if args else RUN
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 150
    rescore = "--rescore" in sys.argv
    pfile = ROOT / "results" / f"loanwords_{run}.preds.json"
    saved = json.loads(pfile.read_text(encoding="utf-8")) if pfile.exists() else {}
    if rescore:
        if not saved or any(len(saved.get(k, [])) != len(saved[SETS[0]]) for k in SETS):
            sys.exit(f"--rescore: {pfile.name} missing or incomplete (run without --rescore to finish it)")
        n = len(saved[SETS[0]])                                     # the saved n, not the default
    mdir = ROOT / "results" / "kaggle" / run / "model"
    cfg = json.loads((mdir / "run_args.json").read_text(encoding="utf-8"))
    _, train_en, _, _ = training_pieces(cfg, ROOT / "results" / "subgraph" / f"{run}.pieces.json")
    R2.set_held_out(0)
    gen = R2.GenV2("tamil_like", "abugida_pua", 0)
    sets = {}
    for pool, name in zip(("train", "held"), SETS):
        R2.LOAN.update(rate=0.5, pool=pool)
        rng = random.Random(31 if pool == "train" else 32)          # the same draws for every run (minus sentences in its training)
        items = []
        while len(items) < n:
            ex = gen.example(rng)
            it = gen.item(ex) if ex else None
            if it and "LOAN." in it["analysis"] and it["english"] not in train_en:
                words = [(w.gloss, gen.rep.script.word(w.phonemes)) for cl in ex["clauses"] for w in gen.rep.words(cl)]
                items.append({**it, "words": words})
        sets[name] = items
    R2.LOAN.update(rate=0.0, pool="train")
    model = tok = None
    res, examples = {}, []
    for name, items in sets.items():
        preds = saved.setdefault(name, [])                          # resumes a stopped run
        if any(p["english"] != it["english"] for p, it in zip(preds, items)):
            sys.exit(f"{pfile.name} does not match this run's sentences")
        for i in range(len(preds), len(items), 8):
            if model is None:
                assert not rescore
                model, tok, _ = load_trained(mdir)
            out = translate(model, tok, [x["english"] for x in items[i:i + 8]], max_new_tokens=700)
            preds += [{"english": x["english"], "gold": x["analysis"], "pred": o} for x, o in zip(items[i:i + 8], out)]
            tmp = pfile.with_suffix(".tmp")                         # a kill mid-write must not corrupt the file
            tmp.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, pfile)
            print(f"  {name} {len(preds)}/{len(items)}", flush=True)
        preds = [p["pred"] for p in preds]
        rec = collections.Counter()                                  # what the meaning did with each loan noun
        false_loans = 0
        tot = mean_ok = gl_ok = sent_ok = 0
        by = collections.defaultdict(lambda: [0, 0])                 # group -> [loans, glyph word right]
        close = []                                                   # closeness of each wrong loan glyph word
        for it, p in zip(items, preds):
            ana, _, gly = p.partition("=>")
            ana, gly = ana.strip(), gly.strip()
            sent_ok += gly == it["target"]
            pred_loans = collections.Counter(re.findall(r"LOAN\.[a-z]+", ana))
            bare = collections.Counter(w for pre, w in re.findall(r"(LOAN\.|NAME\.|PRON\.)?([a-z]+)", ana) if not pre)
            gold_loans = collections.Counter(re.findall(r"LOAN\.([a-z]+)", it["analysis"]))
            false_loans += sum(c for w, c in collections.Counter(re.findall(r"LOAN\.([a-z]+)", ana)).items()
                               if w not in gold_loans)
            gold_glyphs = [g for _, g in it["words"]]
            ok = [False] * len(gold_glyphs)
            for blk in difflib.SequenceMatcher(a=gold_glyphs, b=gly.split(), autojunk=False).get_matching_blocks():
                for j in range(blk.size):
                    ok[blk.a + j] = True
            for k, (gloss, g) in enumerate(it["words"]):
                m = re.match(r"LOAN\.([a-z]+)", gloss)
                if not m:
                    continue
                tot += 1
                if pred_loans[f"LOAN.{m.group(1)}"] > 0:
                    mean_ok += 1
                    pred_loans[f"LOAN.{m.group(1)}"] -= 1
                    rec["LOAN.word right"] += 1
                elif bare[m.group(1)] > 0:
                    bare[m.group(1)] -= 1
                    rec["marker missing (bare word)"] += 1
                else:
                    rec["other"] += 1
                gl_ok += ok[k]
                for grp in groups(m.group(1)):
                    by[grp][0] += 1
                    by[grp][1] += ok[k]
                if not ok[k]:
                    close.append(max((difflib.SequenceMatcher(a=g, b=w, autojunk=False).ratio()
                                      for w in gly.split()), default=0.0))
            if len(examples) < 12 and name.startswith("new") and "LOAN." in it["analysis"]:
                examples.append((it["english"], it["analysis"], ana))
        res[name] = {"loans": tot, "meaning_right": mean_ok / tot, "glyph_right": gl_ok / tot,
                     "sentences_right": sent_ok / len(items), "groups": {g: tuple(by[g]) for g in GROUPS},
                     "wrong": len(close), "near_miss": sum(c >= 0.75 for c in close) / max(1, len(close)),
                     "closeness_median": sorted(close)[len(close) // 2] if close else None,
                     "recognition": {k: rec[k] / tot for k in RECOG}, "false_loans_per_sentence": false_loans / len(items)}
    write(run, n, res, examples)


def write(run: str, n: int, res: dict, examples: list) -> None:
    def rate(x):
        return f"{x[1] / x[0]:.2f} ({x[0]})" if x[0] else "- (0)"

    L = ["# Per-loanword accuracy (`analysis/loan_words.py`)", "",
         f"Run {run}. {n} generated sentences per set; every loan noun in the gold answer is scored.", "",
         "| set | loan nouns | meaning right (LOAN.word) | glyph word right | sentences fully right |", "|---|---|---|---|---|"]
    for name, r in res.items():
        L.append(f"| {name} | {r['loans']} | **{r['meaning_right']:.2f}** | **{r['glyph_right']:.2f}** | "
                 f"{r['sentences_right']:.2f} |")
    L += ["", "## Rule or list: glyph word right by kind of loan (n loans)", "",
          "Closeness compares whole glyph words, including the shared LOAN ending and case/plural ending, so it "
          "runs high even for a wrong stem; compare it across runs, not as an absolute.", "",
          "| set | " + " | ".join(GROUPS) + " | wrong loans | near misses (closeness >= 0.75) | median closeness |",
          "|---|" + "---|" * (len(GROUPS) + 3)]
    for name, r in res.items():
        med = "-" if r["closeness_median"] is None else f"{r['closeness_median']:.2f}"
        L.append(f"| {name} | " + " | ".join(rate(r["groups"][g]) for g in GROUPS) +
                 f" | {r['wrong']} | {r['near_miss']:.0%} | {med} |")
    if all("recognition" in r for r in res.values()):
        L += ["", "## Recognition: what the meaning did with each loan noun", "",
              "| set | " + " | ".join(RECOG) + " | unmatched LOAN marks per sentence (false loans + misspelled keys) |", "|---|" + "---|" * (len(RECOG) + 1)]
        for name, r in res.items():
            L.append(f"| {name} | " + " | ".join(f"{r['recognition'][k]:.2f}" for k in RECOG) +
                     f" | {r['false_loans_per_sentence']:.2f} |")
    L += ["", "## Examples (new loans): gold meaning vs the model's", ""]
    for e, g, a in examples:
        L += [f"- {e}  \n  gold: `{g}`  \n  model: `{a}`"]
    (ROOT / "results" / f"loanwords_{run}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "results" / f"loanwords_{run}.json").write_text(json.dumps({"run": run, "n": n, "sets": res}, indent=1),
                                                            encoding="utf-8")
    print("\n".join(L[:16]))


if __name__ == "__main__":
    main()
