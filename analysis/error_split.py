"""Understanding vs translation errors for every analysis-first v2 model (2026-10-26).

    python analysis/error_split.py        -> results/error_split.md + .json

Every test item has two gold answers (the meaning analysis and the glyphs), so each output falls in one group:
  both right
  translation error      analysis right, glyphs wrong
  lucky glyphs           analysis wrong, glyphs right
  understanding error    analysis wrong, and the glyphs are exactly the model's own analysis rendered by the language
                         (the model translated faithfully what it understood; a named doer's gender is not in the
                         analysis, so both genders are tried)
  both wrong             analysis wrong, and the glyphs do not match even the model's own analysis
  word not in language   analysis wrong and it uses a word key the language does not have (help instead of
                         help>ACT, 'tom' as a common noun), so it cannot be rendered (until 2026-10-26 these were
                         counted as 'unreadable'; analysis/unreadable_check.py found only 0-2 truly unreadable)
  unreadable             analysis wrong and not readable as an analysis at all
Generated test sentences: the groups computed by analysis/subgraph_check.py (results/subgraph/<run>.json; 'local' is
a translation error, 'propagation' an understanding error). Tatoeba 600: computed here from the saved predictions.
Sealed v2 (147 in scope): only the four totals from the per-entry right/wrong flags of analysis/score_sealed.py
(totals-only rule; outputs were never saved, so own-analysis rendering is not possible there).
"""

from __future__ import annotations

import collections
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sealed" / "app"))
import train.represent_v2 as R2  # noqa: E402
from analysis.score_sealed import DRAFT  # noqa: E402
from conlang.v2 import from_analysis  # noqa: E402
from english_parse import spell_name, with_gender  # noqa: E402

GROUPS = ["both right", "translation error", "lucky glyphs", "understanding error", "both wrong",
          "word not in language", "unreadable"]
SUB = {"correct": "both right", "local": "translation error", "restoration": "lucky glyphs",
       "propagation": "understanding error", "both wrong": "both wrong", "unreadable analysis": "unreadable"}


def split(gen, items) -> collections.Counter:
    """items: (gold analysis, gold glyphs, model output 'analysis => glyphs')."""
    c = collections.Counter()
    for ana_g, gly_g, p in items:
        ana_p, _, gly_p = p.partition("=>")
        ana_p, gly_p = ana_p.strip(), gly_p.strip()
        a_ok, g_ok = ana_p == ana_g, gly_p == gly_g
        if a_ok:
            c["both right" if g_ok else "translation error"] += 1
            continue
        if g_ok:
            c["lucky glyphs"] += 1
            continue
        own, missing = set(), False
        try:
            cls, _ = from_analysis(ana_p, spell_name)
            if cls:
                for gs in itertools.product("mf", repeat=min(len(cls), 6)):
                    own.add(" ".join(gen.rep.target(with_gender(cl, g)) for cl, g in zip(cls, gs + ("m",) * 9)))
        except KeyError:
            missing = True                                       # a word key the language does not have
        except Exception:
            pass
        c["understanding error" if gly_p in own else
          ("both wrong" if own else ("word not in language" if missing else "unreadable"))] += 1
    return c


def main() -> None:
    gen = R2.GenV2("tamil_like", "abugida_pua", 0)
    test = json.loads((ROOT / "results" / "tatoeba_test.json").read_text(encoding="utf-8"))
    in_scope = {e["id"] for e in json.loads(DRAFT.read_text(encoding="utf-8"))["entries"] if e["scope"] == "in"}
    runs = sorted(p.stem for p in (ROOT / "results" / "tatoeba").glob("*.json"))
    res = {}
    for run in runs:
        r = json.loads((ROOT / "results" / "tatoeba" / f"{run}.json").read_text(encoding="utf-8"))
        if not r.get("cot") or "preds" not in r:
            continue
        ra = ROOT / "results" / "kaggle" / run.partition("@epoch")[0] / "model" / "run_args.json"
        if ra.exists() and R2.language_of_run(json.loads(ra.read_text(encoding="utf-8"))):
            g = R2.GenV2("tamil_like", "abugida_pua", 0)          # sound-rule run: its language, re-rendered golds
            golds = R2.render_targets([t["analysis"] for t in test], [t["target"] for t in test])
            R2.language_of_run({})
            row = {"tatoeba": split(g, [(t["analysis"], gl or "\0", p) for t, gl, p in zip(test, golds, r["preds"])])}
        else:
            row = {"tatoeba": split(gen, [(t["analysis"], t["target"], p) for t, p in zip(test, r["preds"])])}
        sg = ROOT / "results" / "subgraph" / f"{run}.json"
        if sg.exists():
            e = json.loads(sg.read_text(encoding="utf-8")).get("errors") or {}
            row["generated"] = collections.Counter({SUB[k]: v for k, v in e.items()})
        ss = ROOT / "results" / "sealed" / f"{run}.json"
        if ss.exists():
            its = [i for i in json.loads(ss.read_text(encoding="utf-8"))["items"]
                   if i.get("scope") == "in" and i["id"] in in_scope]
            cf = ROOT / "results" / "sealed" / f"{run}.categories.json"       # category TOTALS only, added 30 Oct
            c = collections.Counter()
            for i in its:
                if "category" in i:                      # re-scored since 30 Oct: the full split, totals only
                    c[i["category"]] += 1
                else:
                    c[("both right" if i["exact"] else "translation error") if i["meaning"] else
                      ("lucky glyphs" if i["exact"] else "analysis wrong, glyphs wrong")] += 1
            if cf.exists():
                c = collections.Counter(json.loads(cf.read_text(encoding="utf-8"))["category_totals"])
            row["sealed"] = c
        res[run] = row
        print(run, {k: dict(v) for k, v in row.items()}, flush=True)
    write(res)


def pct(c: collections.Counter, k: str) -> str:
    n = sum(c.values())
    return f"{c.get(k, 0) / n:.0%}" if n else "-"


def write(res: dict) -> None:
    md = ["# Understanding vs translation errors (`analysis/error_split.py`)", "",
          "Generated-sentence groups come from subgraph_check, which does not separate 'word not in language' from",
          "'unreadable' (both are under 'unreadable' there).", "",
          "Share of test items in each group. Translation error = analysis right, glyphs wrong. Understanding error =",
          "analysis wrong, glyphs exactly the model's own analysis rendered (faithful translation of a misreading).",
          "Both wrong = analysis wrong and glyphs not even its own analysis.", ""]
    for test, n_note in (("generated", "300 generated sentences (from results/subgraph)"),
                         ("tatoeba", "Tatoeba 600")):
        md += [f"## {n_note}", "", "| run | n | " + " | ".join(GROUPS) + " |", "|---|---|" + "---|" * len(GROUPS)]
        for run, row in res.items():
            if test in row:
                c = row[test]
                md.append(f"| {run} | {sum(c.values())} | " + " | ".join(pct(c, g) for g in GROUPS) + " |")
        md.append("")
    g4 = ["both right", "translation error", "lucky glyphs", "analysis wrong, glyphs wrong"]
    md += ["## Sealed v2, 147 in scope (totals only)", "", "| run | n | " + " | ".join(g4) + " |",
           "|---|---|" + "---|" * len(g4)]
    full = {r: row["sealed"] for r, row in res.items() if "sealed" in row and "understanding error" in row["sealed"]
            or "sealed" in row and "both wrong" in row["sealed"]}
    for run, row in res.items():
        if "sealed" in row:
            c = collections.Counter(row["sealed"])
            if run in full:                              # collapse the full split to the four columns above
                c["analysis wrong, glyphs wrong"] = sum(c[g] for g in GROUPS[3:])
            md.append(f"| {run} | {sum(c[g] for g in g4)} | " + " | ".join(pct(collections.Counter(
                {g: c[g] for g in g4}), g) for g in g4) + " |")
    if full:
        md += ["", "Sealed, full split (re-scored since 30 Oct: per-entry category labels only, no text):", "",
               "| run | n | " + " | ".join(GROUPS) + " |", "|---|---|" + "---|" * len(GROUPS)]
        md += [f"| {run} | {sum(c.values())} | " + " | ".join(pct(c, g) for g in GROUPS) + " |" for run, c in full.items()]
    # writing judged on the model's OWN meaning (review, 30 Oct; the user's measure): among items whose own meaning
    # can be rendered, the share whose glyphs are not that rendering; reading = share of items with the meaning wrong
    md += ["", "## Writing judged on the model's own meaning (30 Oct)", "",
           "Writing error = glyphs differ from the model's own meaning rendered (right or wrong meaning), among items "
           "whose own meaning can be rendered (unreadable / word-not-in-language items excluded; they count as "
           "reading errors). Reading error = meaning wrong, share of all items.", "",
           "| run | test | renderable | writing error | reading error | writing errors / reading errors |",
           "|---|---|---|---|---|---|"]
    for run, row in res.items():
        for test in ("generated", "tatoeba", "sealed"):
            c = row.get(test)
            if not c or (test == "sealed" and run not in full):
                continue
            n = sum(c.values())
            unr = c.get("unreadable", 0) + c.get("word not in language", 0)
            w = c.get("translation error", 0) + c.get("both wrong", 0) + c.get("lucky glyphs", 0)
            rd = c.get("understanding error", 0) + c.get("both wrong", 0) + c.get("lucky glyphs", 0) + unr
            md.append(f"| {run} | {test} | {n - unr}/{n} | {w / max(1, n - unr):.2f} | {rd / n:.2f} | {w}/{rd} |")
    (ROOT / "results" / "error_split.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (ROOT / "results" / "error_split.json").write_text(json.dumps({r: {k: dict(v) for k, v in row.items()}
                                                                   for r, row in res.items()}, indent=1),
                                                       encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
