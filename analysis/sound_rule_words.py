"""Sound-rule language: does the model write words it never saw? (local; generated sentences only, never the sealed set)

    python analysis/sound_rule_words.py [RUN]   -> results/sound_rule_words_<RUN>.md + .json   (RUN: e2b-v2fix-soundrule)
    python analysis/sound_rule_words.py RUN --from FILE.json   (a result file elsewhere, e.g. a smoke test)

Reads the run's own test "sound rule: new words (never in training)" (train/represent_v2.py eval_sets; every item and
prediction saved by train/run.py). In the sound-rule language a content word is its English word respelled by fixed
letter rules (conlang/v2.py loan_sounds), and 20% of content words (seeded) never appear in training. For every word
of every gold answer:
  new word      its English base word was held out of training; glyph word right = the gold glyph word (sounds + the
                sentence's endings) is in the model's glyphs, aligned
  seen word     every other content word in the same sentences (the control: same sentences, same model)
  recognised    the model's MEANING contains the held-out word's key (it read the English word as a word of the
                language), whatever it wrote in glyphs
  closeness     for a wrong new word: best character similarity to any word the model wrote (near miss >= 0.75)
Rule-learning predicts new words close to seen ones; list-learning predicts new words near 0 (as v2, where an unseen
root is random and cannot be guessed). Predictions are in results/findings_2026-10-24.md.
"""

from __future__ import annotations

import difflib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sealed" / "app"))                 # the reader's code only (no sealed data)

RUN = "e2b-v2fix-soundrule"
SET = "sound rule: new words (never in training)"


def aligned_ok(gold: list[str], pred: list[str]) -> list[bool]:
    ok = [False] * len(gold)
    for blk in difflib.SequenceMatcher(a=gold, b=pred, autojunk=False).get_matching_blocks():
        for j in range(blk.size):
            ok[blk.a + j] = True
    return ok


def main() -> None:
    import train.represent_v2 as R2
    from conlang.v2 import from_analysis
    from english_parse import spell_name, with_gender
    args = [a for i, a in enumerate(sys.argv[1:]) if not a.startswith("--") and (i == 0 or sys.argv[i] != "--from")]
    run = args[0] if args else RUN
    f = Path(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else \
        ROOT / "results" / "kaggle" / run / "results" / f"{run}.json"
    r = json.loads(f.read_text(encoding="utf-8"))
    cfg = r["args"]
    assert R2.language_of_run(cfg), f"{run} is not a sound-rule run"
    g = r["eval_gen"][SET]
    rep = R2.RepV2(cfg.get("preset", "tamil_like"), cfg.get("script", "abugida_pua"), int(cfg.get("seed", 0)))
    held = R2.SOUND["held"]
    from conlang.v2 import CONTENT_ROLES
    content = {c.partition(":")[2] for c in rep.lang.roots if c.partition(":")[0] in CONTENT_ROLES and c != "V:be"}
    tot = {"new": [0, 0], "seen": [0, 0]}                         # [words, glyph word right]
    recog, close, skipped, examples = [0, 0], [], 0, []
    for it, p in zip(g["items"], g["preds"]):
        ana_p, _, gly_p = p.partition("=>")
        pred_keys = set(R2.analysis_keys(ana_p))
        cls, _ = from_analysis(it["analysis"], spell_name)
        words = None
        for gs in itertools.product("mf", repeat=min(len(cls), 6)):  # the gold's own name genders
            ws = [w for c, x in zip(cls, gs + ("m",) * 9) for w in rep.words(with_gender(c, x))]
            if " ".join(rep.script.word(w.phonemes) for w in ws) == it["target"]:
                words = ws
                break
        if words is None:
            skipped += 1
            continue
        gold = [rep.script.word(w.phonemes) for w in words]
        ok = aligned_ok(gold, gly_p.split())
        for w, glyph, right in zip(words, gold, ok):
            bases = w.gloss.split("-")[0].split("+")
            if not any(b in content for b in bases):
                continue                                          # grammar words (pronouns, articles, case words)
            kind = "new" if set(bases) & held else "seen"
            tot[kind][0] += 1
            tot[kind][1] += right
            if kind == "new":
                recog[0] += 1
                recog[1] += any(k in pred_keys for k in R2.analysis_keys(it["analysis"])
                                if set(R2.bases_of(k)) & set(bases))
                if not right:
                    close.append(max((difflib.SequenceMatcher(a=glyph, b=x, autojunk=False).ratio()
                                      for x in gly_p.split()), default=0.0))
        if len(examples) < 10:
            examples.append((it["english"], it["new_words"], it["analysis"], ana_p.strip()))
    R2.language_of_run({})
    res = {"run": run, "sentences": len(g["items"]), "skipped (gold not re-rendered)": skipped,
           "sentence_glyph_exact": g["exact"], "sentence_meaning_exact": g.get("analysis_exact"),
           "new_words": tot["new"][0], "new_glyph_right": tot["new"][1] / max(1, tot["new"][0]),
           "seen_words": tot["seen"][0], "seen_glyph_right": tot["seen"][1] / max(1, tot["seen"][0]),
           "new_recognised_in_meaning": recog[1] / max(1, recog[0]),
           "new_wrong": len(close), "near_miss": sum(c >= 0.75 for c in close) / max(1, len(close)),
           "closeness_median": sorted(close)[len(close) // 2] if close else None}
    L = [f"# Sound-rule language: words never seen in training (`analysis/sound_rule_words.py`), run {run}", "",
         f"{res['sentences']} test sentences that each use at least one held-out word "
         f"(sentence glyphs exact {res['sentence_glyph_exact']:.3f}"
         + (f", meaning exact {res['sentence_meaning_exact']:.3f}" if res["sentence_meaning_exact"] is not None else "")
         + f"; {skipped} skipped: gold not re-rendered).", "",
         "| words | n | glyph word right |", "|---|---|---|",
         f"| new (never in training) | {res['new_words']} | **{res['new_glyph_right']:.2f}** |",
         f"| seen (same sentences) | {res['seen_words']} | {res['seen_glyph_right']:.2f} |", "",
         f"New words read as words of the language (their key in the model's meaning): "
         f"{res['new_recognised_in_meaning']:.2f}. Wrong new words: {res['new_wrong']}, near misses (closeness >= 0.75) "
         f"{res['near_miss']:.0%}, median closeness "
         + ("-" if res["closeness_median"] is None else f"{res['closeness_median']:.2f}") + ".", "",
         "## Examples", ""]
    for e, nw, ga, pa in examples:
        L += [f"- {e}  (new: {', '.join(nw)})  \n  gold: `{ga}`  \n  model: `{pa}`"]
    out = (f.parent if "--from" in sys.argv else ROOT / "results") / f"sound_rule_words_{run}"
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
