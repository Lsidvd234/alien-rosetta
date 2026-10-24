"""Final 20k comparison: 2x2 with three random languages each (mean and spread), plus the
cause tests (character-level Latin, character-level accented Latin). Writes results/final_20k.md.

    python analysis/final_20k.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curve_table import content_word_accuracy  # noqa: E402
from conlang.dataset import PRESETS, TEST_SPLITS, build  # noqa: E402
from conlang.frames import Vocab  # noqa: E402
from train.run import unknown_chars  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CELLS = ["easy-latin", "easy-tamil", "hard-latin", "hard-tamil"]
CAUSE = [("easy", "Latin, normal", "c-easy-latin-20k"), ("easy", "Latin, 1 char/token", "m2-easy-latinchar-20k"),
         ("easy", "Tamil script", "c-easy-tamil-20k"), ("easy", "Invented glyphs (PUA)", "p-easy-pua-20k"),
         ("hard", "Latin, normal", "c-hard-latin-20k"), ("hard", "Latin, 1 char/token", "m2-hard-latinchar-20k"),
         ("hard", "Accented Latin, 1 char/token", "m3-hard-accentchar-20k"), ("hard", "Tamil script", "c-hard-tamil-20k"),
         ("hard", "Invented glyphs (PUA)", "p-hard-pua-20k"),
         ("easy", "Invented glyphs, 1 new token each", "p-easy-puatok-20k"),
         ("hard", "Invented glyphs, 1 new token each", "p-hard-puatok-20k")]
FIXES = [("Baseline", "new glyph tokens start as copies of random words", "p-hard-puatok-50k"),
         ("Option 1", "each glyph starts as the embedding of its Latin sound", "g1-hard-latininit-50k"),
         ("Option 2", "baseline + glyph<->Latin transliteration examples", "g2-hard-translit-50k"),
         ("Option 3a", "baseline + plain-text phase (50k glyph-only sentences) first", "g3-hard-plaintext-50k"),
         ("Control", "each glyph starts at a WRONG sound (shuffled)", "s4-hard-shuffledinit-50k"),
         ("Target", "Latin script", "c-hard-latin-50k"),
         ("Target", "Tamil script", "g-hard-tamil-50k")]


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    V = Vocab.large()
    content = set(V.nouns) | set(V.locations) | set(V.transitive + V.intransitive) | set(V.adjectives)
    cache: dict = {}

    def score(name: str) -> dict | None:
        p = ROOT / "results" / "kaggle" / name / "results" / f"{name}.json"
        if not p.exists():
            return None
        r = json.loads(p.read_text(encoding="utf-8"))
        a = r["args"]
        key = (a["preset"], a["script"], a["seed"])
        if key not in cache:
            cache[key] = build(PRESETS[a["preset"]], a["script"], n_train=50000, n_test=int(a["n_test"]),
                               seed=int(a["seed"]))
        t = tok
        if a.get("add_tokens") == "True":               # same single-glyph tokens as the run (M3)
            t = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
            t.add_tokens(unknown_chars(t, cache[key]))
        return {"cwa": content_word_accuracy(r, cache[key], t, content),
                "exact": r["eval"]["test_iid"]["teacher_forced"]["exact_match"],
                "tok": r["data"]["mean_target_tokens"], "min": r["training"]["seconds"] / 60}

    out = ["# Final 20k results (Qwen3-0.6B-Base, LoRA r=16, 3 epochs, 20,000 sentences)", "",
           "Three random languages per cell (seeds 0, 1, 2; same grammar, different invented vocabulary).",
           "Content-word accuracy = every token of the content word right. Exact = whole sentence right (test_iid).", "",
           "| cell | content-word acc (seed 0 / 1 / 2) | mean | sd | exact iid mean | tokens/sent | train min |",
           "|---|---|---|---|---|---|---|"]
    for cell in CELLS:
        res = [score(f"{p}-{cell}-20k") for p in ("c", "s1", "s2")]
        res = [x for x in res if x]
        c = [x["cwa"] for x in res]
        sd = st.stdev(c) if len(c) > 1 else 0.0
        out.append(f"| {cell} | {' / '.join(f'{v:.3f}' for v in c)} | {st.mean(c):.3f} | {sd:.3f} | "
                   f"{st.mean(x['exact'] for x in res):.3f} | {st.mean(x['tok'] for x in res):.1f} | "
                   f"{st.mean(x['min'] for x in res):.1f} |")
    out += ["", "## Cause tests (seed 0, 20k)", "",
            "| language | target written as | tokens/sent | content-word acc | exact iid | train min |",
            "|---|---|---|---|---|---|"]
    for lang, label, name in CAUSE:
        x = score(name)
        if x:
            out.append(f"| {lang} | {label} | {x['tok']:.1f} | {x['cwa']:.3f} | {x['exact']:.3f} | {x['min']:.1f} |")
    out += ["", "## Invented glyphs (PUA) by training size", "",
            "| language | glyphs as | sentences | tokens/sent | content-word acc | exact iid | exact comp | exact morph | exact long | train min |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for lang in ("easy", "hard"):
        for label, kind in (("3 byte tokens each", "pua"), ("1 new token each", "puatok")):
            for n in (20, 35, 50, 80):
                name = f"p-{lang}-{kind}-{n}k"
                x = score(name)
                if x:
                    ex = json.loads((ROOT / "results" / "kaggle" / name / "results" / f"{name}.json")
                                    .read_text(encoding="utf-8"))["eval"]
                    e = " | ".join(f"{ex[s]['teacher_forced']['exact_match']:.3f}" for s in TEST_SPLITS)
                    out.append(f"| {lang} | {label} | {n}k | {x['tok']:.1f} | {x['cwa']:.3f} | {e} | {x['min']:.1f} |")
    out += ["", "1-new-token runs: 20k used mean-embedding init, 50k used copy-of-random-ASCII-token init."]
    out += ["", "## Making hard grammar + invented glyphs generalise (50k, glyphs as 1 new token each)", "",
            "Each fix changes one thing against the baseline. Latin and Tamil script rows are the targets.", "",
            "| run | change | content-word acc | exact iid | exact comp | exact morph | exact long | train min |",
            "|---|---|---|---|---|---|---|---|"]
    for label, change, name in FIXES:
        x = score(name)
        if x:
            ex = json.loads((ROOT / "results" / "kaggle" / name / "results" / f"{name}.json")
                            .read_text(encoding="utf-8"))["eval"]
            e = " | ".join(f"{ex[s]['teacher_forced']['exact_match']:.3f}" for s in TEST_SPLITS)
            out.append(f"| {label} | {change} | {x['cwa']:.3f} | {e} | {x['min']:.1f} |")
    out += ["", "Option 2 trains on 2x the examples (translation + transliteration), so ~2x the compute. "
            "Option 3a adds a plain-text phase on 50k extra sentences before translation. One seed each; "
            "re-runs with saved weights (s1/s2/s3-*) reproduced every score exactly."]
    text = "\n".join(out) + "\n"
    (ROOT / "results" / "final_20k.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
