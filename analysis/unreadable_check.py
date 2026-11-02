"""Why are some predicted analyses unreadable? (2026-10-26; Tatoeba only, never the sealed set)

    python analysis/unreadable_check.py        -> results/unreadable_check.md

For every run with saved Tatoeba predictions, the outputs whose analysis cannot be read back into events
(conlang.v2.from_analysis fails or returns nothing) are sorted into:
  no '=>'        the output never reached the glyphs (cut by the length limit, or the model stopped / looped)
  looping        the analysis repeats one field or token run many times (a degenerate loop)
  bad field      a field name or ending the format does not have (e.g. an invented relation)
  other          readable separators but the reader of the format rejects it (first issue message counted)
plus the output length in tokens, to see whether the 500-token limit of analysis/score_tatoeba.py was hit.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sealed" / "app"))
from conlang.v2 import FIELDS, from_analysis  # noqa: E402
from english_parse import spell_name  # noqa: E402

NO_ARROW = "no '=>'"
FIELD_RE = re.compile(r"(?:^|\s)([A-Z]+)=")


def kind(pred: str) -> tuple[str, str]:
    ana, sep, _ = pred.partition("=>")
    ana = ana.strip()
    toks = ana.split()
    if toks and max(collections.Counter(toks).values()) >= 6:
        return "looping", ""
    if not sep:
        return NO_ARROW, ""
    bad = [f for f in FIELD_RE.findall(ana) if f not in FIELDS]
    if bad:
        return "bad field", bad[0]
    try:
        cls, issues = from_analysis(ana, spell_name)
    except Exception as e:                                     # noqa: BLE001
        return "other", type(e).__name__ + ": " + str(e)[:60]
    if not cls:
        return "other", (issues[0][:60] if issues else "empty")
    return "readable", ""


def main() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT / "results" / "kaggle" / "e2b-v2fix-cot-80k-noheld" / "model")
    md = ["# Why predicted analyses are unreadable (Tatoeba 600; `analysis/unreadable_check.py`)", "",
          "| run | unreadable | no '=>' | looping | bad field | other | at the 500-token limit | top reasons |",
          "|---|---|---|---|---|---|---|---|"]
    for f in sorted((ROOT / "results" / "tatoeba").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if not r.get("cot") or "preds" not in r:
            continue
        c, why, at_limit = collections.Counter(), collections.Counter(), 0
        for p in r["preds"]:
            k, w = kind(p)
            if k == "readable":
                continue
            c[k] += 1
            if w:
                why[w] += 1
            if len(tok(p, add_special_tokens=False)["input_ids"]) >= 495:
                at_limit += 1
        n = sum(c.values())
        top = "; ".join(f"{k} ({v})" for k, v in why.most_common(3))
        md.append(f"| {f.stem} | {n} | {c[NO_ARROW]} | {c['looping']} | {c['bad field']} | {c['other']} | "
                  f"{at_limit} | {top} |")
        print(md[-1], flush=True)
    (ROOT / "results" / "unreadable_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
