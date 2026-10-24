"""Stricter filters for reader-labelled Tatoeba sentences (paraphrase run 3), from the error patterns found by hand.

    python scripts/tatoeba_filter.py [--show 100]   -> results/feasibility_tatoeba_filtered.json + a fresh check sample

Input: results/feasibility_tatoeba_accepted.json (sentences the reader read with no issue / guess / unknown word).
Drops (patterns that were wrong in the first 99-sentence hand check):
  quantity-of   "no one", "none of", "a couple of", "dozens of", "lots of" ... (read as an owner / greeting)
  time-slot     a time word (morning, night, week, day ...) that lands outside a time field (T, M, FOR, DURING, BEFORE,
                AFTER, UNTIL, SINCE): "an appointment yesterday morning" -> appointment_morning, "her tomorrow night"
  phrasal       verb + particle + determiner ("took on a new secretary"): the particle is read as a relation field
Prints a fresh random sample (excluding the 99 already checked) for the next hand check.
"""

from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUANT = re.compile(r"\b(no one|no-one|none of|couple of|dozens? of|lots? of|plenty of|hundreds of|thousands of|"
                   r"a number of|kind of|sort of|a bit of|most of|all of|some of|each of|both of|one of|many of)\b", re.I)
TIME = {"morning", "mornings", "night", "nights", "evening", "evenings", "afternoon", "afternoons", "week", "weeks",
        "weekend", "weekends", "day", "days", "month", "months", "year", "years", "hour", "hours", "minute",
        "minutes", "second", "seconds", "summer", "winter", "spring", "autumn", "today", "tonight", "tomorrow",
        "yesterday"}
TIME_FIELDS = {"T", "M", "FOR", "DURING", "BEFORE", "AFTER", "UNTIL", "SINCE"}
PHRASAL = re.compile(r"\b[a-z]+(?:ed|s)?\s+(on|off|out|up|over|down|back|away|in)\s+"
                     r"(a|an|the|my|his|her|our|their|your|this|that|these|those)\b", re.I)
REL_WORDS = {"TO": {"to", "into"}, "WITH": {"with"}, "ON": {"on", "onto", "upon"}, "THROUGH": {"through"},
             "FROM": {"from"}, "FOR": {"for"}, "ABOUT": {"about"}, "THAN": {"than"}, "UNDER": {"under", "below",
             "beneath", "underneath"}, "NEAR": {"near", "beside", "by"}, "BEHIND": {"behind"}, "FRONT": {"front"},
             "WITHOUT": {"without"}, "BEFORE": {"before"}, "AFTER": {"after"}, "AROUND": {"around"},
             "BETWEEN": {"between"}, "DURING": {"during"}, "UNTIL": {"until", "till"}, "SINCE": {"since"},
             "AGAINST": {"against"}, "L": {"in", "at", "on", "inside", "within"}}
FIELD_RE = re.compile(r"(?<![A-Za-z0-9_.>^+&|])([A-Z]+)=(\S+)")


def reason_v1(x: dict) -> str | None:
    """The first three filters only (to rebuild which 100 sentences the second check saw)."""
    en, ana = x["english"], x["analysis"]
    if QUANT.search(en):
        return "q"
    for f, v in FIELD_RE.findall(ana):
        if set(re.split(r"[\^&|+_.>]", v.lower())) & TIME and f not in TIME_FIELDS:
            return "t"
    return "p" if PHRASAL.search(en) else None


def reason(x: dict) -> str | None:
    en, ana = x["english"], x["analysis"]
    if QUANT.search(en):
        return "quantity-of / no one"
    for f, v in FIELD_RE.findall(ana):
        parts = set(re.split(r"[\^&|+_.>]", v.lower()))
        if parts & TIME and f not in TIME_FIELDS:
            return "time word outside a time field"
    if PHRASAL.search(en):
        return "phrasal verb + object"
    if re.search(r"\bof\b", en, re.I):
        return "contains 'of'"
    words = set(re.findall(r"[a-z]+", en.lower()))
    for f, _ in FIELD_RE.findall(ana):
        need = REL_WORDS.get(f)
        if need and not (need & words):
            return "relation field without its English word"
    return None


def main() -> None:
    show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 100
    acc = json.loads((ROOT / "results" / "feasibility_tatoeba_accepted.json").read_text(encoding="utf-8"))
    checked = {x["english"] for x in random.Random(0).sample(acc, min(30, len(acc)))}
    checked |= {x["english"] for x in random.Random(99).sample(acc, 70)}
    prev = [x for x in acc if x["english"] not in checked and not reason_v1(x)]       # the 2nd check's 100
    checked |= {x["english"] for x in random.Random(2026).sample(prev, 100)}
    drops = collections.Counter()
    keep = []
    for x in acc:
        r = reason(x)
        if r:
            drops[r] += 1
        else:
            keep.append(x)
    (ROOT / "results" / "feasibility_tatoeba_filtered.json").write_text(json.dumps(keep, ensure_ascii=False),
                                                                        encoding="utf-8")
    print(f"accepted {len(acc):,} -> kept {len(keep):,} ({len(keep) / len(acc):.0%}); dropped {dict(drops)}")
    print(f"kept = {len(keep) / 60000:.1%} of the 60,000 sampled sentences "
          f"(full 2.04M Tatoeba -> about {len(keep) / 60000 * 2039120:,.0f})")
    fresh = [x for x in keep if x["english"] not in checked]
    for i, x in enumerate(random.Random(3033).sample(fresh, show), 1):
        print(f"{i}. {x['english']} | {x['analysis']}")


if __name__ == "__main__":
    main()
