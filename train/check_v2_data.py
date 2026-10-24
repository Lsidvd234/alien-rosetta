"""Check the v2 training-data generator against the sealed-set reader (two independent implementations).

    python -m train.check_v2_data [n]

Generates n clean examples (no typos, no synonyms, single spaces, sentence case), reads each English input with the
reader in sealed/app/english_parse.py, and compares the meanings (analysis). Agreement means the generator's labels
follow the same conventions the sealed set is scored with; disagreements are listed by kind of utterance so that
generator bugs (wrong English for a meaning) and reader limits (English it reads differently) can be told apart.
Also prints vocabulary coverage of a training build. Uses generated English only; never looks at the sealed set.
"""

from __future__ import annotations

import collections
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sealed" / "app"))

from english_parse import Parser, render  # noqa: E402

from train.represent_v2 import GenV2, build_training, coverage  # noqa: E402

CLEAN = {"noise": "none", "lexical": "plain", "spacing": "single", "case": "sentence"}


def main(n: int = 600) -> None:
    gen = GenV2("tamil_like", "abugida_pua", 0)
    p = Parser()
    rng = random.Random(11)
    stats = collections.Counter()
    by_kind = collections.defaultdict(collections.Counter)
    bad = collections.defaultdict(list)
    t0 = time.time()
    while stats["total"] < n:
        ex = gen.example(rng, force=dict(CLEAN))
        it = gen.item(ex) if ex else None
        if not it:
            continue
        kind = ex["choices"]["kind"] if len(ex["clauses"]) == 1 else "several events"
        stats["total"] += 1
        res = p.parse(it["english"])
        if not res.clauses:
            stats["unread"] += 1
            by_kind[kind]["unread"] += 1
            bad[kind].append((it["english"], "UNREAD " + "; ".join(res.issues)[:120], it["analysis"]))
            continue
        got = render(res.clauses, p.reps)["analysis"]
        if got == it["analysis"]:
            stats["same"] += 1
            by_kind[kind]["same"] += 1
        else:
            stats["different"] += 1
            by_kind[kind]["different"] += 1
            bad[kind].append((it["english"], got, it["analysis"]))
    print(f"{dict(stats)}  agreement {stats['same'] / stats['total']:.3f}  ({time.time() - t0:.0f} s)")
    for k, c in sorted(by_kind.items()):
        tot = sum(c.values())
        print(f"  {k:18} n={tot:4}  same {c['same'] / tot:.2f}  different {c['different']}  unread {c['unread']}")
    for k, items in bad.items():
        print(f"\n--- {k}: {len(items)}")
        for en, got, want in items[:8]:
            print(f"   {en}\n      reader   : {got}\n      generator: {want}")
    if "--coverage" in sys.argv:
        items, info = build_training("tamil_like", "abugida_pua", 0, n=int(sys.argv[sys.argv.index("--coverage") + 1]))
        print("\ncoverage", coverage(items, gen), info["n"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 600)
