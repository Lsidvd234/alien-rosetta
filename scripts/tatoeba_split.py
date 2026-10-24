"""Split the filtered, reader-labelled Tatoeba sentences into run-3 training data and a held-out real-English test set.

    python scripts/tatoeba_split.py [--test 600]

Input : results/feasibility_tatoeba_filtered.json (strict filters; hand-checked label precision ~88%, 263/299)
Output: train/data/tatoeba_train.json   -> mixed into training by build_training(real_file=...) (paraphrase run 3)
        results/tatoeba_test.json        -> second real-English test set for every model (analysis/score_tatoeba.py)
The test set never overlaps the training file, and excludes the 299 hand-checked sentences (unbiased sample).
Tatoeba sentences are CC BY 2.0 FR (tatoeba.org); the language labels come from the sealed-set reader, used only after
the sealed set was frozen (sealed/FROZEN.md).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> None:
    n_test = int(sys.argv[sys.argv.index("--test") + 1]) if "--test" in sys.argv else 600
    from tatoeba_filter import reason_v1
    acc = json.loads((ROOT / "results" / "feasibility_tatoeba_accepted.json").read_text(encoding="utf-8"))
    kept = json.loads((ROOT / "results" / "feasibility_tatoeba_filtered.json").read_text(encoding="utf-8"))
    checked = {x["english"] for x in random.Random(0).sample(acc, 30)}
    checked |= {x["english"] for x in random.Random(99).sample(acc, 70)}
    prev = [x for x in acc if x["english"] not in checked and not reason_v1(x)]
    checked |= {x["english"] for x in random.Random(2026).sample(prev, 100)}
    fresh = [x for x in kept if x["english"] not in checked]
    checked |= {x["english"] for x in random.Random(3033).sample(fresh, 100)}
    seen, uniq = set(), []
    for x in kept:                                       # one entry per English sentence
        if x["english"] not in seen:
            seen.add(x["english"])
            uniq.append({"english": x["english"], "analysis": x["analysis"], "target": x["target"]})
    pool = [x for x in uniq if x["english"] not in checked]
    rng = random.Random(4242)
    test = rng.sample(pool, n_test)
    tset = {x["english"] for x in test}
    train = [x for x in uniq if x["english"] not in tset]
    (ROOT / "train" / "data" / "tatoeba_train.json").write_text(json.dumps(train, ensure_ascii=False), encoding="utf-8")
    (ROOT / "results" / "tatoeba_test.json").write_text(json.dumps(test, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(uniq):,} unique sentences -> train {len(train):,} ({len(train) / 80000:.1%} of 80k), test {len(test)}")


if __name__ == "__main__":
    main()
