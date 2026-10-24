"""Summarise Kaggle run results.

1) Summary table: model, language, script, N, tokens/sentence, per-token accuracy p,
   exact match (= greedy), p^L prediction, chrF, for each test split.
2) Controlled word-level analysis (needs per_item_ok, saved by newer runs): word error rate
   by number of tokens in the word, for CONTENT words only (grammar words like "the"/"not"
   are short AND frequent, which would confound the comparison), split by how often that
   exact word form appeared in the training subset.

Usage:
    python analysis/analyze.py results/kaggle            # scans */results/*.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conlang.dataset import PRESETS, TEST_SPLITS, build  # noqa: E402
from conlang.frames import Vocab  # noqa: E402

LANG = {"english_like": "easy", "tamil_like": "hard"}
FREQ_BUCKETS = [(0, 4), (5, 19), (20, 10**9)]
TOK_BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 9), (10, 10**6)]


def bucket(x, buckets):
    for a, b in buckets:
        if a <= x <= b:
            return f"{a}" if a == b else (f"{a}+" if b > 10**5 else f"{a}-{b}")
    return "?"


def load(root: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.rglob("results/*.json"))]


def summary(runs: list[dict]) -> None:
    print("| model | language | script | N | split | target tok/sent | token acc p | exact | p^L pred | chrF (gen) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        a = r["args"]
        for split in TEST_SPLITS:
            tf, g = r["eval"][split]["teacher_forced"], r["eval"][split]["generation"]
            print(f"| {a['model'].split('/')[-1]} | {LANG.get(a['preset'], a['preset'])} | {a['script']} | "
                  f"{a['n_train']} | {split} | {tf['mean_target_tokens']:.1f} | {tf['token_accuracy']:.3f} | "
                  f"{tf['exact_match']:.3f} | {tf['predicted_exact_from_p^L']:.3f} | {g['chrf']:.1f} |")


def word_analysis(r: dict) -> None:
    from transformers import AutoTokenizer

    a = r["args"]
    if "per_item_ok" not in r["eval"]["test_iid"]["teacher_forced"]:
        print(f"  (skip word analysis for {a['model']} {a['preset']} {a['script']}: no per-item data)")
        return
    vocab = Vocab.large()
    content = set(vocab.nouns) | set(vocab.locations) | set(vocab.transitive + vocab.intransitive) | set(vocab.adjectives)
    data = build(PRESETS[a["preset"]], a["script"], n_train=50000, n_test=int(a["n_test"]), seed=int(a["seed"]))
    train = data["splits"]["train"][:int(a["n_train"])]
    freq = Counter(w for it in train for w in it["target"].split())
    tok = AutoTokenizer.from_pretrained(a["model"])

    table = defaultdict(lambda: [0, 0])     # (kind, freq bucket, tok bucket) -> [words, errors]
    for split in TEST_SPLITS:
        tf = r["eval"][split]["teacher_forced"]
        items = {it["id"]: it for it in data["splits"][split]}
        for iid, okstr in zip(tf["per_item_ids"], tf["per_item_ok"]):
            it = items[iid]
            target, gloss = it["target"], it["gloss"].split()
            offs = tok(target, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            words: dict[int, list[bool]] = defaultdict(list)
            for (st, _), ch in zip(offs, okstr[:-1]):
                if st < len(target) and target[st] == " ":
                    st += 1
                words[target[:st].count(" ")].append(ch == "1")
            tw = target.split()
            for wi, oks in words.items():
                lemma = gloss[wi].split("-")[0].split(".")[0]
                kind = "content" if lemma in content else "grammar"
                key = (kind, bucket(freq[tw[wi]], FREQ_BUCKETS), bucket(len(oks), TOK_BUCKETS))
                table[key][0] += 1
                table[key][1] += not all(oks)

    print(f"\n### {a['model'].split('/')[-1]} | {LANG.get(a['preset'])} | {a['script']} | N={a['n_train']}")
    print("word error rate (words) by tokens-in-word; rows = kind x times the word form was seen in training")
    cols = [bucket(b[0], TOK_BUCKETS) for b in TOK_BUCKETS]
    print("| kind | seen in train | " + " | ".join(f"{c} tok" for c in cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for kind in ("content", "grammar"):
        for fb in [bucket(b[0], FREQ_BUCKETS) for b in FREQ_BUCKETS]:
            cells = []
            for c in cols:
                n, e = table[(kind, fb, c)]
                cells.append(f"{e/n:.2f} ({n})" if n else "-")
            print(f"| {kind} | {fb} | " + " | ".join(cells) + " |")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/kaggle")
    runs = [r for r in load(root) if r["args"].get("max_steps", "0") in ("0", 0)]
    if not runs:
        print("no full runs found"); return
    summary(runs)
    for r in runs:
        word_analysis(r)


if __name__ == "__main__":
    main()
