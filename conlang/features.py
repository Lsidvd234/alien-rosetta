"""Token-cost and grammar measurements for each language x script (no training needed).

For every (language, script) pair:
- tokens_per_word / tokens_per_sentence   how much the tokenizer fragments the text
- bytes_per_token
- broken_token_rate      tokens that are not whole characters (partial UTF-8 byte pieces)
- morphemes_per_word     how many pieces (root + endings) each word is built from
- type_token_ratio       distinct word forms / words
- chars_per_word

Usage:
    python -m conlang.features --tokenizers Qwen/Qwen2.5-0.5B-Instruct unsloth/gemma-3-1b-it
    python -m conlang.features --variants      # also every one-dial variant of the easy language
"""

from __future__ import annotations

import argparse
import json

from .dataset import build
from .grammar import ENGLISH_LIKE, TAMIL_LIKE, dial_variants
from .script import SCRIPTS


def grammar_features(items: list[dict]) -> dict:
    words = [w for it in items for w in it["target"].split()]
    b = sum(it["stats"]["boundaries"] for it in items)
    return {
        "morphemes_per_word": sum(it["stats"]["morphemes_per_word"] for it in items) / len(items),
        "sandhi_rate": sum(it["stats"]["sandhi_changes"] for it in items) / b if b else 0.0,
        "type_token_ratio": len(set(words)) / len(words),
        "chars_per_word": sum(map(len, words)) / len(words),
        "words_per_sentence": len(words) / len(items),
    }


def script_features(items: list[dict], tokenizer) -> dict:
    n_tok = n_words = n_bytes = broken = 0
    for it in items:
        text = it["target"]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        n_tok += len(ids)
        n_words += len(text.split())
        n_bytes += len(text.encode("utf-8"))
        broken += sum("�" in tokenizer.decode([i]) for i in ids)
    return {"tokens_per_word": n_tok / n_words, "tokens_per_sentence": n_tok / len(items),
            "bytes_per_token": n_bytes / n_tok, "broken_token_rate": broken / n_tok}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizers", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=2000, help="training sentences to measure on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", action="store_true", help="include one-dial variants of the easy language")
    a = ap.parse_args()

    toks = {}
    if a.tokenizers:
        from transformers import AutoTokenizer
        toks = {name: AutoTokenizer.from_pretrained(name) for name in a.tokenizers}

    langs = {"easy": ENGLISH_LIKE, "hard": TAMIL_LIKE}
    if a.variants:
        langs.update({f"easy+{k}": v for k, v in dial_variants().items() if k not in ("base", "all_flipped")})

    rows = []
    for lname, cfg in langs.items():
        for script in SCRIPTS:
            items = build(cfg, script, n_train=a.n, n_test=50, seed=a.seed)["splits"]["train"]
            row = {"language": lname, "script": script, **grammar_features(items)}
            for tname, tok in toks.items():
                for k, v in script_features(items, tok).items():
                    row[f"{tname.split('/')[-1]}:{k}"] = v
            rows.append(row)
    print(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
