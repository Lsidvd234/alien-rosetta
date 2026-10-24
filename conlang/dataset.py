"""Build a dataset (one training pool + four fixed test splits) for one language + script.

Training sizes are NESTED: train.jsonl is shuffled once, and a run with N
examples uses its first N lines. So every training size is scored on the SAME
test sets, and bigger N only adds data.

Test splits (fixed, `n_test` each), so results show learning, not memorisation:
- test_iid    held-out sentences of the same kinds as training
- test_comp   verb+object pairs never seen together in training (about 10% of pairs held out)
- test_morph  a feature combination never seen in training: past + negated + plural subject
- test_long   the longest sentence type (adjectives on both nouns + a location), never in training
Test sentences whose words never appear anywhere in the training pool are dropped
(they would be unlearnable). Each item lists its content words so the evaluator
can also report accuracy on items whose words all appear in a given first-N subset.

Usage:
    python -m conlang.dataset --preset english_like --script latin --n-train 50000 --out data/easy_latin
    python -m conlang.dataset --preset tamil_like --script abugida_tamil --out data/hard_tamil
    python -m conlang.dataset --list-variants
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

from .frames import NP, Frame, Vocab, english, sample_frame
from .grammar import ENGLISH_LIKE, TAMIL_LIKE, GrammarConfig, Language, dial_variants
from .script import SCRIPTS, Script, render_sentence

PRESETS = {"english_like": ENGLISH_LIKE, "tamil_like": TAMIL_LIKE}
TEST_SPLITS = ("test_iid", "test_comp", "test_morph", "test_long")


def _np_dict(x):
    return asdict(x) if isinstance(x, NP) else x


def heldout_pair(seed: int, verb: str, noun: str, frac: float = 0.10) -> bool:
    h = hashlib.md5(f"{seed}|{verb}|{noun}".encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32 < frac


def is_long(f: Frame) -> bool:
    return (f.loc is not None and f.obj is not None and isinstance(f.subj, NP)
            and f.subj.adj is not None and f.obj.adj is not None)


def split_of(f: Frame, vocab: Vocab, seed: int) -> str | None:
    if is_long(f):
        return "test_long"
    if f.obj is not None and heldout_pair(seed, f.verb, f.obj.noun):
        return "test_comp"
    _, number, _ = f.subject_png(vocab)
    if f.tense == "past" and f.neg and number == "pl":
        return "test_morph"
    return None


def _item(f: Frame, lang: Language, script: Script, vocab: Vocab) -> dict:
    real = lang.realize(f)
    return {
        "english": english(f, vocab),
        "target": render_sentence(real.words, script),
        "gloss": " ".join(w.gloss for w in real.words),
        "content": f.content_words(),
        "frame": {"subj": _np_dict(f.subj), "verb": f.verb, "obj": _np_dict(f.obj),
                  "loc": f.loc, "tense": f.tense, "neg": f.neg},
        "stats": {"words": len(real.words), "boundaries": real.boundaries,
                  "sandhi_changes": real.sandhi_changes,
                  "morphemes_per_word": sum(real.morphemes) / len(real.morphemes)},
    }


def build(config: GrammarConfig, script_name: str, n_train: int = 50000, n_test: int = 500,
          seed: int = 0, lexicon_seed: int | None = None, vocab: Vocab | None = None,
          n_extra: int = 0) -> dict:
    """Returns {"meta": ..., "splits": {name: [items]}}.

    `seed` controls which sentences are sampled; `lexicon_seed` (default: seed)
    controls the invented words. Same lexicon seed + different script = the
    same language written differently.
    `n_extra` appends more training sentences AFTER the test sets are drawn, so the
    first `n_train` training sentences and every test set are identical to a build
    without them (lets runs go beyond the standard 50k pool).
    """
    vocab = vocab or Vocab.large()
    lang = Language(config, seed if lexicon_seed is None else lexicon_seed, vocab)
    script = Script(script_name, config.phonology)
    rng = random.Random(seed)

    want = {"train": n_train, "test_iid": n_test, "test_comp": n_test, "test_morph": n_test}
    frames: dict[str, list[Frame]] = {k: [] for k in want}
    long_frames: list[Frame] = []
    seen: set[str] = set()
    tries, max_tries = 0, 400 * (n_train + 4 * n_test)

    def add(f: Frame, bucket: list) -> None:
        en = english(f, vocab)
        if en not in seen:
            seen.add(en)
            bucket.append(f)

    while any(len(frames[k]) < want[k] for k in want) and tries < max_tries:
        tries += 1
        f = sample_frame(rng, vocab)
        split = split_of(f, vocab, seed)
        if split == "test_long":
            if len(long_frames) < n_test:
                add(f, long_frames)
            continue
        if split is None:
            # ordinary sentence: fill training first, then the iid test
            split = "train" if len(frames["train"]) < n_train else "test_iid"
        if len(frames[split]) < want[split]:
            add(f, frames[split])
    while len(long_frames) < n_test and tries < max_tries:
        tries += 1
        f = sample_frame(rng, vocab, long=True)
        if split_of(f, vocab, seed) == "test_long":
            add(f, long_frames)
    extra: list[Frame] = []
    while len(extra) < n_extra and tries < max_tries + 400 * n_extra:
        tries += 1
        f = sample_frame(rng, vocab)
        if split_of(f, vocab, seed) is None:
            add(f, extra)

    splits = {k: [_item(f, lang, script, vocab) for f in v] for k, v in frames.items()}
    splits["test_long"] = [_item(f, lang, script, vocab) for f in long_frames]

    # drop test items with a word that never appears in the training pool (the first
    # n_train only, so extra sentences never change the test sets)
    train_words = {w for it in splits["train"] for w in it["content"]}
    splits["train"] += [_item(f, lang, script, vocab) for f in extra]
    dropped = {}
    for k in TEST_SPLITS:
        keep = [it for it in splits[k] if all(w in train_words for w in it["content"])]
        dropped[k] = len(splits[k]) - len(keep)
        splits[k] = keep
    for name, items in splits.items():
        for i, it in enumerate(items):
            it["id"] = f"{name}-{i:06d}"

    meta = {
        "config": config.as_dict(), "script": script_name, "seed": seed, "lexicon_seed": lang.seed,
        "vocab": vocab.name, "vocab_size": vocab.size(),
        "train_vocab_coverage": len(train_words) / vocab.size(),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "dropped_unseen_word_tests": dropped,
        "heldout_rule": "md5(seed|verb|noun) < 10% -> verb+object pair only in test_comp",
        "grammatical_endings": {g: script.word(f) for g, f in lang.grams.items()},
        "irregular_past": {v: script.word(f) for v, f in lang.irregular_past.items()},
        "lexicon": {c: script.word(r) for c, r in lang.roots.items()},
    }
    return {"meta": meta, "splits": splits}


def write(data: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(data["meta"], ensure_ascii=False, indent=1), encoding="utf-8")
    for name, items in data["splits"].items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for it in items:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")


def _parse_set(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, v = p.split("=", 1)
        if v in ("True", "False"):
            v = v == "True"
        else:
            try:
                v = float(v) if "." in v else int(v)
            except ValueError:
                pass
        out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS) + list(dial_variants()), default="english_like")
    ap.add_argument("--set", nargs="*", default=[], help="override dials, e.g. word_order=SOV sandhi=True")
    ap.add_argument("--script", choices=SCRIPTS, default="latin")
    ap.add_argument("--n-train", type=int, default=50000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--vocab", choices=["large", "small"], default="large")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lexicon-seed", type=int, default=None)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--list-variants", action="store_true")
    a = ap.parse_args()

    if a.list_variants:
        for name, cfg in dial_variants().items():
            print(name, json.dumps(cfg.as_dict()))
        return
    base = PRESETS.get(a.preset) or dial_variants()[a.preset]
    cfg = replace(base, **_parse_set(a.set))
    vocab = Vocab.large() if a.vocab == "large" else Vocab.small()
    data = build(cfg, a.script, a.n_train, a.n_test, a.seed, a.lexicon_seed, vocab)
    if a.out:
        write(data, a.out)
    m = data["meta"]
    print(json.dumps({"split_sizes": m["split_sizes"], "vocab_size": m["vocab_size"],
                      "train_vocab_coverage": round(m["train_vocab_coverage"], 3),
                      "dropped_unseen_word_tests": m["dropped_unseen_word_tests"]}))
    for it in data["splits"]["train"][:4]:
        print(f"  {it['english']}\n    -> {it['target']}\n    gloss: {it['gloss']}")


if __name__ == "__main__":
    main()
