"""Scoring model outputs against the exact answer key.

- exact_match     whole sentence identical (main metric)
- chrf            character n-gram F-score (n=1..6, beta=2), corpus level, spaces ignored
                  (same definition as sacreBLEU's default chrF)
- word_accuracy   share of word positions that are exactly right
"""

from __future__ import annotations

import unicodedata
from collections import Counter


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip())


def exact_match(preds: list[str], refs: list[str]) -> float:
    return sum(_norm(p) == _norm(r) for p, r in zip(preds, refs)) / max(1, len(refs))


def _ngrams(s: str, n: int) -> Counter:
    s = s.replace(" ", "")
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(preds: list[str], refs: list[str], max_n: int = 6, beta: float = 2.0) -> float:
    """Corpus-level chrF in [0, 100]."""
    match = [0] * max_n
    hyp_total = [0] * max_n
    ref_total = [0] * max_n
    for p, r in zip(preds, refs):
        p, r = _norm(p), _norm(r)
        for n in range(1, max_n + 1):
            hp, rf = _ngrams(p, n), _ngrams(r, n)
            match[n - 1] += sum((hp & rf).values())
            hyp_total[n - 1] += sum(hp.values())
            ref_total[n - 1] += sum(rf.values())
    precs, recs = [], []
    for n in range(max_n):
        if hyp_total[n] and ref_total[n]:
            precs.append(match[n] / hyp_total[n])
            recs.append(match[n] / ref_total[n])
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return 100 * (1 + b2) * p * r / (b2 * p + r)


def word_accuracy(preds: list[str], refs: list[str]) -> float:
    right = total = 0
    for p, r in zip(preds, refs):
        pw, rw = _norm(p).split(), _norm(r).split()
        total += max(len(pw), len(rw))
        right += sum(a == b for a, b in zip(pw, rw))
    return right / max(1, total)


def score(preds: list[str], refs: list[str]) -> dict:
    return {"exact_match": exact_match(preds, refs), "chrf": chrf(preds, refs),
            "word_accuracy": word_accuracy(preds, refs), "n": len(refs)}
