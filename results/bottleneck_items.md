# Where exactly does the writer fail? (`analysis/bottleneck_items.py`)

reader = e2b-v2fix-passive0-meaning, writer = e2b-v2fix-passive0-writer (given the GOLD meaning for every sentence). Key counts = word keys in e2b-v2fix-passive0's training analyses (the writer's data). Sealed: totals only.

## Tatoeba 600 (n = 600)

Reader right 362, writer from gold right 377, pipeline right 289.

| | writer (gold meaning) right | writer wrong | writer accuracy |
|---|---|---|---|
| reader right | 282 | 80 | 0.78 |
| reader wrong | 95 | 143 | 0.40 |

| least-trained word key of the gold meaning | n | writer accuracy | reader accuracy |
|---|---|---|---|
| a key never trained (0) | 194 | 0.20 | 0.25 |
| rarest key 1-24 | 94 | 0.82 | 0.70 |
| all keys >= 25 | 312 | 0.84 | 0.79 |

| meaning length | n | writer accuracy |
|---|---|---|
| <= 4 keys | 467 | 0.66 |
| 5-7 keys | 128 | 0.50 |
| >= 8 keys | 5 | 0.60 |

## generated: in-distribution mix (n = 300)

Reader right 260, writer from gold right 250, pipeline right 217.

| | writer (gold meaning) right | writer wrong | writer accuracy |
|---|---|---|---|
| reader right | 217 | 43 | 0.83 |
| reader wrong | 33 | 7 | 0.82 |

| least-trained word key of the gold meaning | n | writer accuracy | reader accuracy |
|---|---|---|---|
| a key never trained (0) | 1 | 0.00 | 1.00 |
| rarest key 1-24 | 188 | 0.80 | 0.85 |
| all keys >= 25 | 110 | 0.89 | 0.89 |
| no word keys | 1 | 1.00 | 1.00 |

| meaning length | n | writer accuracy |
|---|---|---|
| <= 4 keys | 118 | 0.89 |
| 5-7 keys | 74 | 0.88 |
| >= 8 keys | 108 | 0.74 |

## generated: held-out voice = passive (n = 300)

Reader right 0, writer from gold right 215, pipeline right 1.

| | writer (gold meaning) right | writer wrong | writer accuracy |
|---|---|---|---|
| reader right | 0 | 0 | nan |
| reader wrong | 215 | 85 | 0.72 |

| least-trained word key of the gold meaning | n | writer accuracy | reader accuracy |
|---|---|---|---|
| rarest key 1-24 | 236 | 0.69 | 0.00 |
| all keys >= 25 | 64 | 0.80 | 0.00 |

| meaning length | n | writer accuracy |
|---|---|---|
| <= 4 keys | 62 | 0.79 |
| 5-7 keys | 62 | 0.82 |
| >= 8 keys | 176 | 0.65 |

## sealed (totals only) (n = 147)

Reader right 69, writer from gold right 104, pipeline right 63.

| | writer (gold meaning) right | writer wrong | writer accuracy |
|---|---|---|---|
| reader right | 61 | 8 | 0.88 |
| reader wrong | 43 | 35 | 0.55 |

| least-trained word key of the gold meaning | n | writer accuracy | reader accuracy |
|---|---|---|---|
| a key never trained (0) | 47 | 0.34 | 0.13 |
| rarest key 1-24 | 15 | 0.60 | 0.87 |
| all keys >= 25 | 85 | 0.93 | 0.59 |

| meaning length | n | writer accuracy |
|---|---|---|
| <= 4 keys | 82 | 0.73 |
| 5-7 keys | 54 | 0.80 |
| >= 8 keys | 11 | 0.09 |

