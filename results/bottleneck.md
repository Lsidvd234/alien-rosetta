# Reader or writer? Two-model pipeline vs one model (`analysis/bottleneck.py`)

reader = e2b-v2fix-passive0-meaning (English -> meaning), writer = e2b-v2fix-passive0-writer (meaning -> glyphs), one model = e2b-v2fix-passive0. Exact-match shares; the sealed row is totals only.

| test | n | reading | writing (gold meaning) | pipeline | one model: glyphs | one model: meaning | pipeline failures: reader wrong / writer wrong |
|---|---|---|---|---|---|---|---|
| Tatoeba 600 | 600 | 0.603 | 0.628 | 0.482 | 0.478 | 0.603 | 231 / 80 |
| generated: in-distribution mix | 300 | 0.867 | 0.833 | 0.723 | 0.613 | 0.823 | 40 / 43 |
| generated: held-out voice = passive | 300 | 0.000 | 0.717 | 0.003 | 0.003 | 0.000 | 299 / 0 |
| sealed (totals only) | 147 | 0.469 | 0.707 | 0.429 | 0.449 | 0.476 | 76 / 8 |
