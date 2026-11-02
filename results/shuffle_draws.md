# Claim 1 (language): own sound vs other wrong-sound draws (31 Oct 2026)

Same language, data, seed and settings; only the wrong-sound permutation differs (`--shuffle-seed`; draw 1 = the
original s4 run, shuffle seed = seed 0). Teacher-forced whole-sentence exact on 500 items per split (generation exact
on 100 after the slash). Paired McNemar on the same items (code: `analysis/significance_more.py` helpers).
p1 added 31 Oct 13:10 (shuffle seed 1).

| split | own sound (g1) | wrong sound, draw 1 (s4) | wrong sound, draw p1 | wrong sound, draw p2 |
|---|---|---|---|---|
| test_iid | 0.990 / 1.00 | 0.982 / 0.99 | 0.988 / 0.98 | 0.982 / 1.00 |
| test_comp | 0.986 / 0.98 | 0.978 / 0.97 | 0.994 / 0.98 | 0.952 / 0.97 |
| test_morph | 0.972 / 0.98 | 0.860 / 0.83 | 0.984 / 0.99 | 0.310 / 0.34 |
| test_long | 0.974 / 0.99 | 0.650 / 0.74 | 0.162 / 0.14 | 0.730 / 0.76 |

| comparison | split | A | B | A-only / B-only right | p |
|---|---|---|---|---|---|
| own vs p2 | test_iid | 0.990 | 0.982 | 5 / 1 | 0.22 |
| own vs p2 | test_comp | 0.986 | 0.952 | 22 / 5 | 0.0015 |
| own vs p2 | test_morph | 0.972 | 0.310 | 334 / 3 | 4.6e-95 |
| own vs p2 | test_long | 0.974 | 0.730 | 127 / 5 | 1.2e-31 |
| draw 1 vs p2 | test_iid | 0.982 | 0.982 | 4 / 4 | 1 |
| draw 1 vs p2 | test_comp | 0.978 | 0.952 | 19 / 6 | 0.015 |
| draw 1 vs p2 | test_morph | 0.860 | 0.310 | 288 / 13 | 1.1e-68 |
| draw 1 vs p2 | test_long | 0.650 | 0.730 | 78 / 118 | 0.0052 |
| own vs p1 | test_iid | 0.990 | 0.988 | 3 / 2 | 1 |
| own vs p1 | test_comp | 0.986 | 0.994 | 0 / 4 | 0.12 |
| own vs p1 | test_morph | 0.972 | 0.984 | 3 / 9 | 0.15 |
| own vs p1 | test_long | 0.974 | 0.162 | 406 / 0 | 1.2e-122 |
| draw 1 vs p1 | test_morph | 0.860 | 0.984 | 2 / 64 | 6e-17 |
| draw 1 vs p1 | test_long | 0.650 | 0.162 | 263 / 19 | 4.4e-56 |
| p2 vs p1 | test_morph | 0.310 | 0.984 | 1 / 338 | 6.1e-100 |
| p2 vs p1 | test_long | 0.730 | 0.162 | 291 / 7 | 1.6e-76 |

Reading (three wrong draws): every wrong-sound draw fails badly on at least one generalisation test, but a
different one each time: draw 1 is weak on both (morphology 0.86, long 0.65), p2 collapses on morphology (0.31),
p1 matches the own-sound start on morphology (0.984, not significantly different) and collapses on long sentences
(0.162; 406 vs 0 items). The own-sound start is the only one of the four good on all four tests. All four reach a
near-zero training loss. So claim 1's language result holds across draws as "the wrong-sound start is unreliable
out of distribution", not as "the wrong sound always costs X on test T": the cost on any single test ranges from
none (p1, morphology) to -0.81 (p1, long sentences).
