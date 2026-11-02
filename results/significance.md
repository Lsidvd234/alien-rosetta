# Significance of the headline differences (`analysis/significance.py`)

Paired tests on the same sentences: McNemar exact test (only sentences where the two runs disagree count) and a bootstrap 95% interval of the difference. 'significant' = p < 0.05. Seed rows show the size of pure training noise. The sealed set is compared from totals only (unpaired, so conservative).

### Generated test: sentence exact (glyph answer fully right), 300 sentences

| comparison | A | B | A - B | 95% CI | A right, B wrong | B right, A wrong | McNemar p | |
|---|---|---|---|---|---|---|---|---|
| seed noise: baseline seed 1 vs seed 0 | 0.627 | 0.607 | +0.020 | [-0.030, +0.070] | 31 | 25 | 0.5 | noise (n=300) |
| seed noise: real text seed 1 vs seed 0 | 0.610 | 0.587 | +0.023 | [-0.027, +0.073] | 34 | 27 | 0.44 | noise (n=300) |
| claim 4: real text vs baseline (seed 0) | 0.587 | 0.607 | -0.020 | [-0.070, +0.030] | 25 | 31 | 0.5 | noise (n=300) |
| claim 4: real text vs baseline (seed 1) | 0.610 | 0.627 | -0.017 | [-0.067, +0.033] | 26 | 31 | 0.6 | noise (n=300) |
| claim 3: word floor 25 vs 12 (seed 0) | 0.720 | 0.607 | +0.113 | [+0.063, +0.163] | 49 | 15 | 2.4e-05 | significant (n=300) |
| claim 3: word floor 25 vs 12 (vs seed 1) | 0.720 | 0.627 | +0.093 | [+0.040, +0.147] | 50 | 22 | 0.0013 | significant (n=300) |
| 6 epochs vs 4 | 0.660 | 0.607 | +0.053 | [+0.007, +0.100] | 34 | 18 | 0.036 | significant (n=300) |
| claim 6: focal loss vs baseline | 0.567 | 0.607 | -0.040 | [-0.090, +0.010] | 23 | 35 | 0.15 | noise (n=300) |
| claim 6: paraphrases vs baseline | 0.633 | 0.607 | +0.027 | [-0.020, +0.073] | 31 | 23 | 0.34 | noise (n=300) |

### Tatoeba: meaning exact, 600 sentences

| comparison | A | B | A - B | 95% CI | A right, B wrong | B right, A wrong | McNemar p | |
|---|---|---|---|---|---|---|---|---|
| seed noise: baseline seed 1 vs seed 0 | 0.625 | 0.617 | +0.008 | [-0.015, +0.032] | 27 | 22 | 0.57 | noise (n=600) |
| seed noise: real text seed 1 vs seed 0 | 0.928 | 0.923 | +0.005 | [-0.012, +0.022] | 16 | 13 | 0.71 | noise (n=600) |
| claim 4: real text vs baseline (seed 0) | 0.923 | 0.617 | +0.307 | [+0.268, +0.345] | 189 | 5 | 1.8e-49 | significant (n=600) |
| claim 4: real text vs baseline (seed 1) | 0.928 | 0.625 | +0.303 | [+0.265, +0.342] | 186 | 4 | 6.8e-50 | significant (n=600) |
| claim 3: word floor 25 vs 12 (seed 0) | 0.603 | 0.617 | -0.013 | [-0.037, +0.010] | 22 | 30 | 0.33 | noise (n=600) |
| claim 3: word floor 25 vs 12 (vs seed 1) | 0.603 | 0.625 | -0.022 | [-0.047, +0.003] | 22 | 35 | 0.11 | noise (n=600) |
| 6 epochs vs 4 | 0.563 | 0.617 | -0.053 | [-0.080, -0.027] | 19 | 51 | 0.00017 | significant (n=600) |
| claim 6: focal loss vs baseline | 0.588 | 0.617 | -0.028 | [-0.050, -0.008] | 13 | 30 | 0.014 | significant (n=600) |
| claim 6: paraphrases vs baseline | 0.615 | 0.617 | -0.002 | [-0.025, +0.022] | 24 | 25 | 1 | noise (n=600) |

### Sealed set, meaning exact, in scope (totals only; unpaired, conservative)

| comparison | A | B | A - B | 95% CI | |
|---|---|---|---|---|---|
| seed noise: baseline seed 1 vs seed 0 | 0.463 | 0.476 | -0.013 | [-0.127, +0.101] | inside noise |
| seed noise: real text seed 1 vs seed 0 | 0.558 | 0.558 | +0.000 | [-0.114, +0.114] | inside noise |
| claim 4: real text vs baseline (seed 0) | 0.558 | 0.476 | +0.082 | [-0.032, +0.196] | inside noise |
| claim 4: real text vs baseline (seed 1) | 0.558 | 0.463 | +0.095 | [-0.019, +0.209] | inside noise |
| claim 3: word floor 25 vs 12 (seed 0) | 0.503 | 0.476 | +0.027 | [-0.087, +0.141] | inside noise |
| claim 3: word floor 25 vs 12 (vs seed 1) | 0.503 | 0.463 | +0.040 | [-0.074, +0.154] | inside noise |
| 6 epochs vs 4 | 0.449 | 0.476 | -0.027 | [-0.141, +0.087] | inside noise |
| claim 6: focal loss vs baseline | 0.442 | 0.476 | -0.034 | [-0.148, +0.080] | inside noise |
| claim 6: paraphrases vs baseline | 0.449 | 0.476 | -0.027 | [-0.141, +0.087] | inside noise |
| claim 4: real text vs baseline, seed average (worst case: seeds fully correlated) | 0.558 | 0.470 | +0.089 | [-0.025, +0.202] | inside noise |
| claim 3: floor 25 vs baseline seed average (worst case) | 0.503 | 0.470 | +0.033 | [-0.081, +0.148] | inside noise |

