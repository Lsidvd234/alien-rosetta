# Significance of the remaining headline differences (`analysis/significance_more.py`)

Paired where both runs answered the same items and per-item results are saved (McNemar exact test + bootstrap 95% interval); otherwise unpaired (Fisher exact test + normal 95% interval; conservative). 'significant' = p < 0.05. Claims 3, 4 and 6 are in results/significance.md. Sealed comparisons are out of scope here.

### Claim 1 (language): glyph start = own Latin sound (A) vs a deliberately wrong sound (B)

Stage 1 v1, hard grammar, 50k, 3 epochs, seed 0. Teacher-forced whole-sentence exact on 500 items per split (the metric of the report table); an item is right when every target token is right. Same item ids in both runs, so paired.

| comparison | A | B | A - B | 95% CI | test | p | |
|---|---|---|---|---|---|---|---|
| g1-hard-latininit-50k vs s4-hard-shuffledinit-50k, test_iid | 0.990 | 0.982 | +0.008 | [-0.002, +0.020] | paired McNemar (6 A-only right, 2 B-only right) | 0.29 | noise (n=500) |
| g1-hard-latininit-50k vs s4-hard-shuffledinit-50k, test_comp | 0.986 | 0.978 | +0.008 | [-0.004, +0.020] | paired McNemar (7 A-only right, 3 B-only right) | 0.34 | noise (n=500) |
| g1-hard-latininit-50k vs s4-hard-shuffledinit-50k, test_morph | 0.972 | 0.860 | +0.112 | [+0.082, +0.142] | paired McNemar (60 A-only right, 4 B-only right) | 7.4e-14 | significant (n=500) |
| g1-hard-latininit-50k vs s4-hard-shuffledinit-50k, test_long | 0.974 | 0.650 | +0.324 | [+0.280, +0.368] | paired McNemar (168 A-only right, 6 B-only right) | 3.1e-42 | significant (n=500) |

s1-hard-latininit-50k has identical per-item results to g1-hard-latininit-50k: True (same run settings and seed; not an independent replicate).

### Claim 1 (maths): real-digit start (meaning) vs wrong-digit (shuffled) vs neutral (mean)

Exact match, 300 items per test. Only totals are saved (per-item predictions were not stored by these runs), so unpaired; runs of the same seed share their test items, so this is conservative. Seeds are tested separately: the shuffled start differs between seeds far more than binomial noise (different wrong permutations), so pooling seeds would overstate certainty.

| comparison | A | B | A - B | 95% CI | test | p | |
|---|---|---|---|---|---|---|---|
| seed 0, add: 4+4: meaning vs shuffled (maths-meaning vs maths-shuffled) | 0.887 | 0.083 | +0.803 | [+0.756, +0.851] | unpaired Fisher (266/300 vs 25/300) | 3.1e-98 | significant |
| seed 0, add: 4+4: meaning vs mean (maths-meaning vs maths-mean) | 0.887 | 0.020 | +0.867 | [+0.827, +0.906] | unpaired Fisher (266/300 vs 6/300) | 1.6e-121 | significant |
| seed 0, mul: up to 2x2 (trained lengths): meaning vs shuffled (maths-meaning vs maths-shuffled) | 0.957 | 0.690 | +0.267 | [+0.209, +0.324] | unpaired Fisher (287/300 vs 207/300) | 7.3e-19 | significant |
| seed 0, mul: up to 2x2 (trained lengths): meaning vs mean (maths-meaning vs maths-mean) | 0.957 | 0.623 | +0.333 | [+0.274, +0.393] | unpaired Fisher (287/300 vs 187/300) | 1.2e-25 | significant |
| seed 0, mul: 3x1: meaning vs shuffled (maths-meaning vs maths-shuffled) | 0.653 | 0.363 | +0.290 | [+0.213, +0.367] | unpaired Fisher (196/300 vs 109/300) | 1.5e-12 | significant |
| seed 0, mul: 3x1: meaning vs mean (maths-meaning vs maths-mean) | 0.653 | 0.320 | +0.333 | [+0.258, +0.409] | unpaired Fisher (196/300 vs 96/300) | 3.3e-16 | significant |
| seed 0, mul: 3x2: meaning vs shuffled (maths-meaning vs maths-shuffled) | 0.130 | 0.000 | +0.130 | [+0.092, +0.168] | unpaired Fisher (39/300 vs 0/300) | 9.7e-13 | significant |
| seed 0, mul: 3x2: meaning vs mean (maths-meaning vs maths-mean) | 0.130 | 0.020 | +0.110 | [+0.069, +0.151] | unpaired Fisher (39/300 vs 6/300) | 2.1e-07 | significant |
| seed 0, mul: 3x3: meaning vs shuffled (maths-meaning vs maths-shuffled) | 0.027 | 0.000 | +0.027 | [+0.008, +0.045] | unpaired Fisher (8/300 vs 0/300) | 0.0075 | significant |
| seed 0, mul: 3x3: meaning vs mean (maths-meaning vs maths-mean) | 0.027 | 0.000 | +0.027 | [+0.008, +0.045] | unpaired Fisher (8/300 vs 0/300) | 0.0075 | significant |
| seed 1, add: 4+4: meaning vs shuffled (maths-meaning-s1 vs maths-shuffled-s1) | 0.840 | 0.600 | +0.240 | [+0.171, +0.309] | unpaired Fisher (252/300 vs 180/300) | 6.5e-11 | significant |
| seed 1, add: 4+4: meaning vs mean (maths-meaning-s1 vs maths-mean-s1) | 0.840 | 0.153 | +0.687 | [+0.628, +0.745] | unpaired Fisher (252/300 vs 46/300) | 7.9e-69 | significant |
| seed 1, mul: up to 2x2 (trained lengths): meaning vs shuffled (maths-meaning-s1 vs maths-shuffled-s1) | 0.943 | 0.767 | +0.177 | [+0.122, +0.231] | unpaired Fisher (283/300 vs 230/300) | 5.5e-10 | significant |
| seed 1, mul: up to 2x2 (trained lengths): meaning vs mean (maths-meaning-s1 vs maths-mean-s1) | 0.943 | 0.640 | +0.303 | [+0.243, +0.364] | unpaired Fisher (283/300 vs 192/300) | 3.7e-21 | significant |
| seed 1, mul: 3x1: meaning vs shuffled (maths-meaning-s1 vs maths-shuffled-s1) | 0.690 | 0.403 | +0.287 | [+0.210, +0.363] | unpaired Fisher (207/300 vs 121/300) | 2.2e-12 | significant |
| seed 1, mul: 3x1: meaning vs mean (maths-meaning-s1 vs maths-mean-s1) | 0.690 | 0.323 | +0.367 | [+0.292, +0.441] | unpaired Fisher (207/300 vs 97/300) | 2.2e-19 | significant |
| seed 1, mul: 3x2: meaning vs shuffled (maths-meaning-s1 vs maths-shuffled-s1) | 0.163 | 0.033 | +0.130 | [+0.083, +0.177] | unpaired Fisher (49/300 vs 10/300) | 6.8e-08 | significant |
| seed 1, mul: 3x2: meaning vs mean (maths-meaning-s1 vs maths-mean-s1) | 0.163 | 0.007 | +0.157 | [+0.114, +0.200] | unpaired Fisher (49/300 vs 2/300) | 1.6e-13 | significant |
| seed 1, mul: 3x3: meaning vs shuffled (maths-meaning-s1 vs maths-shuffled-s1) | 0.023 | 0.000 | +0.023 | [+0.006, +0.040] | unpaired Fisher (7/300 vs 0/300) | 0.015 | significant |
| seed 1, mul: 3x3: meaning vs mean (maths-meaning-s1 vs maths-mean-s1) | 0.023 | 0.000 | +0.023 | [+0.006, +0.040] | unpaired Fisher (7/300 vs 0/300) | 0.015 | significant |

### Claim 2 (causal): one multiplication fact removed; problems that avoid it (A) vs need it (B)

Trained sizes, real-digit start, 300 items each. Different problems in the two groups, so unpaired. Last rows: does the scratchpad help on problems needing the removed pair (same seed, same items: unpaired is conservative).

| comparison | A | B | A - B | 95% CI | test | p | |
|---|---|---|---|---|---|---|---|
| maths-meaning-drop56: avoids vs needs 5x6 | 0.923 | 0.047 | +0.877 | [+0.838, +0.915] | unpaired Fisher (277/300 vs 14/300) | 1.2e-121 | significant |
| maths-meaning-drop34: avoids vs needs 3x4 | 0.957 | 0.103 | +0.853 | [+0.812, +0.895] | unpaired Fisher (287/300 vs 31/300) | 1.3e-114 | significant |
| maths-meaning-scratch-drop56: avoids vs needs 5x6 | 1.000 | 0.263 | +0.737 | [+0.687, +0.787] | unpaired Fisher (300/300 vs 79/300) | 1.3e-96 | significant |
| needs 5x6: scratchpad vs no scratchpad (maths-meaning-scratch-drop56 vs -drop56) | 0.263 | 0.047 | +0.217 | [+0.161, +0.272] | unpaired Fisher (79/300 vs 14/300) | 5.8e-14 | significant |
| scratch-drop56, needs-5x6 items: digit steps not using vs using the pair | 0.872 | 0.490 | +0.382 | [+0.325, +0.439] | unpaired Fisher (674/773 vs 173/353); steps within the same items | 9.4e-41 | significant |
| maths-meaning-drop22: avoids vs needs 2x2 | | | not tested | | run not in results/kaggle (pending) | | |

### Claim 2 (descriptive): trained structures (A, original test_iid) vs held-out structures (B)

v1 E1 runs at 5 epochs, free generation, glyph output exact. B pools the 5 strict held-out structure groups (passive, object cleft, subject cleft, object topicalised, name as subject; 100 items each, only items that contain the structure). Different sentences, so unpaired. Last row per run: the best single held-out group.

| comparison | A | B | A - B | 95% CI | test | p | |
|---|---|---|---|---|---|---|---|
| e1a-hard-rep-5ep: trained iid vs 5 held-out structures pooled | 0.976 | 0.010 | +0.966 | [+0.950, +0.982] | unpaired Fisher (488/500 vs 5/500) | 9.3e-265 | significant |
| e1a-hard-rep-5ep: trained iid vs best held-out group (held-out order = subject cleft) | 0.976 | 0.020 | +0.956 | [+0.925, +0.987] | unpaired Fisher (488/500 vs 2/100) | 3.8e-96 | significant |
| e1b-hard-rep-cot-5ep: trained iid vs 5 held-out structures pooled | 0.988 | 0.024 | +0.964 | [+0.948, +0.980] | unpaired Fisher (494/500 vs 12/500) | 7.5e-263 | significant |
| e1b-hard-rep-cot-5ep: trained iid vs best held-out group (held-out order = subject cleft) | 0.988 | 0.090 | +0.898 | [+0.841, +0.955] | unpaired Fisher (494/500 vs 9/100) | 4.7e-89 | significant |
| e1b-hard-rep-cot-5ep-seed1: trained iid vs 5 held-out structures pooled | 0.974 | 0.018 | +0.956 | [+0.938, +0.974] | unpaired Fisher (487/500 vs 9/500) | 6.4e-256 | significant |
| e1b-hard-rep-cot-5ep-seed1: trained iid vs best held-out group (held-out order = object cleft) | 0.974 | 0.040 | +0.934 | [+0.893, +0.975] | unpaired Fisher (487/500 vs 4/100) | 5e-91 | significant |

### Claim 5: meaning first (E1b) vs direct (E1a), 3 vs 5 epochs

v1 E1 runs, free generation, glyph output exact (eval_gen, per item saved). Seed 0 runs share their test items (paired); 'seed 1' is a different random language (different items, unpaired). The 3- and 5-epoch runs share the original splits and held-out letter-drop items but not the 5-6 adjectives set (that row falls back to unpaired).

| comparison | A | B | A - B | 95% CI | test | p | |
|---|---|---|---|---|---|---|---|
| E1b vs E1a, 3 epochs: original test_iid (e1b-hard-rep-cot-50k vs e1a-hard-rep-50k) | 0.958 | 0.256 | +0.702 | [+0.660, +0.742] | paired McNemar (353 A-only right, 2 B-only right) | 1.7e-102 | significant (n=500) |
| E1b vs E1a, 3 epochs: original test_long (e1b-hard-rep-cot-50k vs e1a-hard-rep-50k) | 0.838 | 0.060 | +0.778 | [+0.740, +0.816] | paired McNemar (395 A-only right, 6 B-only right) | 2.2e-108 | significant (n=500) |
| E1b vs E1a, 3 epochs: extrapolation: 5-6 adjectives on a noun (e1b-hard-rep-cot-50k vs e1a-hard-rep-50k) | 0.580 | 0.250 | +0.330 | [+0.230, +0.430] | paired McNemar (34 A-only right, 1 B-only right) | 2.1e-09 | significant (n=100) |
| E1b vs E1a, 3 epochs: held-out noise = drop (e1b-hard-rep-cot-50k vs e1a-hard-rep-50k) | 0.510 | 0.160 | +0.350 | [+0.240, +0.460] | paired McNemar (40 A-only right, 5 B-only right) | 7.9e-08 | significant (n=100) |
| E1b vs E1a, 5 epochs: original test_iid (e1b-hard-rep-cot-5ep vs e1a-hard-rep-5ep) | 0.988 | 0.976 | +0.012 | [-0.004, +0.028] | paired McNemar (11 A-only right, 5 B-only right) | 0.21 | noise (n=500) |
| E1b vs E1a, 5 epochs: original test_long (e1b-hard-rep-cot-5ep vs e1a-hard-rep-5ep) | 0.870 | 0.854 | +0.016 | [-0.028, +0.060] | paired McNemar (64 A-only right, 56 B-only right) | 0.52 | noise (n=500) |
| E1b vs E1a, 5 epochs: extrapolation: 5-6 adjectives on a noun (e1b-hard-rep-cot-5ep vs e1a-hard-rep-5ep) | 0.710 | 0.530 | +0.180 | [+0.100, +0.270] | paired McNemar (20 A-only right, 2 B-only right) | 0.00012 | significant (n=100) |
| E1b vs E1a, 5 epochs: held-out noise = drop (e1b-hard-rep-cot-5ep vs e1a-hard-rep-5ep) | 0.540 | 0.420 | +0.120 | [+0.020, +0.220] | paired McNemar (19 A-only right, 7 B-only right) | 0.029 | significant (n=100) |
| E1b seed 1 vs E1a, 5 epochs (other language): original test_iid (e1b-hard-rep-cot-5ep-seed1 vs e1a-hard-rep-5ep) | 0.974 | 0.976 | -0.002 | [-0.021, +0.017] | unpaired Fisher (487/500 vs 488/500); different items | 1 | noise |
| E1b seed 1 vs E1a, 5 epochs (other language): original test_long (e1b-hard-rep-cot-5ep-seed1 vs e1a-hard-rep-5ep) | 0.910 | 0.854 | +0.056 | [+0.016, +0.096] | unpaired Fisher (455/500 vs 427/500); different items | 0.0079 | significant |
| E1b seed 1 vs E1a, 5 epochs (other language): extrapolation: 5-6 adjectives on a noun (e1b-hard-rep-cot-5ep-seed1 vs e1a-hard-rep-5ep) | 0.750 | 0.530 | +0.220 | [+0.090, +0.350] | unpaired Fisher (75/100 vs 53/100); different items | 0.0019 | significant |
| E1b seed 1 vs E1a, 5 epochs (other language): held-out noise = drop (e1b-hard-rep-cot-5ep-seed1 vs e1a-hard-rep-5ep) | 0.570 | 0.420 | +0.150 | [+0.013, +0.287] | unpaired Fisher (57/100 vs 42/100); different items | 0.047 | significant |
| E1a 5 vs 3 epochs: original test_iid (e1a-hard-rep-5ep vs e1a-hard-rep-50k) | 0.976 | 0.256 | +0.720 | [+0.680, +0.758] | paired McNemar (361 A-only right, 1 B-only right) | 7.7e-107 | significant (n=500) |
| E1a 5 vs 3 epochs: original test_long (e1a-hard-rep-5ep vs e1a-hard-rep-50k) | 0.854 | 0.060 | +0.794 | [+0.756, +0.832] | paired McNemar (402 A-only right, 5 B-only right) | 5.6e-112 | significant (n=500) |
| E1a 5 vs 3 epochs: extrapolation: 5-6 adjectives on a noun (e1a-hard-rep-5ep vs e1a-hard-rep-50k) | 0.530 | 0.250 | +0.280 | [+0.150, +0.410] | unpaired Fisher (53/100 vs 25/100); different items | 8e-05 | significant |
| E1a 5 vs 3 epochs: held-out noise = drop (e1a-hard-rep-5ep vs e1a-hard-rep-50k) | 0.420 | 0.160 | +0.260 | [+0.170, +0.360] | paired McNemar (28 A-only right, 2 B-only right) | 8.7e-07 | significant (n=100) |
| E1b 5 vs 3 epochs: original test_iid (e1b-hard-rep-cot-5ep vs e1b-hard-rep-cot-50k) | 0.988 | 0.958 | +0.030 | [+0.012, +0.048] | paired McNemar (18 A-only right, 3 B-only right) | 0.0015 | significant (n=500) |
| E1b 5 vs 3 epochs: original test_long (e1b-hard-rep-cot-5ep vs e1b-hard-rep-cot-50k) | 0.870 | 0.838 | +0.032 | [-0.010, +0.072] | paired McNemar (64 A-only right, 48 B-only right) | 0.16 | noise (n=500) |
| E1b 5 vs 3 epochs: extrapolation: 5-6 adjectives on a noun (e1b-hard-rep-cot-5ep vs e1b-hard-rep-cot-50k) | 0.710 | 0.580 | +0.130 | [-0.001, +0.261] | unpaired Fisher (71/100 vs 58/100); different items | 0.076 | noise |
| E1b 5 vs 3 epochs: held-out noise = drop (e1b-hard-rep-cot-5ep vs e1b-hard-rep-cot-50k) | 0.540 | 0.510 | +0.030 | [-0.050, +0.110] | paired McNemar (10 A-only right, 7 B-only right) | 0.63 | noise (n=100) |

