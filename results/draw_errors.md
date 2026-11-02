# Why each wrong-sound start fails on a different test (`analysis/draw_errors.py`)

Own sound vs three wrong-sound draws (same language, data, seed, test items). Glyphs are named by their REAL sound; 'start' = the sound the run's embedding started from. Teacher-forced token errors unless stated.

## 0. Summary per run (teacher-forced exact; glide = error rate on glyph v in test_morph)

| run | start of v | iid | comp | morph | long | v error (morph) |
|---|---|---|---|---|---|---|
| own | v | 0.990 | 0.986 | 0.972 | 0.974 | 0.000 |
| draw0 | zh | 0.982 | 0.978 | 0.860 | 0.650 | 0.063 |
| p1 | k | 0.988 | 0.994 | 0.984 | 0.162 | 0.000 |
| p2 | r | 0.982 | 0.952 | 0.310 | 0.730 | 0.427 |
| own-vswap | r | 0.986 | 0.992 | 0.976 | 0.514 | 0.003 |
| p2-vfix | v | 0.994 | 0.992 | 0.976 | 0.984 | 0.004 |
| p1-ts1 | k | 0.990 | 0.992 | 0.978 | 0.978 | 0.005 |
| p1-ts2 | k | 0.996 | 0.990 | 0.976 | 0.976 | 0.000 |
| own-ts1 | v | 0.990 | 0.992 | 0.974 | 0.964 | 0.001 |
| own-ts2 | v | 0.990 | 0.978 | 0.882 | 0.960 | 0.000 |

## 1. Error rate by position in the word (teacher-forced, per glyph token)

0 = last glyph of a word (where case/tense endings sit), 3 = fourth from the end or earlier.

**test_morph**

| run | last | 2nd last | 3rd last | earlier |
|---|---|---|---|---|
| own | 0.000 | 0.000 | 0.000 | 0.001 |
| draw0 | 0.000 | 0.028 | 0.001 | 0.002 |
| p1 | 0.000 | 0.000 | 0.000 | 0.001 |
| p2 | 0.003 | 0.191 | 0.000 | 0.001 |
| own-vswap | 0.000 | 0.000 | 0.000 | 0.001 |
| p2-vfix | 0.000 | 0.002 | 0.000 | 0.001 |
| p1-ts1 | 0.000 | 0.003 | 0.000 | 0.001 |
| p1-ts2 | 0.001 | 0.000 | 0.000 | 0.001 |
| own-ts1 | 0.000 | 0.000 | 0.001 | 0.001 |
| own-ts2 | 0.000 | 0.001 | 0.024 | 0.002 |

**test_long**

| run | last | 2nd last | 3rd last | earlier |
|---|---|---|---|---|
| own | 0.000 | 0.001 | 0.001 | 0.001 |
| draw0 | 0.006 | 0.008 | 0.018 | 0.012 |
| p1 | 0.067 | 0.100 | 0.101 | 0.094 |
| p2 | 0.002 | 0.009 | 0.007 | 0.014 |
| own-vswap | 0.021 | 0.030 | 0.041 | 0.040 |
| p2-vfix | 0.001 | 0.000 | 0.001 | 0.001 |
| p1-ts1 | 0.000 | 0.000 | 0.000 | 0.001 |
| p1-ts2 | 0.000 | 0.000 | 0.001 | 0.001 |
| own-ts1 | 0.000 | 0.001 | 0.001 | 0.001 |
| own-ts2 | 0.001 | 0.000 | 0.000 | 0.001 |

## 2. Where in a long sentence the first error falls (test_long)

Error rate by token index (10-token bins), and the first error's relative position in failed items.

| run | 0-9 | 10-19 | 20-29 | 30-39 | 40-49 | 50-59 | 60-69 | 70-79 | 80-89 | 90+ | failed items | median first error at |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| own | 0.000 | 0.001 | 0.001 | 0.001 | 0.001 | 0.000 | 0.000 | - | - | - | 13 | 0.49 |
| draw0 | 0.000 | 0.004 | 0.046 | 0.010 | 0.002 | 0.001 | 0.000 | - | - | - | 175 | 0.47 |
| p1 | 0.000 | 0.068 | 0.322 | 0.101 | 0.001 | 0.001 | 0.000 | - | - | - | 419 | 0.38 |
| p2 | 0.001 | 0.021 | 0.025 | 0.003 | 0.001 | 0.006 | 0.000 | - | - | - | 135 | 0.35 |
| own-vswap | 0.001 | 0.025 | 0.126 | 0.046 | 0.001 | 0.000 | 0.000 | - | - | - | 243 | 0.45 |
| p2-vfix | 0.000 | 0.001 | 0.000 | 0.000 | 0.001 | 0.001 | 0.000 | - | - | - | 8 | 0.63 |
| p1-ts1 | 0.000 | 0.001 | 0.000 | 0.000 | 0.001 | 0.001 | 0.000 | - | - | - | 11 | 0.45 |
| p1-ts2 | 0.001 | 0.001 | 0.000 | 0.000 | 0.001 | 0.001 | 0.000 | - | - | - | 12 | 0.40 |
| own-ts1 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.000 | - | - | - | 18 | 0.50 |
| own-ts2 | 0.000 | 0.000 | 0.002 | 0.002 | 0.001 | 0.001 | 0.000 | - | - | - | 20 | 0.56 |

## 3. Error rate per glyph (teacher-forced)

Glyphs with an error rate of at least 0.02 in some wrong draw on test_morph or test_long; the start sound each draw gave that glyph in brackets.

**test_morph**

| glyph (real sound) | uses | own | draw0 | p1 | p2 | own-vswap | p2-vfix | p1-ts1 | p1-ts2 | own-ts1 | own-ts2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v | 779 | 0.000 | 0.063 [zh] | 0.000 [k] | 0.427 [r] | 0.003 [r] | 0.004 [v] | 0.005 [k] | 0.000 [k] | 0.001 [v] | 0.000 [v] |
| uu | 366 | 0.000 | 0.003 [ee] | 0.000 [tx] | 0.000 [y] | 0.000 [uu] | 0.000 [y] | 0.000 [tx] | 0.000 [tx] | 0.000 [uu] | 0.115 [uu] |

**test_long**

| glyph (real sound) | uses | own | draw0 | p1 | p2 | own-vswap | p2-vfix | p1-ts1 | p1-ts2 | own-ts1 | own-ts2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r | 590 | 0.000 | 0.012 [uu] | 0.163 [e] | 0.022 [n] | 0.053 [v] | 0.002 [n] | 0.002 [e] | 0.000 [e] | 0.000 [r] | 0.000 [r] |
| c | 447 | 0.004 | 0.018 [k] | 0.150 [r] | 0.016 [l] | 0.078 [c] | 0.000 [l] | 0.000 [r] | 0.000 [r] | 0.000 [c] | 0.000 [c] |
| k | 447 | 0.000 | 0.011 [oo] | 0.148 [ii] | 0.013 [zh] | 0.063 [k] | 0.002 [zh] | 0.000 [ii] | 0.000 [ii] | 0.002 [k] | 0.002 [k] |
| uu | 457 | 0.000 | 0.022 [ee] | 0.147 [tx] | 0.013 [y] | 0.057 [uu] | 0.000 [y] | 0.000 [tx] | 0.000 [tx] | 0.002 [uu] | 0.002 [uu] |
| n | 596 | 0.002 | 0.017 [m] | 0.146 [uu] | 0.017 [i] | 0.057 [n] | 0.000 [i] | 0.000 [uu] | 0.000 [uu] | 0.000 [n] | 0.000 [n] |
| p | 434 | 0.002 | 0.007 [r] | 0.141 [o] | 0.014 [ng] | 0.071 [p] | 0.000 [ng] | 0.000 [o] | 0.000 [o] | 0.002 [p] | 0.002 [p] |
| zh | 399 | 0.000 | 0.015 [i] | 0.138 [n] | 0.003 [tx] | 0.073 [zh] | 0.000 [tx] | 0.000 [n] | 0.003 [n] | 0.000 [zh] | 0.000 [zh] |
| rx | 462 | 0.000 | 0.011 [v] | 0.134 [t] | 0.006 [e] | 0.050 [rx] | 0.000 [e] | 0.000 [t] | 0.000 [t] | 0.000 [rx] | 0.000 [rx] |
| ny | 503 | 0.000 | 0.020 [o] | 0.133 [y] | 0.008 [t] | 0.060 [ny] | 0.000 [t] | 0.000 [y] | 0.000 [y] | 0.000 [ny] | 0.000 [ny] |
| ee | 446 | 0.000 | 0.038 [l] | 0.121 [p] | 0.016 [o] | 0.049 [ee] | 0.000 [o] | 0.000 [p] | 0.000 [p] | 0.000 [ee] | 0.000 [ee] |
| t | 474 | 0.002 | 0.011 [ny] | 0.120 [nx] | 0.015 [u] | 0.053 [t] | 0.002 [u] | 0.006 [nx] | 0.002 [nx] | 0.002 [t] | 0.004 [t] |
| tx | 972 | 0.002 | 0.003 [c] | 0.113 [ee] | 0.012 [uu] | 0.042 [tx] | 0.000 [uu] | 0.000 [ee] | 0.000 [ee] | 0.001 [tx] | 0.003 [tx] |
| ng | 622 | 0.005 | 0.016 [lx] | 0.113 [oo] | 0.011 [c] | 0.051 [ng] | 0.003 [c] | 0.002 [oo] | 0.002 [oo] | 0.003 [ng] | 0.002 [ng] |
| y | 996 | 0.000 | 0.011 [nx] | 0.105 [m] | 0.015 [aa] | 0.028 [y] | 0.000 [aa] | 0.000 [m] | 0.001 [m] | 0.002 [y] | 0.000 [y] |
| u | 1028 | 0.003 | 0.021 [p] | 0.104 [lx] | 0.005 [lx] | 0.043 [u] | 0.002 [lx] | 0.001 [lx] | 0.001 [lx] | 0.002 [u] | 0.000 [u] |
| lx | 998 | 0.001 | 0.006 [u] | 0.095 [aa] | 0.011 [v] | 0.031 [lx] | 0.000 [r] | 0.000 [aa] | 0.000 [aa] | 0.001 [lx] | 0.002 [lx] |
| oo | 793 | 0.001 | 0.019 [e] | 0.095 [i] | 0.006 [ee] | 0.029 [oo] | 0.001 [ee] | 0.000 [i] | 0.003 [i] | 0.000 [oo] | 0.001 [oo] |
| v | 1161 | 0.001 | 0.006 [zh] | 0.084 [k] | 0.032 [r] | 0.031 [r] | 0.000 [v] | 0.000 [k] | 0.001 [k] | 0.001 [v] | 0.001 [v] |
| nx | 1007 | 0.000 | 0.009 [tx] | 0.083 [ng] | 0.010 [k] | 0.039 [nx] | 0.001 [k] | 0.000 [ng] | 0.000 [ng] | 0.001 [nx] | 0.000 [nx] |
| o | 1422 | 0.000 | 0.017 [ii] | 0.079 [zh] | 0.012 [m] | 0.031 [o] | 0.000 [m] | 0.000 [zh] | 0.000 [zh] | 0.000 [o] | 0.001 [o] |
| aa | 673 | 0.001 | 0.019 [rx] | 0.077 [ny] | 0.012 [p] | 0.037 [aa] | 0.000 [p] | 0.000 [ny] | 0.000 [ny] | 0.000 [aa] | 0.000 [aa] |
| i | 2264 | 0.000 | 0.008 [n] | 0.077 [u] | 0.011 [oo] | 0.031 [i] | 0.000 [oo] | 0.000 [u] | 0.000 [u] | 0.000 [i] | 0.000 [i] |
| virama | 680 | 0.000 | 0.004 [-] | 0.076 [-] | 0.001 [-] | 0.034 [-] | 0.000 [-] | 0.000 [-] | 0.000 [-] | 0.000 [-] | 0.004 [-] |
| l | 983 | 0.001 | 0.007 [y] | 0.067 [rx] | 0.012 [ny] | 0.030 [l] | 0.002 [ny] | 0.004 [rx] | 0.004 [rx] | 0.004 [l] | 0.002 [l] |
| ii | 930 | 0.000 | 0.009 [aa] | 0.063 [c] | 0.011 [nx] | 0.020 [ii] | 0.000 [nx] | 0.000 [c] | 0.000 [c] | 0.002 [ii] | 0.001 [ii] |
| m | 1383 | 0.001 | 0.009 [ng] | 0.062 [v] | 0.004 [ii] | 0.025 [m] | 0.001 [ii] | 0.001 [v] | 0.001 [v] | 0.001 [m] | 0.001 [m] |
| e | 2650 | 0.000 | 0.008 [t] | 0.044 [l] | 0.004 [rx] | 0.017 [e] | 0.000 [rx] | 0.000 [l] | 0.000 [l] | 0.000 [e] | 0.001 [e] |

## 4. Which glyph is written instead of which (generation, 100 items per test)

Character substitutions from aligning gold and predicted text (difflib); the most frequent per run.

**test_morph**

| run | top substitutions (gold -> written, count) | top dropped glyphs |
|---|---|---|
| own | - | - |
| draw0 | - | - |
| p1 | - | - |
| p2 | - | v 2, i 2 |
| own-vswap | - | - |
| p2-vfix | - | - |
| p1-ts1 | - | - |
| p1-ts2 | - | - |
| own-ts1 | - | - |
| own-ts2 | - | - |

**test_long**

| run | top substitutions (gold -> written, count) | top dropped glyphs |
|---|---|---|
| own | - | - |
| draw0 | - | - |
| p1 | - | ' ' 4, u 3, v 2, r 2 |
| p2 | virama->e 1 | - |
| own-vswap | p->t 1, o->ii 1, y->rx 1, uu->e 1, lx->nx 1, lx->ny 1 | r 1 |
| p2-vfix | - | - |
| p1-ts1 | - | - |
| p1-ts2 | - | - |
| own-ts1 | - | - |
| own-ts2 | - | - |

## 5. Start and final embeddings of the glyphs

Cosine similarity between glyph embeddings. 'start' = delta_init, 'final' = trained (saved glyph_tokens.pt). For each run: the most similar pairs of glyphs at the start and at the end.

- own: no glyph_tokens.pt
- draw0, start: n~i 0.45, n~u 0.42, ny~oo 0.42, tx~u 0.41, r~lx 0.41
- draw0, final: n~i 0.40, e~u 0.39, n~u 0.39, ny~oo 0.38, e~i 0.38
- p1, start: y~zh 0.45, y~ee 0.42, p~r 0.42, ee~ii 0.41, n~i 0.41
- p1, final: y~zh 0.40, ee~ii 0.39, p~r 0.38, n~i 0.38, zh~rx 0.37
- p2, start: r~o 0.45, o~aa 0.42, rx~ee 0.42, ng~aa 0.41, tx~t 0.41
- p2, final: o~aa 0.41, r~o 0.40, ng~ny 0.38, tx~t 0.38, ng~aa 0.38
  - test_long confusions: virama->e (1): start 0.21, final 0.27
- own-vswap, start: n~m 0.45, p~m 0.42, e~o 0.42, c~p 0.41, u~uu 0.41
- own-vswap, final: n~m 0.43, e~o 0.41, p~m 0.40, u~uu 0.39, l~v 0.39
  - test_long confusions: p->t (1): start 0.38, final 0.37; o->ii (1): start 0.09, final 0.13; y->rx (1): start 0.03, final 0.09; uu->e (1): start 0.05, final 0.10
- p2-vfix, start: r~o 0.45, o~aa 0.42, rx~ee 0.42, ng~aa 0.41, tx~t 0.41
- p2-vfix, final: r~o 0.41, o~aa 0.40, tx~t 0.38, r~lx 0.38, ng~aa 0.38
- p1-ts1, start: y~zh 0.45, y~ee 0.42, p~r 0.42, ee~ii 0.41, n~i 0.41
- p1-ts1, final: y~zh 0.40, n~i 0.39, ee~ii 0.38, p~r 0.38, c~e 0.36
- p1-ts2, start: y~zh 0.45, y~ee 0.42, p~r 0.42, ee~ii 0.41, n~i 0.41
- p1-ts2, final: y~zh 0.41, n~i 0.38, ee~ii 0.38, p~r 0.38, zh~rx 0.37
- own-ts1, start: n~m 0.45, p~m 0.42, e~o 0.42, c~p 0.41, u~uu 0.41
- own-ts1, final: n~m 0.43, e~o 0.42, p~m 0.40, r~l 0.39, n~r 0.39
- own-ts2, start: n~m 0.45, p~m 0.42, e~o 0.42, c~p 0.41, u~uu 0.41
- own-ts2, final: n~m 0.43, e~o 0.42, p~m 0.40, u~uu 0.39, r~l 0.38

