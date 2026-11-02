# Glyph embeddings: start vs after training, and across runs (`analysis/glyph_embedding_drift.py`)

27 glyph token embeddings (1024-d), centred on the base model's mean token embedding. Same model, language, data and setting in every run; only the start and the training seed differ.

## 1. How much each run's glyphs moved

cos(start, final): 1 = did not turn at all, 0 = ended unrelated to where it started. Change = |final - start| / |start|. Mean over the 27 glyphs (min-max in brackets).

| run | start | cos(start, final) | change |
|---|---|---|---|
| own | own sound | 0.953 (0.39-0.99) | 0.377 (0.19-3.51) |
| own-ts1 | own sound | 0.954 (0.40-0.98) | 0.378 (0.20-3.54) |
| own-ts2 | own sound | 0.953 (0.38-0.99) | 0.377 (0.19-3.53) |
| own-vswap | own, v<->r | 0.953 (0.39-0.99) | 0.377 (0.19-3.52) |
| draw0 | wrong sound | 0.951 (0.39-0.98) | 0.393 (0.21-3.64) |
| p1 | wrong sound | 0.950 (0.38-0.98) | 0.401 (0.24-3.67) |
| p1-ts1 | wrong sound | 0.951 (0.39-0.98) | 0.398 (0.24-3.71) |
| p1-ts2 | wrong sound | 0.951 (0.40-0.98) | 0.399 (0.24-3.70) |
| p2 | wrong sound | 0.953 (0.39-0.98) | 0.385 (0.22-3.67) |
| p2-vfix | wrong, v fixed | 0.953 (0.40-0.98) | 0.389 (0.22-3.71) |
| copy-tr | copied token | 0.941 (0.91-0.97) | 0.389 (0.26-0.50) |
| copy-pl | copied token | 0.964 (0.95-0.98) | 0.293 (0.21-0.37) |
| random | random | 0.975 (0.96-0.99) | 0.239 (0.17-0.29) |
| randtok | copied token | 0.971 (0.96-0.99) | 0.263 (0.18-0.34) |

## 2. Same place after training?

Mean over glyphs of cos(glyph's final vector in run A, same glyph's final vector in run B). 1 = identical; runs that started at the same place sit high just because they started there, so compare with the start matrix (in the PNG).

| | own | own-ts1 | own-ts2 | own-vswap | draw0 | p1 | p1-ts1 | p1-ts2 | p2 | p2-vfix | copy-tr | copy-pl | random | randtok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| own | 1.00 | 0.98 | 0.98 | 0.93 | 0.26 | 0.23 | 0.23 | 0.23 | 0.23 | 0.25 | 0.07 | 0.03 | 0.01 | 0.03 |
| own-ts1 | 0.98 | 1.00 | 0.98 | 0.93 | 0.26 | 0.23 | 0.23 | 0.23 | 0.23 | 0.25 | 0.07 | 0.03 | 0.01 | 0.03 |
| own-ts2 | 0.98 | 0.98 | 1.00 | 0.93 | 0.26 | 0.23 | 0.23 | 0.23 | 0.23 | 0.25 | 0.07 | 0.03 | 0.01 | 0.03 |
| own-vswap | 0.93 | 0.93 | 0.93 | 1.00 | 0.25 | 0.23 | 0.23 | 0.23 | 0.25 | 0.23 | 0.07 | 0.04 | 0.01 | 0.03 |
| draw0 | 0.26 | 0.26 | 0.26 | 0.25 | 1.00 | 0.26 | 0.26 | 0.26 | 0.23 | 0.23 | 0.03 | 0.02 | 0.02 | 0.02 |
| p1 | 0.23 | 0.23 | 0.23 | 0.23 | 0.26 | 1.00 | 0.97 | 0.97 | 0.26 | 0.26 | 0.04 | 0.03 | 0.02 | 0.03 |
| p1-ts1 | 0.23 | 0.23 | 0.23 | 0.23 | 0.26 | 0.97 | 1.00 | 0.97 | 0.26 | 0.26 | 0.04 | 0.04 | 0.02 | 0.03 |
| p1-ts2 | 0.23 | 0.23 | 0.23 | 0.23 | 0.26 | 0.97 | 0.97 | 1.00 | 0.26 | 0.26 | 0.04 | 0.04 | 0.02 | 0.03 |
| p2 | 0.23 | 0.23 | 0.23 | 0.25 | 0.23 | 0.26 | 0.26 | 0.26 | 1.00 | 0.93 | 0.03 | 0.03 | 0.03 | 0.03 |
| p2-vfix | 0.25 | 0.25 | 0.25 | 0.23 | 0.23 | 0.26 | 0.26 | 0.26 | 0.93 | 1.00 | 0.03 | 0.03 | 0.03 | 0.03 |
| copy-tr | 0.07 | 0.07 | 0.07 | 0.07 | 0.03 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 1.00 | 0.94 | 0.00 | 0.95 |
| copy-pl | 0.03 | 0.03 | 0.03 | 0.04 | 0.02 | 0.03 | 0.04 | 0.04 | 0.03 | 0.03 | 0.94 | 1.00 | 0.00 | 0.97 |
| random | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.03 | 0.00 | 0.00 | 1.00 | 0.00 |
| randtok | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.95 | 0.97 | 0.00 | 1.00 |

## 3. Same shape after training?

Correlation between two runs' 27x27 glyph-to-glyph similarity matrices: 1 = the glyphs are arranged the same way relative to each other, wherever the cloud sits.

| | own | own-ts1 | own-ts2 | own-vswap | draw0 | p1 | p1-ts1 | p1-ts2 | p2 | p2-vfix | copy-tr | copy-pl | random | randtok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| own | 1.00 | 1.00 | 1.00 | 0.98 | 0.02 | 0.07 | 0.05 | 0.05 | 0.06 | 0.03 | 0.31 | 0.18 | 0.23 | 0.17 |
| own-ts1 | 1.00 | 1.00 | 1.00 | 0.98 | 0.02 | 0.07 | 0.06 | 0.06 | 0.06 | 0.03 | 0.31 | 0.18 | 0.24 | 0.18 |
| own-ts2 | 1.00 | 1.00 | 1.00 | 0.98 | 0.02 | 0.07 | 0.06 | 0.06 | 0.06 | 0.03 | 0.30 | 0.17 | 0.23 | 0.17 |
| own-vswap | 0.98 | 0.98 | 0.98 | 1.00 | 0.02 | 0.07 | 0.05 | 0.05 | 0.06 | 0.03 | 0.30 | 0.18 | 0.25 | 0.17 |
| draw0 | 0.02 | 0.02 | 0.02 | 0.02 | 1.00 | 0.14 | 0.14 | 0.14 | -0.12 | -0.10 | 0.05 | 0.03 | 0.14 | 0.03 |
| p1 | 0.07 | 0.07 | 0.07 | 0.07 | 0.14 | 1.00 | 0.99 | 1.00 | -0.02 | -0.03 | 0.07 | 0.05 | 0.15 | 0.06 |
| p1-ts1 | 0.05 | 0.06 | 0.06 | 0.05 | 0.14 | 0.99 | 1.00 | 1.00 | -0.03 | -0.03 | 0.06 | 0.05 | 0.15 | 0.05 |
| p1-ts2 | 0.05 | 0.06 | 0.06 | 0.05 | 0.14 | 1.00 | 1.00 | 1.00 | -0.03 | -0.03 | 0.07 | 0.04 | 0.16 | 0.05 |
| p2 | 0.06 | 0.06 | 0.06 | 0.06 | -0.12 | -0.02 | -0.03 | -0.03 | 1.00 | 0.98 | 0.10 | 0.06 | 0.12 | 0.09 |
| p2-vfix | 0.03 | 0.03 | 0.03 | 0.03 | -0.10 | -0.03 | -0.03 | -0.03 | 0.98 | 1.00 | 0.09 | 0.06 | 0.11 | 0.09 |
| copy-tr | 0.31 | 0.31 | 0.30 | 0.30 | 0.05 | 0.07 | 0.06 | 0.07 | 0.10 | 0.09 | 1.00 | 0.91 | 0.33 | 0.92 |
| copy-pl | 0.18 | 0.18 | 0.17 | 0.18 | 0.03 | 0.05 | 0.05 | 0.04 | 0.06 | 0.06 | 0.91 | 1.00 | 0.21 | 0.95 |
| random | 0.23 | 0.24 | 0.23 | 0.25 | 0.14 | 0.15 | 0.15 | 0.16 | 0.12 | 0.11 | 0.33 | 0.21 | 1.00 | 0.24 |
| randtok | 0.17 | 0.18 | 0.17 | 0.17 | 0.03 | 0.06 | 0.05 | 0.05 | 0.09 | 0.09 | 0.92 | 0.95 | 0.24 | 1.00 |

## Summary: average over run pairs (start -> after training)

| pair of runs | n pairs | same place: start -> final | same shape: start -> final |
|---|---|---|---|
| own sound vs own sound | 3 | 1.00 -> 0.98 | 1.00 -> 1.00 |
| own sound vs wrong sound | 15 | 0.23 -> 0.24 | 0.10 -> 0.05 |
| own sound vs copied token | 9 | 0.01 -> 0.05 | -0.04 -> 0.22 |
| own sound vs random | 3 | -0.01 -> 0.01 | 0.02 -> 0.23 |
| wrong sound vs wrong sound | 10 | 0.47 -> 0.47 | 0.39 -> 0.32 |
| wrong sound vs copied token | 15 | 0.00 -> 0.03 | -0.03 -> 0.06 |
| wrong sound vs random | 5 | 0.01 -> 0.02 | 0.03 -> 0.14 |
| copied token vs copied token | 3 | 1.00 -> 0.95 | 1.00 -> 0.93 |
| copied token vs random | 3 | -0.00 -> 0.00 | -0.04 -> 0.26 |

## 4. Wrong-sound runs: does a glyph drift toward its REAL sound?

Per glyph: cos(final, the start it was given) vs cos(final, its own sound's start (= the own run's start)). Glyphs whose given start already equals the own start are left out.

| run | glyphs | cos to given (wrong) start | cos to own-sound start | glyphs nearer own sound |
|---|---|---|---|---|
| own-vswap | 2 | 0.975 | 0.384 | 0 / 2 |
| draw0 | 26 | 0.972 | 0.223 | 0 / 26 |
| p1 | 26 | 0.972 | 0.196 | 0 / 26 |
| p1-ts1 | 26 | 0.973 | 0.199 | 0 / 26 |
| p1-ts2 | 26 | 0.972 | 0.194 | 0 / 26 |
| p2 | 26 | 0.974 | 0.198 | 0 / 26 |
| p2-vfix | 25 | 0.974 | 0.189 | 0 / 25 |
| copy-tr | 27 | 0.941 | 0.065 | 0 / 27 |
| copy-pl | 27 | 0.964 | 0.029 | 0 / 27 |
| random | 27 | 0.975 | 0.006 | 0 / 27 |
| randtok | 27 | 0.971 | 0.027 | 0 / 27 |

## Pictures

- `results/glyph_embeddings/1_movement_with_random_randtok.png`: cos(start, final), run x glyph.
- `results/glyph_embeddings/2_same_place_same_shape_with_random_randtok.png`: the four run x run matrices.
- `results/glyph_embeddings/3_start_to_final_pca_with_random_randtok.png`: start -> final arrows for 6 glyphs, all runs.

