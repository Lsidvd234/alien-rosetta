# Follow-ups 5-7 (laptop)

## 5. English forgetting

WikiText-2 test: 40 chunks x 256 tokens (never trained on). Everyday: the 30 sentences of test 21. 'Masked' = glyph tokens removed from the prediction. Leak = plain-English continuations that contain glyphs.

| model | WikiText-2 ppl | masked | everyday ppl | masked | probability on glyphs | leak |
|---|---|---|---|---|---|---|
| Base Qwen3-0.6B-Base | 22.9 | 22.9 | 20.3 | 20.3 | 0.000 | 0% |
| Option 1 | 4207.6 | 521.9 | 372.0 | 246.7 | 0.195 | 47% |
| R1 | 35.6 | 34.9 | 22.7 | 22.7 | 0.000 | 0% |
| E1a (plain) | 34.3 | 33.6 | 23.5 | 23.5 | 0.000 | 0% |
| E1b (analysis) | 35.5 | 34.9 | 23.0 | 23.0 | 0.000 | 0% |

## 6. E1b with the LoRA adapter scaled at inference

| measure | strength 0.25 | strength 0.5 | strength 0.75 | strength 1.0 |
|---|---|---|---|---|
| WikiText-2 ppl (20 chunks) | 24.6 | 26.2 | 29.6 | 36.0 |
| everyday ppl | 18.9 | 19.0 | 20.4 | 23.0 |
| in-distribution mix | 0.00 | 0.00 | 0.08 | 0.70 |
| held-out voice = passive | 0.00 | 0.00 | 0.04 | 0.14 |
| held-out synonym pairs | 0.00 | 0.00 | 0.02 | 0.06 |
| R1 rewrite synonym [train] | 0.00 | 0.00 | 0.00 | 0.36 |
| R1 rewrite typo [train] | 0.00 | 0.00 | 0.02 | 0.52 |
| original test_iid (first 200) | 0.00 | 0.00 | 0.02 | 0.94 |

## 7. Per-epoch checkpoints (exact match of the glyph output)

Subsets for laptop time: first 100 items of each original split (label says 200: name kept for the lookup), first 30 of every other group, identical for every checkpoint (about +/-9 points per cell).

| group | R1 ep1 | R1 ep2 | R1 ep3 | E1a (plain) ep1 | E1a (plain) ep2 | E1a (plain) ep3 | E1b (analysis) ep1 | E1b (analysis) ep2 | E1b (analysis) ep3 |
|---|---|---|---|---|---|---|---|---|---|
| in-distribution mix | 0.03 | 0.37 | 0.33 | 0.07 | 0.20 | 0.33 | 0.10 | 0.30 | 0.77 |
| held-out case = sentence | 0.13 | 0.20 | 0.30 | 0.07 | 0.27 | 0.33 | 0.07 | 0.30 | 0.87 |
| held-out noise = drop | 0.00 | 0.07 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.07 | 0.43 |
| held-out order = object topicalised | 0.00 | 0.00 | 0.00 | 0.00 | 0.07 | 0.07 | 0.07 | 0.07 | 0.13 |
| held-out voice = passive | 0.00 | 0.13 | 0.13 | 0.03 | 0.03 | 0.03 | 0.07 | 0.07 | 0.07 |
| held-out synonym pairs | 0.03 | 0.23 | 0.37 | 0.00 | 0.03 | 0.07 | 0.03 | 0.00 | 0.07 |
| extrapolation: 5-6 adjectives on a noun | 0.13 | 0.17 | 0.17 | 0.13 | 0.20 | 0.23 | 0.17 | 0.27 | 0.60 |
| extrapolation: 5 clauses | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | 0.30 | 0.00 | 0.20 | 0.30 |
| refusal [heldout] | 0.93 | 0.80 | 0.80 | 0.90 | 0.83 | 0.60 | 0.83 | 0.83 | 0.77 |
| R1 rewrite synonym [train] | 0.07 | 0.67 | 0.80 | 0.00 | 0.03 | 0.07 | 0.03 | 0.07 | 0.40 |
| R1 rewrite typo [train] | 0.00 | 0.10 | 0.13 | 0.03 | 0.07 | 0.13 | 0.03 | 0.07 | 0.60 |
| R1 structure 3 adjectives [heldout] | 0.00 | 0.23 | 0.37 | 0.00 | 0.00 | 0.13 | 0.00 | 0.17 | 0.80 |
| R1 structure three sentences [heldout] | 0.00 | 0.10 | 0.13 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.87 |
| original test_iid (first 200) | 0.07 | 0.92 | 1.00 | 0.03 | 0.12 | 0.33 | 0.06 | 0.39 | 0.94 |
| original test_comp (first 200) | 0.00 | 0.88 | 0.98 | 0.00 | 0.01 | 0.27 | 0.00 | 0.21 | 0.92 |
| original test_morph (first 200) | 0.06 | 0.85 | 0.99 | 0.04 | 0.10 | 0.26 | 0.06 | 0.25 | 0.92 |
| original test_long (first 200) | 0.00 | 0.78 | 1.00 | 0.00 | 0.00 | 0.05 | 0.00 | 0.06 | 0.90 |
