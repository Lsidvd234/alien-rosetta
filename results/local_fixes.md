# Inference-time fixes without retraining (L2 sentence splitting, L3 lower-casing)

50 items per group from E1's evaluation groups; exact match of the glyph output. Cells: as given -> with the fix. Split rule: break at '.', ';', '!' and ', and '; translate each part; join. Lower-casing also lower-cases names (none in these groups' casing rows except the in-distribution mix).

| fix | group | Option 1 | R1 | E1a (plain) | E1b (analysis) |
|---|---|---|---|---|---|
| split | R1 structure two sentences [train] | 0.00 -> 1.00 | 0.88 -> 1.00 | 0.10 -> 0.06 | 0.88 -> 0.88 |
| split | R1 structure three sentences [heldout] | 0.00 -> 0.98 | 0.16 -> 1.00 | 0.02 -> 0.06 | 0.82 -> 0.84 |
| split | extrapolation: 5 clauses | 0.00 -> 0.00 | 0.00 -> 0.02 | 0.16 -> 0.22 | 0.30 -> 0.54 |
| split | in-distribution mix | 0.10 -> 0.12 | 0.34 -> 0.34 | 0.28 -> 0.28 | 0.70 -> 0.74 |
| lowercase | R1 rewrite casing [train] | 0.56 -> 1.00 | 0.78 -> 0.96 | 0.24 -> 0.26 | 0.94 -> 0.96 |
| lowercase | R1 rewrite casing [heldout] | 0.02 -> 1.00 | 0.28 -> 0.98 | 0.10 -> 0.18 | 0.72 -> 0.98 |
| lowercase | held-out case = sentence | 0.06 -> 0.06 | 0.32 -> 0.30 | 0.34 -> 0.36 | 0.82 -> 0.78 |
| lowercase | in-distribution mix | 0.10 -> 0.12 | 0.34 -> 0.32 | 0.28 -> 0.24 | 0.70 -> 0.68 |
