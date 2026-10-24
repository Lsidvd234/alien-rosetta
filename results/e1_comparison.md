# Option 1 vs R1 vs E1a vs E1b on the same test items

Exact match of the glyph output (free generation; for the original splits of Option 1 and R1, teacher-forced exact match, which equals greedy generation for these models). All models: Qwen3-0.6B-Base + LoRA, seed 0.

E1's randomly held-out variants (never in E1 training): case=sentence, det_sg=one, noise=drop, order=object cleft, order=object topicalised, order=subject cleft, subject=name, voice=passive. Rows starting 'R1' are R1's categories; their [train]/[heldout] labels refer to R1's training, not E1's.

| group | n | Option 1 | R1 | E1a (plain) | E1b (analysis) |
|---|---|---|---|---|---|
| original test_iid | 500 | 0.99 | 0.99 | 0.26 | 0.96 |
| original test_comp | 500 | 0.99 | 0.99 | 0.22 | 0.93 |
| original test_morph | 500 | 0.97 | 0.97 | 0.27 | 0.93 |
| original test_long | 500 | 0.97 | 0.98 | 0.06 | 0.84 |
| in-distribution mix | 100 | 0.09 | 0.24 | 0.30 | 0.76 |
| held-out case = sentence | 100 | 0.07 | 0.31 | 0.34 | 0.74 |
| held-out det_sg = one | 100 | 0.03 | 0.22 | 0.42 | 0.74 |
| held-out noise = drop | 100 | 0.00 | 0.08 | 0.16 | 0.51 |
| held-out order = object cleft | 100 | 0.02 | 0.07 | 0.07 | 0.12 |
| held-out order = object topicalised | 100 | 0.04 | 0.09 | 0.13 | 0.20 |
| held-out order = subject cleft | 100 | 0.00 | 0.00 | 0.02 | 0.07 |
| held-out subject = name | 100 | 0.00 | 0.10 | 0.00 | 0.00 |
| held-out voice = passive | 100 | 0.05 | 0.12 | 0.07 | 0.12 |
| held-out synonym pairs | 100 | 0.02 | 0.28 | 0.06 | 0.07 |
| extrapolation: 5-6 adjectives on a noun | 100 | 0.08 | 0.11 | 0.25 | 0.58 |
| extrapolation: 5 clauses | 100 | 0.00 | 0.00 | 0.20 | 0.36 |
| refusal [train] | 100 | 0.00 | 1.00 | 1.00 | 1.00 |
| refusal [heldout] | 100 | 0.00 | 0.82 | 0.64 | 0.73 |
| R1 rewrite passive [train] | 50 | 0.00 | 0.94 | 0.00 | 0.00 |
| R1 rewrite passive [heldout] | 50 | 0.00 | 0.94 | 0.00 | 0.00 |
| R1 rewrite fronting [train] | 50 | 0.00 | 1.00 | 0.26 | 0.98 |
| R1 rewrite fronting [heldout] | 50 | 0.00 | 0.00 | 0.00 | 0.00 |
| R1 rewrite filler [train] | 50 | 0.02 | 0.96 | 0.40 | 0.68 |
| R1 rewrite filler [heldout] | 50 | 0.08 | 1.00 | 0.38 | 0.80 |
| R1 rewrite determiner [train] | 50 | 0.30 | 1.00 | 0.18 | 0.94 |
| R1 rewrite determiner [heldout] | 50 | 0.04 | 0.98 | 0.20 | 0.88 |
| R1 rewrite numeral [train] | 50 | 0.00 | 1.00 | 0.30 | 0.94 |
| R1 rewrite numeral [heldout] | 50 | 0.00 | 1.00 | 0.18 | 0.94 |
| R1 rewrite tense [train] | 50 | 0.76 | 1.00 | 0.38 | 1.00 |
| R1 rewrite tense [heldout] | 50 | 0.98 | 0.98 | 0.54 | 0.98 |
| R1 rewrite synonym [train] | 50 | 0.12 | 0.74 | 0.08 | 0.36 |
| R1 rewrite synonym [heldout] | 50 | 0.00 | 0.30 | 0.10 | 0.12 |
| R1 rewrite typo [train] | 50 | 0.04 | 0.12 | 0.12 | 0.52 |
| R1 rewrite typo [heldout] | 50 | 0.02 | 0.04 | 0.10 | 0.50 |
| R1 rewrite spacing [train] | 50 | 0.02 | 1.00 | 0.34 | 0.94 |
| R1 rewrite spacing [heldout] | 50 | 0.04 | 1.00 | 0.38 | 0.98 |
| R1 rewrite casing [train] | 50 | 0.56 | 0.78 | 0.24 | 0.94 |
| R1 rewrite casing [heldout] | 50 | 0.02 | 0.28 | 0.10 | 0.72 |
| R1 structure 2 adjectives [train] | 50 | 0.18 | 0.92 | 0.16 | 0.92 |
| R1 structure 3 adjectives [heldout] | 50 | 0.00 | 0.30 | 0.08 | 0.86 |
| R1 structure verb without object [train] | 50 | 0.00 | 0.74 | 0.02 | 0.00 |
| R1 structure name as subject [train] | 50 | 0.00 | 0.32 | 0.00 | 0.00 |
| R1 structure name as object [heldout] | 50 | 0.00 | 0.32 | 0.50 | 0.96 |
| R1 structure two sentences [train] | 50 | 0.00 | 0.88 | 0.10 | 0.88 |
| R1 structure three sentences [heldout] | 50 | 0.00 | 0.16 | 0.02 | 0.80 |
| R1 refusal [train] | 50 | 0.00 | 1.00 | 1.00 | 1.00 |
| R1 refusal [heldout] | 50 | 0.00 | 0.82 | 0.74 | 0.70 |
