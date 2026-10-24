# E1b error breakdown: analysis (understanding the English) vs glyphs (translation)

For every item: was the analysis the model wrote exactly right, and was the glyph output right? Source: eval_gen per-item results of e1b-hard-rep-cot-50k.

| group | n | both right | analysis right, glyphs WRONG (translation error) | analysis WRONG, glyphs right | both wrong (understanding error) |
|---|---|---|---|---|---|
| original test_iid | 500 | 96% | 3% | 0% | 1% |
| original test_comp | 500 | 93% | 5% | 0% | 2% |
| original test_morph | 500 | 93% | 5% | 0% | 1% |
| original test_long | 500 | 84% | 14% | 0% | 3% |
| in-distribution mix | 100 | 76% | 6% | 0% | 18% |
| held-out case = sentence | 100 | 74% | 6% | 0% | 20% |
| held-out det_sg = one | 100 | 72% | 3% | 2% | 23% |
| held-out noise = drop | 100 | 51% | 7% | 0% | 42% |
| held-out order = object cleft | 100 | 12% | 1% | 0% | 87% |
| held-out order = object topicalised | 100 | 20% | 1% | 0% | 79% |
| held-out order = subject cleft | 100 | 7% | 0% | 0% | 93% |
| held-out subject = name | 100 | 0% | 3% | 0% | 97% |
| held-out voice = passive | 100 | 12% | 2% | 0% | 86% |
| held-out synonym pairs | 100 | 4% | 0% | 3% | 93% |
| extrapolation: 5-6 adjectives on a noun | 100 | 58% | 19% | 0% | 23% |
| extrapolation: 5 clauses | 100 | 36% | 32% | 0% | 32% |
| **all groups pooled** | 3200 | 70% | 7% | 0% | 23% |
