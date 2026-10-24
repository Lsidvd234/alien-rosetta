# E1 with 5 epochs: E1a, E1b and an E1b replicate

Free generation, exact match of the glyph output. 5-epoch runs: RunPod V100 (fp16, torch 2.6, transformers 5.16.1), otherwise identical settings to the 3-epoch Kaggle runs. Held-out voice/order groups of the 5-epoch runs are strict (only items that contain the variant); the 3-epoch cells for those rows are left blank (see e1_heldout_corrected.md). Seed 1 = a new random language instance (different test items). n = 500 per original split, 100 per E1 group, 50 per R1 group.

| group | n | E1a 3ep | E1b 3ep | E1a 5ep | E1b 5ep | E1b 5ep seed 1 |
|---|---|---|---|---|---|---|
| original test_iid | 500 | 0.26 | 0.96 | 0.98 | 0.99 | 0.97 |
| original test_comp | 500 | 0.22 | 0.93 | 0.97 | 0.97 | 0.97 |
| original test_morph | 500 | 0.27 | 0.93 | 0.96 | 0.97 | 0.96 |
| original test_long | 500 | 0.06 | 0.84 | 0.85 | 0.87 | 0.91 |
| in-distribution mix | 100 | 0.30 | 0.76 | 0.76 | 0.80 | 0.78 |
| held-out case = sentence | 100 | 0.34 | 0.74 | 0.76 | 0.80 | 0.86 |
| held-out det_sg = one | 100 | 0.42 | 0.74 | 0.80 | 0.83 | 0.85 |
| held-out noise = drop | 100 | 0.16 | 0.51 | 0.42 | 0.54 | 0.57 |
| held-out order = object cleft | 100 |  |  | 0.01 | 0.02 | 0.04 |
| held-out order = object topicalised | 100 |  |  | 0.00 | 0.00 | 0.01 |
| held-out order = subject cleft | 100 |  |  | 0.02 | 0.09 | 0.02 |
| held-out subject = name | 100 | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 |
| held-out voice = passive | 100 |  |  | 0.00 | 0.01 | 0.02 |
| held-out synonym pairs | 100 | 0.06 | 0.07 | 0.13 | 0.10 | 0.17 |
| extrapolation: 5-6 adjectives on a noun | 100 | 0.25 | 0.58 | 0.53 | 0.71 | 0.75 |
| extrapolation: 5 clauses | 100 | 0.20 | 0.36 | 0.48 | 0.48 | 0.58 |
| refusal [train] | 100 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| refusal [heldout] | 100 | 0.64 | 0.73 | 0.76 | 0.79 | 0.83 |
| R1 structure 2 adjectives [train] | 50 | 0.16 | 0.92 | 0.98 | 1.00 | 1.00 |
| R1 structure 3 adjectives [heldout] | 50 | 0.08 | 0.86 | 0.94 | 0.98 | 0.98 |
| R1 structure verb without object [train] | 50 | 0.02 | 0.00 | 0.02 | 0.02 | 0.00 |
| R1 structure name as subject [train] | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R1 structure name as object [heldout] | 50 | 0.50 | 0.96 | 1.00 | 0.98 | 0.92 |
| R1 structure two sentences [train] | 50 | 0.10 | 0.88 | 0.94 | 0.96 | 0.96 |
| R1 structure three sentences [heldout] | 50 | 0.02 | 0.80 | 0.98 | 0.86 | 0.94 |
| R1 refusal [train] | 50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| R1 refusal [heldout] | 50 | 0.74 | 0.70 | 0.78 | 0.78 | 0.76 |

| run | training minutes | target tokens/s |
|---|---|---|
| E1a 3ep | 54 | 1743 |
| E1b 3ep | 74 | 1962 |
| E1a 5ep | 76 | 2089 |
| E1b 5ep | 88 | 2737 |
| E1b 5ep seed 1 | 88 | 2788 |
