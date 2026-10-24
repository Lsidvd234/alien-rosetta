# R1 (retrained) vs Option 1

Both: Qwen3-0.6B-Base + LoRA r=16, 3 epochs, glyphs as new tokens starting at their Latin sounds, seed 0. R1 adds varied English, new structures, refusals and plain-English replay (train/augment.py). Teacher-forced exact match. [train] = this kind of variant was in R1's training data; [heldout] = it never was (tests generalisation). Option 1 never saw any of them.

## The four original test splits

| split | Option 1 | R1 |
|---|---|---|
| test_iid | 0.990 | 0.986 |
| test_comp | 0.986 | 0.988 |
| test_morph | 0.972 | 0.970 |
| test_long | 0.974 | 0.982 |

Training time: Option 1 49.6 min, R1 58.8 min.

## New categories (100 items each)

| category | Option 1 | R1 | R1 on Kaggle (check) |
|---|---|---|---|
| rewrite passive [train] | 0.00 | 0.93 | 0.93 |
| rewrite passive [heldout] | 0.00 | 0.94 | 0.94 |
| rewrite fronting [train] | 0.00 | 0.99 | 0.99 |
| rewrite fronting [heldout] | 0.00 | 0.00 | 0.00 |
| rewrite filler [train] | 0.01 | 0.99 | 0.99 |
| rewrite filler [heldout] | 0.06 | 0.99 | 0.99 |
| rewrite determiner [train] | 0.16 | 0.98 | 0.98 |
| rewrite determiner [heldout] | 0.03 | 0.98 | 0.98 |
| rewrite numeral [train] | 0.00 | 1.00 | 1.00 |
| rewrite numeral [heldout] | 0.00 | 1.00 | 1.00 |
| rewrite tense [train] | 0.81 | 0.99 | 0.99 |
| rewrite tense [heldout] | 0.96 | 0.99 | 0.99 |
| rewrite synonym [train] | 0.14 | 0.78 | 0.78 |
| rewrite synonym [heldout] | 0.00 | 0.15 | 0.15 |
| rewrite typo [train] | 0.03 | 0.13 | 0.13 |
| rewrite typo [heldout] | 0.07 | 0.12 | 0.12 |
| rewrite spacing [train] | 0.05 | 0.99 | 0.99 |
| rewrite spacing [heldout] | 0.03 | 0.98 | 0.98 |
| rewrite casing [train] | 0.41 | 0.72 | 0.72 |
| rewrite casing [heldout] | 0.04 | 0.29 | 0.29 |
| structure 2 adjectives [train] | 0.09 | 0.96 | 0.96 |
| structure 3 adjectives [heldout] | 0.00 | 0.30 | 0.31 |
| structure verb without object [train] | 0.00 | 0.83 | 0.83 |
| structure name as subject [train] | 0.00 | 0.47 | 0.47 |
| structure name as object [heldout] | 0.00 | 0.22 | 0.22 |
| structure two sentences [train] | 0.00 | 0.90 | 0.90 |
| structure three sentences [heldout] | 0.00 | 0.06 | 0.06 |
| refusal [train] | 0.00 | 1.00 | 1.00 |
| refusal [heldout] | 0.00 | 0.85 | 0.85 |

## Training data used by R1

| kind | examples |
|---|---|
| none | 25091 |
| passive | 2156 |
| fronting | 2016 |
| typo | 2781 |
| numeral | 2607 |
| spacing | 2899 |
| determiner | 1787 |
| filler | 2950 |
| tense | 3019 |
| synonym | 1783 |
| casing | 2911 |
| 2 adjectives | 1500 |
| verb without object | 1000 |
| name as subject | 1500 |
| two sentences | 1000 |
| refusal | 1500 |
| replay | 5000 |
| names_used | 1461 |
