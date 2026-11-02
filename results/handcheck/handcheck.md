# Tatoeba hand-check (`analysis/handcheck_tatoeba.py`)

Gold meanings judged by the student without knowing the group. gain = real15 right and baseline wrong (real15's output is the gold); control = random other sentences.

| group | judged | gold right | gold wrong | not sure | share right (95% CI, sure answers) |
|---|---|---|---|---|---|
| gain | 50 | 30 | 19 | 1 | 0.61 [0.47, 0.74] |
| control | 20 | 13 | 7 | 0 | 0.65 [0.43, 0.82] |

The control sentences are not typical: 20 of its 20 sentences are ones both models get right (the 411 non-gain sentences are mostly easy ones), so it measures the gold on easier sentences.

Gain vs control, gold right: 30/49 vs 13/20, Fisher exact p = 1.00 (no detectable difference; with 20 control sentences only a large difference would show).

Estimated real gains: 0.61 x 189 = about 116 of real15's 189 Tatoeba gains have a correct gold meaning (95% range 89-139); in the rest real15 matches a gold judged wrong.

## Notes

- 4: V=like.PAST
