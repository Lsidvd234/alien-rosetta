# Missing fact 5x6 in the scratchpad (`analysis/scratch_missing_fact.py`), run maths-meaning-scratch-drop56-tw

Headline, comparable to the old run's 0.49 (173 of 353): test 'mul: needs dropped pair 5x6 (trained lengths)', 5x6 steps right by position against the gold working 0.49 (353); written steps exact (value = 5x6 + the carry the model wrote) 0.50 (353).

Below: written steps that multiply 5 and 6 (either order), every multiplication test (the 3-digit tests are mostly outside the trained lengths, so these pool easier and harder problems). Gold 5x6 = 42 in base 7 (+ carry).

| steps | exact | tens digit right | units digit right | wrong: tens right | wrong: units = an operand | top-3 implied facts cover | one fact per problem | neighbour facts exact |
|---|---|---|---|---|---|---|---|---|
| 516 | 0.50 | 0.83 | 0.50 | 0.66 | 0.35 | 0.85 | 0.60 (65) | 0.96 (987) |

Implied facts (written value minus written carry, base 7): 42 x260, 36 x117, 45 x62, 41 x53, 46 x14, 33 x8

By order: 5x6 0.51 (245), 6x5 0.50 (271)

By test: mul: 3x1 0.53 (19); mul: 3x2 0.64 (66); mul: 3x3 0.40 (50); mul: needs dropped pair 5x6 (trained lengths) 0.50 (353); mul: up to 2x2 (trained lengths) 0.46 (28)

By test, by position: mul: 3x1 0.30 (27); mul: 3x2 0.15 (89); mul: 3x3 0.02 (120); mul: needs dropped pair 5x6 (trained lengths) 0.49 (353); mul: up to 2x2 (trained lengths) 0.46 (28)
