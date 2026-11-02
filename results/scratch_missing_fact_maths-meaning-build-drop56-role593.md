# Missing fact 5x6 in the scratchpad (`analysis/scratch_missing_fact.py`), run maths-meaning-build-drop56-role593

Headline, comparable to the old run's 0.49 (173 of 353): test 'mul: needs dropped pair 5x6 (trained lengths)', 5x6 steps right by position against the gold working 0.54 (353); written steps exact (value = 5x6 + the carry the model wrote) 0.54 (353).

Below: written steps that multiply 5 and 6 (either order), every multiplication test (the 3-digit tests are mostly outside the trained lengths, so these pool easier and harder problems). Gold 5x6 = 42 in base 7 (+ carry).

| steps | exact | tens digit right | units digit right | wrong: tens right | wrong: units = an operand | top-3 implied facts cover | one fact per problem | neighbour facts exact |
|---|---|---|---|---|---|---|---|---|
| 487 | 0.51 | 0.60 | 0.51 | 0.18 | 0.63 | 0.98 | 0.93 (58) | 1.00 (823) |

Implied facts (written value minus written carry, base 7): 42 x248, 36 x163, 33 x67, 34 x8, 45 x1

By order: 5x6 0.97 (240), 6x5 0.06 (247)

By test: mul: 3x1 0.55 (22); mul: 3x2 0.42 (57); mul: 3x3 0.44 (27); mul: needs dropped pair 5x6 (trained lengths) 0.54 (353); mul: up to 2x2 (trained lengths) 0.39 (28)

By test, by position: mul: 3x1 0.19 (27); mul: 3x2 0.06 (89); mul: 3x3 0.03 (120); mul: needs dropped pair 5x6 (trained lengths) 0.54 (353); mul: up to 2x2 (trained lengths) 0.39 (28)
