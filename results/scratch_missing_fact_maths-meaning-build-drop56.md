# Missing fact 5x6 in the scratchpad (`analysis/scratch_missing_fact.py`), run maths-meaning-build-drop56

Headline, comparable to the old run's 0.49 (173 of 353): test 'mul: needs dropped pair 5x6 (trained lengths)', 5x6 steps right by position against the gold working 0.07 (353); written steps exact (value = 5x6 + the carry the model wrote) 0.07 (353).

Below: written steps that multiply 5 and 6 (either order), every multiplication test (the 3-digit tests are mostly outside the trained lengths, so these pool easier and harder problems). Gold 5x6 = 42 in base 7 (+ carry).

| steps | exact | tens digit right | units digit right | wrong: tens right | wrong: units = an operand | top-3 implied facts cover | one fact per problem | neighbour facts exact |
|---|---|---|---|---|---|---|---|---|
| 515 | 0.06 | 0.35 | 0.06 | 0.31 | 0.63 | 0.96 | 0.83 (72) | 1.00 (931) |

Implied facts (written value minus written carry, base 7): 36 x257, 35 x206, 42 x31, 25 x7, 33 x6, 41 x6

By order: 5x6 0.00 (265), 6x5 0.12 (250)

By test: mul: 3x1 0.00 (24); mul: 3x2 0.00 (61); mul: 3x3 0.06 (49); mul: needs dropped pair 5x6 (trained lengths) 0.07 (353); mul: up to 2x2 (trained lengths) 0.07 (28)

By test, by position: mul: 3x1 0.00 (27); mul: 3x2 0.00 (89); mul: 3x3 0.00 (120); mul: needs dropped pair 5x6 (trained lengths) 0.07 (353); mul: up to 2x2 (trained lengths) 0.07 (28)
