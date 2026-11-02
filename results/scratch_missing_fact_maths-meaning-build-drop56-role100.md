# Missing fact 5x6 in the scratchpad (`analysis/scratch_missing_fact.py`), run maths-meaning-build-drop56-role100

Headline, comparable to the old run's 0.49 (173 of 353): test 'mul: needs dropped pair 5x6 (trained lengths)', 5x6 steps right by position against the gold working 0.50 (353); written steps exact (value = 5x6 + the carry the model wrote) 0.50 (353).

Below: written steps that multiply 5 and 6 (either order), every multiplication test (the 3-digit tests are mostly outside the trained lengths, so these pool easier and harder problems). Gold 5x6 = 42 in base 7 (+ carry).

| steps | exact | tens digit right | units digit right | wrong: tens right | wrong: units = an operand | top-3 implied facts cover | one fact per problem | neighbour facts exact |
|---|---|---|---|---|---|---|---|---|
| 505 | 0.47 | 0.86 | 0.47 | 0.73 | 0.63 | 0.90 | 0.66 (65) | 1.00 (893) |

Implied facts (written value minus written carry, base 7): 42 x237, 45 x159, 36 x61, 33 x17, 35 x16, 34 x11

By order: 5x6 0.96 (248), 6x5 0.00 (257)

By test: mul: 3x1 0.47 (30); mul: 3x2 0.37 (65); mul: 3x3 0.41 (29); mul: needs dropped pair 5x6 (trained lengths) 0.50 (353); mul: up to 2x2 (trained lengths) 0.32 (28)

By test, by position: mul: 3x1 0.22 (27); mul: 3x2 0.08 (89); mul: 3x3 0.01 (120); mul: needs dropped pair 5x6 (trained lengths) 0.50 (353); mul: up to 2x2 (trained lengths) 0.32 (28)
