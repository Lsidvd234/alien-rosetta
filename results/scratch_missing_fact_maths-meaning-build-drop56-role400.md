# Missing fact 5x6 in the scratchpad (`analysis/scratch_missing_fact.py`), run maths-meaning-build-drop56-role400

Headline, comparable to the old run's 0.49 (173 of 353): test 'mul: needs dropped pair 5x6 (trained lengths)', 5x6 steps right by position against the gold working 0.73 (353); written steps exact (value = 5x6 + the carry the model wrote) 0.73 (353).

Below: written steps that multiply 5 and 6 (either order), every multiplication test (the 3-digit tests are mostly outside the trained lengths, so these pool easier and harder problems). Gold 5x6 = 42 in base 7 (+ carry).

| steps | exact | tens digit right | units digit right | wrong: tens right | wrong: units = an operand | top-3 implied facts cover | one fact per problem | neighbour facts exact |
|---|---|---|---|---|---|---|---|---|
| 539 | 0.68 | 0.75 | 0.68 | 0.20 | 0.62 | 0.95 | 0.90 (73) | 1.00 (951) |

Implied facts (written value minus written carry, base 7): 42 x367, 36 x128, 34 x18, 234 x13, 33 x9, 225 x4

By order: 5x6 0.93 (266), 6x5 0.44 (273)

By test: mul: 3x1 0.75 (24); mul: 3x2 0.65 (77); mul: 3x3 0.46 (57); mul: needs dropped pair 5x6 (trained lengths) 0.73 (353); mul: up to 2x2 (trained lengths) 0.57 (28)

By test, by position: mul: 3x1 0.41 (27); mul: 3x2 0.16 (89); mul: 3x3 0.07 (120); mul: needs dropped pair 5x6 (trained lengths) 0.73 (353); mul: up to 2x2 (trained lengths) 0.57 (28)
