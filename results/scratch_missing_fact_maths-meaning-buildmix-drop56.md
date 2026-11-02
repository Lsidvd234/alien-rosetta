# Missing fact 5x6 in the scratchpad (`analysis/scratch_missing_fact.py`), run maths-meaning-buildmix-drop56

Headline, comparable to the old run's 0.49 (173 of 353): test 'mul: needs dropped pair 5x6 (trained lengths)', 5x6 steps right by position against the gold working 0.83 (353); written steps exact (value = 5x6 + the carry the model wrote) 0.83 (353).

Below: written steps that multiply 5 and 6 (either order), every multiplication test (the 3-digit tests are mostly outside the trained lengths, so these pool easier and harder problems). Gold 5x6 = 42 in base 7 (+ carry).

| steps | exact | tens digit right | units digit right | wrong: tens right | wrong: units = an operand | top-3 implied facts cover | one fact per problem | neighbour facts exact |
|---|---|---|---|---|---|---|---|---|
| 530 | 0.83 | 0.87 | 0.83 | 0.23 | 0.53 | 0.94 | 0.79 (68) | 1.00 (954) |

Implied facts (written value minus written carry, base 7): 42 x439, 34 x38, 36 x23, 33 x11, 45 x10, 43 x4

By order: 5x6 0.81 (263), 6x5 0.84 (267)

By test: mul: 3x1 0.55 (20); mul: 3x2 0.92 (66); mul: 3x3 0.86 (63); mul: needs dropped pair 5x6 (trained lengths) 0.83 (353); mul: up to 2x2 (trained lengths) 0.71 (28)

By test, by position: mul: 3x1 0.19 (27); mul: 3x2 0.28 (89); mul: 3x3 0.09 (120); mul: needs dropped pair 5x6 (trained lengths) 0.83 (353); mul: up to 2x2 (trained lengths) 0.64 (28)
