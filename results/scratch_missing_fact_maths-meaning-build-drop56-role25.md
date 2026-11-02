# Missing fact 5x6 in the scratchpad (`analysis/scratch_missing_fact.py`), run maths-meaning-build-drop56-role25

Headline, comparable to the old run's 0.49 (173 of 353): test 'mul: needs dropped pair 5x6 (trained lengths)', 5x6 steps right by position against the gold working 0.52 (353); written steps exact (value = 5x6 + the carry the model wrote) 0.52 (353).

Below: written steps that multiply 5 and 6 (either order), every multiplication test (the 3-digit tests are mostly outside the trained lengths, so these pool easier and harder problems). Gold 5x6 = 42 in base 7 (+ carry).

| steps | exact | tens digit right | units digit right | wrong: tens right | wrong: units = an operand | top-3 implied facts cover | one fact per problem | neighbour facts exact |
|---|---|---|---|---|---|---|---|---|
| 526 | 0.49 | 0.67 | 0.49 | 0.35 | 0.65 | 1.00 | 0.92 (62) | 1.00 (955) |

Implied facts (written value minus written carry, base 7): 42 x256, 36 x224, 35 x45, 34 x1

By order: 5x6 1.00 (257), 6x5 0.00 (269)

By test: mul: 3x1 0.59 (22); mul: 3x2 0.43 (67); mul: 3x3 0.38 (56); mul: needs dropped pair 5x6 (trained lengths) 0.52 (353); mul: up to 2x2 (trained lengths) 0.32 (28)

By test, by position: mul: 3x1 0.26 (27); mul: 3x2 0.13 (89); mul: 3x3 0.07 (120); mul: needs dropped pair 5x6 (trained lengths) 0.52 (353); mul: up to 2x2 (trained lengths) 0.32 (28)
