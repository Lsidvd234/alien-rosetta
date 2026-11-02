# Missing fact 5x6 in the scratchpad (`analysis/scratch_missing_fact.py`), run maths-meaning-build-drop56-role5

Headline, comparable to the old run's 0.49 (173 of 353): test 'mul: needs dropped pair 5x6 (trained lengths)', 5x6 steps right by position against the gold working 0.50 (353); written steps exact (value = 5x6 + the carry the model wrote) 0.50 (353).

Below: written steps that multiply 5 and 6 (either order), every multiplication test (the 3-digit tests are mostly outside the trained lengths, so these pool easier and harder problems). Gold 5x6 = 42 in base 7 (+ carry).

| steps | exact | tens digit right | units digit right | wrong: tens right | wrong: units = an operand | top-3 implied facts cover | one fact per problem | neighbour facts exact |
|---|---|---|---|---|---|---|---|---|
| 497 | 0.48 | 0.49 | 0.51 | 0.03 | 0.43 | 0.90 | 0.66 (61) | 0.99 (890) |

Implied facts (written value minus written carry, base 7): 42 x237, 33 x154, 36 x56, 234 x22, 32 x17, 26 x5

By order: 5x6 0.91 (243), 6x5 0.06 (254)

By test: mul: 3x1 0.50 (16); mul: 3x2 0.45 (64); mul: 3x3 0.33 (36); mul: needs dropped pair 5x6 (trained lengths) 0.50 (353); mul: up to 2x2 (trained lengths) 0.36 (28)

By test, by position: mul: 3x1 0.15 (27); mul: 3x2 0.12 (89); mul: 3x3 0.03 (120); mul: needs dropped pair 5x6 (trained lengths) 0.50 (353); mul: up to 2x2 (trained lengths) 0.36 (28)
