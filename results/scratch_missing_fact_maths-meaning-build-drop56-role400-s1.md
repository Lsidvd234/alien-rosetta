# Missing fact 5x6 in the scratchpad (`analysis/scratch_missing_fact.py`), run maths-meaning-build-drop56-role400-s1

Headline, comparable to the old run's 0.49 (173 of 353): test 'mul: needs dropped pair 5x6 (trained lengths)', 5x6 steps right by position against the gold working 0.48 (355); written steps exact (value = 5x6 + the carry the model wrote) 0.48 (355).

Below: written steps that multiply 5 and 6 (either order), every multiplication test (the 3-digit tests are mostly outside the trained lengths, so these pool easier and harder problems). Gold 5x6 = 42 in base 7 (+ carry).

| steps | exact | tens digit right | units digit right | wrong: tens right | wrong: units = an operand | top-3 implied facts cover | one fact per problem | neighbour facts exact |
|---|---|---|---|---|---|---|---|---|
| 494 | 0.45 | 0.63 | 0.45 | 0.33 | 0.59 | 0.99 | 0.91 (58) | 0.99 (871) |

Implied facts (written value minus written carry, base 7): 36 x245, 42 x222, 34 x24, 33 x2, 40 x1

By order: 5x6 0.90 (239), 6x5 0.03 (255)

By test: mul: 3x1 0.33 (21); mul: 3x2 0.45 (38); mul: 3x3 0.38 (48); mul: needs dropped pair 5x6 (trained lengths) 0.48 (355); mul: up to 2x2 (trained lengths) 0.34 (32)

By test, by position: mul: 3x1 0.16 (32); mul: 3x2 0.23 (70); mul: 3x3 0.10 (119); mul: needs dropped pair 5x6 (trained lengths) 0.48 (355); mul: up to 2x2 (trained lengths) 0.34 (32)
