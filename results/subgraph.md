# Compose or match? Seen vs new pieces (Faith and Fate's subgraph-matching test on Alien Rosetta)

Generated test sentences (the same 300 as the frequency-weighted test, none in training); the run's training data rebuilt exactly. Accuracy (number in brackets). `analysis/subgraph_check.py`.

## e2a-v2-80k

**1. Glyph words: exact form seen in training vs new form (root trained this many times)**

| root trained | form seen | new form, known root |
|---|---|---|
| 1-9 | 0.33 (88) | 0.27 (49) |
| 10-29 | 0.86 (292) | 0.64 (116) |
| 30-99 | 0.93 (360) | 0.75 (113) |
| 100+ | 0.91 (310) | 0.86 (65) |
| root never trained | - | - |
| (pronouns and grammar words) | 0.95 (952) | - |

**2. Sentences: every piece of a kind seen vs at least one new**

| piece | all seen | at least one new |
|---|---|---|
| English variant | 0.41 (284) | 0.06 (16) |
| clause template | 0.40 (288) | 0.08 (12) |
| slot shape | 0.39 (299) | 0.00 (1) |
| verb endings | 0.39 (300) | - |
| word form | 0.59 (95) | 0.30 (205) |
| (everything seen) | 0.59 (95) | - |

**3. Sentence accuracy by share of its pieces seen in training**

| pieces seen | sentences right |
|---|---|
| 100% | 0.59 (95) |
| 95%-100% | 0.00 (16) |
| 90%-95% | 0.31 (81) |
| 80%-90% | 0.32 (98) |
| 0%-80% | 0.50 (10) |

## e2b-v2-cot-80k-noheld

**1. Glyph words: exact form seen in training vs new form (root trained this many times)**

| root trained | form seen | new form, known root |
|---|---|---|
| 1-9 | 0.36 (67) | 0.20 (56) |
| 10-29 | 0.87 (316) | 0.80 (110) |
| 30-99 | 0.95 (373) | 0.90 (114) |
| 100+ | 0.93 (305) | 0.88 (60) |
| root never trained | - | 1.00 (1) |
| (pronouns and grammar words) | 0.97 (961) | 1.00 (1) |

**2. Sentences: every piece of a kind seen vs at least one new**

| piece | all seen | at least one new |
|---|---|---|
| clause template | 0.49 (286) | 0.29 (14) |
| slot shape | 0.48 (299) | 0.00 (1) |
| verb endings | 0.48 (300) | - |
| word form | 0.65 (96) | 0.40 (204) |
| (everything seen) | 0.65 (95) | - |

**3. Sentence accuracy by share of its pieces seen in training**

| pieces seen | sentences right |
|---|---|
| 100% | 0.65 (95) |
| 95%-100% | 0.28 (18) |
| 90%-95% | 0.41 (79) |
| 80%-90% | 0.42 (98) |
| 0%-80% | 0.30 (10) |

**4. Where the steps went wrong (meaning first, then glyphs)**

Check of the check: re-rendering the GOLD analysis gives the gold glyphs for 198/200 generated sentences, so about 1% of true propagation errors may be counted as 'both wrong'.

| outcome | sentences | share |
|---|---|---|
| correct | 142 | 0.47 |
| local | 94 | 0.31 |
| propagation | 24 | 0.08 |
| restoration | 1 | 0.00 |
| both wrong | 14 | 0.05 |
| unreadable analysis | 25 | 0.08 |

## e2b-v2fix-cot-80k-noheld

**1. Glyph words: exact form seen in training vs new form (root trained this many times)**

| root trained | form seen | new form, known root |
|---|---|---|
| 1-9 | 1.00 (1) | - |
| 10-29 | 0.91 (367) | 0.84 (140) |
| 30-99 | 0.94 (391) | 0.86 (114) |
| 100+ | 0.93 (300) | 0.84 (55) |
| root never trained | - | - |
| (pronouns and grammar words) | 0.97 (937) | 1.00 (1) |

**2. Sentences: every piece of a kind seen vs at least one new**

| piece | all seen | at least one new |
|---|---|---|
| clause template | 0.61 (287) | 0.62 (13) |
| slot shape | 0.61 (300) | - |
| verb endings | 0.61 (300) | - |
| word form | 0.73 (102) | 0.55 (198) |
| (everything seen) | 0.72 (101) | - |

**3. Sentence accuracy by share of its pieces seen in training**

| pieces seen | sentences right |
|---|---|
| 100% | 0.72 (101) |
| 95%-100% | 0.46 (24) |
| 90%-95% | 0.57 (79) |
| 80%-90% | 0.56 (88) |
| 0%-80% | 0.50 (8) |

**4. Where the steps went wrong (meaning first, then glyphs)**

Check of the check: re-rendering the GOLD analysis gives the gold glyphs for 198/200 generated sentences, so about 1% of true propagation errors may be counted as 'both wrong'.

| outcome | sentences | share |
|---|---|---|
| correct | 180 | 0.60 |
| local | 58 | 0.19 |
| propagation | 30 | 0.10 |
| restoration | 2 | 0.01 |
| both wrong | 8 | 0.03 |
| unreadable analysis | 22 | 0.07 |

## e2b-v2fix-cot-80k

**1. Glyph words: exact form seen in training vs new form (root trained this many times)**

| root trained | form seen | new form, known root |
|---|---|---|
| 1-9 | 1.00 (2) | - |
| 10-29 | 0.82 (358) | 0.69 (133) |
| 30-99 | 0.87 (387) | 0.69 (124) |
| 100+ | 0.91 (305) | 0.75 (59) |
| root never trained | - | - |
| (pronouns and grammar words) | 0.89 (938) | - |

**2. Sentences: every piece of a kind seen vs at least one new**

| piece | all seen | at least one new |
|---|---|---|
| English variant | 0.51 (284) | 0.19 (16) |
| clause template | 0.51 (288) | 0.08 (12) |
| slot shape | 0.49 (299) | 0.00 (1) |
| verb endings | 0.49 (300) | - |
| word form | 0.68 (107) | 0.39 (193) |
| (everything seen) | 0.68 (107) | - |

**3. Sentence accuracy by share of its pieces seen in training**

| pieces seen | sentences right |
|---|---|
| 100% | 0.68 (107) |
| 95%-100% | 0.29 (14) |
| 90%-95% | 0.39 (80) |
| 80%-90% | 0.40 (92) |
| 0%-80% | 0.43 (7) |

**4. Where the steps went wrong (meaning first, then glyphs)**

Check of the check: re-rendering the GOLD analysis gives the gold glyphs for 198/200 generated sentences, so about 1% of true propagation errors may be counted as 'both wrong'.

| outcome | sentences | share |
|---|---|---|
| correct | 147 | 0.49 |
| local | 52 | 0.17 |
| propagation | 31 | 0.10 |
| restoration | 1 | 0.00 |
| both wrong | 19 | 0.06 |
| unreadable analysis | 50 | 0.17 |

## e2b-v2fix-focal-r2

**1. Glyph words: exact form seen in training vs new form (root trained this many times)**

| root trained | form seen | new form, known root |
|---|---|---|
| 1-9 | 1.00 (1) | - |
| 10-29 | 0.86 (367) | 0.79 (140) |
| 30-99 | 0.94 (391) | 0.81 (114) |
| 100+ | 0.94 (300) | 0.85 (55) |
| root never trained | - | - |
| (pronouns and grammar words) | 0.97 (937) | 1.00 (1) |

**2. Sentences: every piece of a kind seen vs at least one new**

| piece | all seen | at least one new |
|---|---|---|
| clause template | 0.57 (287) | 0.46 (13) |
| slot shape | 0.57 (300) | - |
| verb endings | 0.57 (300) | - |
| word form | 0.71 (102) | 0.49 (198) |
| (everything seen) | 0.71 (101) | - |

**3. Sentence accuracy by share of its pieces seen in training**

| pieces seen | sentences right |
|---|---|
| 100% | 0.71 (101) |
| 95%-100% | 0.38 (24) |
| 90%-95% | 0.48 (79) |
| 80%-90% | 0.53 (88) |
| 0%-80% | 0.50 (8) |

**4. Where the steps went wrong (meaning first, then glyphs)**

Check of the check: re-rendering the GOLD analysis gives the gold glyphs for 198/200 generated sentences, so about 1% of true propagation errors may be counted as 'both wrong'.

| outcome | sentences | share |
|---|---|---|
| correct | 169 | 0.56 |
| local | 57 | 0.19 |
| propagation | 28 | 0.09 |
| restoration | 1 | 0.00 |
| both wrong | 16 | 0.05 |
| unreadable analysis | 29 | 0.10 |

## e2b-v2fix-loan-r2

**1. Glyph words: exact form seen in training vs new form (root trained this many times)**

| root trained | form seen | new form, known root |
|---|---|---|
| 1-9 | 1.00 (2) | - |
| 10-29 | 0.87 (351) | 0.79 (143) |
| 30-99 | 0.95 (388) | 0.91 (128) |
| 100+ | 0.95 (292) | 0.88 (64) |
| root never trained | - | - |
| (pronouns and grammar words) | 0.97 (938) | - |

**2. Sentences: every piece of a kind seen vs at least one new**

| piece | all seen | at least one new |
|---|---|---|
| clause template | 0.64 (292) | 0.38 (8) |
| slot shape | 0.64 (299) | 0.00 (1) |
| verb endings | 0.63 (300) | - |
| word form | 0.74 (99) | 0.58 (201) |
| (everything seen) | 0.74 (97) | - |

**3. Sentence accuracy by share of its pieces seen in training**

| pieces seen | sentences right |
|---|---|
| 100% | 0.74 (97) |
| 95%-100% | 0.50 (20) |
| 90%-95% | 0.59 (82) |
| 80%-90% | 0.59 (94) |
| 0%-80% | 0.71 (7) |

**4. Where the steps went wrong (meaning first, then glyphs)**

Check of the check: re-rendering the GOLD analysis gives the gold glyphs for 198/200 generated sentences, so about 1% of true propagation errors may be counted as 'both wrong'.

| outcome | sentences | share |
|---|---|---|
| correct | 189 | 0.63 |
| local | 54 | 0.18 |
| propagation | 22 | 0.07 |
| restoration | 1 | 0.00 |
| both wrong | 11 | 0.04 |
| unreadable analysis | 23 | 0.08 |

## e2b-v2fix-para

**1. Glyph words: exact form seen in training vs new form (root trained this many times)**

| root trained | form seen | new form, known root |
|---|---|---|
| 1-9 | 1.00 (1) | - |
| 10-29 | 0.91 (367) | 0.82 (140) |
| 30-99 | 0.95 (391) | 0.88 (114) |
| 100+ | 0.94 (300) | 0.87 (55) |
| root never trained | - | - |
| (pronouns and grammar words) | 0.97 (937) | 1.00 (1) |

**2. Sentences: every piece of a kind seen vs at least one new**

| piece | all seen | at least one new |
|---|---|---|
| clause template | 0.64 (287) | 0.54 (13) |
| slot shape | 0.63 (300) | - |
| verb endings | 0.63 (300) | - |
| word form | 0.76 (102) | 0.57 (198) |
| (everything seen) | 0.76 (101) | - |

**3. Sentence accuracy by share of its pieces seen in training**

| pieces seen | sentences right |
|---|---|
| 100% | 0.76 (101) |
| 95%-100% | 0.58 (24) |
| 90%-95% | 0.51 (79) |
| 80%-90% | 0.60 (88) |
| 0%-80% | 0.75 (8) |

**4. Where the steps went wrong (meaning first, then glyphs)**

Check of the check: re-rendering the GOLD analysis gives the gold glyphs for 198/200 generated sentences, so about 1% of true propagation errors may be counted as 'both wrong'.

| outcome | sentences | share |
|---|---|---|
| correct | 190 | 0.63 |
| local | 54 | 0.18 |
| propagation | 25 | 0.08 |
| restoration | 0 | 0.00 |
| both wrong | 9 | 0.03 |
| unreadable analysis | 22 | 0.07 |
