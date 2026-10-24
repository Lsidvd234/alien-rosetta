# Test 30: invisible Unicode changes (look the same on screen)

100 new sentences per kind, one change each (or 3 mixed). FIX rows: the same inputs after NFKC normalisation, and after NFKC + removing invisible format characters. Cells: accuracy (share of inputs restored to the exact original text by the fix).

| change | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|
| homoglyph (Cyrillic look-alike letter) | 27.0% | 29.0% | 7.0% | 25.0% |
|    FIX NFKC: homoglyph (Cyrillic look-alike letter) (restored 0.0%) | 27.0% | 29.0% | 7.0% | 25.0% |
|    FIX NFKC + drop invisible: homoglyph (Cyrillic look-alike letter) (restored 0.0%) | 27.0% | 29.0% | 7.0% | 25.0% |
| zero-width space inside a word | 34.0% | 43.0% | 12.0% | 33.0% |
|    FIX NFKC: zero-width space inside a word (restored 0.0%) | 34.0% | 43.0% | 12.0% | 33.0% |
|    FIX NFKC + drop invisible: zero-width space inside a word (restored 100.0%) | 100.0% | 100.0% | 29.0% | 98.0% |
| no-break space between words | 80.0% | 71.0% | 19.0% | 71.0% |
|    FIX NFKC: no-break space between words (restored 100.0%) | 100.0% | 100.0% | 29.0% | 98.0% |
|    FIX NFKC + drop invisible: no-break space between words (restored 100.0%) | 100.0% | 100.0% | 29.0% | 98.0% |
| full-width letters in one word | 16.0% | 5.0% | 4.0% | 19.0% |
|    FIX NFKC: full-width letters in one word (restored 100.0%) | 100.0% | 100.0% | 29.0% | 98.0% |
|    FIX NFKC + drop invisible: full-width letters in one word (restored 100.0%) | 100.0% | 100.0% | 29.0% | 98.0% |
| soft hyphen inside a word | 34.0% | 43.0% | 13.0% | 45.0% |
|    FIX NFKC: soft hyphen inside a word (restored 0.0%) | 34.0% | 43.0% | 13.0% | 45.0% |
|    FIX NFKC + drop invisible: soft hyphen inside a word (restored 100.0%) | 100.0% | 100.0% | 29.0% | 98.0% |
| 3 mixed invisible changes | 1.0% | 2.0% | 1.0% | 0.0% |
|    FIX NFKC: 3 mixed invisible changes (restored 0.0%) | 13.0% | 8.0% | 7.0% | 10.0% |
|    FIX NFKC + drop invisible: 3 mixed invisible changes (restored 39.0%) | 49.0% | 52.0% | 16.0% | 47.0% |
