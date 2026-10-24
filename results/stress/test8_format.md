# Test 8: input-format brittleness

250 new sentences per case, each scored as normally written and changed; the correct translation is the same.

| case | n | example | Option 1: Latin-sound start (normal -> changed) | Option 2: + transliteration (normal -> changed) | Option 3a: plain text first (normal -> changed) | Control: shuffled-sound start (normal -> changed) |
|---|---|---|---|---|---|---|
| double spaces between words | 250 | you  swear | 99.6% -> 4.0% | 98.4% -> 4.0% | 30.8% -> 0.0% | 98.8% -> 2.8% |
| trailing space | 250 | I rack  | 99.6% -> 99.6% | 99.6% -> 100.0% | 29.6% -> 29.2% | 99.6% -> 99.6% |
| leading space | 250 |  she pops | 100.0% -> 99.6% | 99.6% -> 95.6% | 31.6% -> 13.6% | 99.2% -> 99.2% |
| prompt label 'Input/Output' instead of 'English/Target' | 250 | Input: he smiles / Output: /  | 99.6% -> 87.6% | 99.6% -> 28.0% | 28.0% -> 18.4% | 99.2% -> 34.0% |
| no newline after 'Target:' | 250 | English: we die / Target:  | 100.0% -> 82.4% | 99.2% -> 18.0% | 30.0% -> 2.8% | 99.2% -> 60.0% |
| typo (two letters swapped in a word) | 250 | it praeyd | 99.2% -> 7.2% | 98.0% -> 8.8% | 30.0% -> 3.2% | 99.2% -> 9.6% |
| 'the the' (doubled word) | 250 | the the fox lags | 99.6% -> 95.6% | 100.0% -> 98.4% | 23.6% -> 14.0% | 99.6% -> 88.0% |
| 'Translate:' instruction before the prompt | 250 | Translate: / English: I sank / Target: /  | 99.2% -> 91.6% | 100.0% -> 32.0% | 42.4% -> 29.6% | 99.2% -> 68.4% |
