# Test 4: rephrased English, same correct answer

300 new sentences per row (only training words, no held-out combinations, none in training).
Each sentence is scored twice: as normally written, and rephrased. The correct translation is identical.
Synonyms are only used when the synonym is NOT itself a word of the language.

| rephrasing | example | Option 1: Latin-sound start (normal -> rephrased) | Option 2: + transliteration (normal -> rephrased) | Option 3a: plain text first (normal -> rephrased) | Control: shuffled-sound start (normal -> rephrased) |
|---|---|---|---|---|---|
| capitalised + full stop | We sway. | 99.7% -> 99.7% | 100.0% -> 99.7% | 35.7% -> 34.0% | 99.3% -> 99.3% |
| contractions (didn't, won't...) | we won't shy | 99.3% -> 92.7% | 99.7% -> 62.7% | 33.7% -> 27.7% | 99.3% -> 72.3% |
| 'a' instead of 'the' | a cup gazes | 100.0% -> 48.3% | 98.7% -> 37.3% | 34.3% -> 17.3% | 100.0% -> 79.3% |
| passive voice | the rib is sustained by you | 99.0% -> 0.0% | 99.0% -> 0.3% | 0.0% -> 0.0% | 99.3% -> 0.0% |
| synonym (word not in the language) | the hog clung | 99.3% -> 18.7% | 98.0% -> 9.3% | 28.3% -> 0.7% | 99.7% -> 1.7% |
| UPPER CASE | I YELL | 99.3% -> 17.7% | 99.3% -> 5.7% | 57.3% -> 5.0% | 99.7% -> 15.0% |
