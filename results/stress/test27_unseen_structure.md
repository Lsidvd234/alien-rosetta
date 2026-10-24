# Test 27: does an input fail exactly when it contains something never seen in the training English?

Pool: normal sentences, the 4 test splits, every failure kind of tests 17-19, and the invisible-Unicode inputs (the FIX rows and odd inputs excluded). Word classes: NOUN / VERB / ADJ / PRON from the lexicon, function words kept as themselves, anything else UNK (Bogin et al. 2022, 'unobserved local structures').

Cells: share of inputs flagged / accuracy when flagged / accuracy when NOT flagged / share of all failures that are flagged (recall) / share of flagged inputs that fail (precision).

| flag | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|
| unseen word | 56.9% / 17.9% / 66.6% / 76.4% / 82.1% | 56.9% / 20.2% / 63.3% / 74.2% / 79.8% | 56.9% / 4.9% / 12.2% / 58.8% / 95.1% | 56.9% / 18.4% / 62.0% / 73.9% / 81.6% |
| unseen word pair | 91.7% / 36.7% / 63.5% / 95.1% / 63.3% | 91.7% / 36.6% / 63.1% / 95.0% / 63.4% | 91.7% / 7.2% / 17.2% / 92.6% / 92.8% | 91.7% / 35.0% / 61.1% / 94.9% / 65.0% |
| unseen word-class pair | 62.0% / 24.7% / 62.1% / 76.4% / 75.3% | 62.0% / 26.8% / 58.4% / 74.2% / 73.2% | 62.0% / 6.0% / 11.2% / 63.4% / 94.0% | 62.0% / 25.1% / 57.0% / 74.0% / 74.9% |
| unseen word-class triple | 65.0% / 23.2% / 68.0% / 81.7% / 76.8% | 65.0% / 25.0% / 64.4% / 79.7% / 75.0% | 65.0% / 5.1% / 13.4% / 67.1% / 94.9% | 65.0% / 24.2% / 61.4% / 78.5% / 75.8% |

## Per input kind (Option 1: Latin-sound start)

| input kind | n | accuracy | flag: unseen word | flag: unseen word pair | flag: unseen word-class pair | flag: unseen word-class triple |
|---|---|---|---|---|---|---|
| normal | 100 | 100.0% | 0.0% | 64.0% | 0.0% | 0.0% |
| passive voice | 100 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| synonym | 100 | 15.0% | 68.0% | 94.0% | 67.0% | 67.0% |
| UPPER CASE | 100 | 4.0% | 100.0% | 100.0% | 100.0% | 99.0% |
| typo | 100 | 10.0% | 91.0% | 96.0% | 91.0% | 90.0% |
| double spaces | 100 | 1.0% | 0.0% | 64.0% | 0.0% | 0.0% |
| 2 adjectives on a noun | 100 | 2.0% | 0.0% | 100.0% | 0.0% | 100.0% |
| transitive verb without object | 100 | 1.0% | 0.0% | 71.0% | 0.0% | 0.0% |
| test_iid | 100 | 100.0% | 0.0% | 69.0% | 0.0% | 0.0% |
| test_comp | 100 | 98.0% | 0.0% | 76.0% | 0.0% | 0.0% |
| test_morph | 100 | 98.0% | 0.0% | 100.0% | 8.0% | 8.0% |
| test_long | 100 | 99.0% | 0.0% | 100.0% | 0.0% | 0.0% |
| object fronted ('the cat, the dog saw') | 60 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| filler word in front ('okay, ...') | 60 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 'please translate:' in front | 60 | 3.3% | 100.0% | 100.0% | 100.0% | 100.0% |
| this/that/these/those for 'the' | 60 | 3.3% | 100.0% | 100.0% | 100.0% | 100.0% |
| numerals ('two dogs', 'one cat') | 60 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Title Case | 60 | 30.0% | 100.0% | 100.0% | 100.0% | 96.7% |
| full stop and capital | 60 | 100.0% | 100.0% | 100.0% | 100.0% | 96.7% |
| 'is going to' for 'will' | 60 | 60.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| location fronted ('in the X, ...') | 60 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 'did X' for past ('did eat') | 60 | 98.3% | 0.0% | 100.0% | 100.0% | 100.0% |
| 'does X' for present | 60 | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| homoglyph (Cyrillic look-alike letter) | 100 | 27.0% | 100.0% | 100.0% | 100.0% | 96.0% |
| zero-width space inside a word | 100 | 34.0% | 100.0% | 100.0% | 100.0% | 96.0% |
| no-break space between words | 100 | 80.0% | 0.0% | 63.0% | 0.0% | 0.0% |
| full-width letters in one word | 100 | 16.0% | 100.0% | 100.0% | 100.0% | 96.0% |
| soft hyphen inside a word | 100 | 34.0% | 100.0% | 100.0% | 100.0% | 96.0% |
| 3 mixed invisible changes | 100 | 1.0% | 100.0% | 100.0% | 100.0% | 96.0% |
