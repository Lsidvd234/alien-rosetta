# Test 17: search errors vs model errors, beam search, end of sentence, confidence

120 sentences per kind.

## Greedy vs beam-5 accuracy; share of wrong outputs that are SEARCH errors (the model actually scores the correct translation higher than what it wrote)

| input kind | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start
|---|---|---|---|---|
| normal | 99.2% -> 99.2% / search err 0.0% | 99.2% -> 99.2% / search err 0.0% | 28.3% -> 28.3% / search err 9.3% | 100.0% -> 100.0% / search err - |
| passive voice | 0.0% -> 0.0% / search err 0.0% | 0.0% -> 0.0% / search err 0.0% | 0.0% -> 0.0% / search err 0.0% | 0.0% -> 0.0% / search err 0.0% |
| synonym | 15.0% -> 15.0% / search err 0.0% | 7.5% -> 6.7% / search err 0.9% | 0.8% -> 0.8% / search err 2.5% | 1.7% -> 1.7% / search err 0.0% |
| UPPER CASE | 9.2% -> 10.0% / search err 0.9% | 4.2% -> 4.2% / search err 1.7% | 1.7% -> 1.7% / search err 0.8% | 7.5% -> 10.8% / search err 2.7% |
| typo | 7.5% -> 7.5% / search err 0.0% | 8.3% -> 8.3% / search err 0.9% | 2.5% -> 3.3% / search err 4.3% | 6.7% -> 7.5% / search err 0.9% |
| double spaces | 6.7% -> 6.7% / search err 1.8% | 3.3% -> 3.3% / search err 0.0% | 0.0% -> 0.0% / search err 0.0% | 4.2% -> 4.2% / search err 0.0% |
| 2 adjectives on a noun | 9.2% -> 11.7% / search err 3.7% | 0.0% -> 0.0% / search err 0.0% | 0.0% -> 0.0% / search err 0.0% | 10.0% -> 11.7% / search err 2.8% |
| transitive verb without object | 0.0% -> 0.0% / search err 0.0% | 0.8% -> 0.0% / search err 0.0% | 0.0% -> 0.0% / search err 0.0% | 0.0% -> 0.0% / search err 0.0% |
| 'Input/Output' prompt labels | 90.0% -> 91.7% / search err 16.7% | 33.3% -> 36.7% / search err 5.0% | 20.0% -> 20.8% / search err 11.5% | 27.5% -> 31.7% / search err 8.0% |

## Confidence in its OWN output (lowest token probability of what it wrote)

Separation = chance a correct output gets higher confidence than a wrong one (0.5 = useless, 1 = perfect).

| model | separation (all kinds pooled) | accuracy |
|---|---|---|
| Option 1: Latin-sound start | 0.93 | 26.3% |
| Option 2: + transliteration | 0.89 | 17.4% |
| Option 3a: plain text first | 0.98 | 5.9% |
| Control: shuffled-sound start | 0.93 | 17.5% |

| input kind: mean confidence of WRONG outputs | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|
| normal | 0.54 | 0.57 | 0.11 | nan |
| passive voice | 0.49 | 0.65 | 0.22 | 0.47 |
| synonym | 0.69 | 0.62 | 0.17 | 0.65 |
| UPPER CASE | 0.49 | 0.41 | 0.19 | 0.43 |
| typo | 0.52 | 0.56 | 0.20 | 0.53 |
| double spaces | 0.47 | 0.40 | 0.21 | 0.48 |
| 2 adjectives on a noun | 0.63 | 0.78 | 0.22 | 0.57 |
| transitive verb without object | 0.59 | 0.60 | 0.53 | 0.58 |
| 'Input/Output' prompt labels | 0.52 | 0.47 | 0.16 | 0.44 |

## End of sentence: probability of stopping at the right place / share with a >10% chance of stopping too early

| input kind | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start
|---|---|---|---|---|
| normal | 1.00 / 0.0% | 1.00 / 0.0% | 1.00 / 0.0% | 1.00 / 0.0% |
| passive voice | 0.95 / 0.8% | 0.95 / 0.0% | 1.00 / 0.0% | 0.95 / 1.7% |
| synonym | 1.00 / 0.0% | 1.00 / 0.0% | 1.00 / 0.0% | 1.00 / 0.0% |
| UPPER CASE | 1.00 / 0.0% | 1.00 / 0.0% | 1.00 / 0.0% | 1.00 / 0.0% |
| typo | 1.00 / 0.0% | 1.00 / 0.0% | 1.00 / 0.0% | 1.00 / 0.0% |
| double spaces | 0.91 / 15.0% | 1.00 / 32.5% | 0.99 / 19.2% | 0.96 / 13.3% |
| 2 adjectives on a noun | 1.00 / 4.2% | 1.00 / 5.0% | 1.00 / 0.0% | 1.00 / 2.5% |
| transitive verb without object | 1.00 / 0.0% | 1.00 / 0.0% | 0.99 / 0.0% | 1.00 / 0.0% |
| 'Input/Output' prompt labels | 1.00 / 0.0% | 1.00 / 0.8% | 1.00 / 0.0% | 0.98 / 5.0% |
