# Test 28: can the INPUT alone tell that the English is off-template? (Mahalanobis distance)

Hidden states of the prompt (mean over its tokens) at the middle and the last layer; a Gaussian is fitted on 2,000 training inputs. AUROC = chance an input of this kind is further from the training inputs than an in-distribution one (normal + the 4 test splits). 1.0 = perfectly separated, 0.5 = invisible.

| input kind | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|
| passive voice | 0.93 / 0.80 (acc 0.0%) | 0.95 / 0.80 (acc 0.0%) | 0.93 / 0.82 (acc 0.0%) | 0.96 / 0.79 (acc 0.0%) |
| synonym | 0.54 / 0.61 (acc 15.0%) | 0.52 / 0.63 (acc 7.0%) | 0.57 / 0.61 (acc 2.0%) | 0.52 / 0.62 (acc 2.0%) |
| UPPER CASE | 1.00 / 1.00 (acc 4.0%) | 1.00 / 1.00 (acc 6.0%) | 1.00 / 1.00 (acc 1.0%) | 1.00 / 1.00 (acc 5.0%) |
| typo | 0.78 / 0.73 (acc 10.0%) | 0.74 / 0.66 (acc 13.0%) | 0.80 / 0.74 (acc 1.0%) | 0.82 / 0.67 (acc 10.0%) |
| double spaces | 1.00 / 1.00 (acc 1.0%) | 1.00 / 1.00 (acc 1.0%) | 1.00 / 1.00 (acc 0.0%) | 1.00 / 0.99 (acc 1.0%) |
| 2 adjectives on a noun | 0.82 / 0.74 (acc 2.0%) | 0.90 / 0.68 (acc 0.0%) | 0.88 / 0.74 (acc 0.0%) | 0.85 / 0.67 (acc 11.0%) |
| transitive verb without object | 0.46 / 0.55 (acc 1.0%) | 0.59 / 0.59 (acc 0.0%) | 0.43 / 0.59 (acc 0.0%) | 0.52 / 0.55 (acc 0.0%) |
| object fronted ('the cat, the dog saw') | 0.96 / 0.82 (acc 0.0%) | 0.99 / 0.88 (acc 0.0%) | 0.94 / 0.91 (acc 0.0%) | 0.98 / 0.65 (acc 0.0%) |
| filler word in front ('okay, ...') | 0.93 / 0.89 (acc 0.0%) | 0.99 / 0.76 (acc 15.0%) | 0.95 / 0.82 (acc 0.0%) | 0.97 / 0.82 (acc 0.0%) |
| 'please translate:' in front | 0.97 / 0.83 (acc 3.3%) | 1.00 / 0.90 (acc 8.3%) | 0.95 / 0.66 (acc 8.3%) | 1.00 / 0.67 (acc 13.3%) |
| this/that/these/those for 'the' | 0.96 / 0.82 (acc 3.3%) | 0.98 / 0.81 (acc 11.7%) | 0.97 / 0.85 (acc 0.0%) | 0.97 / 0.69 (acc 13.3%) |
| numerals ('two dogs', 'one cat') | 0.97 / 0.89 (acc 0.0%) | 1.00 / 0.85 (acc 0.0%) | 0.99 / 0.92 (acc 0.0%) | 0.99 / 0.75 (acc 1.7%) |
| Title Case | 1.00 / 0.99 (acc 30.0%) | 1.00 / 0.89 (acc 30.0%) | 1.00 / 0.95 (acc 11.7%) | 1.00 / 0.96 (acc 30.0%) |
| full stop and capital | 0.78 / 0.61 (acc 100.0%) | 0.81 / 0.61 (acc 100.0%) | 0.79 / 0.69 (acc 26.7%) | 0.74 / 0.58 (acc 98.3%) |
| 'is going to' for 'will' | 0.86 / 0.69 (acc 60.0%) | 0.93 / 0.79 (acc 68.3%) | 0.91 / 0.87 (acc 0.0%) | 0.89 / 0.64 (acc 53.3%) |
| location fronted ('in the X, ...') | 0.96 / 0.65 (acc 0.0%) | 0.99 / 0.73 (acc 5.0%) | 0.96 / 0.86 (acc 0.0%) | 0.98 / 0.45 (acc 0.0%) |
| 'did X' for past ('did eat') | 0.52 / 0.51 (acc 98.3%) | 0.53 / 0.56 (acc 98.3%) | 0.50 / 0.60 (acc 21.7%) | 0.55 / 0.53 (acc 96.7%) |
| 'does X' for present | 0.52 / 0.54 (acc 100.0%) | 0.53 / 0.52 (acc 100.0%) | 0.52 / 0.60 (acc 16.7%) | 0.52 / 0.53 (acc 100.0%) |
| homoglyph (Cyrillic look-alike letter) | 0.95 / 0.82 (acc 27.0%) | 0.97 / 0.74 (acc 29.0%) | 0.95 / 0.81 (acc 7.0%) | 0.98 / 0.77 (acc 25.0%) |
| zero-width space inside a word | 0.96 / 0.81 (acc 34.0%) | 0.98 / 0.71 (acc 43.0%) | 0.98 / 0.80 (acc 12.0%) | 0.99 / 0.77 (acc 33.0%) |
| no-break space between words | 0.84 / 0.65 (acc 80.0%) | 0.90 / 0.60 (acc 71.0%) | 0.89 / 0.62 (acc 19.0%) | 0.90 / 0.61 (acc 71.0%) |
| full-width letters in one word | 1.00 / 0.98 (acc 16.0%) | 1.00 / 0.95 (acc 5.0%) | 1.00 / 0.98 (acc 4.0%) | 1.00 / 0.95 (acc 19.0%) |
| soft hyphen inside a word | 0.96 / 0.84 (acc 34.0%) | 0.97 / 0.76 (acc 43.0%) | 0.99 / 0.83 (acc 13.0%) | 0.98 / 0.78 (acc 45.0%) |
| 3 mixed invisible changes | 1.00 / 0.98 (acc 1.0%) | 1.00 / 0.93 (acc 2.0%) | 1.00 / 0.97 (acc 1.0%) | 1.00 / 0.96 (acc 0.0%) |
| odd inputs | 0.98 / 0.98 | 1.00 / 0.96 | 0.99 / 0.97 | 0.99 / 0.87 |

Cells: AUROC middle layer / last layer (model accuracy on that kind).

## Predicting a WRONG output (all inputs with a known answer, FIX rows excluded)

| signal | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|
| input distance, middle layer | 0.80 | 0.80 | 0.64 | 0.78 |
| input distance, last layer | 0.76 | 0.73 | 0.57 | 0.70 |
| output confidence (lowest token prob), inverted | 0.95 | 0.93 | 0.98 | 0.93 |
| input distance (middle) + output confidence, rank average | 0.93 | 0.91 | 0.89 | 0.91 |
