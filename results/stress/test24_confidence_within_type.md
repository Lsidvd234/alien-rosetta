# Test 24: own-output confidence within each input type

Confidence = lowest token probability of what the model wrote. AUROC = chance a right output gets higher confidence than a wrong one (0.5 useless, 1 perfect; '-' = all right or all wrong). Threshold = the level that keeps 95% of the correct NORMAL outputs.

| | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|
| pooled AUROC (min / mean token) | 0.93 / 0.94 | 0.88 / 0.88 | 0.97 / 0.97 | 0.91 / 0.92 |
| threshold (token probability) | 0.98 | 0.92 | 0.64 | 0.93 |

Cells: accuracy / within-type AUROC (min token) / share of WRONG outputs flagged / accuracy of the outputs NOT flagged.

| input kind | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|
| normal | 98.7% / 0.96 / 50.0% / 99.3% | 98.7% / 0.97 / 50.0% / 99.3% | 37.3% / 0.99 / 98.9% / 98.2% | 98.0% / 0.96 / 66.7% / 99.3% |
| passive voice | 0.0% / - / 99.3% / 0.0% | 0.0% / - / 80.7% / 0.0% | 0.0% / - / 96.0% / 0.0% | 0.0% / - / 97.3% / 0.0% |
| synonym | 12.0% / 0.63 / 67.4% / 14.0% | 6.7% / 0.42 / 69.3% / 0.0% | 0.0% / - / 93.3% / 0.0% | 0.7% / 0.11 / 71.8% / 0.0% |
| UPPER CASE | 11.3% / 0.92 / 100.0% / 100.0% | 4.7% / 0.82 / 98.6% / 60.0% | 2.7% / 0.97 / 97.9% / 50.0% | 9.3% / 0.86 / 100.0% / 100.0% |
| typo | 4.0% / 0.92 / 98.6% / 66.7% | 4.0% / 0.86 / 97.9% / 50.0% | 2.0% / 0.93 / 90.5% / 12.5% | 4.0% / 0.99 / 97.2% / 55.6% |
| double spaces | 7.3% / 0.98 / 99.3% / 87.5% | 4.7% / 0.96 / 100.0% / 100.0% | 0.0% / - / 100.0% / - | 4.0% / 1.00 / 95.8% / 50.0% |
| 2 adjectives on a noun | 9.3% / 0.68 / 97.8% / 0.0% | 0.0% / - / 64.7% / 0.0% | 0.0% / - / 96.0% / 0.0% | 12.7% / 0.90 / 96.9% / 73.3% |
| transitive verb without object | 0.0% / - / 99.3% / 0.0% | 0.7% / 0.43 / 94.0% / 0.0% | 0.0% / - / 80.7% / 0.0% | 0.0% / - / 95.3% / 0.0% |
| 'Input/Output' prompt labels | 90.7% / 0.96 / 100.0% / 100.0% | 34.7% / 0.87 / 100.0% / 100.0% | 24.7% / 0.94 / 89.4% / 73.3% | 36.0% / 0.87 / 100.0% / 100.0% |
