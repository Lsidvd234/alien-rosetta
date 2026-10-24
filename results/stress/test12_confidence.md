# Test 12: does the model know when it is wrong?

200 sentences per kind. Confidence = the model's lowest probability on any token of the correct translation (teacher-forced).
**Separation** = chance that a correct sentence gets higher confidence than a wrong one (0.5 = no idea, 1.0 = perfect warning signal).

| model | separation (all kinds pooled) | normal new sentences: right / mean confidence | passive voice: right / mean confidence | synonym: right / mean confidence | UPPER CASE: right / mean confidence | 2 adjectives on a noun: right / mean confidence |
|---|---|---|---|---|---|---|
| Option 1: Latin-sound start | 1.00 | 98% / 0.98 | 0% / 0.00 | 10% / 0.10 | 8% / 0.09 | 7% / 0.07 |
| Option 2: + transliteration | 1.00 | 100% / 0.99 | 0% / 0.00 | 6% / 0.06 | 7% / 0.06 | 0% / 0.00 |
| Option 3a: plain text first | 1.00 | 34% / 0.35 | 0% / 0.00 | 0% / 0.02 | 1% / 0.02 | 0% / 0.00 |
| Control: shuffled-sound start | 1.00 | 99% / 0.97 | 0% / 0.00 | 2% / 0.03 | 8% / 0.07 | 14% / 0.12 |
