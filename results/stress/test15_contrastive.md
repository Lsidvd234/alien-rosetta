# Test 15: does the model prefer the correct translation over a one-thing-wrong version?

About 400 new sentences; each corrupted in every applicable way. Teacher-forced scoring.
Cells: prefers correct / mean log-probability margin (bigger = more certain).

| corruption | n | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start
|---|---|---|---|---|---|
| wrong agreement ending | 72 | 100.0% / 22.9 | 100.0% / 22.0 | 100.0% / 20.6 | 100.0% / 23.8 |
| negation flipped | 72 | 100.0% / 18.0 | 100.0% / 17.5 | 100.0% / 15.6 | 100.0% / 16.5 |
| wrong tense | 72 | 100.0% / 18.1 | 100.0% / 21.0 | 100.0% / 15.4 | 100.0% / 19.3 |
| object case ending missing | 50 | 100.0% / 71.9 | 100.0% / 73.7 | 100.0% / 68.8 | 100.0% / 78.3 |
| object plural missing | 18 | 100.0% / 22.5 | 100.0% / 23.0 | 100.0% / 10.2 | 100.0% / 21.0 |
| case ending put on the subject | 52 | 100.0% / 70.2 | 100.0% / 94.0 | 100.0% / 69.3 | 100.0% / 92.1 |
| sound changes (sandhi) not applied | 59 | 100.0% / 119.6 | 100.0% / 126.5 | 100.0% / 121.6 | 100.0% / 125.7 |
| irregular verb regularised | 8 | 100.0% / 6.9 | 100.0% / 9.8 | 37.5% / 0.4 | 87.5% / 7.7 |
