# Test 11: output behaviour

300 test_iid sentences. Teacher-forced exact = every next token right given the correct prefix;
free generation = what the model actually writes (greedy).

| model | teacher-forced exact | free generation exact | output too long | output too short | reverse direction (glyphs -> English) exact |
|---|---|---|---|---|---|
| Option 1: Latin-sound start | 99.0% | 99.0% | 0.0% | 0.0% | 0.0% |
| Option 2: + transliteration | 98.7% | 98.7% | 0.0% | 0.0% | 0.0% |
| Option 3a: plain text first | 33.0% | 33.0% | 0.0% | 0.3% | 0.0% |
| Control: shuffled-sound start | 97.7% | 97.7% | 0.0% | 0.0% | 0.0% |
