# Test 10: unusual input

150 cases each, free generation (greedy).

**Two sentences in one input** ('S1. S2.'): which did the output contain?

| model | both, in order | only the first | other |
|---|---|---|---|
| Option 1: Latin-sound start | 0.0% | 2.0% | 98.0% |
| Option 2: + transliteration | 0.0% | 7.3% | 92.7% |
| Option 3a: plain text first | 0.0% | 0.7% | 99.3% |
| Control: shuffled-sound start | 0.0% | 4.0% | 96.0% |

**An unknown English word as the object** (e.g. 'the dog saw the blicket'):

| model | rest of the sentence right | same number of words as expected |
|---|---|---|
| Option 1: Latin-sound start | 99.3% | 100.0% |
| Option 2: + transliteration | 98.0% | 100.0% |
| Option 3a: plain text first | 0.0% | 98.0% |
| Control: shuffled-sound start | 98.7% | 100.0% |
