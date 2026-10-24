# Test 18: what kind of error does each model make?

150 sentences per input kind, free generation; one label per output.

## Option 1: Latin-sound start

| input kind | correct | word(s) dropped | extra word(s) | right stem, wrong ending | wrong word (stem) | verb not last (word order changed) | right words, wrong order | empty output |
|---|---|---|---|---|---|---|---|---|
| normal | 99.3% | 0.0% | 0.0% | 0.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| passive voice | 0.0% | 7.3% | 3.3% | 8.7% | 80.7% | 0.0% | 0.0% | 0.0% |
| synonym | 9.3% | 0.0% | 0.0% | 0.0% | 90.7% | 0.0% | 0.0% | 0.0% |
| UPPER CASE | 7.3% | 0.7% | 0.0% | 0.0% | 92.0% | 0.0% | 0.0% | 0.0% |
| typo | 9.3% | 0.0% | 0.0% | 0.0% | 90.7% | 0.0% | 0.0% | 0.0% |
| double spaces | 3.3% | 0.7% | 92.0% | 0.0% | 4.0% | 0.0% | 0.0% | 0.0% |
| 2 adjectives on a noun | 8.0% | 74.7% | 0.0% | 2.0% | 15.3% | 0.0% | 0.0% | 0.0% |
| transitive verb without object | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| 'Input/Output' prompt labels | 91.3% | 0.0% | 0.0% | 2.0% | 6.7% | 0.0% | 0.0% | 0.0% |
| two sentences at once | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Option 2: + transliteration

| input kind | correct | word(s) dropped | extra word(s) | right stem, wrong ending | wrong word (stem) | verb not last (word order changed) | right words, wrong order | empty output |
|---|---|---|---|---|---|---|---|---|
| normal | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| passive voice | 0.0% | 16.7% | 2.0% | 39.3% | 42.0% | 0.0% | 0.0% | 0.0% |
| synonym | 4.0% | 0.0% | 0.0% | 0.7% | 95.3% | 0.0% | 0.0% | 0.0% |
| UPPER CASE | 4.0% | 0.7% | 0.0% | 0.0% | 95.3% | 0.0% | 0.0% | 0.0% |
| typo | 10.7% | 0.7% | 0.7% | 0.7% | 87.3% | 0.0% | 0.0% | 0.0% |
| double spaces | 2.7% | 43.3% | 16.0% | 0.7% | 37.3% | 0.0% | 0.0% | 0.0% |
| 2 adjectives on a noun | 0.0% | 91.3% | 0.0% | 0.0% | 8.7% | 0.0% | 0.0% | 0.0% |
| transitive verb without object | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| 'Input/Output' prompt labels | 34.0% | 0.7% | 0.0% | 10.0% | 55.3% | 0.0% | 0.0% | 0.0% |
| two sentences at once | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Option 3a: plain text first

| input kind | correct | word(s) dropped | extra word(s) | right stem, wrong ending | wrong word (stem) | verb not last (word order changed) | right words, wrong order | empty output |
|---|---|---|---|---|---|---|---|---|
| normal | 32.7% | 0.0% | 0.0% | 0.0% | 67.3% | 0.0% | 0.0% | 0.0% |
| passive voice | 0.0% | 14.7% | 4.0% | 0.0% | 81.3% | 0.0% | 0.0% | 0.0% |
| synonym | 0.0% | 1.3% | 0.7% | 0.7% | 97.3% | 0.0% | 0.0% | 0.0% |
| UPPER CASE | 1.3% | 12.7% | 6.7% | 0.7% | 78.7% | 0.0% | 0.0% | 0.0% |
| typo | 3.3% | 2.7% | 3.3% | 0.0% | 90.7% | 0.0% | 0.0% | 0.0% |
| double spaces | 0.0% | 92.0% | 0.0% | 0.0% | 8.0% | 0.0% | 0.0% | 0.0% |
| 2 adjectives on a noun | 0.0% | 74.0% | 0.0% | 0.0% | 26.0% | 0.0% | 0.0% | 0.0% |
| transitive verb without object | 0.0% | 0.0% | 1.3% | 0.0% | 98.7% | 0.0% | 0.0% | 0.0% |
| 'Input/Output' prompt labels | 19.3% | 0.7% | 0.0% | 4.7% | 75.3% | 0.0% | 0.0% | 0.0% |
| two sentences at once | 0.0% | 99.3% | 0.0% | 0.0% | 0.7% | 0.0% | 0.0% | 0.0% |

## Control: shuffled-sound start

| input kind | correct | word(s) dropped | extra word(s) | right stem, wrong ending | wrong word (stem) | verb not last (word order changed) | right words, wrong order | empty output |
|---|---|---|---|---|---|---|---|---|
| normal | 99.3% | 0.0% | 0.0% | 0.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| passive voice | 0.0% | 7.3% | 24.7% | 10.7% | 57.3% | 0.0% | 0.0% | 0.0% |
| synonym | 2.0% | 0.0% | 0.0% | 2.0% | 96.0% | 0.0% | 0.0% | 0.0% |
| UPPER CASE | 8.0% | 2.0% | 0.0% | 1.3% | 88.7% | 0.0% | 0.0% | 0.0% |
| typo | 8.0% | 0.7% | 0.0% | 0.0% | 91.3% | 0.0% | 0.0% | 0.0% |
| double spaces | 2.0% | 1.3% | 92.0% | 0.0% | 4.7% | 0.0% | 0.0% | 0.0% |
| 2 adjectives on a noun | 17.3% | 42.7% | 0.0% | 6.0% | 34.0% | 0.0% | 0.0% | 0.0% |
| transitive verb without object | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| 'Input/Output' prompt labels | 34.0% | 0.0% | 0.0% | 2.0% | 64.0% | 0.0% | 0.0% | 0.0% |
| two sentences at once | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

