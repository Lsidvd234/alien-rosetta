# Test 20: structure

## Every sentence template, with how often it appeared in training

| template | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start
|---|---|---|---|---|
| template: subject noun, object none, - (seen 5287x) | 100.0% | 98.8% | 98.8% | 100.0% |
| template: subject noun, object none, location (seen 2266x) | 100.0% | 97.5% | 97.5% | 100.0% |
| template: subject noun, object noun, - (seen 7154x) | 100.0% | 100.0% | 1.2% | 98.8% |
| template: subject noun, object noun, location (seen 3113x) | 100.0% | 98.8% | 0.0% | 97.5% |
| template: subject noun, object noun+adj, - (seen 3842x) | 100.0% | 98.8% | 0.0% | 98.8% |
| template: subject noun, object noun+adj, location (seen 1659x) | 98.8% | 98.8% | 0.0% | 97.5% |
| template: subject noun+adj, object none, - (seen 2829x) | 98.8% | 100.0% | 92.5% | 98.8% |
| template: subject noun+adj, object none, location (seen 1251x) | 100.0% | 97.5% | 92.5% | 98.8% |
| template: subject noun+adj, object noun, - (seen 3858x) | 98.8% | 98.8% | 0.0% | 100.0% |
| template: subject noun+adj, object noun, location (seen 1721x) | 96.2% | 97.5% | 0.0% | 97.5% |
| template: subject noun+adj, object noun+adj, - (seen 2051x) | 97.5% | 96.2% | 0.0% | 97.5% |
| template: subject noun+adj, object noun+adj, location (seen 0x) | 97.5% | 93.8% | 1.2% | 65.0% |
| template: subject pronoun, object none, - (seen 2957x) | 100.0% | 100.0% | 96.2% | 100.0% |
| template: subject pronoun, object none, location (seen 1463x) | 100.0% | 98.8% | 97.5% | 100.0% |
| template: subject pronoun, object noun, - (seen 4801x) | 100.0% | 100.0% | 1.2% | 100.0% |
| template: subject pronoun, object noun, location (seen 2042x) | 97.5% | 98.8% | 0.0% | 98.8% |
| template: subject pronoun, object noun+adj, - (seen 2580x) | 98.8% | 100.0% | 0.0% | 98.8% |
| template: subject pronoun, object noun+adj, location (seen 1126x) | 97.5% | 100.0% | 0.0% | 98.8% |

## Nouns in a role they never had in training

| case | n | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start
|---|---|---|---|---|---|
| role: noun seen in both roles (control) | 200 | 98.5% | 98.5% | 0.0% | 98.5% |

## Adjective+noun pairs, each in 5 different sentences

Cells: share of sentences right / share of pairs right in ALL 5 sentences.

| pairs | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start
|---|---|---|---|---|
| pair: adjective+noun never seen together (x5 contexts) | 100.0% / 100.0% | 100.0% / 100.0% | 30.5% / 0.0% | 100.0% / 100.0% |
| pair: adjective+noun seen together (x5 contexts) | 99.0% / 95.0% | 99.5% / 97.5% | 31.0% / 0.0% | 98.0% / 90.0% |

## Consistency: change only the subject noun (same agreement) -> only that word may change

Cells: pairs where a word OTHER than the subject also changed in the output.

| | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start
|---|---|---|---|---|
| subject-noun swap | 1.0% | 0.5% | 20.0% | 1.0% |

## Accuracy by output length (tokens), all sentences above

Training targets: median 32 characters, longest 59.

| output tokens | n | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start
|---|---|---|---|---|---|
| 1-15 | 16 | 100.0% | 100.0% | 93.8% | 100.0% |
| 16-25 | 329 | 99.7% | 99.4% | 89.7% | 100.0% |
| 26-35 | 865 | 99.5% | 99.4% | 37.9% | 99.2% |
| 36-50 | 1100 | 98.5% | 98.7% | 7.1% | 97.6% |
| 51-999 | 130 | 97.7% | 96.2% | 0.8% | 80.8% |
