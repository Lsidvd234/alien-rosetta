# Test 7: who did what

300 new sentences per case, only words seen in training, none in the training set.
Training never used a transitive verb without an object (the generator always gave it one).

| case | n | example | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|---|---|
| subject and object of a training sentence swapped | 300 | the rotor added the hips | 99.3% | 100.0% | 0.7% | 99.3% |
| verb that always had an object, used without one | 300 | I smack | 0.0% | 0.0% | 0.0% | 0.3% |
| same noun as subject and object | 300 | the cups slam the cup | 97.7% | 98.7% | 0.0% | 97.7% |
| control: ordinary new sentences | 300 | I stew | 99.0% | 98.7% | 27.7% | 98.3% |
