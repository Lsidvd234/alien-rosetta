# Test 9: hard word forms

Up to 300 new sentences per case. Irregular verbs have their own past stem (rarer in training).
Sound changes = places where joining a stem and an ending changes a sound (sandhi).

| case | n | example | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|---|---|
| past tense, irregular verb | 300 | it swore | 96.0% | 98.0% | 14.3% | 94.3% |
| past tense, regular verb | 300 | we fled | 99.3% | 99.3% | 24.7% | 99.0% |
| no sound change at morpheme joins | 300 | we aimed | 99.3% | 100.0% | 42.3% | 99.3% |
| 1 sound change | 300 | he arose | 99.7% | 99.3% | 43.0% | 99.3% |
| 2+ sound changes | 300 | she will glow | 99.0% | 99.3% | 28.0% | 99.0% |
