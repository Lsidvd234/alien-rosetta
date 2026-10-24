# Test 14: change one thing in the English, check the translation changes accordingly

120 sentence pairs per feature (both new, valid). Among pairs where the first sentence was translated right:
cells = changed sentence also right / output did NOT change at all (feature ignored) / something OUTSIDE the expected word changed.

| feature changed | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start
|---|---|---|---|---|
| tense | 99.2% / 0.0% / 0.0% | 99.2% / 0.0% / 0.0% | 97.6% / 0.0% / 2.4% | 99.2% / 0.0% / 0.0% |
| negation | 99.2% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% | 99.2% / 0.0% / 0.0% |
| location added/removed | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% |
| subject number | 100.0% / 0.0% / 0.0% | 99.2% / 0.0% / 0.8% | 97.6% / 0.0% / 2.4% | 100.0% / 0.0% / 0.0% |
| subject noun (same number) | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% | 97.7% / 0.0% / 2.3% | 99.2% / 0.0% / 0.8% |
| adjective added to subject | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% | 95.3% / 0.0% / 0.0% | 99.2% / 0.0% / 0.0% |
| object number | 99.2% / 0.0% / 0.0% | 99.2% / 0.0% / 0.0% | 0.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% |
| pronoun person/number | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% |
| pronoun gender (he/she/it) | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% |
