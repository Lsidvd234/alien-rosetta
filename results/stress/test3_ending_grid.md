# Test 3: every ending combination

4500 new sentences: every tense x negation x subject type x object number, ~25 each.
Only words seen in training; no adjectives or locations; no held-out verb+object pairs; none in the training set.
Held out in training = past + negated + plural subject (the test_morph combination).
Scores: whole sentence right (verb word right) — teacher-forced.

## Seen vs held-out combinations

| model | combinations seen in training | held-out (past + neg + plural subject) |
|---|---|---|
| Option 1: Latin-sound start | 99.1% (99.2%) | 95.0% (95.0%) |
| Option 2: + transliteration | 99.4% (99.5%) | 71.7% (71.7%) |
| Option 3a: plain text first | 32.8% (32.8%) | 30.0% (30.0%) |
| Control: shuffled-sound start | 99.0% (99.0%) | 84.7% (84.7%) |

## By tense

| tense | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|
| present | 99.7% (99.8%) | 99.5% (99.6%) | 32.9% (32.9%) | 99.7% (99.7%) |
| past | 97.2% (97.3%) | 93.5% (93.7%) | 31.7% (31.7%) | 94.9% (95.1%) |
| future | 99.5% (99.6%) | 99.5% (99.6%) | 33.3% (33.3%) | 99.4% (99.5%) |

## By neg

| neg | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|
| affirmative | 99.2% (99.3%) | 99.4% (99.6%) | 33.0% (33.0%) | 99.1% (99.2%) |
| negated | 98.4% (98.5%) | 95.6% (95.6%) | 32.2% (32.2%) | 96.9% (96.9%) |

## By subject

| subject | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|
| noun, singular | 99.6% (99.8%) | 99.6% (99.8%) | 32.9% (32.9%) | 99.3% (99.6%) |
| noun, plural | 98.4% (98.4%) | 98.7% (99.1%) | 33.1% (33.1%) | 96.0% (96.0%) |
| pronoun 1sg | 99.6% (99.8%) | 99.8% (99.8%) | 33.3% (33.3%) | 99.3% (99.6%) |
| pronoun 2sg | 98.9% (98.9%) | 99.6% (99.6%) | 32.2% (32.2%) | 99.3% (99.3%) |
| pronoun 3sgm | 98.0% (98.0%) | 99.6% (99.6%) | 32.4% (32.4%) | 97.8% (97.8%) |
| pronoun 3sgf | 99.1% (99.1%) | 99.8% (99.8%) | 32.9% (32.9%) | 98.7% (98.7%) |
| pronoun 3sgn | 99.3% (99.3%) | 99.6% (99.6%) | 31.8% (31.8%) | 99.1% (99.1%) |
| pronoun 1pl | 99.1% (99.3%) | 85.1% (85.3%) | 32.2% (32.2%) | 98.0% (98.2%) |
| pronoun 2pl | 97.6% (97.6%) | 95.8% (95.8%) | 32.9% (32.9%) | 99.1% (99.1%) |
| pronoun 3pl | 98.7% (98.9%) | 97.8% (98.0%) | 32.4% (32.4%) | 93.3% (93.6%) |

## By object

| object | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|
| no object | 99.9% (99.9%) | 97.7% (97.8%) | 97.5% (97.5%) | 98.2% (98.2%) |
| object singular | 97.9% (97.9%) | 97.6% (97.7%) | 0.2% (0.2%) | 98.3% (98.3%) |
| object plural | 98.7% (98.9%) | 97.2% (97.4%) | 0.1% (0.1%) | 97.5% (97.8%) |

## Tense x negation x subject number (whole sentence right)

| tense | negated | subject number | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|---|---|
| present | no | singular | 99.6% | 99.6% | 32.7% | 99.3% |
| present | no | plural | 99.3% | 98.7% | 32.7% | 99.3% |
| present | yes | singular | 100.0% | 99.8% | 33.1% | 100.0% |
| present | yes | plural | 100.0% | 99.7% | 33.0% | 100.0% |
| past | no | singular | 98.4% | 99.8% | 32.9% | 99.1% |
| past | no | plural | 98.0% | 98.0% | 33.0% | 97.3% |
| past | yes | singular | 96.9% | 98.9% | 30.7% | 96.0% |
| past | yes | plural **(held out)** | 95.0% | 71.7% | 30.0% | 84.7% |
| future | no | singular | 100.0% | 100.0% | 33.3% | 99.8% |
| future | no | plural | 99.7% | 100.0% | 33.7% | 99.3% |
| future | yes | singular | 99.6% | 99.8% | 32.9% | 99.3% |
| future | yes | plural | 98.7% | 98.0% | 33.7% | 99.0% |
