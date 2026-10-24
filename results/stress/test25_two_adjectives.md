# Test 25: two adjectives on a noun - which translation does the model prefer?

200 sentences; the subject has 2 adjectives (never in training). Each candidate is scored (teacher-forced log-probability) and the greedy output is matched against the candidates.

## Highest-scoring candidate

| candidate | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|
| both adjectives, English order (assumed gold) | 21.5% | 0.0% | 6.5% | 42.5% |
| both adjectives, reversed order | 0.5% | 0.0% | 0.0% | 0.0% |
| first adjective only | 3.5% | 98.5% | 79.5% | 10.5% |
| second adjective only | 0.0% | 1.0% | 0.0% | 0.0% |
| no adjective | 0.0% | 0.0% | 0.0% | 0.0% |
| both adjectives, noun dropped | 74.5% | 0.5% | 14.0% | 47.0% |

## What the model actually wrote

| output equals | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|
| both adjectives, English order (assumed gold) | 9.0% | 0.0% | 0.0% | 14.5% |
| both adjectives, reversed order | 0.0% | 0.0% | 0.0% | 0.0% |
| first adjective only | 0.5% | 66.5% | 3.5% | 0.5% |
| second adjective only | 0.0% | 0.5% | 0.0% | 0.0% |
| no adjective | 0.0% | 0.0% | 0.0% | 0.0% |
| both adjectives, noun dropped | 20.5% | 0.0% | 0.0% | 9.5% |
| none of these | 70.0% | 33.0% | 96.5% | 75.5% |
