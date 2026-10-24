# Test 2: content-word accuracy by how often the word appeared in training

All four test splits pooled. A content word counts as right only if every token of it is right
(teacher-forced). Frequency = number of training sentences (of the 50k) containing the word's stem.

| model | seen 1-15x | seen 16-25x | seen 26-35x | seen 36-45x | seen 46-60x | seen 61+x |
|---|---|---|---|---|---|---|
| Glyphs, random-word start (before) | 1.1% | 6.3% | 84.0% | 97.2% | 98.5% | 96.1% |
| Option 1: Latin-sound start | 99.8% | 99.5% | 99.0% | 99.5% | 99.7% | 99.4% |
| Option 2: + transliteration | 96.7% | 98.2% | 97.3% | 98.8% | 99.3% | 95.6% |
| Option 3a: plain text first | 79.3% | 84.0% | 24.7% | 70.5% | 94.8% | 87.8% |
| Control: shuffled-sound start | 89.7% | 91.4% | 96.3% | 98.4% | 97.3% | 96.1% |
| Latin script | 98.7% | 99.2% | 95.3% | 98.2% | 98.6% | 95.6% |
| Tamil script | 98.9% | 99.3% | 97.4% | 99.3% | 99.3% | 98.3% |

Test words per bucket: 1-15x: 458, 16-25x: 1263, 26-35x: 1164, 36-45x: 2489, 46-60x: 2350, 61+x: 181
