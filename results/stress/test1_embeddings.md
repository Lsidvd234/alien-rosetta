# Test 1: where did the glyph embeddings end up?

Each glyph's embedding (trained, and at the start of training) is compared with the embedding
of the Latin letters for its true sound, taken from Qwen's own table.

- **similarity to own sound**: mean cosine similarity between a glyph and its sound's embedding
- **nearest sound = own sound**: share of glyphs whose most similar sound (among all the script's sounds) is their own
- **sound-structure match**: correlation between how similar glyphs are to each other and how similar their sounds are (1 = the glyphs reproduce the sound layout)
- **moved**: how far training moved the embedding, relative to its starting length

| model | similarity to own sound (start -> end) | nearest sound = own (start -> end) | sound-structure match (start -> end) | moved |
|---|---|---|---|---|
| Option 1: Latin-sound start | 1.00 -> 0.97 | 100% -> 100% | 1.00 -> 0.99 | 27% |
| Option 2: + transliteration | 0.00 -> 0.05 | 0% -> 23% | -0.09 -> 0.16 | 37% |
| Option 3a: plain text first | 0.00 -> 0.02 | 0% -> 4% | -0.09 -> 0.04 | 28% |
| Control: shuffled-sound start | 0.15 -> 0.15 | 0% -> 0% | -0.06 -> 0.01 | 28% |

## Closest ordinary tokens to some trained glyphs

| model | glyph's true sound | 5 closest tokens in Qwen's vocabulary |
|---|---|---|
| Option 1: Latin-sound start | k | 'k', 'k', 'K', '_k', 'k' |
| Option 1: Latin-sound start | tx | 'tx', 'tx', 'tx', 'TX', '_tx' |
| Option 1: Latin-sound start | p | 'p', 'p', '_p', '.p', 'P' |
| Option 1: Latin-sound start | l | 'l', 'l', '_l', 'L', '(l' |
| Option 1: Latin-sound start | rx | 'rx', 'rx', '(rx', '_rx', '.rx' |
| Option 1: Latin-sound start | u | 'u', 'u', 'U', 'u', 'uC' |
| Option 2: + transliteration | k | 'Extra', 'Extra', 'extra', 'extra', '_extra' |
| Option 2: + transliteration | tx | 'licht', 'Licht', 'light', 'LIGHT', '灯光' |
| Option 2: + transliteration | p | 'imageSize', 'imageName', 'imageURL', 'getImage', 'imageName' |
| Option 2: + transliteration | l | 'rem', 'Rem', 'Rem', 'rem', '_rem' |
| Option 2: + transliteration | rx | 'opers', 'oper', 'OPER', 'ops', 'Oper' |
| Option 2: + transliteration | u | 'lahoma', 'ahoma', 'Oklahoma', '죌', '뭬' |
| Option 3a: plain text first | k | 'Extra', 'Extra', 'extra', 'extra', '_extra' |
| Option 3a: plain text first | tx | 'licht', 'Licht', 'light', 'LIGHT', 'lights' |
| Option 3a: plain text first | p | 'imageSize', 'imageName', 'image', 'imageData', 'imageURL' |
| Option 3a: plain text first | l | 'rem', 'Rem', 'Rem', 'rem', '_rem' |
| Option 3a: plain text first | rx | 'opers', 'oper', 'OPER', 'ops', 'oper' |
| Option 3a: plain text first | u | 'lahoma', 'ahoma', 'Oklahoma', '죌', '뭬' |
| Control: shuffled-sound start | k | 'oo', 'ooo', 'oooo', 'OO', 'oo' |
| Control: shuffled-sound start | tx | 'c', 'c', '_c', 'c', 'C' |
| Control: shuffled-sound start | p | 'r', 'r', 'r', '_r', '(r' |
| Control: shuffled-sound start | l | 'y', 'y', 'yb', 'yw', 'yh' |
| Control: shuffled-sound start | rx | 'v', 'v', 'V', '_v', 'v' |
| Control: shuffled-sound start | u | 'p', 'p', '_p', '.p', 'p' |
