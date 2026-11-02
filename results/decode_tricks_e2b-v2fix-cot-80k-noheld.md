# Decoding tricks on Tatoeba (`analysis/decode_tricks.py`), run e2b-v2fix-cot-80k-noheld, 600 sentences

| decoding | meaning exact | glyphs exact | vs greedy (meaning): fixes / breaks | McNemar p |
|---|---|---|---|---|
| greedy | 0.617 | 0.467 | - | - |
| beam 4 | 0.618 | 0.472 | 5 / 4 | 1 |
| vote 5 | 0.622 | 0.433 | 6 / 3 | 0.51 |

Greedy reproduces the saved Tatoeba predictions on 596 of 600 sentences.
