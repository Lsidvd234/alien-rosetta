# Real-English test 2: 600 held-out Tatoeba sentences (`analysis/score_tatoeba.py`)

Gold labels from the sealed-set reader (label precision ~88% by hand check), so ~0.88 meaning is the ceiling.

| run | glyphs exact | meaning exact | glyph word accuracy |
|---|---|---|---|
| e2a-v2-80k | 0.438 | - | 0.774 |
| e2b-v2-cot-80k-noheld | 0.462 | 0.620 | 0.792 |
| e2b-v2-cot-80k | 0.458 | 0.622 | 0.806 |
| e2b-v2fix-cot-80k-noheld | 0.467 | 0.617 | 0.798 |
| e2b-v2fix-cot-80k | 0.503 | 0.633 | 0.813 |
| e2b-v2fix-focal-r2 | 0.455 | 0.588 | 0.795 |
| e2b-v2fix-loan-r2 | 0.502 | 0.592 | 0.817 |
| e2b-v2fix-para | 0.477 | 0.615 | 0.810 |
| e2b-v2fix-para8 | 0.480 | 0.620 | 0.807 |
