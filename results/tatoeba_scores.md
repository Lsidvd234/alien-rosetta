# Real-English test 2: 600 held-out Tatoeba sentences (`analysis/score_tatoeba.py`)

Gold labels from the sealed-set reader (label precision ~88% by hand check), so ~0.88 meaning is the ceiling.

| run | glyphs exact | meaning exact | glyph word accuracy |
|---|---|---|---|
| e2a-v2-80k | 0.438 | - | 0.774 |
| e2b-v2-cot-80k-noheld | 0.462 | 0.620 | 0.792 |
| e2b-v2-cot-80k | 0.458 | 0.622 | 0.806 |
| e2b-v2fix-6ep | 0.450 | 0.563 | 0.776 |
| e2b-v2fix-cot-80k-noheld | 0.467 | 0.617 | 0.798 |
| e2b-v2fix-cot-80k-noheld@epoch1 | 0.292 | 0.577 | 0.714 |
| e2b-v2fix-cot-80k-noheld@epoch2 | 0.420 | 0.612 | 0.779 |
| e2b-v2fix-cot-80k | 0.503 | 0.633 | 0.813 |
| e2b-v2fix-everyday | 0.605 | 0.725 | 0.869 |
| e2b-v2fix-floor25 | 0.483 | 0.603 | 0.814 |
| e2b-v2fix-focal-r2 | 0.455 | 0.588 | 0.795 |
| e2b-v2fix-loan-r2 | 0.502 | 0.592 | 0.817 |
| e2b-v2fix-loan12 | 0.477 | 0.612 | 0.805 |
| e2b-v2fix-n98k | 0.488 | 0.610 | 0.812 |
| e2b-v2fix-para | 0.477 | 0.615 | 0.810 |
| e2b-v2fix-para8 | 0.480 | 0.620 | 0.807 |
| e2b-v2fix-passive0-meaning | 0.000 | 0.603 | 0.000 |
| e2b-v2fix-passive0-ts1 | 0.468 | 0.597 | 0.803 |
| e2b-v2fix-passive0 | 0.478 | 0.603 | 0.809 |
| e2b-v2fix-passive300 | 0.475 | 0.612 | 0.812 |
| e2b-v2fix-passive50-ts1 | 0.473 | 0.607 | 0.806 |
| e2b-v2fix-passive50 | 0.437 | 0.607 | 0.789 |
| e2b-v2fix-q17 | 0.428 | 0.625 | 0.792 |
| e2b-v2fix-real15-ts1 | 0.892 | 0.928 | 0.965 |
| e2b-v2fix-real15 | 0.890 | 0.923 | 0.963 |
| e2b-v2fix-soundrule | 0.500 | 0.577 | 0.827 |
| e2b-v2fix-ts1 | 0.467 | 0.625 | 0.809 |
