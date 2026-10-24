# Maths notation extension (charter claim 1)

Base-7 alien digits + operator glyphs as new tokens, Qwen3-0.6B-Base + LoRA r=16, 60k mixed examples, 2 epochs (train/maths.py, Kaggle P100, seed 0, 300 test items per row). Glyph tokens start at: **meaning** = the real digit/operator's embedding; **shuffled** = a WRONG digit's; **mean** = no meaning. Exact match (digit accuracy).

| test | meaning | shuffled | mean |
|---|---|---|---|
| to: 1-4 digits (trained lengths) | 0.397 (0.70) | 0.033 (0.56) | 0.043 (0.57) |
| to: 5 digits | 0.000 (0.14) | 0.000 (0.12) | 0.000 (0.11) |
| from: 1-4 digits (trained lengths) | 0.407 (0.72) | 0.053 (0.56) | 0.137 (0.59) |
| from: 5 digits | 0.000 (0.13) | 0.000 (0.10) | 0.000 (0.08) |
| add: up to 3+3 (trained lengths) | 0.977 (0.99) | 0.860 (0.93) | 0.873 (0.95) |
| add: 4+4 | 0.887 (0.97) | 0.083 (0.40) | 0.020 (0.35) |
| add: 5+5 | 0.723 (0.93) | 0.020 (0.32) | 0.000 (0.23) |
| mul: up to 2x2 (trained lengths) | 0.957 (0.98) | 0.690 (0.86) | 0.623 (0.84) |
| mul: 3x1 | 0.653 (0.87) | 0.363 (0.54) | 0.320 (0.50) |
| mul: 3x2 | 0.130 (0.50) | 0.000 (0.25) | 0.020 (0.20) |
| mul: 3x3 | 0.027 (0.38) | 0.000 (0.18) | 0.000 (0.17) |

| | meaning | shuffled | mean |
|---|---|---|---|
| train minutes | 18.3 | 17.8 | 19.3 |
| final loss | 0.0191 | 0.0853 | 0.1479 |
