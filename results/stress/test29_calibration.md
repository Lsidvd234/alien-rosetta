# Test 29: calibration and selective translation

Inputs with a known answer (normal, test splits, failure kinds, rewordings, invisible Unicode; FIX rows excluded). Confidence = probability of the whole output (product of token probabilities) or its lowest token probability. Platt scaling is fitted on even-numbered inputs and evaluated on odd-numbered ones.

| measure | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|
| accuracy | 38.9% | 38.8% | 8.0% | 37.2% |
| ECE, whole-output probability | 0.166 | 0.154 | 0.037 | 0.152 |
| ECE, lowest-token probability | 0.321 | 0.313 | 0.160 | 0.313 |
| ECE after Platt scaling (held-out half) | 0.057 | 0.049 | 0.010 | 0.051 |
| coverage at 99% accuracy (rank by lowest token) | 3.0% | 3.4% | 0.0% | 0.9% |
| coverage at 95% accuracy (rank by lowest token) | 10.2% | 9.2% | 0.0% | 8.6% |
| AURC (lower = better; lowest token) | 0.278 | 0.295 | 0.739 | 0.313 |
| in-distribution only: accuracy | 99.0% | 94.4% | 17.2% | 89.8% |
| in-distribution only: coverage at 99% accuracy | 100.0% | 8.4% | 15.2% | 72.2% |
| in-distribution only: ECE (whole output) | 0.029 | 0.045 | 0.024 | 0.060 |

## Reliability table (Option 1: Latin-sound start, whole-output probability)

| confidence bin | n | mean confidence | accuracy |
|---|---|---|---|
| 0.0-0.1 | 456 | 0.04 | 0.0% |
| 0.1-0.2 | 248 | 0.15 | 0.4% |
| 0.2-0.3 | 190 | 0.25 | 3.2% |
| 0.3-0.4 | 152 | 0.35 | 2.6% |
| 0.4-0.5 | 139 | 0.45 | 6.5% |
| 0.5-0.6 | 95 | 0.55 | 16.8% |
| 0.6-0.7 | 76 | 0.65 | 22.4% |
| 0.7-0.8 | 84 | 0.75 | 27.4% |
| 0.8-0.9 | 94 | 0.86 | 50.0% |
| 0.9-1.0 | 926 | 0.97 | 90.1% |
