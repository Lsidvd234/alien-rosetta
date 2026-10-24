# Learning curves: 2x2 (grammar x script), Qwen3-0.6B-Base, LoRA r=16, 3 epochs

Content-word accuracy = every token of a content word right (script-fair). Exact = whole sentence right (greedy).

| N | cell | tokens/sent | content-word acc | exact iid | exact comp | exact morph | exact long | train min | batch |
|---|---|---|---|---|---|---|---|---|---|
| 5k | easy-latin | 12.5 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 4.0 | 16x2 |
| 5k | easy-tamil | 41.9 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 5.1 | 16x2 |
| 5k | hard-latin | 19.3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3.9 | 16x2 |
| 5k | hard-tamil | 43.5 | 0.001 | 0.002 | 0.000 | 0.000 | 0.000 | 5.4 | 16x2 |
| 10k | easy-latin | 12.5 | 0.113 | 0.032 | 0.000 | 0.022 | 0.000 | 7.5 | 16x2 |
| 10k | easy-tamil | 41.8 | 0.033 | 0.028 | 0.000 | 0.034 | 0.000 | 10.7 | 16x2 |
| 10k | hard-latin | 19.2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 7.8 | 16x2 |
| 10k | hard-tamil | 43.4 | 0.003 | 0.002 | 0.000 | 0.002 | 0.000 | 10.9 | 16x2 |
| 20k | easy-latin | 12.5 | 0.817 | 0.634 | 0.568 | 0.554 | 0.176 | 15.7 | 16x2 |
| 20k | easy-tamil | 42.0 | 0.830 | 0.650 | 0.598 | 0.596 | 0.188 | 21.7 | 16x2 |
| 20k | hard-latin | 19.3 | 0.048 | 0.022 | 0.000 | 0.010 | 0.000 | 15.8 | 16x2 |
| 20k | hard-tamil | 43.4 | 0.232 | 0.100 | 0.002 | 0.072 | 0.000 | 21.3 | 16x2 |
| 50k | easy-latin | 12.5 | 0.984 | 0.960 | 0.964 | 0.964 | 0.856 | 24.9 | 32x1 |
| 50k | easy-tamil | 42.1 | 0.997 | 0.994 | 0.988 | 0.984 | 0.992 | 53.8 | 16x2 |
| 50k | hard-latin | 19.3 | 0.980 | 0.960 | 0.952 | 0.836 | 0.940 | 40.7 | 16x2 |
| 50k | hard-tamil | 43.5 | 0.990 | 0.986 | 0.976 | 0.920 | 0.958 | 53.6 | 32x1 |
