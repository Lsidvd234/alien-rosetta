# Final score vs mean of the last 3 checkpoints (`analysis/checkpoint_average.py`)

Final = the run's end score (300 items). Last-3 = mean of the last 3 training-curve evaluations (100 items each). Only tests whose name contains 'needs' or 'trained lengths'.

## Same-seed re-run pairs (GPU noise only)

| test | run A final | run B final | gap final | A last-3 | B last-3 | gap last-3 |
|---|---|---|---|---|---|---|
| buildmix-drop56-s1: needs 5x6 | 0.72 | 0.80 | 0.08 | 0.69 | 0.79 | 0.10 |
| buildmix-drop66: needs 6x6 | 0.64 | 0.86 | 0.23 | 0.85 | 0.92 | 0.07 |
| buildmix-drop6row: needs 6x4 | 0.43 | 0.43 | 0.00 | 0.42 | 0.42 | 0.00 |
| buildmix-drop6row: needs 6x5 | 0.29 | 0.18 | 0.11 | 0.31 | 0.20 | 0.10 |
| buildmix-drop6row: needs 6x6 | 0.49 | 0.41 | 0.07 | 0.40 | 0.35 | 0.06 |
| maths-meaning-buildadd-drop6row vs maths-meaning-buildadd-drop6row-rep | not both available | | | | | |

Mean gap between same-seed runs: final 0.098, last-3 mean 0.067 (5 tests).

## Every maths run: needs-pair tests

| run | test | final | last-3 mean | last 3 points |
|---|---|---|---|---|
| mean-drop56-bridge | needs dropped pair 5x6 | 0.02 | 0.02 | 0.03, 0.01, 0.03 |
| meaning-build-drop56 | needs dropped pair 5x6 | 0.08 | 0.20 | 0.34, 0.17, 0.08 |
| meaning-build-drop56-role100 | needs dropped pair 5x6 | 0.48 | 0.42 | 0.37, 0.44, 0.44 |
| meaning-build-drop56-role25 | needs dropped pair 5x6 | 0.51 | 0.46 | 0.46, 0.46, 0.46 |
| meaning-build-drop56-role400 | needs dropped pair 5x6 | 0.71 | 0.54 | 0.49, 0.49, 0.65 |
| meaning-build-drop56-role400-s1 | needs dropped pair 5x6 | 0.47 | 0.41 | 0.42, 0.40, 0.41 |
| meaning-build-drop56-role5 | needs dropped pair 5x6 | 0.48 | 0.40 | 0.30, 0.43, 0.46 |
| meaning-build-drop56-role593 | needs dropped pair 5x6 | 0.53 | 0.44 | 0.41, 0.44, 0.48 |
| meaning-buildadd-drop6row | needs dropped pair 6x4 | 0.61 | 0.66 | 0.72, 0.66, 0.59 |
| meaning-buildadd-drop6row | needs dropped pair 6x5 | 0.33 | 0.29 | 0.31, 0.23, 0.32 |
| meaning-buildadd-drop6row | needs dropped pair 6x6 | 0.97 | 0.81 | 0.59, 0.89, 0.97 |
| meaning-buildmix-drop34 | needs dropped pair 3x4 | 0.98 | 0.98 | 0.98, 0.99, 0.97 |
| meaning-buildmix-drop56 | needs dropped pair 5x6 | 0.77 | 0.85 | 0.89, 0.84, 0.82 |
| meaning-buildmix-drop56-s1 | needs dropped pair 5x6 | 0.72 | 0.69 | 0.74, 0.65, 0.69 |
| meaning-buildmix-drop56-s1-fo | needs dropped pair 5x6 | 0.80 | 0.79 | 0.81, 0.81, 0.75 |
| meaning-buildmix-drop66 | needs dropped pair 6x6 | 0.64 | 0.85 | 0.92, 0.95, 0.67 |
| meaning-buildmix-drop66-fo | needs dropped pair 6x6 | 0.86 | 0.92 | 0.94, 0.95, 0.87 |
| meaning-buildmix-drop6row | needs dropped pair 6x4 | 0.43 | 0.42 | 0.44, 0.32, 0.50 |
| meaning-buildmix-drop6row | needs dropped pair 6x5 | 0.29 | 0.31 | 0.33, 0.30, 0.29 |
| meaning-buildmix-drop6row | needs dropped pair 6x6 | 0.49 | 0.40 | 0.33, 0.41, 0.47 |
| meaning-buildmix-drop6row-fo | needs dropped pair 6x4 | 0.43 | 0.42 | 0.41, 0.37, 0.47 |
| meaning-buildmix-drop6row-fo | needs dropped pair 6x5 | 0.18 | 0.20 | 0.17, 0.27, 0.17 |
| meaning-buildmix-drop6row-fo | needs dropped pair 6x6 | 0.41 | 0.35 | 0.23, 0.40, 0.41 |
| meaning-double-drop6row | needs dropped pair 6x4 | 0.65 | 0.62 | 0.62, 0.56, 0.69 |
| meaning-double-drop6row | needs dropped pair 6x5 | 0.02 | 0.01 | 0.00, 0.00, 0.03 |
| meaning-double-drop6row | needs dropped pair 6x6 | 0.00 | 0.00 | 0.00, 0.00, 0.00 |
| meaning-drop22 | needs dropped pair 2x2 | 0.41 | 0.47 | 0.46, 0.48, 0.48 |
| meaning-drop34 | needs dropped pair 3x4 | 0.10 | 0.10 | 0.11, 0.09, 0.10 |
| meaning-drop34-tw | needs dropped pair 3x4 | 0.10 | 0.10 | 0.11, 0.09, 0.10 |
| meaning-drop56 | needs dropped pair 5x6 | 0.05 | 0.07 | 0.08, 0.06, 0.07 |
| meaning-drop56-bridge | needs dropped pair 5x6 | 0.05 | 0.07 | 0.08, 0.06, 0.07 |
| meaning-drop56-bridge2 | needs dropped pair 5x6 | 0.03 | 0.04 | 0.04, 0.03, 0.04 |
| meaning-drop56-tw | needs dropped pair 5x6 | 0.05 | 0.07 | 0.08, 0.06, 0.07 |
| meaning-scratch-drop56 | needs dropped pair 5x6 | 0.26 | 0.27 | 0.27, 0.24, 0.30 |
| meaning-scratch-drop56-tw | needs dropped pair 5x6 | 0.26 | 0.27 | 0.27, 0.24, 0.30 |
| meaning-scratch-drop56-vote | needs dropped pair 5x6 | 0.26 | 0.27 | 0.27, 0.24, 0.30 |
| meaning-scratch-drop65ordered | needs ordered 6x5 | 0.72 | 0.73 | 0.70, 0.75, 0.73 |
| meaning-scratch-drop65ordered | needs the other order 5x6 only | 0.97 | 0.97 | 0.97, 0.97, 0.97 |
| meaning-scratch-drop6row | needs dropped pair 6x4 | 0.00 | 0.00 | 0.00, 0.00, 0.00 |
| meaning-scratch-drop6row | needs dropped pair 6x5 | 0.14 | 0.15 | 0.19, 0.12, 0.14 |
| meaning-scratch-drop6row | needs dropped pair 6x6 | 0.00 | 0.00 | 0.00, 0.00, 0.00 |
| meaning-split-drop6row | needs dropped pair 6x4 | 0.59 | 0.54 | 0.49, 0.53, 0.61 |
| meaning-split-drop6row | needs dropped pair 6x5 | 0.33 | 0.35 | 0.33, 0.35, 0.37 |
| meaning-split-drop6row | needs dropped pair 6x6 | 0.27 | 0.27 | 0.17, 0.40, 0.23 |
| meaning-trick-drop6row | needs dropped pair 6x4 | 0.04 | 0.03 | 0.04, 0.01, 0.04 |
| meaning-trick-drop6row | needs dropped pair 6x5 | 0.23 | 0.29 | 0.27, 0.29, 0.32 |
| meaning-trick-drop6row | needs dropped pair 6x6 | 0.00 | 0.00 | 0.00, 0.00, 0.00 |
| shuffled-drop56-bridge | needs dropped pair 5x6 | 0.01 | 0.00 | 0.00, 0.00, 0.01 |
