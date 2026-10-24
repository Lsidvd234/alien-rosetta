# Was each run still learning? (`analysis/epoch_curve.py`)

Each epoch checkpoint on the same generated test sentences (none trained on) and on 150 of the run's own training examples. The learning rate decays linearly to 0 over the 4 epochs.

| run | epoch | glyphs exact | meaning exact | word acc | train loss | zero-grad (fp32) |
|---|---|---|---|---|---|---|
| e2a-v2-80k | 1 | 0.10 | nan | 0.655 | 0.3090 | 0.2% |
| e2a-v2-80k | 2 | 0.17 | nan | 0.788 | 0.1113 | 1.0% |
| e2a-v2-80k | 3 | 0.30 | nan | 0.850 | 0.0550 | 3.2% |
| e2a-v2-80k | 4 | 0.36 | nan | 0.858 | 0.0310 | 6.6% |
| e2b-v2-cot-80k-noheld | 1 | 0.14 | 0.66 | 0.733 | 0.2054 | 1.7% |
| e2b-v2-cot-80k-noheld | 2 | 0.33 | 0.75 | 0.847 | 0.0880 | 2.8% |
| e2b-v2-cot-80k-noheld | 3 | 0.35 | 0.76 | 0.864 | 0.0403 | 5.7% |
| e2b-v2-cot-80k-noheld | 4 | 0.40 | 0.75 | 0.880 | 0.0261 | 7.7% |
| e2b-v2fix-cot-80k-noheld | 1 | 0.16 | 0.68 | 0.757 | 0.1840 | 0.9% |
| e2b-v2fix-cot-80k-noheld | 2 | 0.30 | 0.68 | 0.842 | 0.0649 | 3.6% |
| e2b-v2fix-cot-80k-noheld | 3 | 0.43 | 0.74 | 0.892 | 0.0214 | 5.2% |
| e2b-v2fix-cot-80k-noheld | 4 | 0.53 | 0.76 | 0.912 | 0.0114 | 8.0% |
| e2b-v2fix-cot-80k | 1 | 0.17 | 0.58 | 0.729 | 0.1736 | 1.9% |
| e2b-v2fix-cot-80k | 2 | 0.23 | 0.54 | 0.781 | 0.0627 | 1.8% |
| e2b-v2fix-cot-80k | 3 | 0.32 | 0.60 | 0.828 | 0.0198 | 5.1% |
| e2b-v2fix-cot-80k | 4 | 0.43 | 0.62 | 0.855 | 0.0084 | 7.0% |
| e2b-v2fix-focal-r2 | 1 | 0.17 | 0.64 | 0.723 | 0.2078 | 0.0% |
| e2b-v2fix-focal-r2 | 2 | 0.29 | 0.69 | 0.831 | 0.0851 | 0.0% |
| e2b-v2fix-focal-r2 | 3 | 0.50 | 0.73 | 0.903 | 0.0401 | 0.0% |
| e2b-v2fix-focal-r2 | 4 | 0.54 | 0.74 | 0.911 | 0.0300 | 0.0% |
| e2b-v2fix-loan-r2 | 1 | 0.16 | 0.64 | 0.747 | 0.1595 | 0.8% |
| e2b-v2fix-loan-r2 | 2 | 0.31 | 0.73 | 0.829 | 0.0522 | 1.4% |
| e2b-v2fix-loan-r2 | 3 | 0.43 | 0.74 | 0.881 | 0.0181 | 4.9% |
| e2b-v2fix-loan-r2 | 4 | 0.51 | 0.75 | 0.908 | 0.0105 | 7.5% |
| e2b-v2fix-para | 1 | 0.19 | 0.67 | 0.760 | 0.1817 | 1.7% |
| e2b-v2fix-para | 2 | 0.30 | 0.72 | 0.840 | 0.0637 | 2.5% |
| e2b-v2fix-para | 3 | 0.44 | 0.74 | 0.896 | 0.0210 | 4.7% |
| e2b-v2fix-para | 4 | 0.58 | 0.75 | 0.915 | 0.0125 | 7.6% |

| run | glyphs exact, epoch 3 → 4 | word acc, epoch 3 → 4 | train loss, epoch 3 → 4 |
|---|---|---|---|
| e2a-v2-80k | 0.30 → 0.36 | 0.850 → 0.858 | 0.0550 → 0.0310 |
| e2b-v2-cot-80k-noheld | 0.35 → 0.40 | 0.864 → 0.880 | 0.0403 → 0.0261 |
| e2b-v2fix-cot-80k-noheld | 0.43 → 0.53 | 0.892 → 0.912 | 0.0214 → 0.0114 |
| e2b-v2fix-cot-80k | 0.32 → 0.43 | 0.828 → 0.855 | 0.0198 → 0.0084 |
| e2b-v2fix-focal-r2 | 0.50 → 0.54 | 0.903 → 0.911 | 0.0401 → 0.0300 |
| e2b-v2fix-loan-r2 | 0.43 → 0.51 | 0.881 → 0.908 | 0.0181 → 0.0105 |
| e2b-v2fix-para | 0.44 → 0.58 | 0.896 → 0.915 | 0.0210 → 0.0125 |
