# Is a misread passive's doer still in the model? (`analysis/inside_passive.py`, prediction R4, redesign)

All four doer x patient number combinations, balanced. Probe A: trained on actives only. Probe B: trained on actives + other passives (R4). Baselines on the same split: bag of words (no model) and layer 0 (word embeddings). Misread = output S= is not the gold doer. Chance 0.5. Rule (amended 28 Oct): readout failure if probe B where misread >= 0.85 and >= 0.10 above the better baseline; representation failure if < 0.60 or within 0.05.

| run | output: doer number right (differ) | output: first noun read as doer | probe A on passives | probe B on passives | bag of words / layer 0 | where misread (n): probe B, bag of words, layer 0 | shuffled control | patient probe | layers A / B | R4 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | - | - | 0.5542 | 0.7042 | 0.6875 / 0.7042 | (-) -, -, - | 0.4625 | 0.8167 | mean:20 / mean:0 | - |
| e2b-v2fix-cot-80k-noheld | 1.0 | 0.0292 | 0.6708 | 0.8708 | 0.6875 / 0.7042 | (3) 1.0, 0.6667, 0.3333 | 0.4875 | 0.9125 | last:20 / last:28 | readout failure |
| e2b-v2fix-passive0 | 0.0 | 0.0792 | 0.5625 | 0.7458 | 0.6875 / 0.7042 | (240) 0.7458, 0.6875, 0.7042 | 0.4833 | 0.9417 | last:19 / last:17 | representation failure |
| e2b-v2fix-passive300 | 0.9821 | 0.0292 | 0.7917 | 0.95 | 0.6875 / 0.7042 | (3) 1.0, 1.0, 1.0 | 0.4667 | 0.875 | last:24 / last:21 | representation failure |
