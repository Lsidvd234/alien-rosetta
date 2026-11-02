# Is a misread passive's doer still in the model? (`analysis/inside_passive.py`, prediction R4, redesign)

All four doer x patient number combinations, balanced. Probe A: trained on actives only. Probe B: trained on actives + other passives (R4). Baselines on the same split: bag of words (no model) and layer 0 (word embeddings). Misread = output S= is not the gold doer. Chance 0.5. Rule (amended 28 Oct): readout failure if probe B where misread >= 0.85 and >= 0.10 above the better baseline; representation failure if < 0.60 or within 0.05.

| run | output: doer number right (differ) | output: first noun read as doer | probe A on passives | probe B on passives | bag of words / layer 0 | where misread (n): probe B, bag of words, layer 0 | shuffled control | patient probe | layers A / B | R4 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | - | - | 0.5333 | 0.7792 | 0.8 / 0.7792 | (-) -, -, - | 0.475 | 0.9042 | mean:6 / mean:0 | - |
| e1b-hard-rep-cot-5ep | 0.0 | 1.0 | 0.5083 | 0.7833 | 0.8 / 0.7792 | (240) 0.7833, 0.8, 0.7792 | 0.4417 | 0.8708 | last:18 / last:14 | representation failure |
