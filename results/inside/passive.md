# Is a misread passive's doer still in the model? (`analysis/inside_passive.py`, prediction R4)

Test passives whose doer and patient differ in number. Probe A: trained on actives only. Probe B: trained on actives + other passives (R4). Chance 0.5.

| run | output: doer number right | output: first noun read as doer | probe A on passives (reads patient) | probe B on passives | probe B where output wrong (n) | shuffled control | patient probe | layers A / B |
|---|---|---|---|---|---|---|---|---|
| base | - | - | 0.0958 (0.9042) | 0.6875 | - (-) | 0.5667 | 0.6875 | mean:22 / mean:18 |
| e1b-hard-rep-cot-5ep | 0.0 | 0.9958 | 0.0 (1.0) | 0.8625 | 0.8625 (240) | 0.4958 | 0.8625 | last:18 / last:25 |
