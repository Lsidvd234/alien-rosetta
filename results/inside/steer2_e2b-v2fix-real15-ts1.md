# Steering test (phase 2, redesigned): e2b-v2fix-real15-ts1

Picked setting: layer 6, position english, alpha 0.035 (share of the layer's hidden-state norm).

| pair | n | flip | random flip | unreadable | other features kept |
|---|---|---|---|---|---|
| tense:PRES>PAST | 100 | 1.00 | 0.00 | 0.00 | 0.53 |
| tense:PAST>PRES | 100 | 0.99 | 0.00 | 0.00 | 0.78 |
| tense:PRES>FUT | 100 | 1.00 | 0.00 | 0.00 | 0.24 |
| tense:FUT>PRES | 100 | 0.98 | 0.00 | 0.00 | 0.15 |
| NEG:no>yes | 100 | 0.95 | 0.00 | 0.03 | 0.61 |
| NEG:yes>no | 100 | 0.98 | 0.00 | 0.01 | 0.58 |
| Q:no>yes | 100 | 0.92 | 0.01 | 0.02 | 0.81 |
| Q:yes>no | 99 | 1.00 | 0.01 | 0.00 | 0.51 |

| feature | real failures | fixed by the push | fixed by a random push |
|---|---|---|---|
| tense | 2 | 2 | 1 |
| NEG | 0 | 0 | 0 |
| Q | 0 | 0 | 0 |
