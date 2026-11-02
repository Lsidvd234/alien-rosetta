# Steering test (phase 2, redesigned): e2b-v2fix-ts1

Picked setting: layer 6, position english, alpha 0.035 (share of the layer's hidden-state norm).

| pair | n | flip | random flip | unreadable | other features kept |
|---|---|---|---|---|---|
| tense:PRES>PAST | 100 | 1.00 | 0.00 | 0.00 | 0.57 |
| tense:PAST>PRES | 100 | 0.99 | 0.00 | 0.01 | 0.78 |
| tense:PRES>FUT | 100 | 1.00 | 0.01 | 0.00 | 0.29 |
| tense:FUT>PRES | 100 | 0.97 | 0.01 | 0.01 | 0.13 |
| NEG:no>yes | 100 | 0.91 | 0.00 | 0.02 | 0.57 |
| NEG:yes>no | 100 | 0.98 | 0.00 | 0.02 | 0.58 |
| Q:no>yes | 100 | 0.95 | 0.00 | 0.01 | 0.78 |
| Q:yes>no | 99 | 0.89 | 0.00 | 0.11 | 0.57 |

| feature | real failures | fixed by the push | fixed by a random push |
|---|---|---|---|
| tense | 1 | 1 | 0 |
| NEG | 0 | 0 | 0 |
| Q | 5 | 5 | 4 |
