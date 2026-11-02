# Steering test (phase 2, redesigned): e2b-v2fix-real15

Picked setting: layer 10, position english, alpha 0.035 (share of the layer's hidden-state norm), picked on 100 OTHER sentences than the ones reported below.

| pair | n | flip | random flip | unreadable | other features kept |
|---|---|---|---|---|---|
| tense:PRES>PAST | 100 | 0.98 | 0.00 | 0.00 | 0.89 |
| tense:PAST>PRES | 100 | 1.00 | 0.00 | 0.00 | 0.95 |
| tense:PRES>FUT | 100 | 0.99 | 0.01 | 0.00 | 0.29 |
| tense:FUT>PRES | 100 | 0.98 | 0.01 | 0.00 | 0.38 |
| NEG:no>yes | 100 | 0.89 | 0.00 | 0.00 | 0.96 |
| NEG:yes>no | 100 | 0.79 | 0.00 | 0.00 | 0.9 |
| Q:no>yes | 100 | 0.92 | 0.00 | 0.01 | 0.85 |
| Q:yes>no | 99 | 0.98 | 0.01 | 0.00 | 0.33 |

| feature | real failures | fixed by the push | fixed by a random push |
|---|---|---|---|
| tense | 0 | 0 | 0 |
| NEG | 0 | 0 | 0 |
| Q | 0 | 0 | 0 |
