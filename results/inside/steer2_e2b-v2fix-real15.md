# Steering test (phase 2, redesigned): e2b-v2fix-real15

Picked setting: layer 6, position all, alpha 0.035 (share of the layer's hidden-state norm).

| pair | n | flip | random flip | unreadable | other features kept |
|---|---|---|---|---|---|
| tense:PRES>PAST | 100 | 1.00 | 0.01 | 0.00 | 0.68 |
| tense:PAST>PRES | 100 | 0.97 | 0.00 | 0.02 | 0.84 |
| tense:PRES>FUT | 100 | 1.00 | 0.01 | 0.00 | 0.23 |
| tense:FUT>PRES | 100 | 0.76 | 0.02 | 0.22 | 0.24 |
| NEG:no>yes | 100 | 0.92 | 0.00 | 0.04 | 0.42 |
| NEG:yes>no | 100 | 0.97 | 0.02 | 0.03 | 0.56 |
| Q:no>yes | 100 | 0.15 | 0.00 | 0.52 | 0.56 |
| Q:yes>no | 99 | 0.85 | 0.01 | 0.06 | 0.22 |

| feature | real failures | fixed by the push | fixed by a random push |
|---|---|---|---|
| tense | 0 | 0 | 0 |
| NEG | 0 | 0 | 0 |
| Q | 0 | 0 | 0 |
