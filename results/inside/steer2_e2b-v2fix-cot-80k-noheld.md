# Steering test (phase 2, redesigned): e2b-v2fix-cot-80k-noheld

Picked setting: layer 6, position all, alpha 0.035 (share of the layer's hidden-state norm).

| pair | n | flip | random flip | unreadable | other features kept |
|---|---|---|---|---|---|
| tense:PRES>PAST | 100 | 1.00 | 0.02 | 0.00 | 0.7 |
| tense:PAST>PRES | 100 | 0.98 | 0.00 | 0.02 | 0.87 |
| tense:PRES>FUT | 100 | 0.98 | 0.01 | 0.00 | 0.25 |
| tense:FUT>PRES | 100 | 0.98 | 0.00 | 0.00 | 0.15 |
| NEG:no>yes | 100 | 0.93 | 0.00 | 0.01 | 0.52 |
| NEG:yes>no | 100 | 0.99 | 0.01 | 0.00 | 0.7 |
| Q:no>yes | 100 | 0.29 | 0.00 | 0.02 | 0.72 |
| Q:yes>no | 99 | 0.81 | 0.02 | 0.09 | 0.79 |

| feature | real failures | fixed by the push | fixed by a random push |
|---|---|---|---|
| tense | 1 | 0 | 0 |
| NEG | 0 | 0 | 0 |
| Q | 0 | 0 | 0 |
