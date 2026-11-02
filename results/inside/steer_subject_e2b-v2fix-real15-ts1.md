# Subject steering: e2b-v2fix-real15-ts1

Picked setting: layer 10, position english, alpha 0.035.

| pair | n | flip | random flip | shuffled-label flip | unreadable | other features kept |
|---|---|---|---|---|---|---|
| 1sg>2sg | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 0.97 |
| 2sg>1sg | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 0.93 |
| 1sg>3sgm | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| 3sgm>1sg | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 0.88 |
| 3sgm>3sgf | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 0.95 |
| 3sgf>3sgm | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 0.88 |
| 1sg>1pl | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 0.95 |
| 1pl>1sg | 40 | 0.97 | 0.00 | 0.00 | 0.00 | 0.47 |
| 3sgm>3pl | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 0.97 |
| 3pl>3sgm | 40 | 0.97 | 0.00 | 0.00 | 0.00 | 0.97 |
| name>noun | 40 | 0.55 | 0.00 | 0.00 | 0.00 | 0.97 |
| noun>name | 40 | 0.57 | 0.03 | 0.00 | 0.00 | 0.97 |

| setting | real subject failures | fixed by the push | fixed by a random push | shuffled labels | unreadable |
|---|---|---|---|---|---|
| picked [10, 'english', 0.035] | 1 | 1 | 0 | 0 | 0 |
| tense setting [6, 'all', 0.035] | 1 | 1 | 0 | 1 | 0 |

Real failures (written -> gold : after the push):

- name -> noun : noun  (Maria usually goes to school on bike.)
