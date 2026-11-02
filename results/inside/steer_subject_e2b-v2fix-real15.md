# Subject steering: e2b-v2fix-real15

Picked setting: layer 6, position english, alpha 0.02.

| pair | n | flip | random flip | shuffled-label flip | unreadable | other features kept |
|---|---|---|---|---|---|---|
| 1sg>2sg | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 0.9 |
| 2sg>1sg | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| 1sg>3sgm | 40 | 0.97 | 0.00 | 0.00 | 0.00 | 0.97 |
| 3sgm>1sg | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| 3sgm>3sgf | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| 3sgf>3sgm | 40 | 0.93 | 0.00 | 0.00 | 0.00 | 0.97 |
| 1sg>1pl | 40 | 0.78 | 0.00 | 0.00 | 0.00 | 0.97 |
| 1pl>1sg | 40 | 0.72 | 0.00 | 0.00 | 0.00 | 0.97 |
| 3sgm>3pl | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| 3pl>3sgm | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| name>noun | 40 | 0.85 | 0.00 | 0.00 | 0.00 | 1.0 |
| noun>name | 40 | 0.07 | 0.00 | 0.00 | 0.00 | 0.95 |

| setting | real subject failures | fixed by the push | fixed by a random push | shuffled labels | unreadable |
|---|---|---|---|---|---|
| picked [6, 'english', 0.02] | 4 | 4 | 0 | 1 | 0 |
| tense setting [6, 'all', 0.035] | 4 | 2 | 1 | 1 | 2 |

Real failures (written -> gold : after the push):

- none -> noun : noun  (The victory was dearly won.)
- name -> noun : noun  (Maria usually goes to school on bike.)
- none -> noun : noun  (Come in, the door's open.)
- name -> noun : noun  (War seemed a long way off.)
