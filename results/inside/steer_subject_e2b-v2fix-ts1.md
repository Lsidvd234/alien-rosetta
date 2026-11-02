# Subject steering: e2b-v2fix-ts1

Picked setting: layer 6, position english, alpha 0.02.

| pair | n | flip | random flip | shuffled-label flip | unreadable | other features kept |
|---|---|---|---|---|---|---|
| 1sg>2sg | 40 | 0.97 | 0.00 | 0.00 | 0.00 | 1.0 |
| 2sg>1sg | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| 1sg>3sgm | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 0.97 |
| 3sgm>1sg | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| 3sgm>3sgf | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| 3sgf>3sgm | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 0.95 |
| 1sg>1pl | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| 1pl>1sg | 40 | 0.25 | 0.00 | 0.00 | 0.00 | 0.95 |
| 3sgm>3pl | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| 3pl>3sgm | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| name>noun | 40 | 0.88 | 0.00 | 0.00 | 0.07 | 1.0 |
| noun>name | 40 | 0.05 | 0.03 | 0.00 | 0.03 | 0.9 |

| setting | real subject failures | fixed by the push | fixed by a random push | shuffled labels | unreadable |
|---|---|---|---|---|---|
| picked [6, 'english', 0.02] | 10 | 10 | 1 | 1 | 0 |
| tense setting [6, 'all', 0.035] | 10 | 6 | 2 | 3 | 4 |

Real failures (written -> gold : after the push):

- group -> none : none  (Tom and Mary are worried about you.)
- none -> noun : noun  (The victory was dearly won.)
- none -> noun : noun  (Someone please stop this nightmare.)
- name -> noun : noun  (Maria usually goes to school on bike.)
- name -> noun : noun  (Ken met Tom here.)
- name -> noun : noun  (Mum plays tennis.)
- noun -> none : none  (Your mother is worried about you.)
- none -> 1sg : 1sg  (Hey, could I talk to you for a sec?)
- noun -> name : name  (Anton is talking to me.)
- name -> noun : noun  (War seemed a long way off.)
