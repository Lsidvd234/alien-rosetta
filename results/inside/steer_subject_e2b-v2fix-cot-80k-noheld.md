# Subject steering: e2b-v2fix-cot-80k-noheld

Picked setting: layer 6, position english, alpha 0.02.

| pair | n | flip | random flip | shuffled-label flip | unreadable | other features kept |
|---|---|---|---|---|---|---|
| 1sg>2sg | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 0.97 |
| 2sg>1sg | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| 1sg>3sgm | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| 3sgm>1sg | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| 3sgm>3sgf | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| 3sgf>3sgm | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 0.97 |
| 1sg>1pl | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| 1pl>1sg | 40 | 0.53 | 0.00 | 0.00 | 0.00 | 1.0 |
| 3sgm>3pl | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| 3pl>3sgm | 40 | 1.00 | 0.00 | 0.00 | 0.00 | 1.0 |
| name>noun | 40 | 0.65 | 0.00 | 0.00 | 0.00 | 1.0 |
| noun>name | 40 | 0.05 | 0.00 | 0.00 | 0.03 | 1.0 |

| setting | real subject failures | fixed by the push | fixed by a random push | shuffled labels | unreadable |
|---|---|---|---|---|---|
| picked [6, 'english', 0.02] | 12 | 11 | 3 | 3 | 0 |
| tense setting [6, 'all', 0.035] | 12 | 10 | 3 | 5 | 1 |

Real failures (written -> gold : after the push):

- none -> name : name  (Tom fell apart.)
- group -> none : none  (Tom and Mary are worried about you.)
- none -> noun : noun  (The victory was dearly won.)
- none -> noun : none  (Someone please stop this nightmare.)
- name -> noun : noun  (Maria usually goes to school on bike.)
- name -> 2sg : 2sg  (You said it, Tom.)
- name -> noun : noun  (Ken met Tom here.)
- name -> noun : noun  (Mum plays tennis.)
- noun -> none : none  (Your mother is worried about you.)
- 3sgm -> none : none  (He is satisfied.)
- noun -> name : name  (Anton is talking to me.)
- name -> noun : noun  (War seemed a long way off.)
