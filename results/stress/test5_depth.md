# Test 5: longer and deeper sentences

150 new sentences per level. Training had at most one adjective per noun, and never an
adjective on both nouns together with a location (that is the held-out 'long' structure).
Assumed rule for 2+ adjectives (never seen): each adjective is its own word before the noun, in English order.

| level | example | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|---|
| 0. plain: subject verb object | the lung mocked the kit | 100.0% | 100.0% | 0.0% | 99.3% |
| 1. + location | the hat admired the oaks in the shop | 98.7% | 98.0% | 0.7% | 99.3% |
| 2. + 1 adjective on the subject | the tribal tam spun the poplar in the palace | 98.0% | 96.7% | 0.7% | 96.7% |
| 3. adjective on both nouns + location (= held-out 'long') | the early rotor raised the deluxe walnut in the hank | 95.3% | 91.3% | 0.0% | 66.7% |
| 4. 2 adjectives on the subject, 1 on the object | the loyal great ploughs puffed the nuts crops in the pub | 0.7% | 0.0% | 0.0% | 0.0% |
| 5. 2 adjectives on each noun | the bad speedy submarine hums the spoken serial nib in the portal | 0.0% | 0.0% | 0.0% | 0.0% |
| 6. 3 adjectives on each noun | the equal blue quiet haircuts buckled the whole loud yellow drum in the room | 0.0% | 0.0% | 0.0% | 0.0% |
| 7. level 3 + past + negated + plural subject (two held-out things at once) | the green tams did not cough the warm spades in the tent | 96.0% | 78.0% | 0.7% | 50.0% |
