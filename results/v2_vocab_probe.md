# Do the v1-trained models struggle with the v2 vocabulary?

Every saved model, 50 sentences per set, same sentence shape in every set (the [describing] noun VERB the [describing] noun; one tense; optional 'not'; lower case); only where the words come from changes. Answers from conlang/v2.py for each model's own language seed. Cell = glyph answer exact (analysis exact for models that write one). `analysis/v2_vocab_probe.py`.

| model | 1. v1 words (control) | 2. new root words | 3. built words | 4. everyday additions |
|---|---|---|---|---|
| Option 1 (sound start) | 1.00 | 0.00 | 0.00 | 0.00 |
| Option 2 (transliteration) | 1.00 | 0.00 | 0.00 | 0.00 |
| Option 3a (plain text first) | 0.00 | 0.00 | 0.00 | 0.00 |
| control (wrong sounds) | 1.00 | 0.00 | 0.00 | 0.00 |
| R1 | 0.98 | 0.00 | 0.00 | 0.00 |
| E1a 3ep | 0.12 | 0.00 | 0.00 | 0.00 |
| E1b 3ep | 1.00 (1.00) | 0.00 (0.46) | 0.00 (0.00) | 0.00 (0.34) |
| E1a 5ep | 0.96 | 0.00 | 0.00 | 0.00 |
| E1b 5ep | 0.98 (0.98) | 0.00 (0.48) | 0.00 (0.00) | 0.00 (0.38) |
| E1b 5ep seed 1 | 0.98 (1.00) | 0.00 (0.50) | 0.00 (0.00) | 0.00 (0.40) |

## Wrong examples, E1b 5ep (first 5 per set): analysis wanted vs written

**1. v1 words (control)**
- `the bisexual doodles commemorate the pew`  
  want `S=bisexual+doodle.PL V=commemorate.PRES O=pew`  
  got  `S=bisexual+doodle V=commemorate.PRES O=pew`

**2. new root words**
- `the mems resell the lookup`  
  want `S=mem.PL V=resell.PRES O=lookup`  
  got  `S=mems V=resell.PRES O=lookup`
- `the reform settled the alphabets`  
  want `S=reform V=settle.PAST O=alphabet.PL`  
  got  `S=reform V=settle.PAST O=alphabet.PL`
- `the tyrosine underestimates the eighteenth amounts`  
  want `S=tyrosine V=underestimate.PRES O=eighteenth+amount.PL`  
  got  `S=tyrosine V=underestimate.PRES O=eighth+amount.PL`
- `the cranky acquaintances hoarded the chassis`  
  want `S=cranky+acquaintance.PL V=hoard.PAST O=chassis`  
  got  `S=cranky+acquaintances V=hoard.PAST O=chassis`
- `the edges will cull the cases`  
  want `S=edge.PL V=cull.FUT O=case.PL`  
  got  `S=edge.PL V=cull.FUT O=case.PL`

**3. built words**
- `the cutters symbolize the informant`  
  want `S=cut>AGENT.PL V=symbol>VERB.PRES O=inform>AGENT`  
  got  `S=cutter.PL V=symbolize.PRES O=informant`
- `the rigidity solidified the elasticities`  
  want `S=rigid>QUAL V=solid>VERB.PAST O=elastic>QUAL.PL`  
  got  `S=rigidity V=solidify.PAST O=elasticity.PL`
- `the stimulation revolutionizes the ceremonial selectors`  
  want `S=stimulate>ACT V=revolution>VERB.PRES O=ceremony>ADJ+select>AGENT.PL`  
  got  `S=stimulation V=revolutionize.PRES O=ceremonial+selector.PL`
- `the spinal councilmen undressed the opposition`  
  want `S=spine>ADJ+council_man.PL V=dress>NOT.PAST O=oppose>ACT`  
  got  `S=spinal+consultant.PL V=undress.PAST O=opposition`
- `the rarities will fortify the lenders`  
  want `S=rare>QUAL.PL V=fort>VERB.FUT O=lend>AGENT.PL`  
  got  `S=rarity.PL V=fortify.FUT O=lender.PL`

**4. everyday additions**
- `the emperors tell the member`  
  want `S=emperor.PL V=tell.PRES O=member`  
  got  `S=empire.PL V=say.PRES O=member`
- `the criminal told the neighbours`  
  want `S=criminal V=tell.PAST O=neighbour.PL`  
  got  `S=criminal V=inform.PAST O=neighbor.PL`
- `the anybody gets the boring cops`  
  want `S=anybody V=get.PRES O=boring+cop.PL`  
  got  `S=anybody V=obtain.PRES O=boring+cop.PL`
- `the tired sayings said the pupil`  
  want `S=tired+saying.PL V=say.PAST O=pupil`  
  got  `S=tired+saying.PL V=say.PAST O=pupil`
- `the grooms will say the emperors`  
  want `S=groom.PL V=say.FUT O=emperor.PL`  
  got  `S=groom.PL V=say.FUT O=emperor.PL`
