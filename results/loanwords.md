# Per-loanword accuracy (`analysis/loan_words.py`)

Run e2b-v2fix-loan-r2. 120 generated sentences per set; every loan noun in the gold answer is scored.

| set | loan nouns | meaning right (LOAN.word) | glyph word right | sentences fully right |
|---|---|---|---|---|
| seen loans | 179 | **0.53** | **0.48** | 0.27 |
| new loans (never trained) | 187 | **0.35** | **0.18** | 0.11 |

## Examples (new loans): gold meaning vs the model's

- they might acquire Rical under the misfit downstairs  
  gold: `S=PRON.3pl V=acquire.PRES.MAY M=downstairs O=NAME.Rical UNDER=LOAN.misfit`  
  model: `S=PRON.3pl V=acquire.PRES.MAY M=downstairs O=NAME.Rical UNDER=LOAN.misfit`
- will you all be visual in front of the soapbox to arrange the fortune and the mighty residential purposes haven't been thought without the beau that Rore didn't outlaw the joinery and the really doctoral baguettes from the defenses  
  gold: `S=PRON.2pl V=be.FUT.Q FRONT=LOAN.soapbox A=visual ; C=to V=arrange O=fortune ; V=think.PRES.PERF.NEG O=mighty+residential+purpose.PL WITHOUT=LOAN.beau ; C=that S=NAME.Rore V=outlaw.PAST.NEG O=LOAN.joinery&really+doctor>ADJ+LOAN.baguette.PL FROM=defense.PL`  
  model: `S=PRON.2pl V=be.FUT.Q FRONT=LOAN.soapbox A=visual ; C=to V=arrange O=fortune ; S=mighty+residential+purpose.PL V=be.PRES.PERF.NEG WITHOUT=LOAN.beau A=thought ; C=that S=NAME.Rore V=outlaw.PAST.NEG O=joinery&really+doctor>ADJ+baguette.PL FROM=defense.PL`
- DISTRACT THE TOMBOY.  
  gold: `V=distract O=LOAN.tomboy`  
  model: `V=distract O=tomboy`
- My Eighteen Nutcracker Must Have Hurt On My Cerebellum?  
  gold: `S=PRON.1sg^eighteen+nutcracker V=hurt.PRES.PERF.MUST.Q ON=PRON.1sg^LOAN.cerebellum`  
  model: `S=PRON.1sg^eighteen+nutcracker V=hurt.PRES.PERF.MUST.Q ON=PRON.1sg^cerebellum`
- LINUU AND NALU WILL BE VERY BELOVED, HERBAL, FLAMMABLE AND KEPT; KINUUVUN OR LAAVIPU WON'T ENCOURAGE THE FIDDLER'S PALATINE AND VOVII.  
  gold: `S=NAME.Linuu&NAME.Nalu V=be.FUT A=very+dear+herb>ADJ+flammable+kept ; S=NAME.Kinuuvun|NAME.Laavipu V=encourage.FUT.NEG O=fiddle>AGENT^LOAN.palatine&NAME.Vovii`  
  model: `S=NAME.Linuu&NAME.Nalu V=be.FUT A=very+dear+herb>ADJ+flammable+kept ; S=NAME.Kinuuvun|NAME.Laavipu V=encourage.FUT.NEG O=fiddle>AGENT^palatine&NAME.Vovii`
- It  will  be  Vipiiyer  and  this  anything  that  will  finish  towards  this  coach  below  this  too  manly  scrim;  it  is  the  cornice  that  does  fqap  Lumuce  and  the  haywire  a  lot  
  gold: `S=NAME.Vipiiyer&this+anything V=finish.FUT TO=this+coach UNDER=this+too+man>ADJ+LOAN.scrim ; S=LOAN.cornice V=flap.PRES M=much O=NAME.Lumuce&LOAN.haywire`  
  model: `S=NAME.Vipiiyer&this+anything V=finish.FUT TO=this+coach UNDER=this+too+man>ADJ+scrim ; S=LOAN.cornice V=rap.PRES M=much O=NAME.Lumuce&LOAN.haywire`
- last summer Munimuum or every glass would tell her to every borrower on many culverts respectively; when did we abuse this fool.  
  gold: `S=NAME.Munimuum|every+glass T=last+summer V=tell.PRES.WOULD M=respective>ADV O=PRON.3sgf TO=every+borrow>AGENT ON=many+LOAN.culvert.PL ; S=PRON.1pl V=abuse.PAST M=when O=this+fool`  
  model: `S=NAME.Munimuum|every+glass T=last+summer V=tell.PRES.WOULD M=respective>ADV O=PRON.3sgf TO=every+borrow>AGENT ON=many+culvert.PL ; S=PRON.1pl V=abuse.PAST M=when O=this+fool`
- Another aviary would be a patriotic dirty equine fjord since Liyer when the old cystic slaves must be in an orchard.  
  gold: `S=another+LOAN.aviary V=be.PRES.WOULD SINCE=NAME.Liyer P=patriotic+dirt>ADJ+equine+LOAN.fjord ; C=when S=old+cyst>ADJ+LOAN.slave.PL V=be.PRES.MUST L=orchard`  
  model: `S=another+aviary V=be.PRES.WOULD SINCE=NAME.Liyer P=patriotic+dirt>ADJ+equine+LOAN.ford.PL ; C=when S=old+cyst>ADJ+slave.PL V=be.PRES.MUST L=orchard`
- Tell you all the too ugly warethogs quite infinitely  
  gold: `V=tell M=quite+finite>NOT>ADV O=too+ugly+LOAN.warthog.PL TO=PRON.2pl`  
  model: `V=tell M=quite+finite>NOT>ADV O=too+ugly+LOAN.warethog.PL TO=PRON.2pl`
- OKAY AND WAS I MORE BETTER AND LINK THIS DIALECT AND CIIPI TO THIS STATESMAN  
  gold: `I=okay ; S=PRON.1sg V=be.PAST.Q A=more+better ; V=link O=this+dialect&NAME.Ciipi TO=this+LOAN.statesman`  
  model: `I=okay ; S=PRON.1sg V=be.PAST.Q A=more+better ; V=link O=this+dialect&NAME.Ciipi TO=this+LOAN.statesman`
- Why  didn't  their  negative  and  teenage  sandbags  succumb  and  why  hasn't  Calii  packed  that  nanotube.  
  gold: `S=PRON.3pl^negative+teenage+LOAN.sandbag.PL V=succumb.PAST.NEG M=why ; S=NAME.Calii V=pack.PRES.PERF.NEG M=why O=that+LOAN.nanotube`  
  model: `S=PRON.3pl^negative+teenage+LOAN.sandbag.PL V=succumb.PAST.NEG M=why ; S=NAME.Calii V=pack.PRES.PERF.NEG M=why O=that+nanotube`
- don't  still  demand  his  really  given  candidate  and  my  really  sad  magnetite  very  weell  please  
  gold: `I=please V=demand.NEG M=still+very+well O=PRON.3sgm^really+given+LOAN.candidate&PRON.1sg^really+sad+LOAN.magnetite`  
  model: `I=please V=demand.NEG M=still+very+well O=PRON.3sgm^really+given+candidate&PRON.1sg^really+sad+magnetite`
