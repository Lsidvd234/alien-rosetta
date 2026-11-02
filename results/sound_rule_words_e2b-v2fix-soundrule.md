# Sound-rule language: words never seen in training (`analysis/sound_rule_words.py`), run e2b-v2fix-soundrule

300 test sentences that each use at least one held-out word (sentence glyphs exact 0.453, meaning exact 0.727; 2 skipped: gold not re-rendered).

| words | n | glyph word right |
|---|---|---|
| new (never in training) | 470 | **0.67** |
| seen (same sentences) | 1459 | 0.96 |

New words read as words of the language (their key in the model's meaning): 0.87. Wrong new words: 156, near misses (closeness >= 0.75) 58%, median closeness 0.78.

## Examples

- predict  that  contrary  lipid  across  them;  what  locks  her  although  Viruu  and  the  too  made  headlight  of  the  professor  can  set  her  while  please  waste  Lolucii  randomly  (new: lipid)  
  gold: `V=predict O=that+contrary+lipid THROUGH=PRON.3pl ; S=what V=lock.PRES O=PRON.3sgf ; C=although S=NAME.Viruu&professor^too+made+headlight V=set.PRES.CAN O=PRON.3sgf ; C=while I=please V=waste M=random>ADV O=NAME.Lolucii`  
  model: `V=predict O=that+contrary+lipid THROUGH=PRON.3pl ; S=what V=lock.PRES O=PRON.3sgf ; C=although S=NAME.Viruu&professor^too+made+headlight V=set.PRES.CAN O=PRON.3sgf ; C=while I=please V=waste M=random>ADV O=NAME.Lolucii`
- no and last year it and a poetic director were tunneling a greenish writer or Vemor  (new: last, year)  
  gold: `I=no ; S=PRON.3sgn&poet>ADJ+direct>AGENT T=last+year V=tunnel.PAST.PROG O=greenish+write>AGENT|NAME.Vemor`  
  model: `I=no ; S=PRON.3sgn&poet>ADJ+direct>AGENT T=last+year V=tunnel.PAST.PROG O=greenish+write>AGENT|NAME.Vemor`
- Can't the boy's means and Talutuun tunnel Vuluunum's unfriendly vans together again each manth and announce Cini and Notim out of that temperature above Kumim really formally  (new: boy, formal, temperature, van)  
  gold: `S=boy^means&NAME.Talutuun T=each+month V=tunnel.PRES.CAN.NEG.Q M=together+again O=NAME.Vuluunum^friend>ADJ>NOT+van.PL ; V=announce M=really+formal>ADV O=NAME.Cini&NAME.Notim FROM=that+temperature ON=NAME.Kumim`  
  model: `S=boy^means&NAME.Talutuun T=each+month V=tunnel.PRES.CAN.NEG.Q M=together+again O=NAME.Vuluunum^friend>ADJ>NOT+van.PL ; V=announce M=really+formal>ADV O=NAME.Cini&NAME.Notim FROM=that+temperature ON=NAME.Kumim`
- many  duets  had  not  been  inevitable  till  few  collective  diorectives  soon  and  he  and  the  illustration  will  not  be  swelling  and  these  subways  and  Liko  can  not  be  at  this  shuttle  abroad  and  how  famous  am  I  (new: duet)  
  gold: `S=many+duet.PL V=be.PAST.PERF.NEG M=soon UNTIL=few+collective+directive.PL A=inevitable ; S=PRON.3sgm&illustrate>ACT V=swell.FUT.PROG.NEG ; S=this+subway.PL&NAME.Liko V=be.PRES.CAN.NEG M=abroad L=this+shuttle ; S=PRON.1sg V=be.PRES A=how+fame>ADJ`  
  model: `S=many+duet.PL V=be.PAST.PERF.NEG M=soon UNTIL=few+collective+directive.PL A=inevitable ; S=PRON.3sgm&illustrate>ACT V=swell.FUT.PROG.NEG ; S=this+subway.PL&NAME.Liko V=be.PRES.CAN.NEG M=abroad L=this+shuttle ; S=PRON.1sg V=be.PRES A=how+fame>ADJ`
- already chuck Lituum and she will not be conmplicated from Ciipi under the diamonds.  (new: chuck)  
  gold: `V=chuck M=already O=NAME.Lituum ; V=complicate.FUT.NEG O=PRON.3sgf FROM=NAME.Ciipi UNDER=diamond.PL`  
  model: `V=chuck M=already O=NAME.Lituum ; V=complicate.FUT.NEG O=PRON.3sgf FROM=NAME.Ciipi UNDER=diamond.PL`
- thanks Mumar  (new: mar)  
  gold: `I=thanks Y=NAME.Mumar`  
  model: `I=thanks Y=NAME.Mumar`
- THE ARM COULD HAVE ASKED  (new: ask)  
  gold: `S=arm V=ask.PAST.PERF.CAN`  
  model: `S=arm V=ask.PAST.PERF.CAN`
- let's  proceed  (new: proceed)  
  gold: `S=PRON.1pl V=proceed.LET`  
  model: `S=PRON.1pl V=proceed.LET`
- should they be the dukes, but you quite together proceeded.  (new: proceed)  
  gold: `S=PRON.3pl V=be.PRES.SHOULD.Q P=duke.PL ; C=but S=PRON.2sg V=proceed.PAST M=quite+together`  
  model: `S=PRON.3pl V=be.PRES.SHOULD.Q P=duke.PL ; C=but S=PRON.2sg V=proceed.PAST M=quite+together`
- Hurl Her Haigher Globalisation.  (new: globalisation, higher, hurl)  
  gold: `V=hurl O=PRON.3sgf^higher+globalisation`  
  model: `V=hurl O=PRON.3sgf^high>MORE+globalisation`
