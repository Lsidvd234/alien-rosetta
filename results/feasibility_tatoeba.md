# Real-English feasibility (`scripts/book_feasibility.py`)

Source: 60,000 sentences (tatoeba). Strict filter: 3-15 words, simple prose, the reader finds no issue / no guess / no unknown word.

| step | sentences | share of all |
|---|---|---|
| sentences | 60,000 | 100.0% |
| too long / short | 2,622 | 4.4% |
| digits / symbols | 1,690 | 2.8% |
| simple prose | 55,688 | 92.8% |
| unknown word | 6,718 | 11.2% |
| reader issue | 12,868 | 21.4% |
| reader had to guess (note) | 20,109 | 33.5% |
| render failed | 0 | 0.0% |
| ACCEPTED | 15,993 | 26.7% |

Accepted: 15,993 = **26.7%** of the sentences tried (= 20.0% of an 80k training set from this sample).

Most frequent unknown words: as (750), anymore (168), interested (100), killed (92), others (60), ones (59), kill (58), willing (52), fired (48), shopping (46), either (42), forward (41), fishing (36), ordered (35), smoking (31), nor (31), forced (30), sex (30), o'clock (26), murder (26), dating (23), handed (23), whenever (22), sooner (22), stepped (21)

## Random sample of 30 accepted sentences (check: is the meaning right?)

1. I haven't read it!  
   `S=PRON.1sg V=read.PRES.PERF.NEG O=PRON.3sgn`
2. Yanni washed the figs.  
   `S=NAME.Yanni V=wash.PAST O=fig.PL`
3. Who doesn't admire them?  
   `S=who V=admire.PRES.NEG O=PRON.3pl`
4. You guys can't scare me.  
   `S=PRON.2pl V=scare.PRES.CAN.NEG O=PRON.1sg`
5. Our children really like animals.  
   `S=PRON.1pl^child.PL V=like.PRES M=really O=animal.PL`
6. Yanni was really scared.  
   `V=scare.PAST M=really O=NAME.Yanni`
7. Tom was surprised by the sight.  
   `S=sight V=surprise.PAST O=NAME.Tom`
8. The flood caused material losses.  
   `S=flood V=cause.PAST O=material_loss.PL`
9. You should not be going.  
   `S=PRON.2sg V=go.PRES.PROG.SHOULD.NEG`
10. What's wrong with me? Why don't you like me?  
   `S=what V=be.PRES WITH=PRON.1sg A=wrong ; S=PRON.2sg V=like.PRES.NEG M=why O=PRON.1sg`
11. No one has called me today.  
   `I=no S=one V=call.PRES.PERF M=today O=PRON.1sg`
12. Let's meet next week.  
   `S=PRON.1pl T=next+week V=meet.LET`
13. He has something in his hands.  
   `S=PRON.3sgm V=have.PRES O=something L=PRON.3sgm^hand.PL`
14. They're too noisy; I can't concentrate.  
   `S=PRON.3pl V=be.PRES A=too+noise>ADJ ; S=PRON.1sg V=concentrate.PRES.CAN.NEG`
15. Tom is going to beat you.  
   `S=NAME.Tom V=beat.FUT O=PRON.2sg`
16. Tom is mentioned in the book.  
   `V=mention.PRES O=NAME.Tom L=book`
17. I'm allergic to shellfish.  
   `S=PRON.1sg V=be.PRES TO=shellfish A=allergic`
18. Doesn't Tom ever wear a suit?  
   `S=NAME.Tom V=wear.PRES.NEG.Q M=ever O=suit`
19. We had some fun.  
   `S=PRON.1pl V=have.PAST O=some+fun`
20. He was satisfied with the interview.  
   `V=satisfy.PAST O=PRON.3sgm WITH=interview>ACT`
21. Watch out for cats.  
   `V=watch M=out FOR=cat.PL`
22. I didn't attack anybody.  
   `S=PRON.1sg V=attack.PAST.NEG O=anybody`
23. That's a whole other rabbit hole.  
   `S=that V=be.PRES P=whole+other+rabbit_hole`
24. I've found a new job.  
   `S=PRON.1sg V=find.PRES.PERF O=new+job`
25. Could you tell me all the details?  
   `S=PRON.2sg V=tell.PAST.CAN.Q O=all+detail.PL TO=PRON.1sg`
26. Those times are long gone.  
   `S=that+time.PL V=be.PRES A=long+gone`
27. What did Tom know?  
   `S=NAME.Tom V=know.PAST O=what`
28. I like your country a lot.  
   `S=PRON.1sg V=like.PRES M=much O=PRON.2sg^country`
29. I don't want a wife.  
   `S=PRON.1sg V=want.PRES.NEG O=wife`
30. Why don't we try calling Tom?  
   `Y=NAME.Tom S=PRON.1pl V=try.PRES.NEG M=why O=call>ACT`
