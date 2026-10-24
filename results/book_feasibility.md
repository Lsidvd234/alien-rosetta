# Real-English feasibility (`scripts/book_feasibility.py`)

Source: 15,765 sentences from the 8,000 FineWeb-Edu replay passages. Strict filter: 3-15 words, simple prose, the reader finds no issue / no guess / no unknown word.

| step | sentences | share of all |
|---|---|---|
| sentences | 15,765 | 100.0% |
| too long / short | 10,181 | 64.6% |
| digits / symbols | 1,727 | 11.0% |
| simple prose | 3,857 | 24.5% |
| unknown word | 1,266 | 8.0% |
| reader issue | 1,178 | 7.5% |
| reader had to guess (note) | 976 | 6.2% |
| render failed | 0 | 0.0% |
| ACCEPTED | 437 | 2.8% |

Accepted sentences would be **0.5%** of an 80k training set.

Most frequent unknown words: as (181), however (18), throughout (13), others (12), researchers (12), findings (10), simply (8), thousands (7), either (7), bursa (6), drugs (6), etc (6), ones (5), completed (5), warming (5), processing (5), currently (5), recorded (4), interested (4), experiencing (4), blocked (4), placed (4), monarch (4), nobles (4), closely (4)

## Random sample of 30 accepted sentences (check: is the meaning right?)

1. Eliminate unnecessary tasks.  
   `V=eliminate O=necessary>NOT+task.PL`
2. Dozens of people did.  
   `S=people.PL^dozen.PL V=do.PAST`
3. Rapid, rhythmic, involuntary movements of the eyeball are common.  
   `S=eyeball^rapid+rhythm>ADJ+voluntary>NOT+move>ACT.PL V=be.PRES A=common`
4. Stories can be fact or fiction, and prose or poetry.  
   `S=story.PL V=be.PRES.CAN P=fact|fiction|prose|poetry`
5. A truly successful study of race should be made in a laboratory of sorts.  
   `S=race^successful+study V=be.PRES.SHOULD M=true>ADV L=sort.PL^laboratory A=made`
6. A pap smear can detect the presence of cancer cells at an early stage.  
   `S=pap_smear V=detect.PRES.CAN O=cancer_cell.PL^presence L=early+stage`
7. In our continent, unfortunately, that is not always the case.  
   `S=that V=be.PRES.NEG M=fortunate>NOT>ADV+always L=PRON.1pl^continent P=case`
8. Please click here to access the full paper.  
   `I=please V=click M=here O=full+paper TO=access`
9. What do you think?  
   `S=PRON.2sg V=think.PRES O=what`
10. It contains technical articles on sound, vibration, speech and other topics.  
   `S=PRON.3sgn V=contain.PRES O=technical+article.PL ON=sound&vibrate>ACT&speech&other+topic.PL`
11. So it is with birds.  
   `C=so S=PRON.3sgn V=be.PRES WITH=bird.PL`
12. It's a plane!  
   `S=PRON.3sgn V=be.PRES P=plane`
13. Subsequent activities will not take a long time.  
   `S=subsequent+active>QUAL.PL V=take.FUT.NEG O=long+time`
14. Their response will be second nature.  
   `S=PRON.3pl^response V=be.FUT P=second+nature`
15. These tumors are very rare in children.  
   `S=this+tumor.PL V=be.PRES L=child.PL A=very+rare`
16. What is the setting?  
   `S=setting V=be.PRES P=what`
17. Such collaboration will open up new horizons for manufacturing.  
   `S=such+collaborate>ACT V=open.FUT M=up O=new+horizon.PL FOR=manufacture>ACT`
18. What experience have you had with educational tools and resources on the Web?  
   `S=PRON.2sg V=have.PRES.PERF O=what+experience WITH=educational+tool.PL&resource.PL ON=web`
19. Life must have been very hard.  
   `S=life V=be.PRES.PERF.MUST A=very+hard`
20. The event remains one of America's greatest environmental disasters.  
   `S=event V=remain.PRES O=NAME.America^greatest+environment>ADJ+disaster.PL^one`
21. Consider a shop.  
   `V=consider O=shop`
22. TITLE contains plain text and entities; it may not contain other markup.  
   `S=title V=contain.PRES O=plain+text&entity.PL ; S=PRON.3sgn V=contain.PRES.MAY.NEG O=other+markup`
23. I hear this question every day in my practice.  
   `S=PRON.1sg T=every+day V=hear.PRES O=this+question L=PRON.1sg^practice`
24. What dangers will they recognise?  
   `S=PRON.3pl V=recognise.FUT O=what+danger.PL`
25. More information is available here.  
   `S=more+inform>ACT V=be.PRES M=here A=avail>ABLE`
26. So a solution is in the works.  
   `C=so S=solution V=be.PRES L=works`
27. Injuries on school grounds are fairly common.  
   `S=injury.PL V=be.PRES M=fair>ADV ON=school_ground.PL A=common`
28. For the most part he succeeds in both efforts.  
   `S=PRON.3sgm V=succeed.PRES FOR=most+part L=both+effort.PL`
29. Details about the free population were not specified in the count.  
   `V=specify.PAST.NEG O=detail.PL ABOUT=free+population L=count>ACT`
30. Viral respiratory infections often worsen asthma.  
   `S=viral+respiratory+infect>ACT.PL V=worsen.PRES M=often O=asthma`
