# --rep-everyday vs the English reader (`analysis/everyday_check.py`)

3000 generated examples each, switch off / on (same seed). Agreement = the reader's analysis of the clean
English equals the generator's analysis exactly.

| examples | off: agree (n) | on: agree (n) |
|---|---|---|
| all | 0.93 (3000) | 0.92 (3000) |
| English name | 0.67 (3) | 0.89 (754) |
| by | - | 0.73 (30) |
| det enough | - | 0.81 (26) |
| det first | - | 0.93 (59) |
| det other | - | 0.85 (53) |
| det own | - | 0.88 (43) |
| det second | - | 0.95 (66) |
| det third | - | 0.90 (60) |
| how: ahead | - | 0.86 (14) |
| how: almost | - | 1.00 (1) |
| how: certainly | - | 0.75 (4) |
| how: even | - | 0.83 (6) |
| how: everywhere | - | 0.75 (12) |
| how: finally | - | 1.00 (4) |
| how: forever | - | 0.86 (29) |
| how: indeed | - | 1.00 (3) |
| how: inside | - | 0.82 (28) |
| how: just | - | 1.00 (7) |
| how: maybe | - | 0.89 (9) |
| how: once | - | 0.93 (29) |
| how: outside | - | 0.88 (17) |
| how: perhaps | - | 0.75 (4) |
| how: probably | - | 0.89 (9) |
| how: really | 0.90 (51) | 0.89 (53) |
| how: somewhere | - | 0.85 (20) |
| how: suddenly | - | 0.80 (5) |
| how: twice | - | 0.84 (31) |
| particle | 0.98 (54) | 0.89 (85) |
| people | - | 0.89 (9) |
| reflexive | - | 0.83 (23) |
| time | 1.00 (1) | 1.00 (5) |
| verb do | - | 1.00 (20) |
| verb have | - | 0.90 (29) |
| verb like | 1.00 (3) | 0.94 (18) |

Disagreements (up to 4 per feature): English / generator / reader

- English name: `Let us pop that Neil couldn't be in front of those records upstairs`  
  gen `S=PRON.1pl V=pop.LET ; C=that S=NAME.Neil V=be.PAST.CAN.NEG M=upstairs FRONT=that+record.PL`  
  reader `S=PRON.1pl V=pop.LET O=that ; S=NAME.Neil V=be.PAST.CAN.NEG M=upstairs FRONT=that+record.PL`
- how: certainly: `Taco and Takal must be unbeaten and do not certainly whine Vike and these identical organisms please!`  
  gen `S=NAME.Taco&NAME.Takal V=be.PRES.MUST A=unbeaten ; I=please V=whine.NEG M=certainly O=NAME.Vike&this+identical+organism.PL`  
  reader `S=NAME.Taco&NAME.Takal V=be.PRES.MUST A=unbeaten ; I=please S=NAME.Taco&NAME.Takal V=whine.PRES.MUST.NEG M=certainly O=NAME.Vike&this+identical+organism.PL`
- how: really: `Before Pavokiim isn't hushing, really critically.`  
  gen `C=before S=NAME.Pavokiim V=hush.PRES.PROG.NEG ; M=really+critical>ADV`  
  reader `C=before S=NAME.Pavokiim V=hush.PRES.PROG.NEG M=really+critical>ADV`
- English name: `Thanks, and there could be the practical semi within the street, and Mia's really slow term will be mailed`  
  gen `I=thanks ; S=practice>ADJ+semi V=be.PAST.CAN L=street ; V=mail.FUT O=NAME.Mia^really+slow+term`  
  reader `I=thanks ; S=practice>ADJ+semi V=be.PAST.CAN L=street ; S=practice>ADJ+semi V=mail.FUT O=NAME.Mia^really+slow+term`
- by: `Recognize the black questionable net of the trainer till Nancy by the portion of the someone.`  
  gen `V=recognize O=train>AGENT^black+questionable+net UNTIL=NAME.Nancy NEAR=someone^portion`  
  reader `V=recognize O=someone^train>AGENT^black+questionable+net UNTIL=NAME.Nancy NEAR=portion`
- English name: `Recognize the black questionable net of the trainer till Nancy by the portion of the someone.`  
  gen `V=recognize O=train>AGENT^black+questionable+net UNTIL=NAME.Nancy NEAR=someone^portion`  
  reader `V=recognize O=someone^train>AGENT^black+questionable+net UNTIL=NAME.Nancy NEAR=portion`
- English name: `He will be giving this giveaway and our fishery and Mary sell their husband our vice on our flag?`  
  gen `S=PRON.3sgm V=give.FUT.PROG O=this+giveaway ; S=PRON.1pl^fishery&NAME.Mary V=sell.PRES.Q O=PRON.1pl^vice ON=PRON.1pl^flag TO=PRON.3pl^husband`  
  reader `S=PRON.3sgm V=give.FUT.PROG ; S=this+giveaway&PRON.1pl^fishery&NAME.Mary V=sell.PRES.Q O=PRON.1pl^vice ON=PRON.1pl^flag TO=PRON.3pl^husband`
- how: really: `Without the queen's carbons, lure Keyur underneath you all really continually and Leon and the minister's company don't only reprise Martin.`  
  gen `V=lure M=really+continual>ADV O=NAME.Keyur WITHOUT=queen^carbon.PL UNDER=PRON.2pl ; S=NAME.Leon&minister^company V=reprise.PRES.NEG M=only O=NAME.Martin`  
  reader `S=carbon.PL V=lure.PRES WITHOUT=queen^ ; S=NAME.Keyur&NAME.Leon&minister^company V=reprise.PRES.NEG M=really+continual>ADV+only O=NAME.Martin UNDER=PRON.2pl`
- det enough: `Ah chairmen, or enough most said worldwide blind modern climaxes will have dealt you with another human with some underpants that Martin will not be this most extreme correspondence for the most same disguises through the communication`  
  gen `I=ah Y=chair_man.PL ; C=or S=enough+most+said+worldwide+blind+modern+climax.PL V=deal.FUT.PERF O=PRON.2sg WITH=another+human WITH=some+underpants ; C=that S=NAME.Martin V=be.FUT.NEG FOR=most+same+disguise.PL THROUGH=communicate>ACT P=this+most+extreme+correspond>ACT`  
  reader `I=ah S=chair_man.PL|enough+most+said+worldwide+blind+modern+climax.PL V=deal.FUT.PERF O=PRON.2sg WITH=another+human WITH=some+underpants ; C=that S=NAME.Martin V=be.FUT.NEG FOR=most+same+disguise.PL THROUGH=communicate>ACT P=this+most+extreme+correspond>ACT`
- how: forever: `It or the employment can commute beneath the underground car about the conscious things forever, but please do not fuse tonight.`  
  gen `S=PRON.3sgn|employ>ACT V=commute.PRES.CAN M=forever UNDER=underground+car ABOUT=conscious+thing.PL ; C=but I=please V=fuse.NEG M=tonight`  
  reader `S=PRON.3sgn|employ>ACT V=commute.PRES.CAN M=forever UNDER=underground+car ABOUT=conscious+thing.PL ; C=but I=please S=PRON.3sgn|employ>ACT V=fuse.PRES.CAN.NEG M=tonight`
- how: maybe: `It and Laatil did not crash, so always maybe agree`  
  gen `S=PRON.3sgn&NAME.Laatil V=crash.PAST.NEG ; C=so V=agree M=always+maybe`  
  reader `S=PRON.3sgn&NAME.Laatil V=crash.PAST.NEG M=so+always+maybe O=agree>ACT`
- by: `You all will be more given by the scissors than it`  
  gen `S=PRON.2pl V=be.FUT NEAR=scissors THAN=PRON.3sgn A=more+given`  
  reader `S=PRON.2pl V=be.FUT A=more+given ; NEAR=scissors THAN=PRON.3sgn`
- reflexive: `We taught ourselves in this architecture everywhere, and the tough brigade mustn't be scared with the good braces`  
  gen `S=PRON.1pl V=teach.PAST M=everywhere O=PRON.1pl L=this+architecture ; V=scare.PRES.MUST.NEG O=tough+brigade WITH=good+brace.PL`  
  reader `S=PRON.1pl V=teach.PAST M=everywhere O=PRON.1pl L=this+architecture ; S=PRON.1pl V=scare.PRES.MUST.NEG O=tough+brigade WITH=good+brace.PL`
- how: everywhere: `We taught ourselves in this architecture everywhere, and the tough brigade mustn't be scared with the good braces`  
  gen `S=PRON.1pl V=teach.PAST M=everywhere O=PRON.1pl L=this+architecture ; V=scare.PRES.MUST.NEG O=tough+brigade WITH=good+brace.PL`  
  reader `S=PRON.1pl V=teach.PAST M=everywhere O=PRON.1pl L=this+architecture ; S=PRON.1pl V=scare.PRES.MUST.NEG O=tough+brigade WITH=good+brace.PL`
- det own: `Alice sexually increased and how excellent are the priest's regulatory different centuries and are you all not her own children with a vision next morning and plan Kevin please`  
  gen `S=NAME.Alice V=increase.PAST M=sexual>ADV ; S=priest^regulatory+different+century.PL V=be.PRES A=how+excellent ; S=PRON.2pl T=next+morning V=be.PRES.NEG.Q WITH=vision P=PRON.3sgf^own+child.PL ; I=please V=plan O=NAME.Kevin`  
  reader `S=NAME.Alice V=increase.PAST M=sexual>ADV ; S=priest^regulatory+different+century.PL V=be.PRES A=how+excellent ; I=please Y=NAME.Kevin S=PRON.2pl T=next+morning V=be.PRES.NEG.Q WITH=vision P=PRON.3sgf^own+child.PL&plan>ACT`
- reflexive: `I must not be shifting myself from three really digital minis behind the traffic after this so pure physiology will be so timeless from this alignment`  
  gen `S=PRON.1sg V=shift.PRES.PROG.MUST.NEG O=PRON.1sg FROM=three+really+digit>ADJ+mini.PL BEHIND=traffic ; C=after S=this+so+pure+physiology V=be.FUT FROM=this+align>ACT A=so+timeless`  
  reader `S=PRON.1sg V=shift.PRES.PROG.MUST.NEG O=PRON.1sg ; S=so+pure+physiology V=be.FUT FROM=three+really+digit>ADJ+mini.PL BEHIND=traffic AFTER=this FROM=this+align>ACT A=so+timeless`
- reflexive: `Does the lower cosmetic attractive healthy tenderness roll every evening, and it was yourselves that you all did assemble by email`  
  gen `S=low>MORE+cosmetic+attractive+health>ADJ+tender>QUAL T=every+evening V=roll.PRES.Q ; S=PRON.2pl V=assemble.PAST O=PRON.2pl WITH=email`  
  reader `V=do.PRES O=lower>ACT ; S=cosmetic+attractive+health>ADJ+tender>QUAL T=every+evening V=roll.PRES ; S=PRON.2pl V=assemble.PAST O=PRON.2pl WITH=email`
- by: `Does the lower cosmetic attractive healthy tenderness roll every evening, and it was yourselves that you all did assemble by email`  
  gen `S=low>MORE+cosmetic+attractive+health>ADJ+tender>QUAL T=every+evening V=roll.PRES.Q ; S=PRON.2pl V=assemble.PAST O=PRON.2pl WITH=email`  
  reader `V=do.PRES O=lower>ACT ; S=cosmetic+attractive+health>ADJ+tender>QUAL T=every+evening V=roll.PRES ; S=PRON.2pl V=assemble.PAST O=PRON.2pl WITH=email`
- how: twice: `Irritate Vuni, and lay Tocam's chilies and Nora downstairs twice!`  
  gen `V=irritate O=NAME.Vuni ; V=lay M=downstairs+twice O=NAME.Tocam^chilli.PL&NAME.Nora`  
  reader `V=irritate O=NAME.Vuni ; V=lay M=downstairs+twice O=NAME.Tocam^chili.PL&NAME.Nora`
- how: inside: `The proceeds must give Tina around the marc because there can not be the trio and Leon at his mountain and you all must not have offered the merle to Tracy out of four arks and Novikum must have earned us inside`  
  gen `S=proceeds V=give.PRES.MUST O=NAME.Tina AROUND=marc ; C=because S=trio&NAME.Leon V=be.PRES.CAN.NEG L=PRON.3sgm^mountain ; S=PRON.2pl V=offer.PRES.PERF.MUST.NEG O=merle TO=NAME.Tracy FROM=four+ark.PL ; S=NAME.Novikum V=earn.PRES.PERF.MUST M=inside O=PRON.1pl`  
  reader `S=proceeds V=give.PRES.MUST O=NAME.Tina AROUND=marc ; C=because S=trio V=be.PRES.CAN.NEG ; S=NAME.Leon&PRON.2pl V=offer.PRES.PERF.MUST.NEG O=merle L=PRON.3sgm^mountain TO=NAME.Tracy FROM=four+ark.PL ; S=NAME.Novikum V=earn.PRES.PERF.MUST M=inside O=PRON.1pl`
- how: really: `Please, so there were not the little decent wallets in the sporty cemeteries; yesterday that saw is realized during that tod until that hormone really genuinely`  
  gen `I=please ; C=so S=little+decent+wallet.PL V=be.PAST.NEG L=sporty+cemetery.PL ; V=realize.PRES M=yesterday+really+genuine>ADV O=that+saw DURING=that+tod UNTIL=that+hormone`  
  reader `I=please ; C=so S=little V=be.PAST.NEG ; S=decent+wallet.PL L=sporty+cemetery.PL ; V=realize.PRES M=yesterday+really+genuine>ADV O=that+saw DURING=that+tod UNTIL=that+hormone`
- particle: `While I and the old cedar must talk for the pizzas, please go away after that japan!`  
  gen `C=while S=PRON.1sg&old+cedar V=talk.PRES.MUST FOR=pizza.PL ; I=please V=go M=away AFTER=that+japan`  
  reader `C=while I=please S=PRON.1sg&old+cedar V=talk.PRES.MUST M=away O=go>ACT FOR=pizza.PL AFTER=that+japan`
- reflexive: `It could stir itself personally and even add it`  
  gen `S=PRON.3sgn V=stir.PAST.CAN M=person>ADJ>ADV O=PRON.3sgn ; V=add M=even O=PRON.3sgn`  
  reader `S=PRON.3sgn V=stir.PAST.CAN M=person>ADJ>ADV O=PRON.3sgn ; S=PRON.3sgn V=add.PAST.CAN M=even O=PRON.3sgn`
- how: even: `It could stir itself personally and even add it`  
  gen `S=PRON.3sgn V=stir.PAST.CAN M=person>ADJ>ADV O=PRON.3sgn ; V=add M=even O=PRON.3sgn`  
  reader `S=PRON.3sgn V=stir.PAST.CAN M=person>ADJ>ADV O=PRON.3sgn ; S=PRON.3sgn V=add.PAST.CAN M=even O=PRON.3sgn`
- verb have: `While my playoff won't be with his permissions, have this socialization.`  
  gen `C=while S=PRON.1sg^playoff V=be.FUT.NEG WITH=PRON.3sgm^permission.PL ; V=have O=this+socialize>ACT`  
  reader `C=while S=PRON.1sg^playoff V=be.FUT.NEG WITH=PRON.3sgm^ ; S=permission.PL V=have.PRES O=this+socialize>ACT`
- how: once: `Techniques or Lia, types hadn't pinged and the value must mince bunches and Timecu once and these doodles could be on this more healthy way last summer?`  
  gen `S=type.PL V=ping.PAST.PERF.NEG O=technique.PL|NAME.Lia ; S=value V=mince.PRES.MUST M=once O=bunch.PL&NAME.Timecu ; S=this+doodle.PL T=last+summer V=be.PAST.CAN.Q ON=this+more+health>ADJ+way`  
  reader `S=type.PL V=ping.PAST.PERF.NEG O=technique.PL|NAME.Lia ; S=value V=mince.PRES.MUST ; S=bunch.PL&NAME.Timecu&this+doodle.PL T=last+summer V=be.PAST.CAN.Q M=once ON=this+more+health>ADJ+way`
- verb like: `Didn't Lituuman pray till the fertility away that I might be this leash after eight facilities; the racial nonsense and the alcoholic organization might be upon the meeting; he was liking Pora and Cuucu a lot!`  
  gen `S=NAME.Lituuman V=pray.PAST.NEG.Q M=away UNTIL=fertile>QUAL ; C=that S=PRON.1sg V=be.PRES.MAY AFTER=eight+facility.PL P=this+leash ; S=race>ADJ+nonsense&alcohol>ADJ+organize>ACT V=be.PRES.MAY ON=meeting ; S=PRON.3sgm V=like.PAST.PROG M=much O=NAME.Pora&NAME.Cuucu`  
  reader `S=NAME.Lituuman V=pray.PAST.NEG.Q M=away O=that UNTIL=fertile>QUAL ; S=PRON.1sg V=be.PRES.MAY AFTER=eight+facility.PL P=this+leash ; S=race>ADJ+nonsense&alcohol>ADJ+organize>ACT V=be.PRES.MAY ON=meeting ; S=PRON.3sgm V=like.PAST.PROG M=much O=NAME.Pora&NAME.Cuucu`
- particle: `Didn't Lituuman pray till the fertility away that I might be this leash after eight facilities; the racial nonsense and the alcoholic organization might be upon the meeting; he was liking Pora and Cuucu a lot!`  
  gen `S=NAME.Lituuman V=pray.PAST.NEG.Q M=away UNTIL=fertile>QUAL ; C=that S=PRON.1sg V=be.PRES.MAY AFTER=eight+facility.PL P=this+leash ; S=race>ADJ+nonsense&alcohol>ADJ+organize>ACT V=be.PRES.MAY ON=meeting ; S=PRON.3sgm V=like.PAST.PROG M=much O=NAME.Pora&NAME.Cuucu`  
  reader `S=NAME.Lituuman V=pray.PAST.NEG.Q M=away O=that UNTIL=fertile>QUAL ; S=PRON.1sg V=be.PRES.MAY AFTER=eight+facility.PL P=this+leash ; S=race>ADJ+nonsense&alcohol>ADJ+organize>ACT V=be.PRES.MAY ON=meeting ; S=PRON.3sgm V=like.PAST.PROG M=much O=NAME.Pora&NAME.Cuucu`
- how: inside: `Why shouldn't the permission institute the diet, and hint about those mos inside, and Puuramon and the third hellfire would be sweetest and anterior, and which client mustn't this bacterial confident concurrent proud course finance`  
  gen `S=permission V=institute.PRES.SHOULD.NEG M=why O=diet ; V=hint M=inside ABOUT=that+mo.PL ; S=NAME.Puuramon&third+hellfire V=be.PRES.WOULD A=sweet>MOST+anterior ; S=this+bacterial+confident+concurrent+proud+course V=finance.PRES.MUST.NEG O=which+client`  
  reader `S=permission V=institute.PRES.SHOULD.NEG M=why+inside O=diet&hint>ACT ABOUT=that+mo.PL ; S=NAME.Puuramon&third+hellfire V=be.PRES.WOULD A=sweet>MOST+anterior ; S=this+bacterial+confident+concurrent+proud+course V=finance.PRES.MUST.NEG O=which+client`
- det third: `Why shouldn't the permission institute the diet, and hint about those mos inside, and Puuramon and the third hellfire would be sweetest and anterior, and which client mustn't this bacterial confident concurrent proud course finance`  
  gen `S=permission V=institute.PRES.SHOULD.NEG M=why O=diet ; V=hint M=inside ABOUT=that+mo.PL ; S=NAME.Puuramon&third+hellfire V=be.PRES.WOULD A=sweet>MOST+anterior ; S=this+bacterial+confident+concurrent+proud+course V=finance.PRES.MUST.NEG O=which+client`  
  reader `S=permission V=institute.PRES.SHOULD.NEG M=why+inside O=diet&hint>ACT ABOUT=that+mo.PL ; S=NAME.Puuramon&third+hellfire V=be.PRES.WOULD A=sweet>MOST+anterior ; S=this+bacterial+confident+concurrent+proud+course V=finance.PRES.MUST.NEG O=which+client`
- det other: `Where will that historical superhuman bespoke and magical word have to tag that address and they and Mia had been too afraid and it is the really purple and indigenous services of the other informant that do lessen.`  
  gen `S=that+historical+superhuman+bespoke+magic>ADJ+word V=tag.FUT.MUST M=where O=that+address ; S=PRON.3pl&NAME.Mia V=be.PAST.PERF A=too+afraid ; S=other+inform>AGENT^really+purple+indigenous+service.PL V=lessen.PRES`  
  reader `S=that+historical+superhuman+bespoke+magic>ADJ+word V=tag.FUT.MUST M=where ; S=that+address&PRON.3pl&NAME.Mia V=be.PAST.PERF A=too+afraid ; S=other+inform>AGENT^really+purple+indigenous+service.PL V=lessen.PRES`
- how: outside: `Nora will not be multiple during it on this sample outside.`  
  gen `S=NAME.Nora V=be.FUT.NEG M=outside DURING=PRON.3sgn ON=this+sample A=multiple`  
  reader `S=NAME.Nora V=be.FUT.NEG DURING=PRON.3sgn ON=this+sample A=multiple+outside`
- det third: `The manipulators of the third dealer are not recovering the transponder of the sister after the defendant of the contributor`  
  gen `S=third+deal>AGENT^manipulate>AGENT.PL V=recover.PRES.PROG.NEG O=sister^transponder AFTER=contribute>AGENT^defend>AGENT`  
  reader `S=third+deal>AGENT^manipulate>AGENT.PL V=recover.PRES.PROG.NEG O=contribute>AGENT^sister^transponder AFTER=defend>AGENT`
- how: twice: `He can say enough eighty traffic in some selfless peace twice last month?`  
  gen `S=PRON.3sgm T=last+month V=say.PRES.CAN.Q M=twice O=enough+eighty+traffic L=some+selfless+peace`  
  reader `S=PRON.3sgm V=say.PRES.CAN O=enough+eighty ; S=traffic T=last+month M=twice L=some+selfless+peace`
- det enough: `He can say enough eighty traffic in some selfless peace twice last month?`  
  gen `S=PRON.3sgm T=last+month V=say.PRES.CAN.Q M=twice O=enough+eighty+traffic L=some+selfless+peace`  
  reader `S=PRON.3sgm V=say.PRES.CAN O=enough+eighty ; S=traffic T=last+month M=twice L=some+selfless+peace`
- det third: `It was I that did guess her until work behind the third peachy colons please.`  
  gen `S=PRON.1sg V=guess.PAST O=PRON.3sgf ; C=until I=please V=work BEHIND=third+peach>ADJ+colon.PL`  
  reader `I=please S=PRON.1sg V=guess.PAST O=PRON.3sgf UNTIL=work>ACT BEHIND=third+peach>ADJ+colon.PL`
- how: twice: `The so easy dads would audition after the side twice; you must be in front of the disorders, but annoy you all densely`  
  gen `S=so+easy+dad.PL V=audition.PRES.WOULD M=twice AFTER=side ; S=PRON.2sg V=be.PRES.MUST FRONT=order>NOT.PL ; C=but V=annoy M=dense>ADV O=PRON.2pl`  
  reader `S=so+easy+dad.PL V=audition.PRES.WOULD M=twice AFTER=side ; S=PRON.2sg V=be.PRES.MUST FRONT=order>NOT.PL ; C=but S=PRON.2sg V=annoy.PRES.MUST M=dense>ADV O=PRON.2pl`
- det enough: `Let's close the ghastly alliance. let us trump that they are not in front of enough summertime`  
  gen `S=PRON.1pl V=close.LET O=ghastly+alliance ; S=PRON.1pl V=trump.LET ; C=that S=PRON.3pl V=be.PRES.NEG FRONT=enough+summertime`  
  reader `S=PRON.1pl V=close.LET O=ghastly+alliance ; S=PRON.1pl V=trump.LET O=that ; S=PRON.3pl V=be.PRES.NEG FRONT=enough+summertime`
- particle: `Pinipu may be destroyed by Laamil with a fandom, and will Never march inside nine accesses about two very ultimate francs although you should be dismantled by me away, and let us permit the horrors religiously`  
  gen `S=NAME.Laamil V=destroy.PRES.MAY O=NAME.Pinipu WITH=fandom ; S=NAME.Never V=march.FUT.Q L=nine+access.PL ABOUT=two+very+ultimate+franc.PL ; C=although S=PRON.1sg V=dismantle.PRES.SHOULD M=away O=PRON.2sg ; S=PRON.1pl V=permit.LET M=religious>ADV O=horror.PL`  
  reader `S=NAME.Laamil V=destroy.PRES.MAY O=NAME.Pinipu WITH=fandom ; S=NAME.Laamil V=march.FUT M=never L=nine+access.PL ABOUT=two+very+ultimate+franc.PL ; C=although S=PRON.1sg V=dismantle.PRES.SHOULD M=away O=PRON.2sg ; S=PRON.1pl V=permit.LET M=religious>ADV O=horror.PL`
- how: probably: `The manuscript. they might not care without that too strict mainstream although he does not probably obey us or the bye!`  
  gen `S=manuscript ; S=PRON.3pl V=care.PRES.MAY.NEG WITHOUT=that+too+strict+mainstream ; C=although S=PRON.3sgm V=obey.PRES.NEG M=probably O=PRON.1pl|bye`  
  reader `S=manuscript ; S=PRON.3pl V=care.PRES.MAY.NEG O=too+strict+mainstream WITHOUT=that ; C=although S=PRON.3sgm V=obey.PRES.NEG M=probably O=PRON.1pl|bye`
- det enough: `Enough closures might not approve Rapoya and remain Karu behind the his`  
  gen `S=enough+close>ACT.PL V=approve.PRES.MAY.NEG O=NAME.Rapoya ; V=remain O=NAME.Karu BEHIND=hi.PL`  
  reader `None`
- det own: `How adequate will you all be because he was peaceful, traditional and dental around your own weird cents around his vat, then work soon; please usually say each low bumpy and incoherent capital from every basketball.`  
  gen `S=PRON.2pl V=be.FUT A=how+adequate ; C=because S=PRON.3sgm V=be.PAST AROUND=PRON.2sg^own+weird+cent.PL AROUND=PRON.3sgm^vat A=peaceful+tradition>ADJ+dental ; C=then V=work M=soon ; I=please V=say M=usually O=each+low+bumpy+coherent>NOT+capital FROM=every+basketball`  
  reader `S=PRON.2pl V=be.FUT A=how+adequate ; C=because S=PRON.3sgm V=be.PAST M=then+soon AROUND=PRON.2sg^own+weird+cent.PL AROUND=PRON.3sgm^vat P=peaceful+tradition>ADJ+dental+work>ACT ; I=please V=say M=usually O=each+low+bumpy+coherent>NOT+capital FROM=every+basketball`
- particle: `There is Ian at the beams, and which student wouldn't that situation read, and Vivo might have surcharged Mivaki, and don't get me away underneath the lunch`  
  gen `S=NAME.Ian V=be.PRES L=beam.PL ; S=that+situation V=read.PRES.WOULD.NEG O=which+student ; S=NAME.Vivo V=surcharge.PRES.PERF.MAY O=NAME.Mivaki ; V=get.NEG M=away O=PRON.1sg UNDER=lunch`  
  reader `S=NAME.Ian V=be.PRES L=beam.PL ; S=that+situation V=read.PRES.WOULD.NEG O=which+student ; S=NAME.Vivo V=surcharge.PRES.PERF.MAY O=NAME.Mivaki ; S=NAME.Vivo V=get.PRES.PERF.MAY.NEG M=away O=PRON.1sg UNDER=lunch`
- how: forever: `Before Kuyin and Vocaa are godly forever, in the educational batteries`  
  gen `C=before S=NAME.Kuyin&NAME.Vocaa V=be.PRES M=forever A=godly ; L=educational+battery.PL`  
  reader `C=before S=NAME.Kuyin&NAME.Vocaa V=be.PRES M=forever L=educational+battery.PL A=godly`
- det third: `What might they have retailed after the third contexts and the partisanship could be behind the module before sue this affront till Monica around these corks; he shouldn't be a baker during the initiatives.`  
  gen `S=PRON.3pl V=retail.PRES.PERF.MAY O=what ; C=after S=third+context.PL&partisanship V=be.PAST.CAN BEHIND=module ; C=before V=sue O=this+affront UNTIL=NAME.Monica AROUND=this+cork.PL ; S=PRON.3sgm V=be.PRES.SHOULD.NEG DURING=initiative.PL P=bake>AGENT`  
  reader `S=PRON.3pl V=retail.PRES.PERF.MAY O=what AFTER=third+context.PL ; S=partisanship V=be.PAST.CAN BEHIND=module BEFORE=sue>ACT UNTIL=NAME.Monica AROUND=this+cork.PL P=this+affront ; S=PRON.3sgm V=be.PRES.SHOULD.NEG DURING=initiative.PL P=bake>AGENT`
- det own: `She and his very safe blade might not be very funny since our own very tops blazes`  
  gen `S=PRON.3sgf&PRON.3sgm^very+safe+blade V=be.PRES.MAY.NEG SINCE=PRON.1pl^own+very+tops+blaze.PL A=very+fun>ADJ`  
  reader `S=PRON.3sgf&PRON.3sgm^very+safe+blade V=be.PRES.MAY.NEG SINCE=PRON.1pl^own+very+top.PL P=very+fun>ADJ+blaze.PL`
- how: somewhere: `Rumuu should double five gaps or one sixty confusion before one agreement somewhere!`  
  gen `S=NAME.Rumuu V=double.PRES.SHOULD M=somewhere O=five+gap.PL|one+sixty+confuse>ACT BEFORE=one+agree>ACT`  
  reader `S=NAME.Rumuu V=double.PRES.SHOULD O=five+gap.PL|one+sixty ; S=confuse>ACT M=somewhere BEFORE=one+agree>ACT`
- how: somewhere: `She and that fuel have fixed that scrapbook and Naru, and uncertainties or Kira had been most sheer during the remotest portrayal somewhere, and who would she be issuing, and blast.`  
  gen `S=PRON.3sgf&that+fuel V=fix.PRES.PERF O=that+scrapbook&NAME.Naru ; S=certainty>NOT.PL|NAME.Kira V=be.PAST.PERF M=somewhere DURING=remote>MOST+portray>ACT A=most+sheer ; S=PRON.3sgf V=issue.PRES.PROG.WOULD O=who ; V=blast`  
  reader `S=PRON.3sgf&that+fuel V=fix.PRES.PERF ; S=that+scrapbook|NAME.Naru|certainty>NOT.PL|NAME.Kira V=be.PAST.PERF M=somewhere DURING=remote>MOST+portray>ACT A=most+sheer ; S=PRON.3sgf V=issue.PRES.PROG.WOULD O=who ; S=PRON.3sgf V=blast.PRES.PROG.WOULD`
- det second: `How do they lace that really good and mid investigator and this mantel and Yetum might be the second amazing figures every day`  
  gen `S=PRON.3pl V=lace.PRES M=how O=that+really+good+mid+investigate>AGENT ; S=this+mantel&NAME.Yetum T=every+day V=be.PRES.MAY P=second+amazing+figure.PL`  
  reader `S=PRON.3pl V=lace.PRES M=how ; S=that+really+good+mid+investigate>AGENT&this+mantel&NAME.Yetum T=every+day V=be.PRES.MAY P=second+amazing+figure.PL`
- det other: `You may purchase Nico and the other utopia among the statistics and do not issue him after the doodle and what may it not refer.`  
  gen `S=PRON.2sg V=purchase.PRES.MAY O=NAME.Nico&other+utopia BETWEEN=statistic.PL ; V=issue.NEG O=PRON.3sgm AFTER=doodle ; S=PRON.3sgn V=refer.PRES.MAY.NEG O=what`  
  reader `S=PRON.2sg V=purchase.PRES.MAY O=NAME.Nico&other+utopia BETWEEN=statistic.PL ; S=PRON.2sg V=issue.PRES.MAY.NEG O=PRON.3sgm AFTER=doodle ; S=PRON.3sgn V=refer.PRES.MAY.NEG O=what`
- how: once: `You will be beneath the cloud once and it is projected and let us shift the fitch instead of the likely intervention towards the ministerial gears.`  
  gen `S=PRON.2sg V=be.FUT M=once UNDER=cloud ; V=project.PRES O=PRON.3sgn ; S=PRON.1pl V=shift.LET O=fitch FOR=likely+intervention TO=ministerial+gear.PL`  
  reader `S=PRON.2sg V=be.FUT M=once UNDER=cloud ; S=PRON.2sg V=project.PRES O=PRON.3sgn ; S=PRON.1pl V=shift.LET O=fitch FOR=likely+intervention TO=ministerial+gear.PL`
- how: forever: `Oliver must be parsed by Cipe and that adage downstairs, and insert Victor with the rainy bonus today forever`  
  gen `S=NAME.Cipe&that+adage V=parse.PRES.MUST M=downstairs O=NAME.Oliver ; V=insert M=today+forever O=NAME.Victor WITH=rain>ADJ+bonus`  
  reader `S=NAME.Cipe&that+adage V=parse.PRES.MUST M=downstairs O=NAME.Oliver ; S=NAME.Cipe&that+adage V=insert.PRES.MUST M=today+forever O=NAME.Victor WITH=rain>ADJ+bonus`
- det other: `When can the other amphibious spanks eat a translucent and dramatic blood, and whose rooftop will this be, and don't watch these rumours`  
  gen `S=other+amphibious+spank.PL V=eat.PRES.CAN M=when O=translucent+dramatic+blood ; S=this V=be.FUT P=who^rooftop ; V=watch.NEG O=this+rumour.PL`  
  reader `S=other+amphibious+spank.PL V=eat.PRES.CAN M=when O=translucent+dramatic+blood ; S=this V=be.FUT P=who^rooftop ; S=this V=watch.FUT.NEG O=this+rumour.PL`
- det first: `Who will they be, and care since the first lesson since the commuter, and whose chunk was this unless don't always gather Kara yesterday`  
  gen `S=PRON.3pl V=be.FUT P=who ; V=care SINCE=first+lesson SINCE=commute>AGENT ; S=this V=be.PAST P=who^chunk ; C=unless V=gather.NEG M=always+yesterday O=NAME.Kara`  
  reader `S=PRON.3pl V=be.FUT P=who ; S=PRON.3pl V=care.FUT SINCE=first+lesson SINCE=commute>AGENT ; S=this V=be.PAST P=who^chunk ; C=unless V=gather.NEG M=always+yesterday O=NAME.Kara`
- det other: `Can't you and the other evil be inflicting Nicole and the primer a lot, but shrink please`  
  gen `S=PRON.2sg&other+evil V=inflict.PRES.PROG.CAN.NEG.Q M=much O=NAME.Nicole&primer ; C=but I=please V=shrink`  
  reader `S=PRON.2sg&other+evil V=inflict.PRES.PROG.CAN.NEG.Q M=much O=NAME.Nicole&primer ; C=but I=please S=PRON.2sg&other+evil V=shrink.PRES.PROG.CAN`
- how: outside: `Katie will be made by Martin and the warner's hajj outside and the so annual, possible blasphemy might be so renewable`  
  gen `S=NAME.Martin&warn>AGENT^hajj V=make.FUT M=outside O=NAME.Katie ; S=so+annual+possible+blasphemy V=be.PRES.MAY A=so+renewable`  
  reader `S=NAME.Martin V=make.FUT O=NAME.Katie ; S=warn>AGENT^hajj&so+annual+possible+blasphemy V=be.PRES.MAY M=outside A=so+renewable`
- how: twice: `Tomorrow one more fifteen, same, autonomous stranger can go to Ian. might they have wondered it without one crest traditionally. we have achieved them from the coolant twice. deserve every information to another really feasible court quite heavily`  
  gen `S=one+more+fifteen+same+autonomous+stranger V=go.PRES.CAN M=tomorrow TO=NAME.Ian ; S=PRON.3pl V=wonder.PRES.PERF.MAY.Q M=tradition>ADJ>ADV O=PRON.3sgn WITHOUT=one+crest ; S=PRON.1pl V=achieve.PRES.PERF M=twice O=PRON.3pl FROM=coolant ; V=deserve M=quite+heavy>ADV O=every+inform>ACT TO=another+really+feasible+court`  
  reader `S=same+autonomous+stranger V=go.PRES.CAN M=tomorrow O=one+more+fifteen TO=NAME.Ian ; S=PRON.3pl V=wonder.PRES.PERF.MAY.Q M=tradition>ADJ>ADV O=PRON.3sgn WITHOUT=one+crest ; S=PRON.1pl V=achieve.PRES.PERF M=twice O=PRON.3pl FROM=coolant ; V=deserve M=quite+heavy>ADV O=every+inform>ACT TO=another+really+feasible+court`
- how: somewhere: `Were so shallow enrollments her so considerate, invalid, evolutionary bourne inside, and this calcium has been gambled, and her cursor was not flirting somewhere because it would be realized today now`  
  gen `S=so+shallow+enroll>ACT.PL V=be.PAST.Q M=inside P=PRON.3sgf^so+considerate+valid>NOT+evolutionary+bourne ; V=gamble.PRES.PERF O=this+calcium ; S=PRON.3sgf^cursor V=flirt.PAST.PROG.NEG M=somewhere ; C=because V=realize.PRES.WOULD M=today+now O=PRON.3sgn`  
  reader `S=so+shallow+enroll>ACT.PL V=be.PAST.Q M=inside P=PRON.3sgf^so+considerate+valid>NOT+evolutionary+bourne ; S=so+shallow+enroll>ACT.PL V=gamble.PRES.PERF O=this+calcium ; S=PRON.3sgf^cursor V=flirt.PAST.PROG.NEG M=somewhere ; C=because V=realize.PRES.WOULD M=today+now O=PRON.3sgn`
- how: inside: `Were so shallow enrollments her so considerate, invalid, evolutionary bourne inside, and this calcium has been gambled, and her cursor was not flirting somewhere because it would be realized today now`  
  gen `S=so+shallow+enroll>ACT.PL V=be.PAST.Q M=inside P=PRON.3sgf^so+considerate+valid>NOT+evolutionary+bourne ; V=gamble.PRES.PERF O=this+calcium ; S=PRON.3sgf^cursor V=flirt.PAST.PROG.NEG M=somewhere ; C=because V=realize.PRES.WOULD M=today+now O=PRON.3sgn`  
  reader `S=so+shallow+enroll>ACT.PL V=be.PAST.Q M=inside P=PRON.3sgf^so+considerate+valid>NOT+evolutionary+bourne ; S=so+shallow+enroll>ACT.PL V=gamble.PRES.PERF O=this+calcium ; S=PRON.3sgf^cursor V=flirt.PAST.PROG.NEG M=somewhere ; C=because V=realize.PRES.WOULD M=today+now O=PRON.3sgn`
- how: really: `Paruvo can not be underneath this service and Kopenii and Rivum will not correlate really naturally?`  
  gen `S=NAME.Paruvo V=be.PRES.CAN.NEG UNDER=this+service ; S=NAME.Kopenii&NAME.Rivum V=correlate.FUT.NEG.Q M=really+natural>ADV`  
  reader `S=NAME.Paruvo V=be.PRES.CAN.NEG UNDER=this ; S=service&NAME.Kopenii&NAME.Rivum V=correlate.FUT.NEG.Q M=really+natural>ADV`
- how: everywhere: `Topple, then it will be I that am going to advise them and shut Lekinu behind Patrick among the so cold helping everywhere!`  
  gen `V=topple ; C=then S=PRON.1sg V=advise.FUT O=PRON.3pl ; V=shut M=everywhere O=NAME.Lekinu BEHIND=NAME.Patrick BETWEEN=so+cold+helping`  
  reader `V=topple ; C=then S=PRON.1sg V=advise.FUT O=PRON.3pl ; S=PRON.1sg V=shut.FUT M=everywhere O=NAME.Lekinu BEHIND=NAME.Patrick BETWEEN=so+cold+helping`
- how: perhaps: `They might perhaps throw it to Velumin; what will she causeway that Tunuu did sell the arithmetic of the neighbour to the wives of the owner behind the hackney of the thrower last day; you were helping the local and industrial backseat?`  
  gen `S=PRON.3pl V=throw.PRES.MAY M=perhaps O=PRON.3sgn TO=NAME.Velumin ; S=PRON.3sgf V=causeway.FUT O=what ; C=that S=NAME.Tunuu T=last+day V=sell.PAST O=neighbour^arithmetic TO=owner^wife.PL BEHIND=throw>AGENT^hackney ; S=PRON.2sg V=help.PAST.PROG.Q O=local+industry>ADJ+backseat`  
  reader `S=PRON.3pl V=throw.PRES.MAY M=perhaps O=PRON.3sgn TO=NAME.Velumin ; S=PRON.3sgf V=causeway.FUT O=what ; C=that S=NAME.Tunuu T=last+day V=sell.PAST O=throw>AGENT^owner^neighbour^arithmetic TO=wife.PL BEHIND=hackney ; S=PRON.2sg V=help.PAST.PROG.Q O=local+industry>ADJ+backseat`
- people: `You or both vague securities had also gone over soon, and please trust Culuukaam and your methadone, and she or the club sees the worse carrier during the proteins, and is there we or Mine near those people for them`  
  gen `S=PRON.2sg|both+vague+security.PL V=go.PAST.PERF M=also+over+soon ; I=please V=trust O=NAME.Culuukaam&PRON.2sg^methadone ; S=PRON.3sgf|club V=see.PRES O=worse+carry>AGENT DURING=protein.PL ; S=PRON.1pl|NAME.Mine V=be.PRES.Q NEAR=that+people.PL FOR=PRON.3pl`  
  reader `S=PRON.2sg|both+vague+security.PL V=go.PAST.PERF M=also+over+soon ; I=please V=trust ; S=NAME.Culuukaam|PRON.2sg^methadone|PRON.3sgf|club V=see.PRES O=worse+carry>AGENT DURING=protein.PL ; S=PRON.1pl|PRON.1sg^ V=be.PRES.Q NEAR=that+people.PL FOR=PRON.3pl`
- how: suddenly: `Tuyal was suddenly telling these dukes this really better demo on these lovers, and live Vaaletiin or Karen quite recently, and it may not be most younger and stark!`  
  gen `S=NAME.Tuyal V=tell.PAST.PROG M=suddenly O=this+really+better+demo ON=this+love>AGENT.PL TO=this+duke.PL ; V=live M=quite+recent>ADV O=NAME.Vaaletiin|NAME.Karen ; S=PRON.3sgn V=be.PRES.MAY.NEG A=most+younger+stark`  
  reader `S=NAME.Tuyal V=tell.PAST.PROG M=suddenly O=this+really+better+demo TO=this+duke.PL ; S=NAME.Vaaletiin&NAME.Karen&PRON.3sgn V=be.PRES.MAY.NEG M=quite+recent>ADV ON=this+love>AGENT.PL&live>ACT A=most+younger+stark`
- by: `That music isn't by that too speedy intent, and Puuvacaa or one user couldn't think the spectrometry onto Katie a lot`  
  gen `S=that+music V=be.PRES.NEG NEAR=that+too+speed>ADJ+intent ; S=NAME.Puuvacaa|one+use>AGENT V=think.PAST.CAN.NEG M=much O=spectrometry ON=NAME.Katie`  
  reader `S=that+music V=be.PRES.NEG NEAR=that P=too+speed>ADJ+intent ; S=NAME.Puuvacaa|one+use>AGENT V=think.PAST.CAN.NEG M=much O=spectrometry ON=NAME.Katie`
- how: everywhere: `This week because every harmless place should not pass Nucum her for Mila near no warm tow, and designate you all everywhere`  
  gen `T=this+week ; C=because S=every+harmless+place V=pass.PRES.SHOULD.NEG O=PRON.3sgf FOR=NAME.Mila NEAR=no+warm+tow TO=NAME.Nucum ; V=designate M=everywhere O=PRON.2pl`  
  reader `T=this+week ; C=because S=every+harmless+place V=pass.PRES.SHOULD.NEG O=PRON.3sgf FOR=NAME.Mila NEAR=no+warm+tow TO=NAME.Nucum ; S=every+harmless+place V=designate.PRES.SHOULD M=everywhere O=PRON.2pl`
- how: forever: `Teach Peraan or the happiness against the stage. Lekama and Nunul must not be too dry under these too maximum elevations until these systems. had Kevin and the lit controls of the criminal been by bus forever each night. do not welcome the irritation or the surplus with the fridge!`  
  gen `V=teach O=NAME.Peraan|happy>QUAL AGAINST=stage ; S=NAME.Lekama&NAME.Nunul V=be.PRES.MUST.NEG UNDER=this+too+maximum+elevate>ACT.PL UNTIL=this+system.PL A=too+dry ; S=NAME.Kevin&criminal^lit+control.PL T=each+night V=be.PAST.PERF.Q M=forever WITH=bus ; V=welcome.NEG O=irritate>ACT|surplus WITH=fridge`  
  reader `V=teach O=NAME.Peraan|happy>QUAL AGAINST=stage ; S=NAME.Lekama&NAME.Nunul V=be.PRES.MUST.NEG UNDER=this.PL UNTIL=this+system.PL P=too+dry+too+maximum+elevate>ACT.PL ; S=NAME.Kevin&criminal^lit+control.PL T=each+night V=be.PAST.PERF.Q M=forever WITH=bus ; V=welcome.NEG O=irritate>ACT|surplus WITH=fridge`
- how: inside: `Riyonir was really easy inside, and where must Pekuu have reopened the too dietary holders, or one money and Caanin could be mader until the bastille than Paapi.`  
  gen `S=NAME.Riyonir V=be.PAST M=inside A=really+easy ; S=NAME.Pekuu V=reopen.PRES.PERF.MUST M=where O=too+dietary+hold>AGENT.PL ; C=or S=one+money&NAME.Caanin V=be.PAST.CAN UNTIL=bastille THAN=NAME.Paapi A=made>MORE`  
  reader `S=NAME.Riyonir V=be.PAST M=inside A=really+easy ; S=NAME.Pekuu V=reopen.PRES.PERF.MUST M=where O=too+dietary+hold>AGENT.PL ; C=or S=one+money&NAME.Caanin V=be.PAST.CAN UNTIL=bastille THAN=NAME.Paapi A=mad>MORE`
- det first: `She and the ownership are on Naliir; felt the first misogyny or the his actually`  
  gen `S=PRON.3sgf&ownership V=be.PRES ON=NAME.Naliir ; V=felt M=actual>ADV O=first+misogyny|hi.PL`  
  reader `None`
- det second: `It might not be in front of Colin, and Mary and the second gold are very exquisitely printing Nancy and Kiiruu.`  
  gen `S=PRON.3sgn V=be.PRES.MAY.NEG FRONT=NAME.Colin ; S=NAME.Mary&second+gold V=print.PRES.PROG M=very+exquisite>ADV O=NAME.Nancy&NAME.Kiiruu`  
  reader `S=PRON.3sgn V=print.PRES.PROG.MAY.NEG M=very+exquisite>ADV O=NAME.Nancy&NAME.Kiiruu FRONT=NAME.Colin&NAME.Mary&second+gold_are`
- det own: `Its own mainstream and it was offering them to me during her and how will you all report the ground and when didn't you all help this innovative, unnecessary circus!`  
  gen `S=PRON.3sgn^own+mainstream ; S=PRON.3sgn V=offer.PAST.PROG O=PRON.3pl TO=PRON.1sg DURING=PRON.3sgf ; S=PRON.2pl V=report.FUT M=how O=ground ; S=PRON.2pl V=help.PAST.NEG M=when O=this+innovative+necessary>NOT+circus`  
  reader `S=PRON.3sgn^own+mainstream&PRON.3sgn V=offer.PAST.PROG O=PRON.3pl TO=PRON.1sg DURING=PRON.3sgf ; S=PRON.2pl V=report.FUT M=how O=ground ; S=PRON.2pl V=help.PAST.NEG M=when O=this+innovative+necessary>NOT+circus`
- how: ahead: `Please bid Ruyu and Lia and this gymnastics could be next to these propositions ahead.`  
  gen `I=please V=bid O=NAME.Ruyu&NAME.Lia ; S=this+gymnastics V=be.PAST.CAN M=ahead NEAR=this+propose>ACT.PL`  
  reader `I=please V=bid ; S=NAME.Ruyu&NAME.Lia&this+gymnastics V=be.PAST.CAN M=ahead NEAR=this+propose>ACT.PL`
- det second: `Do not very rarely think the second greatest regional calligraphy`  
  gen `V=think.NEG M=very+rare>ADV O=second+great>MOST+region>ADJ+calligraphy`  
  reader `V=think.NEG M=very+rare>ADV O=second+greatest+region>ADJ+calligraphy`
- det first: `Was this contact by this board before this too relative war when the first multilingual statuses might sometimes be instituting it, and one too uneven normal best lord is not going to lounge`  
  gen `S=this+contact V=be.PAST.Q NEAR=this+board BEFORE=this+too+relative+war ; C=when S=first+multilingual+status.PL V=institute.PRES.PROG.MAY M=sometimes O=PRON.3sgn ; S=one+too+uneven+normal+best+lord V=lounge.FUT.NEG`  
  reader `S=this+contact V=be.PAST.Q NEAR=this+board BEFORE=this P=too+relative+war ; C=when S=first+multilingual+status.PL V=institute.PRES.PROG.MAY M=sometimes O=PRON.3sgn ; S=one+too+uneven+normal+best+lord V=lounge.FUT.NEG`
- verb have: `Must I have me a lot, and please trump against him a lot tonight`  
  gen `S=PRON.1sg V=have.PRES.MUST.Q M=much O=PRON.1sg ; I=please V=trump M=much+tonight AGAINST=PRON.3sgm`  
  reader `I=please S=PRON.1sg V=have.PRES.MUST.Q M=much+much+tonight O=PRON.1sg&trump>ACT AGAINST=PRON.3sgm`
- verb have: `Goodbye tourists, and the package could be having me, and do not announce us or Keyu ahead please!`  
  gen `I=goodbye Y=tourist.PL ; S=package V=have.PAST.PROG.CAN O=PRON.1sg ; I=please V=announce.NEG M=ahead O=PRON.1pl|NAME.Keyu`  
  reader `I=goodbye S=tourist.PL&package V=have.PAST.PROG.CAN O=PRON.1sg ; I=please S=tourist.PL&package V=announce.PAST.PROG.CAN.NEG M=ahead O=PRON.1pl|NAME.Keyu`
- how: ahead: `Goodbye tourists, and the package could be having me, and do not announce us or Keyu ahead please!`  
  gen `I=goodbye Y=tourist.PL ; S=package V=have.PAST.PROG.CAN O=PRON.1sg ; I=please V=announce.NEG M=ahead O=PRON.1pl|NAME.Keyu`  
  reader `I=goodbye S=tourist.PL&package V=have.PAST.PROG.CAN O=PRON.1sg ; I=please S=tourist.PL&package V=announce.PAST.PROG.CAN.NEG M=ahead O=PRON.1pl|NAME.Keyu`
- det first: `Must she be within its cave inside before she and Vuvi can be the first endless pines inside the mountains tomorrow and usually publish a mediation and Pali`  
  gen `S=PRON.3sgf V=be.PRES.MUST.Q M=inside L=PRON.3sgn^cave ; C=before S=PRON.3sgf&NAME.Vuvi V=be.PRES.CAN M=tomorrow L=mountain.PL P=first+endless+pine.PL ; V=publish M=usually O=mediate>ACT&NAME.Pali`  
  reader `S=PRON.3sgf V=be.PRES.MUST.Q M=inside L=PRON.3sgn^cave ; C=before S=PRON.3sgf&NAME.Vuvi V=be.PRES.CAN M=tomorrow L=mountain.PL P=first+endless+pine.PL ; S=PRON.3sgf&NAME.Vuvi V=publish.PRES.CAN M=usually O=mediate>ACT&NAME.Pali`
