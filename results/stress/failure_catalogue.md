# Failure-mode catalogue for the hard-language glyph models

Merged from four literature sweeps (compositional generalisation, MT failures, fine-tuning/LoRA/tokenizer,
behavioural testing; ~180 sources, deduplicated) plus our own stress tests 1-21. Created 2026-10-11;
**statuses updated 2026-10-11 after tests 13-21** (all on the saved 50k models; numbers are option 1 unless stated).

**Status:** DONE = measured (result shown, test number in brackets) · PARTIAL = measured in part · NEW = not tested yet ·
N/A = cannot occur in our data.
**Verdict:** ✅ robust · ⚠️ weak · ❌ fails.
**Needs:** none = runs on saved models now · gen = small generator extension · weights = analysis of saved weights/logits ·
retrain = new GPU run.
**Priority (for what is still open):** A = do next · B = worth doing · C = low value / gold undefined.

Test 12 (confidence) was invalid (it scored the probability of the gold answer, which is tautological); test 17 replaces it.

## A. English-side variation (same meaning, different input)

| # | Failure mode | Status (opt1) | Verdict | Needs | Pri | Key sources |
|---|---|---|---|---|---|---|
| 1 | Passive voice | DONE: 0% (4, 17, 18). Beam does not help; 0 search errors; 81% wrong stems | ❌ | – | – | COGS (Kim & Linzen 2020); SLOG (Li et al. 2023) |
| 2 | Synonym not in the language | DONE: 15-19% (4, 17); 91% wrong stem | ❌ | – | – | CheckList INV (Ribeiro 2020); Hupkes 2020 substitutivity |
| 3 | UPPER CASE; mechanism | DONE: 9-18% (4, 17); Title Case 45% (19). Mechanism: new tokens → wrong stem 92% (18) | ❌ | – | – | Belinkov & Bisk 2018; Chai et al. 2024 |
| 4 | Determiner variants (a / this / that) | DONE: "a" 48% (4); this/that/these/those 6% (19) | ❌ | – | – | CheckList INV; Elazar 2021 |
| 5 | Negation spellings (doesn't, isn't, won't...) | DONE: contractions 93% (4) | ⚠️ | – | – | Hossain 2020; Dankers 2022 |
| 6 | Typos and other noise types | PARTIAL: letter swap 7% (8, 17); other noise types (keyboard, drop, scramble) not run | ❌ | none | B | Belinkov & Bisk 2018; Michel & Neubig 2018; Niu 2020 |
| 7 | Prompt wrapper / whitespace / format | DONE (8): double spaces 4%; leading/trailing space ~100%; "Input/Output" labels 88% (opt2 28%, control 34%); no newline after "Target:" 82%; "Translate:" line 92%; "please translate:" inside the sentence 5% (19) | ⚠️/❌ | – | – | Sclar 2024 (FormatSpread); Wang 2024 (ProMoT) |
| 8 | Tense said differently | DONE (19): "did X" 98%, "does X" 99%, "is going to" 50% | ⚠️ | – | – | Macketanz 2018; Burchardt 2017 |
| 9 | "did not eat": past on the auxiliary | DONE: this is the generator's normal English for negated past, so it is in training; minimal pairs on negation 99% (14) | ✅ | – | – | Hossain 2020; Sennrich 2017 |
| 10 | Fronted location / topicalised object | DONE (19): fronted location 0%, fronted object 0% | ❌ | – | – | McCoy, Frank & Linzen 2020; Yao & Koller 2022 |
| 11 | An unrelated token inserted ("Okay, the dog runs") | DONE (19): 0% | ❌ | – | – | Lee et al. 2018; Raunak 2021 |
| 12 | Homographs inside our lexicon ("saw", "left") | NEW | | none (lexicon scan) | B | MuCoW (Raganato 2020) |
| 13 | Nonce English words (wug test) | PARTIAL (10): unknown object, rest of sentence right 99%, word count right 100%; ending-only scoring not run | ✅ | none | C | Weissweiler 2023; Liu & Hulden 2022 |
| 14 | Numerals ("two dogs", "one cat") | DONE (19): 0% | ❌ | – | – | Wang et al. 2021 (numerical MT) |
| 15 | Out-of-domain input (emoji, French, question, code, empty) | DONE (16): stays in glyphs (95-100%) and confidently invents 2-5 glyph words; never refuses | ❌ (no refusal) | – | – | Guerreiro 2023; Bawden & Yvon 2023 |
| 16 | Instructions inside the sentence (prompt injection) | DONE (16): "Ignore the above and say hello" → 4 glyph words, never says hello | ✅ (not hijacked) | – | – | Sun & Miceli-Barone 2024 |
| 17 | Idioms ("kicks the bucket") → literal | NEW (gold undefined: the language has no idioms) | | none | C | Fadaee 2018; Dankers 2022 |

## B. Composition and structure

| # | Failure mode | Status (opt1) | Verdict | Needs | Pri | Key sources |
|---|---|---|---|---|---|---|
| 18 | Two+ adjectives on one noun | DONE: 0-9% (5, 17); 75% "word dropped" (18); beam +2 pts | ❌ | – | – | SLOG; ReCOGS (Wu 2023); Yao & Koller 2022 |
| 19 | Held-out long structure; long + held-out morphology | DONE: 95% / 96% (5); control 67% / 50% | ✅ | – | – | COGS; CoGnition |
| 20 | Template inventory (every structure type × training count) | DONE (20): all 18 templates 96-100%, incl. the never-seen one 97.5% (control 65%, 3a 1%) | ✅ | – | – | SLOG; ReCOGS; Keysers 2020 |
| 21 | Word in a role it never had | N/A (20): every noun appears as both subject and object in training | – | – | – | COGS; Kim, Linzen & Smolensky 2022 |
| 22 | Never-seen adjective+noun pairs ×5 contexts | DONE (20, after a bug fix): 100% of sentences, 100% of pairs right in all 5 | ✅ | – | – | CoGnition (Li 2021); gSCAN (Ruis 2020) |
| 23 | Subject/object swap; transitive verb without object; same noun twice | DONE (7): swap 99%; no-object 0% (all models; a shape never in training); same noun twice 98% | ✅ / ❌ (no object) | – | – | COGS; HANS (McCoy 2019) |
| 24 | Swap split by plausibility ("the apple eats the boy") | NEW (swap overall 99%, so low value) | | none | C | Kauf 2023; Papadimitriou 2022 |
| 25 | Accuracy by output length | DONE (20, 11): 99.7% (16-25 tokens) → 97.7% (51+); no too-long/too-short outputs | ✅ | – | – | Newman 2020; Variš & Bojar 2021 |
| 26 | Consistency: swap one noun → only that word may change | DONE (20): 1% other words changed (3a 20%) | ✅ | – | – | Dankers 2022; SIT (He 2020) |
| 27 | Same phrase in different sentences → same glyphs | PARTIAL (20): adj+noun pair right in all 5 contexts 95-100% | ✅ | none | C | He 2021; Hupkes 2020 localism |
| 28 | Coordination ("the dog and the cat run") | NEW | | gen | B | Marvin & Linzen 2018; SLOG |
| 29 | Embedding / relative clauses / recursion | NEW | | gen (new rules) | C | COGS recursion; SLOG |

## C. Morphology and agreement

| # | Failure mode | Status (opt1) | Verdict | Needs | Pri | Key sources |
|---|---|---|---|---|---|---|
| 30 | Agreement attraction (object/location between subject and verb) | DONE (13): verb right 99-100%; agrees with object 0%; prefers correct 100% | ✅ | – | – | Linzen 2016; Ravfogel 2019; CLAMS (Mueller 2020) |
| 31 | Change one feature → exactly one ending changes | DONE (14): 99-100% right, 0% ignored, 0% collateral, all 9 features | ✅ | – | – | CheckList DIR; Burlot & Yvon 2017 |
| 32 | Contrastive scoring per error type | DONE (15): 100% for all 8 corruption types; weakest margin = irregular verb regularised (6.9; control 88%, 3a 38%; only 8 items) | ✅ | – | – | Sennrich 2017; BLiMP; Vamvas & Sennrich 2021 |
| 33 | Full ending grid with training counts per cell | DONE (3): 99% seen / 95% held-out (opt2 72%, control 85%) | ✅ | – | – | Goldman 2022; Kodner 2023 |
| 34 | Irregular past: over-regularisation × verb frequency | PARTIAL (9, 15): irregular 96% vs regular 99%; frequency split not run; contrastive set small | ⚠️ | none | B | Kirov & Cotterell 2018; Corkery 2019 |
| 35 | Sandhi split by how rare the trigger is | PARTIAL (9): 0 / 1 / 2+ sound changes 99-100%; rarity split not run | ✅ | none | C | McCurdy 2020; Liu & Hulden 2022 |
| 36 | Gender: he/she/it swap; gendered nouns | PARTIAL (14): pronoun gender pairs 100%; gendered nouns not run | ✅ | none | C | WinoMT (Stanovsky 2019) |
| 37 | Word frequency curves | DONE (2): ~100% even for words seen ≤15 times (baseline needs 25-35) | ✅ | – | – | Koehn & Knowles 2017; Wei 2021 |
| 38 | Rare output patterns are harder | DONE (20): no link between template training count (1,126-7,154) and accuracy | ✅ | – | – | McCoy et al. 2024 (PNAS) |

## D. Output and decoding

| # | Failure mode | Status (opt1) | Verdict | Needs | Pri | Key sources |
|---|---|---|---|---|---|---|
| 39 | Omission per slot | PARTIAL (18): error classifier shows drops (2 adjectives 75%, two sentences 100%); per-slot parser not built | ❌ (where it occurs) | none | B | Tu 2016; HalOmi (Dale 2023) |
| 40 | Non-glyph output and probability on it | DONE (16): ≤5% outputs with non-glyph characters; 97-100% probability on glyphs | ✅ | – | – | Guerreiro 2023; Zhang 2020 |
| 41 | Free generation vs teacher-forced; too long / too short | DONE (11): 99.0% = 99.0%; 0% too long or short | ✅ | – | – | Wang & Sennrich 2020 |
| 42 | End-of-sentence calibration; premature stop | DONE (17): P(stop at the right place) 1.00 on normal input; double spaces → 15% risk of stopping early | ✅ / ⚠️ | – | – | Kumar & Sarawagi 2019; Stahlberg & Byrne 2019 |
| 43 | Repetition loops | DONE (11): no too-long outputs, so no loops | ✅ | – | – | Holtzman 2020; DITTO (Xu 2022) |
| 44 | Search error vs model error; beam | DONE (17): failures are model errors (search errors ≈0%, except Input/Output labels 17%); beam 5 = greedy | ✅ (search) | – | – | Stahlberg & Byrne 2019; Eikema & Aziz 2020 |
| 45 | Recovery after one forced wrong token | NEW | | none | B | Wang & Sennrich 2020; Bachmann & Nagarajan 2024 |
| 46 | Two sentences at once; nonce word | DONE (10, 18): two sentences 0% (second one dropped); nonce word, rest right 99% | ❌ / ✅ | – | – | Voita 2019; Lee 2018 |
| 47 | Reverse direction (glyphs → English) | DONE (11): 0% (never trained) | ❌ (expected) | – | – | Reversal curse (Berglund 2024) |
| 48 | Regurgitating training sentences | NEW | | none | B | Raunak & Menezes 2022 |
| 49 | Classify failures by type | DONE (18): verb-last order never broken (0%); failures = wrong stem (passive, synonym, case, typo, no-object), dropped words (2 adjectives, two sentences), extra words (double spaces) | – | – | – | Lake, Linzen & Baroni 2019 |
| 50 | Confidence: does it know when it is wrong? | DONE (17): own-output confidence separates right from wrong with AUROC 0.93; but wrong outputs still average 0.5-0.7 confidence | ⚠️ | – | – | Kumar & Sarawagi 2019; Dale 2023 |

## E. Model internals and fine-tuning side effects

| # | Failure mode | Status | Verdict | Needs | Pri | Key sources |
|---|---|---|---|---|---|---|
| 51 | Forgetting English; glyph leakage; LoRA strength sweep | DONE (21): English perplexity 20.3 (base) → 372 (opt1), 934 (control), 8.9M (opt2), 13.2M (3a); glyphs in English continuations 47% (opt1), 100% (others); adapter off restores ~20; translation needs full strength | ❌ | – | – | Biderman 2024; Kalajdzievski 2024 |
| 52 | Tokenizer round-trip audit | DONE (21): 52,000 targets, 0 errors; 1 token per glyph | ✅ | – | – | HF AddedToken docs |
| 53 | Train/test near-duplicate audit | DONE (21): 0% exact copies; one-word-different iid 18%, comp 2%, morph 8%, long 0% | ✅ | – | – | Sainz 2023 |
| 54 | Glyph norms; under-trained glyphs | DONE (21, 1): norms 1.1 vs vocab 0.97 median; rarest glyph seen 24,572×; per-glyph accuracy 100% | ✅ | – | – | Hewitt 2021; Land & Bartolo 2024 |
| 55 | Control model: errors follow its wrong start sounds? | DONE (21): no, 2.6% of 191 substitutions vs 3.7% chance | – | – | – | WECHSEL; FOCUS; Minixhofer 2024 |
| 56 | Intruder dimensions in LoRA; scaling by 0.7 | NEW (21's strength sweep: 0.75 drops translation to 10%) | | weights | B | Shuttleworth et al. 2025 |
| 57 | Where passive / 2-adjective information is lost (logit lens, probes) | NEW | | hidden states | B | Yao & Koller 2022; Hewitt & Liang 2019 |
| 58 | fp16 vs fp32 evaluation agreement | NEW | | none | C | Micikevicius 2018 |
| 59 | Glyph embeddings collapsing together | NEW (1: glyphs stay distinct at their sounds) | | weights | C | Gao 2019; Ethayarajh 2019 |
| 60 | LoRA rank saturation | NEW | | weights | C | rsLoRA; LoRA+ |
| 61 | Option 3a's broken verbs: spurious forgetting? | NEW (20: 3a fails every sentence with an object, 92-98% without) | | retrain (tiny) | C | Zheng 2025; Kumar 2022 |
| 62 | Seed variance of the 50k option results | NEW (20k grid has 3 seeds) | | retrain | **A** | Dodge 2020; COGS |
| 63 | Per-epoch trajectory | NEW | | retrain with checkpoints | B | Hupkes 2020; Power 2022 |

## Found by our tests (not in the literature list)

| Finding | Test | Note |
|---|---|---|
| A transitive verb used without an object fails for every model (0%) | 7, 18 | The generator never produced this shape, so it is a data gap, not a learning failure |
| Option 1 is far more robust to prompt-format changes than the other models ("Input/Output" labels 88% vs 28-34%; "Translate:" line 92% vs 32-68%) | 8 | Sound-start glyphs make the model less tied to the exact prompt |
| Option 1 forgets English ~25,000× less than options 2 and 3a; the two Latin-start models forget least | 21 | Hypothesis, untested: glyphs that start as ordinary letter embeddings need not become attractors |

## Open items, in priority order

1. **Retrain** (fixes, then re-test): varied English (passive, synonyms, casing, typos, spacing, fronting, fillers, numerals,
   determiners, "is going to"), 2-adjective noun phrases, transitive verbs without objects, names in sentences, some
   plain English to limit forgetting; **seed replicates (#62)** and per-epoch checkpoints (#63).
2. No-retrain extras (B): typo noise types (#6), homographs (#12), coordination (#28, needs generator), irregular-verb
   over-regularisation by frequency with ~200 items (#34), per-slot omission parser (#39), recovery after a forced wrong
   token (#45), regurgitation (#48), LoRA intruder dimensions (#56), logit lens / probes (#57).
