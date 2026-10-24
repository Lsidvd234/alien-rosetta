# More test ideas for the saved glyph models (beyond the 63-item catalogue)

Created 2026-10-11. Literature/practitioner sweep for tests that can run on the **saved 50k models** (option 1, control,
option 2, 3a) without retraining. Deduplicated against `failure_catalogue.md` (#1-#63) and against the already-planned
list (per-word error localisation, passive "first noun = agent", within-input-type AUROC, 2-adjective candidate
scoring, forgetting decomposition glyph rows vs LoRA, prompt-format variants, irregulars x frequency, seeds, per-epoch
checkpoints). Where an idea touches a catalogue item, the overlap is stated.

**Cost** = rough time on the 4 GB laptop GPU (Qwen3-0.6B in fp16, ~1k test sentences unless stated).
**Priority**: A = do next, B = worth doing, C = low value / nice-to-have.
**Sources**: [v] = found and checked in this sweep; [m] = standard reference cited from memory, not re-checked here.

One fact the setup makes unusually cheap: the generator knows the **gold word alignment** (which English word produced
which glyph word, and which feature produced which suffix). Several ideas below (1, 3, 5, 24) use this as ground truth,
which most interpretability papers lack.

---

## 1. Mechanistic interpretability

**1. Source-attribution alignment vs gold alignment**
- Probes: does each output glyph word (stem and each suffix) get its evidence from the right English word? On failures
  (passive, synonym, typo) which English token actually drove the wrong stem?
- How: with Inseq (integrated gradients or input x gradient, plus attention), attribute each generated glyph token to
  the source tokens; take the argmax source word per target word and score alignment error rate (AER) against the
  generator's gold alignment. Split by suffix type: ACC suffix should point at the object (or "the"/verb?), tense suffix at
  the auxiliary ("will", "did") or the verb inflection.
- Informative: low AER on normal input but scrambled attribution on passive/fronting means the model's lookup is
  position-based, not word-based. Tense suffix attributing to "will"/"did" vs the verb's "-ed" says what cue it uses.
- Cost: ~15 min for 500 sentences. **A**
- Sources: Sarti et al. 2023, Inseq (ACL demo) [v]; Ferrando et al. 2022, ALTI+ (EMNLP) [v]; Sundararajan et al. 2017,
  integrated gradients [m].

**2. Source vs target-prefix contribution per token**
- Probes: when the model fails (or invents words for nonsense input, #15), is it ignoring the source and running on its
  own glyph language model?
- How: compute the share of attribution mass on source tokens vs previously generated glyph tokens for every output
  token (Inseq or ALTI-style aggregation); compare distributions for correct outputs, failure sets and nonsense inputs.
- Informative: if the source share drops sharply on failures, it is a cheap hallucination detector and explains "never
  refuses"; if it stays high, the model is reading the source but mapping it wrongly.
- Cost: ~10 min (reuses idea 1). **B**
- Sources: Zaranis et al. 2024, context contributions in LLM-based MT (Findings EMNLP) [v]; Ferrando et al. 2022 [v].

**3. Activation patching on minimal pairs (where each feature is read and written)**
- Probes: at which layer and token position is tense / negation / plural / person / case moved from the English side
  to the output? Is there one "agreement" site, as in natural-language models?
- How: take the minimal pairs from test 14 (one feature differs); run the clean and corrupted prompt, patch the residual
  stream (then attention/MLP outputs) from clean into corrupted at each (layer, source position), and measure the logit
  difference between the two correct suffix glyphs at the suffix position. Use logit difference, not probability, as
  recommended.
- Informative: a sharp layer x position map per feature; e.g. person agreement read from the subject token at layers
  k-m. Comparing option 1 vs control shows whether the better generaliser uses a more localised circuit.
- Cost: ~20-40 min for 5 features x 50 pairs x 28 layers. **A**
- Sources: Zhang & Nanda 2024, best practices of activation patching (ICLR) [v]; Ferrando & Costa-jussa 2024,
  subject-verb agreement circuits across languages [v]; Meng et al. 2022, causal tracing (ROME) [m].

**4. Logit lens: decision layer for suffixes, and is there an English latent?**
- Probes: (a) at what depth does the correct suffix glyph become top-1 in normal sentences (catalogue #57 only asks
  this for passive/2-adjective information loss); (b) when emitting the first glyph of a stem, does a middle layer decode
  the English word ("dog") or its Latin sound before the glyph, i.e. is English a pivot?
- How: apply the final norm + unembedding to each layer's residual at each output position; record the first layer
  where the gold glyph is top-1 and the rank of the English lemma token / the Latin sound token in middle layers.
  Optionally fit a tuned lens on 2k training sentences for a cleaner read.
- Informative: suffixes decided late but stems early (or the reverse) tells where to look for failures; an English pivot
  in option 1 but not the control would support the "sound start helps" story mechanistically.
- Cost: ~10 min (logit lens), ~30 min with tuned lens. **A**
- Sources: Wendler et al. 2024, "Do Llamas work in English?" (ACL) [v]; nostalgebraist 2020, logit lens blog [m];
  Belrose et al. 2023, tuned lens [m].

**5. Causal probing of grammatical features (probe, then erase)**
- Probes: are tense / number / person / role (agent vs patient) linearly encoded on the English tokens, and does the
  model actually *use* that direction?
- How: linear probes on residual states of the verb / subject / object tokens per layer (with a control task), then
  erase the direction with LEACE/INLP during generation and check whether the corresponding suffix becomes random while
  other suffixes stay right. Distinct from #57, which probes for lost passive/2-adjective information.
- Informative: "decodable but not used" vs "decodable and causal" per feature; erasing "role" on passive input tells
  whether the model has any agent representation at all.
- Cost: ~30 min. **B**
- Sources: Elazar et al. 2021, amnesic probing (TACL) [m]; Belrose et al. 2023, LEACE [m]; Hewitt & Liang 2019
  (already in catalogue) for control tasks.

**6. Steering vectors for grammatical features**
- Probes: can one direction flip a feature in the output without touching the English? Tests whether the grammar is
  stored as clean linear features.
- How: contrastive activation addition: mean residual difference between past and future (or affirmative vs negated)
  prompts at one layer; add +/- alpha at the source positions of new sentences; measure the fraction of outputs whose
  tense suffix flips and nothing else changes (reuse the test-14 collateral-change scorer).
- Informative: clean flips = linear, composable features; also try a "passive -> active" vector to see if a steering
  fix exists for the 0% passive result.
- Cost: ~20 min. **B**
- Sources: Rimsky et al. 2024, contrastive activation addition (ACL) [v]; Turner et al. 2023, activation addition [m].

**7. Attention-head ablation map**
- Probes: which heads carry word reordering (SVO -> SOV), agreement and case? Are there a few critical heads?
- How: mean-ablate each head (28 layers x 16 heads) one at a time on 200 dev sentences; record per-feature accuracy
  (order, ACC/LOC, agreement, stem). Then ablate the top-5 jointly.
- Informative: a handful of critical heads = fragile, localised mechanism; diffuse = redundant. Compare with the LoRA
  layer ablation (idea 13).
- Cost: ~40-60 min (448 ablations x 200 forward passes, teacher-forced scoring only). **B**
- Sources: Michel et al. 2019, "Are sixteen heads really better than one?" [m]; Voita et al. 2019, specialised heads [m].

**8. "Translate-mode" function vector into the base model**
- Probes: is the adapter's effect summarised by one task vector? Relates to the 47% glyph leakage in English.
- How: average the (adapter-on minus adapter-off) residual at the last prompt token over 500 prompts; add it to the
  base model (adapter off) and see whether it starts emitting glyphs / translating; subtract it from the adapted model on
  English continuations and see whether leakage drops.
- Informative: a single vector that toggles glyph mode would give a cheap leakage fix and show the task is a "mode".
- Cost: ~15 min. **C**
- Sources: Todd et al. 2024, function vectors (ICLR) [m].

## 2. Representation and embedding analysis

**9. Layer-wise CKA: base vs adapted, option 1 vs control**
- Probes: where in depth does LoRA rewrite English processing, and does option 1 rewrite less than the control (a
  candidate explanation for its 2.5x lower forgetting and better generalisation)?
- How: linear CKA between base and each adapted model's hidden states on (a) English source tokens of test prompts,
  (b) plain English text; per layer. Also CKA option 1 vs control.
- Informative: option 1 staying close to base on plain English but diverging on prompts = isolated task circuit.
- Cost: ~10 min. **B**
- Sources: Kornblith et al. 2019, CKA (ICML) [m].

**10. Do glyphs still behave like their sounds in context?**
- Probes: catalogue #59 covers static embedding collapse; this asks about contextual representations: does the model
  process a glyph sequence like the Latin transliteration at depth?
- How: feed the gold glyph target and its Latin transliteration (same tokens-per-sound where possible) as continuations
  after the same source; per layer, cosine / RSA between position-matched hidden states. Compare option 1 vs control
  (control should not match).
- Informative: high similarity in early layers that fades = sound prior used only as a starting point; persistent
  similarity = the model really reads glyphs as letters.
- Cost: ~10 min. **C**
- Sources: Minixhofer et al. 2022, WECHSEL; Dobler & de Melo 2023, FOCUS (both already in catalogue) [m].

**11. Embedding-row swap at inference (functional test of the glyph rows)**
- Probes: how much of translation depends on the trained glyph rows vs the LoRA layers? (The planned decomposition is
  about forgetting; this is about the translation itself.)
- How: (a) reset all glyph rows to their initial Latin-sound vectors and re-run greedy translation; (b) reset only input
  side or only output side (Qwen3-0.6B ties input and output embeddings, so check `tie_word_embeddings` and untie in
  memory if needed); (c) permute rows of two glyphs and check the outputs swap accordingly.
- Informative: if (a) leaves accuracy high, LoRA did the work and the rows barely matter; if it collapses, the rows
  moved functionally even though norms look normal (#54).
- Cost: ~10 min. **A**
- Sources: Hewitt 2021 (embedding init, in catalogue) [m]; HF `tie_word_embeddings` config [m].

**12. Transliteration leakage: does option 1 read Latin sounds as glyphs?**
- Probes: because glyph rows start at Latin sounds, does a romanised target word inside the English get "copied"?
- How: replace an English noun by its romanisation in the invented language (e.g. "the nai runs"); score whether the
  output stem is the right glyph word. Compare with the control (shuffled sounds) and a random romanised nonce word.
- Informative: option 1 >> control means the sound prior is still live as a lexical shortcut (a new spurious path);
  equal means the prior only helped optimisation.
- Cost: ~5 min. **B**
- Sources: own idea; motivated by WECHSEL/FOCUS init literature [m].

**13. Language-specific (glyph) neurons and leakage**
- Probes: is "glyph mode" carried by a small set of MLP neurons, and are those the cause of glyph leakage into English?
- How: LAPE-style: activation probability of each MLP neuron on glyph output vs plain English; take the most selective
  1%; deactivate them on English continuations (measure leakage and perplexity) and on translation (measure accuracy).
- Informative: if leakage drops with little translation loss, a neuron mask is a free mitigation and localises the
  forgetting. Complements the planned rows-vs-LoRA decomposition.
- Cost: ~20 min. **B**
- Sources: Tang et al. 2024, language-specific neurons / LAPE (ACL) [v].

## 3. LoRA-specific analyses

**14. Per-layer and per-module LoRA ablation (which layers carry the grammar)**
- Probes: which depth ranges and which modules (q/k/v/o vs gate/up/down) are needed for stems, word order, case, agreement?
- How: zero the LoRA update in one block of 4 layers at a time (7 runs), then per module type (7 runs); score the
  standard test set with the per-feature scorer (stem, order, each suffix). Also "keep only block i" runs.
- Informative: e.g. lexicon in early MLPs, reordering in middle attention, suffixes late; tells you whether a smaller
  adapter (fewer layers) would suffice and where failures like passive must be fixed.
- Cost: ~30 min (14-20 eval runs x ~1.5 min, teacher-forced). **A**
- Sources: Zhou et al. 2024, LoRA-drop (output-based layer importance) [v]; Hu et al. 2021, LoRA [m].

**15. LoRA output-norm diagnostics on failing inputs**
- Probes: do failing inputs (passive, uppercase, fillers) produce abnormal adapter activations, i.e. the adapter "does
  not fire" or fires in the wrong place?
- How: hook every LoRA module and record ||BAx|| per layer per token; compare normal vs each failure set; correlate with
  correctness.
- Informative: an adapter that goes quiet on unseen English forms supports "LoRA keyed on surface form"; could also be
  an OOD signal (see idea 21).
- Cost: ~5 min. **B**
- Sources: Zhou et al. 2024, LoRA-drop (LoRA output magnitude as importance) [v].

**16. SVD truncation of the LoRA update, per feature**
- Probes: how many directions of each dW = BA are needed, and does lexicon need more rank than grammar? This is a
  concrete way to run catalogue #60 (rank saturation); listed for the per-feature breakdown.
- How: for k = 1, 2, 4, 8, 16 keep the top-k singular directions of every dW, evaluate with the per-feature scorer.
- Informative: if grammar survives at k=2 but stems need 16, rank is spent on the lexicon; if all saturate at k<8,
  r=16 is oversized.
- Cost: ~15 min. **B** (overlaps #60)
- Sources: Hu et al. 2021, LoRA subspace analysis [m]; Shuttleworth et al. 2025 (in catalogue).

**17. Alignment of dW with base weights (amplification factor)**
- Probes: does the update amplify directions already in the base model (re-using English features) or add new ones?
  Compare option 1 vs control vs option 2.
- How: Hu et al.'s ||U^T W V|| / ||dW|| projection per layer, plus principal angles between dW's top singular vectors
  and W's.
- Informative: option 1 re-using base directions more would match its lower forgetting.
- Cost: ~5 min (weights only). **C**
- Sources: Hu et al. 2021, LoRA section 7 [m].

## 4. Data attribution and memorisation

**18. k-nearest training neighbours for failures**
- Probes: what training sentences does the model "think" a failing input is? Does the wrong stem come from the nearest
  neighbours?
- How: embed all 50k training sources (mean of last-layer, or layer of maximal probe accuracy) once; for each failing
  test input retrieve the 10 nearest; check whether the wrong output stem/structure appears in those neighbours'
  targets. Optionally kNN-vote backoff for low-confidence items.
- Informative: "passive input's neighbours are active sentences with the nouns swapped" would explain the 81% wrong
  stems directly; neighbour-explained errors = retrieval-like behaviour.
- Cost: ~15 min (50k forward passes, no generation). **A**
- Sources: Rajani et al. 2020, kNN representations for explaining/improving models [v].

**19. Influence of training pairs on specific outputs (LoRA influence functions)**
- Probes: which training examples most support a particular (wrong) output; is behaviour driven by a few examples?
- How: DataInf (closed-form influence for LoRA) or TracIn with only LoRA gradients on a 5k random training subset plus
  the top-200 kNN candidates from idea 18; rank by influence for 20 chosen failures and 20 successes.
- Informative: consistent overlap with idea 18 validates kNN as the cheap proxy; if influence is diffuse, failures are
  rule-level, not example-level.
- Cost: ~45-90 min (per-example LoRA gradients, 5k items). **C**
- Sources: Kwon et al. 2024, DataInf (ICLR) [v]; Pruthi et al. 2020, TracIn [m].

**20. Membership-inference loss gap (memorisation without regurgitation)**
- Probes: catalogue #48 checks whether outputs copy training sentences; this checks whether the model assigns training
  targets systematically higher likelihood than equally-formed unseen ones.
- How: per-token loss and Min-K% prob on 2k training pairs vs 2k fresh generator pairs of the same templates; report
  AUC. Also sample 500 outputs from an empty / "Source: ." prompt at T=1 and count exact training targets.
- Informative: AUC ~0.5 = rule learning; clearly >0.5 = memorisation pressure (useful when deciding epochs).
- Cost: ~10 min. **B**
- Sources: Shi et al. 2024, Min-K% prob (ICLR) [v]; Carlini et al. 2021, training-data extraction [m].

## 5. Calibration, uncertainty and abstention

**21. Input-side OOD detector (Mahalanobis) for abstention**
- Probes: the model never refuses (#15) and output confidence still averages 0.5-0.7 on wrong outputs; can it tell from
  the *input* that the English is off-template?
- How: fit mean/covariance of input-encoding embeddings (mean of source-token states, a middle and the last layer) on
  training sources; score relative Mahalanobis distance for normal test inputs vs every failure set (passive, synonyms,
  numerals, fronting, nonsense...). Report AUROC and combine with output confidence.
- Informative: high AUROC = a free "I can't translate this" switch; tells whether failures are visible in the input
  representation before decoding.
- Cost: ~10 min. **A**
- Sources: Ren et al. 2023, OOD detection and selective generation for conditional LMs (ICLR) [v]; Lee et al. 2018,
  Mahalanobis OOD [m].

**22. Calibration package: ECE, reliability diagram, temperature scaling, risk-coverage**
- Probes: AUROC (#50) only measures ranking; is the confidence numerically right, and how much can be translated at a
  target accuracy? (The planned within-input-type AUROC is a ranking metric too.)
- How: on a mixed pool (normal + all stress sets), bin sequence confidence -> ECE and reliability diagram; fit a single
  temperature on dev; plot risk-coverage, report AURC and coverage at 99% accuracy.
- Informative: "covers 80% of inputs at 99% accuracy" is a deployable statement; overconfidence pattern guides
  temperature or abstention design.
- Cost: ~5 min (reuses test-17 scores). **A**
- Sources: Guo et al. 2017, calibration of modern nets [m]; Geifman & El-Yaniv 2017, selective classification [m];
  Kumar & Sarawagi 2019 (in catalogue).

**23. Sampling-based uncertainty (sequence/semantic entropy)**
- Probes: greedy = beam (#44), so is the distribution still diverse on failures? Does disagreement among samples
  detect errors better than greedy confidence?
- How: 10 samples at T=1 per input; entropy over distinct outputs (exact-match clusters = meaning clusters here);
  AUROC vs correctness, compared with #50.
- Informative: confident-and-consistent wrong outputs (low entropy) on passive would show the errors are systematic
  mis-rules, not uncertainty.
- Cost: ~20 min. **B**
- Sources: Farquhar et al. 2024, semantic entropy (Nature) [v]; Kuhn et al. 2023 [m].

**24. Conformal abstention with an error guarantee**
- Probes: can we give a finite-sample guarantee like "<=2% wrong among answered"?
- How: on a calibration split of the mixed pool, choose the confidence threshold by conformal risk control; test on a
  held-out split per input type; also the sample-until-covered variant of conformal LM with 10 samples.
- Informative: whether the guarantee holds under the shift to unseen failure types (it should break; that is the
  interesting result).
- Cost: ~10 min. **C**
- Sources: Quach et al. 2024, conformal language modeling (ICLR) [v]; Angelopoulos et al. 2022/2024, conformal risk
  control [m].

## 6. Compositional-generalisation analysis

**25. Per-item "unobserved local structure" predictor**
- Probes: one rule that explains all stress-test failures: does an item fail exactly when it contains a local source
  structure (word bigram / POS bigram / parent-child pair) never seen in training?
- How: from the 50k training sources build the set of POS-tag bigrams and trigrams (and word bigrams); for every
  test item from tests 1-21 flag unseen local structures; logistic regression / accuracy table of correctness vs flag.
- Informative: if the flag predicts >90% of failures (passive "was eaten by", "two dogs", fronted PP...), the catalogue
  collapses to one mechanism and tells the retrain exactly what to add.
- Cost: ~5 min CPU, no GPU. **A**
- Sources: Bogin et al. 2022, unobserved local structures (EMNLP) [v]; Keysers et al. 2020 (in catalogue).

**26. Atom and compound divergence of each stress set (MCD-style numbers)**
- Probes: #20 cites Keysers but compound divergence was never computed. How far is each test set from training, and
  is accuracy a smooth function of it?
- How: atoms = words / features, compounds = (head, dependent) pairs from the generator's trees; compute the Chernoff
  divergence of each test set vs training; plot accuracy against compound divergence for all sets.
- Informative: a smooth curve = generalisation degrades with distance; a cliff = some compound types are "never"
  generalised. Gives a single axis to report all stress results on.
- Cost: ~10 min CPU. **B**
- Sources: Keysers et al. 2020, CFQ/MCD (ICLR) [m]; Shaw et al. 2021, TMCD [m].

**27. Word-order permutation sensitivity**
- Probes: does the model still output a confident well-formed translation for scrambled English ("dog the runs the")?
  How much does it rely on bag-of-words vs order?
- How: random permutations (5 per sentence) plus partial ones (swap 2 words); record whether output equals the
  unpermuted translation, whether roles change, and confidence.
- Informative: identical outputs for shuffles = lexical lookup + defaults (the passive failure suggests position
  heuristics instead; this distinguishes the two); high confidence on shuffles strengthens the abstention case.
- Cost: ~5 min. **B**
- Sources: Sinha et al. 2021, UnNatural language inference (ACL) [v].

**28. Irrelevant context before the prompt**
- Probes: catalogue #11 inserts a filler inside the sentence; this adds unrelated English (one sentence, one paragraph)
  or a previous translation pair *before* "Source:".
- How: prepend 0 / 1 / 3 / 10 sentences of unrelated English or of other training pairs; measure accuracy and whether
  words leak from the distractor.
- Informative: tells whether the model can be used in batch/document settings and whether it attends only to the
  final source line.
- Cost: ~10 min. **B**
- Sources: Shi et al. 2023, LLMs easily distracted by irrelevant context (ICML) [m].

**29. Is in-context learning still alive after fine-tuning?**
- Probes: can 2-3 demonstrations of a failing construction (passive -> translation) in the prompt rescue it, or has
  format specialisation removed ICL? (Different from the planned prompt-format variants, which change the wrapper.)
- How: k = 0/1/3 demos of passive (or numerals) with gold glyph outputs before the test prompt; same with demos of
  normal sentences as control.
- Informative: if demos help, cheap few-shot patches exist; if not, the adapter has killed ICL, a known LoRA side effect.
- Cost: ~10 min. **B**
- Sources: Wang et al. 2024, ProMoT / format specialisation (in catalogue); Zaranis et al. 2024 (fine-tuning changes
  few-shot context use) [v].

## 7. Robustness and adversarial inputs

**30. Imperceptible Unicode perturbations**
- Probes: homoglyphs (Cyrillic "о"/"а", Greek "ο"), zero-width space/joiner, non-breaking space, full-width letters,
  soft hyphen, bidi override; none change what a human sees.
- How: one and three injections per sentence at random positions; accuracy, and whether the stem of the touched word
  goes wrong (reuse the test-18 classifier). Include NFKC-normalised input as the "fix" condition.
- Informative: expected large drops like typos (7%); if NFKC recovers it, a one-line preprocessing fix; also shows the
  token-level mechanism (new tokens -> wrong stem) generalises.
- Cost: ~5 min. **A**
- Sources: Boucher et al. 2022, "Bad Characters: imperceptible NLP attacks" (IEEE S&P; arXiv 2021) [v].

**31. Glyph characters inside the English input**
- Probes: what happens if the source already contains PUA glyphs (the glyph word for "dog" in place of "dog", or a
  random glyph)? Copy, translate, or derail?
- How: 200 sentences with one noun replaced by its glyph form; 200 with a random glyph string; score stem and rest.
- Informative: copying = the model can "pass through" target words (useful for names); derailing = glyphs in the
  source are a strong trigger, i.e. an attack surface.
- Cost: ~5 min. **C**
- Sources: own idea (related to Boucher et al. 2022 [v]).

**32. Gradient-guided minimal attacks and universal triggers**
- Probes: smallest meaning-preserving edit that flips tense/negation/subject in the output; and whether one short
  prefix breaks or controls all translations.
- How: HotFlip/GCG-style token search over English tokens, restricted to plausible words (lexicon + stop words) so the
  English meaning stays the same; targeted objective = log-prob of the flipped suffix. Universal trigger: optimise a
  3-token prefix over 100 sentences.
- Informative: number of edits needed is a robustness score per feature; a universal trigger that forces negation
  everywhere would be a striking failure.
- Cost: ~30-60 min. **B**
- Sources: Ebrahimi et al. 2018, HotFlip [m]; Wallace et al. 2019, universal adversarial triggers [m]; Zou et al.
  2023, GCG [m].

## 8. Test-time interventions (mitigations, no retraining)

**33. Normalise the English with the base model, then translate**
- Probes: can the base model's untouched English knowledge rescue passive / synonym / numeral / fronting / uppercase
  failures if used as a front-end?
- How: adapter off, few-shot prompt the base model to rewrite each English sentence into the training template
  ("the dog eats the apple"); adapter on, translate the rewrite. Score end-to-end and the rewrite step separately.
- Informative: a big recovery means the failures are only an input-coverage problem; also gives a strong baseline for
  the planned retrain.
- Cost: ~20 min (two generations per item). **A-** (top-11)
- Sources: Deng et al. 2023, "Rephrase and Respond" [m]; idea is standard pre-editing for MT [m].

**34. Source-amplifying decoding (context-aware / contrastive decoding)**
- Probes: are wrong stems driven by the glyph-LM prior? Amplifying the difference between with-source and
  without-source logits should fix prior-driven errors.
- How: logits = (1+a) * logp(y | source) - a * logp(y | empty source), a in {0.5, 1}; also contrastive decoding against
  the control model. Evaluate on failure sets and normal set.
- Informative: no gain = errors are source misreadings (consistent with #44 model errors); gain on some sets = a
  decoding-only fix.
- Cost: ~10 min. **B**
- Sources: Shi et al. 2024, context-aware decoding [m]; Li et al. 2023, contrastive decoding (ACL) [m].

**35. Oracle prefix: give the first k correct glyph words**
- Probes: complement to #45 (forced *wrong* token): on failing inputs, does forcing the correct first word (or stem)
  fix the rest? Locates whether the error is one early decision or spread.
- How: for passive / synonym / 2-adjective failures, force gold prefix of k = 1, 2, 3 words, greedy the rest, score.
- Informative: if one correct word fixes the sentence, failures are a single early mis-decision (e.g. which noun is
  subject); if not, the source representation is wrong throughout.
- Cost: ~5 min. **B**
- Sources: Wang & Sennrich 2020; Bachmann & Nagarajan 2024 (both in catalogue).

**36. MBR / majority vote over samples**
- Probes: does any sample-based decoding beat greedy on the failure sets?
- How: 20 samples at T=0.7, choose the exact-match majority or chrF-MBR.
- Informative: likely no gain (model errors); a quick confirmation that closes the decoding question.
- Cost: ~20 min. **C**
- Sources: Freitag et al. 2022, MBR with neural metrics [m]; Eikema & Aziz 2020 (in catalogue).

## 9. Numerics and evaluation sanity

**37. Quantisation robustness (8-bit / 4-bit)**
- Probes: do the glyph rows and the small LoRA update survive int8 / NF4? Non-Latin scripts are reported to suffer most.
- How: load the merged model with bitsandbytes 8-bit and 4-bit; rerun the standard set and test-14 minimal pairs.
- Informative: drop concentrated on suffixes or rare glyphs would show where precision matters; matters for any demo.
- Cost: ~10 min. **B**
- Sources: Marchisio et al. 2024, how quantization affects multilingual LLMs (Findings EMNLP) [v]; Dettmers et al.
  2023, QLoRA [m].

**38. Batch and padding invariance of reported numbers**
- Probes: do greedy outputs change with batch size or padding side? A silent source of noise in every result above.
- How: run 500 prompts at batch 1, 8, 32, left vs right padding; count differing outputs and max logit difference.
- Informative: any difference with right padding = a bug in batched eval; small batch-size differences quantify
  numerical noise floor for the 1-2 point effects in the catalogue.
- Cost: ~5 min. **C** (but cheap insurance)
- Sources: He / Thinking Machines Lab 2025, "Defeating nondeterminism in LLM inference" (blog) [v]; HF Transformers
  docs on left padding for decoder-only generation [m].

---

## Ranked top-10

| Rank | Idea | Why first | Cost |
|---|---|---|---|
| 1 | 25. Unobserved-local-structure predictor | CPU-only, may explain every failure in the catalogue with one rule and directly specifies the retrain data | 5 min |
| 2 | 14. Per-layer / per-module LoRA ablation | Says where grammar vs lexicon live; reused by 3, 4, 7 | 30 min |
| 3 | 21. Input-side Mahalanobis OOD + abstention | Directly attacks "never refuses"; cheap | 10 min |
| 4 | 1. Attribution alignment vs gold alignment | Gold alignments are free here; shows which English cue each suffix uses and what drives wrong stems | 15 min |
| 5 | 3. Activation patching on minimal pairs | Causal map of where each feature is read/written; minimal pairs already exist | 20-40 min |
| 6 | 18. kNN training neighbours of failures | Cheap data-level explanation of the 81-92% wrong-stem errors | 15 min |
| 7 | 4. Logit lens: suffix decision layer + English latent | Quick, and tests the sound-init story mechanistically | 10 min |
| 8 | 11. Embedding-row swap at inference | Separates "rows vs LoRA" for translation (complements planned forgetting split) | 10 min |
| 9 | 22. ECE + risk-coverage + temperature scaling | Turns AUROC 0.93 into a usable "coverage at 99% accuracy" | 5 min |
| 10 | 30. Imperceptible Unicode perturbations (+ NFKC fix) | New, realistic failure family; one-line fix to test | 5 min |

Next in line: 33 (base-model normalisation front-end), 27 (word-order shuffle), 35 (oracle prefix), 13 (glyph neurons
and leakage), 23 (sample entropy).

## Sources checked in this sweep [v]

- Sarti et al. 2023, Inseq: https://aclanthology.org/2023.acl-demo.40
- Ferrando et al. 2022, ALTI+ / source and target interpretations: https://aclanthology.org/2022.emnlp-main.599
- Zaranis et al. 2024, context contributions in LLM-based MT: https://arxiv.org/abs/2410.16246
- Zhang & Nanda 2024, activation patching best practices: https://arxiv.org/abs/2309.16042
- Ferrando & Costa-jussa 2024, subject-verb agreement circuits across languages: https://arxiv.org/abs/2410.06496
- Wendler et al. 2024, Do Llamas work in English?: https://aclanthology.org/2024.acl-long.820
- Rimsky et al. 2024, contrastive activation addition: https://aclanthology.org/2024.acl-long.828/
- Tang et al. 2024, language-specific neurons (LAPE): https://aclanthology.org/2024.acl-long.309
- Zhou et al. 2024/2025, LoRA-drop: https://arxiv.org/abs/2402.07721
- Kwon et al. 2024, DataInf: https://arxiv.org/abs/2310.00902
- Rajani et al. 2020, kNN representations: https://arxiv.org/abs/2010.09030
- Shi et al. 2024, Min-K% prob: https://arxiv.org/abs/2310.16789
- Ren et al. 2023, OOD detection and selective generation for conditional LMs: https://arxiv.org/abs/2209.15558
- Farquhar et al. 2024, semantic entropy (Nature): https://pmc.ncbi.nlm.nih.gov/articles/PMC11186750
- Quach et al. 2024, conformal language modeling: https://arxiv.org/abs/2306.10193
- Bogin et al. 2022, unobserved local structures: https://aclanthology.org/2022.emnlp-main.175
- Sinha et al. 2021, UnNatural language inference: https://aclanthology.org/2021.acl-long.569
- Boucher et al. 2022 (arXiv 2021), Bad Characters: https://arxiv.org/abs/2106.09898
- Marchisio et al. 2024, quantization and multilingual LLMs: https://aclanthology.org/2024.findings-emnlp.935/
- He / Thinking Machines Lab 2025, defeating nondeterminism: https://simonwillison.net/2025/Sep/11/defeating-nondeterminism

All [m] references are well-known papers cited from memory; check year/venue before putting them in a write-up.
