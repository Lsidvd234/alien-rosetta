# Why our translator only generalises to what we trained, and what "representative data" can and cannot fix

Literature review for the Alien Rosetta project (Qwen3-0.6B-Base + LoRA, English -> invented Tamil-like glyph language).
Date: 2026-10-12. Scope: web literature only; no code was run.

Citation convention: every source in this file was found and checked during this search (title, authors, venue, URL). Claims marked **[unverified]** come from memory or from my own reasoning about how a result transfers to our setup.

---

## 0. Executive summary

1. **What we are seeing is the textbook result, not a bug in our pipeline.** A network fine-tuned on a narrow input distribution learns the cheapest rule that fits that distribution (shortcut learning / simplicity bias). It generalises reliably only to recombinations of *local structures it has already seen* (Bogin et al. 2022). Benchmarks built to probe this (SCAN, COGS, CFQ, PCFG-SET) show the same pattern we see: ~95-99% on in-distribution or recombination tests, and near 0-35% on unseen *structures*, deeper recursion, or longer outputs. Our R1 pattern (passive transfers to unseen verbs, but object fronting does not follow from location fronting; 2->3 adjectives fails) is exactly the "lexical vs structural generalisation" split described by Kim & Linzen 2020 and Yao & Koller 2022.

2. **The user's instinct is right, and the literature backs it, with one caveat.** The evidence strongly favours *structural diversity of the training distribution* over adding targeted fixes. Examples: diverse-structure sampling gives 200x data efficiency (Oren et al. 2021); sampling meaning structures *uniformly* does almost as well as sampling from the test distribution (Yao & Koller 2024); more distinct primitives lead to near-perfect one-shot generalisation (Patel et al. 2022); more complex/diverse datasets generalise better (Zhou, Jiang & Bansal 2023); task diversity beats examples-per-task (Zhang, Wang & Charton 2024). **The caveat:** none of this lets a model handle structures *of a kind* that is absent from training. Diversity widens the support; it does not remove it. No finite training set can guarantee all unseen cases (no-free-lunch, Wolpert 1996). A realistic claim is "generalises to constructions held out *at random* from a broad, independently specified distribution of English". "Passes every test we never thought of" is not a realistic claim.

3. **Our failures fall into two families, and they need different remedies** (this split follows Shaw et al. 2021):
   - **(A) Input-variation failures** (synonyms, paraphrase, casing, typos, names, fillers): these are about *English*, which the base model already knows. The fix is to stop overwriting that knowledge and to let it be used. Options: a broader English side (LLM paraphrases), gentler adaptation (fewer or upper layers, adapter scaling, KL or replay), or a pipeline that normalises English first.
   - **(B) Structural / productive failures** (object fronting, 3 adjectives, 3 sentences, no-object verbs): these are compositional-generalisation failures. Fixes: generate English from a *broad grammar* with variable counts, sample uniformly over structures, decompose (sentence splitting / meaning frames), and accept that extrapolating beyond the trained count range stays fragile (Anil et al. 2022; Zhou et al. 2024).

4. **The biggest single opportunity:** we own a grammar generator, so we can (i) sample English from a *grammar of English*, not from one template, (ii) hold out whole construction families at random, MCD-style, to *measure* generalisation to unseen structures honestly, and (iii) optionally sample *many* invented languages (meta-learning, Lake & Baroni 2023). Option (iii) is real but research-grade and evidenced only in tiny domains.

5. **Ranked plan (details in section 11):**
   - E0: build a sealed challenge suite first.
   - E1: "grammar-of-English" generator with uniform structure sampling and LLM paraphrases, with random construction-family hold-outs.
   - E2: knowledge-preserving adaptation (adapter scaling at inference, which is free; upper-layer LoRA; replay/KL).
   - E3: interlingua pipeline baseline (LLM -> meaning frame -> deterministic realiser).
   - E4: count-range and decomposition study for adjectives and sentences.
   - E5 (stretch): multi-language meta-training.

---

## 1. Why fine-tuned networks generalise only within the support of the training data

### Evidence

- **SCAN.** Lake & Baroni, "Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks", ICML 2018. https://arxiv.org/abs/1711.00350
  - Seq2seq models succeed when test commands differ slightly from training ("mix-and-match").
  - They "fail spectacularly" when the test needs systematic composition, e.g. a primitive seen only in isolation, or longer outputs.
- **COGS.** Kim & Linzen, "COGS: A Compositional Generalization Challenge Based on Semantic Interpretation", EMNLP 2020. https://aclanthology.org/2020.emnlp-main.731
  - In-distribution accuracy 96-99%; generalisation accuracy 16-35%, with high seed variance.
  - The gaps include new combinations of familiar *structures* (e.g. a PP modifying the subject when only object-PPs were seen) and deeper recursion.
- **CFQ / MCD splits.** Keysers et al., "Measuring Compositional Generalization: A Comprehensive Method on Realistic Data", ICLR 2020. https://research.google/pubs/measuring-compositional-generalization-a-comprehensive-method-on-realistic-data/
  - Accuracy correlates strongly and negatively with *compound divergence* between train and test.
  - Accuracy is >95% at low divergence and <20% at maximum divergence. Atoms (words) are shared; what differs is how they combine.
- **Compositionality decomposed.** Hupkes, Dankers, Mul & Bruni, JAIR 67, 2020. https://jair.org/index.php/jair/article/view/11674
  - Splits "compositional generalisation" into five tests: systematicity, productivity (longer than seen), substitutivity (synonyms), localism, and overgeneralisation.
  - Our stress tests map onto these almost one-to-one:
    - synonyms -> substitutivity
    - 3 adjectives / 3 sentences -> productivity
    - object fronting -> systematicity
- **Structural vs lexical generalisation.** Yao & Koller, "Structural generalization is hard for sequence-to-sequence models", EMNLP 2022. https://arxiv.org/abs/2210.13050
  - Seq2seq models (including pretrained ones) "achieve very low accuracy in generalizing to linguistic structures that were not seen in training".
  - The related COGS literature reports pretrained Transformers near 0% on structural splits such as subject-PP and on recursion depth beyond training (summarised in https://arxiv.org/pdf/2303.13716 and related work).
- **Unobserved local structures.** Bogin, Gupta & Berant, "Unobserved Local Structures Make Compositional Generalization Hard", EMNLP 2022. https://aclanthology.org/2022.emnlp-main.175
  - Different models fail on the *same* test instances.
  - A test instance is hard if it contains a *local structure* (a small sub-graph of the program) that was never seen in training. This predicts instance-level failure better than alternatives.
  - **This is the single best predictive model of our results:**
    - "Passive with an unseen verb" contains no new local structure (passive was seen; the verb is just a slot), so it works.
    - "Object fronting" contains a new local structure (object-in-initial-position) that "location fronting" does not supply, so it fails.
- **Shortcut learning.** Geirhos et al., "Shortcut Learning in Deep Neural Networks", Nature Machine Intelligence 2(11), 2020. https://arxiv.org/abs/2004.07780
  - Networks adopt decision rules that work on the training/benchmark distribution but fail to transfer.
  - **Our case:** "first noun = doer" and "position k in English -> slot k in target" are perfect shortcuts on template data.
- **Simplicity bias.** Shah et al., "The Pitfalls of Simplicity Bias in Neural Networks", NeurIPS 2020. https://arxiv.org/abs/2006.07710
  - When a simple feature suffices, SGD-trained nets use *only* the simple feature, even when it is less predictive than a complex one.
  - Positional template filling is the "simple feature" of our task.
- **HANS.** McCoy, Pavlick & Linzen, "Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference", ACL 2019. https://aclanthology.org/P19-1334/
  - BERT trained on MNLI adopts lexical-overlap/subsequence heuristics and collapses on a controlled set where the heuristics fail.
  - This is the canonical demonstration that high held-out accuracy can hide a heuristic. Our "held-out combinations 95-99%" is the analogue of MNLI accuracy.
- **Training details matter too.** Csordás, Irie & Schmidhuber, "The Devil is in the Detail: Simple Tricks Improve Systematic Generalization of Transformers", EMNLP 2021. https://aclanthology.org/2021.emnlp-main.49
  - Early stopping on an IID validation set, embedding scaling and relative positional encoding change systematic-generalisation results drastically (COGS 35% -> 81%).
  - **Implication for us:** model selection on an IID dev set (the default in our runs) can pick the *least* systematic checkpoint. **[unverified]**: whether this applies to LoRA on a pretrained 0.6B model; worth logging challenge-suite accuracy per epoch.

### How this applies to us

On template data, every rule that maps *English position* to *target slot* is consistent with 100% of the training set. The model has no pressure to learn "who-did-what-to-whom" (thematic roles) rather than "word 2 is the doer". R1 added passive, so the positional shortcut became inconsistent *for that construction*. The model then learned something role-like that transferred to unseen verbs (local structure seen). But no training example ever made "fronted object" distinct from "fronted location", so nothing forced a general "constituent movement" rule. Bogin et al. say this is predictable.

---

## 2. What makes training data "representative"? Diversity vs quantity

### Evidence

- **Structural diversity beats quantity.** Oren, Herzig & Berant, "Finding needles in a haystack: Sampling Structurally-diverse Training Sets from Synthetic Data for Compositional Generalization", EMNLP 2021. https://aclanthology.org/2021.emnlp-main.843
  - Abstract templates are obtained by replacing constants with types, giving 251 templates.
  - *Uniform-over-templates* (UAT) or compound-max-entropy sampling of 5K synthetic examples beats 1M uniformly sampled ones on compositional splits: 41.4 vs 37.7 EM, a "200x improvement in data efficiency".
  - IID accuracy is unchanged.
  - **Key lesson:** the *distribution over structures* matters, not the count of sentences.
- **Uniform sampling over meanings works almost as well as knowing the test.** Yao & Koller, "Simple and effective data augmentation for compositional generalization", NAACL 2024. https://aclanthology.org/2024.naacl-long.25/
  - Sample meaning representations, then back-translate them to text.
  - "Sampling from a uniform distribution performs almost as well as sampling from the test distribution", and greatly outperforms sampling from the training distribution.
  - **This is the most direct evidence for the user's goal:** a *test-agnostic* sampling principle (uniform over structures) recovers most of the benefit of test-targeted sampling.
- **More distinct primitives -> near-perfect one-shot generalisation.** Patel, Bhattamishra, Blunsom & Goyal, "Revisiting the Compositional Generalization Abilities of Neural Sequence Models", ACL 2022. https://aclanthology.org/2022.acl-short.46
  - Same training-set size (<=13.2k on SCAN); add new primitives ("swim", "clap"...) that follow the same grammar.
  - SCAN add-jump accuracy rises from ~0 to near 100% as extra primitives grow to 50-100 (their Fig. 2).
  - Mechanism: the embedding of the isolated primitive moves closer to the class of other primitives (Figs. 3-4).
  - **Lesson:** *variety within a category* is what makes the model treat the category as a variable rather than memorising members.
- **Dataset complexity helps.** Zhou, Jiang & Bansal, "Data Factors for Better Compositional Generalization", EMNLP 2023. https://aclanthology.org/2023.emnlp-main.898
  - More complex/diverse training sets give better generalisation on several challenges. They give "more diverse examples" and they "prevent ungeneralizable memorization due to reduced example repetition frequency".
  - On synthetic data, simple examples induce stronger compositionality than hard ones; on natural data, a balanced mix works best.
- **Task diversity beats examples per task.** Zhang, Wang & Charton, "Instruction Diversity Drives Generalization To Unseen Tasks", arXiv 2024. https://arxiv.org/abs/2402.10891
  - With string-rewriting (Markov-algorithm) tasks: "Generalization emerges once a diverse enough set of tasks is provided, even though very few examples are provided for each task."
  - Semantic diversity of rules matters, not just their count.
- **Diversity also matters in context.** Levy, Bogin & Berant, "Diverse Demonstrations Improve In-context Compositional Generalization", ACL 2023. https://arxiv.org/abs/2212.06800
  - Selecting demonstrations that *collectively cover* the needed structures beats selecting the most similar ones, both in-context and with fine-tuning.
- **Quality and diversity over quantity in fine-tuning.** Zhou et al., "LIMA: Less Is More for Alignment", NeurIPS 2023. https://arxiv.org/abs/2305.11206
  - 1,000 diverse, curated examples suffice for broad behaviour, because knowledge comes from pretraining.
  - This is a different task (alignment); cite only as support for "diversity > volume".
- **Pretraining helps compositional generalisation, but not completely.** Furrer, van Zee, Scales & Schärli, "Compositional Generalization in Semantic Parsing: Pre-training vs. Specialized Architectures", 2020. https://arxiv.org/abs/2007.08970
  - MLM pretraining rivals SCAN-specific architectures on primitive hold-outs and helps on CFQ.
  - Qiu et al. (EMNLP 2022, below) show fine-tuning has flat or negative scaling on compositional splits.

### Is there evidence that structural diversity (not targeted fixes) gives generalisation to *unseen* structures?

Yes, with a precise meaning:
- **Oren et al. and Yao & Koller:** the test structures are unseen *compositions* of seen atoms, and diversity is *not* chosen with the test in mind (uniform over templates / meanings).
- **Patel:** an unseen *member* of a category generalises once the category is diverse.
- **What is *not* shown:** diversity making a model handle a *construction type* whose local structures never occur in training (Bogin). Diversity raises the chance that a test construction's local pieces are covered. It does not create pieces from nothing.

### How this applies to us

- Our English side has **one template**. In Oren's terms we trained on 1 abstract template (plus R1's handful). This is the extreme low-diversity corner of every curve above.
- The correct move is not "add object fronting". It is to replace the template with a **probabilistic grammar of English clauses** whose rules are taken from a *description of English* (clause types, voice, constituent order, modification, coordination, determiners, numerals, optional arguments). Then sample **uniformly over structure types**, not over sentences.
- Within each lexical category, use **many distinct English words** (Patel), e.g. hundreds of English nouns mapping onto the target lexicon. See section 3 for synonyms.

---

## 3. Using the base model's English knowledge instead of overwriting it

### Evidence

- **Feature distortion.** Kumar, Raghunathan, Jones, Ma & Liang, "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution", ICLR 2022 (oral). https://arxiv.org/abs/2202.10054
  - Full fine-tuning gets +2% ID but -7% OOD vs linear probing on 10 shifts.
  - Lower-layer features get distorted while the head is learned.
  - LP-FT (probe first, then fine-tune) gives +10% OOD over FT.
  - **[unverified]** transfer to generative LoRA: the analogue would be "train the new glyph embeddings / output head first, then LoRA", to avoid large early gradients through the network.
- **Surgical fine-tuning.** Lee et al., "Surgical Fine-Tuning Improves Adaptation to Distribution Shifts", ICLR 2023. https://arxiv.org/abs/2210.11466
  - Tuning only a subset of layers matches or beats full fine-tuning.
  - *Which* subset depends on the shift type: input-level shifts favour early layers, output-level shifts favour later layers.
  - **For us [unverified reasoning]:** the shift we want to *keep invariant* is the English input (keep lower and middle layers that encode English lexical semantics), and what is new is the output language. That argues for **LoRA on upper layers only**, plus trainable glyph embeddings and LM head rows.
- **LoRA forgets less but learns less.** Biderman et al., "LoRA Learns Less and Forgets Less", TMLR 2024. https://arxiv.org/abs/2405.09673
  - LoRA preserves base-model capabilities outside the target domain better than full fine-tuning.
- **Forgetting localises in "intruder dimensions".** Shuttleworth, Andreas, Torralba & Sharma, "LoRA vs Full Fine-tuning: An Illusion of Equivalence", NeurIPS 2025. https://arxiv.org/abs/2410.21228
  - LoRA creates high-singular-value directions that are absent in the base model, and forgetting is concentrated there.
  - *Scaling them down* restores pretraining-distribution modelling with little task loss.
  - **Cheap test for us:** scale the LoRA delta (alpha multiplier) at inference.
- **Weight-space interpolation.** Wortsman et al., "Robust fine-tuning of zero-shot models" (WiSE-FT), CVPR 2022. https://arxiv.org/abs/2109.01903
  - Interpolating fine-tuned and pretrained weights improves OOD by 4-6 points at no ID cost and no extra compute.
  - LoRA-scale-at-inference is the LoRA analogue. **[unverified]** for generation tasks with new tokens: the glyph embeddings must stay fully trained, so only the LoRA delta should be scaled.
- **Prefix/prompt tuning.**
  - Li & Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation", ACL 2021. https://aclanthology.org/2021.acl-long.353
    - Freezes the LM; "extrapolates better to examples with topics that are unseen during training".
  - Qiu et al., "Evaluating the Impact of Model Scale for Compositional Generalization in Semantic Parsing", EMNLP 2022. https://aclanthology.org/2022.emnlp-main.624
    - Fine-tuning has flat or negative scaling on compositional OOD; prompt tuning "can outperform fine-tuning".
  - **Caveat for us [unverified]:** a 0.6B model learning a whole new script and grammar may underfit with prompt tuning alone.
- **Regularising toward the base model.**
  - Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (EWC), PNAS 2017. https://pmc.ncbi.nlm.nih.gov/articles/PMC5380101
  - Aghajanyan et al., "Better Fine-Tuning by Reducing Representational Collapse" (R3F/R4F), ICLR 2021. https://arxiv.org/abs/2008.03156
  - Yang et al., "Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning" (SDFT), ACL 2024. https://aclanthology.org/2024.acl-long.58
  - Ibrahim et al., "Simple and Scalable Strategies to Continually Pre-train Large Language Models", TMLR 2024 (replay of previous data prevents forgetting). https://arxiv.org/abs/2403.08763
- **Pretrained models are more robust OOD, and fine-tuned ones can be too.**
  - Hendrycks et al., "Pretrained Transformers Improve Out-of-Distribution Robustness", ACL 2020. https://aclanthology.org/2020.acl-main.244
  - Mosbach et al., "Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation", Findings ACL 2023. https://arxiv.org/abs/2305.16938
    - Fine-tuned LMs *can* generalise OOD (e.g. on HANS) comparably to ICL. The outcome depends on model size and number of examples.
- **Synonyms as a known failure mode.** Gan et al., "Towards Robustness of Text-to-SQL Models against Synonym Substitution" (Spider-Syn), ACL 2021. https://aclanthology.org/2021.acl-long.195
  - Fine-tuned parsers rely on lexical matching, and accuracy "dramatically drops" with non-adversarial synonyms.
  - Synonym annotations or adversarial training help.

### Why our model ignored Qwen's synonym knowledge (diagnosis, **[unverified]** but testable)

1. **Closed-class mapping.** With one English word per target lexeme, the cheapest solution is a lookup on *token identity*, not on meaning. Patel's result says the model will only treat a slot as "any word meaning X" if it sees *many surface forms* per meaning.
2. **Adaptation everywhere.** LoRA on all linear layers lets lower layers re-purpose English representations for the template (feature distortion).
3. **Weak base knowledge.** Qwen3-0.6B's own synonym knowledge may be weak. **This must be measured before blaming the training.** E.g. prompt the base model: "A word that means 'big' is ..."; or check the embedding/hidden-state cosine of synonym pairs vs random pairs.

### Actionable

- Many-to-one lexicon on the English side, built *from a general resource* (WordNet synsets, or an LLM asked for synonyms of each lexeme), never from the test list. Hold out a random 30% of synonyms per lexeme for evaluation.
- Adapter-scale sweep at inference (free).
- Upper-layer-only LoRA vs all-layer LoRA, at the same data.
- 10-20% replay of generic English text (LM loss), or a KL-to-base term on the English prompt tokens.

---

## 4. Paraphrasing the English side (LLM / back-translation / self-instruct)

### Evidence

- **Synthetic grammar + paraphrase.** Wang, Berant & Liang, "Building a Semantic Parser Overnight", ACL 2015. https://aclanthology.org/P15-1129/
  - The original "generate canonical utterances from a grammar, then paraphrase" recipe; this is our situation.
- **AutoQA.** Xu, Semnani, Campagna & Lam, "AutoQA: From Databases To QA Semantic Parsers With Only Synthetic Training Data", EMNLP 2020. https://arxiv.org/abs/2010.04806
  - Template synthesis + automatic (filtered) paraphrasing gives a parser trained on *only synthetic data* that is within 6.4 points of a model trained on human-written data, when tested on natural questions.
  - **This is the closest published analogue of "make the synthetic English side representative".**
- **Limits of paraphrase data.** Herzig & Berant, "Don't paraphrase, detect! Rapid and Effective Data Collection for Semantic Parsing", EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1394/
  - Paraphrasing synthetic utterances leaves a *distribution mismatch* with real user language (51.3% vs 70.6% for their detection approach).
  - Paraphrases inherit the structural bias of the seed grammar.
- **Projection onto the synthetic support.** Marzoev et al., "Unnatural Language Processing: Bridging the Gap Between Synthetic and Natural Language Data", 2020. https://arxiv.org/abs/2004.13645
  - The *inverse* strategy: train only on synthetic data, then at test time *project* natural utterances onto the support of the synthetic language. This matches models trained on natural data in several domains.
  - This is the theoretical basis of a "normalise-then-translate" pipeline (section 5).
- **Back-translation.** Sennrich, Haddow & Birch, "Improving Neural Machine Translation Models with Monolingual Data", ACL 2016. https://aclanthology.org/P16-1009/
  - Standard MT augmentation. **For us:** we can sample target-side meanings and generate *many* English renderings (the Yao & Koller recipe).
- **Self-Instruct.** Wang et al., "Self-Instruct: Aligning Language Models with Self-Generated Instructions", ACL 2023. https://aclanthology.org/2023.acl-long.754
  - LLM-generated diversity with filtering (dedup and similarity filters).
- **LLM-generated natural perturbations as a test.** Chang et al., "Dr.Spider: A Diagnostic Evaluation Benchmark towards Text-to-SQL Robustness", ICLR 2023. https://arxiv.org/abs/2301.08881
  - Even the most robust model drops 14% overall and 50.7% on the hardest perturbation.
  - Natural-language perturbations were generated with PLMs. **Useful as a recipe for our *evaluation* paraphrases**, which must come from a different generator/prompt than training paraphrases.
- **Natural variation vs compositional generalisation.** Shaw, Chang, Pasupat & Toutanova, "Compositional Generalization and Natural Language Variation: Can a Semantic Parsing Approach Handle Both?", ACL 2021. https://arxiv.org/abs/2010.12725
  - Grammar-based methods handle composition; pretrained seq2seq handles variation; neither does both alone (the motivation for NQG-T5).
  - **Our two failure families are exactly this dichotomy.**
- **Recent LLM augmentation work** (e.g. "Diversity-Oriented Data Augmentation with Large Language Models", 2025, https://arxiv.org/abs/2502.11671) reports diversity gains. Some sample-selection studies find OOD gains are "seldom and marginal" (https://arxiv.org/abs/2501.08276). Paraphrase augmentation is not a free win.

### Limits relevant to us

1. **The paraphraser can break meaning.** The target is computed from the *meaning*, so every paraphrase must keep it exactly (role swaps in passives, scope of negation). Filter by having a *second* model parse the paraphrase back to the meaning frame, and keep only exact matches (AutoQA-style filtered paraphrasing).
2. **Paraphrases only cover what the paraphraser tends to produce.** LLM paraphrase distributions are themselves narrow (formal register, few typos, rarely 4 coordinated clauses). They fix family (A) much more than family (B).
3. **Using the same LLM/prompt for train and test paraphrases leaks the distribution.** Use different models or prompts, or human-written sentences, for the sealed suite.

---

## 5. Generalisation by construction: interlingua / meaning-frame pipelines

### Evidence

- **Paraphrase into a controlled sublanguage, with constrained decoding.** Shin et al., "Constrained Language Models Yield Few-Shot Semantic Parsers", EMNLP 2021. https://arxiv.org/abs/2104.08768
  - LMs paraphrase inputs into an English-like canonical sublanguage under grammar-constrained decoding, which is then mapped deterministically.
  - Strong few-shot parsers on Overnight, Break and SMCalFlow.
- **Grammar-constrained decoding.** Geng, Josifoski, Peyrard & West, "Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning", EMNLP 2023. https://aclanthology.org/2023.emnlp-main.674
  - Grammar-constrained LMs beat unconstrained ones and sometimes fine-tuned task models. Input-dependent grammars are supported.
- **Decompose-then-parse with LLMs.** Drozdov et al., "Compositional Semantic Parsing with Large Language Models", ICLR 2023. https://arxiv.org/abs/2209.15003
  - Dynamic least-to-most prompting: 95.0% CFQ with 1% of the data; 99.2% COGS. Decomposition + exemplar selection is what drives compositional generalisation.
- **Hybrid grammar + neural.** NQG-T5 (Shaw et al. 2021, above): a high-precision grammar model with a pretrained seq2seq fallback.
- **Abstract syntax as interlingua.** Ranta, Angelov, Gruzitis & Kolachina, "Abstract Syntax as Interlingua: Scaling Up the Grammatical Framework from Controlled Languages to Robust Pipelines", Computational Linguistics 46(2), 2020. https://research.chalmers.se/en/publication/517996
  - The GF tradition: language-neutral abstract syntax + per-language concrete grammars, with deterministic generation. This is effectively what our grammar generator already is on the target side.
- **Learning a new language from a grammar book.** Tanzer et al., "A Benchmark for Learning to Translate a New Language from One Grammar Book" (MTOB, Kalamang), ICLR 2024. https://arxiv.org/abs/2309.16575
  - LLMs with grammar and wordlist in context reach 45.8 chrF En->Kalamang vs 57.0 for a human learner.
  - Shows the *in-context* route works partially, with large models.

### Trade-offs for us

| | End-to-end LoRA (current) | Pipeline: English -> meaning frame (LLM, constrained) -> deterministic realiser |
|---|---|---|
| Target-grammar correctness | Learned (95-99% on seen structures) | **100% by construction** for any frame the realiser supports |
| English variation (synonyms, passive, fronting, fillers, casing) | Must be covered by data | Handled by the LLM's English competence. **[unverified]** for 0.6B; probably needs Qwen3-1.7B/4B in 4-bit |
| New structures (3 adjectives, 3 sentences) | Productivity is fragile | Free *if* the frame schema has lists (adjectives[], clauses[]) |
| Things not in the frame schema (e.g. relative clauses) | Fails silently with garbage | Fails *loudly* (parse error / refusal). Arguably better behaviour |
| Course-project value | "Model learned the language" | "Model learned to *understand English into a frame*". Arguably less of a language-learning result |
| Cost | ~1h/run on T4 | Frame-parser fine-tune (small) or few-shot + constrained decoding; realiser is the existing generator |

**Honest framing:** the pipeline shows *what is achievable* and serves as an upper-bound baseline. It does not answer "can the end-to-end model learn the language". Report both.

---

## 6. Meta-learning for compositionality: training on many invented languages

### Evidence

- **MLC.** Lake & Baroni, "Human-like systematic generalization through a meta-learning neural network", Nature 623, 2023. https://www.cs.princeton.edu/node/5254 (Nature: s41586-023-06668-3)
  - A standard Transformer is meta-trained on a stream of episodes. Each episode has its own randomly sampled grammar, and the model must infer and apply it from study examples.
  - It reaches human-like systematicity and flexibility, which neither symbolic models nor ordinary nets achieve.
- **Precursors and relatives.**
  - Lake, "Compositional generalization through meta sequence-to-sequence learning", NeurIPS 2019. https://arxiv.org/abs/1906.05381
  - Conklin, Wang, Smith & Titov, "Meta-Learning to Compositionally Generalize", ACL 2021. https://aclanthology.org/2021.acl-long.258
  - McCoy & Griffiths, "Modeling rapid language learning by distilling Bayesian priors into artificial neural networks", Nature Communications 2025. https://ideas.repec.org/a/nat/natcom/v16y2025i1d10.1038_s41467-025-59957-y.html
    - Meta-train on formal languages sampled from a simplicity prior. The network then learns new formal languages from few examples, close to symbolic models.
- **Pre-pretraining on formal languages.** Hu, Petty, Shi, Merrill & Linzen, "Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases", ACL 2025. https://aclanthology.org/2025.acl-long.478/
  - Transfer works when the formal language captures natural-language dependency structure *and* stays within what the architecture can compute.
- **Critique / limits.** Woydt et al., "Fodor and Pylyshyn's Legacy - Still No Human-like Systematic Compositionality in Neural Networks", arXiv 2506.01820 (2025/26). https://arxiv.org/abs/2506.01820
  - 179/200 MLC validation episodes reuse grammar-operation combinations seen in the 100k training episodes.
  - Inputs are at most 10 tokens and outputs at most 8; there are erratic non-systematic errors (e.g. "twice"/"thrice" confusions).
  - **Meta-learning generalises within the meta-distribution of grammars, the same support limit one level up.**
- **LMs have natural-language-shaped inductive biases.** Kallini et al., "Mission: Impossible Language Models", ACL 2024. https://aclanthology.org/2024.acl-long.787
  - GPT-2 learns English-like languages more easily than counting-based "impossible" ones.
  - **Relevance:** our invented grammar should stay within natural-language-like rules; counting-based rules will be harder.

### Relevance to us

**Opportunity.** Our generator can emit unlimited invented languages (different word orders, case markers, agreement, glyph inventories). Two levels:
1. **Multi-language training (cheap, [unverified] benefit).** Train on K sibling languages, each with a language-ID token, sharing the *same English generator*.
   - Hypothesis (by analogy with Patel and Zhang-Wang-Charton): seeing the same English structure map to several different output orders forces the model to represent English *roles* rather than English *positions*. Positional copying stops being a universal shortcut.
   - This directly attacks "first noun = doer".
2. **True MLC-style in-context meta-learning.** Each episode gives the grammar/lexicon or study pairs in context, and the model must translate a query.
   - Much more expensive (long contexts, many episodes). Evidence exists only in tiny domains. Risky for a course timeline.

**What meta-learning will *not* do:** fix English-side variation that never appears in any episode. All languages share the English generator, so if the English is one template, every episode has the same shortcut on the input side.

---

## 7. Task-agnostic augmentation (creating coverage without knowing the test)

Each method below generates new examples from *properties of the training data itself*, not from test categories.

- **GECA.** Andreas, "Good-Enough Compositional Data Augmentation", ACL 2020. https://aclanthology.org/2020.acl-main.676
  - Swap fragments that occur in a shared context. Up to 87% error reduction on SCAN diagnostics, 16% on semantic parsing.
- **R&R.** Akyürek, Akyürek & Andreas, "Learning to Recombine and Resample Data for Compositional Generalization", ICLR 2021. https://arxiv.org/abs/2010.03706
  - Learned recombination plus *resampling that favours rare combinations*, which encourages extrapolation.
- **SeqMix.** Guo, Kim & Rush, "Sequence-Level Mixed Sample Data Augmentation", EMNLP 2020. https://aclanthology.org/2020.emnlp-main.447
  - Soft input/output mixing; about +1 BLEU on MT and gains on SCAN.
- **SUBS.** Yang, Zhang & Yang, "SUBS: Subtree Substitution for Compositional Semantic Parsing", NAACL 2022. https://aclanthology.org/2022.naacl-main.12
  - Swap subtrees with the same semantic function. SOTA on GeoQuery compositional split.
- **CSL.** Qiu et al., "Improving Compositional Generalization with Latent Structure and Data Augmentation", NAACL 2022. https://aclanthology.org/2022.naacl-main.323
  - Induce a quasi-synchronous CFG from the data, sample recombined pairs, add them to T5 fine-tuning. SOTA on tasks that need both natural variation and novel composition.
- **LexSym.** Akyürek & Andreas, "LexSym: Compositionality as Lexical Symmetry", ACL 2023. https://aclanthology.org/2023.acl-long.38
  - Theory: if a task is solvable by a compositional model, there is a data-augmentation scheme (a *symmetry of the data distribution*) that imparts the same inductive bias to *any* model.
  - **This is the principled framing we want:** augment with *transformations that should leave meaning invariant*, chosen from a theory of the language, not from the test.
- **Uniform MR sampling + back-translation.** Yao & Koller 2024 (section 2): the simplest and most relevant to us.

### Applied to us

We do not need to *induce* a grammar (GECA, CSL), because we *have* one. The principled version is:
1. Write down the **symmetry classes of English input that should not change the output**, from a linguistic description of English, not from failures:
   - orthographic: case, whitespace, punctuation, light typos
   - lexical: synonyms from a lexical resource
   - syntactic: voice, constituent fronting of *any* constituent, clefts, dative alternation
   - discourse: one vs several sentences, conjunction
   - pragmatic fillers: discourse markers
2. Implement each as a *generator option*.
3. Sample *uniformly* over combinations.
4. **Hold out at random** a subset of the families, and of instances within families, purely to *measure* generalisation. Never add the held-out families back after seeing results.

The difference from "tailoring": the family list comes from a description of English written *before* testing. The held-out portion is selected at random, and the sealed suite (section 8) is written independently.

---

## 8. Evaluation philosophy: not teaching to the test

### Evidence

- **Critique of IID evaluation.** Linzen, "How Can We Accelerate Progress Towards Human-like Linguistic Generalization?", ACL 2020. https://aclanthology.org/2020.acl-main.465
  - The "PAID" paradigm (pretrain, fine-tune, test IID) rewards exploiting dataset-specific statistics. Argues for evaluation that rewards human-like generalisation.
- **Taxonomy of generalisation research.** Hupkes et al., "A taxonomy and review of generalization research in NLP", Nature Machine Intelligence 5(10), 2023. https://arxiv.org/abs/2210.03050
  - Five axes: motivation, generalisation type, shift type, shift source, shift locus.
  - **Use it to *declare* what kind of generalisation each test measures.**
- **CheckList.** Ribeiro, Wu, Guestrin & Singh, "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList", ACL 2020. https://arxiv.org/abs/2005.04118
  - A capability x test-type matrix for systematic test ideation; users found ~3x more bugs.
- **Contrast sets.** Gardner et al., "Evaluating Models' Local Decision Boundaries via Contrast Sets", Findings EMNLP 2020. https://arxiv.org/abs/2004.02709
  - Minimal meaning-changing perturbations; up to 25% drops.
- **Dynabench.** Kiela et al., "Dynabench: Rethinking Benchmarking in NLP", NAACL 2021. https://aclanthology.org/2021.naacl-main.324
  - Adversarial human-in-the-loop data collection. Note the risk: if failures found this way are fed back into training, the benchmark becomes the training set. That is exactly the user's worry.
- **MCD splits.** Keysers et al. 2020 (section 1): make train and test *compound*-divergent while keeping atoms shared, by algorithm rather than by hand.
- **No free lunch.** Wolpert, "The Lack of A Priori Distinctions Between Learning Algorithms", Neural Computation 8(7), 1996. https://mlanthology.org/neco/1996/wolpert1996neco-lack
  - Averaged over all possible targets, no learner beats another on off-training-set error.
  - Generalisation to unseen cases comes *only* from inductive bias matching the true target. For us, the inductive bias comes from (i) Qwen's pretraining on English, (ii) the structure of our data, and (iii) the architecture/adaptation choices.

### What claims are realistic

- **Not realistic:** "passes all tests automatically". There will always be an English construction (ellipsis, relative clauses, idioms, sarcasm, code-switching) outside the generator's scope. NFL says the off-support behaviour is decided by bias, not data.
- **Realistic and publishable:**
  1. Generalisation to *randomly held-out* structure families from an independently specified distribution of English, with confidence intervals and multiple seeds.
  2. Performance on a *sealed* natural-English suite written by someone else, never used for any design decision.
  3. *Calibrated failure*: out-of-scope input gives refusal or a flagged low-confidence output, not confident garbage.
  4. A *curve* showing how sealed-suite accuracy scales with structural diversity at a fixed example count. This is the scientific answer to the user's question.

### Concrete protocol

1. **Write the capability matrix first** (CheckList x Hupkes axes).
   - Rows: lexical substitution, morphology/tense, voice, constituent order, modification count, clause count, argument structure (intransitive/ditransitive), determiners/quantifiers/numerals, orthographic noise, discourse fillers, names/OOV, out-of-scope/non-English.
   - Columns: seen-construction/unseen-lexeme, unseen-construction, unseen-count, natural-paraphrase.
2. **Split each row's constructions in two:**
   - *dev-challenge*: allowed for diagnosis, may inform *generic* design choices;
   - *sealed*: looked at once, at the end.
3. **Sealed natural sentences:** ~200-400 English sentences written by classmates, or by a *different* LLM with a prompt that does not mention our generator. Constrain them only to the lexicon/meaning space.
   - Annotate each with a meaning frame (LLM draft + manual check).
   - Produce gold targets with the deterministic realiser.
   - Mark sentences whose meaning is outside the frame schema as "out-of-scope, should refuse".
4. **Freeze and hash the sealed file**; record the date. Training data generators may never import it.
5. **Report both** dev-challenge and sealed results, and report whether any decision was made after looking at the sealed set (ideally none).
6. **Model selection:** pick the checkpoint by dev-challenge accuracy, not IID loss (Csordás et al. 2021).

---

## 9. Counting, length and recursion (2 -> 3 adjectives, 2 -> 3 sentences)

### Evidence

- **Length generalisation.** Anil et al., "Exploring Length Generalization in Large Language Models", NeurIPS 2022 (oral). https://arxiv.org/abs/2207.04901
  - Naive fine-tuning shows length-generalisation deficits "independent of model scale".
  - Scratchpad + in-context learning helps dramatically.
- **When Transformers do length-generalise.** Zhou et al., "What Algorithms can Transformers Learn? A Study in Length Generalization", ICLR 2024. https://arxiv.org/abs/2310.16028
  - The RASP-Generalization conjecture: Transformers length-generalise when a *short RASP-L program* solves the task for all lengths. Output formats that make the per-step rule simple and local help; scratchpads that complicate it hurt.
- **Fragility.** Zhou, Alon, Chen, Wang, Agarwal & Zhou, "Transformers Can Achieve Length Generalization But Not Robustly", 2024. https://arxiv.org/abs/2402.09371
  - 2.5x length extrapolation on addition with the right format and position encodings (FIRE, randomized PE, index hints).
  - Large variance across seeds and data order.
- **Position encodings.** Kazemnejad et al., "The Impact of Positional Encoding on Length Generalization in Transformers", NeurIPS 2023. https://arxiv.org/abs/2305.19466
  - NoPE beats RoPE/ALiBi/APE for length generalisation in their setting. **Not actionable for us**: Qwen uses RoPE and we cannot change it in a LoRA run.
- **The EOS decision.** Newman, Hewitt, Liang & Manning, "The EOS Decision and Length Extrapolation", 2020. https://arxiv.org/abs/2010.07174
  - Models stop too early when they have never seen longer outputs. The EOS decision itself encodes "outputs are this long".
  - **Plausibly part of our 3-sentence failure [unverified].**
- **Scratchpads.** Nye et al., "Show Your Work: Scratchpads for Intermediate Computation with Language Models", ICLR 2022 workshop. https://arxiv.org/abs/2112.00114
- **Recursion depth in COGS.** Trained at depth 0-2, tested at 3-12, seq2seq models degrade quickly with depth. Compositional parsers do not degrade (see the ReCOGS / structural generalisation discussion: https://arxiv.org/pdf/2303.13716, https://arxiv.org/abs/2210.13050).
- **Productivity.** Hupkes et al. 2020: productivity (longer than seen) is one of the five tests; the abstract frames it as a core weakness of all three architectures tested.

### Does training on 1..k enable k+1?

**Evidence says: sometimes, slightly, and not robustly.** It works best when:
- (a) the per-item rule is identical and local (each extra adjective adds the same affix in the same place);
- (b) the training counts are *uniformly* spread over 1..k, so the model cannot memorise "max=2";
- (c) k is not tiny.

Going from k=2 to 3 is the hardest case: with only {1,2}, "exactly two" is as simple a hypothesis as "any number".

### Actionable

1. **Count-range study.** Train with counts uniform over 0..5 (adjectives) and 1..4 (sentences). Test at 6-8 and 5-6. Report the extrapolation curve rather than a single point. Use 2-3 seeds; Zhou et al. 2024 show high seed variance.
2. **Decomposition by construction (principled, not test-targeted):**
   - Split multi-sentence input at sentence boundaries, translate each sentence independently, and join. This handles any N with the 1-sentence model.
   - For adjectives, the pipeline frame has an adjectives list.
   - Report this as an architectural choice, separately from end-to-end numbers.
3. **Do not train on counts used in the sealed suite.** Decide the training range *before* looking at the suite (e.g. train 0..5, sealed tests 6..8).

---

## 10. Typos and character-level robustness

### Evidence

- **Noise is type-specific.** Belinkov & Bisk, "Synthetic and Natural Noise Both Break Neural Machine Translation", ICLR 2018. https://arxiv.org/abs/1711.02173
  - Models break under moderate noise.
  - Crucially: "The three broad classes of noise (scrambling, Key, Nat) are not mutually-beneficial. Models trained on one do not perform well on the others", and "only models trained on natural noise can reasonably translate natural noise".
  - A *mixture* of noise types gave robustness to all the types in the mixture.
  - **This is precisely the user's worry, confirmed empirically for noise.**
- **Mild random noise helps.** Karpukhin, Levy, Eisenstein & Ghazvininejad, "Training on Synthetic Noise Improves Robustness to Natural Noise in Machine Translation", W-NUT 2019. https://aclanthology.org/D19-5506/
  - Training on a *mild amount of random* synthetic character noise dramatically improves robustness to natural spelling variation, without hurting clean text.
  - This partly contradicts Belinkov & Bisk on transfer. The difference is plausibly the mixture/randomness of the noise **[unverified]**.
- **BPE-dropout.** Provilkov, Emelianenko & Voita, "BPE-Dropout: Simple and Effective Subword Regularization", ACL 2020. https://aclanthology.org/2020.acl-main.170
  - Stochastic segmentation during training; up to +2.3 BLEU; increases robustness to segmentation variation.
  - A *representation-level* regulariser that needs no typo list.
  - **[unverified]** feasibility with Qwen's byte-level BPE tokenizer in HF `tokenizers` (the BPE model has a dropout parameter, but byte-level merges in Qwen's tokenizer may need custom handling).
- **Spell-correction front end.** Pruthi, Dhingra & Lipton, "Combating Adversarial Misspellings with Robust Word Recognition", ACL 2019. https://aclanthology.org/P19-1561/
  - A word-recognition front end restores BERT sentiment accuracy from 45.8% to 75% under 1-character attacks. It beats adversarial training and off-the-shelf spell-checkers.

### Why we stay at ~13% even after training on typos [unverified diagnosis]

- A typo changes the BPE segmentation completely: "elephant" may be one token, "elepahnt" 3-4 unrelated tokens.
- For a translation task the model must recover the *identity* of the lexeme to choose the target word. That is much harder than keeping sentiment.
- A 0.6B model with LoRA and a small share of typo examples has little signal for this. If R1's typos came from one generator (e.g. only swaps), Belinkov & Bisk predict poor transfer to other typo types.

### Actionable

1. Noise as a *mixture* (swap/drop/insert/keyboard-neighbour/case/whitespace) on a random 10-20% of examples, applied to random words, at mild rates (Karpukhin). Plus BPE-dropout if feasible.
2. Evaluate on a **natural** misspelling list that the training generator never touches. **[unverified]**: such lists exist, e.g. Wikipedia's list of common English misspellings or the Birkbeck spelling error corpus.
3. Pipeline alternative: fuzzy match to the known English lexicon, or an LLM normalisation step (Pruthi; Marzoev "projection").
4. **Be candid in the write-up:** for a closed lexicon, a deterministic normaliser solves typos almost entirely. Whether that counts as "the model learned robustness" is a framing choice to state openly.

---

## 11. Principled (test-agnostic) vs test-targeted interventions

| Intervention | Type | Why |
|---|---|---|
| Grammar-of-English generator, uniform over structure types (Oren; Yao & Koller) | **Principled** | Coverage comes from a description of English, chosen before testing |
| Many surface forms per category: lexicon breadth, synonyms from WordNet/LLM (Patel) | **Principled** | Makes slots variables; synonyms not chosen from the test |
| Random hold-out of construction families for measurement (Keysers MCD) | **Principled evaluation** | Selection by algorithm |
| Mixed random character noise + BPE-dropout (Karpukhin; Provilkov) | **Principled** (invariance family) | Noise model is generic |
| LLM paraphrase of the English side with round-trip filter (AutoQA) | **Principled**, if the paraphrase prompt is generic | Distribution comes from the paraphraser, not from our failures |
| Upper-layer LoRA / adapter scaling / replay / KL (Lee; Shuttleworth; Wortsman; Ibrahim) | **Principled** (inductive bias) | Changes how the model learns, not what it sees |
| Multi-language / meta-training (Lake & Baroni; McCoy & Griffiths) | **Principled** (inductive bias) | Removes positional shortcut by varying output order |
| Pipeline: frame parsing + realiser, sentence splitting (Shin; Drozdov) | **Architectural** | Generalisation by construction; changes what is being claimed |
| "Add object fronting", "add 3-adjective examples", "add these synonym pairs" | **Test-targeted patch** | Each fixes exactly the observed failure, as R1 showed |

---

## 12. Ranked experiment plan (cost-aware: Kaggle T4 ~1 h per 50k run, 4 GB laptop GPU for eval)

### E0. Build and freeze the sealed challenge suite (no training; 1-2 days; do first)
- Capability matrix (section 8).
- Dev-challenge vs sealed split by construction.
- ~300 independently written natural English sentences with meaning-frame annotation and realiser gold.
- Include an "out-of-scope" bucket.
- Also run the *base* Qwen synonym probe (section 3).
- Evaluate R0 and R1 on it to get a baseline.
- **Rule:** no category from the sealed half is ever added to training.

### E1. "Representative English" at fixed budget (2-3 runs; core answer to the user's question)
- Replace the template with a probabilistic English clause grammar, written from an English grammar reference, not from the stress tests. It covers:
  - voice
  - fronting of any constituent
  - 0-5 adjectives
  - 1-4 clauses / sentences
  - intransitive / transitive / ditransitive
  - determiners, quantifiers, numerals
  - fillers / discourse markers
  - names
  - case / whitespace variation
  - mixed typo noise
- Sample *uniformly over structure signatures* (Oren UAT, Yao & Koller), not over sentences.
- Plus 20-30% LLM paraphrases with round-trip frame filtering.
- Randomly hold out ~20% of structure signatures, and of synonyms, for measurement.
- **Same 50k examples as R0/R1.**
- Compare R0 vs R1 vs E1 on random-held-out signatures, dev-challenge and sealed.
- Optional dose-response: E1 with 10 / 50 / 250 structure types at fixed 50k. This is the scientific answer to "does diversity, not tailoring, give unseen-structure generalisation?"

### E2. Keep Qwen's English (cheap; mostly eval-time)
- (a) **Adapter-scale sweep** at inference on E1's adapter: LoRA delta x {0.25, 0.5, 0.75, 1.0}, glyph embeddings untouched (WiSE-FT / intruder-dimension logic). Zero training cost.
- (b) LoRA on upper half of layers only (+ glyph embeddings / lm_head rows) vs all layers.
- (c) +15% generic-English replay or a KL-to-base term.
- **Metrics:** unseen-synonym and paraphrase buckets, plus IID accuracy (to see the trade-off).

### E3. Interlingua pipeline baseline (1-2 days; strong reference point)
- Qwen3 (0.6B fine-tuned, or 1.7B/4B few-shot in 4-bit) maps English -> JSON meaning frame under grammar-constrained decoding (Geng; Shin).
- The existing deterministic realiser produces glyph text.
- Sentence splitting gives multi-sentence input for free.
- Evaluate on the same sealed suite.
- Report as "generalisation by construction", separate from the end-to-end result.

### E4. Productivity study (2 runs, can share with E1)
- Train counts uniform 0..5 adjectives / 1..4 sentences.
- Test 6-8 / 5-6.
- 2-3 seeds; report the extrapolation curve and seed variance.
- Compare against sentence-split decoding.

### E5 (stretch). Multi-language training (2-3 runs)
- Same E1 English generator; K=4-8 sibling target languages that differ in word order and case marking, with a language-ID token.
- Test whether role-based (not positional) mapping improves on passive / fronting / unseen structures in the *original* language.
- Full MLC-style in-context meta-learning is likely out of scope for the course timeline.

### Reporting
- For every run: IID, random-held-out structures, dev-challenge, sealed, out-of-scope refusal rate.
- 2+ seeds where the budget allows.
- State explicitly in the write-up:
  - which categories were declared in advance;
  - which were held out at random;
  - that the sealed set was never used for any decision.

---

## 13. What is not possible (be candid in the report)

- **No training set, however "representative", guarantees correct behaviour on structures absent from it** (NFL; Bogin et al.). A generator of English is itself a finite description. Real English will exceed it: ellipsis, relative clauses, idioms, questions, imperatives.
- **Extrapolating counts (k -> k+1 and beyond) stays fragile** even in the best published settings (Zhou et al. 2024; Anil et al. 2022). Decomposition is the reliable route.
- **A 0.6B model may simply lack strong synonym/paraphrase knowledge.** Preservation techniques cannot recover what the base model does not have. Measure this first (E0 probe).
- **Meta-learning moves the support boundary up one level** (to the distribution of grammars). It does not remove it (Woydt et al. critique).
- **The "pipeline" route changes the claim.** It shows the system works, not that the network learned the language.

---

## 14. Source list (all found during this search)

1. Lake & Baroni 2018, ICML. https://arxiv.org/abs/1711.00350
2. Kim & Linzen 2020, EMNLP (COGS). https://aclanthology.org/2020.emnlp-main.731
3. Keysers et al. 2020, ICLR (CFQ/MCD). https://research.google/pubs/measuring-compositional-generalization-a-comprehensive-method-on-realistic-data/
4. Hupkes, Dankers, Mul & Bruni 2020, JAIR. https://jair.org/index.php/jair/article/view/11674
5. Yao & Koller 2022, EMNLP. https://arxiv.org/abs/2210.13050
6. Bogin, Gupta & Berant 2022, EMNLP. https://aclanthology.org/2022.emnlp-main.175
7. Geirhos et al. 2020, Nature Machine Intelligence. https://arxiv.org/abs/2004.07780
8. Shah et al. 2020, NeurIPS. https://arxiv.org/abs/2006.07710
9. McCoy, Pavlick & Linzen 2019, ACL. https://aclanthology.org/P19-1334/
10. Csordás, Irie & Schmidhuber 2021, EMNLP. https://aclanthology.org/2021.emnlp-main.49
11. Oren, Herzig & Berant 2021, EMNLP. https://aclanthology.org/2021.emnlp-main.843
12. Yao & Koller 2024, NAACL. https://aclanthology.org/2024.naacl-long.25/
13. Patel, Bhattamishra, Blunsom & Goyal 2022, ACL. https://aclanthology.org/2022.acl-short.46
14. Zhou, Jiang & Bansal 2023, EMNLP. https://aclanthology.org/2023.emnlp-main.898
15. Zhang, Wang & Charton 2024, arXiv. https://arxiv.org/abs/2402.10891
16. Levy, Bogin & Berant 2023, ACL. https://arxiv.org/abs/2212.06800
17. Zhou et al. 2023, NeurIPS (LIMA). https://arxiv.org/abs/2305.11206
18. Furrer, van Zee, Scales & Schärli 2020, arXiv. https://arxiv.org/abs/2007.08970
19. Qiu et al. 2022, EMNLP (model scale). https://aclanthology.org/2022.emnlp-main.624
20. Shaw, Chang, Pasupat & Toutanova 2021, ACL (NQG-T5). https://arxiv.org/abs/2010.12725
21. Kumar et al. 2022, ICLR (LP-FT). https://arxiv.org/abs/2202.10054
22. Lee et al. 2023, ICLR (surgical FT). https://arxiv.org/abs/2210.11466
23. Biderman et al. 2024, TMLR. https://arxiv.org/abs/2405.09673
24. Shuttleworth, Andreas, Torralba & Sharma 2025, NeurIPS. https://arxiv.org/abs/2410.21228
25. Wortsman et al. 2022, CVPR (WiSE-FT). https://arxiv.org/abs/2109.01903
26. Hu et al. 2021 (LoRA). https://arxiv.org/abs/2106.09685
27. Li & Liang 2021, ACL (prefix-tuning). https://aclanthology.org/2021.acl-long.353
28. Kirkpatrick et al. 2017, PNAS (EWC). https://pmc.ncbi.nlm.nih.gov/articles/PMC5380101
29. Aghajanyan et al. 2021, ICLR (R3F). https://arxiv.org/abs/2008.03156
30. Yang et al. 2024, ACL (SDFT). https://aclanthology.org/2024.acl-long.58
31. Ibrahim et al. 2024, TMLR (replay). https://arxiv.org/abs/2403.08763
32. Hendrycks et al. 2020, ACL. https://aclanthology.org/2020.acl-main.244
33. Mosbach et al. 2023, Findings ACL. https://arxiv.org/abs/2305.16938
34. Gan et al. 2021, ACL (Spider-Syn). https://aclanthology.org/2021.acl-long.195
35. Wang, Berant & Liang 2015, ACL (Overnight). https://aclanthology.org/P15-1129/
36. Xu, Semnani, Campagna & Lam 2020, EMNLP (AutoQA). https://arxiv.org/abs/2010.04806
37. Herzig & Berant 2019, EMNLP. https://aclanthology.org/D19-1394/
38. Marzoev et al. 2020, arXiv. https://arxiv.org/abs/2004.13645
39. Sennrich, Haddow & Birch 2016, ACL. https://aclanthology.org/P16-1009/
40. Wang et al. 2023, ACL (Self-Instruct). https://aclanthology.org/2023.acl-long.754
41. Chang et al. 2023, ICLR (Dr.Spider). https://arxiv.org/abs/2301.08881
42. Diversity-oriented data augmentation with LLMs, 2025. https://arxiv.org/abs/2502.11671 ; LLM augmentation sample selection, 2025. https://arxiv.org/abs/2501.08276
43. Shin et al. 2021, EMNLP. https://arxiv.org/abs/2104.08768
44. Geng, Josifoski, Peyrard & West 2023, EMNLP. https://aclanthology.org/2023.emnlp-main.674
45. Drozdov et al. 2023, ICLR. https://arxiv.org/abs/2209.15003
46. Ranta, Angelov, Gruzitis & Kolachina 2020, Computational Linguistics. https://research.chalmers.se/en/publication/517996
47. Tanzer et al. 2024, ICLR (MTOB). https://arxiv.org/abs/2309.16575
48. Lake & Baroni 2023, Nature (MLC). https://www.cs.princeton.edu/node/5254
49. Lake 2019, NeurIPS (meta seq2seq). https://arxiv.org/abs/1906.05381
50. Conklin, Wang, Smith & Titov 2021, ACL. https://aclanthology.org/2021.acl-long.258
51. McCoy & Griffiths 2025, Nature Communications. https://ideas.repec.org/a/nat/natcom/v16y2025i1d10.1038_s41467-025-59957-y.html
52. Hu, Petty, Shi, Merrill & Linzen 2025, ACL. https://aclanthology.org/2025.acl-long.478/
53. Woydt et al. 2025/26, arXiv (MLC critique). https://arxiv.org/abs/2506.01820
54. Kallini et al. 2024, ACL. https://aclanthology.org/2024.acl-long.787
55. Andreas 2020, ACL (GECA). https://aclanthology.org/2020.acl-main.676
56. Akyürek, Akyürek & Andreas 2021, ICLR (R&R). https://arxiv.org/abs/2010.03706
57. Guo, Kim & Rush 2020, EMNLP (SeqMix). https://aclanthology.org/2020.emnlp-main.447
58. Yang, Zhang & Yang 2022, NAACL (SUBS). https://aclanthology.org/2022.naacl-main.12
59. Qiu et al. 2022, NAACL (CSL). https://aclanthology.org/2022.naacl-main.323
60. Akyürek & Andreas 2023, ACL (LexSym). https://aclanthology.org/2023.acl-long.38
61. Linzen 2020, ACL. https://aclanthology.org/2020.acl-main.465
62. Hupkes et al. 2023, Nature Machine Intelligence (taxonomy). https://arxiv.org/abs/2210.03050
63. Ribeiro et al. 2020, ACL (CheckList). https://arxiv.org/abs/2005.04118
64. Gardner et al. 2020, Findings EMNLP (contrast sets). https://arxiv.org/abs/2004.02709
65. Kiela et al. 2021, NAACL (Dynabench). https://aclanthology.org/2021.naacl-main.324
66. Wolpert 1996, Neural Computation (NFL). https://mlanthology.org/neco/1996/wolpert1996neco-lack
67. Anil et al. 2022, NeurIPS. https://arxiv.org/abs/2207.04901
68. Zhou et al. 2024, ICLR (RASP-L). https://arxiv.org/abs/2310.16028
69. Zhou, Alon, Chen, Wang, Agarwal & Zhou 2024 (length gen. not robust). https://arxiv.org/abs/2402.09371
70. Kazemnejad et al. 2023, NeurIPS. https://arxiv.org/abs/2305.19466
71. Newman, Hewitt, Liang & Manning 2020 (EOS decision). https://arxiv.org/abs/2010.07174
72. Nye et al. 2021/22 (scratchpads). https://arxiv.org/abs/2112.00114
73. Belinkov & Bisk 2018, ICLR. https://arxiv.org/abs/1711.02173
74. Karpukhin, Levy, Eisenstein & Ghazvininejad 2019, W-NUT. https://aclanthology.org/D19-5506/
75. Provilkov, Emelianenko & Voita 2020, ACL (BPE-dropout). https://aclanthology.org/2020.acl-main.170
76. Pruthi, Dhingra & Lipton 2019, ACL. https://aclanthology.org/P19-1561/
77. ReCOGS discussion of recursion/structural splits. https://arxiv.org/pdf/2303.13716
