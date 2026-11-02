# Alien Rosetta: teaching a small language model a language it has never seen

Course project, AI60213 Foundation of Large Language Models (IIT Kharagpur), Kavinraj (24IM10055), October-November
2026. This file explains the whole project for a reader who has never seen it: the question, what was built, how it
is tested, every claim with its evidence and its weaknesses, what is still running, and where everything is.
The full write-up with every table is `../docs/Alien_Rosetta_project_report.pdf` (~35 pages); the code is explained
file by file in `../docs/Alien_Rosetta_code_guide.pdf`. Raw results and pre-registered predictions are in
`results/findings_2026-10-24.md` (a running lab notebook, newest entries at the end).

---

## 1. The question in one paragraph

What does it take for a small pretrained language model to learn a language it has never seen, and where does that
learning stop? Real low-resource languages cannot answer this cleanly: the model may have seen fragments in
pre-training, and there is no exact answer key. So the project **invents** a language from scratch, with an exact
grammar, its own vocabulary and its own script of glyphs no tokenizer has ever seen, and teaches Qwen3-0.6B-Base to
translate English into it. Because every sentence has a provably correct translation and a provably correct
"meaning" (who did what, tense, negation, ...), every failure can be located: did the model misunderstand the
English, or did it understand and write the new language wrong? A second, smaller domain (arithmetic in an invented
base-7 notation) checks that the findings are not specific to translation.

## 2. What was built

**The language (`conlang/`, v2 since 13 Oct).** A generator with grammar "dials". The language used throughout is
Tamil-like: subject-object-verb order, case endings, every tense/person/number an ending, sound changes where pieces
join, 11,078 root words plus words built from pieces (help -> help>ACT "helping"), compounds and loanwords. It
is written in **27 invented glyphs in the Unicode Private Use Area** (`abugida_pua`): no tokenizer has them, so they
are added as new tokens. Every glyph's sound is known.

**The meaning line.** Each example is English -> `analysis => glyphs`, where the analysis is a compact meaning
(`S=PRON.3sgf V=adjust.PAST O=PRON.3sgf^dress>ACT`). The generator produces both from one meaning, so both are exact.
Main runs write the meaning first ("CoT" runs), so errors can be split into reading errors (meaning wrong) and
writing errors (meaning right, glyphs wrong).

**Training data (`train/represent_v2.py`).** 80,000 generated English sentences with all surface variety the
generator knows (questions, commands, passives, clefts, ...), 2,500 refusal examples (non-English input), 6,000
lines of plain English replay text against forgetting; a word floor (`--rep-min-count 12`: every word seen >= 12
times). Optional switches add real Tatoeba sentences with reader-made labels, hold out structures, add loanwords,
add common English the generator left out, and so on. Every switch is off by default and was checked to leave the
default data byte-identical.

**Models and training (`train/run.py`, `kaggle_tools/`).** Qwen3-0.6B-Base (one run with Qwen3-1.7B-Base planned),
LoRA rank 16 on all projections, lr 2e-4, 4 epochs, effective batch 32; new glyph tokens get their own trainable
embeddings, initialised at the embedding of the Latin letter for their sound. Trained on free Kaggle T4 GPUs (~6 GPU-h
per translation run, ~0.5-1 h per maths run), launched from the laptop through the Kaggle API on three accounts. 123
run folders exist in `results/kaggle/`.

**The maths domain (`train/maths.py`).** Seven new digit glyphs (base 7) and new operator glyphs; tasks: decimal <->
alien numeral, addition, multiplication, trained on short numbers and tested on longer ones. Options: what the
glyphs start from (meaning / shuffled meaning / no meaning), removing single multiplication facts from training, and
several "scratchpad" formats that write out the long multiplication.

## 3. How it is tested

| Test | What | Why it is trustworthy / its weakness |
|---|---|---|
| Generated test sets | 300 sentences per family, never in training; in-distribution mixes, each kind of utterance, held-out structures | exact answers; but same generator as training |
| Held-out structures | e.g. passives, clefts, name subjects never seen in training | exact; tests compositional generalisation |
| **Tatoeba 600** | 600 real, everyday English sentences, held out | real English; gold labels come from an automatic rule-based reader, and a blind hand-check found only ~0.6 of them right |
| **Sealed set** | 147 in-scope sentences of free English written by the student, frozen before it influenced any design; scored on the laptop only, totals only (no per-sentence inspection, by rule) | the fairest test; small (147), so only differences > ~0.10 show without pairing |
| Statistics | paired McNemar tests on the same sentences, bootstrap intervals, Wilson intervals; two training seeds for the key runs | |
| Pre-registration | every new run or analysis since ~24 Oct has its prediction and decision rule written in `results/findings_2026-10-24.md` before it runs; failed predictions are reported as failed | some predictions were amended after a flawed pilot; each amendment is dated and labelled |

## 4. The claims, their evidence and how strong they are

Ratings are the project's own (strong / moderate / weak), with the main reason. Numbers are from result files.

**Claim 1. A new symbol is learned best when its embedding starts from what it stands for.** Strong (language),
moderate-strong (maths).
- Language (v1, single seed): glyph tokens initialised at their Latin sound reach 0.99/0.99/0.97/0.97 on the four
  generated test splits vs 0.57/0.51/0.42/0.00 when initialised at an unrelated token. A separate control that starts
  every glyph at a WRONG sound (one permutation, one seed) reaches 0.98/0.98/0.86/0.65: a distinct letter-like start
  gives the in-distribution gain, the right sound adds generalisation (morphology +0.11, long sentences +0.32).
- Maths: digits initialised at the real digits add 4+4 (longer than trained) at 0.86 (two-seed mean) vs 0.09 for a
  neutral start. The shuffled (wrong-digit) start varied 0.08 vs 0.60 between seeds; a swap test (30 Oct) showed this
  is caused by the starting permutation, not the data: seed-0 data with the near-right permutation 0.58, seed-1 data
  with the scrambled one 0.05. A near-miss start behaves almost like the right one.

**Claim 2. Generalisation stops at what training covers.** Strong as description; strong causal evidence for one
structure and one domain (single seeds).
- Held-out sentence structures score 0.00-0.09 while trained ones are ~0.97 (zero coverage fails; a small dose
  is enough, see below).
- **Structure dose (28-29 Oct):** only plain passives held out, then 0 / 50 / 300 put back (of ~88k examples):
  passive meaning 0.00 / 0.53 / 0.75; the rest of the language unchanged; real English unchanged.
- Maths: removing 5x6 from training drops problems that need it to 0.05 while others stay at 0.84-0.92 even when
  memorised swapped twins are excluded (twin-split runs, 29 Oct; also 3x4 and 2x2).

**Claim 3. Rare words are a sightings problem.** Moderate.
- A word floor of 25 (vs 12) lifts generated-sentence accuracy 0.61/0.63 -> 0.72 (significant) but not real English;
  caveat: that run also had 20% more data (a size-matched control, n98k, is queued).
- Loanwords (English nouns respelled by fixed rules): new loans 0.19 vs seen 0.51; letter-by-letter loans 0.23 vs
  multi-letter-rule loans 0.12 (p = 0.005): part rule, mostly list; half of new loans are not even recognised as loans.

**Claim 4. English variety is the biggest gap; real text closes much of it.** Strong on Tatoeba, moderate on sealed.
- 15% real Tatoeba sentences: Tatoeba meaning 0.62 -> 0.92 (both seeds, p < 1e-48); sealed 0.47 -> 0.56, significant
  on both seeds by a paired test (McNemar p = 0.036 / 0.013) but not after a Holm correction across the 21
  claim-level tests (`results/holm.md`); the paired test was chosen after the unpaired one failed, and the sealed
  gold is partly the same reader's output, so the reader-free everyday run is the cleaner sealed evidence. An earlier "significant" from pooling seeds as
  independent was withdrawn (code review, 28 Oct).
- Hand-check: only about +0.19 (range +0.15 to +0.23) of the +0.31 Tatoeba gain rests on correct gold labels.
- **Word coverage without real text (everyday run, 29 Oct):** Tatoeba 0.73 (significant vs baseline), sealed 0.52:
  about half of real text's sealed gain, not distinguishable from it on 147 sentences; partly tailored to Tatoeba.
- The remaining sealed gap is multi-clause sentences (baseline 0.61 single-clause vs 0.17 multi-clause).

**Claim 5. Failures move from writing to reading as the English departs from training.** Moderate as worded now
(re-stated 30 Oct after the external review). The first evidence (share of failures that are misreadings) rises by
construction; it is now measured by judging writing on the model's OWN meaning, right or wrong: on familiar generated
sentences writing and reading errors are about equal (baseline 0.24 vs 0.21), on Tatoeba reading grows faster (0.31
vs 0.38), sealed being added (labels only). Older figures below are kept for reference. (Replaces the original claim 5, "meaning first speeds learning", which held in v1 but did not
replicate in v2; both are reported.) Added after the data.
- Share of failures that are misreadings: 0.25-0.59 generated -> 0.64-0.77 Tatoeba -> 0.88-0.98 sealed -> 0.98-0.99
  held-out structures, the same order in all 10 models.
- Passives (28 Oct): with no passives in training, 100 of 100 failures are reading failures (pre-registered R2,
  holds); a linear probe cannot read the doer's number from the model's states better than a bag-of-words baseline
  (0.746 vs 0.704), but the untrained base model also sits at 0.704, so without a positive control (running on
  models that read passives) R4 is inconclusive, not negative. The model
  reads the patient correctly and invents a phrase field `BY=` for "by Y".
- A plain-English question test of whether the model "forgot" passives was tried twice and judged invalid for 0.6B
  models (even models that read passives correctly fail it); it is not used.
- Pending: a meaning-only model and a separate meaning->glyph writer (reader vs writer bottleneck).

**Claim 6. Model tricks are not the fix.** Moderate. Focal loss, a two-stage English rewriter, paraphrased training
English, 6 epochs, beam search, majority voting: none improves real English; focal loss and 6 epochs make Tatoeba
significantly worse. Pending: model size (Qwen3-1.7B).

**Claim 7. The model uses English meaning directions to write the new language** (added after the data).
Moderate (re-rated 30 Oct): the steering setting was picked on the same sentences it is reported on (a held-out re-run
is running), the subject "repairs" push toward the gold value (oracle), and a causal tense direction is expected in
any translator; steering vectors are a known method. Probes read tense, negation, question, aspect, modality, subject at 0.90-1.00; steering along
mean-difference directions (between the mean hidden states of the two values; not the probes' own weights) flips the
output (tense 0.97-1.00 except one direction of one model at 0.76, negation 0.91-0.99, vs <= 0.02 for random
directions) on two seeds of two models, and pushing along the subject direction repairs real subject errors (11/12, 10/10).

**Candidate claim 8 (maths, 29-30 Oct; gated on replicates). A fact the model never saw can be computed, but only by
recombining facts it has already used in the same role.** With 5x6 and 6x5 removed:

| scratchpad | 5x6 steps right | problems needing 5x6 |
|---|---|---|
| facts recalled | 0.49 (noisy guessing; voting does not help: 0.27 vs 0.26) | 0.26 |
| facts built by one rule (5x6 = 5x4 + 5 + 5) | 0.07: the procedure is followed but the block 5x4 is recalled wrong, because 5x4 was never used as a block | 0.08 |
| facts built by three rules mixed | 0.83: built from blocks seen in that role are right 99.6%, from blocks never seen in that role 17% | 0.77 |
| recall, only the order 6x5 removed (5x6 kept) | 0.99 (6x5 answered from 5x6 without being taught to swap) | 0.72 |

Caveat (review): the model chooses its own route, so "seen block" is confounded with its confidence; Saturday's
mixed-rule runs also force every route on every problem (`--force-routes`). The language side shows the same pattern (passives: "by Y" never seen in the doer role is not mapped to the doer).
Queued to decide it: seed replicate, a second pair (3x4), the largest fact (6x6, only one familiar route), removal of
the whole 6x4/6x5/6x6 corner under five ways of writing facts (recall, mixed rules, pure addition, a base-7 "x6 =
x0 - x" trick, splitting, doubling), and a **role dose curve** (5x4 shown as a block 0/5/25/100/400 times; the
prediction is a rise like the passive dose, with 6x5 as an untouched control). All predictions are written.

## 5. Corrections, failed predictions and withdrawn results (honesty log)

- Sealed claim-4 significance from pooling two seeds as independent: withdrawn, replaced by a paired test.
- Hand-check summary sentence was hard-coded and the control group unrepresentative: fixed, with a Fisher test.
- Loanword grouping missed some multi-letter rules (22% misfiled): re-scored; the conclusion changed (L2 now holds).
- R4 pilot design leaked the answer through English agreement: redesigned with baselines, rule amended (dated).
- Failed as written: R2 second half (writing failures 0.22 vs <= 0.03), R3 (writing change 0.032 vs < 0.03), S2/S3
  (scratchpad consistency), RB1 (one-rule building), RB3 (voting), the swap control (predicted <= 0.40, got 0.72),
  original claim 5 in v2. The plain-English QA test (two versions) judged invalid.
- Reviewer-found bugs fixed before launch: q17 results saved only at the end; a smoke-test name collision that
  would have skipped the real run; scorers crashing on epoch checkpoints; a test-set cap that would have removed all
  5x6 problems from training.

## 6. Limitations and threats to validity

- One base model size so far (0.6B); one invented language family (Tamil-like); one script.
- Most runs are single seed; seed spread on generated sentences is ~0.02, on the sealed set up to ~0.04.
- Tatoeba gold comes from an automatic reader that is right only ~60% of the time by hand check; Tatoeba numbers
  measure agreement with that reader. The sealed set (147) is small.
- Several claims (5, 7, candidate 8) were formed after seeing data; their later tests are pre-registered.
- The generator's English is synthetic and sometimes semantically odd ("the kremlin is procured by the fruit").
- Probes are linear; "not decodable" may mean "not linearly decodable".
- Compute: free Kaggle quota (~30 GPU-h per account per week), so no large sweeps or many seeds.

## 7. What is running / planned (Saturday 31 Oct, three Kaggle accounts)

`LAUNCH_NEXT_WEEK.md` (end) has the exact job lists `kaggle_tools/jobs_{main,second,third}_1031.json`:
size-matched word-floor control (claim 3), sound-rule language (every word = its English word respelled: does a
rule-built vocabulary remove the vocabulary limit?), Qwen3-1.7B (claims 5/6), loanword rate x3 (claim 3), meaning-only
reader and meaning->glyph writer (claim 5), and the maths runs for candidate claim 8. Then laptop scoring
(Tatoeba, sealed, error splits) and the report update. Deadline 13 Nov.

## 8. Where things are

| Path | Contents |
|---|---|
| `conlang/` | the language generator (v1 `grammar.py`, `lexicon.py`, `frames.py`, `script.py`; v2 `v2.py`); old v1 README in `conlang/docs/` |
| `train/` | `run.py` (translation training/eval, all switches), `represent_v2.py` (v2 data), `maths.py` (maths domain), `load.py` |
| `analysis/` | every scorer and analysis (`score_tatoeba.py`, `score_sealed.py`, `significance.py`, `sealed_paired.py`, `error_split.py`, `inside_*.py` probes/steering, `loan_words.py`, `buildmix_blocks.py`, `bottleneck.py`, ...) |
| `kaggle_tools/` | `kaggle_jobs.py` (upload code, push, queue, fetch) and every job list `jobs_*.json` |
| `results/` | `findings_2026-10-24.md` (lab notebook + predictions), `kaggle/<run>/` (each run's results and log), score tables |
| `sealed/` | the sealed test set and its writer app; **never leaves the laptop, never used for design** |
| `scripts/` | data builders (vocabulary, Tatoeba split, paraphrases); `scripts/laptop/` one-off laptop job scripts |
| `two_stage/`, `ceiling/`, `tests/` | the rewriter pipeline (claim 6), a ceiling estimate, unit tests |
| `../docs/` | project report, code guide, results report (PDF + LaTeX) |
| `../PLAN.md` | the project charter (research question, claims, scope) and history |

## 9. Reproducing

Python 3.11; `pip install torch transformers peft scikit-learn scipy`. Set `PYTHONUTF8=1` on Windows. A translation
run: `python -m train.run --preset tamil_like --script abugida_pua --add-tokens --rep --rep-v2 --cot ...` (exact
arguments of every run are in its `results/kaggle/<run>/model/run_args.json` and in the `kaggle_tools/jobs_*.json`
files). A maths run: `python train/maths.py --init meaning --scratch-build-mix --drop-pair 5,6`. Scoring:
`python analysis/score_tatoeba.py RUN`, `python analysis/significance.py`. Unit tests: `python -m pytest tests -q`.
