# Extension: the sound-rule language (design fixed 2026-10-27; code built and checked 2026-10-28)

**Status (user, 2026-10-27):** an extension, built and run ONLY if Kaggle compute is left after every run needed for
the main language (LAUNCH_NEXT_WEEK.md). Not part of the six claims; if run, it is reported as an extension of
claim 3 (vocabulary). No code exists yet.

## The question

In the main language (v2) every content word is invented: *dog* might be a random root, so the model must memorise
~11.6k words one by one, and a word outside the list cannot be written at all ('word not in the language': 37-42% of
the baseline's wrong sealed analyses, 36% on Tatoeba). The loanword runs showed the model can partly learn a
spelling rule for a small set of borrowed nouns (new loans 0.18 vs seen 0.48 at rate 0.04).

**Does a language whose words follow one sound rule remove the vocabulary limit?** With the same grammar, does the
model write words it never saw in training, and how much of "learning a new language" is the vocabulary list?

## How the language works

Everything is the main language (v2, `tamil_like`, script `abugida_pua`, seed 0) except where a word's sounds come
from:

| Part | Main language (v2) | Sound-rule language |
|---|---|---|
| Content words (nouns, verbs, adjectives, adverbs, places: v2 roots N/V/A/R/L, 11,561) | invented random root per word | the English word respelled by the loanword rules: `loan_sounds(word)` in conlang/v2.py, e.g. dog -> t-o-k, run -> r-u-n, happy -> a-p-y, church -> c-u-r-c |
| Built words (`help>ACT`, `kind>LESS`, ...) | invented root + derivation ending | rule-spelled base + the SAME v2 derivation ending |
| Grammar: word order, case endings, tense/negation/question/aspect/modal endings, plural, sound changes where pieces join | v2 | unchanged |
| Pronouns, determiners, connectors, interjections, question words (v2 roots D/C/I/W/PRON, ~75) | invented | unchanged (they are grammar, not vocabulary) |
| Names | spelled by sound (`spell_name`) | unchanged |
| LOAN ending | marks a borrowed noun | not used (every word is rule-built; loans would be identical to ordinary words) |
| Meaning line (analysis) | English keys, v2 notation | unchanged, so meaning scores compare directly with v2 |
| Glyph starting points | each glyph = copy of its Latin sound token | unchanged |

**The rule** (fixed, from the letters, not the pronunciation): letters the language lacks become the nearest sound
(b->p, d->t, f->p, g->k, s/z->c, j->c, q->k, w->v, x->k-c, h dropped); multi-letter rules first (tch/ch/sh->c,
th->t, ph->p, ck->k, qu->k-v, gh dropped, wh->v, ea->ii, ou/ow->a-v, oa->oo, ai/ay->ee); doubled consonants once.
Same function as the loanwords, so the loan results carry over as a reference.

**Same sound, different word.** 1,772 of the 11,561 content roots (15%) share their sound form with another root
(father / batter -> p-a-t-e-r; son / con -> c-o-n; nurse as noun and verb). Allowed, as in real languages (English
has homophones too): the task is English -> language, which stays one answer per sentence; only the reverse
direction would be ambiguous. Measured 2026-10-27 over the v2 vocabulary.

## Run

- Name `e2b-v2fix-soundrule`; the baseline's settings (80k examples, word floor 12, 4 epochs, analysis first,
  Qwen3-0.6B-Base, LoRA + glyph tokens); ~6.1 GPU-h on a T4.
- **New-word test (the key test):** 20% of content roots (seeded) never appear in training; test sentences that use
  them. In v2 an unseen root cannot be guessed (it is random); here a rule-learner writes them.
- Also: the usual generated test, Tatoeba 600 and the sealed set (glyph targets re-rendered from the gold analyses by
  the sound-rule language, locally; the sealed set never leaves the laptop, totals only).
- Compare with the v2 baseline (`e2b-v2fix-cot-80k-noheld`, both seeds) and with the loanword runs.

## To build (only when the compute decision is made)

1. `--rep-sound-rule` switch in train/represent_v2.py (off by default: default training data must stay
   byte-identical, as for every earlier switch), plus the held-out-roots option.
2. Glyph renderer for the rule-spelled roots (reuse `loan_sounds` + the script).
3. CPU smoke test on tiny-qwen3; reader round trip; count of training examples per held-out root = 0.
4. Pre-registered predictions in results/findings before launch (e.g. new-word glyph accuracy, meaning on Tatoeba
   vs the baseline).
5. Job file `kaggle_tools/jobs_soundrule.json`; code dataset re-upload.

Estimated build: half a day of code and checks; then one ~6 GPU-h run.

## Built 2026-10-28 (user: "start coding the sound-language")
- `conlang/v2.py` `SOUND_RULE` switch: content roots (roles N/V/A/R/L, not V:be) = `loan_sounds(word)`, set after every
  v2 draw, so endings, function words and names are v2's exactly; irregular pasts dropped (deviation, see findings).
- `train/represent_v2.py`: `set_sound_rule(on, held_frac)`, `SOUND["held"]` (2,232 words at 0.2, seed
  HELD_OUT_SEED+77), generators skip held words (pools filtered, item() rejects any other way in), build-time
  assertion of zero training sightings, eval set "sound rule: new words (never in training)" (3 x rep-eval-per),
  `language_of_run(cfg)` for laptop rebuilds, `render_targets` (sound-rule golds for Tatoeba / sealed from the gold
  meanings, matching each gold's v2 rendering first; 600/600 Tatoeba). Real text and loans refused with the switch on.
- `train/run.py`: `--rep-sound-rule`, `--rep-sound-held`; the new-word set keeps every item and prediction.
- Scorers: score_tatoeba, score_sealed (re-rendered glyph golds), freq_weighted_eval, subgraph_check, significance,
  error_split (language from run_args); `analysis/sound_rule_words.py` (per-word new vs seen, recognition, near misses).
- Checks (CPU): switch off = byte-identical training data and test sets (3 configurations + eval sets, same hash as
  the code before); switch on: 0 of 6,801 training examples contain a held word; the standard tests contain none;
  switching off restores v2's roots exactly; re-rendering from meaning fails only where v2 also fails; tiny-qwen3 run
  end to end (data, 3 steps, every eval set, result file with the new-word items); sound_rule_words.py gives 1.00 on
  gold answers.
- Job: `kaggle_tools/jobs_soundrule.json` (= the baseline's args + `--rep-sound-rule --rep-sound-held 0.2`), ~6.1 GPU-h.
  Predictions SR1-SR4 in results/findings_2026-10-24.md. Needs the code dataset re-uploaded on the account that runs it.
