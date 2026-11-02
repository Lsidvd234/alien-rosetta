# Orvessa: how far does a small LLM learn a language it has never seen?

Course project for *Foundations of Large Language Models* (AI60213), IIT Kharagpur. Group 14: R S Kavin Raj
(24IM10055).

- **Report (ICLR 2025 format):** [`submission/report/alien_rosetta.pdf`](submission/report/alien_rosetta.pdf)
  (LaTeX source in `submission/report/`)
- **Presentation:** [`submission/Orvessa_presentation.pptx`](submission/Orvessa_presentation.pptx)
- **Lab notebook with the predictions written before each run:** `results/findings_2026-10-24.md`
- **Every run and its scores:** `results/ALL_RUNS.md`

## What was built

- A generator that **invents a complete language** with an exact answer key: Tamil-like grammar (subject-object-verb,
  case and agreement endings, sound changes), 11,078 root words plus built words, and a script of 27 invented glyphs
  (Unicode Private Use Area) that no tokenizer has seen.
- **Qwen3-0.6B-Base + LoRA** (rank 16, all projections), each glyph added as one new token whose embedding starts at
  the embedding of its sound. The model first writes a meaning analysis (`S=NAME.Tom V=beat.FUT O=PRON.2sg`), then
  the glyphs, so every error splits into a misreading of the English or a mistake in the new language.
- Tests: generated sentences, 600 real Tatoeba sentences, and a sealed set of free English written by the author
  (kept off-line and used only as totals). 165 training runs on free Kaggle GPUs.

## Findings

| | Finding | Evidence | Strength |
|---|---|---|---|
| A | Generalisation stops at what training covers | held-out English structures 0.00-0.09; putting back only 50 passives lifts passive reading 0.00 -> 0.53 (0.66 on a second seed); a removed multiplication fact is not inferred (0.05 vs 0.92) | strong |
| B | A spelling rule transfers only when it covers the vocabulary | never-trained words 0.67 when every word follows the rule, vs new loanwords 0.19-0.27 when only a 4-12% minority does | moderate |
| C | Only English variety helps real English | 15% real text: Tatoeba meaning 0.62 -> 0.92 (2 seeds); focal loss, paraphrases, more epochs, word floor, rewriter, beam search, voting and a 3x larger model: null or worse | strong (Tatoeba), moderate (sealed) |
| D | On real English reading fails; with untrained words, both fail | reading errors 0.21 / 0.38 / 0.52 vs writing 0.24 / 0.31 / 0.26 (generated / Tatoeba / sealed); writer from the gold meaning 0.84 vs 0.20 when a word was never trained; probes and steering show most errors are already in the English representation | moderate |
| E | In maths a missing fact is rebuilt only from facts seen in the same role | building scratchpad with three mixed rules 0.77 vs one rule 0.08; showing the block in 5 training workings lifts it 0.00 -> 0.91; 3x4 and 6x6 rebuilt too (one run per condition) | moderate |

## Layout

| Folder | Contents |
|---|---|
| `conlang/` | the language generator: grammar, sound rules, scripts, vocabulary |
| `train/` | data building (`represent_v2.py`), LoRA training and glyph tokens (`run.py`), maths (`maths.py`), grokking (`grok.py`) |
| `analysis/` | scoring and analysis: error split, reader/writer bottleneck, loanwords and sound rule, probes and steering, significance tests |
| `kaggle_tools/` | runs training jobs on free Kaggle GPUs from a laptop: job files with the exact arguments of every run |
| `two_stage/`, `ceiling/` | the two-stage rewriter baseline and the base-model English ceiling |
| `scripts/` | data-building helpers and laptop launchers |
| `tests/` | unit tests (`python -m pytest tests -q`) |
| `results/` | result tables (`*.md`), figures and small JSON summaries |
| `docs/` | earlier results report, code guide, generator notes |
| `submission/` | the final report, presentation, figures and the scripts that build them |

See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for a longer tour.

**Not in this repository:** the sealed test set (never published or used to design training data) and the trained
model checkpoints.

## Running

```bash
pip install torch transformers peft accelerate wordfreq matplotlib
python -m pytest tests -q
# one run (kaggle_tools/jobs_*.json hold the exact arguments of every reported run), e.g. the baseline:
python train/run.py --preset tamil_like --script abugida_pua --add-tokens --model Qwen/Qwen3-0.6B-Base \
  --epochs 4 --batch 16 --grad-accum 2 --new-tok-init latin --rep --rep-v2 --cot --rep-n 80000 \
  --rep-refuse 2500 --rep-replay 6000 --rep-held-frac 0 --rep-min-count 12 --max-batch-tokens 1600 --seed 0
```

## Credits

- Real English sentences: [Tatoeba](https://tatoeba.org), CC-BY 2.0 FR. WikiText-2: CC-BY-SA 3.0.
- Word lists built with WordNet, wordfreq and lemminflect.
- Presentation images: Tamil alphabet chart (public domain)
  and decoder-only transformer diagram by Marxav (CC BY-SA 4.0), all from Wikimedia Commons; Figure 1 of Hu et al.
  2021 (LoRA) and of Turner et al. 2023 (activation addition).
