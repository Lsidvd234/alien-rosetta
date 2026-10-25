# Alien Rosetta: teaching a small LLM a language it has never seen

Course project for *Foundations of Large Language Models* (AI60213), IIT Kharagpur, by R S Kavin Raj.

**Question:** what does it take for a small LLM to learn a language it has never seen, and where does that
learning stop?

**Answer the results support:** a small model learns a new language well once its new symbols start from the right
meaning; beyond that it generalises only as far as its training data covers (in vocabulary, sentence structure and
the variety of English it sees), and the remaining gaps are closed by data design rather than model tricks.

## Setup

- An **invented language** with a rule engine that gives an exact answer key for every sentence: about 11k root words,
  Tamil-like grammar (SOV word order, stacked endings, agreement) and a new glyph script no tokenizer has seen.
- **Model:** Qwen3-0.6B-Base with LoRA; the new glyphs are added as new tokens.
- **Data:** 80k generated English -> language examples, optionally with a meaning-first analysis step
  (`S=dog V=see.PRES O=cat => <glyphs>`), plus real English from Tatoeba for the English-variety runs.
- **Compute:** 40+ LoRA runs on free Kaggle T4/P100 GPUs (two accounts) and RunPod.

## Main results

| Claim | Evidence |
|---|---|
| New symbols are learnable when they start from meaning | maths in glyph notation: digits started at real digit embeddings solve 4+4 addition **86%** vs **34%** (shuffled start) and **9%** (no meaning), means over 2 seeds; glyphs split into byte pieces collapse to ~0% word accuracy |
| Generalisation stops at training coverage of sentence structure | ~97% exact on trained structures, **0-9%** on held-out ones (passives, clefts, fronted objects) |
| ... of vocabulary | rare words fail below roughly 10 sightings in training; a minimum-count data fix (every word seen at least 12 times) targets this |
| ... of English variety | adding 12k real Tatoeba sentences lifts meaning accuracy on 600 held-out real Tatoeba sentences from **62%** to **92%** |
| Writing the meaning first speeds learning, not generalisation | 96% vs 26% at 3 epochs, but both reach ~97% at 5 epochs |
| Model tricks are not the fix | focal loss, paraphrased training data and a two-stage rewriter give no gain on real English |
| Side study: grokking | no grokking within 60k steps (maths) or 12k steps (translation) at weight decay 1.0 with a frozen base |

Tables and figures are in `results/`; the full write-up is `docs/Alien_Rosetta_results_report.pdf`, and
`docs/Alien_Rosetta_code_guide.pdf` walks through the code.

**Not in this repository:** the sealed test set (a set of real sentences written by hand and kept as a final exam,
so it is never published or used to design training data) and the trained model checkpoints (5.6 GB).

## Layout

| Folder | Contents |
|---|---|
| `conlang/` | the language generator: grammar dials, sound rules, scripts, vocabulary |
| `train/` | data building (`represent_v2.py`), LoRA training (`run.py`), maths (`maths.py`), grokking (`grok.py`) |
| `analysis/` | scoring and analysis scripts (frequency-weighted accuracy, subgraph matching, Tatoeba, epoch curves) |
| `kaggle_tools/` | runs training jobs on Kaggle from a laptop: queue, fetch, chunked grokking chains |
| `two_stage/`, `ceiling/` | the two-stage rewriter baseline and the base-model English ceiling |
| `tests/` | unit tests (`python -m pytest tests -q`) |
| `results/` | result tables (`*.md`), figures and small JSON summaries |
| `docs/` | report, code guide, and the original generator notes (`GENERATOR.md`) |

## Running

```bash
pip install torch transformers peft accelerate wordfreq matplotlib
python -m pytest tests -q
# train one run (see kaggle_tools/jobs_*.json for the exact arguments of every reported run)
python train/run.py --preset tamil_like --script abugida_pua --add-tokens --rep --rep-v2 --cot --epochs 4 ...
```

## Data credits

- Real English sentences: [Tatoeba](https://tatoeba.org), CC-BY 2.0 FR.
- WikiText-2 (used to check that English is not forgotten): CC-BY-SA 3.0.
- Word lists built with WordNet, wordfreq and lemminflect.
