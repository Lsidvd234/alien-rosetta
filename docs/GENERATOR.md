# New-language lab: generator (v1)

**Vocabulary:** 4,320 English content words (`conlang/data/vocab_en.json`; rebuild with
`scripts/build_vocab.py`, which needs the `.venv-vocab` env and `NLTK_DATA` on D:). `Vocab.small()`
(48 words) is only for fast tests. **Data:** 50,000-sentence training pool (runs use the first
5k/20k/50k) + 500 per test split; regenerated from the seed in ~6 s, so nothing needs uploading.

Invents languages a model has never seen, in two families, and writes the same
language in four scripts. Every example has an exact answer key.

| Family | Grammar | Sounds | Default script |
|---|---|---|---|
| **easy** (English-like) | SVO, articles, few endings (plural, past, 3sg), helper words for future/negation | short vowels, simple consonants | latin |
| **hard** (Tamil-like) | SOV, no articles, case endings (object, location), every tense an ending, full person/number/gender agreement, negation ending, sound changes where endings join | long vowels, retroflex consonants, longer roots | abugida_tamil |

**Dials** (`conlang/grammar.py`): `word_order`, `adj_order`, `articles`, `acc_marking`, `loc_marking`,
`plural`, `tense`, `agreement`, `negation`, `sandhi`, `phonology`, `irregular_rate`.

**Scripts** (`conlang/script.py`): `latin`, `latin_accent` (stress/long-vowel/retroflex accents),
`abugida_tamil` (real Tamil letters), `abugida_pua` (invented glyphs no tokenizer knows).
With the same lexicon seed, changing the script gives the **same language** written differently.

**Test splits** (`conlang/dataset.py`): `test_iid`, `test_comp` (unseen verb+object pairs),
`test_morph` (past + negated + plural subject, never in training), `test_long`.

## Commands
```bash
python -m pytest tests -q                                         # 16 tests
python -m conlang.dataset --preset english_like --script latin --n-train 50000 --out data/easy_latin
python -m conlang.dataset --preset tamil_like --script abugida_tamil --out data/hard_tamil
python -m conlang.dataset --preset english_like --set word_order=SOV --script latin_accent --out data/x
python -m conlang.dataset --list-variants                         # base + one-dial flips + all_flipped
python -m conlang.features --tokenizers Qwen/Qwen2.5-1.5B-Instruct   # token cost per language x script
```
On Windows set `PYTHONUTF8=1` so non-Latin output prints. Keep Hugging Face caches on D:
(`HF_HOME`), and use `HF_HUB_DISABLE_XET=1` if downloads stall.

## Files
- `conlang/lexicon.py`: sound inventories, seeded roots and endings
- `conlang/frames.py`: meanings (sentence frames), concept lists, English rendering
- `conlang/grammar.py`: dials, presets, invented-language rendering, sound changes
- `conlang/script.py`: the four writing systems
- `conlang/dataset.py`: builds train/test splits (CLI)
- `conlang/evaluate.py`: exact match, chrF, word accuracy
- `conlang/features.py`: difficulty features measured before training (CLI)
- `results/features_v0.json`: first measurements from v0 (old 82-word vocabulary; superseded)
- `scripts/build_vocab.py`: builds the 4,320-word vocabulary file
