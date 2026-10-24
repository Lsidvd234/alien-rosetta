# How different is the paraphrase run's training English? (`analysis/para_diff.py`)

The para run trains on the baseline's examples with **1,106 of about 80,000 (1.4%)** given reworded English (same meaning, checked by the reader). Below: how much those examples changed, and how much NEW English they add.

## 1. How much the changed examples changed

- Word-level edit distance (share of the sentence's words moved / added / removed / replaced): median **0.50**, mean 0.59; 78% of them changed a quarter of their words or more.
- New sentence patterns: 80% of paraphrases have a function-word pattern absent from the baseline's training data, but so do 80% of FRESH generated sentences (the generator is very varied), so this alone shows nothing. For short sentences (<= 8 words): paraphrases 65% vs fresh generated 50%.
- They contain **67** function-word pairs the baseline never trained on (e.g. ''d be', ''d he', ''d her', ''ll 's', ''ll against', ''ll not', ''ll what', ''m by', ''re by', ''re not').


## 2. Statistical test: is the paraphrased English a different kind of English?

A logistic-regression classifier sees only the FUNCTION words of a sentence (1-3-word sequences of the, by, did, n't, there ...; content words, capitals and punctuation removed) and must tell two groups apart; 5-fold cross-validated accuracy, 50% = chance.

| groups | sentences per group | accuracy | p (vs chance) |
|---|---|---|---|
| paraphrases vs their own originals | 1106 | **0.61** | 2.1e-26 |
| paraphrases vs baseline training sentences | 1106 | **0.63** | 2.8e-35 |
| their originals vs baseline training sentences (control) | 1106 | **0.57** | 1.2e-12 |
| baseline vs baseline, two random halves (control) | 1106 | **0.49** | 8.8e-01 |

## 3. Constructions: how often each appears

| construction | baseline training | paraphrases | their originals |
|---|---|---|---|
| passive with 'by' | 2.8% | 6.5% | 2.7% |
| contraction (n't 'll 're 've 'd 'm) | 14.7% | 22.3% | 11.8% |
| phrase moved to the front (... ,) | 3.7% | 24.2% | 3.8% |
| 'there is / are' | 1.5% | 1.3% | 4.2% |
| 'it was ... that' | 0.6% | 0.6% | 0.8% |
| 'by who / by what' | 0.0% | 1.6% | 0.0% |

## 4. Kinds of change in the paraphrased examples

| kind of change | paraphrases | share |
|---|---|---|
| content words reordered | 432 | 39% |
| small words / punctuation only | 412 | 37% |
| phrase moved to the front | 252 | 23% |
| word changed (same meaning) | 202 | 18% |
| contraction | 134 | 12% |
| passive (by ...) | 70 | 6% |
| active (from passive) | 17 | 2% |
| there is / are | 4 | 0% |
| it was ... that | 2 | 0% |

(one paraphrase can have several kinds)

## 5. But the training set as a whole changed little

- Examples changed: 1,106 of ~80,000 = **1.4%**.
- New sentence patterns added: **870** distinct skeletons, against 70,641 distinct skeletons in the baseline's training data (+1.2%); each new pattern appears ~1.0 times, well under the ~10 sightings a word needs (claim 3).
- So the para run saw clearly different English, but too little of it: this is why the result is inconclusive rather than negative.

## Examples (largest changes)

- Remark  
  → **Remark...**  (small words / punctuation only; edit 3.00)
- What can not know these parishes  
  → **These parishes can't be known by what.**  (passive (by ...), word changed (same meaning); edit 2.00; NEW pattern)
- Who was rapping them  
  → **They were being rapped by who.**  (passive (by ...), word changed (same meaning); edit 2.00; NEW pattern)
- What informs us  
  → **We're informed by what.**  (contraction, passive (by ...), word changed (same meaning); edit 2.00; NEW pattern)
- Who deciphers Laaliiyaa  
  → **Laaliiyaa is deciphered by who?**  (passive (by ...), word changed (same meaning); edit 2.00; NEW pattern)
- That academia had not financed him  
  → **He hadn't been financed by that academia.**  (passive (by ...), word changed (same meaning); edit 1.83; NEW pattern)
- Who buys this sweat  
  → **This sweat is bought by who.**  (passive (by ...), word changed (same meaning); edit 1.75; NEW pattern)
- What had not ornamented this urban innovation  
  → **This urban innovation hadn't been ornamented by what.**  (passive (by ...), word changed (same meaning); edit 1.71; NEW pattern)
- Kuuvun will not conform a lot  
  → **A lot Kuuvun won't conform.**  (word changed (same meaning); edit 1.67; NEW pattern)
- You all should be rallying Rakinii  
  → **Rakinii should be being rallied by you all.**  (passive (by ...), word changed (same meaning); edit 1.67; NEW pattern)
- Who will suck them.  
  → **They will be sucked by who.**  (passive (by ...), word changed (same meaning); edit 1.60; NEW pattern)
- What has massaged Piyir!  
  → **Piyir has been massaged by what!**  (passive (by ...), content words reordered; edit 1.60; NEW pattern)
