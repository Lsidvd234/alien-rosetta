# Inside the network: are glyphs put together into words? (`analysis/hidden_word_assembly.py`)

25 training sentences, each with one of 5 frequent content stems (5 per stem), fed with an EMPTY English line, so word identity can only come from the glyphs. CPU, fp32.

## 1. Can the stem be read from the word's LAST glyph? (linear probe, 5 classes, chance 0.200)

Layer 0 = the last glyph's own embedding (an ending glyph shared by many words).

| model | L0 | L1 | L2 | L4 | L6 | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| own letters | 0.28 | 0.32 | 0.32 | 0.32 | 0.36 | 0.36 | 0.40 | 0.40 | 0.32 | 0.28 | 0.32 |

## 2. Attention to the glyph's own word vs the previous word (ratio per glyph, mean over heads)

| model | L0 | L1 | L2 | L4 | L6 | L8 | L12 | L16 | L20 | L24 | mean all layers |
|---|---|---|---|---|---|---|---|---|---|---|---|
| own letters | 17.47 | 15.22 | 12.16 | 13.09 | 4.34 | 4.76 | 5.42 | 5.66 | 16.29 | 5.69 | 10.35 |

## 3. Logit lens: what kind of English token does a glyph's hidden state look like?

Share of the top-10 base-vocabulary tokens that are whole words (3+ letters) / letter pieces (1-2).

| model | L0 | L1 | L2 | L4 | L6 | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| own letters | 0.24/0.65 | 0.70/0.04 | 0.65/0.07 | 0.43/0.12 | 0.30/0.17 | 0.22/0.23 | 0.44/0.21 | 0.35/0.23 | 0.39/0.22 | 0.30/0.34 | 0.14/0.56 |

Plot: `results/hidden_word_assembly.png`.
