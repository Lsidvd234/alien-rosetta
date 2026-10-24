# Test 21: model health

## 1. Did fine-tuning damage plain English? (LoRA strength 0 = adapter off, 1 = as trained)

Base Qwen3-0.6B-Base English perplexity (30 everyday sentences): **20.3**

Cells: English perplexity / probability put on glyph tokens while reading English / translation accuracy (200 new sentences).

| model | strength 0 | strength 0.25 | strength 0.5 | strength 0.75 | strength 1.0 |
|---|---|---|---|---|---|
| Option 1: Latin-sound start | 20.5 / 0.005 / 0.0% | 20.4 / 0.009 / 0.0% | 26.0 / 0.019 / 0.0% | 55.8 / 0.056 / 10.0% | 372.0 / 0.195 / 98.5% |
| Option 2: + transliteration | 23.2 / 0.081 / 0.0% | 38.4 / 0.208 / 0.0% | 340.5 / 0.556 / 0.0% | 230783.8 / 0.773 / 15.5% | 8907560.0 / 0.772 / 98.5% |
| Option 3a: plain text first | 21.3 / 0.033 / 0.0% | 44.9 / 0.243 / 0.0% | 523.3 / 0.690 / 0.0% | 103244.5 / 0.866 / 1.5% | 13215565.0 / 0.874 / 25.0% |
| Control: shuffled-sound start | 20.5 / 0.005 / 0.0% | 21.8 / 0.009 / 0.0% | 29.9 / 0.018 / 0.0% | 73.7 / 0.049 / 15.0% | 933.9 / 0.150 / 99.0% |

## 2. Glyphs leaking into ordinary English continuation (no 'Target:' line)

| model | continuations containing glyphs |
|---|---|
| Option 1: Latin-sound start | 46.7% (e.g. '') |
| Option 2: + transliteration | 100.0% (e.g. '\ue00e\ue066\ue007\ue061\ue00e\ue065\ue00c\ue07f \ue007\ue063\ue00b\ue064\ue00c\ue062 \ue007\ue063\ue00b\ue064\ue00c\ue062\ue00a\ue061\ue009') |
| Option 3a: plain text first | 100.0% (e.g. '\ue001\ue063\ue00e\ue062\ue00f\ue07f \ue00e\ue064\ue00a\ue063\ue004\ue063\ue007\ue07f \ue010\ue064\ue00c\ue063\ue007\ue062\ue00b\ue061\ue009') |
| Control: shuffled-sound start | 100.0% (e.g. '\ue00f\ue062\ue00f\ue064\ue00f\ue07f \ue00b\ue065\ue007\ue063\ue00a\ue068\ue00a\ue07f \ue00c\ue002\ue061\ue00a\ue061\ue009\ue061 \ue00b') |

## 3. Tokenizer round trip

- 52000 targets; decode(encode(x)) != x for **0**.
- Targets where the glyph-token count differs from the glyph count (first 5,000): **0**.
- Non-glyph tokens used inside targets: ' ' x4901.

## 4. Are test sentences near-copies of training sentences?

| split | exact copy of a training sentence | differs from one by a single word |
|---|---|---|
| test_iid | 0.0% | 18.4% |
| test_comp | 0.0% | 1.8% |
| test_morph | 0.0% | 8.0% |
| test_long | 0.0% | 0.0% |

## 5. Glyph embeddings: size vs Qwen's vocabulary, training frequency, accuracy

Qwen vocabulary embedding norm: median 0.97 (5th-95th pct 0.73-1.14).

| model | glyph norm median (min-max) | rarest glyph (training count) | per-glyph accuracy: rarest 5 / rest |
|---|---|---|---|
| Option 1: Latin-sound start | 1.11 (0.44-1.27) | 24572 | 100.0% / 100.0% |
| Option 2: + transliteration | 1.11 (0.99-1.27) | 24572 | 100.0% / 99.9% |
| Option 3a: plain text first | 1.07 (0.95-1.24) | 24572 | 87.9% / 94.9% |
| Control: shuffled-sound start | 1.12 (0.44-1.27) | 24572 | 99.6% / 99.9% |

## 6. Control model: do its glyph mix-ups follow the wrong sounds it started from?

Glyph-for-glyph substitution errors (teacher-forced, 1,200 test sentences): **191**; of these, the wrong glyph was the one whose TRUE sound equals the gold glyph's (wrong) START sound: **5** (2.6%; chance ≈ 3.7%).
