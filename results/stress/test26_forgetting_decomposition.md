# Test 26: where does the English damage come from?

English perplexity on 30 everyday sentences (base Qwen3-0.6B-Base: 20.3). 'Masked' = glyph tokens removed from the softmax: if masked perplexity is near 20, the damage is just probability STOLEN by glyphs; if it stays high, the adapter changed the English predictions themselves. Rows: trained glyph rows vs their starting values; LoRA on (1) or off (0). Translation accuracy on 100 new sentences (LoRA on only).

Cells: perplexity / masked perplexity / probability on glyphs / translation accuracy.

| glyph rows, LoRA | Option 1: Latin-sound start | Option 2: + transliteration | Option 3a: plain text first | Control: shuffled-sound start |
|---|---|---|---|---|
| trained, LoRA 1.0 | 372.0 / 246.7 / 0.195 / 99.0% | 8907560.0 / 136791.2 / 0.772 / 100.0% | 13215565.0 / 88597.4 / 0.874 / 31.0% | 933.9 / 693.0 / 0.150 / 99.0% |
| trained, LoRA 0.0 | 20.5 / 20.3 / 0.005 / - | 23.2 / 20.3 / 0.081 / - | 21.3 / 20.3 / 0.033 / - | 20.5 / 20.3 / 0.005 / - |
| start, LoRA 1.0 | 247.9 / 246.7 / 0.004 / 0.0% | 142090.1 / 136791.2 / 0.033 / 0.0% | 98098.1 / 88597.4 / 0.084 / 0.0% | 697.6 / 693.0 / 0.006 / 0.0% |
| start, LoRA 0.0 | 20.4 / 20.3 / 0.000 / - | 20.3 / 20.3 / 0.000 / - | 20.3 / 20.3 / 0.000 / - | 20.4 / 20.3 / 0.000 / - |
