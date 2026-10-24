# Final 20k results (Qwen3-0.6B-Base, LoRA r=16, 3 epochs, 20,000 sentences)

Three random languages per cell (seeds 0, 1, 2; same grammar, different invented vocabulary).
Content-word accuracy = every token of the content word right. Exact = whole sentence right (test_iid).

| cell | content-word acc (seed 0 / 1 / 2) | mean | sd | exact iid mean | tokens/sent | train min |
|---|---|---|---|---|---|---|
| easy-latin | 0.817 / 0.846 / 0.810 | 0.824 | 0.019 | 0.655 | 12.3 | 15.8 |
| easy-tamil | 0.830 / 0.872 / 0.868 | 0.857 | 0.023 | 0.722 | 38.9 | 20.8 |
| hard-latin | 0.048 / 0.028 / 0.038 | 0.038 | 0.010 | 0.009 | 19.6 | 15.6 |
| hard-tamil | 0.232 / 0.257 / 0.233 | 0.241 | 0.014 | 0.117 | 43.3 | 21.4 |

## Cause tests (seed 0, 20k)

| language | target written as | tokens/sent | content-word acc | exact iid | train min |
|---|---|---|---|---|---|
| easy | Latin, normal | 12.5 | 0.817 | 0.634 | 15.7 |
| easy | Latin, 1 char/token | 30.9 | 0.855 | 0.698 | 18.3 |
| easy | Tamil script | 42.0 | 0.830 | 0.650 | 21.7 |
| easy | Invented glyphs (PUA) | 94.6 | 0.003 | 0.002 | 42.9 |
| hard | Latin, normal | 19.3 | 0.048 | 0.022 | 15.8 |
| hard | Latin, 1 char/token | 42.9 | 0.106 | 0.054 | 21.0 |
| hard | Accented Latin, 1 char/token | 33.1 | 0.016 | 0.010 | 19.8 |
| hard | Tamil script | 43.4 | 0.232 | 0.100 | 21.3 |
| hard | Invented glyphs (PUA) | 90.9 | 0.001 | 0.000 | 38.8 |
| easy | Invented glyphs, 1 new token each | 35.5 | 0.001 | 0.000 | 19.7 |
| hard | Invented glyphs, 1 new token each | 32.6 | 0.012 | 0.000 | 19.1 |

## Invented glyphs (PUA) by training size

| language | glyphs as | sentences | tokens/sent | content-word acc | exact iid | exact comp | exact morph | exact long | train min |
|---|---|---|---|---|---|---|---|---|---|
| easy | 3 byte tokens each | 20k | 94.6 | 0.003 | 0.002 | 0.000 | 0.002 | 0.000 | 42.9 |
| easy | 3 byte tokens each | 50k | 94.8 | 0.992 | 0.978 | 0.982 | 0.950 | 0.962 | 105.7 |
| easy | 3 byte tokens each | 80k | 95.0 | 0.998 | 0.994 | 0.992 | 0.990 | 0.988 | 165.8 |
| easy | 1 new token each | 20k | 35.5 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 19.7 |
| easy | 1 new token each | 50k | 35.6 | 0.992 | 0.990 | 0.984 | 0.958 | 0.946 | 50.2 |
| hard | 3 byte tokens each | 20k | 90.9 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 38.8 |
| hard | 3 byte tokens each | 35k | 91.0 | 0.446 | 0.008 | 0.000 | 0.000 | 0.000 | 72.0 |
| hard | 3 byte tokens each | 50k | 91.0 | 0.636 | 0.376 | 0.144 | 0.274 | 0.010 | 97.2 |
| hard | 3 byte tokens each | 80k | 91.3 | 0.898 | 0.990 | 0.968 | 0.464 | 0.048 | 156.9 |
| hard | 1 new token each | 20k | 32.6 | 0.012 | 0.000 | 0.000 | 0.000 | 0.000 | 19.1 |
| hard | 1 new token each | 50k | 32.6 | 0.755 | 0.570 | 0.506 | 0.420 | 0.000 | 52.0 |

1-new-token runs: 20k used mean-embedding init, 50k used copy-of-random-ASCII-token init.

## Making hard grammar + invented glyphs generalise (50k, glyphs as 1 new token each)

Each fix changes one thing against the baseline. Latin and Tamil script rows are the targets.

| run | change | content-word acc | exact iid | exact comp | exact morph | exact long | train min |
|---|---|---|---|---|---|---|---|
| Baseline | new glyph tokens start as copies of random words | 0.755 | 0.570 | 0.506 | 0.420 | 0.000 | 52.0 |
| Option 1 | each glyph starts as the embedding of its Latin sound | 0.995 | 0.990 | 0.986 | 0.972 | 0.974 | 47.8 |
| Option 2 | baseline + glyph<->Latin transliteration examples | 0.984 | 0.990 | 0.976 | 0.896 | 0.894 | 103.5 |
| Option 3a | baseline + plain-text phase (50k glyph-only sentences) first | 0.740 | 0.320 | 0.002 | 0.266 | 0.002 | 78.8 |
| Control | each glyph starts at a WRONG sound (shuffled) | 0.961 | 0.982 | 0.978 | 0.860 | 0.650 | 46.1 |
| Target | Latin script | 0.980 | 0.960 | 0.952 | 0.836 | 0.940 | 40.7 |
| Target | Tamil script | 0.990 | 0.986 | 0.976 | 0.920 | 0.958 | 53.6 |

Option 2 trains on 2x the examples (translation + transliteration), so ~2x the compute. Option 3a adds a plain-text phase on 50k extra sentences before translation. One seed each; re-runs with saved weights (s1/s2/s3-*) reproduced every score exactly.
