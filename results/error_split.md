# Understanding vs translation errors (`analysis/error_split.py`)

Generated-sentence groups come from subgraph_check, which does not separate 'word not in language' from
'unreadable' (both are under 'unreadable' there).

Share of test items in each group. Translation error = analysis right, glyphs wrong. Understanding error =
analysis wrong, glyphs exactly the model's own analysis rendered (faithful translation of a misreading).
Both wrong = analysis wrong and glyphs not even its own analysis.

## 300 generated sentences (from results/subgraph)

| run | n | both right | translation error | lucky glyphs | understanding error | both wrong | word not in language | unreadable |
|---|---|---|---|---|---|---|---|---|
| e2b-v2fix-6ep | 300 | 66% | 15% | 0% | 9% | 1% | 0% | 8% |
| e2b-v2fix-cot-80k-noheld | 300 | 60% | 19% | 1% | 10% | 3% | 0% | 7% |
| e2b-v2fix-floor25 | 300 | 71% | 12% | 1% | 8% | 2% | 0% | 6% |
| e2b-v2fix-focal-r2 | 300 | 56% | 19% | 0% | 9% | 5% | 0% | 10% |
| e2b-v2fix-loan12 | 300 | 63% | 15% | 0% | 8% | 4% | 0% | 9% |
| e2b-v2fix-n98k | 300 | 66% | 18% | 1% | 7% | 1% | 0% | 7% |
| e2b-v2fix-para | 300 | 63% | 18% | 0% | 8% | 3% | 0% | 7% |
| e2b-v2fix-passive0 | 300 | 61% | 20% | 0% | 9% | 2% | 0% | 8% |
| e2b-v2fix-passive0-ts1 | 300 | 63% | 17% | 0% | 7% | 4% | 0% | 9% |
| e2b-v2fix-passive300 | 300 | 64% | 19% | 0% | 8% | 2% | 0% | 8% |
| e2b-v2fix-passive50 | 300 | 59% | 21% | 0% | 7% | 3% | 0% | 10% |
| e2b-v2fix-passive50-ts1 | 300 | 60% | 21% | 0% | 8% | 3% | 0% | 8% |
| e2b-v2fix-real15 | 300 | 58% | 21% | 0% | 9% | 3% | 0% | 8% |
| e2b-v2fix-real15-ts1 | 300 | 61% | 23% | 0% | 8% | 2% | 0% | 6% |
| e2b-v2fix-soundrule | 300 | 75% | 12% | 1% | 5% | 1% | 0% | 6% |
| e2b-v2fix-ts1 | 300 | 62% | 20% | 1% | 6% | 2% | 0% | 9% |

## Tatoeba 600

| run | n | both right | translation error | lucky glyphs | understanding error | both wrong | word not in language | unreadable |
|---|---|---|---|---|---|---|---|---|
| e2b-v2fix-6ep | 600 | 44% | 13% | 2% | 13% | 9% | 20% | 0% |
| e2b-v2fix-cot-80k-noheld | 600 | 46% | 16% | 1% | 14% | 10% | 14% | 0% |
| e2b-v2fix-cot-80k-noheld@epoch1 | 600 | 28% | 29% | 1% | 10% | 18% | 15% | 0% |
| e2b-v2fix-cot-80k-noheld@epoch2 | 600 | 41% | 20% | 1% | 12% | 11% | 14% | 0% |
| e2b-v2fix-everyday | 600 | 60% | 13% | 1% | 12% | 4% | 12% | 0% |
| e2b-v2fix-floor25 | 600 | 47% | 13% | 1% | 16% | 10% | 12% | 0% |
| e2b-v2fix-focal-r2 | 600 | 44% | 15% | 2% | 16% | 10% | 14% | 0% |
| e2b-v2fix-loan12 | 600 | 46% | 15% | 1% | 16% | 7% | 14% | 0% |
| e2b-v2fix-n98k | 600 | 48% | 13% | 1% | 14% | 11% | 13% | 0% |
| e2b-v2fix-para | 600 | 46% | 15% | 1% | 15% | 8% | 14% | 0% |
| e2b-v2fix-passive0 | 600 | 47% | 14% | 1% | 15% | 9% | 14% | 0% |
| e2b-v2fix-passive0-meaning | 600 | 0% | 60% | 0% | 0% | 25% | 15% | 0% |
| e2b-v2fix-passive0-ts1 | 600 | 46% | 14% | 1% | 14% | 10% | 16% | 0% |
| e2b-v2fix-passive300 | 600 | 46% | 15% | 1% | 16% | 8% | 13% | 0% |
| e2b-v2fix-passive50 | 600 | 43% | 18% | 1% | 15% | 10% | 13% | 0% |
| e2b-v2fix-passive50-ts1 | 600 | 46% | 14% | 1% | 12% | 11% | 14% | 0% |
| e2b-v2fix-q17 | 600 | 42% | 20% | 0% | 13% | 11% | 13% | 0% |
| e2b-v2fix-real15 | 600 | 89% | 3% | 0% | 5% | 1% | 2% | 0% |
| e2b-v2fix-real15-ts1 | 600 | 89% | 4% | 0% | 4% | 1% | 2% | 0% |
| e2b-v2fix-soundrule | 600 | 49% | 8% | 1% | 18% | 8% | 15% | 0% |
| e2b-v2fix-ts1 | 600 | 45% | 17% | 1% | 13% | 10% | 14% | 0% |

## Sealed v2, 147 in scope (totals only)

| run | n | both right | translation error | lucky glyphs | analysis wrong, glyphs wrong |
|---|---|---|---|---|---|
| e2b-v2fix-6ep | 147 | 38% | 7% | 1% | 54% |
| e2b-v2fix-cot-80k-noheld | 147 | 39% | 8% | 2% | 50% |
| e2b-v2fix-cot-80k-noheld@epoch1 | 147 | 23% | 20% | 3% | 54% |
| e2b-v2fix-cot-80k-noheld@epoch2 | 147 | 39% | 7% | 1% | 52% |
| e2b-v2fix-everyday | 147 | 47% | 5% | 1% | 48% |
| e2b-v2fix-floor25 | 147 | 44% | 6% | 1% | 48% |
| e2b-v2fix-focal-r2 | 147 | 37% | 7% | 1% | 54% |
| e2b-v2fix-loan12 | 147 | 42% | 4% | 1% | 53% |
| e2b-v2fix-n98k | 147 | 41% | 5% | 2% | 51% |
| e2b-v2fix-para | 147 | 41% | 4% | 2% | 53% |
| e2b-v2fix-passive0 | 147 | 44% | 4% | 1% | 51% |
| e2b-v2fix-passive0-meaning | 147 | 0% | 47% | 0% | 53% |
| e2b-v2fix-passive0-ts1 | 147 | 41% | 4% | 2% | 53% |
| e2b-v2fix-passive300 | 147 | 37% | 6% | 1% | 55% |
| e2b-v2fix-passive50 | 147 | 37% | 7% | 2% | 54% |
| e2b-v2fix-passive50-ts1 | 147 | 39% | 5% | 3% | 53% |
| e2b-v2fix-real15 | 147 | 55% | 1% | 1% | 44% |
| e2b-v2fix-real15-ts1 | 147 | 54% | 1% | 1% | 44% |
| e2b-v2fix-soundrule | 147 | 40% | 6% | 4% | 50% |
| e2b-v2fix-ts1 | 147 | 41% | 5% | 2% | 52% |

Sealed, full split (re-scored since 30 Oct: per-entry category labels only, no text):

| run | n | both right | translation error | lucky glyphs | understanding error | both wrong | word not in language | unreadable |
|---|---|---|---|---|---|---|---|---|
| e2b-v2fix-6ep | 147 | 38% | 7% | 1% | 20% | 11% | 22% | 0% |
| e2b-v2fix-cot-80k-noheld | 147 | 39% | 8% | 2% | 18% | 10% | 22% | 0% |
| e2b-v2fix-everyday | 147 | 47% | 5% | 1% | 20% | 8% | 19% | 1% |
| e2b-v2fix-floor25 | 147 | 44% | 6% | 1% | 20% | 8% | 20% | 0% |
| e2b-v2fix-focal-r2 | 147 | 37% | 7% | 1% | 28% | 9% | 18% | 0% |
| e2b-v2fix-loan12 | 147 | 42% | 4% | 1% | 21% | 8% | 24% | 0% |
| e2b-v2fix-n98k | 147 | 41% | 5% | 2% | 22% | 8% | 21% | 0% |
| e2b-v2fix-para | 147 | 41% | 4% | 2% | 24% | 8% | 20% | 0% |
| e2b-v2fix-passive0 | 147 | 44% | 4% | 1% | 22% | 7% | 22% | 0% |
| e2b-v2fix-passive0-meaning | 147 | 0% | 47% | 0% | 0% | 29% | 24% | 0% |
| e2b-v2fix-passive0-ts1 | 147 | 41% | 4% | 2% | 19% | 11% | 23% | 0% |
| e2b-v2fix-passive300 | 147 | 37% | 6% | 1% | 26% | 8% | 21% | 0% |
| e2b-v2fix-passive50 | 147 | 37% | 7% | 2% | 24% | 10% | 20% | 0% |
| e2b-v2fix-passive50-ts1 | 147 | 39% | 5% | 3% | 22% | 10% | 21% | 1% |
| e2b-v2fix-real15 | 147 | 55% | 1% | 1% | 30% | 5% | 8% | 0% |
| e2b-v2fix-real15-ts1 | 147 | 54% | 1% | 1% | 31% | 4% | 9% | 0% |
| e2b-v2fix-soundrule | 147 | 40% | 6% | 4% | 18% | 7% | 25% | 0% |
| e2b-v2fix-ts1 | 147 | 41% | 5% | 2% | 23% | 9% | 20% | 0% |

## Writing judged on the model's own meaning (30 Oct)

Writing error = glyphs differ from the model's own meaning rendered (right or wrong meaning), among items whose own meaning can be rendered (unreadable / word-not-in-language items excluded; they count as reading errors). Reading error = meaning wrong, share of all items.

| run | test | renderable | writing error | reading error | writing errors / reading errors |
|---|---|---|---|---|---|
| e2b-v2fix-6ep | generated | 275/300 | 0.18 | 0.19 | 50/57 |
| e2b-v2fix-6ep | tatoeba | 481/600 | 0.30 | 0.44 | 142/262 |
| e2b-v2fix-6ep | sealed | 114/147 | 0.25 | 0.55 | 28/81 |
| e2b-v2fix-cot-80k-noheld | generated | 278/300 | 0.24 | 0.21 | 68/62 |
| e2b-v2fix-cot-80k-noheld | tatoeba | 515/600 | 0.31 | 0.38 | 159/230 |
| e2b-v2fix-cot-80k-noheld | sealed | 114/147 | 0.26 | 0.52 | 30/77 |
| e2b-v2fix-cot-80k-noheld@epoch1 | tatoeba | 512/600 | 0.55 | 0.42 | 284/254 |
| e2b-v2fix-cot-80k-noheld@epoch2 | tatoeba | 514/600 | 0.38 | 0.39 | 197/233 |
| e2b-v2fix-everyday | tatoeba | 530/600 | 0.19 | 0.28 | 103/165 |
| e2b-v2fix-everyday | sealed | 118/147 | 0.17 | 0.48 | 20/71 |
| e2b-v2fix-floor25 | generated | 281/300 | 0.16 | 0.16 | 44/49 |
| e2b-v2fix-floor25 | tatoeba | 527/600 | 0.28 | 0.40 | 145/238 |
| e2b-v2fix-floor25 | sealed | 117/147 | 0.20 | 0.50 | 23/73 |
| e2b-v2fix-focal-r2 | generated | 271/300 | 0.27 | 0.25 | 74/74 |
| e2b-v2fix-focal-r2 | tatoeba | 518/600 | 0.31 | 0.41 | 159/247 |
| e2b-v2fix-focal-r2 | sealed | 121/147 | 0.21 | 0.56 | 25/82 |
| e2b-v2fix-loan12 | generated | 273/300 | 0.22 | 0.22 | 59/65 |
| e2b-v2fix-loan12 | tatoeba | 514/600 | 0.26 | 0.39 | 136/233 |
| e2b-v2fix-loan12 | sealed | 112/147 | 0.17 | 0.54 | 19/79 |
| e2b-v2fix-n98k | generated | 279/300 | 0.22 | 0.16 | 60/47 |
| e2b-v2fix-n98k | tatoeba | 523/600 | 0.29 | 0.39 | 151/234 |
| e2b-v2fix-n98k | sealed | 116/147 | 0.20 | 0.53 | 23/78 |
| e2b-v2fix-para | generated | 278/300 | 0.23 | 0.19 | 63/56 |
| e2b-v2fix-para | tatoeba | 517/600 | 0.28 | 0.39 | 147/231 |
| e2b-v2fix-para | sealed | 117/147 | 0.18 | 0.55 | 21/81 |
| e2b-v2fix-passive0 | generated | 275/300 | 0.24 | 0.19 | 67/57 |
| e2b-v2fix-passive0 | tatoeba | 513/600 | 0.27 | 0.40 | 141/238 |
| e2b-v2fix-passive0 | sealed | 115/147 | 0.16 | 0.52 | 18/77 |
| e2b-v2fix-passive0-meaning | tatoeba | 511/600 | 1.00 | 0.40 | 511/238 |
| e2b-v2fix-passive0-meaning | sealed | 112/147 | 1.00 | 0.53 | 112/78 |
| e2b-v2fix-passive0-ts1 | generated | 273/300 | 0.23 | 0.20 | 64/59 |
| e2b-v2fix-passive0-ts1 | tatoeba | 506/600 | 0.29 | 0.40 | 148/242 |
| e2b-v2fix-passive0-ts1 | sealed | 113/147 | 0.22 | 0.55 | 25/81 |
| e2b-v2fix-passive300 | generated | 277/300 | 0.22 | 0.17 | 62/51 |
| e2b-v2fix-passive300 | tatoeba | 520/600 | 0.28 | 0.39 | 144/233 |
| e2b-v2fix-passive300 | sealed | 116/147 | 0.20 | 0.56 | 23/83 |
| e2b-v2fix-passive50 | generated | 270/300 | 0.27 | 0.21 | 73/62 |
| e2b-v2fix-passive50 | tatoeba | 523/600 | 0.33 | 0.39 | 175/236 |
| e2b-v2fix-passive50 | sealed | 117/147 | 0.24 | 0.56 | 28/83 |
| e2b-v2fix-passive50-ts1 | generated | 277/300 | 0.26 | 0.19 | 72/57 |
| e2b-v2fix-passive50-ts1 | tatoeba | 513/600 | 0.31 | 0.39 | 161/236 |
| e2b-v2fix-passive50-ts1 | sealed | 115/147 | 0.22 | 0.56 | 25/82 |
| e2b-v2fix-q17 | tatoeba | 523/600 | 0.37 | 0.38 | 192/225 |
| e2b-v2fix-real15 | generated | 277/300 | 0.27 | 0.20 | 74/61 |
| e2b-v2fix-real15 | tatoeba | 589/600 | 0.05 | 0.08 | 27/46 |
| e2b-v2fix-real15 | sealed | 135/147 | 0.07 | 0.44 | 10/65 |
| e2b-v2fix-real15-ts1 | generated | 282/300 | 0.27 | 0.16 | 76/49 |
| e2b-v2fix-real15-ts1 | tatoeba | 589/600 | 0.05 | 0.07 | 30/43 |
| e2b-v2fix-real15-ts1 | sealed | 134/147 | 0.07 | 0.44 | 9/65 |
| e2b-v2fix-soundrule | generated | 281/300 | 0.14 | 0.13 | 40/39 |
| e2b-v2fix-soundrule | tatoeba | 512/600 | 0.21 | 0.42 | 106/254 |
| e2b-v2fix-soundrule | sealed | 110/147 | 0.23 | 0.54 | 25/79 |
| e2b-v2fix-ts1 | generated | 273/300 | 0.25 | 0.18 | 68/55 |
| e2b-v2fix-ts1 | tatoeba | 518/600 | 0.32 | 0.38 | 168/225 |
| e2b-v2fix-ts1 | sealed | 118/147 | 0.19 | 0.54 | 23/79 |
