# Does adding a word's letter vectors give the word's vector? (`analysis/letters_sum_to_word.py`)

Words: the 27 word pieces s4-randtok started its glyphs from. Letter sum vs the word's base embedding, cosine (centred). `_` in the spelling = a letter with no glyph (left out of the glyph sums).

## Summary

| letters taken from | mean cos(letter sum, its word) | chance (vs the other 26 words) | own word nearest of 27 |
|---|---|---|---|
| English letters | 0.030 | 0.007 | 2 / 27 |
| own start | 0.019 | 0.008 | 2 / 26 |
| own trained | 0.019 | 0.007 | 2 / 26 |
| p1 start | 0.019 | 0.008 | 2 / 26 |
| p1 trained | 0.020 | 0.008 | 2 / 26 |

## Per word

| word | glyph spelling | letters covered | English | own start | own trained | p1 start | p1 trained |
|---|---|---|---|---|---|---|---|
| 'Extra' | e-_-t-r-_ | 60% | 0.12 | 0.08 | 0.08 | 0.08 | 0.07 |
| 'getStatus' | _-e-t-_-t-_-t-u-_ | 56% | -0.10 | -0.09 | -0.08 | -0.09 | -0.08 |
| 'acente' | _-c-e-n-t-e | 83% | -0.04 | -0.03 | -0.03 | -0.03 | -0.02 |
| ' Dez' | _-e-_ | 33% | -0.02 | -0.06 | -0.06 | -0.06 | -0.06 |
| 'licht' | l-i-c-_-t | 80% | -0.01 | -0.01 | -0.01 | -0.01 | -0.00 |
| ' nap' | n-_-p | 67% | 0.11 | 0.14 | 0.14 | 0.14 | 0.16 |
| 'gas' | _-_-_ | 0% | 0.18 | 0.00 | 0.00 | 0.00 | 0.00 |
| 'ondheim' | o-n-_-_-e-i-m | 71% | -0.03 | -0.04 | -0.04 | -0.04 | -0.04 |
| ' imageSize' | i-m-_-_-e-_-i-_-e | 56% | -0.08 | -0.06 | -0.07 | -0.06 | -0.08 |
| ' motifs' | m-o-t-i-_-_ | 67% | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| ' weight' | _-e-i-_-_-t | 50% | 0.16 | 0.12 | 0.12 | 0.12 | 0.13 |
| ' DMA' | _-m-_ | 33% | 0.03 | 0.01 | 0.01 | 0.01 | 0.01 |
| ' rem' | r-e-m | 100% | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 |
| ' Georgetown' | _-e-o-r-_-e-t-o-_-n | 70% | -0.04 | -0.04 | -0.04 | -0.04 | -0.04 |
| ' Established' | e-_-t-_-_-l-i-_-_-e-_ | 45% | -0.00 | 0.01 | 0.01 | 0.01 | 0.01 |
| ' seks' | _-e-k-_ | 50% | 0.02 | -0.01 | -0.00 | -0.01 | -0.01 |
| 'opers' | o-p-e-r-_ | 80% | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 |
| 'BTC' | _-t-c | 67% | 0.02 | -0.01 | -0.01 | -0.01 | -0.01 |
| 'ylene' | y-l-e-n-e | 100% | 0.09 | 0.09 | 0.10 | 0.09 | 0.10 |
| ' perpetr' | p-e-r-p-e-t-r | 100% | -0.14 | -0.14 | -0.14 | -0.14 | -0.14 |
| 'lahoma' | l-_-_-o-m-_ | 50% | 0.01 | 0.04 | 0.03 | 0.04 | 0.03 |
| ' Pure' | p-u-r-e | 100% | 0.10 | 0.10 | 0.11 | 0.10 | 0.10 |
| 'Hmm' | _-m-m | 67% | 0.05 | 0.02 | 0.02 | 0.02 | 0.03 |
| ' addslashes' | _-_-_-_-l-_-_-_-e-_ | 20% | -0.04 | -0.05 | -0.05 | -0.05 | -0.04 |
| 'avern' | _-v-e-r-n | 80% | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| ' foundational' | _-o-u-n-_-_-t-i-o-n-_-l | 67% | 0.04 | 0.03 | 0.04 | 0.03 | 0.03 |
| 'Chapter' | c-_-_-p-t-e-r | 71% | 0.13 | 0.13 | 0.13 | 0.13 | 0.12 |

