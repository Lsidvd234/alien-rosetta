# Frequency-weighted word accuracy (v2 models)

Every glyph word of the gold answer is right or wrong (aligned with difflib), weighted by w = exp(lambda * (L + T) / 2): L = English frequency (zipf, scaled 0-1), T = how often the word appears in that run's own training data (log scale, 0-1). Common, often-trained words weigh up to e^lambda times more than rare, unseen ones. lambda = 0 is plain word accuracy. Names left out; pronouns and grammar words count as very common. 300 generated test sentences, the same for every run, none in its training data. `analysis/freq_weighted_eval.py`.

| run | sentences exact | word acc, lambda=0 | word acc, lambda=1 | word acc, lambda=2 | word acc, lambda=3 |
|---|---|---|---|---|---|
| e2a-v2-80k | 0.390 | 0.867 | 0.884 | 0.899 | 0.910 |
| e2b-v2-cot-80k-noheld | 0.477 | 0.901 | 0.916 | 0.928 | 0.937 |
| e2b-v2-cot-80k | 0.417 | 0.862 | 0.876 | 0.888 | 0.897 |
| e2b-v2fix-cot-80k-noheld | 0.607 | 0.933 | 0.937 | 0.940 | 0.943 |
| e2b-v2fix-cot-80k | 0.493 | 0.855 | 0.862 | 0.867 | 0.872 |
| e2b-v2fix-focal-r2 | 0.567 | 0.922 | 0.930 | 0.937 | 0.942 |
| e2b-v2fix-loan-r2 | 0.633 | 0.934 | 0.941 | 0.947 | 0.951 |
| e2b-v2fix-para-s1 | 0.653 | 0.947 | 0.952 | 0.955 | 0.958 |
| e2b-v2fix-para | 0.633 | 0.936 | 0.940 | 0.943 | 0.945 |
| e2b-v2fix-para8 | 0.617 | 0.936 | 0.942 | 0.947 | 0.951 |

## Lambda sweep (weighted word accuracy)

| lambda | e2a-v2-80k | e2b-v2-cot-80k-noheld | e2b-v2-cot-80k | e2b-v2fix-cot-80k-noheld | e2b-v2fix-cot-80k | e2b-v2fix-focal-r2 | e2b-v2fix-loan-r2 | e2b-v2fix-para-s1 | e2b-v2fix-para | e2b-v2fix-para8 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.8665 | 0.9006 | 0.8618 | 0.9332 | 0.8547 | 0.9224 | 0.9337 | 0.9473 | 0.9358 | 0.9358 |
| 0.1 | 0.8684 | 0.9023 | 0.8634 | 0.9336 | 0.8555 | 0.9232 | 0.9344 | 0.9478 | 0.9363 | 0.9365 |
| 0.3 | 0.8721 | 0.9055 | 0.8664 | 0.9345 | 0.8569 | 0.9247 | 0.9359 | 0.9487 | 0.9371 | 0.9377 |
| 0.5 | 0.8758 | 0.9086 | 0.8694 | 0.9353 | 0.8583 | 0.9263 | 0.9374 | 0.9496 | 0.9379 | 0.9389 |
| 0.7 | 0.8792 | 0.9116 | 0.8722 | 0.9361 | 0.8597 | 0.9278 | 0.9388 | 0.9505 | 0.9387 | 0.9401 |
| 0.9 | 0.8826 | 0.9145 | 0.8749 | 0.9368 | 0.8610 | 0.9293 | 0.9401 | 0.9513 | 0.9395 | 0.9413 |
| 1 | 0.8842 | 0.9159 | 0.8763 | 0.9372 | 0.8617 | 0.9300 | 0.9408 | 0.9517 | 0.9399 | 0.9418 |
| 2 | 0.8986 | 0.9279 | 0.8879 | 0.9404 | 0.8674 | 0.9365 | 0.9466 | 0.9553 | 0.9431 | 0.9468 |
| 3 | 0.9097 | 0.9368 | 0.8965 | 0.9427 | 0.8716 | 0.9417 | 0.9510 | 0.9578 | 0.9453 | 0.9505 |
| 4 | 0.9174 | 0.9428 | 0.9025 | 0.9440 | 0.8743 | 0.9454 | 0.9539 | 0.9594 | 0.9464 | 0.9531 |
| 5 | 0.9225 | 0.9464 | 0.9061 | 0.9444 | 0.8757 | 0.9478 | 0.9556 | 0.9600 | 0.9466 | 0.9544 |
| 6 | 0.9253 | 0.9481 | 0.9080 | 0.9441 | 0.8761 | 0.9489 | 0.9562 | 0.9600 | 0.9459 | 0.9548 |
| 7 | 0.9265 | 0.9484 | 0.9085 | 0.9431 | 0.8756 | 0.9490 | 0.9560 | 0.9594 | 0.9446 | 0.9545 |
| 8 | 0.9264 | 0.9478 | 0.9080 | 0.9415 | 0.8746 | 0.9484 | 0.9552 | 0.9583 | 0.9428 | 0.9535 |
| 9 | 0.9255 | 0.9465 | 0.9069 | 0.9396 | 0.8731 | 0.9472 | 0.9540 | 0.9569 | 0.9405 | 0.9521 |
| 10 | 0.9239 | 0.9446 | 0.9052 | 0.9374 | 0.8713 | 0.9456 | 0.9524 | 0.9553 | 0.9380 | 0.9503 |
| 11 | 0.9219 | 0.9424 | 0.9032 | 0.9349 | 0.8693 | 0.9436 | 0.9506 | 0.9535 | 0.9353 | 0.9484 |
| 12 | 0.9196 | 0.9400 | 0.9009 | 0.9322 | 0.8672 | 0.9415 | 0.9486 | 0.9515 | 0.9324 | 0.9462 |
| 13 | 0.9170 | 0.9374 | 0.8985 | 0.9294 | 0.8649 | 0.9391 | 0.9465 | 0.9495 | 0.9294 | 0.9439 |
| 14 | 0.9143 | 0.9347 | 0.8959 | 0.9264 | 0.8627 | 0.9367 | 0.9443 | 0.9474 | 0.9263 | 0.9415 |
| 15 | 0.9115 | 0.9319 | 0.8933 | 0.9234 | 0.8603 | 0.9341 | 0.9420 | 0.9452 | 0.9231 | 0.9390 |
| 16 | 0.9086 | 0.9290 | 0.8906 | 0.9203 | 0.8579 | 0.9315 | 0.9397 | 0.9430 | 0.9199 | 0.9365 |
| 17 | 0.9056 | 0.9260 | 0.8878 | 0.9171 | 0.8555 | 0.9289 | 0.9373 | 0.9407 | 0.9167 | 0.9339 |
| 18 | 0.9025 | 0.9230 | 0.8850 | 0.9139 | 0.8531 | 0.9262 | 0.9349 | 0.9384 | 0.9134 | 0.9313 |
| 19 | 0.8994 | 0.9200 | 0.8822 | 0.9107 | 0.8507 | 0.9235 | 0.9325 | 0.9362 | 0.9101 | 0.9287 |
| 20 | 0.8963 | 0.9170 | 0.8794 | 0.9075 | 0.8483 | 0.9208 | 0.9301 | 0.9339 | 0.9068 | 0.9261 |
| 21 | 0.8932 | 0.9140 | 0.8765 | 0.9042 | 0.8459 | 0.9180 | 0.9277 | 0.9316 | 0.9035 | 0.9235 |
| 22 | 0.8901 | 0.9109 | 0.8737 | 0.9010 | 0.8435 | 0.9153 | 0.9253 | 0.9294 | 0.9002 | 0.9209 |
| 23 | 0.8870 | 0.9079 | 0.8708 | 0.8977 | 0.8411 | 0.9126 | 0.9229 | 0.9271 | 0.8969 | 0.9183 |
| 24 | 0.8840 | 0.9049 | 0.8680 | 0.8945 | 0.8387 | 0.9100 | 0.9205 | 0.9249 | 0.8937 | 0.9157 |
| 25 | 0.8809 | 0.9019 | 0.8652 | 0.8913 | 0.8363 | 0.9073 | 0.9182 | 0.9227 | 0.8905 | 0.9132 |
| 26 | 0.8779 | 0.8990 | 0.8625 | 0.8881 | 0.8340 | 0.9047 | 0.9158 | 0.9206 | 0.8873 | 0.9107 |
| 27 | 0.8749 | 0.8961 | 0.8597 | 0.8849 | 0.8317 | 0.9021 | 0.9135 | 0.9185 | 0.8842 | 0.9082 |
| 28 | 0.8719 | 0.8932 | 0.8570 | 0.8818 | 0.8294 | 0.8996 | 0.9112 | 0.9164 | 0.8811 | 0.9057 |
| 29 | 0.8690 | 0.8904 | 0.8544 | 0.8788 | 0.8271 | 0.8971 | 0.9090 | 0.9143 | 0.8780 | 0.9033 |
| 30 | 0.8661 | 0.8876 | 0.8518 | 0.8758 | 0.8249 | 0.8946 | 0.9067 | 0.9123 | 0.8750 | 0.9009 |

Ranking of the runs at every lambda (best first):

- lambda 0 to 0: e2b-v2fix-para-s1 > e2b-v2fix-para > e2b-v2fix-para8 > e2b-v2fix-loan-r2 > e2b-v2fix-cot-80k-noheld > e2b-v2fix-focal-r2 > e2b-v2-cot-80k-noheld > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 0.1 to 0.5: e2b-v2fix-para-s1 > e2b-v2fix-para8 > e2b-v2fix-para > e2b-v2fix-loan-r2 > e2b-v2fix-cot-80k-noheld > e2b-v2fix-focal-r2 > e2b-v2-cot-80k-noheld > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 0.7 to 2: e2b-v2fix-para-s1 > e2b-v2fix-para8 > e2b-v2fix-loan-r2 > e2b-v2fix-para > e2b-v2fix-cot-80k-noheld > e2b-v2fix-focal-r2 > e2b-v2-cot-80k-noheld > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 3 to 3: e2b-v2fix-para-s1 > e2b-v2fix-loan-r2 > e2b-v2fix-para8 > e2b-v2fix-para > e2b-v2fix-cot-80k-noheld > e2b-v2fix-focal-r2 > e2b-v2-cot-80k-noheld > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 4 to 4: e2b-v2fix-para-s1 > e2b-v2fix-loan-r2 > e2b-v2fix-para8 > e2b-v2fix-para > e2b-v2fix-focal-r2 > e2b-v2fix-cot-80k-noheld > e2b-v2-cot-80k-noheld > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 5 to 5: e2b-v2fix-para-s1 > e2b-v2fix-loan-r2 > e2b-v2fix-para8 > e2b-v2fix-focal-r2 > e2b-v2fix-para > e2b-v2-cot-80k-noheld > e2b-v2fix-cot-80k-noheld > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 6 to 13: e2b-v2fix-para-s1 > e2b-v2fix-loan-r2 > e2b-v2fix-para8 > e2b-v2fix-focal-r2 > e2b-v2-cot-80k-noheld > e2b-v2fix-para > e2b-v2fix-cot-80k-noheld > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 14 to 30: e2b-v2fix-para-s1 > e2b-v2fix-loan-r2 > e2b-v2fix-para8 > e2b-v2fix-focal-r2 > e2b-v2-cot-80k-noheld > e2b-v2fix-cot-80k-noheld > e2b-v2fix-para > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k

## Summary: plain and weighted (lambda = 3, fixed in advance) accuracy

| run | word, plain | word, weighted | sentence, plain | sentence, weighted |
|---|---|---|---|---|
| e2a-v2-80k | 0.867 | 0.910 | 0.390 | 0.402 |
| e2b-v2-cot-80k-noheld | 0.901 | 0.937 | 0.477 | 0.507 |
| e2b-v2-cot-80k | 0.862 | 0.897 | 0.417 | 0.426 |
| e2b-v2fix-cot-80k-noheld | 0.933 | 0.943 | 0.607 | 0.615 |
| e2b-v2fix-cot-80k | 0.855 | 0.872 | 0.493 | 0.509 |
| e2b-v2fix-focal-r2 | 0.922 | 0.942 | 0.567 | 0.583 |
| e2b-v2fix-loan-r2 | 0.934 | 0.951 | 0.633 | 0.654 |
| e2b-v2fix-para-s1 | 0.947 | 0.958 | 0.653 | 0.662 |
| e2b-v2fix-para | 0.936 | 0.945 | 0.633 | 0.639 |
| e2b-v2fix-para8 | 0.936 | 0.951 | 0.617 | 0.626 |

## Lambda sweep, full-sentence accuracy (sentences weighted by how common their words are)

| lambda | e2a-v2-80k | e2b-v2-cot-80k-noheld | e2b-v2-cot-80k | e2b-v2fix-cot-80k-noheld | e2b-v2fix-cot-80k | e2b-v2fix-focal-r2 | e2b-v2fix-loan-r2 | e2b-v2fix-para-s1 | e2b-v2fix-para | e2b-v2fix-para8 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.3900 | 0.4767 | 0.4167 | 0.6067 | 0.4933 | 0.5667 | 0.6333 | 0.6533 | 0.6333 | 0.6167 |
| 0.1 | 0.3903 | 0.4776 | 0.4169 | 0.6069 | 0.4937 | 0.5672 | 0.6340 | 0.6536 | 0.6335 | 0.6169 |
| 0.3 | 0.3909 | 0.4795 | 0.4174 | 0.6073 | 0.4945 | 0.5682 | 0.6352 | 0.6540 | 0.6337 | 0.6175 |
| 0.5 | 0.3915 | 0.4814 | 0.4179 | 0.6078 | 0.4953 | 0.5692 | 0.6365 | 0.6545 | 0.6340 | 0.6181 |
| 0.7 | 0.3922 | 0.4833 | 0.4184 | 0.6082 | 0.4961 | 0.5703 | 0.6379 | 0.6550 | 0.6343 | 0.6187 |
| 0.9 | 0.3929 | 0.4852 | 0.4189 | 0.6087 | 0.4970 | 0.5714 | 0.6392 | 0.6556 | 0.6346 | 0.6192 |
| 1 | 0.3933 | 0.4862 | 0.4192 | 0.6090 | 0.4975 | 0.5719 | 0.6399 | 0.6558 | 0.6347 | 0.6196 |
| 2 | 0.3974 | 0.4963 | 0.4225 | 0.6117 | 0.5026 | 0.5775 | 0.6468 | 0.6586 | 0.6366 | 0.6227 |
| 3 | 0.4024 | 0.5069 | 0.4265 | 0.6149 | 0.5088 | 0.5834 | 0.6542 | 0.6619 | 0.6390 | 0.6262 |
| 4 | 0.4082 | 0.5180 | 0.4310 | 0.6185 | 0.5159 | 0.5897 | 0.6620 | 0.6655 | 0.6419 | 0.6300 |
| 5 | 0.4147 | 0.5294 | 0.4362 | 0.6225 | 0.5239 | 0.5962 | 0.6702 | 0.6698 | 0.6453 | 0.6341 |
| 6 | 0.4220 | 0.5413 | 0.4419 | 0.6269 | 0.5328 | 0.6029 | 0.6788 | 0.6746 | 0.6492 | 0.6384 |
| 7 | 0.4299 | 0.5535 | 0.4482 | 0.6317 | 0.5425 | 0.6099 | 0.6878 | 0.6801 | 0.6536 | 0.6431 |
| 8 | 0.4385 | 0.5661 | 0.4550 | 0.6368 | 0.5530 | 0.6170 | 0.6970 | 0.6863 | 0.6585 | 0.6480 |
| 9 | 0.4478 | 0.5788 | 0.4622 | 0.6424 | 0.5642 | 0.6244 | 0.7066 | 0.6933 | 0.6639 | 0.6533 |
| 10 | 0.4576 | 0.5918 | 0.4700 | 0.6483 | 0.5761 | 0.6320 | 0.7164 | 0.7011 | 0.6699 | 0.6589 |
| 11 | 0.4680 | 0.6050 | 0.4782 | 0.6546 | 0.5886 | 0.6397 | 0.7264 | 0.7097 | 0.6763 | 0.6648 |
| 12 | 0.4790 | 0.6183 | 0.4868 | 0.6612 | 0.6017 | 0.6477 | 0.7365 | 0.7192 | 0.6833 | 0.6711 |
| 13 | 0.4905 | 0.6317 | 0.4959 | 0.6683 | 0.6152 | 0.6558 | 0.7468 | 0.7294 | 0.6907 | 0.6778 |
| 14 | 0.5025 | 0.6451 | 0.5055 | 0.6757 | 0.6291 | 0.6641 | 0.7571 | 0.7402 | 0.6986 | 0.6849 |
| 15 | 0.5150 | 0.6585 | 0.5155 | 0.6835 | 0.6434 | 0.6726 | 0.7675 | 0.7517 | 0.7069 | 0.6923 |
| 16 | 0.5279 | 0.6718 | 0.5260 | 0.6916 | 0.6579 | 0.6813 | 0.7778 | 0.7637 | 0.7156 | 0.7001 |
| 17 | 0.5413 | 0.6851 | 0.5369 | 0.7001 | 0.6725 | 0.6902 | 0.7880 | 0.7760 | 0.7247 | 0.7082 |
| 18 | 0.5550 | 0.6983 | 0.5483 | 0.7088 | 0.6872 | 0.6993 | 0.7981 | 0.7885 | 0.7341 | 0.7167 |
| 19 | 0.5691 | 0.7113 | 0.5601 | 0.7179 | 0.7020 | 0.7086 | 0.8081 | 0.8010 | 0.7437 | 0.7256 |
| 20 | 0.5834 | 0.7242 | 0.5724 | 0.7272 | 0.7166 | 0.7180 | 0.8179 | 0.8134 | 0.7536 | 0.7347 |
| 21 | 0.5981 | 0.7369 | 0.5850 | 0.7367 | 0.7311 | 0.7276 | 0.8274 | 0.8257 | 0.7637 | 0.7441 |
| 22 | 0.6130 | 0.7493 | 0.5980 | 0.7464 | 0.7454 | 0.7373 | 0.8368 | 0.8375 | 0.7738 | 0.7537 |
| 23 | 0.6280 | 0.7616 | 0.6114 | 0.7563 | 0.7595 | 0.7471 | 0.8458 | 0.8490 | 0.7840 | 0.7634 |
| 24 | 0.6431 | 0.7735 | 0.6251 | 0.7662 | 0.7732 | 0.7570 | 0.8545 | 0.8599 | 0.7942 | 0.7733 |
| 25 | 0.6583 | 0.7852 | 0.6390 | 0.7762 | 0.7865 | 0.7670 | 0.8630 | 0.8703 | 0.8043 | 0.7833 |
| 26 | 0.6735 | 0.7966 | 0.6532 | 0.7862 | 0.7994 | 0.7769 | 0.8711 | 0.8800 | 0.8143 | 0.7933 |
| 27 | 0.6886 | 0.8077 | 0.6675 | 0.7962 | 0.8119 | 0.7868 | 0.8789 | 0.8892 | 0.8242 | 0.8032 |
| 28 | 0.7036 | 0.8185 | 0.6818 | 0.8060 | 0.8239 | 0.7967 | 0.8864 | 0.8978 | 0.8339 | 0.8131 |
| 29 | 0.7184 | 0.8289 | 0.6962 | 0.8158 | 0.8354 | 0.8064 | 0.8935 | 0.9057 | 0.8434 | 0.8228 |
| 30 | 0.7330 | 0.8389 | 0.7105 | 0.8253 | 0.8463 | 0.8160 | 0.9003 | 0.9131 | 0.8526 | 0.8323 |

## e2a-v2-80k: word accuracy by frequency (number of words in brackets)

| seen in training | rare in English (zipf < 3) | medium (3-4.5) | common (> 4.5) | all |
|---|---|---|---|---|
| never | - | - | 0.99 (99) | 0.99 (99) |
| 1-9 times | 0.27 (26) | 0.30 (108) | 1.00 (3) | 0.31 (137) |
| 10-29 times | 0.64 (11) | 0.80 (382) | 0.80 (15) | 0.79 (408) |
| 30+ times | - | 0.89 (214) | 0.93 (1487) | 0.92 (1701) |

Common, often-trained words it still got wrong: be (25), much (3), PRON.2pl (3), want (2), cause (2), afford (2), when (2), many (2), PRON.3sgm (1), walk (1), grade (1), someone (1), remember (1), update (1), challenge (1)

## e2b-v2-cot-80k-noheld: word accuracy by frequency (number of words in brackets)

| seen in training | rare in English (zipf < 3) | medium (3-4.5) | common (> 4.5) | all |
|---|---|---|---|---|
| never | - | - | 1.00 (102) | 1.00 (102) |
| 1-9 times | 0.23 (30) | 0.30 (93) | - | 0.28 (123) |
| 10-29 times | 0.86 (7) | 0.86 (404) | 0.80 (15) | 0.85 (426) |
| 30+ times | - | 0.93 (212) | 0.95 (1501) | 0.95 (1713) |

Common, often-trained words it still got wrong: be (22), want (2), who (2), grade (1), someone (1), update (1), apple (1), challenge (1), survive (1), work (1), set (1), run (1), cause (1), tired (1), tough (1)

## e2b-v2-cot-80k: word accuracy by frequency (number of words in brackets)

| seen in training | rare in English (zipf < 3) | medium (3-4.5) | common (> 4.5) | all |
|---|---|---|---|---|
| never | - | - | 0.97 (99) | 0.97 (99) |
| 1-9 times | 0.35 (26) | 0.40 (108) | 1.00 (3) | 0.40 (137) |
| 10-29 times | 0.73 (11) | 0.83 (382) | 0.87 (15) | 0.83 (408) |
| 30+ times | - | 0.82 (214) | 0.91 (1487) | 0.90 (1701) |

Common, often-trained words it still got wrong: be (26), PRON.2pl (5), PRON.3sgf (3), when (3), PRON.2sg (2), run (2), promise (2), really (2), who (2), every (2), much (2), take (2), summer (2), PRON.1sg (2), that (2)

## e2b-v2fix-cot-80k-noheld: word accuracy by frequency (number of words in brackets)

| seen in training | rare in English (zipf < 3) | medium (3-4.5) | common (> 4.5) | all |
|---|---|---|---|---|
| never | - | - | 1.00 (97) | 1.00 (97) |
| 1-9 times | - | - | 1.00 (1) | 1.00 (1) |
| 10-29 times | 0.80 (35) | 0.90 (457) | 0.93 (15) | 0.89 (507) |
| 30+ times | - | 0.92 (239) | 0.95 (1462) | 0.94 (1701) |

Common, often-trained words it still got wrong: be (24), write (2), promise (2), away (1), how (1), grade (1), want (1), every (1), year (1), ask (1), apple (1), challenge (1), then (1), case (1), live (1)

## e2b-v2fix-cot-80k: word accuracy by frequency (number of words in brackets)

| seen in training | rare in English (zipf < 3) | medium (3-4.5) | common (> 4.5) | all |
|---|---|---|---|---|
| never | - | - | 0.93 (97) | 0.93 (97) |
| 1-9 times | - | - | 1.00 (2) | 1.00 (2) |
| 10-29 times | 0.43 (35) | 0.81 (442) | 0.86 (14) | 0.79 (491) |
| 30+ times | - | 0.81 (254) | 0.88 (1462) | 0.87 (1716) |

Common, often-trained words it still got wrong: be (28), PRON.2pl (8), PRON.2sg (6), who (5), much (4), PRON.3sgf (4), this (3), when (3), PRON.3sgm (3), how (3), that (3), want (2), cause (2), PRON.3pl (2), thank (2)

## e2b-v2fix-focal-r2: word accuracy by frequency (number of words in brackets)

| seen in training | rare in English (zipf < 3) | medium (3-4.5) | common (> 4.5) | all |
|---|---|---|---|---|
| never | - | - | 1.00 (97) | 1.00 (97) |
| 1-9 times | - | - | 1.00 (1) | 1.00 (1) |
| 10-29 times | 0.66 (35) | 0.86 (457) | 0.87 (15) | 0.84 (507) |
| 30+ times | - | 0.90 (239) | 0.95 (1462) | 0.94 (1701) |

Common, often-trained words it still got wrong: be (20), cause (2), afford (2), promise (2), how (1), grade (1), want (1), someone (1), ask (1), apple (1), survive (1), work (1), set (1), usually (1), tired (1)

## e2b-v2fix-loan-r2: word accuracy by frequency (number of words in brackets)

| seen in training | rare in English (zipf < 3) | medium (3-4.5) | common (> 4.5) | all |
|---|---|---|---|---|
| never | - | - | 1.00 (97) | 1.00 (97) |
| 1-9 times | - | - | 1.00 (2) | 1.00 (2) |
| 10-29 times | 0.63 (35) | 0.86 (443) | 0.88 (16) | 0.84 (494) |
| 30+ times | - | 0.94 (253) | 0.96 (1460) | 0.96 (1713) |

Common, often-trained words it still got wrong: be (18), cause (2), promise (2), how (1), grade (1), someone (1), every (1), year (1), ask (1), update (1), apple (1), protect (1), work (1), set (1), run (1)

## e2b-v2fix-para-s1: word accuracy by frequency (number of words in brackets)

| seen in training | rare in English (zipf < 3) | medium (3-4.5) | common (> 4.5) | all |
|---|---|---|---|---|
| never | - | 0.00 (1) | 0.99 (96) | 0.98 (97) |
| 1-9 times | - | 1.00 (1) | 0.75 (4) | 0.80 (5) |
| 10-29 times | 0.69 (36) | 0.91 (425) | 0.62 (13) | 0.89 (474) |
| 30+ times | - | 0.95 (241) | 0.97 (1441) | 0.96 (1682) |

Common, often-trained words it still got wrong: be (18), speak (2), means (2), give (2), PRON.3sgf (1), PRON.1pl (1), want (1), march (1), play (1), ouch (1), great (1), goodbye (1), away (1), mean (1), dress (1)

## e2b-v2fix-para: word accuracy by frequency (number of words in brackets)

| seen in training | rare in English (zipf < 3) | medium (3-4.5) | common (> 4.5) | all |
|---|---|---|---|---|
| never | - | - | 0.99 (97) | 0.99 (97) |
| 1-9 times | - | - | 1.00 (1) | 1.00 (1) |
| 10-29 times | 0.74 (35) | 0.89 (457) | 0.93 (15) | 0.88 (507) |
| 30+ times | - | 0.92 (239) | 0.95 (1462) | 0.95 (1701) |

Common, often-trained words it still got wrong: be (24), cause (2), write (2), who (2), away (1), how (1), grade (1), someone (1), ask (1), survive (1), set (1), tired (1), each (1), promise (1), study (1)

## e2b-v2fix-para8: word accuracy by frequency (number of words in brackets)

| seen in training | rare in English (zipf < 3) | medium (3-4.5) | common (> 4.5) | all |
|---|---|---|---|---|
| never | - | - | 1.00 (97) | 1.00 (97) |
| 1-9 times | - | - | 1.00 (1) | 1.00 (1) |
| 10-29 times | 0.77 (35) | 0.88 (457) | 0.93 (15) | 0.87 (507) |
| 30+ times | - | 0.93 (239) | 0.96 (1462) | 0.95 (1701) |

Common, often-trained words it still got wrong: be (19), want (2), who (2), grade (1), someone (1), ask (1), apple (1), challenge (1), then (1), case (1), live (1), survive (1), set (1), run (1), cause (1)
