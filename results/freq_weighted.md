# Frequency-weighted word accuracy (v2 models)

Every glyph word of the gold answer is right or wrong (aligned with difflib), weighted by w = exp(lambda * (L + T) / 2): L = English frequency (zipf, scaled 0-1), T = how often the word appears in that run's own training data (log scale, 0-1). Common, often-trained words weigh up to e^lambda times more than rare, unseen ones. lambda = 0 is plain word accuracy. Names left out; pronouns and grammar words count as very common. 300 generated test sentences, the same for every run, none in its training data. `analysis/freq_weighted_eval.py`.

| run | sentences exact | word acc, lambda=0 | word acc, lambda=1 | word acc, lambda=2 | word acc, lambda=3 |
|---|---|---|---|---|---|
| e2a-v2-80k | 0.390 | 0.867 | 0.884 | 0.899 | 0.910 |
| e2b-v2-cot-80k-noheld | 0.477 | 0.901 | 0.916 | 0.928 | 0.937 |
| e2b-v2-cot-80k | 0.417 | 0.862 | 0.876 | 0.888 | 0.897 |
| e2b-v2fix-6ep | 0.660 | 0.945 | 0.948 | 0.951 | 0.953 |
| e2b-v2fix-cot-80k-noheld | 0.607 | 0.933 | 0.937 | 0.940 | 0.943 |
| e2b-v2fix-cot-80k | 0.493 | 0.855 | 0.862 | 0.867 | 0.872 |
| e2b-v2fix-floor25 | 0.720 | 0.955 | 0.958 | 0.961 | 0.963 |
| e2b-v2fix-focal-r2 | 0.567 | 0.922 | 0.930 | 0.937 | 0.942 |
| e2b-v2fix-loan-r2 | 0.633 | 0.934 | 0.941 | 0.947 | 0.951 |
| e2b-v2fix-para | 0.633 | 0.936 | 0.940 | 0.943 | 0.945 |
| e2b-v2fix-para8 | 0.617 | 0.936 | 0.942 | 0.947 | 0.951 |
| e2b-v2fix-real15-ts1 | 0.610 | 0.928 | 0.935 | 0.941 | 0.946 |
| e2b-v2fix-real15 | 0.587 | 0.929 | 0.934 | 0.939 | 0.942 |
| e2b-v2fix-ts1 | 0.627 | 0.935 | 0.941 | 0.947 | 0.950 |

## Lambda sweep (weighted word accuracy)

| lambda | e2a-v2-80k | e2b-v2-cot-80k-noheld | e2b-v2-cot-80k | e2b-v2fix-6ep | e2b-v2fix-cot-80k-noheld | e2b-v2fix-cot-80k | e2b-v2fix-floor25 | e2b-v2fix-focal-r2 | e2b-v2fix-loan-r2 | e2b-v2fix-para | e2b-v2fix-para8 | e2b-v2fix-real15-ts1 | e2b-v2fix-real15 | e2b-v2fix-ts1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.8665 | 0.9006 | 0.8618 | 0.9449 | 0.9332 | 0.8547 | 0.9553 | 0.9224 | 0.9337 | 0.9358 | 0.9358 | 0.9280 | 0.9289 | 0.9350 |
| 0.1 | 0.8684 | 0.9023 | 0.8634 | 0.9453 | 0.9336 | 0.8555 | 0.9556 | 0.9232 | 0.9344 | 0.9363 | 0.9365 | 0.9288 | 0.9295 | 0.9357 |
| 0.3 | 0.8721 | 0.9055 | 0.8664 | 0.9460 | 0.9345 | 0.8569 | 0.9562 | 0.9247 | 0.9359 | 0.9371 | 0.9377 | 0.9303 | 0.9306 | 0.9370 |
| 0.5 | 0.8758 | 0.9086 | 0.8694 | 0.9467 | 0.9353 | 0.8583 | 0.9568 | 0.9263 | 0.9374 | 0.9379 | 0.9389 | 0.9318 | 0.9318 | 0.9384 |
| 0.7 | 0.8792 | 0.9116 | 0.8722 | 0.9474 | 0.9361 | 0.8597 | 0.9574 | 0.9278 | 0.9388 | 0.9387 | 0.9401 | 0.9333 | 0.9328 | 0.9397 |
| 0.9 | 0.8826 | 0.9145 | 0.8749 | 0.9481 | 0.9368 | 0.8610 | 0.9580 | 0.9293 | 0.9401 | 0.9395 | 0.9413 | 0.9347 | 0.9339 | 0.9409 |
| 1 | 0.8842 | 0.9159 | 0.8763 | 0.9484 | 0.9372 | 0.8617 | 0.9583 | 0.9300 | 0.9408 | 0.9399 | 0.9418 | 0.9354 | 0.9344 | 0.9415 |
| 2 | 0.8986 | 0.9279 | 0.8879 | 0.9512 | 0.9404 | 0.8674 | 0.9609 | 0.9365 | 0.9466 | 0.9431 | 0.9468 | 0.9413 | 0.9388 | 0.9466 |
| 3 | 0.9097 | 0.9368 | 0.8965 | 0.9531 | 0.9427 | 0.8716 | 0.9632 | 0.9417 | 0.9510 | 0.9453 | 0.9505 | 0.9455 | 0.9419 | 0.9501 |
| 4 | 0.9174 | 0.9428 | 0.9025 | 0.9542 | 0.9440 | 0.8743 | 0.9649 | 0.9454 | 0.9539 | 0.9464 | 0.9531 | 0.9482 | 0.9437 | 0.9521 |
| 5 | 0.9225 | 0.9464 | 0.9061 | 0.9545 | 0.9444 | 0.8757 | 0.9659 | 0.9478 | 0.9556 | 0.9466 | 0.9544 | 0.9494 | 0.9443 | 0.9528 |
| 6 | 0.9253 | 0.9481 | 0.9080 | 0.9541 | 0.9441 | 0.8761 | 0.9664 | 0.9489 | 0.9562 | 0.9459 | 0.9548 | 0.9494 | 0.9439 | 0.9524 |
| 7 | 0.9265 | 0.9484 | 0.9085 | 0.9530 | 0.9431 | 0.8756 | 0.9665 | 0.9490 | 0.9560 | 0.9446 | 0.9545 | 0.9485 | 0.9427 | 0.9512 |
| 8 | 0.9264 | 0.9478 | 0.9080 | 0.9515 | 0.9415 | 0.8746 | 0.9660 | 0.9484 | 0.9552 | 0.9428 | 0.9535 | 0.9470 | 0.9409 | 0.9494 |
| 9 | 0.9255 | 0.9465 | 0.9069 | 0.9497 | 0.9396 | 0.8731 | 0.9653 | 0.9472 | 0.9540 | 0.9405 | 0.9521 | 0.9449 | 0.9386 | 0.9471 |
| 10 | 0.9239 | 0.9446 | 0.9052 | 0.9476 | 0.9374 | 0.8713 | 0.9643 | 0.9456 | 0.9524 | 0.9380 | 0.9503 | 0.9425 | 0.9359 | 0.9445 |
| 11 | 0.9219 | 0.9424 | 0.9032 | 0.9453 | 0.9349 | 0.8693 | 0.9630 | 0.9436 | 0.9506 | 0.9353 | 0.9484 | 0.9399 | 0.9331 | 0.9418 |
| 12 | 0.9196 | 0.9400 | 0.9009 | 0.9428 | 0.9322 | 0.8672 | 0.9616 | 0.9415 | 0.9486 | 0.9324 | 0.9462 | 0.9371 | 0.9300 | 0.9388 |
| 13 | 0.9170 | 0.9374 | 0.8985 | 0.9403 | 0.9294 | 0.8649 | 0.9601 | 0.9391 | 0.9465 | 0.9294 | 0.9439 | 0.9342 | 0.9268 | 0.9357 |
| 14 | 0.9143 | 0.9347 | 0.8959 | 0.9376 | 0.9264 | 0.8627 | 0.9585 | 0.9367 | 0.9443 | 0.9263 | 0.9415 | 0.9313 | 0.9235 | 0.9326 |
| 15 | 0.9115 | 0.9319 | 0.8933 | 0.9349 | 0.9234 | 0.8603 | 0.9568 | 0.9341 | 0.9420 | 0.9231 | 0.9390 | 0.9282 | 0.9201 | 0.9294 |
| 16 | 0.9086 | 0.9290 | 0.8906 | 0.9322 | 0.9203 | 0.8579 | 0.9551 | 0.9315 | 0.9397 | 0.9199 | 0.9365 | 0.9252 | 0.9166 | 0.9261 |
| 17 | 0.9056 | 0.9260 | 0.8878 | 0.9294 | 0.9171 | 0.8555 | 0.9533 | 0.9289 | 0.9373 | 0.9167 | 0.9339 | 0.9220 | 0.9131 | 0.9229 |
| 18 | 0.9025 | 0.9230 | 0.8850 | 0.9267 | 0.9139 | 0.8531 | 0.9515 | 0.9262 | 0.9349 | 0.9134 | 0.9313 | 0.9189 | 0.9096 | 0.9196 |
| 19 | 0.8994 | 0.9200 | 0.8822 | 0.9239 | 0.9107 | 0.8507 | 0.9496 | 0.9235 | 0.9325 | 0.9101 | 0.9287 | 0.9158 | 0.9061 | 0.9163 |
| 20 | 0.8963 | 0.9170 | 0.8794 | 0.9211 | 0.9075 | 0.8483 | 0.9478 | 0.9208 | 0.9301 | 0.9068 | 0.9261 | 0.9127 | 0.9026 | 0.9130 |
| 21 | 0.8932 | 0.9140 | 0.8765 | 0.9183 | 0.9042 | 0.8459 | 0.9459 | 0.9180 | 0.9277 | 0.9035 | 0.9235 | 0.9096 | 0.8991 | 0.9097 |
| 22 | 0.8901 | 0.9109 | 0.8737 | 0.9155 | 0.9010 | 0.8435 | 0.9441 | 0.9153 | 0.9253 | 0.9002 | 0.9209 | 0.9065 | 0.8956 | 0.9065 |
| 23 | 0.8870 | 0.9079 | 0.8708 | 0.9128 | 0.8977 | 0.8411 | 0.9422 | 0.9126 | 0.9229 | 0.8969 | 0.9183 | 0.9034 | 0.8921 | 0.9032 |
| 24 | 0.8840 | 0.9049 | 0.8680 | 0.9101 | 0.8945 | 0.8387 | 0.9404 | 0.9100 | 0.9205 | 0.8937 | 0.9157 | 0.9004 | 0.8886 | 0.9000 |
| 25 | 0.8809 | 0.9019 | 0.8652 | 0.9074 | 0.8913 | 0.8363 | 0.9385 | 0.9073 | 0.9182 | 0.8905 | 0.9132 | 0.8974 | 0.8852 | 0.8968 |
| 26 | 0.8779 | 0.8990 | 0.8625 | 0.9047 | 0.8881 | 0.8340 | 0.9367 | 0.9047 | 0.9158 | 0.8873 | 0.9107 | 0.8944 | 0.8818 | 0.8937 |
| 27 | 0.8749 | 0.8961 | 0.8597 | 0.9021 | 0.8849 | 0.8317 | 0.9349 | 0.9021 | 0.9135 | 0.8842 | 0.9082 | 0.8915 | 0.8785 | 0.8906 |
| 28 | 0.8719 | 0.8932 | 0.8570 | 0.8995 | 0.8818 | 0.8294 | 0.9331 | 0.8996 | 0.9112 | 0.8811 | 0.9057 | 0.8887 | 0.8752 | 0.8875 |
| 29 | 0.8690 | 0.8904 | 0.8544 | 0.8970 | 0.8788 | 0.8271 | 0.9314 | 0.8971 | 0.9090 | 0.8780 | 0.9033 | 0.8858 | 0.8720 | 0.8845 |
| 30 | 0.8661 | 0.8876 | 0.8518 | 0.8945 | 0.8758 | 0.8249 | 0.9296 | 0.8946 | 0.9067 | 0.8750 | 0.9009 | 0.8831 | 0.8689 | 0.8815 |

Ranking of the runs at every lambda (best first):

- lambda 0 to 0: e2b-v2fix-floor25 > e2b-v2fix-6ep > e2b-v2fix-para > e2b-v2fix-para8 > e2b-v2fix-ts1 > e2b-v2fix-loan-r2 > e2b-v2fix-cot-80k-noheld > e2b-v2fix-real15 > e2b-v2fix-real15-ts1 > e2b-v2fix-focal-r2 > e2b-v2-cot-80k-noheld > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 0.1 to 0.3: e2b-v2fix-floor25 > e2b-v2fix-6ep > e2b-v2fix-para8 > e2b-v2fix-para > e2b-v2fix-ts1 > e2b-v2fix-loan-r2 > e2b-v2fix-cot-80k-noheld > e2b-v2fix-real15 > e2b-v2fix-real15-ts1 > e2b-v2fix-focal-r2 > e2b-v2-cot-80k-noheld > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 0.5 to 0.5: e2b-v2fix-floor25 > e2b-v2fix-6ep > e2b-v2fix-para8 > e2b-v2fix-ts1 > e2b-v2fix-para > e2b-v2fix-loan-r2 > e2b-v2fix-cot-80k-noheld > e2b-v2fix-real15-ts1 > e2b-v2fix-real15 > e2b-v2fix-focal-r2 > e2b-v2-cot-80k-noheld > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 0.7 to 1: e2b-v2fix-floor25 > e2b-v2fix-6ep > e2b-v2fix-para8 > e2b-v2fix-ts1 > e2b-v2fix-loan-r2 > e2b-v2fix-para > e2b-v2fix-cot-80k-noheld > e2b-v2fix-real15-ts1 > e2b-v2fix-real15 > e2b-v2fix-focal-r2 > e2b-v2-cot-80k-noheld > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 2 to 2: e2b-v2fix-floor25 > e2b-v2fix-6ep > e2b-v2fix-para8 > e2b-v2fix-loan-r2 > e2b-v2fix-ts1 > e2b-v2fix-para > e2b-v2fix-real15-ts1 > e2b-v2fix-cot-80k-noheld > e2b-v2fix-real15 > e2b-v2fix-focal-r2 > e2b-v2-cot-80k-noheld > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 3 to 3: e2b-v2fix-floor25 > e2b-v2fix-6ep > e2b-v2fix-loan-r2 > e2b-v2fix-para8 > e2b-v2fix-ts1 > e2b-v2fix-real15-ts1 > e2b-v2fix-para > e2b-v2fix-cot-80k-noheld > e2b-v2fix-real15 > e2b-v2fix-focal-r2 > e2b-v2-cot-80k-noheld > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 4 to 4: e2b-v2fix-floor25 > e2b-v2fix-6ep > e2b-v2fix-loan-r2 > e2b-v2fix-para8 > e2b-v2fix-ts1 > e2b-v2fix-real15-ts1 > e2b-v2fix-para > e2b-v2fix-focal-r2 > e2b-v2fix-cot-80k-noheld > e2b-v2fix-real15 > e2b-v2-cot-80k-noheld > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 5 to 5: e2b-v2fix-floor25 > e2b-v2fix-loan-r2 > e2b-v2fix-6ep > e2b-v2fix-para8 > e2b-v2fix-ts1 > e2b-v2fix-real15-ts1 > e2b-v2fix-focal-r2 > e2b-v2fix-para > e2b-v2-cot-80k-noheld > e2b-v2fix-cot-80k-noheld > e2b-v2fix-real15 > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 6 to 6: e2b-v2fix-floor25 > e2b-v2fix-loan-r2 > e2b-v2fix-para8 > e2b-v2fix-6ep > e2b-v2fix-ts1 > e2b-v2fix-real15-ts1 > e2b-v2fix-focal-r2 > e2b-v2-cot-80k-noheld > e2b-v2fix-para > e2b-v2fix-cot-80k-noheld > e2b-v2fix-real15 > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 7 to 7: e2b-v2fix-floor25 > e2b-v2fix-loan-r2 > e2b-v2fix-para8 > e2b-v2fix-6ep > e2b-v2fix-ts1 > e2b-v2fix-focal-r2 > e2b-v2fix-real15-ts1 > e2b-v2-cot-80k-noheld > e2b-v2fix-para > e2b-v2fix-cot-80k-noheld > e2b-v2fix-real15 > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 8 to 8: e2b-v2fix-floor25 > e2b-v2fix-loan-r2 > e2b-v2fix-para8 > e2b-v2fix-6ep > e2b-v2fix-ts1 > e2b-v2fix-focal-r2 > e2b-v2-cot-80k-noheld > e2b-v2fix-real15-ts1 > e2b-v2fix-para > e2b-v2fix-cot-80k-noheld > e2b-v2fix-real15 > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 9 to 9: e2b-v2fix-floor25 > e2b-v2fix-loan-r2 > e2b-v2fix-para8 > e2b-v2fix-6ep > e2b-v2fix-focal-r2 > e2b-v2fix-ts1 > e2b-v2-cot-80k-noheld > e2b-v2fix-real15-ts1 > e2b-v2fix-para > e2b-v2fix-cot-80k-noheld > e2b-v2fix-real15 > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 10 to 13: e2b-v2fix-floor25 > e2b-v2fix-loan-r2 > e2b-v2fix-para8 > e2b-v2fix-6ep > e2b-v2fix-focal-r2 > e2b-v2-cot-80k-noheld > e2b-v2fix-ts1 > e2b-v2fix-real15-ts1 > e2b-v2fix-para > e2b-v2fix-cot-80k-noheld > e2b-v2fix-real15 > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 14 to 21: e2b-v2fix-floor25 > e2b-v2fix-loan-r2 > e2b-v2fix-para8 > e2b-v2fix-6ep > e2b-v2fix-focal-r2 > e2b-v2-cot-80k-noheld > e2b-v2fix-ts1 > e2b-v2fix-real15-ts1 > e2b-v2fix-cot-80k-noheld > e2b-v2fix-para > e2b-v2fix-real15 > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 22 to 27: e2b-v2fix-floor25 > e2b-v2fix-loan-r2 > e2b-v2fix-para8 > e2b-v2fix-6ep > e2b-v2fix-focal-r2 > e2b-v2-cot-80k-noheld > e2b-v2fix-real15-ts1 > e2b-v2fix-ts1 > e2b-v2fix-cot-80k-noheld > e2b-v2fix-para > e2b-v2fix-real15 > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k
- lambda 28 to 30: e2b-v2fix-floor25 > e2b-v2fix-loan-r2 > e2b-v2fix-para8 > e2b-v2fix-focal-r2 > e2b-v2fix-6ep > e2b-v2-cot-80k-noheld > e2b-v2fix-real15-ts1 > e2b-v2fix-ts1 > e2b-v2fix-cot-80k-noheld > e2b-v2fix-para > e2b-v2fix-real15 > e2a-v2-80k > e2b-v2-cot-80k > e2b-v2fix-cot-80k

## Summary: plain and weighted (lambda = 3, fixed in advance) accuracy

| run | word, plain | word, weighted | sentence, plain | sentence, weighted |
|---|---|---|---|---|
| e2a-v2-80k | 0.867 | 0.910 | 0.390 | 0.402 |
| e2b-v2-cot-80k-noheld | 0.901 | 0.937 | 0.477 | 0.507 |
| e2b-v2-cot-80k | 0.862 | 0.897 | 0.417 | 0.426 |
| e2b-v2fix-6ep | 0.945 | 0.953 | 0.660 | 0.664 |
| e2b-v2fix-cot-80k-noheld | 0.933 | 0.943 | 0.607 | 0.615 |
| e2b-v2fix-cot-80k | 0.855 | 0.872 | 0.493 | 0.509 |
| e2b-v2fix-floor25 | 0.955 | 0.963 | 0.720 | 0.723 |
| e2b-v2fix-focal-r2 | 0.922 | 0.942 | 0.567 | 0.583 |
| e2b-v2fix-loan-r2 | 0.934 | 0.951 | 0.633 | 0.654 |
| e2b-v2fix-para | 0.936 | 0.945 | 0.633 | 0.639 |
| e2b-v2fix-para8 | 0.936 | 0.951 | 0.617 | 0.626 |
| e2b-v2fix-real15-ts1 | 0.928 | 0.946 | 0.610 | 0.623 |
| e2b-v2fix-real15 | 0.929 | 0.942 | 0.587 | 0.588 |
| e2b-v2fix-ts1 | 0.935 | 0.950 | 0.627 | 0.638 |

## Lambda sweep, full-sentence accuracy (sentences weighted by how common their words are)

| lambda | e2a-v2-80k | e2b-v2-cot-80k-noheld | e2b-v2-cot-80k | e2b-v2fix-6ep | e2b-v2fix-cot-80k-noheld | e2b-v2fix-cot-80k | e2b-v2fix-floor25 | e2b-v2fix-focal-r2 | e2b-v2fix-loan-r2 | e2b-v2fix-para | e2b-v2fix-para8 | e2b-v2fix-real15-ts1 | e2b-v2fix-real15 | e2b-v2fix-ts1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.3900 | 0.4767 | 0.4167 | 0.6600 | 0.6067 | 0.4933 | 0.7200 | 0.5667 | 0.6333 | 0.6333 | 0.6167 | 0.6100 | 0.5867 | 0.6267 |
| 0.1 | 0.3903 | 0.4776 | 0.4169 | 0.6601 | 0.6069 | 0.4937 | 0.7200 | 0.5672 | 0.6340 | 0.6335 | 0.6169 | 0.6104 | 0.5866 | 0.6269 |
| 0.3 | 0.3909 | 0.4795 | 0.4174 | 0.6602 | 0.6073 | 0.4945 | 0.7201 | 0.5682 | 0.6352 | 0.6337 | 0.6175 | 0.6112 | 0.5866 | 0.6275 |
| 0.5 | 0.3915 | 0.4814 | 0.4179 | 0.6603 | 0.6078 | 0.4953 | 0.7202 | 0.5692 | 0.6365 | 0.6340 | 0.6181 | 0.6120 | 0.5866 | 0.6281 |
| 0.7 | 0.3922 | 0.4833 | 0.4184 | 0.6605 | 0.6082 | 0.4961 | 0.7204 | 0.5703 | 0.6379 | 0.6343 | 0.6187 | 0.6128 | 0.5866 | 0.6287 |
| 0.9 | 0.3929 | 0.4852 | 0.4189 | 0.6606 | 0.6087 | 0.4970 | 0.7205 | 0.5714 | 0.6392 | 0.6346 | 0.6192 | 0.6136 | 0.5866 | 0.6293 |
| 1 | 0.3933 | 0.4862 | 0.4192 | 0.6607 | 0.6090 | 0.4975 | 0.7206 | 0.5719 | 0.6399 | 0.6347 | 0.6196 | 0.6140 | 0.5866 | 0.6297 |
| 2 | 0.3974 | 0.4963 | 0.4225 | 0.6620 | 0.6117 | 0.5026 | 0.7215 | 0.5775 | 0.6468 | 0.6366 | 0.6227 | 0.6185 | 0.5872 | 0.6333 |
| 3 | 0.4024 | 0.5069 | 0.4265 | 0.6637 | 0.6149 | 0.5088 | 0.7228 | 0.5834 | 0.6542 | 0.6390 | 0.6262 | 0.6233 | 0.5884 | 0.6376 |
| 4 | 0.4082 | 0.5180 | 0.4310 | 0.6659 | 0.6185 | 0.5159 | 0.7245 | 0.5897 | 0.6620 | 0.6419 | 0.6300 | 0.6286 | 0.5902 | 0.6425 |
| 5 | 0.4147 | 0.5294 | 0.4362 | 0.6687 | 0.6225 | 0.5239 | 0.7266 | 0.5962 | 0.6702 | 0.6453 | 0.6341 | 0.6342 | 0.5925 | 0.6479 |
| 6 | 0.4220 | 0.5413 | 0.4419 | 0.6721 | 0.6269 | 0.5328 | 0.7291 | 0.6029 | 0.6788 | 0.6492 | 0.6384 | 0.6403 | 0.5954 | 0.6538 |
| 7 | 0.4299 | 0.5535 | 0.4482 | 0.6760 | 0.6317 | 0.5425 | 0.7320 | 0.6099 | 0.6878 | 0.6536 | 0.6431 | 0.6467 | 0.5988 | 0.6602 |
| 8 | 0.4385 | 0.5661 | 0.4550 | 0.6804 | 0.6368 | 0.5530 | 0.7353 | 0.6170 | 0.6970 | 0.6585 | 0.6480 | 0.6534 | 0.6028 | 0.6671 |
| 9 | 0.4478 | 0.5788 | 0.4622 | 0.6854 | 0.6424 | 0.5642 | 0.7391 | 0.6244 | 0.7066 | 0.6639 | 0.6533 | 0.6605 | 0.6074 | 0.6743 |
| 10 | 0.4576 | 0.5918 | 0.4700 | 0.6910 | 0.6483 | 0.5761 | 0.7433 | 0.6320 | 0.7164 | 0.6699 | 0.6589 | 0.6679 | 0.6126 | 0.6820 |
| 11 | 0.4680 | 0.6050 | 0.4782 | 0.6972 | 0.6546 | 0.5886 | 0.7479 | 0.6397 | 0.7264 | 0.6763 | 0.6648 | 0.6756 | 0.6183 | 0.6900 |
| 12 | 0.4790 | 0.6183 | 0.4868 | 0.7039 | 0.6612 | 0.6017 | 0.7529 | 0.6477 | 0.7365 | 0.6833 | 0.6711 | 0.6836 | 0.6246 | 0.6984 |
| 13 | 0.4905 | 0.6317 | 0.4959 | 0.7111 | 0.6683 | 0.6152 | 0.7583 | 0.6558 | 0.7468 | 0.6907 | 0.6778 | 0.6918 | 0.6315 | 0.7071 |
| 14 | 0.5025 | 0.6451 | 0.5055 | 0.7189 | 0.6757 | 0.6291 | 0.7641 | 0.6641 | 0.7571 | 0.6986 | 0.6849 | 0.7003 | 0.6389 | 0.7160 |
| 15 | 0.5150 | 0.6585 | 0.5155 | 0.7270 | 0.6835 | 0.6434 | 0.7703 | 0.6726 | 0.7675 | 0.7069 | 0.6923 | 0.7090 | 0.6468 | 0.7252 |
| 16 | 0.5279 | 0.6718 | 0.5260 | 0.7356 | 0.6916 | 0.6579 | 0.7768 | 0.6813 | 0.7778 | 0.7156 | 0.7001 | 0.7178 | 0.6553 | 0.7346 |
| 17 | 0.5413 | 0.6851 | 0.5369 | 0.7446 | 0.7001 | 0.6725 | 0.7836 | 0.6902 | 0.7880 | 0.7247 | 0.7082 | 0.7268 | 0.6642 | 0.7442 |
| 18 | 0.5550 | 0.6983 | 0.5483 | 0.7539 | 0.7088 | 0.6872 | 0.7906 | 0.6993 | 0.7981 | 0.7341 | 0.7167 | 0.7360 | 0.6737 | 0.7538 |
| 19 | 0.5691 | 0.7113 | 0.5601 | 0.7634 | 0.7179 | 0.7020 | 0.7979 | 0.7086 | 0.8081 | 0.7437 | 0.7256 | 0.7453 | 0.6835 | 0.7636 |
| 20 | 0.5834 | 0.7242 | 0.5724 | 0.7732 | 0.7272 | 0.7166 | 0.8053 | 0.7180 | 0.8179 | 0.7536 | 0.7347 | 0.7546 | 0.6937 | 0.7735 |
| 21 | 0.5981 | 0.7369 | 0.5850 | 0.7830 | 0.7367 | 0.7311 | 0.8129 | 0.7276 | 0.8274 | 0.7637 | 0.7441 | 0.7640 | 0.7043 | 0.7833 |
| 22 | 0.6130 | 0.7493 | 0.5980 | 0.7930 | 0.7464 | 0.7454 | 0.8206 | 0.7373 | 0.8368 | 0.7738 | 0.7537 | 0.7734 | 0.7151 | 0.7931 |
| 23 | 0.6280 | 0.7616 | 0.6114 | 0.8029 | 0.7563 | 0.7595 | 0.8283 | 0.7471 | 0.8458 | 0.7840 | 0.7634 | 0.7828 | 0.7261 | 0.8029 |
| 24 | 0.6431 | 0.7735 | 0.6251 | 0.8128 | 0.7662 | 0.7732 | 0.8360 | 0.7570 | 0.8545 | 0.7942 | 0.7733 | 0.7922 | 0.7373 | 0.8125 |
| 25 | 0.6583 | 0.7852 | 0.6390 | 0.8226 | 0.7762 | 0.7865 | 0.8436 | 0.7670 | 0.8630 | 0.8043 | 0.7833 | 0.8015 | 0.7485 | 0.8220 |
| 26 | 0.6735 | 0.7966 | 0.6532 | 0.8323 | 0.7862 | 0.7994 | 0.8512 | 0.7769 | 0.8711 | 0.8143 | 0.7933 | 0.8107 | 0.7598 | 0.8313 |
| 27 | 0.6886 | 0.8077 | 0.6675 | 0.8417 | 0.7962 | 0.8119 | 0.8587 | 0.7868 | 0.8789 | 0.8242 | 0.8032 | 0.8197 | 0.7711 | 0.8404 |
| 28 | 0.7036 | 0.8185 | 0.6818 | 0.8510 | 0.8060 | 0.8239 | 0.8660 | 0.7967 | 0.8864 | 0.8339 | 0.8131 | 0.8286 | 0.7823 | 0.8492 |
| 29 | 0.7184 | 0.8289 | 0.6962 | 0.8599 | 0.8158 | 0.8354 | 0.8731 | 0.8064 | 0.8935 | 0.8434 | 0.8228 | 0.8374 | 0.7933 | 0.8578 |
| 30 | 0.7330 | 0.8389 | 0.7105 | 0.8686 | 0.8253 | 0.8463 | 0.8801 | 0.8160 | 0.9003 | 0.8526 | 0.8323 | 0.8459 | 0.8041 | 0.8661 |

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

## e2b-v2fix-6ep: word accuracy by frequency (number of words in brackets)

| seen in training | rare in English (zipf < 3) | medium (3-4.5) | common (> 4.5) | all |
|---|---|---|---|---|
| never | - | - | 0.99 (97) | 0.99 (97) |
| 1-9 times | - | - | 1.00 (1) | 1.00 (1) |
| 10-29 times | 0.83 (35) | 0.91 (457) | 0.93 (15) | 0.91 (507) |
| 30+ times | - | 0.93 (239) | 0.96 (1462) | 0.95 (1701) |

Common, often-trained words it still got wrong: be (20), write (2), PRON.3sgf (1), PRON.1pl (1), want (1), grade (1), someone (1), PRON.3sgn (1), every (1), year (1), ask (1), apple (1), challenge (1), survive (1), set (1)

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

## e2b-v2fix-floor25: word accuracy by frequency (number of words in brackets)

| seen in training | rare in English (zipf < 3) | medium (3-4.5) | common (> 4.5) | all |
|---|---|---|---|---|
| never | - | - | 1.00 (97) | 1.00 (97) |
| 1-9 times | - | - | 1.00 (1) | 1.00 (1) |
| 10-29 times | 0.85 (33) | 0.93 (364) | 0.95 (63) | 0.93 (460) |
| 30+ times | 1.00 (2) | 0.96 (332) | 0.96 (1414) | 0.96 (1748) |

Common, often-trained words it still got wrong: be (14), cause (2), write (2), away (1), grade (1), want (1), someone (1), every (1), year (1), ask (1), challenge (1), work (1), set (1), run (1), afford (1)

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

## e2b-v2fix-real15-ts1: word accuracy by frequency (number of words in brackets)

| seen in training | rare in English (zipf < 3) | medium (3-4.5) | common (> 4.5) | all |
|---|---|---|---|---|
| never | - | - | 1.00 (97) | 1.00 (97) |
| 1-9 times | - | - | 1.00 (1) | 1.00 (1) |
| 10-29 times | 0.80 (35) | 0.85 (494) | 0.93 (27) | 0.85 (556) |
| 30+ times | - | 0.91 (202) | 0.96 (1450) | 0.95 (1652) |

Common, often-trained words it still got wrong: be (22), cause (2), write (2), grade (1), want (1), PRON.3sgn (1), every (1), year (1), apple (1), challenge (1), survive (1), protect (1), work (1), tired (1), grace (1)

## e2b-v2fix-real15: word accuracy by frequency (number of words in brackets)

| seen in training | rare in English (zipf < 3) | medium (3-4.5) | common (> 4.5) | all |
|---|---|---|---|---|
| never | - | - | 1.00 (97) | 1.00 (97) |
| 1-9 times | - | - | 1.00 (1) | 1.00 (1) |
| 10-29 times | 0.60 (35) | 0.89 (494) | 0.93 (27) | 0.87 (556) |
| 30+ times | - | 0.91 (202) | 0.95 (1450) | 0.94 (1652) |

Common, often-trained words it still got wrong: be (25), cause (2), write (2), grade (1), want (1), someone (1), year (1), ask (1), challenge (1), set (1), run (1), afford (1), tired (1), promise (1), grace (1)

## e2b-v2fix-ts1: word accuracy by frequency (number of words in brackets)

| seen in training | rare in English (zipf < 3) | medium (3-4.5) | common (> 4.5) | all |
|---|---|---|---|---|
| never | - | - | 0.99 (97) | 0.99 (97) |
| 1-9 times | - | - | 1.00 (1) | 1.00 (1) |
| 10-29 times | 0.66 (35) | 0.86 (457) | 0.93 (15) | 0.85 (507) |
| 30+ times | - | 0.93 (239) | 0.96 (1462) | 0.96 (1701) |

Common, often-trained words it still got wrong: be (23), write (2), grade (1), want (1), someone (1), ask (1), eye (1), apple (1), survive (1), who (1), cause (1), tired (1), promise (1), study (1), put (1)
