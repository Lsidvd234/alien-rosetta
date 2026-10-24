# Softmax collapse check (`analysis/softmax_collapse.py`, measure in `train/collapse.py`)

Each model's own training examples (first 300, answer given), every answer token. **Zero-gradient share** = tokens whose softmax gives exactly 1.0 for the right token (loss and gradient exactly 0). float32 is what the training loss used; bf16 / fp16 show how close lower precision would be. Gap = right logit minus the best wrong logit (float32 collapse needs ~16+).

| run | answer tokens | mean p(right) | mean loss | gap median | gap 90th pct | zero-grad fp32 | near fp32 (gap>15) | zero-grad fp16 | zero-grad bf16 | logit norm | hidden norm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e2a-v2-80k | 21865 | 0.9847 | 0.0318 | 11.5 | 15.8 | **6.7%** | 13.6% | 75.8% | 87.8% | 1888 | 125 |
| e2b-v2-cot-80k-noheld | 34146 | 0.9894 | 0.0217 | 11.7 | 16.0 | **7.4%** | 14.7% | 78.3% | 89.6% | 2261 | 133 |
| e2b-v2fix-cot-80k-noheld | 34146 | 0.9942 | 0.0105 | 11.9 | 16.2 | **7.9%** | 16.6% | 79.6% | 90.6% | 2312 | 133 |
| e2b-v2fix-cot-80k | 33566 | 0.9945 | 0.0095 | 11.9 | 15.8 | **7.1%** | 14.0% | 80.9% | 91.4% | 2316 | 134 |
| e2b-v2fix-focal-r2 | 34146 | 0.9764 | 0.0291 | 6.3 | 9.1 | **0.0%** | 0.0% | 10.8% | 38.8% | 1910 | 131 |
| e2b-v2fix-loan-r2 | 31835 | 0.9940 | 0.0112 | 11.9 | 16.0 | **7.4%** | 15.6% | 80.8% | 91.3% | 2345 | 133 |
| e2b-v2fix-para | 34146 | 0.9940 | 0.0113 | 11.9 | 16.2 | **7.6%** | 16.6% | 79.3% | 90.7% | 2323 | 133 |
