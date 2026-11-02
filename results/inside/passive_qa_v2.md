# Test A v2: does the model still understand English passives? (`analysis/passive_qa.py`)

Each sentence asked two opposite questions (who did it / who was it done to); cells = share of sentences with BOTH answers right (200 sentences per kind; a position habit scores 0); in brackets the two questions separately. Predictions QA1v2-QA3v2 in results/findings_2026-10-24.md.

| model | passive | active | by decoy (active + place) |
|---|---|---|---|
| base | 0.35 [0.29, 0.42] (did 0.76 / was 0.59) | 0.56 [0.50, 0.63] (did 0.65 / was 0.92) | 0.47 [0.41, 0.54] (did 0.54 / was 0.94) |
| e2b-v2fix-passive0 | 0.12 [0.09, 0.18] (did 0.85 / was 0.27) | 0.07 [0.05, 0.12] (did 0.38 / was 0.65) | 0.04 [0.02, 0.08] (did 0.26 / was 0.70) |
| e2b-v2fix-cot-80k-noheld | 0.22 [0.17, 0.28] (did 0.95 / was 0.27) | 0.33 [0.27, 0.40] (did 0.77 / was 0.56) | 0.26 [0.20, 0.32] (did 0.67 / was 0.58) |

QA1v2 (the base model passes, so the test is fair): **fails**.
QA2v2 (passive0 on passives vs base, gap +0.23): **lost in fine-tuning** (but the fairness check failed: not interpretable).
QA3v2 (passive0 on actives vs base, gap +0.49): **it lost the question-answering more generally**.
