# Test A: does the model still understand English passives? (`analysis/passive_qa.py`)

Forced choice between the doer and the other person, 200 per kind, chance 0.50. Predictions QA1-QA2 in results/findings_2026-10-24.md.

| model | passive | active | by decoy (active + place) |
|---|---|---|---|
| base | 0.90 [0.85, 0.93] | 0.78 [0.71, 0.83] | 0.75 [0.69, 0.80] |
| e2b-v2fix-passive0 | 0.97 [0.94, 0.99] | 0.34 [0.28, 0.41] | 0.28 [0.22, 0.35] |
| e2b-v2fix-cot-80k-noheld | 0.99 [0.97, 1.00] | 0.91 [0.86, 0.94] | 0.96 [0.93, 0.98] |

QA1 (the base model passes, so the test is fair): **fails**.
QA2 (passive0 on passives vs base, gap -0.07): **understanding kept (the failure is in writing the meaning format)** (but QA1 failed: not interpretable).
