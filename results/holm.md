# Holm correction across the claim-level tests (`analysis/holm.py`)

19 tests (family-wise error 0.05). A test survives if its p is below its Holm threshold and every smaller p survived. p-values are copied from the source files.

| claim | test | p | Holm threshold | survives | source |
|---|---|---|---|---|---|
| 4 | real text vs baseline, Tatoeba, seed 1 (McNemar) | 6.8e-50 | 0.0026 | yes | `results/significance.md` |
| 4 | real text vs baseline, Tatoeba, seed 0 (McNemar) | 1.8e-49 | 0.0028 | yes | `results/significance.md` |
| 1 | sound vs WRONG-sound start, v1 long sentences (McNemar) | 3.1e-42 | 0.0029 | yes | `results/significance_more.md` |
| 1 | sound vs WRONG-sound start, v1 morphology test (McNemar) | 7.4e-14 | 0.0031 | yes | `results/significance_more.md` |
| 4 | everyday vs baseline seed 0, Tatoeba (McNemar) | 9e-13 | 0.0033 | yes | `results/findings_2026-10-24.md` |
| 4 | everyday vs baseline seed 1, Tatoeba (McNemar) | 2.7e-11 | 0.0036 | yes | `results/findings_2026-10-24.md` |
| 3 | floor 25 vs baseline, generated sentences (McNemar) | 2.4e-05 | 0.0038 | yes | `results/significance.md` |
| 6 | 6 epochs vs 4, Tatoeba (McNemar) | 0.00017 | 0.0042 | yes | `results/significance.md` |
| 3 | floor 25 vs baseline seed 1, generated (McNemar) | 0.0013 | 0.0045 | yes | `results/significance.md` |
| 3 | new loans: letter-by-letter vs multi-letter rule (Fisher) | 0.005 | 0.0050 | yes | `results/findings_2026-10-24.md` |
| 4 | real text vs baseline, sealed, seed 1 (paired McNemar) | 0.013 | 0.0056 | no | `results/sealed_paired.md` |
| 6 | focal loss vs baseline, Tatoeba (McNemar) | 0.014 | 0.0063 | no | `results/significance.md` |
| 4 | real text vs baseline, sealed, seed 0 (paired McNemar) | 0.036 | 0.0071 | no | `results/sealed_paired.md` |
| 6 | 6 epochs vs 4, generated (McNemar) | 0.036 | 0.0083 | no | `results/significance.md` |
| 4 | everyday vs baseline seed 1, sealed (paired McNemar) | 0.039 | 0.0100 | no | `results/findings_2026-10-24.md` |
| 4 | real text vs baseline, sealed, seeds together (sign test) | 0.05 | 0.0125 | no | `results/sealed_paired.md` |
| 4 | everyday vs baseline seed 0, sealed (paired McNemar) | 0.18 | 0.0167 | no | `results/findings_2026-10-24.md` |
| 6 | majority vote vs greedy, Tatoeba (McNemar) | 0.51 | 0.0250 | no | `results/decode_tricks_*.md` |
| 6 | beam search vs greedy, Tatoeba (McNemar) | 1 | 0.0500 | no | `results/decode_tricks_*.md` |

Not tested here: 1: maths add 4+4, real-digit vs neutral start (2 seeds) (results/maths.md (no test stored)).
