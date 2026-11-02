# Rare words in real English (Tatoeba, 600 sentences; `analysis/tatoeba_rare_words.py`)

Share of gold meaning word keys found in the predicted meaning, by how often the run's own training data contained the key (count in brackets). Last columns: of the sentences whose meaning is wrong, how many miss a key seen fewer than 25 times in training.

| run | seen never | seen 1-11 | seen 12-24 | seen 25-99 | seen 100+ | wrong sentences | ... missing a rare key |
|---|---|---|---|---|---|---|---|
| e2b-v2fix-cot-80k-noheld | 0.41 (217) | 0.91 (22) | 0.93 (129) | 0.95 (484) | 0.97 (1410) | 230 | 124 (54%) |
| e2b-v2fix-ts1 | 0.40 (217) | 0.91 (22) | 0.95 (129) | 0.96 (484) | 0.97 (1410) | 225 | 123 (55%) |
| e2b-v2fix-floor25 | 0.37 (217) | 0.86 (22) | - (0) | 0.95 (601) | 0.98 (1422) | 238 | 127 (53%) |
| e2b-v2fix-real15 | 0.65 (20) | 0.84 (45) | 0.97 (142) | 0.98 (515) | 1.00 (1540) | 46 | 18 (39%) |
| e2b-v2fix-real15-ts1 | 0.60 (20) | 0.84 (45) | 0.96 (142) | 0.98 (515) | 1.00 (1540) | 43 | 18 (42%) |
