# Claim 4 from inside the model (read-out)

| run | feature | majority (real) | probe on real | model output on real | output wrong (n, readable) | majority where wrong | probe right where output wrong | output unreadable (n) | probe right where unreadable | probe right where output right | probe = the wrong output | layer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | tense | 0.61 | 0.97 | nan | - | - | - | - | - | - | - | mean:6 |
| base | NEG | 0.86 | 0.99 | nan | - | - | - | - | - | - | - | mean:3 |
| base | Q | 0.90 | 0.98 | nan | - | - | - | - | - | - | - | last:20 |
| base | PROG | 0.91 | 0.98 | nan | - | - | - | - | - | - | - | mean:18 |
| base | PERF | 0.93 | 0.95 | nan | - | - | - | - | - | - | - | mean:1 |
| base | modal | 0.88 | 0.98 | nan | - | - | - | - | - | - | - | mean:8 |
| base | subject | 0.22 | 0.83 | nan | - | - | - | - | - | - | - | mean:18 |
| e2b-v2fix-cot-80k-noheld | tense | 0.61 | 0.99 | 0.98 | 3 | 0.6667 | 0.3333 | 10 | 0.8 | 0.9966 | 0.6667 | mean:18 |
| e2b-v2fix-cot-80k-noheld | NEG | 0.86 | 0.99 | 0.98 | 0 | - | - | 10 | 1.0 | 0.9932 | - | mean:15 |
| e2b-v2fix-cot-80k-noheld | Q | 0.90 | 0.99 | 0.98 | 0 | - | - | 10 | 1.0 | 0.9881 | - | last:17 |
| e2b-v2fix-cot-80k-noheld | PROG | 0.91 | 0.98 | 0.97 | 8 | 0.625 | 0.375 | 10 | 1.0 | 0.9863 | 0.625 | last:18 |
| e2b-v2fix-cot-80k-noheld | PERF | 0.93 | 0.95 | 0.94 | 27 | 0.8519 | 0.3704 | 10 | 1.0 | 0.9805 | 0.6296 | mean:18 |
| e2b-v2fix-cot-80k-noheld | modal | 0.88 | 0.99 | 0.98 | 3 | 1.0 | 0.3333 | 10 | 1.0 | 0.9949 | 0.6667 | mean:19 |
| e2b-v2fix-cot-80k-noheld | subject | 0.22 | 0.95 | 0.96 | 12 | 0.5 | 0.25 | 10 | 0.4 | 0.9758 | 0.75 | last:21 |
| e2b-v2fix-real15-ts1 | tense | 0.61 | 0.99 | 0.99 | 3 | 0.6667 | 0.0 | 0 | - | 0.995 | - | mean:18 |
| e2b-v2fix-real15-ts1 | NEG | 0.86 | 1.00 | 1.00 | 0 | - | - | 0 | - | 0.9983 | - | last:13 |
| e2b-v2fix-real15-ts1 | Q | 0.90 | 1.00 | 1.00 | 0 | - | - | 0 | - | 0.9983 | - | last:17 |
| e2b-v2fix-real15-ts1 | PROG | 0.91 | 0.97 | 1.00 | 1 | 1.0 | 0.0 | 0 | - | 0.9683 | - | last:16 |
| e2b-v2fix-real15-ts1 | PERF | 0.93 | 0.96 | 1.00 | 1 | 1.0 | 0.0 | 0 | - | 0.9666 | - | mean:19 |
| e2b-v2fix-real15-ts1 | modal | 0.88 | 0.99 | 1.00 | 0 | - | - | 0 | - | 0.995 | - | mean:19 |
| e2b-v2fix-real15-ts1 | subject | 0.22 | 0.90 | 1.00 | 1 | 1.0 | 0.0 | 0 | - | 0.8965 | - | last:18 |
| e2b-v2fix-real15 | tense | 0.61 | 0.98 | 1.00 | 1 | 1.0 | 0.0 | 1 | 0.0 | 0.9849 | - | last:14 |
| e2b-v2fix-real15 | NEG | 0.86 | 1.00 | 1.00 | 0 | - | - | 1 | 1.0 | 0.9967 | - | mean:10 |
| e2b-v2fix-real15 | Q | 0.90 | 1.00 | 1.00 | 0 | - | - | 1 | 1.0 | 0.9983 | - | last:20 |
| e2b-v2fix-real15 | PROG | 0.91 | 0.99 | 0.99 | 2 | 1.0 | 0.0 | 1 | 1.0 | 0.9916 | - | mean:18 |
| e2b-v2fix-real15 | PERF | 0.93 | 0.97 | 1.00 | 1 | 1.0 | 0.0 | 1 | 1.0 | 0.9682 | - | mean:18 |
| e2b-v2fix-real15 | modal | 0.88 | 1.00 | 1.00 | 0 | - | - | 1 | 1.0 | 0.9967 | - | mean:17 |
| e2b-v2fix-real15 | subject | 0.22 | 0.92 | 0.99 | 4 | 1.0 | 0.25 | 1 | 0.0 | 0.9261 | - | last:21 |
| e2b-v2fix-ts1 | tense | 0.61 | 0.99 | 0.98 | 3 | 1.0 | 0.6667 | 6 | 0.3333 | 0.9966 | 0.3333 | last:21 |
| e2b-v2fix-ts1 | NEG | 0.86 | 0.99 | 0.99 | 0 | - | - | 6 | 1.0 | 0.9899 | - | mean:15 |
| e2b-v2fix-ts1 | Q | 0.90 | 0.98 | 0.98 | 5 | 1.0 | 0.2 | 6 | 1.0 | 0.983 | 0.8 | last:12 |
| e2b-v2fix-ts1 | PROG | 0.91 | 0.97 | 0.98 | 6 | 0.6667 | 0.3333 | 6 | 1.0 | 0.9762 | 0.6667 | last:18 |
| e2b-v2fix-ts1 | PERF | 0.93 | 0.98 | 0.94 | 28 | 0.8929 | 0.7143 | 6 | 1.0 | 0.9929 | 0.2857 | mean:9 |
| e2b-v2fix-ts1 | modal | 0.88 | 0.99 | 0.99 | 2 | 1.0 | 0.0 | 6 | 1.0 | 0.9966 | 1.0 | mean:19 |
| e2b-v2fix-ts1 | subject | 0.22 | 0.96 | 0.97 | 10 | 0.6 | 0.1 | 6 | 0.5 | 0.9829 | 0.9 | last:21 |
