# English-understanding ceiling (generated statements, 300 test sentences)

Facts per sentence from its known meaning. Word level = share of facts right; sentence level = all of a sentence's facts right. Chance: tense 0.33, negated 0.50, number 0.50; open questions ~0. `ceiling/ceiling.py`.

| method | word level | sentence level | doer | action | thing done to | place | tense | negated | number |
|---|---|---|---|---|---|---|---|---|---|
| method1-questions__Qwen/Qwen3-0.6B-Base | 0.649 | 0.103 | 0.775 | 0.800 | 0.489 | 0.895 | 0.430 | 0.587 | 0.756 |
| method1-questions__Qwen/Qwen3-0.6B | 0.725 | 0.223 | 0.720 | 0.850 | 0.435 | 0.842 | 0.560 | 0.803 | 0.796 |
| method2-format__Qwen/Qwen3-0.6B-Base | 0.450 | 0.013 | 0.257 | 0.460 | 0.326 | 0.158 | 0.303 | 0.617 | 0.701 |
| trained-meaning__e2b-v2fix-cot-80k-noheld | 0.987 | 0.943 | 0.954 | 0.987 | 0.957 | 1.000 | 1.000 | 1.000 | 0.995 |

## method3-probe__Qwen/Qwen3-0.6B-Base (accuracy of linear probes; chance {'word role (majority class)': 0.760752688172043, 'tense': 0.3333333333333333, 'negated': 0.5, 'number': 0.5})

| layer | word role (doer/action/object/place/other) | tense | negated | number |
|---|---|---|---|---|
| 7 | 0.886 | 0.993 | 1.000 | 0.960 |
| 14 | 0.894 | 0.967 | 1.000 | 0.930 |
| 21 | 0.884 | 0.987 | 1.000 | 0.940 |
| 28 | 0.872 | 0.937 | 0.993 | 0.945 |
