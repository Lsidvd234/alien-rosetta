# Alien Rosetta: results of every run

Generated 2026-11-01 21:45 by `analysis/all_runs.py` (refresh: `python analysis/all_runs.py`). Sealed-set numbers are totals from `results/sealed_scores.md`; '-' = metric not measured for that run.

**Research question:** what does it take for a small LLM to learn a language it has never seen, and where does that learning stop?

**Claims:** (1) new symbols are learnable when they start from meaning; generalisation stops at training coverage of (2) sentence structure, (3) vocabulary and (4) English variety; (5) writing the meaning first speeds learning, not generalisation; (6) model tricks are not the fix.

## 1. Stage 1: model and size pick

Which base model to use: Qwen3-0.6B-Base (l06) vs Qwen3-1.7B-Base (l17) on the easy (Latin) and hard (Tamil script) languages. 1.7B was better but still failing at 20k x 2 epochs; 0.6B was kept, since capacity was not the bottleneck (PLAN.md, stage 1).

| run | settings | steps | exact iid | exact comp | exact morph | exact long | word acc iid (generated, 100) | final loss |
|---|---|---|---|---|---|---|---|---|
| l06-easy-latin | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=latin, n_train=20000, epochs=2.0 | 1250/1250 | 0.114 | 0.012 | 0.082 | 0.000 | 0.679 | 1.191 |
| l06-easy-latin-50k3 | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=latin, n_train=50000, epochs=3.0 | 4689/4689 | 0.960 | 0.964 | 0.964 | 0.856 | 0.998 | 0.014 |
| l06-hard-tamil | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_tamil, n_train=20000, epochs=2.0 | 1250/1250 | 0.010 | 0.000 | 0.012 | 0.000 | 0.106 | 0.545 |
| l17-easy-latin | model=Qwen/Qwen3-1.7B-Base, preset=english_like, script=latin, n_train=20000, epochs=2.0 | 1250/1250 | 0.306 | 0.122 | 0.270 | 0.002 | 0.843 | 0.548 |
| l17-hard-tamil | model=Qwen/Qwen3-1.7B-Base, preset=tamil_like, script=abugida_tamil, n_train=20000, epochs=2.0 | 1250/1250 | 0.036 | 0.000 | 0.022 | 0.000 | 0.198 | 0.413 |

## 2. Script x grammar grid (2x2, sizes and seeds)

{easy, hard} grammar x {Latin, Tamil script} at 5k/10k/20k/50k sentences; s1/s2 = seeds 1/2 at 20k (a new seed is a new random language with the same grammar). At 20k, content-word accuracy: easy 0.82-0.86, hard Latin 0.04, hard Tamil 0.24 (results/final_20k.md). Exact here is teacher-forced whole-sentence match on 500 items.

| run | settings | steps | exact iid | exact comp | exact morph | exact long | word acc iid (generated, 100) | final loss |
|---|---|---|---|---|---|---|---|---|
| c-easy-latin-10k | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=latin, n_train=10000, epochs=3.0 | 939/939 | 0.032 | 0.000 | 0.022 | 0.000 | 0.547 | 1.720 |
| c-easy-latin-20k | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=latin, n_train=20000, epochs=3.0 | 1875/1875 | 0.634 | 0.568 | 0.554 | 0.176 | 0.947 | 0.205 |
| c-easy-latin-5k | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=latin, n_train=5000, epochs=3.0 | 471/471 | 0.000 | 0.000 | 0.000 | 0.000 | 0.486 | 2.486 |
| c-easy-tamil-10k | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=abugida_tamil, n_train=10000, epochs=3.0 | 939/939 | 0.028 | 0.000 | 0.034 | 0.000 | 0.518 | 0.545 |
| c-easy-tamil-20k | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=abugida_tamil, n_train=20000, epochs=3.0 | 1875/1875 | 0.650 | 0.598 | 0.596 | 0.188 | 0.932 | 0.047 |
| c-easy-tamil-5k | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=abugida_tamil, n_train=5000, epochs=3.0 | 471/471 | 0.000 | 0.000 | 0.000 | 0.000 | 0.486 | 0.783 |
| c-hard-latin-10k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=latin, n_train=10000, epochs=3.0 | 939/939 | 0.000 | 0.000 | 0.000 | 0.000 | 0.083 | 1.742 |
| c-hard-latin-20k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=latin, n_train=20000, epochs=3.0 | 1875/1875 | 0.022 | 0.000 | 0.010 | 0.000 | 0.139 | 0.910 |
| c-hard-latin-50k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=latin, n_train=50000, epochs=3.0 | 4689/4689 | 0.960 | 0.952 | 0.836 | 0.940 | 0.997 | 0.007 |
| c-hard-latin-5k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=latin, n_train=5000, epochs=3.0 | 471/471 | 0.000 | 0.000 | 0.000 | 0.000 | 0.083 | 2.202 |
| c-hard-tamil-10k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_tamil, n_train=10000, epochs=3.0 | 939/939 | 0.002 | 0.000 | 0.002 | 0.000 | 0.094 | 0.670 |
| c-hard-tamil-20k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_tamil, n_train=20000, epochs=3.0 | 1875/1875 | 0.100 | 0.002 | 0.072 | 0.000 | 0.386 | 0.264 |
| c-hard-tamil-5k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_tamil, n_train=5000, epochs=3.0 | 471/471 | 0.002 | 0.000 | 0.000 | 0.000 | 0.083 | 0.911 |
| g-easy-tamil-50k | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=abugida_tamil, n_train=50000, epochs=3.0 | 4689/4689 | 0.994 | 0.988 | 0.984 | 0.992 | 0.998 | 0.001 |
| g-hard-tamil-50k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_tamil, n_train=50000, epochs=3.0 | 4689/4689 | 0.986 | 0.976 | 0.920 | 0.958 | 1.000 | 0.002 |
| s1-easy-latin-20k | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=latin, n_train=20000, epochs=3.0, target_tok=bpe | 1875/1875 | 0.680 | 0.624 | 0.630 | 0.258 | 0.906 | 0.171 |
| s1-easy-tamil-20k | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=abugida_tamil, n_train=20000, epochs=3.0, target_tok=bpe | 1875/1875 | 0.750 | 0.682 | 0.680 | 0.322 | 0.944 | 0.040 |
| s1-hard-latin-20k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=latin, n_train=20000, epochs=3.0, target_tok=bpe | 1875/1875 | 0.004 | 0.000 | 0.002 | 0.000 | 0.124 | 0.996 |
| s1-hard-tamil-20k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_tamil, n_train=20000, epochs=3.0, target_tok=bpe | 1875/1875 | 0.120 | 0.000 | 0.082 | 0.000 | 0.405 | 0.292 |
| s2-easy-latin-20k | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=latin, n_train=20000, epochs=3.0, target_tok=bpe | 1875/1875 | 0.652 | 0.510 | 0.530 | 0.166 | 0.919 | 0.212 |
| s2-easy-tamil-20k | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=abugida_tamil, n_train=20000, epochs=3.0, target_tok=bpe | 1875/1875 | 0.766 | 0.686 | 0.676 | 0.294 | 0.953 | 0.038 |
| s2-hard-latin-20k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=latin, n_train=20000, epochs=3.0, target_tok=bpe | 1875/1875 | 0.000 | 0.000 | 0.000 | 0.000 | 0.157 | 0.971 |
| s2-hard-tamil-20k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_tamil, n_train=20000, epochs=3.0, target_tok=bpe | 1875/1875 | 0.130 | 0.000 | 0.074 | 0.000 | 0.371 | 0.291 |

## 3. Cause tests: tokenisation and invented glyphs

Is it token count or familiar characters? m2/m3 = one character per token (Latin / accented Latin); p-*-pua = invented glyphs split into byte pieces (collapse at 20k, recover slowly with more data); p-*-puatok = each glyph one new token. Byte pieces, not token count, cause the collapse; whole-glyph tokens fix the easy language fully but the hard one only partly (results/final_20k.md, PLAN.md).

| run | settings | steps | exact iid | exact comp | exact morph | exact long | word acc iid (generated, 100) | final loss |
|---|---|---|---|---|---|---|---|---|
| m2-easy-latinchar-20k | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=latin, n_train=20000, epochs=3.0, target_tok=char | 1875/1875 | 0.698 | 0.646 | 0.676 | 0.238 | 0.856 | 0.044 |
| m2-hard-latinchar-20k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=latin, n_train=20000, epochs=3.0, target_tok=char | 1875/1875 | 0.054 | 0.000 | 0.032 | 0.000 | 0.262 | 0.317 |
| m3-hard-accentchar-20k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=latin_accent, n_train=20000, epochs=3.0, target_tok=char | 1875/1875 | 0.010 | 0.000 | 0.016 | 0.000 | 0.115 | 0.622 |
| p-easy-pua-20k | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=abugida_pua, n_train=20000, epochs=3.0, target_tok=bpe | 1875/1875 | 0.002 | 0.000 | 0.002 | 0.000 | 0.494 | 0.286 |
| p-easy-pua-50k | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=abugida_pua, n_train=50000, epochs=3.0, target_tok=bpe | 4689/4689 | 0.978 | 0.982 | 0.950 | 0.962 | 0.998 | 0.001 |
| p-easy-pua-80k | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=abugida_pua, n_train=80000, epochs=3.0, target_tok=bpe | 7500/7500 | 0.994 | 0.992 | 0.990 | 0.988 | 1.000 | 0.000 |
| p-easy-puatok-20k | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=abugida_pua, n_train=20000, epochs=3.0, target_tok=bpe, add_tokens | 1875/1875 | 0.000 | 0.000 | 0.000 | 0.000 | 0.488 | 0.774 |
| p-easy-puatok-50k | model=Qwen/Qwen3-0.6B-Base, preset=english_like, script=abugida_pua, n_train=50000, epochs=3.0, target_tok=bpe, add_tokens, new_tok_init=copy | 4689/4689 | 0.990 | 0.984 | 0.958 | 0.946 | 0.998 | 0.003 |
| p-hard-pua-20k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_pua, n_train=20000, epochs=3.0, target_tok=bpe | 1875/1875 | 0.000 | 0.000 | 0.000 | 0.000 | 0.083 | 0.304 |
| p-hard-pua-35k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_pua, n_train=35000, epochs=3.0, target_tok=bpe | 3282/3282 | 0.008 | 0.000 | 0.000 | 0.000 | 0.525 | 0.128 |
| p-hard-pua-50k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_pua, n_train=50000, epochs=3.0, target_tok=bpe | 4689/4689 | 0.376 | 0.144 | 0.274 | 0.010 | 0.755 | 0.033 |
| p-hard-pua-80k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_pua, n_train=80000, epochs=3.0, target_tok=bpe | 7500/7500 | 0.990 | 0.968 | 0.464 | 0.048 | 1.000 | 0.000 |
| p-hard-puatok-20k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_pua, n_train=20000, epochs=3.0, target_tok=bpe, add_tokens | 1875/1875 | 0.000 | 0.000 | 0.000 | 0.000 | 0.112 | 0.820 |
| p-hard-puatok-50k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_pua, n_train=50000, epochs=3.0, target_tok=bpe, add_tokens, new_tok_init=copy | 4689/4689 | 0.570 | 0.506 | 0.420 | 0.000 | 0.876 | 0.084 |

## 4. Glyph starting points (claim 1, stage 1)

Hard language, glyphs as one token each, 50k, baseline = p-hard-puatok-50k (exact 0.570/0.506/0.420/0.000). g1 starts each glyph at the embedding of its sound (fixes it fully), g2 adds transliteration examples, g3 trains on plain glyph text first (worse); s1-s3 are the same runs re-trained with saved weights, s4 starts every glyph at a WRONG sound (control).

| run | settings | steps | exact iid | exact comp | exact morph | exact long | word acc iid (generated, 100) | final loss |
|---|---|---|---|---|---|---|---|---|
| g1-hard-latininit-50k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_pua, n_train=50000, epochs=3.0, target_tok=bpe, add_tokens, new_tok_init=latin | 4689/4689 | 0.990 | 0.986 | 0.972 | 0.974 | 1.000 | 0.002 |
| g1-own-ts1 | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_pua, n_train=50000, epochs=3.0, target_tok=bpe, add_tokens, new_tok_init=latin | 4689/4689 | 0.990 | 0.992 | 0.974 | 0.964 | 1.000 | 0.000 |
| g1-own-ts2 | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_pua, n_train=50000, epochs=3.0, target_tok=bpe, add_tokens, new_tok_init=latin | 4689/4689 | 0.990 | 0.978 | 0.882 | 0.960 | 1.000 | 0.001 |
| g2-hard-translit-50k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_pua, n_train=50000, epochs=3.0, target_tok=bpe, add_tokens, new_tok_init=copy, aux_translit | 9375/9375 | 0.990 | 0.976 | 0.896 | 0.894 | 0.997 | 0.002 |
| g3-hard-plaintext-50k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_pua, n_train=50000, epochs=3.0, target_tok=bpe, add_tokens, new_tok_init=copy, mono_n=50000 | 4689/4689 | 0.320 | 0.002 | 0.266 | 0.002 | 0.776 | 0.145 |
| s1-hard-latininit-50k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_pua, n_train=50000, epochs=3.0, target_tok=bpe, add_tokens, new_tok_init=latin | 4689/4689 | 0.990 | 0.986 | 0.972 | 0.974 | 1.000 | 0.002 |
| s2-hard-translit-50k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_pua, n_train=50000, epochs=3.0, target_tok=bpe, add_tokens, new_tok_init=copy, aux_translit | 9375/9375 | 0.990 | 0.976 | 0.896 | 0.894 | 0.997 | 0.002 |
| s3-hard-plaintext-50k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_pua, n_train=50000, epochs=3.0, target_tok=bpe, add_tokens, new_tok_init=copy, mono_n=50000 | 4689/4689 | 0.320 | 0.002 | 0.266 | 0.002 | 0.776 | 0.145 |
| s4-hard-shuffledinit-50k | model=Qwen/Qwen3-0.6B-Base, preset=tamil_like, script=abugida_pua, n_train=50000, epochs=3.0, target_tok=bpe, add_tokens, new_tok_init=latin_shuffled | 4689/4689 | 0.982 | 0.978 | 0.860 | 0.650 | 0.997 | 0.002 |

## 5. R1 / E1: robustness and the reasoning step (v1 language)

R1 = varied English for robustness; E1a = direct translation, E1b = writes the meaning analysis first (the reasoning step). At 3 epochs E1b looked far better (0.96 vs 0.26 exact) but at 5 epochs both reach 0.97-0.99: the analysis step speeds learning, it does not generalise further. Unseen structures stay near 0 for all (passive 0.00-0.02, clefts 0.01-0.09) (results/e1_5ep.md, e1_comparison.md).

| run | settings | steps | exact orig iid | analysis exact orig iid | exact in-dist mix | held-out structures, mean exact | held-out passive exact |
|---|---|---|---|---|---|---|---|
| e1a-hard-rep-50k | epochs=3.0, rep | 3330/3330 | 0.256 | - | 0.300 | 0.151 | 0.070 |
| e1a-hard-rep-5ep | epochs=5.0, rep | 5550/5550 | 0.976 | - | 0.760 | 0.254 | 0.000 |
| e1b-hard-rep-cot-50k | epochs=3.0, cot, rep | 3330/3330 | 0.958 | 0.984 | 0.760 | 0.312 | 0.120 |
| e1b-hard-rep-cot-5ep | epochs=5.0, cot, rep | 5550/5550 | 0.988 | 0.992 | 0.800 | 0.286 | 0.010 |
| e1b-hard-rep-cot-5ep-seed1 | epochs=5.0, cot, rep, seed=1 | 5550/5550 | 0.974 | 0.980 | 0.780 | 0.296 | 0.020 |
| r1-hard-aug-50k | epochs=3.0, aug | 5766/5766 | - | - | - | - | - |

## 6. v2 language: E2a / E2b

The full language v2 (11k roots, built words, new sentence kinds), 80k examples. E2a direct, E2b analysis first; 'noheld' = no English variants held out; 'v2fix' = every word seen at least 12 times (word floor). e2b-v2fix-cot-80k-noheld is the baseline for all extensions.

| run | settings | steps | gen exact (100) | gen analysis exact (100) | gen sentence exact (300) | Tatoeba meaning | Tatoeba glyphs | sealed glyphs | sealed meaning | sealed word acc | sealed refused (out of scope) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e2a-v2-80k | epochs=4.0, rep_n=80000, rep_held_frac=0.2 | 11064/11064 | 0.470 | - | 0.390 | - | 0.438 | 0.401 | - | 0.711 | 0.840 |
| e2b-v2-cot-80k | epochs=4.0, cot, rep_n=80000, rep_held_frac=0.2 | 11064/11064 | 0.510 | 0.780 | 0.417 | 0.622 | 0.458 | 0.408 | 0.415 | 0.723 | 0.920 |
| e2b-v2-cot-80k-noheld | epochs=4.0, cot, rep_n=80000 | 11064/11064 | 0.550 | 0.860 | 0.477 | 0.620 | 0.462 | 0.422 | 0.463 | 0.733 | 0.840 |
| e2b-v2fix-cot-80k | epochs=4.0, cot, rep_n=80000, rep_held_frac=0.2, rep_min_count=12 | 11064/11064 | 0.610 | 0.790 | 0.493 | 0.633 | 0.503 | 0.449 | 0.469 | 0.745 | 0.920 |
| e2b-v2fix-cot-80k-noheld | epochs=4.0, cot, rep_n=80000, rep_min_count=12 | 11064/11064 | 0.610 | 0.820 | 0.607 | 0.617 | 0.467 | 0.422 | 0.476 | 0.749 | 0.720 |

## 7. Extensions on the v2 baseline (claims 3, 4, 6)

One change each vs the baseline: focal loss, loanwords (0.04), paraphrased English (para, para8), real Tatoeba sentences (real15), training-seed replicates (ts1, real15-ts1), 6 epochs, word floor 25. para-s1 (--seed 1 = a different language) is excluded: results/excluded/README.md. Real text is the clear win; focal, loans, paraphrases and 6 epochs do not help on real English (results/findings_2026-10-24.md).

| run | settings | steps | gen exact (100) | gen analysis exact (100) | gen sentence exact (300) | Tatoeba meaning | Tatoeba glyphs | sealed glyphs | sealed meaning | sealed word acc | sealed refused (out of scope) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e2b-v2fix-6ep | epochs=6.0, cot, rep_n=80000, rep_min_count=12 | 16596/16596 | 0.600 | 0.820 | 0.660 | 0.563 | 0.450 | 0.395 | 0.449 | 0.729 | 0.920 |
| e2b-v2fix-everyday | epochs=4.0, cot, rep_n=80000, rep_min_count=12 | 11064/11064 | 0.670 | 0.880 | - | 0.725 | 0.605 | 0.476 | 0.517 | 0.769 | 0.920 |
| e2b-v2fix-floor25 | epochs=4.0, cot, rep_n=80000, rep_min_count=25 | 13304/13304 | 0.660 | 0.860 | 0.720 | 0.603 | 0.483 | 0.456 | 0.503 | 0.738 | 0.920 |
| e2b-v2fix-focal | pending | - | - | - | - | - | - | - | - | - | - |
| e2b-v2fix-focal-r2 | epochs=4.0, cot, rep_n=80000, rep_min_count=12, loss=focal, focal_gamma=2.0 | 11064/11064 | 0.560 | 0.800 | 0.567 | 0.588 | 0.455 | 0.388 | 0.442 | 0.725 | 0.840 |
| e2b-v2fix-loan | pending | - | - | - | - | - | - | - | - | - | - |
| e2b-v2fix-loan-r2 | epochs=4.0, cot, rep_n=80000, rep_min_count=12, rep_loan_rate=0.04 | 11064/11064 | 0.730 | 0.880 | 0.633 | 0.592 | 0.502 | 0.429 | 0.442 | 0.733 | 0.760 |
| e2b-v2fix-loan12 | epochs=4.0, cot, rep_n=80000, rep_min_count=12, rep_loan_rate=0.12 | 11064/11064 | 0.620 | 0.780 | - | 0.612 | 0.477 | - | - | - | - |
| e2b-v2fix-n98k | epochs=4.0, cot, rep_n=97905, rep_min_count=12 | 13304/13304 | 0.630 | 0.840 | - | 0.610 | 0.488 | - | - | - | - |
| e2b-v2fix-para | epochs=4.0, cot, rep_n=80000, rep_min_count=12, rep_paraphrases=/kaggle/working/paraphrases_seed0.json | 11064/11064 | 0.560 | 0.810 | 0.633 | 0.615 | 0.477 | 0.429 | 0.449 | 0.737 | 0.920 |
| e2b-v2fix-para8 | epochs=4.0, cot, rep_n=80000, rep_min_count=12, rep_paraphrases=train/data/paraphrases_v2_seed0.json | 11064/11064 | 0.620 | 0.830 | 0.617 | 0.620 | 0.480 | 0.408 | 0.469 | 0.733 | 0.880 |
| e2b-v2fix-passive0 | epochs=4.0, cot, rep_n=80000, rep_min_count=12 | 11064/11064 | 0.560 | 0.800 | - | 0.603 | 0.478 | 0.449 | 0.476 | 0.740 | 0.840 |
| e2b-v2fix-passive0-meaning | epochs=4.0, cot, rep_n=80000, rep_min_count=12 | 11064/11064 | 0.000 | 0.850 | - | 0.603 | 0.000 | - | - | - | - |
| e2b-v2fix-passive0-ts1 | epochs=4.0, cot, rep_n=80000, rep_min_count=12, train_seed=1 | 11064/11064 | 0.620 | 0.820 | - | 0.597 | 0.468 | - | - | - | - |
| e2b-v2fix-passive0-writer | epochs=4.0, cot, rep_n=80000, rep_min_count=12 | 11064/11064 | 0.810 | - | - | - | - | - | - | - | - |
| e2b-v2fix-passive300 | epochs=4.0, cot, rep_n=80000, rep_min_count=12 | 11064/11064 | 0.600 | 0.860 | - | 0.612 | 0.475 | 0.388 | 0.435 | 0.725 | 0.880 |
| e2b-v2fix-passive50 | epochs=4.0, cot, rep_n=80000, rep_min_count=12 | 11064/11064 | 0.530 | 0.810 | - | 0.607 | 0.437 | 0.388 | 0.435 | 0.724 | 1.000 |
| e2b-v2fix-passive50-ts1 | epochs=4.0, cot, rep_n=80000, rep_min_count=12, train_seed=1 | 11064/11064 | 0.600 | 0.860 | - | 0.607 | 0.473 | - | - | - | - |
| e2b-v2fix-q17 | epochs=2.0, cot, rep_n=80000, rep_min_count=12 | 5532/5532 | 0.400 | 0.870 | - | 0.625 | 0.428 | - | - | - | - |
| e2b-v2fix-real15 | epochs=4.0, cot, rep_n=80000, rep_min_count=12, rep_real=train/data/tatoeba_train.json | 11064/11064 | 0.640 | 0.840 | 0.587 | 0.923 | 0.890 | 0.558 | 0.558 | 0.796 | 0.960 |
| e2b-v2fix-real15-ts1 | epochs=4.0, cot, rep_n=80000, rep_min_count=12, rep_real=train/data/tatoeba_train.json, train_seed=1 | 11064/11064 | 0.560 | 0.810 | 0.610 | 0.928 | 0.892 | 0.551 | 0.558 | 0.795 | 0.760 |
| e2b-v2fix-soundrule | epochs=4.0, cot, rep_n=80000, rep_min_count=12 | 11064/11064 | 0.710 | 0.810 | - | 0.577 | 0.500 | - | - | - | - |
| e2b-v2fix-ts1 | epochs=4.0, cot, rep_n=80000, rep_min_count=12, train_seed=1 | 11064/11064 | 0.540 | 0.860 | 0.627 | 0.625 | 0.467 | 0.435 | 0.463 | 0.751 | 0.960 |

## 8. Maths in glyph notation (claims 1 and 2)

Base-7 arithmetic in glyph digits and operators; trained lengths add <= 3+3, mul <= 2x2. Starts: meaning (real digits), shuffled (wrong digits), mean (none); partial starts; longer training lengths; dropped digit pairs (no training multiplication uses that single-digit product); scratchpad (long-multiplication working before the answer). Exact match on 300 items per test (full tables: results/maths.md).

| run | settings | final loss | add <=3+3 | add 4+4 | add 5+5 | mul <=2x2 | mul 3x1 | mul 3x2 | mul 3x3 | other tests |
|---|---|---|---|---|---|---|---|---|---|---|
| maths-digitsonly | init=digits_only, epochs=2.0, add_max=3, mul_max=2x2 | 0.019 | 0.967 | 0.870 | 0.463 | 0.977 | 0.637 | 0.080 | 0.013 | - |
| maths-mean | init=mean, epochs=2.0 | 0.148 | 0.873 | 0.020 | 0.000 | 0.623 | 0.320 | 0.020 | 0.000 | - |
| maths-mean-drop56-bridge | init=mean, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6 | 0.133 | 0.903 | 0.007 | 0.000 | 0.630 | 0.293 | 0.003 | 0.000 | needs pair 5x6 0.023; avoids pair 5x6 0.603 |
| maths-mean-longer | init=mean, epochs=2.0, add_max=4, mul_max=3x2 | 0.198 | 0.883 | 0.423 | 0.073 | 0.693 | 0.633 | 0.133 | 0.000 | 6+6 0.010; 4x2 0.030; 4x3 0.000 |
| maths-mean-s1 | init=mean, seed=1, epochs=2.0, add_max=3, mul_max=2x2 | 0.145 | 0.897 | 0.153 | 0.017 | 0.640 | 0.323 | 0.007 | 0.000 | - |
| maths-meaning | init=meaning, epochs=2.0 | 0.019 | 0.977 | 0.887 | 0.723 | 0.957 | 0.653 | 0.130 | 0.027 | - |
| maths-meaning-build-drop56 | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6, scratch | 0.012 | 0.987 | 0.923 | 0.767 | 0.913 | 0.140 | 0.000 | 0.000 | needs pair 5x6 0.080; avoids pair 5x6 1.000 |
| maths-meaning-build-drop56-role100 | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6, scratch | 0.015 | 0.990 | 0.910 | 0.750 | 0.907 | 0.127 | 0.000 | 0.000 | needs pair 5x6 0.483; avoids pair 5x6 1.000 |
| maths-meaning-build-drop56-role25 | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6, scratch | 0.014 | 0.990 | 0.903 | 0.743 | 0.920 | 0.153 | 0.003 | 0.000 | needs pair 5x6 0.507; avoids pair 5x6 1.000 |
| maths-meaning-build-drop56-role400 | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6, scratch | 0.015 | 0.993 | 0.903 | 0.773 | 0.940 | 0.147 | 0.000 | 0.000 | needs pair 5x6 0.707; avoids pair 5x6 0.997 |
| maths-meaning-build-drop56-role400-s1 | init=meaning, seed=1, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6, scratch | 0.014 | 0.983 | 0.843 | 0.680 | 0.913 | 0.157 | 0.000 | 0.000 | needs pair 5x6 0.467; avoids pair 5x6 0.990 |
| maths-meaning-build-drop56-role5 | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6, scratch | 0.015 | 0.983 | 0.933 | 0.777 | 0.930 | 0.143 | 0.000 | 0.000 | needs pair 5x6 0.480; avoids pair 5x6 1.000 |
| maths-meaning-build-drop56-role593 | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6, scratch | 0.016 | 0.987 | 0.923 | 0.733 | 0.940 | 0.157 | 0.000 | 0.000 | needs pair 5x6 0.527; avoids pair 5x6 1.000 |
| maths-meaning-buildadd-drop6row | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=6,4;6,5;6,6, scratch | 0.019 | 0.977 | 0.887 | 0.690 | 0.883 | 0.167 | 0.000 | 0.000 | needs pair 6x4 0.613; needs pair 6x5 0.335; needs pair 6x6 0.971; avoids every dropped pair 0.980 |
| maths-meaning-buildadd-drop6row-rep | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=6,4;6,5;6,6, scratch | 0.017 | 0.980 | 0.890 | 0.633 | 0.820 | 0.160 | 0.000 | 0.000 | needs pair 6x4 0.430; needs pair 6x5 0.227; needs pair 6x6 0.671; avoids every dropped pair 0.973 |
| maths-meaning-buildadd-drop6row-s1 | init=meaning, seed=1, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=6,4;6,5;6,6, scratch | 0.011 | 0.993 | 0.833 | 0.547 | 0.880 | 0.190 | 0.003 | 0.000 | needs pair 6x4 0.553; needs pair 6x5 0.258; needs pair 6x6 0.681; avoids every dropped pair 0.973 |
| maths-meaning-buildmix-drop34 | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=3,4, scratch | 0.017 | 0.990 | 0.920 | 0.697 | 0.920 | 0.150 | 0.000 | 0.000 | needs pair 3x4 0.977; avoids pair 3x4 0.987 |
| maths-meaning-buildmix-drop56 | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6, scratch | 0.014 | 0.987 | 0.717 | 0.350 | 0.940 | 0.137 | 0.000 | 0.000 | needs pair 5x6 0.767; avoids pair 5x6 0.980 |
| maths-meaning-buildmix-drop56-s1 | init=meaning, seed=1, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6, scratch | 0.013 | 0.990 | 0.877 | 0.727 | 0.917 | 0.183 | 0.000 | 0.000 | needs pair 5x6 0.720; avoids pair 5x6 0.987 |
| maths-meaning-buildmix-drop56-s1-fo | init=meaning, seed=1, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6, scratch | 0.014 | 0.993 | 0.867 | 0.720 | 0.930 | 0.190 | 0.000 | 0.000 | needs pair 5x6 0.800; avoids pair 5x6 0.983 |
| maths-meaning-buildmix-drop66 | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=6,6, scratch | 0.017 | 0.983 | 0.840 | 0.677 | 0.917 | 0.273 | 0.017 | 0.000 | needs pair 6x6 0.635; avoids pair 6x6 0.970 |
| maths-meaning-buildmix-drop66-fo | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=6,6, scratch | 0.011 | 0.983 | 0.853 | 0.637 | 0.883 | 0.210 | 0.017 | 0.000 | needs pair 6x6 0.865; avoids pair 6x6 0.970 |
| maths-meaning-buildmix-drop6row | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=6,4;6,5;6,6, scratch | 0.022 | 0.980 | 0.937 | 0.803 | 0.733 | 0.127 | 0.000 | 0.000 | needs pair 6x4 0.430; needs pair 6x5 0.287; needs pair 6x6 0.486; avoids every dropped pair 0.957 |
| maths-meaning-buildmix-drop6row-fo | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=6,4;6,5;6,6, scratch | 0.019 | 0.980 | 0.937 | 0.773 | 0.707 | 0.143 | 0.000 | 0.000 | needs pair 6x4 0.430; needs pair 6x5 0.179; needs pair 6x6 0.414; avoids every dropped pair 0.983 |
| maths-meaning-double-drop6row | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=6,4;6,5;6,6, scratch | 0.021 | 0.983 | 0.833 | 0.520 | 0.740 | 0.177 | 0.007 | 0.000 | needs pair 6x4 0.650; needs pair 6x5 0.016; needs pair 6x6 0.000; avoids every dropped pair 0.990 |
| maths-meaning-drop22 | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=2,2 | 0.022 | 0.987 | 0.813 | 0.433 | 0.943 | 0.670 | 0.173 | 0.017 | needs pair 2x2 0.408; avoids pair 2x2 0.913 |
| maths-meaning-drop34 | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=3,4 | 0.023 | 0.997 | 0.920 | 0.747 | 0.840 | 0.550 | 0.133 | 0.020 | needs pair 3x4 0.103; avoids pair 3x4 0.957 |
| maths-meaning-drop34-tw | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=3,4 | 0.023 | 0.997 | 0.920 | 0.747 | 0.840 | 0.550 | 0.133 | 0.020 | needs pair 3x4 0.103; avoids pair 3x4 0.957 |
| maths-meaning-drop56 | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6 | 0.020 | 0.990 | 0.903 | 0.690 | 0.833 | 0.627 | 0.143 | 0.013 | needs pair 5x6 0.047; avoids pair 5x6 0.923 |
| maths-meaning-drop56-bridge | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6 | 0.020 | 0.990 | 0.903 | 0.690 | 0.833 | 0.627 | 0.143 | 0.013 | needs pair 5x6 0.047; avoids pair 5x6 0.923 |
| maths-meaning-drop56-bridge2 | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6 | 0.040 | 0.987 | 0.907 | 0.650 | 0.870 | 0.673 | 0.117 | 0.017 | needs pair 5x6 0.030; avoids pair 5x6 0.940 |
| maths-meaning-drop56-tw | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6 | 0.020 | 0.990 | 0.903 | 0.690 | 0.833 | 0.627 | 0.143 | 0.013 | needs pair 5x6 0.047; avoids pair 5x6 0.923 |
| maths-meaning-longer | init=meaning, epochs=2.0, add_max=4, mul_max=3x2 | 0.040 | 0.983 | 0.947 | 0.860 | 0.943 | 0.910 | 0.327 | 0.077 | 6+6 0.760; 4x2 0.193; 4x3 0.027 |
| maths-meaning-s1 | init=meaning, seed=1, epochs=2.0, add_max=3, mul_max=2x2 | 0.017 | 0.993 | 0.840 | 0.520 | 0.943 | 0.690 | 0.163 | 0.023 | - |
| maths-meaning-scratch | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, scratch | 0.021 | 0.977 | 0.860 | 0.413 | 1.000 | 0.137 | 0.000 | 0.000 | - |
| maths-meaning-scratch-drop56 | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6, scratch | 0.018 | 0.980 | 0.930 | 0.770 | 0.933 | 0.140 | 0.000 | 0.000 | needs pair 5x6 0.263; avoids pair 5x6 1.000 |
| maths-meaning-scratch-drop56-tw | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6, scratch | 0.018 | 0.980 | 0.930 | 0.770 | 0.933 | 0.140 | 0.000 | 0.000 | needs pair 5x6 0.263; avoids pair 5x6 1.000 |
| maths-meaning-scratch-drop56-vote | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6, scratch | 0.018 | 0.980 | 0.930 | 0.770 | 0.933 | 0.140 | 0.000 | 0.000 | needs pair 5x6 0.263; avoids pair 5x6 1.000 |
| maths-meaning-scratch-drop65ordered | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, scratch | 0.021 | 0.977 | 0.873 | 0.770 | 0.917 | 0.137 | 0.000 | 0.000 | - |
| maths-meaning-scratch-drop6row | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=6,4;6,5;6,6, scratch | 0.021 | 0.980 | 0.927 | 0.697 | 0.720 | 0.170 | 0.023 | 0.000 | needs pair 6x4 0.000; needs pair 6x5 0.139; needs pair 6x6 0.000; avoids every dropped pair 0.993 |
| maths-meaning-scratch-longer | init=meaning, epochs=2.0, add_max=4, mul_max=3x2, scratch | 0.019 | 0.987 | 0.957 | 0.860 | 0.993 | 1.000 | 0.983 | 0.000 | 6+6 0.597; 4x2 0.010; 4x3 0.000 |
| maths-meaning-split-drop6row | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=6,4;6,5;6,6, scratch | 0.018 | 0.980 | 0.877 | 0.620 | 0.770 | 0.140 | 0.000 | 0.000 | needs pair 6x4 0.587; needs pair 6x5 0.331; needs pair 6x6 0.271; avoids every dropped pair 0.983 |
| maths-meaning-trick-drop6row | init=meaning, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=6,4;6,5;6,6, scratch | 0.012 | 0.987 | 0.917 | 0.710 | 0.767 | 0.130 | 0.000 | 0.000 | needs pair 6x4 0.037; needs pair 6x5 0.227; needs pair 6x6 0.000; avoids every dropped pair 1.000 |
| maths-opsonly | init=ops_only, epochs=2.0, add_max=3, mul_max=2x2 | 0.125 | 0.883 | 0.063 | 0.000 | 0.660 | 0.320 | 0.013 | 0.000 | - |
| maths-shuffled | init=shuffled, epochs=2.0 | 0.085 | 0.860 | 0.083 | 0.020 | 0.690 | 0.363 | 0.000 | 0.000 | - |
| maths-shuffled-drop56-bridge | init=shuffled, epochs=2.0, add_max=3, mul_max=2x2, drop_pair=5,6 | 0.073 | 0.870 | 0.043 | 0.013 | 0.590 | 0.317 | 0.000 | 0.000 | needs pair 5x6 0.007; avoids pair 5x6 0.580 |
| maths-shuffled-s0-perms1 | init=shuffled, epochs=2.0, add_max=3, mul_max=2x2 | 0.064 | 0.973 | 0.580 | 0.160 | 0.863 | 0.490 | 0.023 | 0.003 | - |
| maths-shuffled-s1 | init=shuffled, seed=1, epochs=2.0, add_max=3, mul_max=2x2 | 0.081 | 0.967 | 0.600 | 0.180 | 0.767 | 0.403 | 0.033 | 0.000 | - |
| maths-shuffled-s1-perms0 | init=shuffled, seed=1, epochs=2.0, add_max=3, mul_max=2x2 | 0.109 | 0.893 | 0.053 | 0.010 | 0.677 | 0.330 | 0.017 | 0.000 | - |

## 9. Grokking (side study)

Long training with weight decay 1.0 to see whether test accuracy jumps late. One row per chain (chunks merged). Translation chunk c2 resumed from a stale checkpoint (step 4000) and was replaced by the r8k rerun from step 8000. Curves: results/grok/*.png.

| chain | chunks | steps done/total | train | test | extra | logit norm | note |
|---|---|---|---|---|---|---|---|
| grok-maths-frac06-r30k | grok-maths-frac06-r30k-c0 | 60000/60000 | 0.987 | 0.837 (new pairs, same sizes) | 0.090 (3-digit (longer)) | 5602.900 | - |
| grok-maths-frac06 | grok-maths-frac06-c0, grok-maths-frac06-c1, grok-maths-frac06-r20k-c0, grok-maths-frac06-r30k-c0 | 60000/60000 | 0.987 | 0.837 (new pairs, same sizes) | 0.090 (3-digit (longer)) | 5602.900 | grok-maths-frac06-c2: started from step 0 (checkpoint not picked up); discarded |
| grok-maths-mean | grok-maths-mean-c0, grok-maths-mean-c1 | 16000/30000 | 1.000 | 0.683 (new pairs, same sizes) | 0.110 (3-digit (longer)) | 1128.600 | - |
| grok-maths-meaning | grok-maths-meaning-c0 | 8000/8000 | 1.000 | 0.830 (new pairs, same sizes) | 0.460 (3-digit (longer)) | 1814.300 | - |
| grok-maths-stablemax | grok-maths-stablemax-c0 | 23000/30000 | 0.000 | 0.000 (new pairs, same sizes) | 0.000 (3-digit (longer)) | nan | - |
| grok-maths-wd1-60k | grok-maths-wd1-60k-c0, grok-maths-wd1-60k-c1, grok-maths-wd1-60k-c2 | 60000/60000 | 1.000 | 0.727 (new pairs, same sizes) | 0.110 (3-digit (longer)) | 5724.500 | - |
| grok-maths-wd1 | grok-maths-wd1-c0, grok-maths-wd1-c1 | 30000/30000 | 0.983 | 0.727 (new pairs, same sizes) | 0.145 (3-digit (longer)) | 3736.400 | - |
| grok-translate-wd1-r8k | grok-translate-wd1-r8k-c0 | 12000/12000 | 0.927 | 0.803 (known words, new combinations) | 0.492 (new clause structure) | 3132.200 | step 8000-12000 rerun chunk; merged into grok-translate-wd1 |
| grok-translate-wd1 | grok-translate-wd1-c0, grok-translate-wd1-c1, grok-translate-wd1-r8k-c0 | 12000/12000 | 0.927 | 0.803 (known words, new combinations) | 0.492 (new clause structure) | 3132.200 | grok-translate-wd1-c2: resumed from step 4000 (stale checkpoint version), redid 400 |

## 10. Faith and Fate replication on real models (presentation)

Qwen2.5-1.5B/3B-Instruct prompted with the paper's step-by-step multiplication scratchpad, 50 problems per size; accuracy = last number in the answer equals the product.

| run | model | 1x1 | 2x1 | 2x2 | 3x1 | 3x2 | 3x3 |
|---|---|---|---|---|---|---|---|
| ff-real-5shot | Qwen2.5-1.5B-Instruct | 0.58 | 0.62 | 0.60 | 0.18 | 0.22 | 0.02 |
| ff-real-5shot | Qwen2.5-3B-Instruct | 0.58 | 0.80 | 0.56 | 0.48 | 0.34 | 0.46 |
| ff-real-models | Qwen2.5-1.5B-Instruct | 0.60 | 0.60 | 0.48 | 0.14 | 0.18 | 0.02 |
| ff-real-models | Qwen2.5-3B-Instruct | 0.72 | 0.90 | 0.90 | 0.76 | 0.50 | 0.82 |

## 11. Data generation and smoke test

| run | what it is |
|---|---|
| paragen-v2-s0 | paraphrase generation for the para runs (data, not a model); output merged elsewhere |
| paragen-v2-s0b | paraphrase generation for the para runs (data, not a model); 6671 records |
| smoke | smoke test of the pipeline, not a result |

## Headline numbers per claim

| claim | runs and numbers |
|---|---|
| 1. New symbols are learnable when they start from what they stand for (sound / real digit) | g1-hard-latininit-50k exact 0.990/0.986/0.972/0.974 vs p-hard-puatok-50k 0.570/0.506/0.420/0.000; maths add 4+4 two-seed means: meaning 0.86 (0.887/0.840), shuffled 0.34 (0.083/0.600: unstable), mean 0.09 (0.020/0.153); digits-only start 0.87, operators-only 0.06 |
| 2. Generalisation stops at training coverage of structure | E1 at 5 epochs: ~0.97 on trained structures, 0.00-0.09 on held-out ones; maths dropped pairs: problems needing the removed fact 0.05 (5x6) / 0.10 (3x4) vs 0.92 / 0.96 avoiding it |
| 3. ... of vocabulary | rare words fail below ~10 sightings; word floor 25 vs floor 12 baseline seeds: generated sentence exact 0.720 vs 0.607/0.627 (p < 0.002; but floor 25 also trained on ~20% more examples: 13,304 vs 11,064 steps), but Tatoeba meaning 0.603 vs 0.617/0.625 and sealed (v2, 147) meaning 0.503 vs 0.476/0.463 (no significant gain on real English: about half of Tatoeba's wrong sentences (54%) miss a word the generated data never had, 217 occurrences at 0.41, mostly common words/uses the generator does not produce, which no floor reaches; 30% have every word right) |
| 4. ... of English variety | real15 vs baseline seeds: Tatoeba meaning 0.923 vs 0.617/0.625; sealed (v2, 147) glyphs 0.558 vs 0.422/0.435, sealed meaning 0.558 vs 0.476/0.463; 6 epochs: generated 0.660 (above baseline 0.607) but Tatoeba meaning 0.563 (lowest) and sealed meaning 0.449. Caveat: Tatoeba gold and real15's training labels come from the same automatic reader, so the sealed gap (sealed v2 +0.089; paired McNemar per seed p = 0.036 and 0.013, significant; unpaired totals not) is the fairer size |
| 5. Writing the meaning first speeds learning, not generalisation | E1b vs E1a: 0.96 vs 0.26 exact at 3 epochs, both 0.97-0.99 at 5 epochs |
| 6. Model tricks are not the fix | sealed v2 glyphs: focal-r2 0.388, para 0.429, two-stage rewriter 0.422 vs baseline 0.422: no gain on real English; Tatoeba (paired): focal -0.028 (p=0.014) and 6 epochs -0.053 (p=0.0002) significantly worse, paraphrases 0.00 |
| Inside the model (claim 4) | probes (corrected 2026-10-25): where the output is wrong the probe is mostly wrong too (0.41 vs majority 0.79, 107 items, both seeds), so errors are mostly in the representation, not a lost read-out; steering at layer 6 flips tense 0.98-1.00 and negation 0.93-0.99 vs random <=0.02 (directions are used, but entangled) |
| Side study: grokking | no grokking: translation 12k steps (test 0.80), maths 40% 60k steps (test 0.73; wd 0.1 for the first 16k, wd 1.0 after), maths with 60% of the table 60k steps (test rising slowly 0.74 -> 0.84, no jump); logit norm rises instead of falling |

## Runs still running or queued

| run | status |
|---|---|
| e2b-v2fix-floor25 | done (in the tables above) |
| maths-meaning-drop22 | done (in the tables above) |
| maths-meaning-scratch | done (in the tables above) |
| maths-meaning-scratch-drop56 | done (in the tables above) |
| grok-maths-frac06 (last chunk) | done |
