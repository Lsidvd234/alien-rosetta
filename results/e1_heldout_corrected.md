# Correction: E1 held-out structure scores, counting only items that really contain the structure

When a held-out variant cannot apply to a clause (passive needs an object and a known participle; clefts and
topicalisation need the right parts), `train/represent.py` silently falls back to the canonical form, so part of each
"held-out" evaluation group does not contain the held-out structure. Items were re-classified by pattern
(passive: be-form + participle + "by"; clefts: "it is/was/will be ... that"; topicalised: a fronted noun phrase
followed by a comma). Exact match of the glyph output, from the per-item results of the E1 runs.

| held-out group | items WITH the structure | E1a | E1b | items that fell back | E1a | E1b |
|---|---|---|---|---|---|---|
| voice = passive | 62 | 0.00 | 0.00 | 38 | 0.18 | 0.32 |
| order = subject cleft | 76 | 0.01 | 0.05 | 24 | 0.04 | 0.12 |
| order = object cleft | 66 | 0.00 | 0.02 | 34 | 0.21 | 0.32 |
| order = object topicalised | 81 | 0.02 | 0.04 | 19 | 0.58 | 0.89 |

Conclusion (stronger than before): structure types absent from training are essentially never handled (0.00-0.05).
The pooled figures in e1_comparison.md (0.07-0.20) are inflated by the fallback items. Fixed for future runs:
`represent.eval_sets(..., strict=True)` keeps only items in which the forced variant was actually used.
