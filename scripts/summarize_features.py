"""Print a compact table from results/features_*.json."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/features_v0.json"
rows = json.load(open(path, encoding="utf-8-sig"))
toks = sorted({k.split(":")[0] for r in rows for k in r if ":" in k})
short = {t: t.split("-")[0] for t in toks}
print(f"{'variant':22s} {'script':14s} {'morph/w':>7s} {'sandhi':>6s} {'order':>5s} {'TTR':>5s} "
      + " ".join(f"{short[t] + ' tok/w':>12s}" for t in toks) + f" {'broken%':>8s}")
for r in rows:
    print(f"{r['variant']:22s} {r['script']:14s} {r['morphemes_per_word']:7.2f} {r['sandhi_rate']:6.2f} "
          f"{r['order_distance']:5.2f} {r['type_token_ratio']:5.3f} "
          + " ".join(f"{r[t + ':tokens_per_word']:12.2f}" for t in toks)
          + f" {100 * r[toks[0] + ':broken_token_rate']:7.1f}%")
