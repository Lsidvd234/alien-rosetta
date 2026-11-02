"""Mixed-rule scratchpad: is an unseen fact built from blocks seen in that role? (local, CPU; predictions RM2 / RC3)

    python analysis/buildmix_blocks.py RUN [RUN ...]   -> results/buildmix_blocks.md + .json
    RUN = a --scratch-build-mix --drop-pair run under results/kaggle (e.g. maths-meaning-buildmix-drop56)

Rebuilds the run's training data (same seed, pair and rules), counts how often each fact occurs as a BUILDING BLOCK
in a training working, then reads every written step of the dropped fact in the "needs dropped pair" test: which
block the model built it from, whether that block was ever a block in training, and whether the step is right.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train.maths as M  # noqa: E402


def tokens(work: str):
    return work.split(M.SCR["ans"])[0].replace(M.SCR[";"], " ").split(" ")


def main() -> None:
    rows, out = [], {}
    for run in [a for a in sys.argv[1:] if not a.startswith("--")]:
        d = json.loads((ROOT / "results" / "kaggle" / run / "results" / f"{run}.json").read_text(encoding="utf-8"))
        args = d["args"]
        prs = [tuple(int(x) for x in q.split(",")) for q in args["drop_pair"].split(";")]   # one pair or several
        pair = prs[0] if len(prs) == 1 else prs
        M.MIX.update(on=True, avoid={pr for x, y in prs for pr in ((x, y), (y, x))})
        train, _ = M.build(int(args["n"]), int(args["per_test"]), int(args["seed"]), None, None, pair)
        train = M.scratchify(train, True)
        M.MIX.update(on=False, avoid=set())
        seen = collections.Counter()
        for it in train:
            if it["key"][0] == "mul":
                for t in tokens(it["target"]):
                    parts = t.split(M.SCR["="])
                    if len(parts) > 2:
                        seen[M.readable(parts[1][:3])] += 1
        names = [k for k in d["tests"] if k.startswith("mul: needs dropped pair")]
        heads = {f"{x}x{y}" for a_, b_ in prs for x, y in ((a_, b_), (b_, a_))}
        by = collections.defaultdict(lambda: [0, 0])
        per_head = collections.defaultdict(lambda: [0, 0])
        for p in [q for k in names for q in d["tests"][k]["preds_all"]]:
            for t in tokens(p):
                r = M.readable(t)
                if r[:3] not in heads:
                    continue
                s_ = M._step(t)
                per_head[r[:3]][0] += 1
                per_head[r[:3]][1] += bool(s_ and s_[0] * s_[1] + s_[2] == s_[3])
                parts = r.split("=")
                blk = parts[1][:3] if len(parts) > 2 else "(none)"
                if str(args.get("scratch_build_add")) == "True":      # pure addition: the route = what is counted
                    blk = f"count {parts[1][0]}s" if len(parts) > 1 and parts[1] else "(none)"
                s = M._step(t)
                by[blk][0] += 1
                by[blk][1] += bool(s and s[0] * s[1] + s[2] == s[3])
        n = sum(v[0] for v in by.values())
        sb = [v for b, v in by.items() if seen[b]]
        ub = [v for b, v in by.items() if not seen[b]]
        res = {"pair": prs, "seed": int(args["seed"]),
               "needs_exact": {k[24:].split(" (")[0]: d["tests"][k]["exact"] for k in names},
               "per_fact": {h: {"steps": v[0], "right": v[1] / v[0]} for h, v in per_head.items()},
               "steps": n, "share_seen_block": sum(v[0] for v in sb) / max(1, n),
               "right_seen_block": sum(v[1] for v in sb) / max(1, sum(v[0] for v in sb)),
               "right_unseen_block": (sum(v[1] for v in ub) / sum(v[0] for v in ub)) if ub else None,
               "by_block": {b: {"steps": v[0], "right": v[1] / v[0], "seen_as_block": seen[b]} for b, v in by.items()}}
        out[run] = res
        rows.append(f"| {run} | {', '.join(f'{x}x{y}' for x, y in prs)} | {res['seed']} | "
                    + ", ".join(f"{k} {v:.2f}" for k, v in res["needs_exact"].items()) + f" | {n} | "
                    f"{res['share_seen_block']:.2f} | {res['right_seen_block']:.2f} | "
                    + ("-" if res["right_unseen_block"] is None else f"{res['right_unseen_block']:.2f}") + " | "
                    + ", ".join(f"{b} {v[0]} ({v[1] / v[0]:.2f}{'' if seen[b] else ', never a block'})"
                                for b, v in sorted(by.items(), key=lambda x: -x[1][0])) + " | "
                    + ", ".join(f"{h} {v[1] / v[0]:.2f} ({v[0]})" for h, v in sorted(per_head.items())) + " |")
    L = ["# Mixed-rule scratchpad: unseen fact built from which blocks? (`analysis/buildmix_blocks.py`)", "",
         "| run | missing fact | seed | needs-pair exact | its written steps | share from a seen block | right: seen "
         "block | right: never-a-block | blocks used (steps, right) | per fact written (right, steps) |",
         "|---|---|---|---|---|---|---|---|---|---|"] + rows
    (ROOT / "results" / "buildmix_blocks.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "results" / "buildmix_blocks.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
