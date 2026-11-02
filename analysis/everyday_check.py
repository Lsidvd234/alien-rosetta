"""Check --rep-everyday against the English reader (2026-10-26).

    python analysis/everyday_check.py [N]      -> results/everyday_check.md

The everyday additions must be labelled exactly as the reader (sealed/app/english_parse.py) labels real English, or
the model learns a convention the Tatoeba and sealed gold answers do not use. Generates N examples with the switch
on and N with it off (same seed), reads each example's clean English (no typos, sentence case, single spaces, plain
words, the example's end mark: as scripts/build_paraphrases_v2.py checks paraphrases) and compares the reader's
analysis with the generator's. Agreement is reported overall and for examples that contain each new feature.
Generated English only; never the sealed set.
"""

from __future__ import annotations

import collections
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sealed" / "app"))
import train.represent_v2 as R2  # noqa: E402

NAMES = {n for ns in R2.ENGLISH_NAMES.values() for n in ns}
BY_RE = re.compile(r"\bby (the|this|that|these|those|my|your|his|her|its|our|their|" + "|".join(R2.BY_WITH) + r")\b")


def features(raw: str, analysis: str) -> list[str]:
    low = raw.lower()
    toks = set(re.findall(r"[a-z]+", low))
    m = " ".join(re.findall(r"\bM=(\S+)", analysis)).replace("+", " ").split()
    f = []
    for v in ("have", "do", "like"):
        if re.search(rf"\bV={v}\b", analysis):
            f.append(f"verb {v}")
    if "people" in toks:
        f.append("people")
    if re.search(r"\btime\b", analysis):
        f.append("time")
    if toks & set(R2.REFL_EN.values()):
        f.append("reflexive")
    if BY_RE.search(low) and not re.search(r"\b(was|were|been|be|is|are)\b \w+ by\b", low):
        f.append("by")
    if set(re.findall(r"[A-Z][a-z]+", raw)) & NAMES:
        f.append("English name")
    for w in sum(R2.X_HOW.values(), []):
        if w in m:
            f.append(f"how: {w}")
    for p in R2.X_PARTICLES:
        if p in m:
            f.append("particle")
            break
    for d in ("other", "first", "second", "third", "enough", "own"):
        if re.search(rf"\b{d}\+", analysis):
            f.append(f"det {d}")
    return f


def run(gen, parser, render, n: int, seed: int) -> tuple[collections.Counter, collections.Counter, list]:
    rng = random.Random(seed)
    ok, tot, bad = collections.Counter(), collections.Counter(), []
    made = 0
    while made < n:
        ex = gen.example(rng)
        it = gen.item(ex) if ex else None
        if not it:
            continue
        made += 1
        ch = ex["choices"]
        probe = gen.surface(random.Random(0), ex["raw"], dict(ch, noise="none", spacing="single", case="sentence",
                                                               lexical="plain"))
        try:
            r = parser.parse(probe)
            got = render(r.clauses, parser.reps)["analysis"] if r.clauses and not r.issues else None
        except Exception:
            got = None
        same = got == it["analysis"]
        for k in ["all"] + features(ex["raw"], it["analysis"]):
            tot[k] += 1
            ok[k] += same
            if not same and k != "all" and sum(1 for b in bad if b[0] == k) < 4:
                bad.append((k, probe, it["analysis"], got))
    return ok, tot, bad


def main() -> None:
    from english_parse import Parser, render
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    parser = Parser()
    R2.set_held_out(0.0)
    res = {}
    for on in (False, True):
        R2.EVERYDAY["on"] = on
        gen = R2.GenV2("tamil_like", "abugida_pua", 0)
        res[on] = run(gen, parser, render, n, 4242)
        R2.EVERYDAY["on"] = False
        print(f"everyday {on}: reader agrees on {res[on][0]['all']}/{res[on][1]['all']}", flush=True)
    ok0, tot0, _ = res[False]
    ok1, tot1, bad = res[True]
    md = ["# --rep-everyday vs the English reader (`analysis/everyday_check.py`)", "",
          f"{n} generated examples each, switch off / on (same seed). Agreement = the reader's analysis of the clean",
          "English equals the generator's analysis exactly.", "",
          "| examples | off: agree (n) | on: agree (n) |", "|---|---|---|"]
    for k in sorted(set(tot1) | set(tot0), key=lambda k: (k != "all", k)):
        a = f"{ok0[k] / tot0[k]:.2f} ({tot0[k]})" if tot0[k] else "-"
        b = f"{ok1[k] / tot1[k]:.2f} ({tot1[k]})" if tot1[k] else "-"
        md.append(f"| {k} | {a} | {b} |")
    md += ["", "Disagreements (up to 4 per feature): English / generator / reader", ""]
    for k, probe, gold, got in bad:
        md.append(f"- {k}: `{probe}`  \n  gen `{gold}`  \n  reader `{got}`")
    (ROOT / "results" / "everyday_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
