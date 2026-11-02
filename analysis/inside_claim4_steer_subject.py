"""Claim 4 from inside the model, phase 2 for the SUBJECT: do the subject directions get used, and does a push toward
the gold subject repair real-English subject errors? (local; generated sentences + held-out Tatoeba; never sealed)

    python analysis/inside_claim4_steer_subject.py RUN [RUN ...] [--n 40]   -> results/inside/steer_subject_<run>.json/.md

Why: in the tense/NEG/Q steering (inside_claim4_steer2.py, 2026-10-25) the real-failure stage was empty, because the
baseline's real-English errors are not tense/NEG/Q errors; the subject is the feature it gets wrong on Tatoeba (22/600)
and the one the probe reads worst where the output is wrong (0.32). Same machinery as steer2:

  Direction   mean hidden state (English words) of generated sentences with subject B minus subject A, unit length;
              push = alpha x layer's mean hidden-state norm x direction.
  Stage 1     sweep on one pair (1sg -> 3sgm): 6 layers x positions (english, all) x alphas 0.02-0.1; picks the most
              flips over a random push of the same size at damage <= 10% (the tense setting may not suit the subject).
  Stage 2     at the picked setting: person (1sg<->2sg, 1sg<->3sgm), gender (3sgm<->3sgf), number (1sg<->1pl,
              3sgm<->3pl), kind (name<->noun); flip / random flip / unreadable / other features kept.
  Stage 3     every Tatoeba sentence whose unsteered output has the wrong subject, pushed from the written subject
              toward the gold one; fixed vs random. Also at the tense setting (layer 6, all, 0.035) for comparison.
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.inside_claim4 import OUT, features, generated, real  # noqa: E402
from analysis.inside_claim4_steer2 import Checkpoint, layer_stats, layers_of, outputs, random_unit, rates  # noqa: E402
from train.load import load_trained  # noqa: E402

FEAT = "subject"
PAIRS = [("1sg", "2sg"), ("2sg", "1sg"), ("1sg", "3sgm"), ("3sgm", "1sg"), ("3sgm", "3sgf"), ("3sgf", "3sgm"),
         ("1sg", "1pl"), ("1pl", "1sg"), ("3sgm", "3pl"), ("3pl", "3sgm"), ("name", "noun"), ("noun", "name")]
POSITIONS = ("english", "all")
ALPHAS = (0.02, 0.035, 0.05, 0.1)
TENSE_SETTING = (6, "all", 0.035)                    # picked by steer2 for tense on both runs


def run(name: str, n: int) -> dict:
    model, tok, _ = load_trained(ROOT / "results" / "kaggle" / name / "model")
    n_layers = len(layers_of(model))
    layers = sorted({max(1, round(n_layers * f)) for f in (0.2, 0.35, 0.5, 0.65, 0.8, 0.95)} | {TENSE_SETTING[0]})
    by_val = {}
    for it in generated(6000):                       # 6000 (not 3000): ~70-100 fit sentences for the rarer pronouns
        by_val.setdefault(str(features(it["analysis"])[FEAT]), []).append(it["english"])
    # small values (~90 sentences) are split: everything but the last n fits the direction, the last n are the test
    fit = {k: v[:min(300, len(v) - n)] for k, v in by_val.items() if len(v) > n + 20}
    test = {k: v[-n:] for k, v in by_val.items() if k in fit}
    stats = {k: layer_stats(model, tok, s, layers) for k, s in fit.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    ck = Checkpoint(OUT / f"steer_subject_{name}_n{n}.partial")

    def gen(key, sents, *push):
        return ck.get(key, lambda: outputs(model, tok, sents, *push))

    def direction(a, b, L):
        d = stats[b][L][0].mean(0) - stats[a][L][0].mean(0)
        return d / d.norm(), (stats[a][L][1] + stats[b][L][1]) / 2

    def shuffled(a, b, L, key):
        """Stronger control than a random vector (which is near-orthogonal to everything in d=1024): the mean
        difference of a random split of the a and b sentences together, i.e. the same recipe with the labels shuffled."""
        x = torch.cat([stats[a][L][0], stats[b][L][0]])
        p = torch.randperm(len(x), generator=torch.Generator().manual_seed(zlib.crc32(f"shuffle:{key}".encode())))
        k = len(stats[a][L][0])
        d = x[p[k:]].mean(0) - x[p[:k]].mean(0)
        return d / d.norm()

    out = {"run": name, "layers": layers, "fit_sizes": {k: len(v) for k, v in fit.items()}, "stage1": {},
           "stage2": {}, "stage3": {}}
    # stage 1
    a, b = "1sg", "3sgm"
    before = [features(o) for o in gen("s1:before", test[a])]
    best = loose = None
    for L in layers:
        d, scale = direction(a, b, L)
        r = random_unit(d.shape, f"s1:L{L}")
        for pos in POSITIONS:
            for al in ALPHAS:
                k = f"L{L}:{pos}:{al}"
                fl, br, kp = rates(gen(f"s1:{k}", test[a], L, al * scale * d, pos), FEAT, b, before)
                rf, _, _ = rates(gen(f"s1:{k}:random", test[a], L, al * scale * r, pos), FEAT, b, before)
                out["stage1"][f"L{L}:{pos}:{al}"] = {"flip": fl, "broken": br, "others_kept": kp, "random_flip": rf}
                print(f"  stage1 L{L} {pos:7s} a={al}: flip {fl:.2f} (random {rf:.2f}) broken {br:.2f}", flush=True)
                # pick: most flips over random, at damage <= 10% AND other features kept >= 0.8 (the tense/NEG
                # setting changed other features too often: kept 0.15-0.87); fallback without the kept rule
                if br <= 0.10 and (kp is None or kp >= 0.8) and (best is None or fl - rf > best[0]):
                    best = (fl - rf, L, pos, al)
                if br <= 0.10 and (loose is None or fl - rf > loose[0]):
                    loose = (fl - rf, L, pos, al)
    _, L, pos, al = best or loose or (0, *TENSE_SETTING)
    out["picked"] = {"layer": L, "position": pos, "alpha": al}
    print(f"  picked layer {L}, {pos}, alpha {al}", flush=True)
    # stage 2
    for a, b in PAIRS:
        if a not in test or b not in stats:
            continue
        k = f"s2:{a}>{b}"
        before = [features(o) for o in gen(f"{k}:before", test[a])]
        d, scale = direction(a, b, L)
        fl, br, kp = rates(gen(k, test[a], L, al * scale * d, pos), FEAT, b, before)
        rf, _, _ = rates(gen(f"{k}:random", test[a], L, al * scale * random_unit(d.shape, k), pos), FEAT, b, before)
        sf, _, _ = rates(gen(f"{k}:shuffled", test[a], L, al * scale * shuffled(a, b, L, k), pos), FEAT, b, before)
        out["stage2"][f"{a}>{b}"] = {"n": len(test[a]), "flip": fl, "broken": br, "others_kept": kp, "random_flip": rf,
                                     "shuffled_flip": sf}
        print(f"  stage2 {a:>5s}>{b:5s}: flip {fl:.2f} (random {rf:.2f}, shuffled labels {sf:.2f}) broken {br:.2f} "
              f"others kept {kp}", flush=True)
    # stage 3, at the picked setting and at the tense setting
    reals = real()
    base = gen("s3:base", [it["english"] for it in reals])
    cases, skipped = [], 0
    for it, o in zip(reals, base):
        g, p = features(it["analysis"]), features(o)
        if g and p and str(g[FEAT]) != str(p[FEAT]):
            if str(p[FEAT]) in stats and str(g[FEAT]) in stats:
                cases.append((it["english"], str(p[FEAT]), str(g[FEAT])))
            else:
                skipped += 1
    for label, (L3, pos3, al3) in (("picked", (L, pos, al)), ("tense setting", TENSE_SETTING)):
        if L3 not in layers:
            continue
        fixed = rfixed = sfixed = broke = 0
        examples = []
        for eng, wrote, gold in cases:
            d, scale = direction(wrote, gold, L3)
            k = f"s3:{label}:{eng}"
            o1 = gen(k, [eng], L3, al3 * scale * d, pos3)[0]
            o2 = features(gen(f"{k}:random", [eng], L3, al3 * scale * random_unit(d.shape, k), pos3)[0])
            f1 = features(o1)
            fixed += bool(f1) and str(f1[FEAT]) == gold
            broke += not f1
            rfixed += bool(o2) and str(o2[FEAT]) == gold
            o3 = features(gen(f"{k}:shuffled", [eng], L3, al3 * scale * shuffled(wrote, gold, L3, k), pos3)[0])
            sfixed += bool(o3) and str(o3[FEAT]) == gold
            examples.append({"english": eng, "wrote": wrote, "gold": gold, "steered": f1 and str(f1[FEAT])})
        out["stage3"][label] = {"setting": [L3, pos3, al3], "n_failures": len(cases), "skipped": skipped,
                                "fixed": fixed, "random_fixed": rfixed, "shuffled_fixed": sfixed, "unreadable": broke, "cases": examples}
        print(f"  stage3 {label}: {fixed}/{len(cases)} real subject failures fixed (random {rfixed}, shuffled labels {sfixed}, unreadable "
              f"{broke}, {skipped} skipped)", flush=True)
    (OUT / f"steer_subject_{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = [f"# Subject steering: {name}", "", f"Picked setting: layer {L}, position {pos}, alpha {al}.", "",
          "| pair | n | flip | random flip | shuffled-label flip | unreadable | other features kept |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {k} | {v['n']} | {v['flip']:.2f} | {v['random_flip']:.2f} | {v['shuffled_flip']:.2f} | {v['broken']:.2f} | "
           f"{v['others_kept'] if v['others_kept'] is None else round(v['others_kept'], 2)} |"
           for k, v in out["stage2"].items()]
    md += ["", "| setting | real subject failures | fixed by the push | fixed by a random push | shuffled labels | unreadable |",
           "|---|---|---|---|---|---|"]
    md += [f"| {k} {v['setting']} | {v['n_failures']} | {v['fixed']} | {v['random_fixed']} | {v['shuffled_fixed']} | {v['unreadable']} |"
           for k, v in out["stage3"].items()]
    md += ["", "Real failures (written -> gold : after the push):", ""]
    md += [f"- {c['wrote']} -> {c['gold']} : {c['steered']}  ({c['english']})"
           for c in out["stage3"].get("picked", {}).get("cases", [])]
    (OUT / f"steer_subject_{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    ck.done()
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    n = int(a[a.index("--n") + 1]) if "--n" in a else 40
    for r in [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] != "--n")]:
        run(r, n)
