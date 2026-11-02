"""New claim 5, prediction R4: when a held-out English structure is misread, is its meaning still inside the model?
(local; generated sentences only, never the sealed set)

    python analysis/inside_passive.py [RUN ...] [--n 600]    -> results/inside/passive4_<run>.json + passive4.md
    python analysis/inside_passive.py --v2 e2b-v2fix-passive0 base   the binding R4 test: v2 language and generator
                                                             (base = untrained Qwen3-0.6B-Base, v2 items)
    RUN = a run folder under results/kaggle (default e1b-hard-rep-cot-5ep, the v1 model with passive held out), or
          "base" (untrained Qwen3-0.6B-Base: probes only). Pilot on v1; the real test is the v2 passive-0 model.

Every v1 passive test item is misread (0.97-1.00 reading failures), typically by position: "this laptop is booed by
us" -> S=laptop. The question is whether the doer is still represented and only the readout fails (R4 >= 0.85), or
the representation itself follows the surface order (< 0.60).

Items: single-clause sentences of the run's own language (Rep tamil_like / abugida_pua / seed 0), a transitive verb,
a noun object, a noun or pronoun doer (no names: name-as-subject is held out too; no "you": 2sg/2pl look alike), every
other choice drawn as in training. All four doer x patient number combinations, balanced (redesign, 28 Oct): the
pilot kept only doer != patient, where the auxiliary (was/were agrees with the patient) gives the doer's number away
and a bag-of-words classifier already scored 0.89. Now the doer's number is independent of the auxiliary.

  probe A  trained on ACTIVE sentences only (the structures the model was fine-tuned on); read on passives.
           Shows what the model's trained "doer" reading does on a passive (position -> near 0).
  probe B  (R4) trained on actives + 40% of the passives, layer chosen on another 20% of the passives, scored on the
           last 40%: is the doer linearly decodable from the fine-tuned model's states on passives at all?
           Reported overall and on the passives whose OUTPUT read the doer wrong.
  control  probe B with shuffled labels (should be ~0.5); and probe B for the PATIENT's number.
  baselines on the same split (R4 amendment): bag-of-words logistic regression on the English (no model), and the
           probe B recipe on layer 0 only (word embeddings, mean over the English).
  output   the model's own analysis of every test passive: doer number right, and "position reading" (the output's
           S= is the gold patient). "Misread" = the output's S= is not the gold doer.
Prediction R4 as amended 28 Oct (results/findings_2026-10-24.md): readout failure if probe B on misread passives is
>= 0.85 AND >= 0.10 above the better baseline; representation failure if < 0.60 or within 0.05 of it; else
inconclusive.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.inside_claim4 import OUT, hidden  # noqa: E402

RUN = "e1b-hard-rep-cot-5ep"
FORCE = {"order": "canonical", "noise": "none", "lexical": "plain", "object": "noun"}


V2 = "--v2" in sys.argv          # v2 runs (e2b-v2fix-passive0): items from the v2 generator, which the run was trained on
FORCE_V2 = {"kind": "statement", "predicate": "transitive", "order": "canonical", "noise": "none", "lexical": "plain",
            "spacing": "single", "case": "sentence", "vocative": "none"}


def num(tok: str) -> str:
    """sg / pl of an analysis argument: PRON.1pl, dog.PL, big+dog; v2 owners: PRON.1sg^dog.PL -> the owned noun."""
    tok = tok.rsplit("^", 1)[-1]
    if tok.startswith("PRON."):
        return "pl" if "pl" in tok else "sg"
    return "pl" if tok.endswith(".PL") else "sg"


def args_of(analysis: str) -> dict:
    first = analysis.split(" ; ")[0]
    return {k: v for k, v in re.findall(r"(?:^| )([SO])=([^ ]+)", first)}


def items(n: int, voice: str, seed: int) -> list[dict]:
    if V2:
        return items_v2(n, voice, seed)
    from train.represent import Rep
    rep = Rep("tamil_like", "abugida_pua", 0)
    rng = random.Random(seed)
    out, seen, count = [], set(), {(d, p): 0 for d in ("sg", "pl") for p in ("sg", "pl")}
    while len(out) < n:
        force = {**FORCE, "voice": voice, "subject": rng.choice(["noun", "pronoun"])}
        ex = rep.example(rng, force=force, n_clauses=1, adj_counts={0: 0.6, 1: 0.4}, allow_held=False)
        if not ex or ex["english"] in seen or (voice == "passive" and ("voice", "passive") not in ex["used"]):
            continue
        a = args_of(ex["analysis"])
        if "S" not in a or "O" not in a or a["S"] in ("PRON.2sg", "PRON.2pl") or a["S"].startswith("NAME."):
            continue
        d, p = num(a["S"]), num(a["O"])
        if count[d, p] >= n // 4:
            continue
        count[d, p] += 1
        seen.add(ex["english"])
        out.append({"english": ex["english"], "analysis": ex["analysis"], "doer": d, "patient": p,
                    "S": a["S"], "O": a["O"]})
    return out


def items_v2(n: int, voice: str, seed: int) -> list[dict]:
    """As items(), from the v2 generator (seed-0 language), single event, noun or pronoun doer and patient; no names,
    no "you", no joined phrases (A&B), balanced over the four doer x patient number cells."""
    import train.represent_v2 as R2
    R2.set_held_out(0)                                   # passive0 holds out only plain passives; allow them here
    gen = R2.GenV2("tamil_like", "abugida_pua", 0)
    rng = random.Random(seed)
    out, seen, count = [], set(), {(d, p): 0 for d in ("sg", "pl") for p in ("sg", "pl")}
    tries = 0
    while len(out) < n and tries < 400 * n:
        tries += 1
        force = {**FORCE_V2, "voice": voice, "subject": rng.choice(["noun", "pronoun"]),
                 "object": rng.choice(["noun", "pronoun"])}
        ex = gen.example(rng, force=force, n_events=1, allow_held=True)
        it = gen.item(ex) if ex else None
        passive = any(("voice", v) in ex["used"] for v in ("passive", "passive without doer"))   # active is not
        if not it or it["english"] in seen or passive != (voice == "passive"):                  # recorded in used
            continue
        a = args_of(it["analysis"])
        if "S" not in a or "O" not in a or any(x.startswith(("PRON.2", "NAME.")) or "&" in x or "NAME." in x
                                               for x in (a["S"], a["O"])):
            continue
        d, p = num(a["S"]), num(a["O"])
        if count[d, p] >= n // 4:
            continue
        count[d, p] += 1
        seen.add(it["english"])
        out.append({"english": it["english"], "analysis": it["analysis"], "doer": d, "patient": p,
                    "S": a["S"], "O": a["O"]})
    return out


def fit(xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(xtr)
    return sc, LogisticRegression(max_iter=2000, C=0.05).fit(sc.transform(xtr), ytr)


def acc(model, x, y):
    sc, clf = model
    return float((clf.predict(sc.transform(x)) == y).mean())


def best_layer(reps: dict, tr_idx, ytr, va_idx, yva, x_of) -> tuple[str, int, float]:
    """The (representation, layer) whose probe, trained on tr_idx, scores best on va_idx."""
    best = None
    for rep, X in reps.items():
        for L in range(X.shape[1]):
            m = fit(x_of(X, tr_idx, L), ytr)
            a = acc(m, x_of(X, va_idx, L), yva)
            if best is None or a > best[2]:
                best = (rep, L, a)
    return best


def analyse(run: str, n: int) -> dict:
    from train.load import load_trained, translate
    if run == "base":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base",  # noqa: E501
                                                     dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    else:
        model, tok, _ = load_trained(ROOT / "results" / "kaggle" / run / "model")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    act, pas = items(n, "active", 501), items(n, "passive", 502)
    print(f"{run}: {len(act)} active, {len(pas)} passive sentences (doer and patient differ in number)", flush=True)
    a_last, a_mean = hidden(model, tok, [x["english"] for x in act])
    p_last, p_mean = hidden(model, tok, [x["english"] for x in pas])
    reps_a, reps_p = {"last": a_last, "mean": a_mean}, {"last": p_last, "mean": p_mean}
    ya = np.array([x["doer"] for x in act])
    yp = np.array([x["doer"] for x in pas])
    yp_pat = np.array([x["patient"] for x in pas])
    rng = np.random.default_rng(0)
    ia = rng.permutation(len(act))
    a_tr, a_va = ia[: int(0.8 * len(ia))], ia[int(0.8 * len(ia)):]
    ip = rng.permutation(len(pas))
    p_tr, p_va, p_te = ip[: int(0.4 * len(ip))], ip[int(0.4 * len(ip)): int(0.6 * len(ip))], ip[int(0.6 * len(ip)):]

    res = {"run": run, "n_active": len(act), "n_passive": len(pas)}
    # probe A: actives only, layer chosen on actives
    rep, L, va = best_layer(reps_a, a_tr, ya[a_tr], a_va, ya[a_va], lambda X, i, l: X[i, l].astype(np.float32))
    mA = fit(reps_a[rep][a_tr, L].astype(np.float32), ya[a_tr])
    predA = mA[1].predict(mA[0].transform(reps_p[rep][p_te, L].astype(np.float32)))
    res["probe_A"] = {"layer": f"{rep}:{L}", "active_val": round(va, 4),
                      "passive_doer": round(float((predA == yp[p_te]).mean()), 4),
                      "passive_reads_patient": round(float((predA == yp_pat[p_te]).mean()), 4)}

    # probe B (R4): actives + 40% passives; layer chosen on 20% passives; scored on the other 40%
    def stack(X_a, X_p, ia_, ip_, L_):
        return np.concatenate([X_a[ia_, L_], X_p[ip_, L_]]).astype(np.float32)

    def choose(y_a, y_p):
        best = None
        for rep_ in reps_a:
            for L_ in range(reps_a[rep_].shape[1]):
                m = fit(stack(reps_a[rep_], reps_p[rep_], a_tr, p_tr, L_), np.concatenate([y_a[a_tr], y_p[p_tr]]))
                v = acc(m, reps_p[rep_][p_va, L_].astype(np.float32), y_p[p_va])
                if best is None or v > best[2]:
                    best = (rep_, L_, v, m)
        return best

    rep, L, va, mB = choose(ya, yp)
    predB = mB[1].predict(mB[0].transform(reps_p[rep][p_te, L].astype(np.float32)))
    res["probe_B"] = {"layer": f"{rep}:{L}", "passive_val": round(va, 4),
                      "passive_doer": round(float((predB == yp[p_te]).mean()), 4),
                      "per_layer_passive_test": {}}
    for rep_ in reps_a:                                              # the whole layer curve, for the figure
        for L_ in range(reps_a[rep_].shape[1]):
            m = fit(stack(reps_a[rep_], reps_p[rep_], a_tr, p_tr, L_), np.concatenate([ya[a_tr], yp[p_tr]]))
            res["probe_B"]["per_layer_passive_test"][f"{rep_}:{L_}"] = round(
                acc(m, reps_p[rep_][p_te, L_].astype(np.float32), yp[p_te]), 4)
    # controls: shuffled labels, and the patient's number with the same recipe
    ys_a, ys_p = rng.permutation(ya), rng.permutation(yp)
    rep_s, L_s, _, mS = choose(ys_a, ys_p)
    res["control_shuffled"] = round(acc(mS, reps_p[rep_s][p_te, L_s].astype(np.float32), ys_p[p_te]), 4)
    ya_pat = np.array([x["patient"] for x in act])
    rep_q, L_q, _, mQ = choose(ya_pat, yp_pat)
    res["probe_B_patient"] = {"layer": f"{rep_q}:{L_q}",
                              "passive_patient": round(acc(mQ, reps_p[rep_q][p_te, L_q].astype(np.float32),
                                                           yp_pat[p_te]), 4)}

    # baselines (R4 amendment): what the surface words alone, or the word embeddings, give on the same split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    tr_en = [act[i]["english"] for i in a_tr] + [pas[i]["english"] for i in p_tr]
    cv = CountVectorizer(token_pattern=r"[A-Za-z']+", lowercase=True).fit(tr_en)
    bow = LogisticRegression(max_iter=4000, C=1.0).fit(cv.transform(tr_en), np.concatenate([ya[a_tr], yp[p_tr]]))
    predW = bow.predict(cv.transform([pas[i]["english"] for i in p_te]))
    m0 = fit(stack(reps_a["mean"], reps_p["mean"], a_tr, p_tr, 0), np.concatenate([ya[a_tr], yp[p_tr]]))
    pred0 = m0[1].predict(m0[0].transform(reps_p["mean"][p_te, 0].astype(np.float32)))
    res["baselines"] = {"bag_of_words": round(float((predW == yp[p_te]).mean()), 4),
                        "layer0_mean": round(float((pred0 == yp[p_te]).mean()), 4)}

    if run != "base":                                                # the model's own reading of the test passives
        te = [pas[i] for i in p_te]
        outs = []
        for s in range(0, len(te), 16):
            outs += translate(model, tok, [x["english"] for x in te[s:s + 16]], max_new_tokens=60)
        got = [args_of(o.split("=>")[0].strip()) for o in outs]
        differ = np.array([x["doer"] != x["patient"] for x in te])
        doer_ok = np.array([("S" in g) and num(g["S"]) == x["doer"] for g, x in zip(got, te)])
        position = np.array([g.get("S") == x["O"] for g, x in zip(got, te)])
        exact = np.array([o.split("=>")[0].strip() == x["analysis"] for o, x in zip(outs, te)])
        wrong = np.array([g.get("S") != x["S"] for g, x in zip(got, te)])     # misread: S= is not the gold doer
        res["output"] = {"analysis_exact": round(float(exact.mean()), 4),
                         "doer_number_right (doer, patient differ)": round(float(doer_ok[differ].mean()), 4),
                         "position_reading (S = gold patient)": round(float(position.mean()), 4),
                         "n_doer_wrong": int(wrong.sum()),
                         "probe_B_where_output_doer_wrong": round(float((predB[wrong] == yp[p_te][wrong]).mean()), 4)
                         if wrong.any() else None,
                         "probe_A_where_output_doer_wrong": round(float((predA[wrong] == yp[p_te][wrong]).mean()), 4)
                         if wrong.any() else None,
                         "bag_of_words_where_output_doer_wrong": round(float((predW[wrong] == yp[p_te][wrong]).mean()), 4)
                         if wrong.any() else None,
                         "layer0_where_output_doer_wrong": round(float((pred0[wrong] == yp[p_te][wrong]).mean()), 4)
                         if wrong.any() else None}
        if wrong.any():                                              # the amended R4 rule
            pb = res["output"]["probe_B_where_output_doer_wrong"]
            base = max(res["output"]["bag_of_words_where_output_doer_wrong"], res["output"]["layer0_where_output_doer_wrong"])
            res["R4"] = ("readout failure" if pb >= 0.85 and pb - base >= 0.10 else
                         "representation failure" if pb < 0.60 or pb - base < 0.05 else "inconclusive")
        res["samples"] = [{"english": x["english"], "gold": x["analysis"], "model": o.split("=>")[0].strip()}
                          for x, o in list(zip(te, outs))[:12]]
    print(json.dumps({k: v for k, v in res.items() if k not in ("samples",)}, indent=1)[:3000], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"passive4{'v2' if V2 else ''}_{run}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return res


def table() -> None:
    L = ["# Is a misread passive's doer still in the model? (`analysis/inside_passive.py`, prediction R4, redesign)", "",
         "All four doer x patient number combinations, balanced. Probe A: trained on actives only. Probe B: trained on "
         "actives + other passives (R4). Baselines on the same split: bag of words (no model) and layer 0 (word "
         "embeddings). Misread = output S= is not the gold doer. Chance 0.5. Rule (amended 28 Oct): readout failure if "
         "probe B where misread >= 0.85 and >= 0.10 above the better baseline; representation failure if < 0.60 or "
         "within 0.05.", "",
         "| run | output: doer number right (differ) | output: first noun read as doer | probe A on passives | "
         "probe B on passives | bag of words / layer 0 | where misread (n): probe B, bag of words, layer 0 | "
         "shuffled control | patient probe | layers A / B | R4 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in sorted(OUT.glob(f"passive4{'v2' if V2 else ''}_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        o = r.get("output", {})
        g = (lambda k: "-" if o.get(k) is None else o[k])
        L.append(f"| {r['run']} | {g('doer_number_right (doer, patient differ)')} | "
                 f"{g('position_reading (S = gold patient)')} | {r['probe_A']['passive_doer']} | "
                 f"{r['probe_B']['passive_doer']} | {r['baselines']['bag_of_words']} / {r['baselines']['layer0_mean']} | "
                 f"({g('n_doer_wrong')}) {g('probe_B_where_output_doer_wrong')}, "
                 f"{g('bag_of_words_where_output_doer_wrong')}, {g('layer0_where_output_doer_wrong')} | "
                 f"{r['control_shuffled']} | {r['probe_B_patient']['passive_patient']} | "
                 f"{r['probe_A']['layer']} / {r['probe_B']['layer']} | {r.get('R4', '-')} |")
    (OUT / f"passive4{'v2' if V2 else ''}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--table"]:
        table()
    else:
        n = int(a[a.index("--n") + 1]) if "--n" in a else 600
        for r in [x for x in a if not x.startswith("--") and not x.isdigit()] or [RUN]:
            analyse(r, n)
        table()
