"""Test A (claim 5): does a fine-tuned model still UNDERSTAND English passives, outside the meaning format?
(local; template English sentences only, never the sealed set)

    python analysis/passive_qa.py [RUN ...] [--n 200]    -> results/inside/passive_qa.md + passive_qa_<model>.json
    python analysis/passive_qa.py --v2 [RUN ...]         -> passive_qa_v2.md: every sentence asked two opposite
        questions ("Who did the helping?" -> doer, "Who was helped?" -> the other); "understood" = both right, so a
        position habit (always the last person) scores 0. Redesign after v1 showed exactly that habit (28 Oct).
    RUN = run folders under results/kaggle (default: e2b-v2fix-passive0 = passives never in training, and
          e2b-v2fix-cot-80k-noheld = the baseline, which saw passives); the untrained Qwen3-0.6B-Base always runs first.

passive0 misreads every held-out passive in its meaning line ("by Y" becomes an invented BY= phrase, no doer). Two
explanations give that same line: (1) fine-tuning made it forget how English passives work; (2) it still understands
them but cannot write an unseen structure in the meaning format. This asks the models in plain English, "who did it?",
with no meaning format:
  - forced choice, no generation: which answer does the model find more likely, the doer ("the doctors") or the other
    person ("the teacher")? (sum of answer-token log-probabilities after a 4-example prompt; chance 0.50). A model that
    answers in the wrong format cannot fail this way.
  - both people are human nouns (no "the animate one did it" shortcut); the question names neither ("Who did the
    helping?"), so the answer is not given away by the question.
  - three kinds of sentence, n each: passive ("The teacher was helped by the doctors."), active ("The doctors helped
    the teacher."), and a "by" decoy, an active with a place after "by" ("The doctors helped the teacher by the
    river."), so copying the noun after "by" fails on decoys.
  - all four doer / other-person number combinations, balanced; verbs and nouns are common ones from the language's
    English word lists; the 4 prompt examples use words outside the test.
Predictions (results/findings_2026-10-24.md, written before this ran): QA1 the base model passes (>= 0.85 passive and
active, >= 0.80 decoy), else the test is not fair; QA2 passive0 on passives >= base - 0.10 = understanding kept (the
failure is in writing the meaning format), < base - 0.20 = forgotten, in between = partly.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "inside"
RUNS = ["e2b-v2fix-passive0", "e2b-v2fix-cot-80k-noheld"]
KINDS = ["passive", "active", "by decoy (active + place)"]
SHOTS = [("The pilot was thanked by the farmers.", "Who did the thanking?", "the farmers", "the pilot"),
         ("The nurse called the boys.", "Who did the calling?", "the nurse", "the boys"),
         ("The kings fed the baker by the lake.", "Who did the feeding?", "the kings", "the baker"),
         ("The singers were visited by the judge.", "Who did the visiting?", "the judge", "the singers")]
NOT_AGENTIVE = {"know", "think", "want", "need", "feel", "put", "give", "found", "mean", "like", "hope", "believe",
                "remember", "forget", "understand", "own", "cost", "lack", "include", "contain", "become", "use",
                "see", "hear", "love", "hate", "prefer", "deserve", "seem", "send", "show", "bring", "owe", "lend",
                "live", "stay", "try", "deal", "felt", "set", "start", "win", "cause", "plan", "report", "read"}
NOT_PEOPLE = {"someone", "somebody", "anyone", "anybody", "everyone", "everybody", "nobody", "god", "human", "guy",
              "person", "people", "individual", "self", "sir", "lord", "mom"}
V2 = "--v2" in sys.argv
SHOTS_V2 = [("The pilot was thanked by the farmers.", "Who did the thanking?", "the farmers", "the pilot"),
            ("The nurse called the boys.", "Who was called?", "the boys", "the nurse"),
            ("The kings fed the baker by the lake.", "Who did the feeding?", "the kings", "the baker"),
            ("The singers were visited by the judge.", "Who was visited?", "the singers", "the judge")]
SHOT_WORDS = {"pilot", "farmer", "nurse", "boy", "king", "baker", "singer", "judge", "lake",
              "thank", "call", "feed", "visit"}


def words():
    """Common transitive verbs and human nouns from the language's English lists (the generator's own pools)."""
    import train.represent_v2 as R2
    gen = R2.GenV2("tamil_like", "abugida_pua", 0)
    top = (lambda xs, w, k: [x for x, _ in sorted(zip(xs, w), key=lambda t: -t[1])][:k])
    verbs = [v for v in top(gen.tverbs, gen.tverb_w, 400)
             if v[0].isalpha() and v[3].isalpha() and v[4].isalpha() and v[0] not in SHOT_WORDS
             and v[0] not in {"have", "get", "say", "tell", "be", "let", "make", "do"} | NOT_AGENTIVE][:120]
    people = [n for n in top(gen.people, gen.people_w, 300)
              if n[1] and n[0].isalpha() and n[1].isalpha() and n[0] not in SHOT_WORDS | NOT_PEOPLE
              and not n[3]][:80]
    places = [p for p in top(gen.places, gen.places_w, 200) if p[0].isalpha() and p[0] not in SHOT_WORDS][:40]
    return verbs, people, places, R2.ing_of


def items(n: int, seed: int = 909) -> list[dict]:
    verbs, people, places, ing_of = words()
    rng = random.Random(seed)
    out = []
    for kind in KINDS:
        for i in range(n):
            d_pl, o_pl = (i % 4) // 2 == 1, i % 2 == 1                      # balanced number combinations
            v = rng.choice(verbs)
            d, o = rng.sample(people, 2)
            dn, on = (d[1] if d_pl else d[0]), (o[1] if o_pl else o[0])
            base, past, part = v[0], v[3], v[4]
            if kind == "passive":
                s = f"The {on} {'were' if o_pl else 'was'} {part} by the {dn}."
            elif kind == "active":
                s = f"The {dn} {past} the {on}."
            else:
                s = f"The {dn} {past} the {on} by the {rng.choice(places)[0]}."
            out.append({"kind": kind, "sentence": s, "question": f"Who did the {ing_of(base)}?",
                        "doer": f"the {dn}", "other": f"the {on}", "pair": len(out)})
            if V2:                                             # the opposite question about the same sentence
                out.append({"kind": kind, "sentence": s, "question": f"Who was {part}?",
                            "doer": f"the {on}", "other": f"the {dn}", "pair": len(out) - 1})
    return out


def prompt_of(it: dict) -> str:
    shots = "".join(f"Sentence: {s}\nQuestion: {q}\nAnswer: {a}\n\n" for s, q, a, _ in (SHOTS_V2 if V2 else SHOTS))
    return shots + f"Sentence: {it['sentence']}\nQuestion: {it['question']}\nAnswer:"


@torch.no_grad()
def logprob(model, tok, context: str, answer: str) -> float:
    dev = next(model.parameters()).device
    c = tok(context, add_special_tokens=False)["input_ids"]
    a = tok(" " + answer, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([c + a], device=dev)
    logits = model(input_ids=ids).logits[0, len(c) - 1:-1].float()
    return float(torch.log_softmax(logits, -1).gather(1, torch.tensor(a, device=dev)[:, None]).sum())


def wilson(k: int, n: int) -> tuple[float, float]:
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def run_model(name: str, tests: list[dict]) -> dict:
    if name == "base":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B-Base", dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    else:
        from train.load import load_trained
        model, tok, _ = load_trained(ROOT / "results" / "kaggle" / name / "model")
        model.eval()
    res = {"model": name, "by_kind": {}}
    for kind in KINDS:
        its = [t for t in tests if t["kind"] == kind]
        ok = {}
        for t in its:
            p = prompt_of(t)
            ok[id(t)] = logprob(model, tok, p, t["doer"]) > logprob(model, tok, p, t["other"])
        if V2:                                                 # a sentence counts only if both questions are right
            pairs = {}
            for t in its:
                pairs.setdefault(t["pair"], []).append(ok[id(t)])
            right, n = sum(all(v) for v in pairs.values()), len(pairs)
            did = [ok[id(t)] for t in its if t["question"].startswith("Who did")]
            was = [ok[id(t)] for t in its if t["question"].startswith("Who was")]
            extra = {"who_did": sum(did) / len(did), "who_was": sum(was) / len(was)}
        else:
            right, n, extra = sum(ok.values()), len(its), {}
        lo, hi = wilson(right, n)
        res["by_kind"][kind] = {"n": n, "right": right, "acc": right / n, "ci": [lo, hi], **extra}
        print(f"  {name} {kind}: {right}/{n} = {right / n:.3f} {extra}", flush=True)
    del model
    torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"passive_qa{'_v2' if V2 else ''}_{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main() -> None:
    a = sys.argv[1:]
    n = int(a[a.index("--n") + 1]) if "--n" in a else 200
    runs = [x for x in a if not x.startswith("--") and not x.isdigit()] or RUNS
    tests = items(n)
    print(f"{len(tests)} questions; e.g. {tests[0]['sentence']} {tests[0]['question']} -> {tests[0]['doer']}",
          flush=True)
    results = [run_model(m, tests) for m in ["base"] + runs]
    base = results[0]["by_kind"]
    tag = "_v2" if V2 else ""
    L = [f"# Test A{' v2' if V2 else ''}: does the model still understand English passives? (`analysis/passive_qa.py`)",
         "", (f"Each sentence asked two opposite questions (who did it / who was it done to); cells = share of "
              f"sentences with BOTH answers right ({n} sentences per kind; a position habit scores 0); in brackets "
              f"the two questions separately. Predictions QA1v2-QA3v2 in results/findings_2026-10-24.md."
              if V2 else f"Forced choice between the doer and the other person, {n} per kind, chance 0.50. "
              "Predictions QA1-QA2 in results/findings_2026-10-24.md."), "",
         "| model | " + " | ".join(KINDS) + " |", "|---|" + "---|" * len(KINDS)]
    for r in results:
        L.append(f"| {r['model']} | " + " | ".join(
            f"{r['by_kind'][k]['acc']:.2f} [{r['by_kind'][k]['ci'][0]:.2f}, {r['by_kind'][k]['ci'][1]:.2f}]"
            + (f" (did {r['by_kind'][k]['who_did']:.2f} / was {r['by_kind'][k]['who_was']:.2f})" if V2 else "")
            for k in KINDS) + " |")
    bar = 0.70 if V2 else 0.85
    fair = base["passive"]["acc"] >= bar and base["active"]["acc"] >= bar and (V2 or base[KINDS[2]]["acc"] >= 0.80)
    L += ["", f"QA1{'v2' if V2 else ''} (the base model passes, so the test is fair): **{'holds' if fair else 'fails'}**."]
    for r in results[1:]:
        if r["model"] == "e2b-v2fix-passive0":
            gap = base["passive"]["acc"] - r["by_kind"]["passive"]["acc"]
            v = ("understanding kept (the failure is in writing the meaning format)" if gap <= 0.10 else
                 "lost in fine-tuning" if gap > 0.20 else "partly lost")
            L.append(f"QA2{'v2' if V2 else ''} (passive0 on passives vs base, gap {gap:+.2f}): **{v}**"
                     + ("" if fair else " (but the fairness check failed: not interpretable)") + ".")
            if V2:
                ga = base["active"]["acc"] - r["by_kind"]["active"]["acc"]
                L.append(f"QA3v2 (passive0 on actives vs base, gap {ga:+.2f}): **"
                         + ("the loss is specific to passives" if ga <= 0.10 else
                            "it lost the question-answering more generally") + "**.")
    (OUT / f"passive_qa{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
