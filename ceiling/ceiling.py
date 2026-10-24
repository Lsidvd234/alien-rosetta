"""English-understanding ceiling: how much of these sentences the base model understands BEFORE any training.

    python ceiling/ceiling.py qa     [--model Qwen/Qwen3-0.6B-Base]   method 1: plain-English comprehension questions
    python ceiling/ceiling.py fmt    [--model ...]                    method 2: our meaning format, few-shot, lenient
    python ceiling/ceiling.py probe  [--model ...]                    method 3: linear probes on frozen hidden states
    python ceiling/ceiling.py trained [--run e2b-v2fix-cot-80k-noheld] reference: a trained model's meaning output
    python ceiling/ceiling.py table                                   all results -> results/ceiling.md

Test sentences: 300 single generated statements (one event, so 'who did it' has one answer; clean spelling), plus 3,000
other generated statements to train the probe classifiers. The facts asked about come from each sentence's known
meaning (conlang/v2.py ClauseV2): doer, action, thing done to, place, tense, negated?, one or many doers. Every method
is scored per fact (word level) and per sentence (all facts right), with the chance level of each fact type. Uses
generated sentences only; never the sealed set.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train.represent_v2 as R2  # noqa: E402
from conlang.frames import NP, PRONOUNS  # noqa: E402
from conlang.v2 import NPX, Coord, Name  # noqa: E402

OUT = ROOT / "results" / "ceiling"
CLEAN = {"noise": "none", "lexical": "plain", "spacing": "single", "case": "sentence", "end": "full stop",
         "kind": "statement"}
PRON_FORMS = {"1sg": {"i", "me"}, "2sg": {"you"}, "3sgm": {"he", "him"}, "3sgf": {"she", "her"}, "3sgn": {"it"},
              "1pl": {"we", "us"}, "2pl": {"you"}, "3pl": {"they", "them"}}
CHANCE = {"tense": 1 / 3, "negated": 1 / 2, "number": 1 / 2}


# --- data: sentences and their facts ---------------------------------------------------------------------------------
class Forms:
    """English forms of a word key (for lenient answer matching): dog -> dog, dogs; eat -> eat, eats, ate, eaten..."""

    def __init__(self, gen: R2.GenV2):
        self.m: dict[str, set[str]] = collections.defaultdict(set)
        for sg, pl, k, _ in gen.nouns + gen.places:
            self.m[k] |= {sg} | ({pl} if pl else set())
        for w, k in gen.adjs + gen.manner:
            self.m[k].add(w)
        for b, k, third, past, part, _ in gen.verbs:
            self.m[k] |= {b, third, past, part, R2.ing_of(b)}
        self.m["be"] |= {"be", "is", "are", "am", "was", "were", "been", "being"}

    def of(self, x) -> set[str]:
        if x is None:
            return set()
        if isinstance(x, str):
            return set(PRON_FORMS.get(x, {x}))
        if isinstance(x, Name):
            return {x.name.lower()}
        if isinstance(x, Coord):
            return set().union(*(self.of(i) for i in x.items))
        if isinstance(x, NP):
            return {f.lower() for f in self.m.get(x.noun, {x.noun})}
        return set()


def facts(cl, forms: Forms) -> dict:
    """The facts of one event that have a definite answer."""
    f = {}
    subj = Name(cl.subj_name, cl.subj_ph) if cl.subj_name else cl.subj
    obj = Name(cl.obj_name, cl.obj_ph) if cl.obj_name else cl.obj
    if subj is not None and not isinstance(subj, Coord):
        f["doer"] = sorted(forms.of(subj))
        if isinstance(subj, str):
            if subj not in ("2sg", "2pl"):                       # 'you' can be one person or many
                f["number"] = "many" if PRONOUNS[subj][2] == "pl" else "one"
        elif isinstance(subj, NP) and subj.noun:
            f["number"] = "many" if subj.plural else "one"
        elif isinstance(subj, Name):
            f["number"] = "one"
    if cl.verb:
        f["action"] = sorted(forms.m.get(cl.verb, {cl.verb}) or {cl.verb})
    if obj is not None and not isinstance(obj, Coord):
        f["thing done to"] = sorted(forms.of(obj))
    place = cl.loc or next((x.noun for fld, x in cl.obls if fld == "L" and isinstance(x, NP)), None)
    if place:
        f["place"] = sorted(forms.m.get(place, {place}))
    if cl.tense:
        f["tense"] = cl.tense
    if cl.verb:
        f["negated"] = "yes" if cl.neg else "no"
    return f


def build(n_test: int = 300, n_train: int = 3000) -> dict:
    R2.set_held_out(0)
    gen = R2.GenV2("tamil_like", "abugida_pua", 0)
    forms = Forms(gen)
    rng = random.Random(31337)
    seen, rows = set(), []
    while len(rows) < n_test + n_train:
        ex = gen.example(rng, force=dict(CLEAN), n_events=1)
        it = gen.item(ex) if ex else None
        if not it or it["english"] in seen:
            continue
        cl = ex["clauses"][0]
        if cl.link or cl.intj or cl.voc is not None:
            continue
        f = facts(cl, forms)
        if "action" not in f:
            continue
        seen.add(it["english"])
        rows.append({"english": it["english"], "analysis": it["analysis"], "facts": f})
    return {"test": rows[:n_test], "train": rows[n_test:]}


def data() -> dict:
    p = OUT / "data.json"
    if not p.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(build(), ensure_ascii=False), encoding="utf-8")
    return json.loads(p.read_text(encoding="utf-8"))


# --- scoring -----------------------------------------------------------------------------------------------------------
def norm_words(s: str) -> set[str]:
    return set(re.findall(r"[a-z]+", s.lower()))


def right(fact: str, gold, answer: str) -> bool:
    a = answer.lower()
    if fact == "tense":
        found = [t for t in ("past", "present", "future") if t in a]
        return found[:1] == [gold]
    if fact in ("negated", "number"):
        opts = ("yes", "no") if fact == "negated" else ("one", "many")
        found = [o for o in opts if re.search(rf"\b{o}\b", a)]
        return found[:1] == [gold]
    return bool(norm_words(answer) & set(gold))


def summarise(name: str, test: list[dict], answers: list[dict]) -> dict:
    per = collections.defaultdict(lambda: [0, 0])
    sent_ok = 0
    for row, ans in zip(test, answers):
        ok_all = True
        for fact, gold in row["facts"].items():
            if fact not in ans:
                continue
            ok = right(fact, gold, ans[fact])
            per[fact][0] += ok
            per[fact][1] += 1
            ok_all &= ok
        sent_ok += ok_all
    tot = [sum(v[0] for v in per.values()), sum(v[1] for v in per.values())]
    res = {"method": name, "word": tot[0] / max(1, tot[1]), "sentence": sent_ok / len(test),
           "per_fact": {k: v[0] / v[1] for k, v in per.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{re.sub(r'[^A-Za-z0-9.-]+', '_', name)}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return res


# --- models --------------------------------------------------------------------------------------------------------------
def load(model: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float16)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    return m.to("cuda").eval(), tok


@torch.no_grad()
def complete(m, tok, prompts: list[str], max_new: int, batch: int = 16) -> list[str]:
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True).to("cuda")
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        out += [t.split("\n")[0].strip() for t in tok.batch_decode(g[:, enc["input_ids"].shape[1]:],
                                                                  skip_special_tokens=True)]
    return out


QUESTIONS = {"doer": "Who or what does the action?", "action": "What is the action (the verb)?",
             "thing done to": "Who or what is the action done to?", "place": "Where does it happen?",
             "tense": "Is it past, present or future?", "negated": "Is the sentence negated (yes or no)?",
             "number": "Is the doer one or many?"}
QA_SHOTS = [("The old farmer did not sell the two horses in the market.",
             {"doer": "farmer", "action": "sell", "thing done to": "horses", "place": "market", "tense": "past",
              "negated": "yes", "number": "one"}),
            ("The children will paint the fence.",
             {"doer": "children", "action": "paint", "thing done to": "fence", "tense": "future", "negated": "no",
              "number": "many"}),
            ("She sleeps in the kitchen.",
             {"doer": "she", "action": "sleep", "place": "kitchen", "tense": "present", "negated": "no",
              "number": "one"})]


def cmd_qa(model: str) -> None:
    d = data()
    m, tok = load(model)
    prompts, keys = [], []
    for i, row in enumerate(d["test"]):
        for fact in row["facts"]:
            shots = "".join(f"Sentence: {s}\nQuestion: {QUESTIONS[fact]}\nAnswer: {a[fact]}\n\n"
                            for s, a in QA_SHOTS if fact in a)
            prompts.append(shots + f"Sentence: {row['english']}\nQuestion: {QUESTIONS[fact]}\nAnswer:")
            keys.append((i, fact))
    outs = complete(m, tok, prompts, 8)
    answers = [dict() for _ in d["test"]]
    for (i, fact), o in zip(keys, outs):
        answers[i][fact] = o
    summarise(f"method1-questions__{model}", d["test"], answers)


FMT_SHOTS = [("The old farmer did not sell the two horses in the market.",
              "S=old+farmer V=sell.PAST.NEG O=two+horse.PL L=market"),
             ("The children will paint the fence.", "S=child.PL V=paint.FUT O=fence"),
             ("She sleeps in the kitchen.", "S=PRON.3sgf V=sleep.PRES L=kitchen")]


def fmt_to_answers(text: str) -> dict:
    """Read our meaning format back into facts, leniently (only heads and grammar tags matter)."""
    f = dict(re.findall(r"\b([SVOL])=(\S+)", text))
    ans = {}

    def head(v):
        v = v.split("^")[-1].split("&")[0]
        v = re.sub(r"\.PL$", "", v).split("+")[-1]
        if v.startswith("PRON."):
            return " ".join(PRON_FORMS.get(v[5:], {v[5:]}))
        if v.startswith("NAME."):
            return v[5:]
        return v.split(">")[0].replace("_", " ")
    if "S" in f:
        ans["doer"] = head(f["S"])
        ans["number"] = "many" if f["S"].endswith(".PL") or f["S"] in ("PRON.1pl", "PRON.2pl", "PRON.3pl") else "one"
    if "V" in f:
        parts = f["V"].split(".")
        ans["action"] = parts[0].split(">")[0]
        ans["tense"] = {"PAST": "past", "PRES": "present", "FUT": "future"}.get(next((p for p in parts[1:]
                                                                                     if p in ("PAST", "PRES", "FUT")),
                                                                                    ""), "")
        ans["negated"] = "yes" if "NEG" in parts else "no"
    if "O" in f:
        ans["thing done to"] = head(f["O"])
    if "L" in f:
        ans["place"] = head(f["L"])
    return ans


def cmd_fmt(model: str) -> None:
    d = data()
    m, tok = load(model)
    shots = "".join(f"Sentence: {s}\nMeaning: {a}\n\n" for s, a in FMT_SHOTS)
    outs = complete(m, tok, [shots + f"Sentence: {r['english']}\nMeaning:" for r in d["test"]], 60)
    answers = []
    for row, o in zip(d["test"], outs):
        a = fmt_to_answers(o)
        answers.append({k: a.get(k, "") for k in row["facts"]})        # a missing field counts as wrong
    summarise(f"method2-format__{model}", d["test"], answers)


def cmd_trained(run: str) -> None:
    from train.load import load_trained, translate
    d = data()
    m, tok, _ = load_trained(ROOT / "results" / "kaggle" / run / "model")
    outs = []
    for i in range(0, len(d["test"]), 8):
        outs += translate(m, tok, [r["english"] for r in d["test"][i:i + 8]], max_new_tokens=700)
    answers = []
    for row, o in zip(d["test"], outs):
        a = fmt_to_answers(o.partition("=>")[0])
        answers.append({k: a.get(k, "") for k in row["facts"]})
    summarise(f"trained-meaning__{run}", d["test"], answers)


# --- method 3: probes ------------------------------------------------------------------------------------------------
@torch.no_grad()
def features(m, tok, rows: list[dict], layers: list[int]):
    """Per sentence: mean hidden state (per layer); per word: mean over its tokens (per layer)."""
    sent = {l: [] for l in layers}
    words = {l: [] for l in layers}
    word_labels = []
    tok.padding_side = "right"
    for r in rows:
        text = r["english"]
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        hs = m(**enc.to("cuda"), output_hidden_states=True).hidden_states
        spans = [(mm.start(), mm.end(), mm.group().lower()) for mm in re.finditer(r"[A-Za-z']+", text)]
        roles = {}
        for fact in ("doer", "action", "thing done to", "place"):
            for g in r["facts"].get(fact, []):
                roles.setdefault(g, fact)
        for l in layers:
            h = hs[l][0].float()
            sent[l].append(h.mean(0).cpu())
            for s0, s1, w in spans:
                idx = [i for i, (a, b) in enumerate(offs) if a < s1 and b > s0]
                if idx:
                    words[l].append(h[idx].mean(0).cpu())
        for s0, s1, w in spans:
            if any(a < s1 and b > s0 for a, b in offs):
                word_labels.append(roles.get(w, "other"))
    return ({l: torch.stack(v).numpy() for l, v in sent.items()},
            {l: torch.stack(v).numpy() for l, v in words.items()}, word_labels)


def cmd_probe(model: str) -> None:
    import time
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def clf():                      # hidden states have a few huge dimensions: scale first, or lbfgs crawls
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.5))
    d = data()
    m, tok = load(model)
    t0 = time.time()
    n_layers = m.config.num_hidden_layers
    layers = sorted({n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers})
    s_tr, w_tr, wl_tr = features(m, tok, d["train"], layers)
    s_te, w_te, wl_te = features(m, tok, d["test"], layers)
    print(f"features done in {time.time() - t0:.0f} s; {len(wl_tr)} train words", flush=True)
    res = {"method": f"method3-probe__{model}", "layers": {}}
    for l in layers:
        out = {}
        pred = clf().fit(w_tr[l], wl_tr).predict(w_te[l])
        out["word role (doer/action/object/place/other)"] = float((pred == __import__("numpy").array(wl_te)).mean())
        for fact in ("tense", "negated", "number"):
            ytr = [r["facts"].get(fact) for r in d["train"]]
            yte = [r["facts"].get(fact) for r in d["test"]]
            itr = [i for i, y in enumerate(ytr) if y]
            ite = [i for i, y in enumerate(yte) if y]
            c = clf().fit(s_tr[l][itr], [ytr[i] for i in itr])
            out[fact] = float((c.predict(s_te[l][ite]) == __import__("numpy").array([yte[i] for i in ite])).mean())
        res["layers"][l] = out
        print(f"layer {l}: {out}  ({time.time() - t0:.0f} s)", flush=True)
    major = collections.Counter(wl_te).most_common(1)[0][1] / len(wl_te)
    res["chance"] = {"word role (majority class)": major, **CHANCE}
    best = max(res["layers"].values(), key=lambda o: o["word role (doer/action/object/place/other)"])
    res["best_layer"] = best
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{re.sub(r'[^A-Za-z0-9.-]+', '_', res['method'])}.json").write_text(json.dumps(res, indent=1),
                                                                                encoding="utf-8")
    print(json.dumps(res, indent=1))


def cmd_table() -> None:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*.json")) if p.name != "data.json"]
    facts_order = ["doer", "action", "thing done to", "place", "tense", "negated", "number"]
    out = ["# English-understanding ceiling (generated statements, 300 test sentences)", "",
           "Facts per sentence from its known meaning. Word level = share of facts right; sentence level = all of a "
           "sentence's facts right. Chance: tense 0.33, negated 0.50, number 0.50; open questions ~0. "
           "`ceiling/ceiling.py`.", "",
           "| method | word level | sentence level | " + " | ".join(facts_order) + " |",
           "|---|---|---|" + "---|" * len(facts_order)]
    for r in rows:
        if "per_fact" in r:
            out.append(f"| {r['method']} | {r['word']:.3f} | {r['sentence']:.3f} | "
                       + " | ".join(f"{r['per_fact'].get(f, float('nan')):.3f}" for f in facts_order) + " |")
    for r in rows:
        if "layers" in r:
            out += ["", f"## {r['method']} (accuracy of linear probes; chance {r['chance']})", "",
                    "| layer | " + " | ".join(next(iter(r["layers"].values()))) + " |",
                    "|---|" + "---|" * len(next(iter(r["layers"].values())))]
            for l, o in r["layers"].items():
                out.append(f"| {l} | " + " | ".join(f"{v:.3f}" for v in o.values()) + " |")
    (ROOT / "results" / "ceiling.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["qa", "fmt", "probe", "trained", "table", "data"])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--run", default="e2b-v2fix-cot-80k-noheld")
    a = ap.parse_args()
    {"qa": lambda: cmd_qa(a.model), "fmt": lambda: cmd_fmt(a.model), "probe": lambda: cmd_probe(a.model),
     "trained": lambda: cmd_trained(a.run), "table": cmd_table, "data": data}[a.cmd]()
