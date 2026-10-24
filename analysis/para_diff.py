"""How different is the paraphrase run's training English from the baseline's? (local, no GPU)

    python analysis/para_diff.py      -> results/para_diff.md + results/fig_para_diff.png

The para run (e2b-v2fix-para) trains on exactly the baseline's 80k examples, except that 1,106 of them have their
English replaced by a paraphrase that the reader reads to the same meaning (results/kaggle/e2b-v2fix-para/
paraphrases_seed0.json). For those examples this measures:
  1. how much the English changed: word-level edit distance, share of words moved / added / removed
  2. what kind of change: contraction, phrase moved to the front, passive <-> active, 'there is/are', other order
  3. NEW English for the model: a sentence's "skeleton" keeps the function words (the, did, by, there, n't ...) and
     replaces every content word by W. A skeleton never seen in the baseline's training data is a sentence pattern the
     baseline never trained on. Same for function-word bigrams.
And for the whole training set: how many skeletons / bigrams the para run adds (the honest size of the change).
"""

from __future__ import annotations

import collections
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sealed" / "app"))

from english_parse import FUNCTION  # noqa: E402

FUNC = {w.lower() for w in FUNCTION} | {"n't", "'ll", "'re", "'s", "'ve", "'d", "'m", "by", "there", "it", "that",
                                         "not", "do", "does", "did", "will", "shall", "be", "is", "are", "was", "were",
                                         "been", "being", "have", "has", "had", "to", "of", "in", "on", "at"}


def toks(s: str) -> list[str]:
    s = s.lower()
    s = re.sub(r"(\w)(n't|'ll|'re|'s|'ve|'d|'m)\b", r"\1 \2", s)
    return re.findall(r"[a-z]+|n't|'[a-z]+|[,.!?;]", s)


def skeleton(s: str) -> str:
    out = []
    for t in toks(s):
        if not (t[0].isalpha() or t[0] == "'" or t == "n't"):
            continue                                   # punctuation ignored: adding a full stop is not a new pattern
        w = t if t in FUNC else "W"
        if not (w == "W" and out and out[-1] == "W"):
            out.append(w)
    return " ".join(out)


def fbigrams(s: str) -> set:
    t = [x for x in toks(s) if x in FUNC]
    return set(zip(t, t[1:]))


def change_types(a: str, b: str) -> list[str]:
    ta, tb = toks(a), toks(b)
    kinds = []
    if any(x in tb for x in ("n't", "'ll", "'re", "'s", "'ve", "'d", "'m")) and not any(
            x in ta for x in ("n't", "'ll", "'re", "'ve", "'d", "'m")):
        kinds.append("contraction")
    if " by " in f" {b.lower()} " and " by " not in f" {a.lower()} ":
        kinds.append("passive (by ...)")
    if " by " in f" {a.lower()} " and " by " not in f" {b.lower()} ":
        kinds.append("active (from passive)")
    if re.match(r"(?i)\s*there (is|are|was|were|will|might|may|can|could|should|must)", b) and not \
            re.match(r"(?i)\s*there ", a):
        kinds.append("there is / are")
    if re.match(r"(?i)\s*it (is|was) .* that", b) and not re.match(r"(?i)\s*it (is|was)", a):
        kinds.append("it was ... that")
    ca = [x for x in ta if x not in FUNC and x[0].isalpha()]
    cb = [x for x in tb if x not in FUNC and x[0].isalpha()]
    if sorted(ca) == sorted(cb) and ca != cb:
        kinds.append("content words reordered")
        if "," in b and "," not in a:
            kinds.append("phrase moved to the front")
    if sorted(ca) != sorted(cb):
        kinds.append("word changed (same meaning)")
    if not kinds:
        kinds.append("small words / punctuation only")
    return kinds


def main() -> None:
    para = json.loads((ROOT / "results" / "kaggle" / "e2b-v2fix-para" / "paraphrases_seed0.json")
                      .read_text(encoding="utf-8"))
    base_en = json.loads((ROOT / "results" / "subgraph" / "e2b-v2fix-cot-80k-noheld.pieces.json")
                         .read_text(encoding="utf-8"))["train_en"]
    base_en = [s for s in base_en if len(s) < 300]                 # translation examples (replay passages out)
    base_sk = collections.Counter(skeleton(s) for s in base_en)
    base_bg = collections.Counter(b for s in base_en for b in fbigrams(s))
    rows = []
    for a, b in para.items():
        ta, tb = toks(a), toks(b)
        sm = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
        ed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
        rows.append({"a": a, "b": b, "edit": ed / max(len(ta), 1), "kinds": change_types(a, b),
                     "new_skel": base_sk[skeleton(b)] == 0, "old_skel_new": base_sk[skeleton(a)] == 0,
                     "new_bigrams": len([x for x in fbigrams(b) if base_bg[x] == 0])})
    n = len(rows)
    kinds = collections.Counter(k for r in rows for k in r["kinds"])
    new_sk = sum(r["new_skel"] for r in rows)
    new_sk_set = {skeleton(r["b"]) for r in rows if r["new_skel"]}
    new_bg_set = {x for r in rows for x in fbigrams(r["b"]) if base_bg[x] == 0}
    eds = sorted(r["edit"] for r in rows)
    med = eds[n // 2]
    para_sk = collections.Counter(base_sk)
    for a, b in para.items():                                          # the para run's training skeletons
        para_sk[skeleton(a)] -= 1
        para_sk[skeleton(b)] += 1
    ctrl = controls(base_en, base_sk, para)
    clf = classify(base_en, para)
    cons = constructions(base_en, para)
    L = ["# How different is the paraphrase run's training English? (`analysis/para_diff.py`)", "",
         f"The para run trains on the baseline's examples with **{n:,} of about 80,000 (1.4%)** given reworded "
         "English (same meaning, checked by the reader). Below: how much those examples changed, and how much NEW "
         "English they add.", "",
         "## 1. How much the changed examples changed", "",
         f"- Word-level edit distance (share of the sentence's words moved / added / removed / replaced): median "
         f"**{med:.2f}**, mean {sum(eds) / n:.2f}; {sum(e >= .25 for e in eds) / n:.0%} of them changed a quarter "
         f"of their words or more.",
         f"- New sentence patterns: {new_sk / n:.0%} of paraphrases have a function-word pattern absent from the "
         f"baseline's training data, but so do {ctrl['fresh_new']:.0%} of FRESH generated sentences (the generator is "
         f"very varied), so this alone shows nothing. For short sentences (<= 8 words): paraphrases "
         f"{ctrl['para_short_new']:.0%} vs fresh generated {ctrl['fresh_short_new']:.0%}.",
         f"- They contain **{len(new_bg_set)}** function-word pairs the baseline never trained on (e.g. "
         + ", ".join(f"'{a} {b}'" for a, b in sorted(new_bg_set)[:10]) + ").", "",
         "", "## 2. Statistical test: is the paraphrased English a different kind of English?", "",
         "A logistic-regression classifier sees only the FUNCTION words of a sentence (1-3-word sequences of the, by, "
         "did, n't, there ...; content words, capitals and punctuation removed) and must tell two groups apart; "
         "5-fold cross-validated accuracy, 50% = chance.", "",
         "| groups | sentences per group | accuracy | p (vs chance) |", "|---|---|---|---|"]
    for name, (acc, m, pv) in clf.items():
        L.append(f"| {name} | {m} | **{acc:.2f}** | {pv:.1e} |")
    L += ["", "## 3. Constructions: how often each appears", "",
          "| construction | baseline training | paraphrases | their originals |", "|---|---|---|---|"]
    for k, (x, y, z) in cons.items():
        L.append(f"| {k} | {x:.1%} | {y:.1%} | {z:.1%} |")
    L += ["", "## 4. Kinds of change in the paraphrased examples", "",
          "| kind of change | paraphrases | share |", "|---|---|---|"]
    for k, c in kinds.most_common():
        L.append(f"| {k} | {c} | {c / n:.0%} |")
    L += ["", "(one paraphrase can have several kinds)", "",
          "## 5. But the training set as a whole changed little", "",
          f"- Examples changed: {n:,} of ~80,000 = **1.4%**.",
          f"- New sentence patterns added: **{len(new_sk_set)}** distinct skeletons, against {len(base_sk):,} "
          f"distinct skeletons in the baseline's training data (+{len(new_sk_set) / len(base_sk):.1%}); each new "
          f"pattern appears ~{new_sk / max(1, len(new_sk_set)):.1f} times, well under the ~10 sightings a word needs "
          "(claim 3).",
          "- So the para run saw clearly different English, but too little of it: this is why the result is "
          "inconclusive rather than negative.", "",
          "## Examples (largest changes)", ""]
    for r in sorted(rows, key=lambda r: -r["edit"])[:12]:
        L.append(f"- {r['a']}  \n  → **{r['b']}**  ({', '.join(r['kinds'])}; edit {r['edit']:.2f}"
                 f"{'; NEW pattern' if r['new_skel'] else ''})")
    (ROOT / "results" / "para_diff.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    plot(rows, kinds, n, cons, clf)
    print("\n".join(L[:20]))


def fword_text(s: str) -> str:
    return " ".join(t for t in toks(s) if t in FUNC or t in ("n't", "'ll", "'re", "'ve", "'d", "'m", "'s"))


def classify(base_en: list[str], para: dict) -> dict:
    import random
    from scipy.stats import binomtest
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    rng = random.Random(0)
    P = list(para.values())
    O = list(para.keys())
    B = rng.sample(base_en, 2 * len(P))
    tests = {"paraphrases vs their own originals": (P, O),
             "paraphrases vs baseline training sentences": (P, B[:len(P)]),
             "their originals vs baseline training sentences (control)": (O, B[:len(O)]),
             "baseline vs baseline, two random halves (control)": (B[:len(P)], B[len(P):])}
    out = {}
    for name, (a, b) in tests.items():
        X = [fword_text(x) for x in a + b]
        y = [1] * len(a) + [0] * len(b)
        vec = CountVectorizer(ngram_range=(1, 3), token_pattern=r"[^ ]+", min_df=2)
        Xv = vec.fit_transform(X)
        pred = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), Xv, y, cv=5)
        k = sum(int(p == t) for p, t in zip(pred, y))
        out[name] = (k / len(y), len(a), binomtest(k, len(y), 0.5, alternative="greater").pvalue)
    return out


CONS = {"passive with 'by'": r"\b(is|are|was|were|be|been|being)\s+\w+\s+by\b",
        "contraction (n't 'll 're 've 'd 'm)": r"(n't|'ll|'re|'ve|'d|'m)\b",
        "phrase moved to the front (... ,)": r"^(in|on|at|with|without|behind|under|near|for|from|to|after|before|during"
                                             r"|until|since|through|around|between|against|about|yesterday|today|"
                                             r"tomorrow|tonight|soon|now|then|here|quite|really|very|next|last|every)"
                                             r"\b[^,]*,",
        "'there is / are'": r"^there (is|are|was|were|will|might|may|can|could|should|must|would)\b",
        "'it was ... that'": r"^it (is|was) .+ that\b",
        "'by who / by what'": r"\bby (who|what|whom)\b"}


def constructions(base_en: list[str], para: dict) -> dict:
    out = {}
    for k, rx in CONS.items():
        r = re.compile(rx, re.I)
        f = lambda xs: sum(bool(r.search(x.strip())) for x in xs) / len(xs)  # noqa: E731
        out[k] = (f(base_en), f(list(para.values())), f(list(para.keys())))
    return out


def controls(base_en, base_sk, para) -> dict:
    sys.path.insert(0, str(ROOT / "analysis"))
    import train.represent_v2 as R2
    from freq_weighted_eval import test_items
    bset = set(base_en)
    gen = R2.GenV2("tamil_like", "abugida_pua", 0)
    fresh = [it["english"] for it, _ in test_items(gen, 700) if it["english"] not in bset][:600]
    short = [x for x in fresh if len(toks(x)) <= 8]
    ps = [b for a, b in para.items() if len(toks(a)) <= 8]
    new = lambda xs: sum(base_sk[skeleton(x)] == 0 for x in xs) / max(1, len(xs))  # noqa: E731
    return {"fresh_new": new(fresh), "fresh_short_new": new(short), "para_short_new": new(ps)}


def plot(rows, kinds, n, cons, clf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2), gridspec_kw={"width_ratios": [1, 1.2, 1.4, 1.1]})
    ax[0].hist([min(r["edit"], 2) for r in rows], bins=20, color="#4a6fd0")
    ax[0].set(xlabel="share of words changed (edit distance / length)", ylabel="paraphrased examples",
              title=f"How much each of the {n:,} changed")
    ks = kinds.most_common()
    ax[1].barh([k for k, _ in ks][::-1], [c / n for _, c in ks][::-1], color="#4a6fd0")
    ax[1].set(xlabel="share of paraphrases", title="Kind of change")
    names = [k for k in cons if max(cons[k]) >= 0.005]
    y = range(len(names))
    ax[2].barh([i + 0.2 for i in y], [cons[k][0] for k in names], 0.4, color="#999999", label="baseline training")
    ax[2].barh([i - 0.2 for i in y], [cons[k][1] for k in names], 0.4, color="#2e8b57", label="paraphrases")
    ax[2].set_yticks(list(y), names)
    ax[2].set(xlabel="share of sentences", title="Constructions")
    ax[2].legend(fontsize=8)
    labels = {"paraphrases vs their own originals": "para vs their originals",
              "paraphrases vs baseline training sentences": "para vs baseline",
              "their originals vs baseline training sentences (control)": "originals vs baseline",
              "baseline vs baseline, two random halves (control)": "baseline vs baseline"}
    xs = [labels.get(k, k) for k in clf]
    ax[3].bar(xs, [v[0] for v in clf.values()], color=["#2e8b57", "#2e8b57", "#999999", "#999999"])
    ax[3].axhline(0.5, color="black", lw=1, ls="--")
    ax[3].set(ylim=(0.4, 0.75), ylabel="classifier accuracy (5-fold)", title="Tell them apart by function words?")
    ax[3].tick_params(axis="x", labelsize=8, labelrotation=25)
    ax[3].text(3.4, 0.505, "chance", fontsize=8, ha="right")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    try:
        fig.savefig(ROOT / "results" / "fig_para_diff.png", dpi=150)
    except OSError as e:
        print("figure not saved:", e)


if __name__ == "__main__":
    main()
