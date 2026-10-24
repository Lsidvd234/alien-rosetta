"""Stress tests for the hard-language invented-glyph models (all local, no retraining).

    python analysis/stress_tests.py embed    # 1. where did the glyph embeddings end up?
    python analysis/stress_tests.py freq     # 2. word accuracy by how often the word was seen in training
    python analysis/stress_tests.py grid     # 3. every ending combination, scored separately (needs a GPU, ~5 min)

Writes results/stress/<test>.md (+ figures in results/figures/).
Set HF_HOME to the cache holding Qwen3-0.6B-Base (D:/Downloads/wm-feasibility/hf_cache).
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conlang.dataset import PRESETS, TEST_SPLITS, build, heldout_pair  # noqa: E402
from conlang.frames import NP, PRONOUNS, TENSES, Frame, Vocab, english  # noqa: E402
from conlang.grammar import Language  # noqa: E402
from conlang.script import Script, render_sentence  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "kaggle"
OUT = ROOT / "results" / "stress"
BASE = "Qwen/Qwen3-0.6B-Base"

# runs with saved weights (hard language, glyphs as 1 new token each, 50k)
MODELS = {"Option 1: Latin-sound start": "s1-hard-latininit-50k",
          "Option 2: + transliteration": "s2-hard-translit-50k",
          "Option 3a: plain text first": "s3-hard-plaintext-50k",
          "Control: shuffled-sound start": "s4-hard-shuffledinit-50k"}
# runs compared offline from their saved predictions (no weights needed)
FREQ_RUNS = {"Glyphs, random-word start (before)": "p-hard-puatok-50k",
             "Option 1: Latin-sound start": "g1-hard-latininit-50k",
             "Option 2: + transliteration": "g2-hard-translit-50k",
             "Option 3a: plain text first": "g3-hard-plaintext-50k",
             "Control: shuffled-sound start": "s4-hard-shuffledinit-50k",
             "Latin script": "c-hard-latin-50k",
             "Tamil script": "g-hard-tamil-50k"}


def available(models: dict) -> dict:
    return {k: v for k, v in models.items() if (RUNS / v / "results" / f"{v}.json").exists()}


def write(name: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (OUT / f"{name}.md").write_text(text, encoding="utf-8")
    print(text)


def stem(gloss_word: str) -> str:
    return gloss_word.split("-")[0].split(".")[0]


# --- 1. glyph embeddings ---------------------------------------------------------------

def base_embeddings() -> torch.Tensor:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    path = hf_hub_download(BASE, "model.safetensors")
    with safe_open(path, "pt") as f:
        return f.get_tensor("model.embed_tokens.weight").float()


def test_embed() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    W = base_embeddings()
    n_old = len(tok)
    cos = torch.nn.functional.cosine_similarity

    lines = ["# Test 1: where did the glyph embeddings end up?", "",
             "Each glyph's embedding (trained, and at the start of training) is compared with the embedding",
             "of the Latin letters for its true sound, taken from Qwen's own table.", "",
             "- **similarity to own sound**: mean cosine similarity between a glyph and its sound's embedding",
             "- **nearest sound = own sound**: share of glyphs whose most similar sound (among all the "
             "script's sounds) is their own",
             "- **sound-structure match**: correlation between how similar glyphs are to each other and how "
             "similar their sounds are (1 = the glyphs reproduce the sound layout)",
             "- **moved**: how far training moved the embedding, relative to its starting length", "",
             "| model | similarity to own sound (start -> end) | nearest sound = own (start -> end) | "
             "sound-structure match (start -> end) | moved |",
             "|---|---|---|---|---|"]
    examples = []
    for label, run in available(MODELS).items():
        g = torch.load(RUNS / run / "model" / "glyph_tokens.pt", map_location="cpu", weights_only=True)
        keep = [i for i, s in enumerate(g["sounds"]) if s]           # the virama has no sound
        sounds = [g["sounds"][i] for i in keep]
        uniq = sorted(set(sounds))
        S = torch.stack([W[tok(s, add_special_tokens=False)["input_ids"]].mean(0) for s in uniq])
        own = torch.tensor([uniq.index(s) for s in sounds])

        def stats(D: torch.Tensor) -> tuple[float, float, float]:
            D = D[keep]
            sim = cos(D[:, None], S[None], dim=-1)                   # glyph x sound
            own_sim = sim[torch.arange(len(keep)), own].mean().item()
            nearest = (sim.argmax(1) == own).float().mean().item()
            gg = cos(D[:, None], D[None], dim=-1)
            ss = cos(S[own][:, None], S[own][None], dim=-1)
            m = ~torch.eye(len(keep), dtype=torch.bool)
            r = torch.corrcoef(torch.stack([gg[m], ss[m]]))[0, 1].item()
            return own_sim, nearest, r

        a, b = stats(g["delta_init"]), stats(g["delta"])
        moved = ((g["delta"] - g["delta_init"]).norm(dim=1) / g["delta_init"].norm(dim=1))[keep].mean().item()
        lines.append(f"| {label} | {a[0]:.2f} -> {b[0]:.2f} | {a[1]:.0%} -> {b[1]:.0%} | "
                     f"{a[2]:.2f} -> {b[2]:.2f} | {moved:.0%} |")
        # closest ordinary tokens to a few trained glyphs
        norm = torch.nn.functional.normalize
        sim_all = norm(g["delta"][keep], dim=1) @ norm(W[:n_old], dim=1).T
        top = sim_all.topk(5, dim=1).indices
        for i in range(0, len(keep), max(1, len(keep) // 6))[:6]:
            near = ", ".join(repr(tok.decode([t]).strip()) for t in top[i].tolist())
            examples.append(f"| {label} | {sounds[i]} | {near} |")
    lines += ["", "## Closest ordinary tokens to some trained glyphs", "",
              "| model | glyph's true sound | 5 closest tokens in Qwen's vocabulary |", "|---|---|---|"] + examples
    write("test1_embeddings", lines)


# --- 2. accuracy by word frequency ------------------------------------------------------

BUCKETS = [(1, 15), (16, 25), (26, 35), (36, 45), (46, 60), (61, 10**9)]


def bucket_name(a: int, b: int) -> str:
    return f"{a}+" if b >= 10**9 else f"{a}-{b}"


def test_freq() -> None:
    from transformers import AutoTokenizer
    from train.run import unknown_chars

    lines = ["# Test 2: content-word accuracy by how often the word appeared in training", "",
             "All four test splits pooled. A content word counts as right only if every token of it is right",
             "(teacher-forced). Frequency = number of training sentences (of the 50k) containing the word's stem.", ""]
    cache: dict = {}
    table: dict[str, dict] = {}
    counts_shown = None
    for label, run in available(FREQ_RUNS).items():
        r = json.loads((RUNS / run / "results" / f"{run}.json").read_text(encoding="utf-8"))
        a = r["args"]
        key = (a["preset"], a["script"])
        if key not in cache:
            data = build(PRESETS[a["preset"]], a["script"], n_train=50000, n_test=int(a["n_test"]), seed=int(a["seed"]))
            freq = Counter(s for it in data["splits"]["train"][:int(a["n_train"])]
                           for s in {stem(w) for w in it["gloss"].split()})
            cache[key] = (data, freq)
        data, freq = cache[key]
        tok = AutoTokenizer.from_pretrained(BASE)
        if a.get("add_tokens") == "True":
            tok.add_tokens(unknown_chars(tok, data))
        V = Vocab.large()
        content = set(V.nouns) | set(V.locations) | set(V.transitive + V.intransitive) | set(V.adjectives)
        res = {bucket_name(*b): [0, 0] for b in BUCKETS}
        for split in TEST_SPLITS:
            tf = r["eval"][split]["teacher_forced"]
            items = {it["id"]: it for it in data["splits"][split]}
            for iid, ok in zip(tf["per_item_ids"], tf["per_item_ok"]):
                it = items[iid]
                t, gl = it["target"], it["gloss"].split()
                offs = tok(t, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
                words: dict[int, list[bool]] = {}
                for (st, _), ch in zip(offs, ok[:-1]):
                    if st < len(t) and t[st] == " ":
                        st += 1
                    words.setdefault(t[:st].count(" "), []).append(ch == "1")
                for wi, oks in words.items():
                    s = stem(gl[wi])
                    if s not in content:
                        continue
                    n = freq.get(s, 0)
                    for b in BUCKETS:
                        if b[0] <= n <= b[1]:
                            cell = res[bucket_name(*b)]
                            cell[0] += all(oks)
                            cell[1] += 1
        table[label] = res
        counts_shown = res
    names = [bucket_name(*b) for b in BUCKETS]
    lines += ["| model | " + " | ".join(f"seen {n}x" for n in names) + " |",
              "|---|" + "---|" * len(names)]
    for label, res in table.items():
        lines.append(f"| {label} | " + " | ".join(
            f"{res[n][0] / res[n][1]:.1%}" if res[n][1] else "-" for n in names) + " |")
    if counts_shown:
        lines += ["", "Test words per bucket: " + ", ".join(f"{n}x: {counts_shown[n][1]}" for n in names)]
    write("test2_word_frequency", lines)
    plot_freq(table, names)


def plot_freq(table: dict, names: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Glyphs, random-word start (before)": "#9e9e9e", "Option 1: Latin-sound start": "#2ca02c",
              "Option 2: + transliteration": "#17becf", "Option 3a: plain text first": "#ff7f0e",
              "Control: shuffled-sound start": "#8c564b", "Latin script": "#1f77b4", "Tamil script": "#d62728"}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for label, res in table.items():
        ys = [100 * res[n][0] / res[n][1] if res[n][1] else None for n in names]
        ax.plot(range(len(names)), ys, marker="o", color=colors.get(label), label=label, lw=2,
                ls="--" if label in ("Latin script", "Tamil script") else "-")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}" for n in names])
    ax.set_xlabel("Times the word appeared in the 50k training sentences")
    ax.set_ylabel("Content words fully correct (%)")
    ax.set_title("Hard language, 50k: accuracy by how often a word was seen in training", fontsize=11)
    ax.grid(alpha=0.3)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.1))
    out = ROOT / "results" / "figures"
    fig.savefig(out / "fig7_word_frequency.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "fig7_word_frequency.pdf", bbox_inches="tight")
    print("wrote fig7_word_frequency")


# --- 3. every ending combination ----------------------------------------------------------

def grid_frames(vocab: Vocab, train_words: set, train_en: set, seed: int, per_cell: int) -> list[tuple[dict, Frame]]:
    """Sentences for every combination of tense x negation x subject x object number. Only words
    seen in training, no adjectives/locations (so only the endings vary), no held-out verb+object
    pairs, and never a sentence that is in the training set."""
    rng = random.Random(1234)
    nouns = [n for n in vocab.nouns if f"N:{n}" in train_words]
    trans = [v for v in vocab.transitive if f"V:{v}" in train_words]
    intr = [v for v in vocab.intransitive if f"V:{v}" in train_words]
    subjects = [("noun, singular", False), ("noun, plural", True)] + [(f"pronoun {k}", k) for k in PRONOUNS]
    out = []
    for tense in TENSES:
        for neg in (False, True):
            for sname, s in subjects:
                for oname in ("no object", "object singular", "object plural"):
                    got, tries = 0, 0
                    while got < per_cell and tries < 5000:
                        tries += 1
                        subj = NP(rng.choice(nouns), s) if isinstance(s, bool) else s
                        if oname == "no object":
                            verb, obj = rng.choice(intr), None
                        else:
                            verb, obj = rng.choice(trans), NP(rng.choice(nouns), oname == "object plural")
                            if heldout_pair(seed, verb, obj.noun):
                                continue
                        f = Frame(subj, verb, obj, None, tense, neg)
                        if english(f, vocab) in train_en:
                            continue
                        out.append(({"tense": tense, "neg": neg, "subject": sname, "object": oname}, f))
                        got += 1
    return out


def test_grid(per_cell: int = 25) -> None:
    from train.load import load_trained
    from train.run import encode, teacher_forced

    models = {k: v for k, v in MODELS.items() if (RUNS / v / "model" / "glyph_tokens.pt").exists()}
    first = next(iter(models.values()))
    cfg = json.loads((RUNS / first / "model" / "run_args.json").read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    data = build(PRESETS[cfg["preset"]], cfg["script"], n_train=50000, n_test=int(cfg["n_test"]), seed=seed)
    train = data["splits"]["train"][:int(cfg["n_train"])]
    vocab = Vocab.large()
    lang = Language(PRESETS[cfg["preset"]], seed, vocab)
    script = Script(cfg["script"], PRESETS[cfg["preset"]].phonology)
    frames = grid_frames(vocab, {w for it in train for w in it["content"]}, {it["english"] for it in train},
                         seed, per_cell)
    items = []
    for feats, f in frames:
        real = lang.realize(f)
        items.append({"english": english(f, vocab), "target": render_sentence(real.words, script),
                      "gloss": " ".join(w.gloss for w in real.words), "feats": feats, "verb": f.verb})
    print(f"{len(items)} grid sentences")

    results: dict[str, list[bool]] = {}
    verb_ok: dict[str, list[bool]] = {}
    for label, run in models.items():
        model, tok, _ = load_trained(RUNS / run / "model")
        device = next(model.parameters()).device
        amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if device.type == "cuda" \
            else (lambda: torch.autocast("cpu", enabled=False))
        tf = teacher_forced(model, tok, [encode(tok, it) for it in items], 16, device, amp)
        results[label] = [set(s) == {"1"} for s in tf["per_item_ok"]]
        # was the verb word (stem + its endings) right?
        vo = []
        for it, ok in zip(items, tf["per_item_ok"]):
            t, gl = it["target"], it["gloss"].split()
            offs = tok(t, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            words: dict[int, list[bool]] = {}
            for (st, _), ch in zip(offs, ok[:-1]):
                if st < len(t) and t[st] == " ":
                    st += 1
                words.setdefault(t[:st].count(" "), []).append(ch == "1")
            vi = [i for i, g in enumerate(gl) if stem(g) == it["verb"]]
            vo.append(all(words.get(vi[-1], [False])) if vi else False)
        verb_ok[label] = vo
        print(label, f"exact {sum(results[label]) / len(items):.3f}")
        del model
        torch.cuda.empty_cache()
    report_grid(items, results, verb_ok)


def report_grid(items: list[dict], results: dict, verb_ok: dict) -> None:
    labels = list(results)

    def rate(xs, mask):
        sel = [x for x, m in zip(xs, mask) if m]
        return sum(sel) / len(sel) if sel else float("nan")

    heldout = [it["feats"]["tense"] == "past" and it["feats"]["neg"] and
               (it["feats"]["subject"] == "noun, plural" or it["feats"]["subject"] in
                ("pronoun 1pl", "pronoun 2pl", "pronoun 3pl")) for it in items]
    lines = ["# Test 3: every ending combination", "",
             f"{len(items)} new sentences: every tense x negation x subject type x object number, "
             f"~{len(items) // (3 * 2 * 10 * 3)} each.",
             "Only words seen in training; no adjectives or locations; no held-out verb+object pairs; none in the training set.",
             "Held out in training = past + negated + plural subject (the test_morph combination).",
             "Scores: whole sentence right (verb word right) — teacher-forced.", "",
             "## Seen vs held-out combinations", "",
             "| model | combinations seen in training | held-out (past + neg + plural subject) |", "|---|---|---|"]
    for lab in labels:
        seen = [not h for h in heldout]
        lines.append(f"| {lab} | {rate(results[lab], seen):.1%} ({rate(verb_ok[lab], seen):.1%}) | "
                     f"{rate(results[lab], heldout):.1%} ({rate(verb_ok[lab], heldout):.1%}) |")
    for feat, values in (("tense", list(TENSES)), ("neg", [False, True]),
                         ("subject", ["noun, singular", "noun, plural"] + [f"pronoun {k}" for k in PRONOUNS]),
                         ("object", ["no object", "object singular", "object plural"])):
        lines += ["", f"## By {feat}", "", "| " + feat + " | " + " | ".join(labels) + " |",
                  "|---|" + "---|" * len(labels)]
        for v in values:
            mask = [it["feats"][feat] == v for it in items]
            name = {False: "affirmative", True: "negated"}.get(v, v) if feat == "neg" else v
            lines.append(f"| {name} | " + " | ".join(
                f"{rate(results[l], mask):.1%} ({rate(verb_ok[l], mask):.1%})" for l in labels) + " |")
    # tense x negation x subject number, the cells around the held-out one
    lines += ["", "## Tense x negation x subject number (whole sentence right)", "",
              "| tense | negated | subject number | " + " | ".join(labels) + " |",
              "|---|---|---|" + "---|" * len(labels)]
    plural = ("noun, plural", "pronoun 1pl", "pronoun 2pl", "pronoun 3pl")
    for tense in TENSES:
        for neg in (False, True):
            for num in ("singular", "plural"):
                mask = [it["feats"]["tense"] == tense and it["feats"]["neg"] == neg and
                        ((it["feats"]["subject"] in plural) == (num == "plural")) for it in items]
                tag = " **(held out)**" if (tense == "past" and neg and num == "plural") else ""
                lines.append(f"| {tense} | {'yes' if neg else 'no'} | {num}{tag} | " + " | ".join(
                    f"{rate(results[l], mask):.1%}" for l in labels) + " |")
    write("test3_ending_grid", lines)
    (OUT / "test3_ending_grid_items.json").write_text(json.dumps(
        {"items": items, "exact": results, "verb_ok": verb_ok}, ensure_ascii=False), encoding="utf-8")


# --- shared: score saved models on new items -----------------------------------------------

class Ctx:
    """The run's language, script, training set and vocabulary, rebuilt from the saved run args."""

    def __init__(self):
        first = next(v for v in MODELS.values() if (RUNS / v / "model" / "run_args.json").exists())
        self.cfg = json.loads((RUNS / first / "model" / "run_args.json").read_text(encoding="utf-8"))
        self.seed = int(self.cfg["seed"])
        preset = PRESETS[self.cfg["preset"]]
        data = build(preset, self.cfg["script"], n_train=50000, n_test=int(self.cfg["n_test"]), seed=self.seed)
        self.train = data["splits"]["train"][:int(self.cfg["n_train"])]
        self.train_en = {it["english"] for it in self.train}
        self.train_words = {w for it in self.train for w in it["content"]}
        self.vocab = Vocab.large()
        self.lang = Language(preset, self.seed, self.vocab)
        self.script = Script(self.cfg["script"], preset.phonology)
        v = self.vocab
        self.nouns = [n for n in v.nouns if f"N:{n}" in self.train_words]
        self.locs = [n for n in v.locations if f"L:{n}" in self.train_words]
        self.adjs = [n for n in v.adjectives if f"A:{n}" in self.train_words]
        self.trans = [n for n in v.transitive if f"V:{n}" in self.train_words]
        self.intr = [n for n in v.intransitive if f"V:{n}" in self.train_words]

    def item(self, f: Frame, english_text: str | None = None, **extra) -> dict:
        real = self.lang.realize(f)
        return {"english": english_text or english(f, self.vocab), "target": render_sentence(real.words, self.script),
                "gloss": " ".join(w.gloss for w in real.words), **extra}


def run_models(items: list[dict]) -> dict[str, list[str]]:
    """Teacher-forced per-token correctness ('1'/'0' per target token, last = end of sentence)
    for every saved model."""
    from train.load import load_trained
    from train.run import encode, teacher_forced
    out = {}
    for label, run in MODELS.items():
        if not (RUNS / run / "model" / "glyph_tokens.pt").exists():
            continue
        model, tok, _ = load_trained(RUNS / run / "model")
        device = next(model.parameters()).device
        amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if device.type == "cuda" \
            else (lambda: torch.autocast("cpu", enabled=False))
        tf = teacher_forced(model, tok, [encode(tok, it) for it in items], 16, device, amp)
        out[label] = tf["per_item_ok"]
        print(label, f"exact {sum(set(s) == {'1'} for s in out[label]) / len(items):.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    return out


CRUDE = {"cum", "condom", "condoms", "dildo", "dildos", "sex", "porn", "penis", "vagina", "boob", "boobs",
         "nipple", "nipples", "orgasm", "sperm", "semen", "anus", "butt", "ass", "bitch", "bastard", "slut",
         "whore", "rape", "raped", "rapes", "pervert", "perverts", "perverted", "erection", "masturbate"}


def example(items: list[dict], idx: list[int], key: str = "english") -> str:
    """A short, clean example for the report table (the large English word list has a few crude words)."""
    ok = [items[i][key] for i in idx if not (set(items[i][key].lower().replace(".", "").split()) & CRUDE)]
    return min(ok, key=len) if ok else ""


def exact_rate(oks: list[str], idx: list[int]) -> float:
    return sum(set(oks[i]) == {"1"} for i in idx) / len(idx) if idx else float("nan")


# --- 4. rephrased English, same answer ---------------------------------------------------

SYN = {  # word -> (synonym, plural or 3rd-person form, past form); synonyms must not be words of the language
    "N": {"dog": ("hound", "hounds"), "car": ("automobile", "automobiles"), "child": ("kid", "kids"),
          "stone": ("rock", "rocks"), "ship": ("vessel", "vessels"), "woman": ("lady", "ladies"),
          "cup": ("mug", "mugs"), "doctor": ("physician", "physicians"), "teacher": ("instructor", "instructors"),
          "student": ("pupil", "pupils"), "bucket": ("pail", "pails"), "sofa": ("couch", "couches"),
          "lawyer": ("attorney", "attorneys"), "photo": ("picture", "pictures"), "bag": ("sack", "sacks"),
          "plate": ("dish", "dishes"), "coat": ("jacket", "jackets"), "pig": ("hog", "hogs"),
          "rabbit": ("bunny", "bunnies"), "kitten": ("kitty", "kitties")},
    "L": {"house": ("home",), "road": ("street",), "shop": ("store",), "forest": ("woods",),
          "river": ("stream",), "jail": ("prison",), "hill": ("mound",), "field": ("meadow",)},
    "A": {"big": ("large",), "small": ("little",), "quick": ("fast",), "happy": ("glad",), "smart": ("clever",),
          "old": ("aged",), "angry": ("mad",), "rich": ("wealthy",), "sad": ("unhappy",), "huge": ("enormous",),
          "beautiful": ("pretty",), "quiet": ("silent",), "wet": ("damp",), "strong": ("powerful",),
          "brave": ("courageous",), "stupid": ("foolish",), "odd": ("strange",), "rude": ("impolite",),
          "loud": ("noisy",), "ill": ("sick",)},
    "V": {"see": ("notice", "notices", "noticed"), "help": ("assist", "assists", "assisted"),
          "buy": ("purchase", "purchases", "purchased"), "hate": ("despise", "despises", "despised"),
          "fight": ("battle", "battles", "battled"), "try": ("attempt", "attempts", "attempted"),
          "seek": ("search", "searches", "searched"), "shout": ("yell", "yells", "yelled"),
          "choose": ("select", "selects", "selected"), "repair": ("fix", "fixes", "fixed"),
          "destroy": ("ruin", "ruins", "ruined"), "gather": ("collect", "collects", "collected"),
          "grab": ("seize", "seizes", "seized"), "steal": ("rob", "robs", "robbed"),
          "speak": ("talk", "talks", "talked"), "hide": ("conceal", "conceals", "concealed")},
}
IRREGULAR_PARTICIPLE = {"see": "seen", "take": "taken", "find": "found", "hold": "held", "catch": "caught",
                        "build": "built", "bring": "brought", "eat": "eaten", "write": "written", "break": "broken",
                        "steal": "stolen", "choose": "chosen", "hide": "hidden", "throw": "thrown", "give": "given",
                        "drive": "driven", "forget": "forgotten", "beat": "beaten", "shake": "shaken", "know": "known"}
OBJ_PRONOUN = {"1sg": "me", "2sg": "you", "3sgm": "him", "3sgf": "her", "3sgn": "it", "1pl": "us",
               "2pl": "you all", "3pl": "them"}


def passive(f: Frame, vocab: Vocab) -> str | None:
    """'the cat was seen by the dog' for a transitive frame (None if no participle is known)."""
    base, _, past = vocab.verb_info[f.verb]
    part = IRREGULAR_PARTICIPLE.get(f.verb) or (past if past.endswith("ed") else None)
    if part is None or f.obj is None:
        return None
    pl = f.obj.plural
    be = {"present": "are" if pl else "is", "past": "were" if pl else "was", "future": "will"}[f.tense]
    aux = [be, "not", "be"] if (f.tense == "future" and f.neg) else ([be, "be"] if f.tense == "future" else
                                                                      ([be, "not"] if f.neg else [be]))
    obj = " ".join(["the"] + ([f.obj.adj] if f.obj.adj else []) + [vocab.noun_info[f.obj.noun][0] if pl else f.obj.noun])
    if isinstance(f.subj, str):
        agent = OBJ_PRONOUN[f.subj]
    else:
        s = f.subj
        agent = " ".join(["the"] + ([s.adj] if s.adj else []) + [vocab.noun_info[s.noun][0] if s.plural else s.noun])
    out = f"{obj} {' '.join(aux)} {part} by {agent}"
    return out + (f" in the {f.loc}" if f.loc else "")


def synonym_english(f: Frame, vocab: Vocab, ctx_words: set) -> tuple[str, str] | None:
    """Replace one content word by a synonym that is NOT a word of the language. Returns (english, word)."""
    words = english(f, vocab).split()
    cands = []
    for np in (f.subj, f.obj):
        if isinstance(np, NP):
            if np.noun in SYN["N"]:
                syn = SYN["N"][np.noun]
                cands.append((vocab.noun_info[np.noun][0] if np.plural else np.noun, syn[1] if np.plural else syn[0]))
            if np.adj in SYN["A"]:
                cands.append((np.adj, SYN["A"][np.adj][0]))
    if f.loc in SYN["L"]:
        cands.append((f.loc, SYN["L"][f.loc][0]))
    if f.verb in SYN["V"]:
        base, third, past = vocab.verb_info[f.verb]
        sb, st, sp = SYN["V"][f.verb]
        cands += [(base, sb), (third, st), (past, sp)]
    for old, new in cands:
        if old in words and new.split()[0] not in ctx_words:
            i = words.index(old)
            return " ".join(words[:i] + [new] + words[i + 1:]), f"{old}->{new}"
    return None


def contracted(text: str) -> str | None:
    out = (text.replace("does not", "doesn't").replace("do not", "don't").replace("did not", "didn't")
           .replace("will not", "won't"))
    return out if out != text else None


def test_paraphrase(per_variant: int = 300) -> None:
    ctx = Ctx()
    v = ctx.vocab
    language_words = set(v.nouns) | set(v.locations) | set(v.transitive) | set(v.intransitive) | set(v.adjectives)
    syn_nouns = [n for n in SYN["N"] if n in ctx.nouns and SYN["N"][n][0] not in language_words]
    syn_adjs = [a for a in SYN["A"] if a in ctx.adjs and SYN["A"][a][0] not in language_words]
    syn_locs = [l for l in SYN["L"] if l in ctx.locs and SYN["L"][l][0] not in language_words]
    syn_verbs = [x for x in SYN["V"] if x in ctx.trans + ctx.intr and SYN["V"][x][0] not in language_words]
    rng = random.Random(7)

    def frame(force_syn: bool = False) -> Frame:
        def np(adj_p=0.35):
            pool = syn_nouns if (force_syn and rng.random() < 0.5) else ctx.nouns
            adj = (rng.choice(syn_adjs if force_syn and rng.random() < 0.5 else ctx.adjs)
                   if rng.random() < adj_p else None)
            return NP(rng.choice(pool), rng.random() < 0.4, adj)
        subj = rng.choice(list(PRONOUNS)) if rng.random() < 0.3 else np()
        vpool = [x for x in syn_verbs if x in ctx.trans] if (force_syn and rng.random() < 0.4) else ctx.trans
        if rng.random() < 0.7 and vpool:
            verb, obj = rng.choice(vpool), np()
        else:
            verb, obj = rng.choice(ctx.intr), None
        loc = rng.choice(syn_locs if force_syn else ctx.locs) if rng.random() < 0.3 else None
        return Frame(subj, verb, obj, loc, rng.choice(TENSES), rng.random() < 0.3)

    def ok(f: Frame) -> bool:
        if english(f, v) in ctx.train_en:
            return False
        if f.obj is not None and heldout_pair(ctx.seed, f.verb, f.obj.noun):
            return False
        if is_heldout_shape(f, v):
            return False
        return True

    variants = {"capitalised + full stop": [], "contractions (didn't, won't...)": [],
                "'a' instead of 'the'": [], "passive voice": [], "synonym (word not in the language)": [],
                "UPPER CASE": []}
    items = []
    tries = 0
    while any(len(x) < per_variant for x in variants.values()) and tries < 200000:
        tries += 1
        f = frame(force_syn=len(variants["synonym (word not in the language)"]) < per_variant and rng.random() < 0.5)
        if not ok(f):
            continue
        en = english(f, v)
        alts = {"capitalised + full stop": en[0].upper() + en[1:] + ".",
                "contractions (didn't, won't...)": contracted(en),
                "'a' instead of 'the'": (" ".join("a" if w == "the" else w for w in en.split())
                                         if " the " in f" {en} " and not any(isinstance(x, NP) and x.plural
                                                                             for x in (f.subj, f.obj)) else None),
                "passive voice": passive(f, v),
                "synonym (word not in the language)": (synonym_english(f, v, language_words) or (None,))[0],
                "UPPER CASE": en.upper()}
        for name, alt in alts.items():
            if alt and len(variants[name]) < per_variant:
                base_i = len(items)
                items.append(ctx.item(f, en, kind="original", variant=name))
                items.append(ctx.item(f, alt, kind="variant", variant=name))
                variants[name].append(base_i)
                break
    print(f"{len(items)} items")
    res = run_models(items)
    labels = list(res)
    lines = ["# Test 4: rephrased English, same correct answer", "",
             f"{per_variant} new sentences per row (only training words, no held-out combinations, none in training).",
             "Each sentence is scored twice: as normally written, and rephrased. The correct translation is identical.",
             "Synonyms are only used when the synonym is NOT itself a word of the language.", "",
             "| rephrasing | example | " + " | ".join(f"{l} (normal -> rephrased)" for l in labels) + " |",
             "|---|---|" + "---|" * len(labels)]
    for name, idx in variants.items():
        ex = example(items, [i + 1 for i in idx])
        cells = []
        for l in labels:
            a = exact_rate(res[l], idx)
            b = exact_rate(res[l], [i + 1 for i in idx])
            cells.append(f"{a:.1%} -> {b:.1%}")
        lines.append(f"| {name} | {ex} | " + " | ".join(cells) + " |")
    write("test4_rephrased", lines)


def is_heldout_shape(f: Frame, vocab: Vocab) -> bool:
    from conlang.dataset import is_long
    _, number, _ = f.subject_png(vocab)
    return is_long(f) or (f.tense == "past" and f.neg and number == "pl")


# --- 5. longer / deeper sentences ---------------------------------------------------------

LEVELS = [("0. plain: subject verb object", 0, 0, False, False),
          ("1. + location", 0, 0, True, False),
          ("2. + 1 adjective on the subject", 1, 0, True, False),
          ("3. adjective on both nouns + location (= held-out 'long')", 1, 1, True, False),
          ("4. 2 adjectives on the subject, 1 on the object", 2, 1, True, False),
          ("5. 2 adjectives on each noun", 2, 2, True, False),
          ("6. 3 adjectives on each noun", 3, 3, True, False),
          ("7. level 3 + past + negated + plural subject (two held-out things at once)", 1, 1, True, True)]


def test_depth(per_level: int = 150) -> None:
    ctx = Ctx()
    v = ctx.vocab
    rng = random.Random(11)
    items, level_idx = [], {}
    for name, n_sa, n_oa, loc, morph in LEVELS:
        level_idx[name] = []
        tries = 0
        while len(level_idx[name]) < per_level and tries < 100000:
            tries += 1
            sa, oa = rng.sample(ctx.adjs, n_sa), rng.sample(ctx.adjs, n_oa)
            verb = rng.choice(ctx.trans)
            subj = NP(rng.choice(ctx.nouns), True if morph else rng.random() < 0.4)
            obj = NP(rng.choice(ctx.nouns), rng.random() < 0.4)
            if heldout_pair(ctx.seed, verb, obj.noun):
                continue
            tense, neg = ("past", True) if morph else (rng.choice(TENSES), rng.random() < 0.3)
            if not morph and tense == "past" and neg and subj.plural:
                continue
            f = Frame(subj, verb, obj, rng.choice(ctx.locs) if loc else None, tense, neg)
            # English: adjectives joined into the NP's adjective slot; target: one word per adjective
            en_f = Frame(NP(subj.noun, subj.plural, " ".join(sa) or None), verb,
                         NP(obj.noun, obj.plural, " ".join(oa) or None), f.loc, tense, neg)
            en = english(en_f, v)
            if en in ctx.train_en:
                continue
            real = ctx.lang.realize(f)
            words = list(real.words)
            adj_words = lambda adjs: [ctx.lang._word([ctx.lang.roots[f"A:{a}"]], [a], real) for a in adjs]
            n_loc = 1 if f.loc else 0
            obj_at = 1 + n_loc                                 # SOV, no articles: [subj] [loc] [obj] [verb]
            words = adj_words(sa) + words[:obj_at] + adj_words(oa) + words[obj_at:]
            items.append({"english": en, "target": render_sentence(words, ctx.script),
                          "gloss": " ".join(w.gloss for w in words), "level": name})
            level_idx[name].append(len(items) - 1)
    print(f"{len(items)} items")
    res = run_models(items)
    labels = list(res)
    lines = ["# Test 5: longer and deeper sentences", "",
             f"{per_level} new sentences per level. Training had at most one adjective per noun, and never an",
             "adjective on both nouns together with a location (that is the held-out 'long' structure).",
             "Assumed rule for 2+ adjectives (never seen): each adjective is its own word before the noun, in English order.", "",
             "| level | example | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for name, idx in level_idx.items():
        lines.append(f"| {name} | {example(items, idx)} | " +
                     " | ".join(f"{exact_rate(res[l], idx):.1%}" for l in labels) + " |")
    write("test5_depth", lines)


# --- 6. invented names, spelled by sound -----------------------------------------------------

def test_names(n: int = 300) -> None:
    ctx = Ctx()
    rng = random.Random(5)
    cons = ["k", "c", "t", "n", "p", "m", "y", "r", "l", "v"]
    vowels = ["a", "e", "i", "o", "u", "a", "i", "u", "aa", "ii", "uu"]
    codas = ["n", "m", "l", "r", ""]
    lexicon_latin = {Script("latin", PRESETS[ctx.cfg["preset"]].phonology).word(r) for r in ctx.lang.roots.values()}
    items, seen = [], set()
    while len(items) < n:
        syl = rng.choice([2, 2, 3])
        ph = []
        for i in range(syl):
            if not (i == 0 and rng.random() < 0.15):
                ph.append(rng.choice(cons))
            ph.append(rng.choice(vowels))
        if rng.random() < 0.5:
            c = rng.choice(codas)
            if c:
                ph.append(c)
        latin = "".join(ph)
        if latin in seen or latin in lexicon_latin:
            continue
        seen.add(latin)
        name = latin.capitalize()
        verb = rng.choice(ctx.intr)
        en = f"{name} {ctx.vocab.verb_info[verb][2]}"
        items.append({"english": en, "target": ctx.script.word(tuple(ph)), "latin": latin, "phonemes": ph})
    print(f"{len(items)} names")
    # the same names through the transliteration prompt option 2 was trained on (zero-shot for the others)
    translit = [{"prompt_text": f"Latin: {it['latin']}\nGlyphs:\n", "target": it["target"]} for it in items]
    both = run_models(items + translit)              # target = the name word only
    res = {l: v[:len(items)] for l, v in both.items()}
    res_t = {l: v[len(items):] for l, v in both.items()}
    labels = list(res)
    lines = ["# Test 6: invented names, spelled by sound", "",
             f"{n} invented names (never seen; not words of the language), e.g. '{items[0]['english']}'.",
             "The name should be written in glyphs by its sound (as the first word of the translation).",
             "Scored teacher-forced on the name's glyphs only: name fully right / share of glyphs right.", "",
             "| model | name fully right | glyphs right |", "|---|---|---|"]
    for l in labels:
        full = sum(set(s[:-1]) == {"1"} for s in res[l]) / len(items)
        glyph = sum(s[:-1].count("1") for s in res[l]) / sum(len(s) - 1 for s in res[l])
        lines.append(f"| {l} | {full:.1%} | {glyph:.1%} |")
    lines += ["", "## Same names through the transliteration prompt ('Latin: muularo / Glyphs:')", "",
              "Only option 2 was trained on this prompt (on training sentences); for the others it is unseen.", "",
              "| model | name fully right | glyphs right |", "|---|---|---|"]
    for l in labels:
        full = sum(set(s[:-1]) == {"1"} for s in res_t[l]) / len(items)
        glyph = sum(s[:-1].count("1") for s in res_t[l]) / sum(len(s) - 1 for s in res_t[l])
        lines.append(f"| {l} | {full:.1%} | {glyph:.1%} |")
    # by name length (glyphs)
    lines += ["", "## By name length (glyphs)", "", "| glyphs | names | " + " | ".join(labels) + " |",
              "|---|---|" + "---|" * len(labels)]
    lens = sorted({len(it["target"]) for it in items})
    for L in lens:
        idx = [i for i, it in enumerate(items) if len(it["target"]) == L]
        lines.append(f"| {L} | {len(idx)} | " + " | ".join(
            f"{sum(set(res[l][i][:-1]) == {'1'} for i in idx) / len(idx):.1%}" for l in labels) + " |")
    write("test6_names", lines)
    sample_names(items[:8])


def sample_names(items: list[dict]) -> None:
    """Free generation for a few names, shown as Latin-spelled output so it can be read."""
    from train.load import load_trained, translate
    lines = ["", "## Sample outputs (free generation, first word, glyphs read back as sounds)", "",
             "| English | correct | " + " | ".join(l for l, r in MODELS.items()
                                                  if (RUNS / r / "model" / "glyph_tokens.pt").exists()) + " |",
             "|---|---|" + "---|" * sum((RUNS / r / "model" / "glyph_tokens.pt").exists() for r in MODELS.values())]
    ctx_script = None
    outs = {}
    for label, run in MODELS.items():
        if not (RUNS / run / "model" / "glyph_tokens.pt").exists():
            continue
        model, tok, cfg = load_trained(RUNS / run / "model")
        if ctx_script is None:
            sc = Script(cfg["script"], PRESETS[cfg["preset"]].phonology)
            ctx_script = sc.latin_of_glyphs()
            cons_glyphs, signs, virama = set(sc.cons.values()), {g for g in sc.sign.values() if g}, sc.virama
        outs[label] = translate(model, tok, [it["english"] for it in items])
        del model
        torch.cuda.empty_cache()
    def read(s: str) -> str:
        """Glyphs -> sounds; a consonant with no vowel sign carries the inherent 'a', a virama removes it."""
        out = []
        for i, c in enumerate(s):
            snd = ctx_script.get(c, c)
            out.append(snd)
            nxt = s[i + 1] if i + 1 < len(s) else ""
            if c in cons_glyphs and (nxt == "" or (nxt not in signs and nxt != virama)):
                out.append("a")
        return "".join(out)
    for i, it in enumerate(items):
        lines.append(f"| {it['english']} | {it['latin']} | " + " | ".join(
            read(outs[l][i].split(" ")[0]) for l in outs) + " |")
    with open(OUT / "test6_names.md", "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    {"embed": test_embed, "freq": test_freq, "grid": test_grid, "paraphrase": test_paraphrase,
     "depth": test_depth, "names": test_names}[sys.argv[1]]()
