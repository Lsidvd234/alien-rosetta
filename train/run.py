"""One training run: LoRA-fine-tune a base model on one language x script x N, then evaluate.

Evaluation on the four fixed test splits (500 each by default):
- teacher-forced: per-token accuracy p, and "every token right" (identical to greedy
  exact match, since greedy decoding reproduces the answer iff each argmax is right);
  compared with the p^L prediction (is the drop just "more tokens to get right"?);
  word error rate grouped by how many tokens the word was split into.
- free generation on a subset: exact match, chrF, word accuracy, sample outputs.

Usage (also what the Kaggle job calls):
    python -m train.run --model Qwen/Qwen3-0.6B-Base --preset english_like --script latin \
        --n-train 20000 --out results/run.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conlang.dataset import PRESETS, TEST_SPLITS, build  # noqa: E402
from conlang.evaluate import chrf, exact_match, word_accuracy  # noqa: E402
from conlang.script import Script  # noqa: E402

POOL = 50000          # training pool is always built at this size, so first-N subsets are nested
WORD_BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 10), (11, 10**6)]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prompt(english: str) -> str:
    return f"English: {english}\nTarget:\n"


def pick_precision(req: str) -> str:
    if req != "auto":
        return req
    if not torch.cuda.is_available():
        return "fp32"
    major, _ = torch.cuda.get_device_capability(0)
    return "bf16" if major >= 8 else "fp16"     # T4 / P100 have no native bf16


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# --- data -------------------------------------------------------------------------

def encode(tok, it: dict, mode: str = "bpe") -> dict:
    """mode="bpe": normal tokenization of the target. mode="char": every character of the
    target is tokenized on its own (one or more tokens per character, never merged across
    characters), so the text is identical but the split is as fine as possible."""
    ptxt = it["prompt_text"] if "prompt_text" in it else prompt(it["english"])   # "" = plain text (LM)
    p = tok(ptxt, add_special_tokens=False)["input_ids"] if ptxt else []
    target = it["target"]
    if mode == "char":
        ids, offsets = [], []
        for i, ch in enumerate(target):
            piece = tok(ch, add_special_tokens=False)["input_ids"]
            ids += piece
            offsets += [(i, i + 1)] * len(piece)
    else:
        enc = tok(target, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
    t = ids + [tok.eos_token_id]
    return {"ids": p + t, "labels": [-100] * len(p) + t, "n_prompt": len(p),
            "offsets": offsets, "target": target}


def unknown_chars(tok, data: dict) -> list[str]:
    """Characters in any target that the tokenizer splits into several pieces (byte fallback),
    sorted, so a run and its analysis add them in the same order."""
    chars = {ch for split in data["splits"].values() for it in split for ch in it["target"]}
    return sorted(c for c in chars if len(tok(c, add_special_tokens=False)["input_ids"]) > 1)


def donor_tokens(tok, n_old: int, k: int, seed: int) -> list[int]:
    """k distinct ordinary tokens to copy embeddings from: plain ASCII word pieces only (no
    Tamil or other non-Latin text, no special tokens), chosen at random with a fixed seed."""
    rng = random.Random(seed)
    out: list[int] = []
    while len(out) < k:
        i = rng.randrange(1000, n_old - 1000)
        s = tok.decode([i]).strip()
        if len(s) >= 3 and s.isascii() and s.isalpha() and i not in out:
            out.append(i)
    return out


def shuffled_sounds(sounds: list[str], seed: int) -> list[str]:
    """Reassign the sounds among the glyphs so that no glyph keeps its own sound (glyphs with
    no sound, i.e. the virama, stay empty). Same distinct-start structure as the real mapping,
    but the sound information is wrong."""
    rng = random.Random(seed)
    idx = [i for i, s in enumerate(sounds) if s]
    for _ in range(10000):
        perm = idx[:]
        rng.shuffle(perm)
        if all(sounds[i] != sounds[j] for i, j in zip(idx, perm)):
            out = list(sounds)
            for i, j in zip(idx, perm):
                out[i] = sounds[j]
            return out
    raise RuntimeError("no derangement found")


class AddedEmbed(torch.nn.Module):
    """Input embedding where ids [start, start+k) come from a trainable fp32 table."""

    def __init__(self, base, start: int, delta: torch.nn.Parameter):
        super().__init__()
        self.base, self.start, self.delta = base, start, delta

    def forward(self, ids):
        out = self.base(ids)
        m = (ids >= self.start) & (ids < self.start + self.delta.shape[0])
        if m.any():
            out = out.clone()
            out[m] = self.delta[ids[m] - self.start].to(out.dtype)
        return out


class AddedHead(torch.nn.Module):
    """Output layer tied to the same table (Qwen3-0.6B ties input and output embeddings)."""

    def __init__(self, base, start: int, delta: torch.nn.Parameter):
        super().__init__()
        self.base, self.start, self.delta = base, start, delta

    def forward(self, h):
        out = self.base(h)
        new = h @ self.delta.t().to(h.dtype)
        return torch.cat([out[..., :self.start], new.to(out.dtype),
                          out[..., self.start + self.delta.shape[0]:]], dim=-1)


def add_glyph_tokens(model, n_old: int, k: int, seed: int, init: str = "mean",
                     donor_ids: list[int] | None = None) -> torch.nn.Parameter:
    """Give k new tokens (ids n_old..n_old+k-1, inside the model's spare embedding rows) a
    trainable embedding. init="mean": mean embedding + small noise (all start near-identical).
    init="copy": each starts as a copy of a different existing token (`donor_ids`), so they
    are distinct from the start."""
    emb = model.get_input_embeddings()
    assert n_old + k <= emb.weight.shape[0], "not enough spare embedding rows"
    w = emb.weight[:n_old].float()
    if init == "copy":
        delta = torch.nn.Parameter(w[torch.tensor(donor_ids, device=w.device)].clone())
    elif init in ("latin", "latin_shuffled"):         # donor_ids: one list of token ids per new token
        g = torch.Generator(device="cpu").manual_seed(seed)
        rows = []
        for ids in donor_ids:
            if ids:
                rows.append(w[torch.tensor(ids, device=w.device)].mean(0))
            else:                                     # no sound (virama): neutral start
                rows.append(w.mean(0) + torch.randn(w.shape[1], generator=g).to(w.device) * w.std(0) * 0.1)
        delta = torch.nn.Parameter(torch.stack(rows))
    else:
        g = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(k, w.shape[1], generator=g).to(w.device) * w.std(0) * 0.1
        delta = torch.nn.Parameter(w.mean(0) + noise)
    model.model.embed_tokens = AddedEmbed(emb, n_old, delta)
    model.lm_head = AddedHead(model.lm_head, n_old, delta)
    return delta


def save_trained(model, tok, delta, n_old: int, added: list[str], args: dict, path: Path,
                 delta_init=None, sounds: list[str] | None = None) -> None:
    """Everything needed to rebuild the trained model (see train/load.py): LoRA weights,
    the tokenizer with its added glyph tokens, the glyph embedding table (it lives outside
    LoRA, so save_pretrained does not include it) plus its starting values and each glyph's
    sound, and the run arguments."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tok.save_pretrained(str(path))
    if delta is not None:
        torch.save({"start": n_old, "chars": added, "sounds": sounds,
                    "delta": delta.detach().float().cpu(), "delta_init": delta_init},
                   path / "glyph_tokens.pt")
    (path / "run_args.json").write_text(json.dumps(args, indent=1), encoding="utf-8")


def batches_by_length(examples: list[dict], bsz: int, rng: random.Random) -> list[list[dict]]:
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    out = []
    chunk = bsz * 50
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s:s + chunk], key=lambda i: len(examples[i]["ids"]))
        out += [[examples[i] for i in part[j:j + bsz]] for j in range(0, len(part), bsz)]
    rng.shuffle(out)
    return out


def collate(batch: list[dict], pad_id: int, device) -> dict:
    T = max(len(e["ids"]) for e in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), T), -100, dtype=torch.long)
    att = torch.zeros((len(batch), T), dtype=torch.long)
    for i, e in enumerate(batch):
        n = len(e["ids"])
        ids[i, :n] = torch.tensor(e["ids"])
        lab[i, :n] = torch.tensor(e["labels"])
        att[i, :n] = 1
    return {"input_ids": ids.to(device), "labels": lab.to(device), "attention_mask": att.to(device)}


# --- evaluation ------------------------------------------------------------------------

def word_index(target: str, start: int) -> int:
    if start < len(target) and target[start] == " ":
        start += 1
    return target[:start].count(" ")


@torch.no_grad()
def teacher_forced(model, tok, examples: list[dict], bsz: int, device, amp) -> dict:
    model.eval()
    rows = []
    for s in range(0, len(examples), bsz):
        batch = examples[s:s + bsz]
        b = collate(batch, tok.pad_token_id, device)
        with amp():
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        pred = logits[:, :-1].argmax(-1)
        gold = b["input_ids"][:, 1:]
        for i, e in enumerate(batch):
            lo, hi = e["n_prompt"] - 1, len(e["ids"]) - 1          # positions predicting target tokens
            ok = (pred[i, lo:hi] == gold[i, lo:hi]).tolist()
            rows.append({"ok": ok, "offsets": e["offsets"], "target": e["target"]})
        del logits
    n_tok = sum(len(r["ok"]) for r in rows)
    p = sum(sum(r["ok"]) for r in rows) / n_tok
    exact = sum(all(r["ok"]) for r in rows) / len(rows)
    pred_pl = sum(p ** len(r["ok"]) for r in rows) / len(rows)

    # word errors by number of tokens in the word (EOS excluded)
    buckets = {f"{a}-{b}" if a != b else str(a): [0, 0] for a, b in WORD_BUCKETS}
    for r in rows:
        words: dict[int, list[bool]] = {}
        for (st, _), good in zip(r["offsets"], r["ok"][:-1]):
            words.setdefault(word_index(r["target"], st), []).append(bool(good))
        for oks in words.values():
            k = len(oks)
            for a, b in WORD_BUCKETS:
                if a <= k <= b:
                    key = f"{a}-{b}" if a != b else str(a)
                    buckets[key][0] += 1
                    buckets[key][1] += (not all(oks))
                    break
    word_err = {k: {"words": n, "error_rate": (e / n if n else None)} for k, (n, e) in buckets.items()}
    return {"n": len(rows), "token_accuracy": p, "exact_match": exact,
            "predicted_exact_from_p^L": pred_pl, "mean_target_tokens": n_tok / len(rows),
            "word_error_by_tokens_in_word": word_err,
            # per-item token correctness (1/0 per target token incl. EOS) for offline analysis
            "per_item_ok": ["".join("1" if x else "0" for x in r["ok"]) for r in rows]}


@torch.no_grad()
def generate(model, tok, items: list[dict], bsz: int, device, amp) -> dict:
    model.eval()
    tok.padding_side = "left"
    preds = []
    for s in range(0, len(items), bsz):
        batch = items[s:s + bsz]
        enc = tok([prompt(it["english"]) for it in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        max_new = max(len(tok(it["target"], add_special_tokens=False)["input_ids"]) for it in batch) + 16
        with amp():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row in out[:, enc["input_ids"].shape[1]:]:
            preds.append(tok.decode(row, skip_special_tokens=True).split("\n")[0].strip())
    tok.padding_side = "right"
    refs = [it["target"] for it in items]
    return {"n": len(items), "exact_match": exact_match(preds, refs), "chrf": chrf(preds, refs),
            "word_accuracy": word_accuracy(preds, refs),
            "samples": [{"english": it["english"], "gold": it["target"], "pred": p}
                        for it, p in list(zip(items, preds))[:8]]}


@torch.no_grad()
def gen_eval(model, tok, items: list[dict], bsz: int, device, amp, cot: bool) -> dict:
    """Free generation; only the glyph part is compared (for CoT: the text after '=>')."""
    model.eval()
    tok.padding_side = "left"
    preds = []
    for s in range(0, len(items), bsz):
        batch = items[s:s + bsz]
        enc = tok([prompt(it["english"]) for it in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        longest = max(len(tok(it["target"], add_special_tokens=False)["input_ids"]) for it in batch)
        max_new = (2 * longest + 64) if cot else (longest + 16)
        with amp():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row in out[:, enc["input_ids"].shape[1]:]:
            preds.append(tok.decode(row, skip_special_tokens=True).split("\n")[0].strip())
    tok.padding_side = "right"
    glyph = [p.split("=>", 1)[1].strip() if (cot and "=>" in p) else p for p in preds]
    res = {"n": len(items), "exact": sum(g == it["target"] for g, it in zip(glyph, items)) / len(items),
           "samples": [{"english": it["english"], "gold": it["target"], "pred": p} for it, p in list(zip(items, preds))[:5]],
           "per_item_ok": "".join("1" if g == it["target"] else "0" for g, it in zip(glyph, items))}
    if cot:
        ana = [p.split("=>", 1)[0].strip() if "=>" in p else "" for p in preds]
        res["analysis_exact"] = sum(x == it.get("analysis") for x, it in zip(ana, items)) / len(items)
        res["per_item_analysis_ok"] = "".join("1" if x == it.get("analysis") else "0" for x, it in zip(ana, items))
    return res


# --- main --------------------------------------------------------------------------------

def focal_loss(model, b: dict, gamma: float) -> torch.Tensor:
    """Mean over answer tokens of (1-p)^gamma * cross-entropy (Lin et al. 2017), next-token shifted like HF."""
    labels = b["labels"]
    logits = model(**{k: v for k, v in b.items() if k != "labels"}).logits[:, :-1].float()
    tgt = labels[:, 1:]
    mask = tgt != -100
    logp = torch.log_softmax(logits[mask], dim=-1).gather(-1, tgt[mask].unsqueeze(-1)).squeeze(-1)
    return (-(1 - logp.exp()).pow(gamma) * logp).mean()


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--preset", choices=list(PRESETS), default="english_like")
    ap.add_argument("--script", default="latin")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--n-gen", type=int, default=100, help="free-generation items per split")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epochs (smoke tests use a small number)")
    ap.add_argument("--max-minutes", type=float, default=540, help="stop training after this, then evaluate")
    ap.add_argument("--target-tok", choices=["bpe", "char"], default="bpe",
                    help="char = one character per token for the target (cause test M2)")
    ap.add_argument("--add-tokens", action="store_true",
                    help="each character the tokenizer splits into bytes becomes one new trainable token (M3)")
    ap.add_argument("--new-tok-init", choices=["mean", "copy", "latin", "latin_shuffled"], default="mean",
                    help="with --add-tokens: start new tokens at the mean embedding, as a copy of a "
                         "different random ASCII word-piece token, or (latin) as the embedding of the "
                         "Latin letters for the sound the glyph writes")
    ap.add_argument("--aux-translit", action="store_true",
                    help="also train on transliteration (glyphs <-> Latin spelling) of every training "
                         "sentence, alternating direction, mixed with the translation examples")
    ap.add_argument("--mono-n", type=int, default=0,
                    help="first train on this many EXTRA target-language-only sentences (plain next-token "
                         "prediction, no English; same held-out rules as training), then translation")
    ap.add_argument("--mono-epochs", type=float, default=2.0)
    ap.add_argument("--aug", action="store_true",
                    help="R1 retrain data (train/augment.py): varied English, new structures, refusals, replay")
    ap.add_argument("--aug-rate", type=float, default=0.5, help="share of translation pairs with rewritten English")
    ap.add_argument("--aug-struct", type=int, default=5000, help="extra examples with new structures")
    ap.add_argument("--aug-refuse", type=int, default=1500, help="non-English inputs -> refusal output")
    ap.add_argument("--aug-replay", type=int, default=5000, help="plain-English passages (language modelling)")
    ap.add_argument("--aug-eval-per", type=int, default=100, help="items per augmented evaluation category")
    ap.add_argument("--save-every-epoch", action="store_true", help="with --save-adapter: also save after each epoch")
    ap.add_argument("--rep", action="store_true",
                    help="E1: representative English (train/represent.py) instead of the template data")
    ap.add_argument("--cot", action="store_true", help="with --rep: write an analysis of the English, then the glyphs")
    ap.add_argument("--rep-n", type=int, default=30000, help="with --rep: translation examples")
    ap.add_argument("--rep-refuse", type=int, default=1500)
    ap.add_argument("--rep-replay", type=int, default=4000)
    ap.add_argument("--rep-eval-per", type=int, default=100)
    ap.add_argument("--rep-strict", action="store_true",
                    help="with --rep: held-out voice/order groups keep only items that really contain the variant")
    ap.add_argument("--rep-v2", action="store_true",
                    help="with --rep: the v2 language (conlang/v2.py) and its generator (train/represent_v2.py)")
    ap.add_argument("--rep-held-frac", type=float, default=0.2,
                    help="with --rep-v2: share of English variants held out from training (0 = none, final model)")
    ap.add_argument("--rep-held-only", default=None,
                    help="with --rep-v2: hold out only this variant, e.g. voice=passive (replaces --rep-held-frac)")
    ap.add_argument("--rep-held-putback", type=int, default=0,
                    help="with --rep-held-only: put back this many training examples of it (structure dose)")
    ap.add_argument("--rep-min-count", type=int, default=0,
                    help="with --rep-v2: data fix, every word at least this many times (same number of examples)")
    ap.add_argument("--rep-paraphrases", type=Path, default=None,
                    help="with --rep-v2: {english: paraphrase} file (scripts/build_paraphrases.py)")
    ap.add_argument("--rep-real", type=Path, default=None,
                    help="with --rep-v2: real-English examples (english/analysis/target) replacing generated ones")
    ap.add_argument("--rep-loan-rate", type=float, default=0.0,
                    help="with --rep-v2: share of noun phrases that use a borrowed English noun (LOAN.pizza)")
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce",
                    help="focal: each answer token's loss times (1-p)^gamma, so tokens the model already gets right "
                         "count less and rare/hard words count more (charter claim 6)")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--max-batch-tokens", type=int, default=0,
                    help="split a micro-batch whose padded size (examples x longest) exceeds this; 0 = never")
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    ap.add_argument("--grad-ckpt", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-seed", type=int, default=None,
                    help="seed for LoRA init + batch order only (default = --seed). A true replicate: same "
                         "--seed (same language and data), different --train-seed")
    ap.add_argument("--out", type=Path, default=Path("results/run.json"))
    ap.add_argument("--save-adapter", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.train_seed is None:
        a.train_seed = a.seed

    from peft import LoraConfig, get_peft_model
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prec = pick_precision(a.precision)
    amp = (lambda: torch.autocast("cuda", dtype=DTYPES[prec])) if (device == "cuda" and prec != "fp32") \
        else (lambda: torch.autocast("cpu", enabled=False))
    result: dict = {"args": {k: str(v) for k, v in vars(a).items()}, "precision": prec,
                    "env": {"python": platform.python_version(), "torch": torch.__version__,
                            "transformers": transformers.__version__,
                            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}}
    log(f"device={result['env']['gpu']} precision={prec}")

    # data
    t0 = time.time()
    n_beyond = max(0, a.n_train - POOL)
    data = build(PRESETS[a.preset], a.script, n_train=POOL, n_test=a.n_test, seed=a.seed,
                 n_extra=n_beyond + a.mono_n)   # beyond 50k: same tests, pool extended
    train_items = data["splits"]["train"][:a.n_train]
    # plain-text sentences: extra ones after the translation pool, never paired with English
    mono_items = [{"prompt_text": "", "target": it["target"]}
                  for it in data["splits"]["train"][POOL + n_beyond:]]
    aux_items: list[dict] = []
    if a.aux_translit:
        lat = build(PRESETS[a.preset], "latin", n_train=POOL, n_test=a.n_test, seed=a.seed, n_extra=n_beyond)
        for i, (g, l) in enumerate(zip(train_items, lat["splits"]["train"])):
            assert g["english"] == l["english"], "glyph and Latin builds out of step"
            if i % 2 == 0:
                aux_items.append({"prompt_text": f"Glyphs: {g['target']}\nLatin:\n", "target": l["target"]})
            else:
                aux_items.append({"prompt_text": f"Latin: {l['target']}\nGlyphs:\n", "target": g["target"]})
    aug_info = None
    if a.rep:
        if a.rep_v2:
            from train.represent_v2 import build_training as rep2_training
            train_items, rep_info = rep2_training(a.preset, a.script, a.seed, a.rep_n, a.rep_refuse, a.rep_replay,
                                                  a.cot, a.rep_held_frac, a.rep_min_count,
                                                  str(a.rep_paraphrases) if a.rep_paraphrases else None,
                                                  a.rep_loan_rate, str(a.rep_real) if a.rep_real else None,
                                                  a.rep_held_only, a.rep_held_putback)
        else:
            from train.represent import build_training as rep_training
            train_items, rep_info = rep_training(a.preset, a.script, a.seed, a.rep_n, a.rep_refuse, a.rep_replay,
                                                 a.cot)
        aug_info = {"counts": {"examples": rep_info["n"]}, "train_names": []}
        result["rep"] = rep_info
        log(f"representative data: {rep_info['n']} examples; held out: {rep_info['held_out']}")
    elif a.aug:
        from train.augment import build_training
        train_items, aug_info = build_training(train_items, a.preset, a.script, a.seed, a.aug_rate,
                                               a.aug_struct, a.aug_refuse, a.aug_replay)
        log(f"augmented training data: {aug_info['counts']}")
    result["data"] = {"build_seconds": time.time() - t0, "n_train": len(train_items),
                      "test_sizes": {k: len(data["splits"][k]) for k in TEST_SPLITS},
                      "vocab_size": data["meta"]["vocab_size"]}
    log(f"data built: train={len(train_items)} tests={result['data']['test_sizes']}")

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    n_old = len(tok)
    added = unknown_chars(tok, data) if a.add_tokens else []
    if a.add_tokens and (a.aug or a.rep):                          # names may use glyph characters no test sentence has
        extra = sorted({c for it in train_items if it.get("aug") != "replay" for c in it["target"]
                        if c not in added and len(tok(c, add_special_tokens=False)["input_ids"]) > 1})
        added = sorted(set(added) | set(extra))
    if aug_info:
        result["data"]["aug"] = aug_info["counts"]
    if added:
        tok.add_tokens(added)
        result["data"]["added_tokens"] = len(added)
        log(f"added {len(added)} single-character tokens")
    train_ex = [encode(tok, it, a.target_tok) for it in train_items]
    result["data"]["mean_target_tokens"] = sum(len(e["ids"]) - e["n_prompt"] for e in train_ex) / len(train_ex)
    result["data"]["mean_example_tokens"] = sum(len(e["ids"]) for e in train_ex) / len(train_ex)
    aux_ex = [encode(tok, it, a.target_tok) for it in aux_items]
    mono_ex = [encode(tok, it, a.target_tok) for it in mono_items]
    result["data"].update({"aux_translit_examples": len(aux_ex), "mono_examples": len(mono_ex)})
    if aux_ex or mono_ex:
        log(f"extra examples: transliteration={len(aux_ex)} plain-text={len(mono_ex)}")

    # model
    load_dtype = DTYPES[prec] if device == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=load_dtype)
    except TypeError:  # older transformers name the argument torch_dtype
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=load_dtype)
    model.to(device)
    model.config.use_cache = False
    n_params = sum(p.numel() for p in model.parameters())
    use_ckpt = a.grad_ckpt == "on" or (a.grad_ckpt == "auto" and n_params > 1e9)
    if use_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    donors = None
    if added and a.new_tok_init == "copy":
        donors = donor_tokens(tok, n_old, len(added), a.seed)
        result["data"]["donor_tokens"] = [tok.decode([d]) for d in donors]
    elif added and a.new_tok_init in ("latin", "latin_shuffled"):
        sound = Script(a.script, PRESETS[a.preset].phonology).latin_of_glyphs()
        start = [sound.get(ch, "") for ch in added]
        if a.new_tok_init == "latin_shuffled":       # control: every glyph starts at a WRONG sound
            start = shuffled_sounds(start, a.seed)
        donors = [tok(s, add_special_tokens=False)["input_ids"] if s else [] for s in start]
        result["data"]["donor_tokens"] = start
    delta = add_glyph_tokens(model, n_old, len(added), a.seed, a.new_tok_init, donors) if added else None
    delta_init = delta.detach().float().cpu().clone() if delta is not None else None
    if a.train_seed != a.seed:                       # LoRA init from the training seed (language/data keep --seed);
        torch.manual_seed(a.train_seed)              # without --train-seed, runs stay bit-identical to before
    model = get_peft_model(model, lcfg)
    if delta is not None:
        delta.requires_grad_(True)                    # get_peft_model froze it
    for p in model.parameters():                      # adapters in fp32 for stable fp16 training
        if p.requires_grad:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result["model"] = {"name": a.model, "params": n_params, "lora_trainable": trainable,
                       "grad_checkpointing": use_ckpt}
    log(f"model {a.model}: {n_params/1e6:.0f}M params, LoRA {trainable/1e6:.1f}M trainable, ckpt={use_ckpt}")

    def sounds_list() -> list[str]:
        sounds = Script(a.script, PRESETS[a.preset].phonology).latin_of_glyphs() \
            if a.script.startswith("abugida") else {}
        return [sounds.get(ch, "?") for ch in added]

    # training: optional plain-text phase, then translation (+ transliteration if asked)
    rng = random.Random(a.train_seed)                # batch order: training seed
    scaler = torch.amp.GradScaler("cuda", enabled=(prec == "fp16"))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_all = time.time()

    def run_phase(examples: list[dict], epochs: float, tag: str) -> dict:
        steps_per_epoch = math.ceil(len(examples) / (a.batch * a.grad_accum))
        total = a.max_steps or math.ceil(steps_per_epoch * epochs)
        warm = max(1, int(0.03 * total))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
        model.train()
        step, micro, seen_tok, nan_events, t_train = 0, 0, 0, 0, time.time()
        loss_hist: list[float] = []
        log(f"phase {tag}: {len(examples)} examples, {total} steps")
        stop = False
        while not stop:
            for batch in batches_by_length(examples, a.batch, rng):
                # a batch longer than the token budget is processed in pieces (gradients add up), so a few long
                # examples cannot run the GPU out of memory; batches within the budget are unchanged
                T = max(len(e["ids"]) for e in batch)
                k = max(1, math.ceil(len(batch) * T / a.max_batch_tokens)) if a.max_batch_tokens else 1
                pieces = [batch[i::k] for i in range(k)] if k > 1 else [batch]
                n_lab = sum(sum(x != -100 for x in e["labels"]) for e in batch)
                total_loss, bad = 0.0, False
                for piece in pieces:
                    b = collate(piece, tok.pad_token_id, device)
                    share = int((b["labels"] != -100).sum()) / max(1, n_lab)   # token-weighted, as one batch
                    with amp():
                        if a.loss == "focal":
                            loss = focal_loss(model, b, a.focal_gamma) * share / a.grad_accum
                        else:
                            loss = model(**b).loss * share / a.grad_accum
                    if not torch.isfinite(loss):
                        bad = True
                        break
                    scaler.scale(loss).backward()
                    total_loss += loss.item()
                    seen_tok += int((b["labels"] != -100).sum())
                if bad:
                    nan_events += 1
                    if step < 20 and nan_events >= 5:
                        raise RuntimeError(f"non-finite loss early in training (precision={prec}); "
                                           "try --precision fp32 or a smaller learning rate")
                    opt.zero_grad(set_to_none=True)
                    continue
                micro += 1
                loss_hist.append(total_loss * a.grad_accum)
                if micro % a.grad_accum:
                    continue
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                step += 1
                if a.save_every_epoch and a.save_adapter and tag == "translation" and step % steps_per_epoch == 0 \
                        and step < total:
                    ep = step // steps_per_epoch
                    save_trained(model, tok, delta, n_old, added, result["args"],
                                 Path(f"{a.save_adapter}_epoch{ep}"), delta_init=delta_init, sounds=sounds_list())
                    log(f"saved epoch-{ep} checkpoint")
                    model.train()
                elapsed = time.time() - t_train
                if step % 25 == 0 or step == 1:
                    mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
                    recent = loss_hist[-25 * a.grad_accum:]
                    log(f"[{tag}] step {step}/{total} loss={sum(recent)/len(recent):.4f} "
                        f"lr={sched.get_last_lr()[0]:.2e} target_tok/s={seen_tok/elapsed:.0f} peak_mem={mem:.2f}GB")
                if step >= total or time.time() - t_all > a.max_minutes * 60:
                    stop = True
                    break
        secs = time.time() - t_train
        log(f"phase {tag}: {step} steps in {secs/60:.1f} min")
        return {"phase": tag, "examples": len(examples), "steps": step, "planned_steps": total,
                "seconds": secs, "target_tokens_per_second": seen_tok / secs,
                "final_loss": sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:])),
                "nan_events": nan_events, "stopped_by_time": step < total}

    phases = []
    if mono_ex:
        phases.append(run_phase(mono_ex, a.mono_epochs, "plain-text"))
    main_phase = run_phase(train_ex + aux_ex, a.epochs, "translation")
    phases.append(main_phase)
    train_s = time.time() - t_all
    result["training"] = {**main_phase, "seconds": train_s, "phases": phases,
                          "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0}
    log(f"trained in {train_s/60:.1f} min total")
    if a.save_adapter:
        sounds = Script(a.script, PRESETS[a.preset].phonology).latin_of_glyphs() \
            if a.script.startswith("abugida") else {}
        save_trained(model, tok, delta, n_old, added, result["args"], a.save_adapter,
                     delta_init=delta_init, sounds=sounds_list())
        log(f"saved model to {a.save_adapter}")

    # evaluation
    model.config.use_cache = True
    result["eval"] = {}
    t_eval = time.time()
    for split in ([] if (a.cot or a.rep_v2) else TEST_SPLITS):   # CoT / v2: scored by generation below
        items = data["splits"][split]
        tf = teacher_forced(model, tok, [encode(tok, it, a.target_tok) for it in items], a.eval_batch, device, amp)
        gen = generate(model, tok, items[:a.n_gen], a.eval_batch, device, amp)
        tf["per_item_ids"] = [it["id"] for it in items]
        result["eval"][split] = {"teacher_forced": tf, "generation": gen}
        log(f"{split}: token_acc={tf['token_accuracy']:.4f} exact={tf['exact_match']:.3f} "
            f"(p^L predicts {tf['predicted_exact_from_p^L']:.3f}) gen_exact={gen['exact_match']:.3f} chrF={gen['chrf']:.1f}")
    if a.rep and a.rep_v2:
        from train.represent_v2 import eval_sets as rep2_sets
        train_en = {it["english"] for it in train_items if "english" in it}
        result["eval_gen"] = {}
        for name, items in rep2_sets(a.preset, a.script, a.seed, train_en, a.rep_eval_per).items():
            if items:
                result["eval_gen"][name] = gen_eval(model, tok, items, a.eval_batch, device, amp, a.cot)
                g = result["eval_gen"][name]
                log(f"gen eval {name}: n={g['n']} glyph_exact={g['exact']:.3f}"
                    + (f" analysis_exact={g['analysis_exact']:.3f}" if a.cot else ""))
    elif a.rep:
        from train.augment import eval_sets as r1_sets
        from train.augment import frame_of
        from train.represent import Clause, Rep, eval_sets as rep_sets
        rp = Rep(a.preset, a.script, a.seed)
        train_en = {it["english"] for it in train_items if "english" in it}
        groups = {f"original {k}": [{**it, "analysis": rp.analysis(Clause(frame_of(it["frame"])))}
                                    for it in data["splits"][k]] for k in TEST_SPLITS}
        groups.update(rep_sets(a.preset, a.script, a.seed, train_en, a.rep_eval_per, strict=a.rep_strict))
        groups.update({f"R1 {k}": v[:50] for k, v in r1_sets(data["splits"]["test_iid"], a.preset, a.script, a.seed,
                                                               train_en, set(), 50).items()})
        result["eval_gen"] = {}
        for name, items in groups.items():
            if items:
                result["eval_gen"][name] = gen_eval(model, tok, items, a.eval_batch, device, amp, a.cot)
                g = result["eval_gen"][name]
                log(f"gen eval {name}: n={g['n']} glyph_exact={g['exact']:.3f}"
                    + (f" analysis_exact={g['analysis_exact']:.3f}" if a.cot else ""))
    if a.aug:
        from train.augment import eval_sets
        sets = eval_sets(data["splits"]["test_iid"], a.preset, a.script, a.seed,
                         {it["english"] for it in train_items if "english" in it},
                         set(aug_info["train_names"]), a.aug_eval_per)
        result["eval_aug"] = {}
        for name, items in sets.items():
            if not items:
                continue
            tf = teacher_forced(model, tok, [encode(tok, it, a.target_tok) for it in items], a.eval_batch, device, amp)
            result["eval_aug"][name] = {"n": tf["n"], "exact_match": tf["exact_match"],
                                        "token_accuracy": tf["token_accuracy"],
                                        "examples": [it["english"] for it in items[:3]]}
            log(f"aug eval {name}: n={tf['n']} exact={tf['exact_match']:.3f}")
    result["eval_seconds"] = time.time() - t_eval

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {a.out}")
    return result


if __name__ == "__main__":
    main()
