"""Load a model saved by `train.run --save-adapter DIR` and use it.

    from train.load import load_trained, translate, rebuild_data
    model, tok, cfg = load_trained("results/models/g1-hard-latininit-50k")
    translate(model, tok, ["the big dog saw the cat"])
    data = rebuild_data(cfg)            # the exact train/test sentences of that run

Command line:
    python -m train.load DIR "english sentence" ...      translate
    python -m train.load DIR --check                     re-score test_iid, compare with the run's result
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conlang.dataset import PRESETS, build  # noqa: E402
from train.run import POOL, add_glyph_tokens, encode, prompt, teacher_forced  # noqa: E402


def load_trained(path, device: str | None = None, base_model: str | None = None):
    """Returns (model, tokenizer, run_args). `base_model` overrides the model name/path
    stored in the run (e.g. a local copy of Qwen3-0.6B-Base)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(path)
    cfg = json.loads((path / "run_args.json").read_text(encoding="utf-8"))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(str(path))
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model or cfg["model"], dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base_model or cfg["model"], torch_dtype=dtype)
    glyphs = path / "glyph_tokens.pt"
    if glyphs.exists():
        g = torch.load(glyphs, map_location="cpu", weights_only=True)
        assert tok.convert_tokens_to_ids(g["chars"][0]) == g["start"], "tokenizer and glyph table disagree"
        delta = add_glyph_tokens(model, g["start"], len(g["chars"]), 0, "mean")
        with torch.no_grad():
            delta.copy_(g["delta"])
    model = PeftModel.from_pretrained(model, str(path))
    model.to(device).eval()
    model.config.use_cache = True
    return model, tok, cfg


@torch.no_grad()
def translate(model, tok, sentences: list[str], max_new_tokens: int = 128) -> list[str]:
    """Greedy translation, English -> the run's language, exactly as evaluated."""
    device = next(model.parameters()).device
    tok.padding_side = "left"
    enc = tok([prompt(s) for s in sentences], return_tensors="pt", padding=True,
              add_special_tokens=False).to(device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    tok.padding_side = "right"
    return [tok.decode(r, skip_special_tokens=True).split("\n")[0].strip()
            for r in out[:, enc["input_ids"].shape[1]:]]


def rebuild_data(cfg: dict) -> dict:
    """The run's dataset (train pool + the four test splits), rebuilt deterministically."""
    return build(PRESETS[cfg["preset"]], cfg["script"], n_train=POOL, n_test=int(cfg["n_test"]),
                 seed=int(cfg["seed"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("sentences", nargs="*")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--base-model", default=None)
    a = ap.parse_intermixed_args()
    model, tok, cfg = load_trained(a.dir, base_model=a.base_model)
    if a.sentences:
        for s, t in zip(a.sentences, translate(model, tok, a.sentences)):
            print(f"{s}\n  -> {t}")
    if a.check:
        items = rebuild_data(cfg)["splits"]["test_iid"]
        device = next(model.parameters()).device
        amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if device.type == "cuda" \
            else (lambda: torch.autocast("cpu", enabled=False))
        tf = teacher_forced(model, tok, [encode(tok, it, cfg["target_tok"]) for it in items], 16, device, amp)
        print(f"test_iid now: exact match {tf['exact_match']:.3f}, token accuracy {tf['token_accuracy']:.4f}")


if __name__ == "__main__":
    main()
