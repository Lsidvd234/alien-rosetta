"""Learning-curve table for the 2x2 (language x script) across training sizes.

Script-fair metrics only: content-word accuracy (every token of a content word right,
teacher-forced = what greedy decoding produces), exact match per test split, plus
tokens per sentence and training time. Writes results/curve_summary.md and prints it.

    python analysis/curve_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conlang.dataset import PRESETS, TEST_SPLITS, build  # noqa: E402
from conlang.frames import Vocab  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CELLS = {"easy-latin": ("english_like", "latin"), "easy-tamil": ("english_like", "abugida_tamil"),
         "hard-latin": ("tamil_like", "latin"), "hard-tamil": ("tamil_like", "abugida_tamil")}
# run name per (cell, N); the 50k runs for three cells came from the pilot stage
RUNS = {(c, n): f"c-{c}-{n // 1000}k" for c in CELLS for n in (5000, 10000, 20000)}
RUNS.update({("easy-latin", 50000): "l06-easy-latin-50k3", ("easy-tamil", 50000): "g-easy-tamil-50k",
             ("hard-tamil", 50000): "g-hard-tamil-50k", ("hard-latin", 50000): "c-hard-latin-50k"})


def content_word_accuracy(r: dict, data: dict, tok, content: set) -> float:
    right = total = 0
    for split in TEST_SPLITS:
        tf = r["eval"][split]["teacher_forced"]
        items = {it["id"]: it for it in data["splits"][split]}
        for iid, ok in zip(tf["per_item_ids"], tf["per_item_ok"]):
            it = items[iid]
            t, gl = it["target"], it["gloss"].split()
            if r["args"].get("target_tok") == "char":       # one character at a time (cause test M2)
                offs = [(i, i + 1) for i, ch in enumerate(t)
                        for _ in tok(ch, add_special_tokens=False)["input_ids"]]
            else:
                offs = tok(t, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            words: dict[int, list[bool]] = {}
            for (st, _), ch in zip(offs, ok[:-1]):
                if st < len(t) and t[st] == " ":
                    st += 1
                words.setdefault(t[:st].count(" "), []).append(ch == "1")
            for wi, oks in words.items():
                if gl[wi].split("-")[0].split(".")[0] in content:
                    total += 1
                    right += all(oks)
    return right / total


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    V = Vocab.large()
    content = set(V.nouns) | set(V.locations) | set(V.transitive + V.intransitive) | set(V.adjectives)
    data_cache: dict = {}
    rows = []
    for (cell, n), name in sorted(RUNS.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        p = ROOT / "results" / "kaggle" / name / "results" / f"{name}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        preset, script = CELLS[cell]
        if (preset, script) not in data_cache:
            data_cache[(preset, script)] = build(PRESETS[preset], script, n_train=50000,
                                                 n_test=int(r["args"]["n_test"]), seed=int(r["args"]["seed"]))
        cwa = content_word_accuracy(r, data_cache[(preset, script)], tok, content) \
            if "per_item_ok" in r["eval"]["test_iid"]["teacher_forced"] else None
        rows.append({"cell": cell, "n": n, "name": name, "tok_per_sent": r["data"]["mean_target_tokens"],
                     "train_min": r["training"]["seconds"] / 60, "cwa": cwa,
                     "batch": f"{r['args']['batch']}x{r['args']['grad_accum']}",
                     **{s: r["eval"][s]["teacher_forced"]["exact_match"] for s in TEST_SPLITS}})

    lines = ["# Learning curves: 2x2 (grammar x script), Qwen3-0.6B-Base, LoRA r=16, 3 epochs", "",
             "Content-word accuracy = every token of a content word right (script-fair). "
             "Exact = whole sentence right (greedy).", "",
             "| N | cell | tokens/sent | content-word acc | exact iid | exact comp | exact morph | exact long | train min | batch |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        cwa = f"{x['cwa']:.3f}" if x["cwa"] is not None else "-"
        lines.append(f"| {x['n']//1000}k | {x['cell']} | {x['tok_per_sent']:.1f} | {cwa} | {x['test_iid']:.3f} | "
                     f"{x['test_comp']:.3f} | {x['test_morph']:.3f} | {x['test_long']:.3f} | {x['train_min']:.1f} | {x['batch']} |")
    text = "\n".join(lines) + "\n"
    (ROOT / "results" / "curve_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
