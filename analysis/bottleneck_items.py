"""Where exactly does the writer fail? Per-sentence reader x writer, from the gold meaning (user, 01 Nov 2026).

    python analysis/bottleneck_items.py   -> results/bottleneck_items.md + .json (per item; sealed: counts only)

Same models and tests as analysis/bottleneck.py (reader e2b-v2fix-passive0-meaning, writer e2b-v2fix-passive0-writer).
The writer is given the GOLD meaning for every sentence, including the ones the reader gets wrong. Questions:
  1. Are the sentences the reader misreads also the ones the writer cannot write (hard for both), or is the writer
     fine on them? -> 2 x 2 table: reader right/wrong x writer-from-gold right/wrong.
  2. Do writer failures sit on meanings with words the writer never (or rarely) saw in training? -> writer accuracy by
     the least-trained word key of the gold meaning (counts = passive0's training analyses, the writer's data).
  3. By meaning length (number of word keys).
Per-item outputs are saved for Tatoeba and the generated sets; the sealed set gives totals only (no sentence, no
output, no per-item flag is written or printed). Predictions WB1-WB3 in results/findings_2026-10-24.md.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from bottleneck import generate, load, tests  # noqa: E402
from tatoeba_rare_words import keys  # noqa: E402

READER, WRITER, SINGLE = "e2b-v2fix-passive0-meaning", "e2b-v2fix-passive0-writer", "e2b-v2fix-passive0"
SEALED = "sealed (totals only)"


def bucket_seen(m: int | None) -> str:
    if m is None:
        return "no word keys"
    return "a key never trained (0)" if m == 0 else "rarest key 1-24" if m < 25 else "all keys >= 25"


def bucket_len(n: int) -> str:
    return "<= 4 keys" if n <= 4 else "5-7 keys" if n <= 7 else ">= 8 keys"


def main() -> None:
    from train.run import prompt, writer_prompt
    counts = json.loads((ROOT / "results" / "freq_weighted" / f"{SINGLE}.counts.json").read_text(encoding="utf-8"))
    T = tests(SINGLE)
    pred = {k: {} for k in T}
    model, tok = load(READER)
    for k, items in T.items():
        pred[k]["reader"] = generate(model, tok, [prompt(it["english"]) for it in items], 300)
    del model
    torch.cuda.empty_cache()
    model, tok = load(WRITER)
    for k, items in T.items():
        reader_meaning = [p.partition("=>")[0].strip() for p in pred[k]["reader"]]
        pred[k]["writer_gold"] = generate(model, tok, [writer_prompt(it["analysis"]) for it in items], 400)
        pred[k]["writer_pipe"] = generate(model, tok, [writer_prompt(m) for m in reader_meaning], 400)
    del model
    torch.cuda.empty_cache()

    L = ["# Where exactly does the writer fail? (`analysis/bottleneck_items.py`)", "",
         f"reader = {READER}, writer = {WRITER} (given the GOLD meaning for every sentence). Key counts = word keys "
         f"in {SINGLE}'s training analyses (the writer's data). Sealed: totals only.", ""]
    saved, summary = {}, {}
    for k, items in T.items():
        P = pred[k]
        read = [p.partition("=>")[0].strip() == it["analysis"] for p, it in zip(P["reader"], items)]
        wg = [g == it["target"] for g, it in zip(P["writer_gold"], items)]
        wp = [g == it["target"] for g, it in zip(P["writer_pipe"], items)]
        kc = [keys(it["analysis"]) for it in items]
        least = [min((counts.get(x, 0) for x in c), default=None) for c in kc]
        nkeys = [sum(c.values()) for c in kc]
        n = len(items)
        tab = collections.Counter((r, w) for r, w in zip(read, wg))
        by_seen, by_len = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(lambda: [0, 0])
        by_seen_read = collections.defaultdict(lambda: [0, 0])
        for i in range(n):
            b = bucket_seen(least[i])
            by_seen[b][0] += 1; by_seen[b][1] += wg[i]
            by_seen_read[b][0] += 1; by_seen_read[b][1] += read[i]
            bl = bucket_len(nkeys[i])
            by_len[bl][0] += 1; by_len[bl][1] += wg[i]
        rr, rw, wr, ww = tab[(True, True)], tab[(True, False)], tab[(False, True)], tab[(False, False)]
        wr_read = rr / (rr + rw) if rr + rw else float("nan")
        wr_misread = wr / (wr + ww) if wr + ww else float("nan")
        summary[k] = {"n": n, "reader_right": sum(read), "writer_gold_right": sum(wg), "pipeline_right": sum(wp),
                      "reader_right_writer_right": rr, "reader_right_writer_wrong": rw,
                      "reader_wrong_writer_right": wr, "reader_wrong_writer_wrong": ww,
                      "writer_gold_acc_where_reader_right": wr_read, "writer_gold_acc_where_reader_wrong": wr_misread,
                      "by_least_seen_key": {b: v for b, v in by_seen.items()},
                      "reading_by_least_seen_key": {b: v for b, v in by_seen_read.items()},
                      "by_meaning_length": {b: v for b, v in by_len.items()}}
        L += [f"## {k} (n = {n})", "",
              f"Reader right {sum(read)}, writer from gold right {sum(wg)}, pipeline right {sum(wp)}.", "",
              "| | writer (gold meaning) right | writer wrong | writer accuracy |", "|---|---|---|---|",
              f"| reader right | {rr} | {rw} | {wr_read:.2f} |",
              f"| reader wrong | {wr} | {ww} | {wr_misread:.2f} |", "",
              "| least-trained word key of the gold meaning | n | writer accuracy | reader accuracy |", "|---|---|---|---|"]
        for b in ("a key never trained (0)", "rarest key 1-24", "all keys >= 25", "no word keys"):
            if b in by_seen:
                v, vr = by_seen[b], by_seen_read[b]
                L.append(f"| {b} | {v[0]} | {v[1] / v[0]:.2f} | {vr[1] / vr[0]:.2f} |")
        L += ["", "| meaning length | n | writer accuracy |", "|---|---|---|"]
        for b in ("<= 4 keys", "5-7 keys", ">= 8 keys"):
            if b in by_len:
                v = by_len[b]
                L.append(f"| {b} | {v[0]} | {v[1] / v[0]:.2f} |")
        L.append("")
        if k != SEALED:
            saved[k] = [{"english": it["english"], "gold_meaning": it["analysis"], "reader": P["reader"][i],
                         "writer_gold": P["writer_gold"][i], "gold_target": it["target"], "reader_right": read[i],
                         "writer_gold_right": wg[i], "pipeline_right": wp[i], "least_key_count": least[i]}
                        for i, it in enumerate(items)]
    (ROOT / "results" / "bottleneck_items.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "results" / "bottleneck_items.json").write_text(
        json.dumps({"summary": summary, "items": saved}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
