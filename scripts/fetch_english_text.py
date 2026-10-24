"""Download the plain-English text used against forgetting (training replay) and for measuring it.

- FineWeb-Edu (sample-10BT; ODC-By): first ~5,000 documents, cut into passages of at most 320 characters at sentence ends.
  Written to train/data/replay_en.jsonl (shipped with the Kaggle job; used for training).
- WikiText-2 raw test split (CC BY-SA): written to train/data/wikitext2_test.txt (measurement only, never trained on).

Uses the Hugging Face datasets-server rows API, so no multi-GB files are downloaded.
    python scripts/fetch_english_text.py
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "train" / "data"
API = "https://datasets-server.huggingface.co/rows"


def rows(dataset: str, config: str, split: str, offset: int, length: int = 100) -> list[dict]:
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                "offset": offset, "length": length})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
                return [x["row"] for x in json.loads(r.read())["rows"]]
        except Exception as e:                       # rate limits / transient errors: back off and retry
            print(f"  retry {attempt + 1} after {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"giving up on {dataset} offset {offset}")


def passages(text: str, size: int = 320) -> list[str]:   # ~75 tokens: fits training memory
    sents = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    out, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 > size and cur:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return [p for p in out if 120 <= len(p) <= size and p.isascii()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # FineWeb-Edu replay passages
    psg: list[str] = []
    off = 0
    while len(psg) < 6000 and off < 8000:
        for r in rows("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", off):
            psg += passages(r["text"])[:2]               # at most 2 passages per document, for variety
        off += 100
        print(f"fineweb-edu: {off} docs, {len(psg)} passages", flush=True)
        time.sleep(0.5)
    with (OUT / "replay_en.jsonl").open("w", encoding="utf-8") as f:
        for p in psg[:6000]:
            f.write(json.dumps({"text": p}) + "\n")
    # WikiText-2 test (measurement only)
    lines: list[str] = []
    off = 0
    while True:
        rr = rows("Salesforce/wikitext", "wikitext-2-raw-v1", "test", off)
        if not rr:
            break
        lines += [r["text"] for r in rr]
        off += 100
        if len(rr) < 100:
            break
        time.sleep(0.3)
    (OUT / "wikitext2_test.txt").write_text("".join(lines), encoding="utf-8")
    print(f"done: {min(len(psg), 6000)} replay passages, {len(lines)} wikitext lines")


if __name__ == "__main__":
    main()
