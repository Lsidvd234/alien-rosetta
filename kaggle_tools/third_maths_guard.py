"""Third account, 28 Oct (user): queue the maths drop-twin runs LAST, and drop them if the account would overshoot.

    KAGGLE_CONFIG_DIR=../.kaggle3 python kaggle_tools/third_maths_guard.py      (log: results/queue_third_maths.log)

1. Wait until the main third-account queue (results/queue_third_2026-10-28.log) has pushed e2b-v2fix-everyday, so
   the maths runs never take a GPU slot before the passive / everyday runs.
2. Project the account's GPU hours: finished runs = their real time (push -> complete in the log), the runs still
   going = the longest finished run's time (or 6.1 h if none finished). If projected + 2.3 h (maths) > LIMIT, skip.
3. Otherwise run `kaggle_jobs.py queue kaggle_tools/jobs_maths_drop_twin.json` (2 at a time, fetched when done).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_LOG = ROOT / "results" / "queue_third_2026-10-28.log"
LIMIT = 29.0          # GPU-h: the weekly quota is ~30; keep a margin
MATHS = 2.3           # the four maths runs, with overrun
RUNS = ["e2b-v2fix-passive0", "e2b-v2fix-passive50", "e2b-v2fix-passive300", "e2b-v2fix-everyday"]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def times(text: str) -> tuple[dict, dict]:
    """Push and finish times per run from the queue log (today's date)."""
    day = datetime.now().date()
    pushed, done = {}, {}
    for t, name in re.findall(r"^(\d\d:\d\d:\d\d) push (\S+) /code/\S+ version \d+ error\s*$", text, re.M):
        pushed.setdefault(name, t)
    for t, name in re.findall(r"^(\d\d:\d\d:\d\d) (\S+) \S*complete", text, re.M):
        done.setdefault(name, t)
    as_dt = (lambda s: datetime.combine(day, datetime.strptime(s, "%H:%M:%S").time()))
    return {k: as_dt(v) for k, v in pushed.items()}, {k: as_dt(v) for k, v in done.items()}


def main() -> None:
    log("waiting for e2b-v2fix-everyday to be pushed by the main third-account queue")
    while True:
        pushed, _ = times(MAIN_LOG.read_text(encoding="utf-8", errors="replace"))
        if "e2b-v2fix-everyday" in pushed:
            break
        time.sleep(120)
    pushed, done = times(MAIN_LOG.read_text(encoding="utf-8", errors="replace"))
    hours = {}
    for r in RUNS:
        if r in done and r in pushed:
            d = done[r] - pushed[r]
            hours[r] = (d + timedelta(days=1) if d.total_seconds() < 0 else d).total_seconds() / 3600
    est = max(hours.values()) if hours else 6.1
    projected = sum(hours.get(r, est) for r in RUNS)
    log(f"finished runs (h): { {k: round(v, 2) for k, v in hours.items()} }; others assumed {est:.2f} h each")
    log(f"projected passive + everyday: {projected:.1f} GPU-h; + maths {MATHS} = {projected + MATHS:.1f} (limit {LIMIT})")
    if projected + MATHS > LIMIT:
        log("SKIPPED: the maths runs would overshoot the third account; they stay on main for 31 Oct")
        return
    log("queueing kaggle_tools/jobs_maths_drop_twin.json on the third account")
    subprocess.run([sys.executable, "-u", str(ROOT / "kaggle_tools" / "kaggle_jobs.py"), "queue",
                    str(ROOT / "kaggle_tools" / "jobs_maths_drop_twin.json")], env=os.environ, check=False)
    log("maths queue ended")


if __name__ == "__main__":
    main()
