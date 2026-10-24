"""Run a grokking run (train/grok.py) on Kaggle as a chain of chunks, with the curve checked after every chunk.

    set KAGGLE_CONFIG_DIR=...   (second account; omit for the main one)
    python kaggle_tools/grok_chain.py NAME --chunks 3 -- --task maths --steps-total 30000 --chunk-steps 10000 ...

Chunk i is the kernel fllm-NAME-c{i}. After it finishes: download it, merge its curve into results/grok/NAME.json,
plot results/grok/NAME.png, print the latest row, upload its checkpoint as the NEW private dataset grok-ckpt-NAME-c{i},
and push chunk i+1 with that dataset attached (train/grok.py --resume auto finds it). Stops when the run reaches
--steps-total, when test accuracy has been >= 0.95 at the last two evaluations (generalised: nothing left to see), or
on a kernel error. Restartable: chunks already downloaded are not run again. Kaggle shows nothing while a kernel runs,
so this is how the curve is watched mid-run.

One NEW dataset per chunk (2026-10-24): re-versioning one dataset raced. dataset_status said "ready" for the OLD version,
so grok-translate-wd1-c2 resumed from step 4000 instead of 8000 and redid chunk 2. A chunk that ends no further than
the previous one is now reported as a stale resume and the chain stops. A chunk kernel that is already running (chain
restarted mid-chunk) is waited for, not pushed again.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
K = None                                                         # kaggle_jobs, imported in main (plotting needs no API)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "grok"


def status(a, kname: str) -> str:
    try:
        return str(getattr(K.timed(a.kernels_status, K.ref(a, kname)), "status", "")).lower()
    except Exception as e:                                       # network hiccup: try again next minute
        return f"unknown ({type(e).__name__})"


def plot(name: str, curve: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    acc = [k for k in curve[0] if k.startswith(("train (", "test", "extra"))]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    xs = [max(r["step"], 1) for r in curve]
    for k in acc:
        ax[0].plot(xs, [r[k] for r in curve], marker=".", label=k)
    ax[0].set(xscale="log", xlabel="training step (log)", ylabel="exact match", ylim=(-0.02, 1.02), title=name)
    ax[0].legend(fontsize=8)
    if "zero_grad_fp32" in curve[0]:
        ax[1].plot(xs, [r["zero_grad_fp32"] for r in curve], label="zero-gradient share (fp32 softmax = 1.0)")
        ax2 = ax[1].twinx()
        ax2.plot(xs, [r["gap_median"] for r in curve], color="gray", ls="--", label="median logit gap")
        ax[1].set(xscale="log", xlabel="training step (log)", ylabel="zero-gradient share", ylim=(-0.02, 1.02),
                  title="softmax collapse")
        ax2.set_ylabel("median logit gap")
        ax[1].legend(loc="upper left", fontsize=8)
        ax2.legend(loc="lower right", fontsize=8)
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    try:
        fig.savefig(OUT / f"{name}.png", dpi=130)
    except OSError as e:
        print("plot not saved:", e)


def main() -> None:
    if sys.argv[1] == "--plot-only":                             # plot from the saved record (main python env)
        rec = json.loads((OUT / f"{sys.argv[2]}.json").read_text(encoding="utf-8"))
        plot(sys.argv[2], rec["curve"])
        return
    global K
    import kaggle_jobs as K  # noqa: E402
    name = sys.argv[1]
    i = sys.argv.index("--")
    opts, gargs = sys.argv[2:i], sys.argv[i + 1:]
    chunks = int(opts[opts.index("--chunks") + 1]) if "--chunks" in opts else 3
    start_from = opts[opts.index("--start-from") + 1] if "--start-from" in opts else None   # another chain's ckpt
    a = K.api()
    OUT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT / f"{name}.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {"name": name, "chunks": []}
    for c in range(chunks):
        kname = f"{name}-c{c}"
        if kname in rec["chunks"]:
            if kname not in rec.get("uploaded", []) and c + 1 < chunks:   # restart after a crash before the upload
                ck = ROOT / "results" / "kaggle" / kname / "ckpt"
                K.upload_dataset(a, ck, f"grok-ckpt-{name}-c{c}")
                rec.setdefault("uploaded", []).append(kname)
                rec_path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            continue                                             # already done (restart)
        args = ["@grok"] + gargs + (["--resume", "auto"] if c > 0 or start_from else [])
        prev = f"{name}-c{c - 1}"
        old_style = c > 0 and prev in rec.get("uploaded", []) and prev not in rec.get("uploaded_per_chunk", [])
        extra = ([f"grok-ckpt-{name}" if old_style else f"grok-ckpt-{name}-c{c - 1}"] if c > 0
                 else ([start_from] if start_from else []))
        prev_step = rec.get("step", 0) if c > 0 else 0
        st0 = status(a, kname)
        while not any(x in st0 for x in ("running", "queued")):  # already running: a restarted chain just waits
            err = K.cmd_push(a, kname, args, extra_sources=extra)
            if not err:
                break
            if "maximum" in err.lower() or "session" in err.lower():
                time.sleep(120)
                continue
            print("push failed:", err)
            return
        t0 = time.time()
        while True:
            time.sleep(60)
            st = status(a, kname)
            if any(x in st for x in ("complete", "error", "cancel")):
                break
            if int(time.time() - t0) % 900 < 60:
                print(time.strftime("%H:%M"), kname, st, flush=True)
        print(time.strftime("%H:%M"), kname, st, f"{(time.time() - t0) / 60:.0f} min", flush=True)
        K.cmd_fetch(a, kname)
        res_f = ROOT / "results" / "kaggle" / kname / "results" / f"{kname}.json"
        if "complete" not in st or not res_f.exists():
            print("chunk failed; see", ROOT / "results" / "kaggle" / kname)
            return
        res = json.loads(res_f.read_text(encoding="utf-8"))
        if c > 0 and res["step"] <= prev_step:
            print(f"STALE RESUME: chunk ended at step {res['step']}, the previous chunk already reached {prev_step}; "
                  f"it resumed from an old checkpoint. Stopping.")
            return
        rec.update({"curve": res["curve"], "step": res["step"], "steps_total": res["steps_total"],
                    "steps_per_epoch": res["steps_per_epoch"], "data": res.get("data")})
        rec["chunks"].append(kname)
        rec_path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        try:
            plot(name, res["curve"])
        except ImportError:                                      # the Kaggle tool environment has no matplotlib
            import subprocess
            subprocess.run(["python", str(Path(__file__)), "--plot-only", name], check=False)
        last = res["curve"][-1]
        print("LATEST", json.dumps(last), flush=True)
        tests = [k for k in last if k.startswith("test")]
        late = res["curve"][-2:]
        if res["done"]:
            print("run finished (all steps)")
            return
        if tests and len(late) == 2 and all(r[tests[0]] >= 0.95 for r in late):
            print("test accuracy >= 0.95 twice: generalised, stopping early")
            return
        ck = ROOT / "results" / "kaggle" / kname / "ckpt"
        if not (ck / "grok_state.pt").exists():
            print("no checkpoint in", ck)
            return
        K.upload_dataset(a, ck, f"grok-ckpt-{name}-c{c}")
        rec.setdefault("uploaded", []).append(kname)
        rec.setdefault("uploaded_per_chunk", []).append(kname)
        rec_path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("all chunks done")


if __name__ == "__main__":
    main()
