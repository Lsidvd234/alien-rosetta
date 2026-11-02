"""Run training jobs on Kaggle from this machine (uses the Kaggle API token in ~/.kaggle/kaggle.json).

Run with the D: environment that has the `kaggle` package:
    D:\\Downloads\\fllm-project\\.venv-vocab\\Scripts\\python.exe kaggle_tools\\kaggle_jobs.py <command>

Commands:
    code                         upload/refresh the code (conlang + train + vocab) as a private dataset
    push NAME -- ARGS...         push a GPU kernel that runs `train.run` with ARGS (e.g. --model ... --n-train ...)
    status NAME                  print the kernel status
    wait NAME                    poll until the kernel finishes (prints status every minute)
    fetch NAME                   download the kernel output into results/kaggle/NAME/
    queue JOBS.json              run a list of {"name", "args"} jobs, max concurrency Kaggle allows,
                                 fetching each result as it finishes
    upload FOLDER SLUG           upload a local folder (e.g. a grok checkpoint) as the private dataset SLUG
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".kaggle_stage"
DATASET_SLUG = "fllm-conlang-code"

BOOTSTRAP = r'''
import glob, os, subprocess, sys, zipfile
from importlib.metadata import version, PackageNotFoundError

JOB_NAME = __JOB_NAME__
JOB_ARGS = __JOB_ARGS__

def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=False)

def find_code():
    hits = glob.glob("/kaggle/input/**/conlang/__init__.py", recursive=True)
    if not hits:
        for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
            with zipfile.ZipFile(z) as zf:
                if any(n.endswith("conlang/__init__.py") for n in zf.namelist()):
                    zf.extractall("/kaggle/working/code")
        hits = glob.glob("/kaggle/working/code/**/conlang/__init__.py", recursive=True)
    if not hits:
        sh("ls -R /kaggle/input | head -50")
        raise SystemExit("code not found under /kaggle/input")
    return os.path.dirname(os.path.dirname(hits[0]))

def ver(pkg):
    try:
        return tuple(int(x) for x in version(pkg).split(".")[:2])
    except (PackageNotFoundError, ValueError):
        return (0, 0)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
code = find_code()
print("code at", code, flush=True)
sys.path.insert(0, code)
if ver("transformers") < (4, 51) or ver("peft") < (0, 13):
    sh('pip install -q "transformers>=4.51" "peft>=0.13" accelerate')
# Kaggle ships torchao 0.10, which recent peft rejects with an ImportError; we don't use it.
if ver("torchao") != (0, 0) and ver("torchao") < (0, 16):
    sh("pip uninstall -y -q torchao")
print("transformers", version("transformers"), "peft", version("peft"), flush=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")

os.makedirs("/kaggle/working/results", exist_ok=True)
if JOB_ARGS and JOB_ARGS[0].startswith("@paragen"):  # @paragen:SEED:TARGET  paraphrases only (v2), one process/GPU
    import json as _json
    import torch as _torch
    _, pseed, ptarget = JOB_ARGS[0].split(":")
    ng = max(1, _torch.cuda.device_count())
    extra = " ".join(JOB_ARGS[1:])
    procs = [subprocess.Popen(f"cd {code} && CUDA_VISIBLE_DEVICES={g} python scripts/build_paraphrases_v2.py "
                              f"--seed {pseed} --target {ptarget} --engine hf --shard {g} --nshards {ng} "
                              f"--out /kaggle/working/para_part{g}.json {extra} > /kaggle/working/para_part{g}.log 2>&1",
                              shell=True) for g in range(ng)]
    for p_ in procs:
        p_.wait()
    merged = {}
    for g in range(ng):
        sh(f"tail -n 25 /kaggle/working/para_part{g}.log")
        try:
            merged.update(_json.load(open(f"/kaggle/working/para_part{g}.json")))
        except Exception as e:
            print("part", g, "missing:", e)
    _json.dump(merged, open(f"/kaggle/working/paraphrases_v2_seed{pseed}.json", "w"), ensure_ascii=False)
    print("merged paraphrases:", len(merged), flush=True)
    raise SystemExit(0)
if JOB_ARGS and JOB_ARGS[0] == "@ff":               # Faith and Fate: real model step-by-step answers
    sh(f"cd {code}/ff && python run_ff_models.py --out /kaggle/working/ff_answers " + " ".join(JOB_ARGS[1:]))
    raise SystemExit(0)
if JOB_ARGS and JOB_ARGS[0] == "@grok":             # train/grok.py: one chunk of a grokking run (resumable)
    from train.grok import main
    main(JOB_ARGS[1:] + ["--out", f"/kaggle/working/results/{JOB_NAME}.json", "--save", "/kaggle/working/ckpt"])
    raise SystemExit(0)
if JOB_ARGS and JOB_ARGS[0] == "@maths":            # train/maths.py instead of train/run.py
    from train.maths import main
    JOB_ARGS = JOB_ARGS[1:]
else:
    from train.run import main
if JOB_ARGS and JOB_ARGS[0].startswith("@para"):     # @para:SEED:SHARE  paraphrase the training English first
    _, pseed, pshare = JOB_ARGS[0].split(":")
    pout = f"/kaggle/working/paraphrases_seed{pseed}.json"
    sh(f"cd {code} && python scripts/build_paraphrases.py --seed {pseed} --share {pshare} --batch 48 --out {pout}")
    JOB_ARGS = JOB_ARGS[1:] + ["--rep-paraphrases", pout]
main(JOB_ARGS + ["--out", f"/kaggle/working/results/{JOB_NAME}.json"])
'''


def timed(fn, *args, timeout: float = 90, **kw):
    """Call fn with a time limit: a Kaggle API request can hang forever (seen 2026-10-15); raise TimeoutError."""
    import threading
    box: dict = {}

    def run():
        try:
            box["v"] = fn(*args, **kw)
        except Exception as e:                 # passed on to the caller
            box["e"] = e
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"{getattr(fn, '__name__', fn)} took longer than {timeout:.0f} s")
    if "e" in box:
        raise box["e"]
    return box.get("v")


def api() -> KaggleApi:
    a = KaggleApi()
    a.authenticate()
    return a


def user(a: KaggleApi) -> str:
    return a.get_config_value("username")


def cmd_code(a: KaggleApi) -> None:
    folder = STAGE / "dataset"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    zpath = folder / "fllm_code.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for sub in ("conlang", "train"):
            for f in (ROOT / sub).rglob("*"):
                if f.is_file() and "__pycache__" not in f.parts and f.name != "wikitext2_test.txt":  # eval-only file
                    zf.write(f, f.relative_to(ROOT).as_posix())
        for f in (ROOT / "sealed" / "app" / "english_parse.py", ROOT / "scripts" / "build_paraphrases.py",
                  ROOT / "scripts" / "build_paraphrases_v2.py"):
            zf.write(f, f.relative_to(ROOT).as_posix())         # the reader only (no sealed data), for paraphrases
        zf.write(ROOT / "results" / "tatoeba_test.json", "eval/tatoeba_test.json")   # inputs for --extra-translate
        ff = ROOT.parent / "papers"                              # Faith and Fate: our model runner + their generator
        for src, dst in ((ff / "ff_tools" / "run_ff_models.py", "ff/run_ff_models.py"),
                         (ff / "faith-and-fate" / "multiplication" / "generate_scratchpads.py",
                          "ff/generate_scratchpads.py")):
            if src.exists():
                zf.write(src, dst)
    print(f"code zip: {zpath.stat().st_size/1e6:.2f} MB")
    upload_dataset(a, folder, DATASET_SLUG)


def upload_dataset(a: KaggleApi, folder: Path, slug: str) -> None:
    """Create or version the private dataset user/slug from the files in folder; wait until Kaggle has it ready."""
    meta = {"title": slug, "id": f"{user(a)}/{slug}", "licenses": [{"name": "CC0-1.0"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    try:
        a.dataset_status(meta["id"])
        exists = True
    except Exception:
        exists = False
    if exists:
        r = a.dataset_create_version(str(folder), version_notes=time.strftime("%Y-%m-%d %H:%M"),
                                     quiet=True, convert_to_csv=False, dir_mode="skip")
    else:
        r = a.dataset_create_new(str(folder), public=False, quiet=True, convert_to_csv=False, dir_mode="skip")
    print("dataset:", meta["id"], "->", getattr(r, "status", r), getattr(r, "error", ""))
    for _ in range(30):                       # wait until Kaggle has processed it
        try:
            st = a.dataset_status(meta["id"])
        except Exception as e:
            st = str(e)
        print("  status:", st)
        if str(st).lower() == "ready":
            return
        time.sleep(10)


def cmd_push(a: KaggleApi, name: str, args: list[str], machine: str | None = None,
             extra_sources: list[str] | None = None) -> str | None:
    """Push one job; returns the error message (None on success). GPU: env KAGGLE_MACHINE (default T4).
    extra_sources: more private datasets (slugs of this user) to attach, e.g. a checkpoint to resume from."""
    machine = os.environ.get("KAGGLE_MACHINE", "NvidiaTeslaT4") if machine is None else machine
    slug = f"fllm-{name}"
    folder = STAGE / "kernels" / slug
    folder.mkdir(parents=True, exist_ok=True)
    code = BOOTSTRAP.replace("__JOB_NAME__", repr(name)).replace("__JOB_ARGS__", repr(args))
    (folder / "run_job.py").write_text(code, encoding="utf-8")
    meta = {"id": f"{user(a)}/{slug}", "title": slug, "code_file": "run_job.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [f"{user(a)}/{DATASET_SLUG}"] + [f"{user(a)}/{s}" for s in (extra_sources or [])],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    if machine:
        meta["machine_shape"] = machine
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    r = timed(a.kernels_push, str(folder), timeout=180)
    err = getattr(r, "error", None)
    print(time.strftime("%H:%M:%S"), "push", name, getattr(r, "ref", ""),
          "version", getattr(r, "version_number", getattr(r, "versionNumber", "")), "error", err or "")
    if err and machine and ("machine" in err.lower() or "shape" in err.lower()):
        print("retrying without machine_shape")
        return cmd_push(a, name, args, machine="", extra_sources=extra_sources)
    return err or None


def cmd_queue(a: KaggleApi, jobs_file: Path) -> None:
    """Run a list of jobs, at most as many at once as Kaggle allows; fetch each when done."""
    jobs = json.loads(Path(jobs_file).read_text())
    pending, running, done = [], [], []
    # resume: skip jobs already fetched; adopt jobs already running/finished on Kaggle
    for job in jobs:
        name = job["name"]
        if (ROOT / "results" / "kaggle" / name / "results" / f"{name}.json").exists():
            done.append((name, "already fetched"))
            continue
        try:
            st = str(getattr(timed(a.kernels_status, ref(a, name)), "status", "")).lower()
        except Exception:
            st = ""                                    # never pushed
        if "running" in st or "queued" in st:
            running.append(name)
        elif "complete" in st:
            running.append(name)                       # fetched in the loop below
        else:
            pending.append(job)
    print("resume:", len(done), "done,", len(running), "running/finished,", len(pending), "to push")
    while pending or running:
        # push as many as Kaggle accepts
        while pending:
            job = pending[0]
            try:
                err = cmd_push(a, job["name"], job["args"])
            except Exception as e:                     # network drop: wait and retry
                print(time.strftime("%H:%M:%S"), "push failed (network?), retrying later:", type(e).__name__)
                break
            if err and any(k in err.lower() for k in ("maximum", "quota", "limit", "gpu", "accelerator")):
                print(time.strftime("%H:%M:%S"), "not pushed yet (slot or quota):", err[:120], flush=True)
                break                                  # no free GPU slot / quota yet: retry later, never drop the job
            pending.pop(0)
            if err:
                done.append((job["name"], "push-error: " + err))
            else:
                running.append(job["name"])
        time.sleep(60)
        for name in list(running):
            try:
                st = str(getattr(timed(a.kernels_status, ref(a, name)), "status", "")).lower()
            except Exception as e:                       # transient API hiccup (or a request that hung)
                print("status error", name, e)
                continue
            if any(k in st for k in ("complete", "error", "cancel")):
                running.remove(name)
                print(time.strftime("%H:%M:%S"), name, st)
                try:
                    cmd_fetch(a, name)
                except Exception as e:
                    print("fetch error", name, e)
                done.append((name, st))
    print("\nQUEUE FINISHED")
    for name, st in done:
        print(f"  {name}: {st}")


def ref(a: KaggleApi, name: str) -> str:
    return f"{user(a)}/fllm-{name}"


def cmd_status(a: KaggleApi, name: str) -> str:
    s = timed(a.kernels_status, ref(a, name))
    status = str(getattr(s, "status", s))
    msg = getattr(s, "failure_message", getattr(s, "failureMessage", ""))
    print(time.strftime("%H:%M:%S"), status, msg or "")
    return status.lower()


def cmd_wait(a: KaggleApi, name: str) -> None:
    while True:
        st = cmd_status(a, name)
        if any(k in st for k in ("complete", "error", "cancel")):
            return
        time.sleep(60)


def cmd_fetch(a: KaggleApi, name: str) -> None:
    out = ROOT / "results" / "kaggle" / name
    out.mkdir(parents=True, exist_ok=True)
    timed(a.kernels_output, ref(a, name), str(out), timeout=1800)
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"{f.stat().st_size:>10}  {f.relative_to(out)}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__); return
    a = api()
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "code":
        cmd_code(a)
    elif cmd == "push":
        name = rest[0]
        args = rest[rest.index("--") + 1:] if "--" in rest else []
        cmd_push(a, name, args)
    elif cmd == "status":
        cmd_status(a, rest[0])
    elif cmd == "wait":
        cmd_wait(a, rest[0])
    elif cmd == "fetch":
        cmd_fetch(a, rest[0])
    elif cmd == "queue":
        cmd_queue(a, Path(rest[0]))
    elif cmd == "upload":
        upload_dataset(a, Path(rest[0]), rest[1])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
