"""Download a finished Kaggle run (results JSON, log, saved model and per-epoch checkpoints) and check it.

Run with the D: environment that has the `kaggle` package (from new-language/):
    D:\\Downloads\\fllm-project\\.venv-vocab\\Scripts\\python.exe kaggle_tools\\fetch_run.py r1-hard-aug-50k
    ... fetch_run.py r1-hard-aug-50k --status      # only print the run's status
    ... fetch_run.py r1-hard-aug-50k --force       # download again even if already present

Files land in results/kaggle/<NAME>/ :
    results/<NAME>.json                 scores (4 test splits + the augmented categories in "eval_aug")
    model/ , model_epoch1/ , ...        LoRA adapter + tokenizer + glyph_tokens.pt + run_args.json
Load a model locally with train/load.py (see its docstring).

If Kaggle returns file names with a stray Windows prefix (it happened once when a job was pushed from Git Bash:
'C:/Program Files/Git/kaggle/working/...'), the files are moved to the expected place automatically.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kaggle_jobs import api, ref  # noqa: E402

NEEDED_MODEL_FILES = ["adapter_model.safetensors", "adapter_config.json", "glyph_tokens.pt", "run_args.json",
                      "tokenizer.json"]


def status(a, name: str) -> str:
    s = a.kernels_status(ref(a, name))
    st = str(getattr(s, "status", s))
    msg = getattr(s, "failure_message", getattr(s, "failureMessage", "")) or ""
    print(f"{name}: {st} {msg}".strip())
    return st.lower()


def fix_prefixes(out: Path) -> None:
    """Move files saved under a mangled absolute path back to <out>/<name>."""
    for f in list(out.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(out).as_posix()
        if "kaggle/working/" in rel and not rel.startswith(("model", "results")):
            dst = out / rel.split("kaggle/working/", 1)[1]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dst))
            print(f"  moved {rel} -> {dst.relative_to(out).as_posix()}")


def check(out: Path, name: str) -> bool:
    ok = True
    res = out / "results" / f"{name}.json"
    if res.exists():
        r = json.loads(res.read_text(encoding="utf-8"))
        print("\nscores (teacher-forced exact match):")
        for split, v in r.get("eval", {}).items():
            print(f"  {split:<12} {v['teacher_forced']['exact_match']:.3f}")
        if r.get("eval_aug"):
            print("  augmented categories:")
            for k, v in r["eval_aug"].items():
                print(f"    {k:<42} n={v['n']:<4} exact={v['exact_match']:.3f}")
        print(f"  training: {r.get('training', {}).get('seconds', 0) / 60:.1f} min")
    else:
        print(f"MISSING: {res.relative_to(ROOT)}")
        ok = False
    models = sorted(p for p in out.iterdir() if p.is_dir() and p.name.startswith("model"))
    for m in models:
        missing = [f for f in NEEDED_MODEL_FILES if not (m / f).exists()]
        size = sum(f.stat().st_size for f in m.rglob("*") if f.is_file()) / 1e6
        print(f"{m.name:<14} {size:7.1f} MB  " + ("complete" if not missing else f"MISSING {missing}"))
        ok &= not missing
    if not models:
        print("no saved model folder found")
    return ok


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    name = sys.argv[1]
    a = api()
    st = status(a, name)
    if "--status" in sys.argv:
        return
    if not any(k in st for k in ("complete", "error", "cancel")):
        print("not finished yet; nothing downloaded")
        return
    out = ROOT / "results" / "kaggle" / name
    if out.exists() and (out / "results" / f"{name}.json").exists() and "--force" not in sys.argv:
        print(f"already downloaded to {out} (use --force to download again)")
    else:
        out.mkdir(parents=True, exist_ok=True)
        print(f"downloading to {out} ...")
        a.kernels_output(ref(a, name), str(out))
        fix_prefixes(out)
    print("all files present" if check(out, name) else "SOME FILES ARE MISSING (see above)")


if __name__ == "__main__":
    main()
