"""Do the original maths runs and their -fo re-runs train on the same data? (01 Nov 2026)
Builds the training set + tests exactly as train/maths.py main() does (lines up to the leak checks), with the
maths.py of each code upload (a5da7ab = originals, 4af8f56 = -fo re-runs) and each run's saved arguments; compares
sha256 of the training items (order included) and of the random state right after building (= batch order seed)."""
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path("D:/Downloads/fllm-project")
NL = REPO / "new-language"
sys.path.insert(0, str(NL))
TMP = Path(__file__).parent / "maths_versions"
TMP.mkdir(exist_ok=True)


def load(commit):
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:new-language/train/maths.py"],
                         capture_output=True, check=True).stdout
    f = TMP / f"maths_{commit}.py"
    f.write_bytes(src)
    spec = importlib.util.spec_from_file_location(f"maths_{commit}", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_data(m, argv):
    """main()'s data path, stopped before the model is loaded."""
    captured = {}
    orig_from = None
    import transformers
    orig_from = transformers.AutoTokenizer.from_pretrained

    class Stop(Exception):
        pass

    def fake(*a, **k):                         # main loads the tokenizer after the data: stop there
        raise Stop()
    transformers.AutoTokenizer.from_pretrained = fake
    m.AutoTokenizer = transformers.AutoTokenizer if hasattr(m, "AutoTokenizer") else None
    real_build = m.build

    def spy_build(*a, **k):
        tr, te = real_build(*a, **k)
        captured["train"], captured["tests"] = tr, te
        return tr, te
    m.build = spy_build
    real_scr = m.scratchify

    def spy_scr(items, build=False):
        out = real_scr(items, build)
        if items is captured.get("train"):
            captured["train_scr"] = out
        return out
    m.scratchify = spy_scr
    try:
        m.main(argv)
    except Stop:
        pass
    except Exception as e:                     # anything after the data is built is irrelevant here
        captured["stopped_by"] = repr(e)[:120]
    finally:
        transformers.AutoTokenizer.from_pretrained = orig_from
    captured["py_rng"] = hashlib.sha256(repr(random.getstate()).encode()).hexdigest()[:12]
    captured["torch_rng"] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:12]
    return captured


def h(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


for run in ["maths-meaning-buildmix-drop56-s1", "maths-meaning-buildmix-drop66", "maths-meaning-buildmix-drop6row"]:
    args = json.loads((NL / "results/kaggle" / run / "results" / f"{run}.json").read_text(encoding="utf-8"))["args"]
    print(f"## {run}")
    for commit in ("a5da7ab", "4af8f56"):
        jobs = json.loads((NL / "kaggle_tools" / ("jobs_maths_forced_outputs.json" if commit == "4af8f56" else
                                                   "jobs_third_1031.json")).read_text(encoding="utf-8"))
        name = run + ("-fo" if commit == "4af8f56" else "")
        job = next((j for j in jobs if j["name"] == name), None)
        if job is None:
            jobs = json.loads((NL / "kaggle_tools" / "jobs_second_1031.json").read_text(encoding="utf-8"))
            job = next(j for j in jobs if j["name"] == name)
        argv = [x for x in job["args"] if x != "@maths"] + ["--out", str(TMP / "x.json")]
        m = load(commit)
        c = build_data(m, argv)
        tr = c.get("train_scr") or c.get("train")
        print(f"  {commit} ({name}): n_train {len(tr)}  train {h(tr)}  tests {h(c['tests'])}  "
              f"rng py {c['py_rng']} torch {c['torch_rng']}  {c.get('stopped_by', '')}")
