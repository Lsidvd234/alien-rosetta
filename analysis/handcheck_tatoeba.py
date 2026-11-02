"""Hand-check of Tatoeba gold labels: how much of real text's Tatoeba gain is real? (claim 4, 2026-10-27)

    python analysis/handcheck_tatoeba.py build            -> results/handcheck/tatoeba_handcheck.html (+ key.json)
    python analysis/handcheck_tatoeba.py score FILE.json  -> results/handcheck/handcheck.md

Why: Tatoeba's gold meanings and real15's 12k training meanings come from the same automatic reader, so real15 may
score 0.92 partly by copying the reader's habits, including its mistakes. The student judges gold meanings by hand,
without being told which items are which:
  gain    50 of the sentences (189) real15 gets right and the baseline gets wrong (there real15's output IS the gold)
  control 20 random sentences from the other 411
If the gold is wrong much more often in 'gain' than in 'control', part of the gain is the reader's errors learned;
if both are about equally right, the gain is real. Tatoeba is not the sealed set: reading these sentences is fine.
"""

from __future__ import annotations

import json
import math
import random
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "handcheck"
REAL, BASE = "e2b-v2fix-real15", "e2b-v2fix-cot-80k-noheld"

GUIDE = """S= subject, V= verb (with PRES/PAST/FUT, NEG, Q question, PROG -ing, PERF have+done, CAN/MUST/SHOULD/MAY/WOULD),
O= object, P= 'is a ...' noun, A= 'is ...' adjective, M= how/when word (adverb), L= place, T= time,
TO=/FROM=/WITH=/FOR=/ON=/ABOUT=... = phrase with that preposition, C= joining word, I= interjection, Y= person addressed.
PRON.1sg I, 2sg you, 3sgm he, 3sgf she, 1pl we, 3pl they; NAME.Tom a name; word.PL plural; a+b adjective + noun;
x^y x's y (PRON.3sgf^hobby = her hobby); A&B A and B; word>ACT the act of (do>ACT = doing), >ADV -ly, >QUAL -ness,
>AGENT -er; ';' separates clauses."""


def ok(pred: str, gold: dict) -> bool:
    return pred.partition("=>")[0].strip() == gold["analysis"]


def load():
    test = json.loads((ROOT / "results" / "tatoeba_test.json").read_text(encoding="utf-8"))
    r = json.loads((ROOT / "results" / "tatoeba" / f"{REAL}.json").read_text(encoding="utf-8"))["preds"]
    b = json.loads((ROOT / "results" / "tatoeba" / f"{BASE}.json").read_text(encoding="utf-8"))["preds"]
    return test, r, b


def build() -> None:
    test, r, b = load()
    gain = [i for i, g in enumerate(test) if ok(r[i], g) and not ok(b[i], g)]
    rest = [i for i in range(len(test)) if i not in set(gain)]
    rng = random.Random(0)
    items = [(i, "gain") for i in rng.sample(gain, 50)] + [(i, "control") for i in rng.sample(rest, 20)]
    rng.shuffle(items)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "key.json").write_text(json.dumps([{"n": k + 1, "idx": i, "group": g} for k, (i, g) in enumerate(items)],
                                             indent=1), encoding="utf-8")
    rows = "\n".join(
        f"""<div class="item" data-n="{k + 1}"><div class="num">{k + 1} / {len(items)}</div>
<div class="en">{escape(test[i]["english"])}</div><div class="ana">{escape(test[i]["analysis"])}</div>
<div class="opts"><label><input type="radio" name="q{k + 1}" value="right"> right</label>
<label><input type="radio" name="q{k + 1}" value="wrong"> wrong</label>
<label><input type="radio" name="q{k + 1}" value="unsure"> not sure</label>
<input class="note" placeholder="optional: what is wrong"></div></div>""" for k, (i, _) in enumerate(items))
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Tatoeba hand-check</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;background:#fafafa;color:#222}}
.guide{{white-space:pre-wrap;font:12.5px/1.5 Consolas,monospace;background:#eef2f7;padding:12px;border-radius:6px}}
.item{{background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px 12px;margin:10px 0}}
.item.done{{border-color:#7bb07b}} .num{{color:#888;font-size:12px}} .en{{font-size:17px;margin:2px 0 6px}}
.ana{{font:14px Consolas,monospace;background:#f4f4f4;padding:6px;border-radius:4px;word-break:break-word}}
.opts{{margin-top:8px;display:flex;gap:14px;flex-wrap:wrap;align-items:center}} .note{{flex:1;min-width:180px}}
button{{font-size:16px;padding:8px 18px;margin:16px 0}} #count{{margin-left:12px;color:#555}}</style></head><body>
<h2>Tatoeba hand-check ({len(items)} sentences)</h2>
<p>For each English sentence, is the meaning line underneath <b>right</b> (it says what the sentence means, in this
notation)? Judge only the meaning line. Small style choices the notation forces are fine; mark <b>wrong</b> if a
word, tense, negation, question, who-does-what or a missing part is wrong. Use <b>not sure</b> freely.
Your answers are kept in this browser while you work; press <b>Save answers</b> at the end (or any time) and tell
Claude where the file went.</p>
<div class="guide">{escape(GUIDE)}</div>
{rows}
<button id="save">Save answers</button><span id="count"></span>
<script>
const KEY='tatoeba-handcheck-v1';
function collect(){{const a={{}};document.querySelectorAll('.item').forEach(d=>{{const n=d.dataset.n;
 const r=d.querySelector('input[type=radio]:checked');a[n]={{answer:r?r.value:null,note:d.querySelector('.note').value}};
 d.classList.toggle('done',!!r);}});return a;}}
function update(){{const a=collect();try{{localStorage.setItem(KEY,JSON.stringify(a));}}catch(e){{}}
 document.getElementById('count').textContent=Object.values(a).filter(x=>x.answer).length+' of {len(items)} answered';}}
try{{const s=JSON.parse(localStorage.getItem(KEY)||'{{}}');document.querySelectorAll('.item').forEach(d=>{{const x=s[d.dataset.n];
 if(!x)return;if(x.answer){{const r=d.querySelector('input[value='+x.answer+']');if(r)r.checked=true;}}
 d.querySelector('.note').value=x.note||'';}});}}catch(e){{}}
document.addEventListener('change',update);document.addEventListener('input',update);update();
document.getElementById('save').onclick=()=>{{const b=new Blob([JSON.stringify(collect(),null,1)],{{type:'application/json'}});
 const u=URL.createObjectURL(b);const l=document.createElement('a');l.href=u;l.download='tatoeba_handcheck_answers.json';
 l.click();URL.revokeObjectURL(u);}};
</script></body></html>"""
    (OUT / "tatoeba_handcheck.html").write_text(html, encoding="utf-8")
    print("wrote", OUT / "tatoeba_handcheck.html", f"({len(items)} items; key in key.json - do not open while judging)")


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson 95% interval."""
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def ci(k: int, n: int) -> str:
    if not n:
        return "-"
    lo, hi = wilson(k, n)
    return f"{k / n:.2f} [{lo:.2f}, {hi:.2f}]"


def score(path: str) -> None:
    ans = json.loads(Path(path).read_text(encoding="utf-8"))
    key = json.loads((OUT / "key.json").read_text(encoding="utf-8"))
    test, r, b = load()
    md = ["# Tatoeba hand-check (`analysis/handcheck_tatoeba.py`)", "",
          f"Gold meanings judged by the student without knowing the group. gain = real15 right and baseline wrong "
          f"(real15's output is the gold); control = random other sentences.", "",
          "| group | judged | gold right | gold wrong | not sure | share right (95% CI, sure answers) |",
          "|---|---|---|---|---|---|"]
    for grp in ("gain", "control"):
        a = [ans.get(str(k["n"]), {}).get("answer") for k in key if k["group"] == grp]
        rt, wr, un = a.count("right"), a.count("wrong"), a.count("unsure")
        md.append(f"| {grp} | {rt + wr + un} | {rt} | {wr} | {un} | {ci(rt, rt + wr)} |")
    from scipy.stats import fisher_exact
    sure = {g: [ans[str(k["n"])]["answer"] == "right" for k in key if k["group"] == g
                and ans.get(str(k["n"]), {}).get("answer") in ("right", "wrong")] for g in ("gain", "control")}
    n_gain = sum(ok(x, t) and not ok(y, t) for t, x, y in zip(test, r, b))   # all of real15's Tatoeba gains
    ctrl = [k for k in key if k["group"] == "control"]
    easy = sum(ok(r[k["idx"]], test[k["idx"]]) and ok(b[k["idx"]], test[k["idx"]]) for k in ctrl)
    md += ["", f"The control sentences are not typical: {easy} of its {len(ctrl)} sentences are ones both models get "
               "right (the 411 non-gain sentences are mostly easy ones), so it measures the gold on easier sentences."]
    if sure["gain"] and sure["control"]:
        g, c = sure["gain"], sure["control"]
        _, pf = fisher_exact([[sum(g), len(g) - sum(g)], [sum(c), len(c) - sum(c)]])
        same = "no detectable difference" if pf >= 0.05 else "a significant difference"
        md += ["", f"Gain vs control, gold right: {sum(g)}/{len(g)} vs {sum(c)}/{len(c)}, Fisher exact p = {pf:.2f} "
                   f"({same}; with {len(c)} control sentences only a large difference would show)."]
        lo, hi = wilson(sum(g), len(g))
        share = sum(g) / len(g)
        md += ["", f"Estimated real gains: {share:.2f} x {n_gain} = about {share * n_gain:.0f} of real15's {n_gain} "
                   f"Tatoeba gains have a correct gold meaning (95% range {lo * n_gain:.0f}-{hi * n_gain:.0f}); in the "
                   "rest real15 matches a gold judged wrong."]
    notes = [(k["n"], ans[str(k["n"])]["note"]) for k in key if ans.get(str(k["n"]), {}).get("note")]
    if notes:
        md += ["", "## Notes", ""] + [f"- {n}: {t}" for n, t in notes]
    (OUT / "handcheck.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    {"build": lambda: build(), "score": lambda: score(sys.argv[2])}[sys.argv[1]]()
