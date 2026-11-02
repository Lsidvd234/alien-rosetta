"""Build the results report (LaTeX -> PDF) from the result files themselves.

Every table in the report is converted automatically from results/*.md and results/stress/test*.md;
nothing is retyped. Hand-written text (setup, inferences) lives in inferences.py.

    python build_report.py          # writes report.tex next to this file
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "new-language"
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import inferences as INF  # noqa: E402

CHARMAP = {"→": r"$\rightarrow$", "->": r"$\rightarrow$", "×": r"$\times$", "≈": r"$\approx$", "–": "--", "—": "---",
           "≤": r"$\le$", "≥": r"$\ge$", "’": "'", "‘": "`", "“": "``", "”": "''", "…": r"\ldots{}", "·": r"$\cdot$",
           "±": r"$\pm$", "↔": r"$\leftrightarrow$", "✅": "[ok]", "⚠️": "[weak]", "❌": "[fails]", " ": "~"}


def esc(s: str) -> str:
    """Escape plain text for LaTeX; non-ASCII that the font cannot show becomes <U+XXXX>."""
    out = []
    i = 0
    while i < len(s):
        if s.startswith("->", i):
            out.append(CHARMAP["->"]); i += 2; continue
        if s.startswith("⚠️", i):
            out.append(CHARMAP["⚠️"]); i += 2; continue
        ch = s[i]
        if ch in CHARMAP:
            out.append(CHARMAP[ch])
        elif ch in "\\":
            out.append(r"\textbackslash{}")
        elif ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\^{}")
        elif ord(ch) < 128:
            out.append(ch)
        elif unicodedata.category(ch).startswith("L") and ord(ch) < 0x250:
            out.append(ch)                                   # accented Latin letters render fine
        else:
            out.append(r"\texttt{<U+%04X>}" % ord(ch))
        i += 1
    return "".join(out)


def inline(s: str) -> str:
    """Markdown inline -> LaTeX: **bold**, `code`."""
    parts = re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", s)
    out = []
    for p in parts:
        if p.startswith("**") and p.endswith("**"):
            out.append(r"\textbf{" + esc(p[2:-2]) + "}")
        elif p.startswith("`") and p.endswith("`"):
            out.append(r"\texttt{" + esc(p[1:-1]) + "}")
        else:
            out.append(esc(p))
    return "".join(out)


def table(rows: list[list[str]], widths: list[float] | None = None) -> str:
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    first = 0.26 if n > 2 else 0.4
    rest = (0.97 - first) / max(1, n - 1)
    widths = widths or [first + 0.012] + [rest] * (n - 1)
    spec = "".join(f">{{\\raggedright\\arraybackslash}}p{{{w - 0.012:.3f}\\linewidth}}" for w in widths)
    size = r"\scriptsize" if n >= 5 else r"\footnotesize"
    lines = [r"{" + size, r"\begin{longtable}{" + spec + "}", r"\toprule",
             " & ".join(r"\textbf{" + inline(c) + "}" for c in rows[0]) + r" \\", r"\midrule", r"\endhead"]
    for r in rows[1:]:
        lines.append(" & ".join(inline(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{longtable}}", ""]
    return "\n".join(lines)


def md_to_tex(md: str, top: int = 1) -> str:
    """Convert one results markdown file. `top`: the level of '#' -> \\subsection (first # is dropped)."""
    out, tbl, para = [], [], []

    def flush_para():
        if para:
            out.append(inline(" ".join(para)) + "\n")
            para.clear()

    def flush_tbl():
        if tbl:
            out.append(table(tbl))
            tbl.clear()
    for line in md.splitlines():
        s = line.rstrip()
        if s.startswith("|"):
            flush_para()
            cells = [c.strip() for c in s.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            tbl.append(cells)
            continue
        flush_tbl()
        if not s.strip():
            flush_para()
            continue
        m = re.match(r"^(#+)\s+(.*)", s)
        if m:
            flush_para()
            lvl = len(m.group(1))
            if lvl == 1:
                continue                                      # the file title is replaced by our own heading
            cmd = r"\subsubsection*" if lvl == 2 else r"\paragraph*"
            out.append(cmd + "{" + inline(m.group(2)) + "}\n")
            continue
        if re.match(r"^\s*[-*]\s+", s):
            flush_para()
            out.append(r"\begin{itemize}\item " + inline(re.sub(r"^\s*[-*]\s+", "", s)) + r"\end{itemize}" + "\n")
            continue
        if re.match(r"^\d+\.\s+", s):
            flush_para()
            out.append(r"\begin{itemize}\item " + inline(s) + r"\end{itemize}" + "\n")
            continue
        para.append(s.strip())
    flush_para()
    flush_tbl()
    return "\n".join(out)


def split_examples() -> str:
    from conlang.dataset import PRESETS, TEST_SPLITS, build
    d = build(PRESETS["tamil_like"], "latin", n_train=50000, n_test=500, seed=0)
    rows = [["split", "items", "English (input)", "gloss of the correct output", "output, Latin spelling of the same sounds"]]
    for k in ["train"] + list(TEST_SPLITS):
        for it in d["splits"][k][:2]:
            rows.append([k, str(len(d["splits"][k])), it["english"], it["gloss"], it["target"]])
    return table(rows, [0.10, 0.07, 0.28, 0.28, 0.24])


def stress_files() -> list[tuple[int, Path]]:
    out = []
    for p in (RES / "stress").glob("test*.md"):
        m = re.match(r"test(\d+)_", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def main() -> None:
    T = []
    T.append(r"""\documentclass[10pt,a4paper,oneside]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{fontspec}
\usepackage{amsmath,booktabs,longtable,array,enumitem,xcolor,graphicx}
\usepackage[most]{tcolorbox}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\setlist{nosep,leftmargin=*}
\setlength{\parskip}{0.45em}\setlength{\parindent}{0pt}
\setlength{\LTpre}{2pt}\setlength{\LTpost}{6pt}
\renewcommand{\arraystretch}{1.12}
\newtcolorbox{infer}{colback=blue!4,colframe=blue!45!black,boxrule=0.5pt,arc=2pt,left=5pt,right=5pt,
  fonttitle=\bfseries\small,title=Inference,breakable}
\newtcolorbox{caveat}{colback=orange!6,colframe=orange!70!black,boxrule=0.5pt,arc=2pt,left=5pt,right=5pt,
  fonttitle=\bfseries\small,title=Caveat,breakable}
\title{\textbf{Alien Rosetta: results report}\\[0.3em]\large Learning an invented language in invented glyphs with Qwen3-0.6B + LoRA\\[0.2em]
\normalsize Main experiments, test splits, and stress tests 1--30}
\author{AI60213 Foundations of LLMs, course project}
\date{12 October 2026}
\begin{document}
\maketitle
\tableofcontents
\newpage
""")
    T.append(INF.PREFACE)
    T.append(r"\section{Setup}" + "\n" + INF.SETUP)
    T.append(r"\subsection{The four test splits, with real examples}" + "\n" + INF.SPLITS_INTRO)
    T.append(split_examples())
    T.append(INF.SPLITS_AFTER)

    T.append(r"\section{Main results}" + "\n" + INF.MAIN_INTRO)
    T.append(r"\subsection{Learning curves: grammar $\times$ script}" + "\n" +
             r"\textit{Source: \texttt{results/curve\_summary.md}.}" + "\n\n")
    T.append(md_to_tex((RES / "curve_summary.md").read_text(encoding="utf-8")))
    T.append(r"\begin{infer}" + INF.CURVES + r"\end{infer}")
    T.append(r"\subsection{Final tables: seeds, cause tests, invented glyphs, fixes}" + "\n" +
             r"\textit{Source: \texttt{results/final\_20k.md}.}" + "\n\n")
    T.append(md_to_tex((RES / "final_20k.md").read_text(encoding="utf-8")))
    T.append(r"\begin{infer}" + INF.FINAL + r"\end{infer}")
    for fig, cap in INF.FIGURES:
        p = RES / "figures" / fig
        if p.exists():
            T.append(r"\begin{center}\includegraphics[width=0.82\linewidth]{" + p.as_posix() + r"}\\[-0.2em]"
                     r"\small " + cap + r"\end{center}")

    T.append(r"\section{Stress tests on the saved 50k models}" + "\n" + INF.STRESS_INTRO)
    groups = INF.GROUPS
    files = dict(stress_files())
    for title, nums in groups:
        T.append(r"\subsection{" + title + "}")
        for n in nums:
            if n not in files:
                T.append(r"\subsubsection{Test " + str(n) + r" (no result file)}")
                continue
            p = files[n]
            md = p.read_text(encoding="utf-8")
            head = re.match(r"#\s+(.*)", md)
            name = head.group(1) if head else p.stem
            T.append(r"\subsubsection{" + inline(name) + "}")
            T.append(r"\textit{Source: \texttt{results/stress/" + esc(p.name) + r"}.}" + "\n\n")
            if n in INF.CAVEATS:
                T.append(r"\begin{caveat}" + INF.CAVEATS[n] + r"\end{caveat}")
            T.append(md_to_tex(md))
            if n in INF.TEST:
                T.append(r"\begin{infer}" + INF.TEST[n] + r"\end{infer}")
    T.append(r"\section{Retraining: R1 and E1}" + "\n" + INF.RETRAIN_INTRO)
    for title, fname, cav, inf in INF.RETRAIN_PARTS:
        T.append(r"\subsection{" + title + "}")
        T.append(r"\textit{Source: \texttt{results/" + esc(fname) + r"}.}" + "\n\n")
        if cav:
            T.append(r"\begin{caveat}" + cav + r"\end{caveat}")
        T.append(md_to_tex((RES / fname).read_text(encoding="utf-8")))
        T.append(r"\begin{infer}" + inf + r"\end{infer}")
    T.append(r"\section{Overall inference}" + "\n" + INF.OVERALL)
    T.append(r"\section{Caveats and limits}" + "\n" + INF.LIMITS)
    T.append(r"\appendix\section{Where every number comes from}" + "\n{\\sloppy\\raggedright\n" + INF.PROVENANCE + "}\n")
    T.append(r"\end{document}")
    (HERE / "report.tex").write_text("\n".join(T), encoding="utf-8")
    print("wrote", HERE / "report.tex")


if __name__ == "__main__":
    main()
