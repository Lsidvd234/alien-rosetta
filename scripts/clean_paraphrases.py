"""Clean a paraphrase file with the same rules build_paraphrases_v2.py now applies while generating.

    python scripts/clean_paraphrases.py IN.json OUT.json

1. A stray comma / semicolon / colon before the end mark is removed ("Simultaneously,!" -> "Simultaneously!").
2. If the original starts with a question word, the paraphrase must too (or start with a fronted phrase + comma);
   otherwise it is dropped ("what did Noci swing" -> "Noci swung what": echo questions read as odd English).
"""

import json
import re
import sys

WH = r"(how|what|who|whom|whose|where|when|which|why)\b"


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    p = json.load(open(src, encoding="utf-8"))
    out, fixed, dropped = {}, 0, 0
    for k, v in p.items():
        if re.match(r"(?i)\s*" + WH, k) and not re.match(r"(?i)\s*(" + WH + r"|[^,]*,)", v):
            dropped += 1
            continue
        w = re.sub(r"[,;:]+(\s*[.!?]*\s*)$", r"\1", v)
        fixed += w != v
        out[k] = w
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(p)} in, {dropped} dropped (question word moved), {fixed} punctuation fixed, {len(out)} out")


if __name__ == "__main__":
    main()
