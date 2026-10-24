"""Two-stage translation: a frozen English model rewrites hard English into the language's own words, then our trained
translator (unchanged) writes the glyphs.

    rewrite(sentences)  -> per sentence: the text the translator gets, and what happened (kept / rewritten / fallback)

Stage 1, only for sentences the sealed-set reader cannot read already (sentences fully inside the language go to the
translator unchanged, so a rewrite can never spoil them):
  1. retrieval: every word that is not a word of the language gets up to 8 replacement candidates
     (conlang/data/synonym_candidates.json, from WordNet; scripts/build_synonym_candidates.py)
  2. a frozen instruction-tuned model (Qwen3-0.6B / 1.7B / 4B, never trained) rewrites the sentence in simple English,
     using the candidates, keeping names, negation, numbers and questions
  3. checks: every word must be a word of the language (the reader); names, negation, numbers and a question mark must
     survive; up to 2 retries that name the problem; if it still fails, the ORIGINAL sentence is used, so the pipeline
     is never worse than the translator alone.
Stage 2 (two_stage/evaluate.py): the trained translator translates the result.
"""

from __future__ import annotations

import gc
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sealed" / "app"))

from english_parse import FUNCTION, Parser, spell_name, tokenize  # noqa: E402
import train.represent_v2 as R2  # noqa: E402

CANDIDATES = json.loads((ROOT / "conlang" / "data" / "synonym_candidates.json").read_text(encoding="utf-8"))
NEG = {"not", "no", "never", "nothing", "nobody", "none", "nowhere", "neither", "nor"}
HELPERS = {"will", "shall", "can", "could", "may", "might", "must", "should", "would", "did", "does", "do",
           "have", "has", "had"}
NUM = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "twenty",
       "thirty", "forty", "fifty", "hundred", "thousand", "million"}
SYSTEM = ("You rewrite English sentences into simple English for a translator that knows a limited vocabulary.\n"
          "Rules:\n"
          "- Replace each word listed under 'Replace' with one of its allowed replacements, changing its form "
          "(plural, past tense, -ing) to fit the sentence.\n"
          "- Keep every other word, every name, every number, every 'not' and the question mark exactly as they are.\n"
          "- Keep the meaning: do not add or drop information.\n"
          "- Answer with the rewritten sentence only, on one line.")
FEW_SHOT = [  # written by hand for the pipeline (not from any test set)
    ("Replace: enormous -> big, large, huge | hound -> dog, wolf\nSentence: The enormous hound did not bark at Mira.",
     "The big dog did not bark at Mira."),
    ("Replace: purchased -> buy, get | automobile -> car\nSentence: Did Kavi purchase two automobiles yesterday?",
     "Did Kavi buy two cars yesterday?"),
    ("Replace: residence -> house, home | sprinted -> run, race\nSentence: They sprinted to their residence.",
     "They ran to their house."),
]


@dataclass
class Rewrite:
    original: str
    text: str                                    # what the translator gets
    action: str                                  # kept | substituted | rewritten | fallback
    unknown: list = field(default_factory=list)  # words not in the language (before)
    tries: list = field(default_factory=list)    # every model rewrite tried, with the reason it was rejected
    subs: list = field(default_factory=list)     # synonyms replaced directly (purchased->bought)
    original_for_llm: str = ""                   # the sentence after direct synonym replacement


def base_forms(w: str) -> list[str]:
    """Simple de-inflection to find a word's entry in the candidate table (hounds -> hound, purchased -> purchase)."""
    out = [w]
    if w.endswith("ies"):
        out.append(w[:-3] + "y")
    if w.endswith("es"):
        out.append(w[:-2])
    if w.endswith("s"):
        out.append(w[:-1])
    if w.endswith("ied"):
        out.append(w[:-3] + "y")
    if w.endswith("ed"):
        out += [w[:-2], w[:-1], w[:-3] if len(w) > 4 and w[-3] == w[-4] else w[:-2]]
    if w.endswith("ing"):
        out += [w[:-3], w[:-3] + "e", w[:-4] if len(w) > 5 and w[-4] == w[-5] else w[:-3]]
    return list(dict.fromkeys(out))


def candidates_for(w: str) -> list[str]:
    for b in base_forms(w):
        if b in CANDIDATES:
            return [c for c, _, _ in CANDIDATES[b]]
    return []


class Rewriter:
    def __init__(self, model: str = "Qwen/Qwen3-0.6B", device: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.parser = Parser()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device).eval()
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype).to(self.device).eval()

    def close(self) -> None:
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    # --- what the reader thinks of a sentence ---------------------------------------------------------------------
    def readable(self, s: str) -> bool:
        r = self.parser.parse(s)
        return bool(r.clauses) and not r.issues

    def unknown(self, s: str) -> list[str]:
        return list(self.parser.parse(s).unknown)

    @staticmethod
    def names(s: str) -> set[str]:
        toks = [t for piece, _ in tokenize(s) for t in piece]
        return {t.orig for i, t in enumerate(toks) if t.orig[:1].isupper() and i > 0 and spell_name(t.orig)[0]
                and t.low not in ("i",)}

    @staticmethod
    def marks(s: str) -> dict:
        words = [t.low for piece, _ in tokenize(s) for t in piece]          # contractions opened: can't -> can not
        return {"neg": any(w in NEG for w in words),
                "num": sorted(w for w in words if w in NUM) + sorted(re.findall(r"\d+", s)),
                "helpers": sorted(w for w in words if w in HELPERS),
                "q": s.strip().endswith("?")}

    def check(self, original: str, rewrite: str) -> str | None:
        """None if the rewrite is acceptable, else the reason."""
        if not rewrite or "\n" in rewrite.strip():
            return "empty or several lines"
        unk = self.unknown(rewrite)
        if unk:
            return "not words of the language: " + ", ".join(unk)
        lost = self.names(original) - set(re.findall(r"[A-Za-z]+", rewrite))
        if lost:
            return "names changed: " + ", ".join(sorted(lost))
        a, b = self.marks(original), self.marks(rewrite)
        if a["neg"] != b["neg"]:
            return "negation changed"
        if a["num"] != b["num"]:
            return "numbers changed"
        if a["helpers"] != b["helpers"]:
            return "helper verbs changed (keep: " + (", ".join(a["helpers"]) or "none") + ")"
        if a["q"] != b["q"]:
            return "question mark changed"
        return None

    # --- the frozen model ------------------------------------------------------------------------------------------
    def prompt(self, sentence: str, unknown: list[str], feedback: str | None) -> str:
        repl = " | ".join(f"{w} -> {', '.join(candidates_for(w.lower())[:6]) or '(describe it with simple words)'}"
                          for w in unknown)
        user = f"Replace: {repl or '(none; only simplify the sentence)'}\nSentence: {sentence}"
        if feedback:
            user += f"\nYour last answer was rejected ({feedback}). Try again."
        msgs = [{"role": "system", "content": SYSTEM}]
        for u, a in FEW_SHOT:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": user})
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate(self, prompts: list[str]) -> list[str]:
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=self.tok.eos_token_id)
        texts = self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        clean = []
        for t in texts:
            t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
            clean.append(t.splitlines()[0].strip().strip('"') if t else "")
        return clean

    def substitute(self, s: str) -> tuple[str, list[str]]:
        """Replace every word the reader reads as a synonym (read 'purchased' as 'buy') with the language's own word,
        in the same form (bought). No model needed: the reader already knows the right word."""
        lex, v = self.parser.lex, self.parser.lex.v
        done = []
        own = self.own_words()

        def form(w: str) -> str | None:
            lemma = lex.syn_used.get(w)
            if not lemma or w in own:                     # a word of the language means itself: leave it
                return None
            if w in lex.verb or w in lex.part or w in lex.ing:
                if lemma in v.verb_info:
                    base, third, past = v.verb_info[lemma]
                    kinds = {k for _, k in lex.verb.get(w, [])}
                    if w in lex.ing and not kinds:
                        return R2.ing_of(base)
                    if w in lex.part and not kinds:
                        return v.participle.get(lemma) or past
                    return {"third": third, "past": past}.get(next(iter(kinds), "base"), base)
            if w in lex.noun:
                pl = any(p for _, p in lex.noun[w])
                info = v.noun_info.get(lemma)
                return (info[0] if (pl and info and info[0]) else lemma)
            return lemma

        def repl(m):
            w = m.group(0)
            new = form(w.lower())
            if not new or w.lower() in FUNCTION:
                return w
            done.append(f"{w}->{new}")
            return new.capitalize() if w[:1].isupper() else new
        return re.sub(r"[A-Za-z]+", repl, s), done

    def own_words(self) -> set[str]:
        """Every English word form that is a word of the language itself (not only a synonym of one)."""
        if getattr(self, "_own", None) is None:
            v = self.parser.lex.v
            own = set(v.locations) | set(v.adjectives) | set(v.x_adjectives) | set(v.adverbs) | set(v.x_adverbs)
            for n, (pl, _) in v.noun_info.items():
                own |= {n} | ({pl} if pl else set())
            for b, t, pa in v.verb_info.values():
                own |= {b, t, pa, R2.ing_of(b)}
            for (w, _), d in v.derived.items():
                own |= {w} | {d.get(k) for k in ("plural", "third", "past") if d.get(k)}
            for w, c in v.compounds.items():
                own |= {w} | ({c["plural"]} if c.get("plural") else set())
            self._own = own
        return self._own

    @staticmethod
    def mostly_foreign(s: str, unknown: list[str]) -> bool:
        """Not English (French, Hindi, emoji ...): leave it alone so the translator can refuse it."""
        words = re.findall(r"\w+", s)
        if not words:
            return True
        non_latin = sum(not re.fullmatch(r"[A-Za-z']+", w) for w in words)
        return (non_latin + len(unknown)) / len(words) > 0.5

    def rewrite(self, sentences: list[str], retries: int = 2, batch: int = 8) -> list[Rewrite]:
        res = []
        todo = []
        for s in sentences:
            r = Rewrite(original=s, text=s, action="kept")
            sub, done = self.substitute(s)
            r.original_for_llm = sub
            if done:
                r.text, r.action, r.subs = sub, "substituted", done
                s = sub
            if not self.readable(s):
                r.unknown = self.unknown(s)
                if self.mostly_foreign(s, r.unknown):
                    r.action = "foreign"
                else:
                    todo.append(r)
            res.append(r)
        pending = list(todo)
        for attempt in range(retries + 1):
            if not pending:
                break
            nxt = []
            for i in range(0, len(pending), batch):
                chunk = pending[i:i + batch]
                prompts = [self.prompt(r.original_for_llm, r.unknown, r.tries[-1][1] if r.tries else None)
                           for r in chunk]
                for r, out in zip(chunk, self.generate(prompts)):
                    why = self.check(r.original_for_llm, out)
                    r.tries.append((out, why))
                    if why is None:
                        r.text, r.action = out, "rewritten"
                    else:
                        nxt.append(r)
            pending = nxt
        for r in pending:
            r.text, r.action = r.original_for_llm, "fallback"         # (synonyms already replaced stay replaced)
        return res
