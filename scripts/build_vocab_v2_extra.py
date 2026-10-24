"""Build the everyday-English additions to the v2 vocabulary (run once): conlang/data/vocab_v2_extra.json.

vocab_v2.json was built from WordNet with filters that left out some everyday words on purpose (every noun ending in
-ing, so "morning" and "evening" were lost with the gerunds; people nouns outside the curated v1 list; the verbs have,
say, tell, get, do, let, become, seem; small adverbs like "here" and "very"). This file adds them from short hand-made
lists, so nothing in vocab_v2.json changes: conlang/v2.py draws the forms of these words after all v2 draws, and every
v2 word keeps its form. The grammar words (this/that, numbers, all/some/many, what/who/where, hello/thanks, because/if,
the case endings) are part of the grammar in conlang/v2.py, not of this file.

    set NLTK_DATA=D:\\Downloads\\fllm-project\\new-language\\.nltk_data
    D:\\Downloads\\fllm-project\\.venv-vocab\\Scripts\\python.exe scripts\\build_vocab_v2_extra.py
"""

from __future__ import annotations

import json
from pathlib import Path

from lemminflect import getInflection

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "conlang" / "data" / "vocab_v2.json"
OUT = ROOT / "conlang" / "data" / "vocab_v2_extra.json"

# nouns ending in -ing that are things, times or events (not "the act of ...", which is verb + ACT: swimming)
ING_NOUNS = ["morning", "evening", "thing", "wedding", "meeting", "feeling", "beginning", "ending", "painting",
             "drawing", "greeting", "clothing", "ceiling", "spring", "offspring", "sibling", "darling", "lightning",
             "meaning", "setting", "gathering", "blessing", "warning", "saying", "king", "ring", "string", "wing",
             "pudding", "evening", "something", "nothing", "everything", "anything"]
# people (gender m / f, or h = person of either gender) and a few beings
PEOPLE = {
    "m": ["dad", "daddy", "dude", "guy", "king", "knight", "lord", "prince", "emperor", "duke", "groom", "grandpa",
          "grandson", "uncle", "nephew", "gentleman", "sir", "wizard", "god", "monk", "husband", "son", "brother",
          "father", "boy", "man", "grandfather", "bachelor"],
    "f": ["mom", "mum", "mama", "mother", "aunt", "niece", "queen", "princess", "bride", "widow", "grandma",
          "grandmother", "granddaughter", "lady", "madam", "goddess", "witch", "maid", "actress", "wife", "daughter",
          "sister", "girl", "woman"],
    "h": ["adult", "boss", "captain", "champion", "chef", "citizen", "client", "coach", "cop", "cousin", "criminal",
          "customer", "employee", "expert", "fellow", "fool", "genius", "guard", "guest", "hero", "host", "idiot",
          "kid", "mate", "mayor", "member", "minister", "owner", "parent", "partner", "passenger", "patient",
          "president", "prisoner", "pupil", "resident", "saint", "secretary", "senator", "specialist", "spouse",
          "student", "teenager", "thief", "tourist", "twin", "victim", "youth", "angel", "pirate", "baby",
          "neighbor", "neighbour", "colleague", "classmate", "roommate", "teammate", "toddler", "infant",
          "celebrity", "athlete", "nurse", "doctor", "police", "friend", "person", "people", "child", "family",
          "someone", "somebody", "everyone", "everybody", "anyone", "anybody", "nobody", "stranger", "enemy",
          "doctor", "lawyer", "farmer", "soldier", "officer", "scientist", "engineer", "artist", "poet",
          "musician", "professor", "principal", "boyfriend", "girlfriend", "grandparent", "grandchild", "human"],
    "n": ["dragon", "monster", "ghost", "fairy", "robot", "alien"],
}
PLURAL_FIX = {"police": None, "people": None, "offspring": None, "clothing": None, "lightning": None,
              "something": None, "nothing": None, "everything": None, "anything": None, "someone": None,
              "somebody": None, "everyone": None, "everybody": None, "anyone": None, "anybody": None, "nobody": None}
# verbs that the v1 builder skipped (auxiliaries and verbs that take a clause); 'be' is already a word
VERBS = {"have": ("has", "had", "had"), "say": ("says", "said", "said"), "tell": ("tells", "told", "told"),
         "get": ("gets", "got", "gotten"), "do": ("does", "did", "done"), "let": ("lets", "let", "let"),
         "become": ("becomes", "became", "become"), "seem": ("seems", "seemed", "seemed"),
         "appear": ("appears", "appeared", "appeared"), "happen": ("happens", "happened", "happened"),
         "thank": ("thanks", "thanked", "thanked"), "miss": ("misses", "missed", "missed"),
         "meet": ("meets", "met", "met"), "welcome": ("welcomes", "welcomed", "welcomed"),
         "rain": ("rains", "rained", "rained"), "snow": ("snows", "snowed", "snowed")}
# small adverbs (time, place, degree, focus, direction particles of phrasal verbs: sit down, go away)
ADVERBS = ["here", "there", "now", "then", "today", "tomorrow", "yesterday", "tonight", "later", "soon", "again",
           "always", "never", "often", "sometimes", "usually", "already", "still", "yet", "just", "only", "also",
           "even", "ever", "ago", "away", "back", "down", "up", "out", "off", "maybe", "perhaps", "probably",
           "almost", "quite", "rather", "very", "too", "so", "much", "else", "instead", "anyway", "once", "twice",
           "forever", "everywhere", "somewhere", "anywhere", "nowhere", "abroad", "ahead", "apart", "aside",
           "inside", "outside", "upstairs", "downstairs", "indeed", "really", "extremely", "less", "least",
           "finally", "suddenly", "together", "right", "exactly", "certainly", "especially", "home", "on", "over", "around", "along"]
ADJECTIVES = ["welcome", "fine", "okay", "alright", "sorry", "glad", "busy", "ready", "sure", "afraid", "alive",
              "awake", "asleep", "able", "hungry", "thirsty", "tired", "sick", "ill", "nice", "good", "bad", "great",
              "favourite", "favorite", "own", "same", "different", "last", "next", "previous", "whole", "full",
              "empty", "open", "closed", "free", "wrong", "right", "true", "false", "real", "interesting",
              "amazing", "boring", "excited", "bored", "married", "lost", "broken", "finished", "done", "late"]


def main() -> None:
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    nouns = {n["lemma"] for n in v2["nouns"]} | set(v2["locations"])
    verbs = {v["base"] for v in v2["transitive"] + v2["intransitive"]}
    adjs = set(v2["adjectives"])
    derived = {d["word"] for d in v2["derived"]} | {c["word"] for c in v2["compounds"]}
    adverbs = set(v2["adverbs"])
    taken_n = nouns | derived
    out = {"nouns": [], "verbs": [], "adjectives": [], "adverbs": []}
    seen = set()
    for g, words in PEOPLE.items():
        for w in words:
            if w in taken_n or w in seen:
                continue
            seen.add(w)
            pl = PLURAL_FIX.get(w, (getInflection(w, "NNS") or [None])[0]) if w in PLURAL_FIX else \
                (getInflection(w, "NNS") or [None])[0]
            out["nouns"].append({"lemma": w, "plural": pl if pl != w else None, "gender": g})
    for w in ING_NOUNS:
        if w in taken_n or w in seen:
            continue
        seen.add(w)
        pl = None if w in PLURAL_FIX else (getInflection(w, "NNS") or [None])[0]
        out["nouns"].append({"lemma": w, "plural": pl if pl != w else None, "gender": "n"})
    for w, (third, past, part) in VERBS.items():
        if w not in verbs:
            out["verbs"].append({"base": w, "third": third, "past": past, "participle": part})
    for w in ADJECTIVES:
        if w not in adjs and w not in derived and w not in [a for a in out["adjectives"]]:
            out["adjectives"].append(w)
    for w in ADVERBS:
        if w not in adverbs and w not in out["adverbs"]:
            out["adverbs"].append(w)
    out["source"] = {"lists": "hand-made (scripts/build_vocab_v2_extra.py)", "plurals": "lemminflect",
                     "counts": {k: len(v) for k, v in out.items() if isinstance(v, list)}}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["source"], indent=1))
    print("nouns:", " ".join(n["lemma"] for n in out["nouns"]))
    print("verbs:", " ".join(v["base"] for v in out["verbs"]))
    print("adjectives:", " ".join(out["adjectives"]))
    print("adverbs:", " ".join(out["adverbs"]))


if __name__ == "__main__":
    main()
