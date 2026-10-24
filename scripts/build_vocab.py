"""Build the large English vocabulary file (run once; the generator itself needs no extra packages).

Sources: WordNet (part of speech, verb sentence patterns, semantic class),
wordfreq (keep common words; drop nouns rarely used in the plural),
lemminflect (correct plurals, -s forms and past tenses, incl. irregulars).

People-nouns come from a curated list (WordNet's person category contains slurs).

Run with the vocab venv:
    set NLTK_DATA=D:\\Downloads\\fllm-project\\new-language\\.nltk_data
    D:\\Downloads\\fllm-project\\.venv-vocab\\Scripts\\python.exe scripts\\build_vocab.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from lemminflect import getInflection
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

OUT = Path(__file__).resolve().parents[1] / "conlang" / "data" / "vocab_en.json"

PEOPLE = {  # word -> gender (m / f / h = person, either gender)
    "man": "m", "boy": "m", "king": "m", "father": "m", "brother": "m", "son": "m", "husband": "m",
    "uncle": "m", "prince": "m", "grandfather": "m", "nephew": "m", "gentleman": "m",
    "woman": "f", "girl": "f", "queen": "f", "mother": "f", "sister": "f", "daughter": "f", "wife": "f",
    "aunt": "f", "princess": "f", "grandmother": "f", "niece": "f", "lady": "f",
    "teacher": "h", "doctor": "h", "farmer": "h", "student": "h", "child": "h", "friend": "h",
    "baby": "h", "nurse": "h", "driver": "h", "worker": "h", "singer": "h", "writer": "h",
    "artist": "h", "soldier": "h", "guest": "h", "neighbor": "h", "player": "h", "leader": "h",
    "cook": "h", "pilot": "h", "baker": "h", "dancer": "h", "fisherman": "h", "hunter": "h",
    "judge": "h", "lawyer": "h", "merchant": "h", "painter": "h", "poet": "h", "sailor": "h",
    "scientist": "h", "shepherd": "h", "stranger": "h", "tailor": "h", "traveler": "h",
    "villager": "h", "visitor": "h", "waiter": "h", "engineer": "h", "priest": "h", "officer": "h",
}
NOUN_CLASSES = {"noun.animal", "noun.artifact", "noun.food", "noun.plant", "noun.object", "noun.body"}
# a location is a common noun whose meaning falls under one of these (no city/country names)
LOC_HYPERNYMS = {"building.n.01", "room.n.01", "body_of_water.n.01", "geological_formation.n.01",
                 "facility.n.01", "tract.n.01", "way.n.06", "structure.n.01"}
BLOCK = {"ass", "bitch", "dick", "cock", "pussy", "slut", "whore", "fag", "nigger", "cunt", "bastard",
         "tit", "boob", "penis", "vagina", "rape", "fuck", "shit", "piss", "crap", "damn", "sex",
         "kill", "murder", "stab", "shoot", "slaughter", "molest", "seduce", "screw", "hump", "drug"}
NUMBERS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
           "twelve", "twenty", "thirty", "forty", "fifty", "hundred", "thousand", "million", "billion"}
EXTRA_LOCATIONS = ["house", "river", "forest", "market", "garden", "field", "school", "village",
                   "kitchen", "temple", "station", "office", "hospital", "library", "shop", "lake",
                   "park", "street", "room", "hall", "castle", "palace", "harbor", "farm", "yard"]
SKIP_VERBS = {"be", "have", "do", "will", "shall", "may", "might", "can", "could", "would", "should",
              "must", "get", "let", "seem", "become", "appear", "say", "tell"}
WORD = re.compile(r"^[a-z]{3,12}$")
TRANSITIVE_FRAMES = {8, 9, 10, 11}   # Somebody ----s something / somebody / ...
INTRANSITIVE_FRAMES = {1, 2}         # Something ----s / Somebody ----s


def pos_counts(word: str) -> dict:
    c = defaultdict(int)
    for syn in wn.synsets(word):
        for lem in syn.lemmas():
            if lem.name().lower() == word:
                c[syn.pos().replace("s", "a")] += lem.count() + 1
    return c


def dominant(word: str, pos: str) -> bool:
    c = pos_counts(word)
    return bool(c) and max(c, key=c.get) == pos


def main_synset(word: str, pos: str):
    syns = [s for s in wn.synsets(word, pos=pos)]
    return syns[0] if syns else None


def candidates(pos: str) -> list[str]:
    # only lemmas written in lowercase in WordNet: drops proper names (London, Jordan, ...)
    words = {l.name() for s in wn.all_synsets(pos) for l in s.lemmas()}
    return sorted(w for w in words if WORD.match(w) and w not in BLOCK)


def is_location(syn) -> bool:
    closure = {s.name() for s in syn.closure(lambda s: s.hypernyms())}
    return bool(closure & LOC_HYPERNYMS)


def common(w: str, z: float) -> bool:
    return zipf_frequency(w, "en") >= z


def build() -> dict:
    nouns, locations = [], []
    for w, g in PEOPLE.items():
        nouns.append({"lemma": w, "plural": getInflection(w, "NNS")[0], "gender": g})
    for w in candidates("n"):
        if w in PEOPLE or not common(w, 3.0) or not dominant(w, "n"):
            continue
        syn = main_synset(w, "n")
        if syn is None:
            continue
        if w in EXTRA_LOCATIONS or (is_location(syn) and common(w, 3.3)):
            locations.append(w)
            continue
        if syn.lexname() not in NOUN_CLASSES:
            continue
        plural = (getInflection(w, "NNS") or [None])[0]
        if not plural or plural == w and w not in ("fish", "sheep", "deer"):
            continue
        if zipf_frequency(plural, "en") < 2.0:   # rarely pluralised: likely a mass noun
            continue
        nouns.append({"lemma": w, "plural": plural, "gender": "n"})
    for w in EXTRA_LOCATIONS:
        if w not in locations:
            locations.append(w)

    transitive, intransitive = [], []
    for w in candidates("v"):
        if w in SKIP_VERBS or not common(w, 3.3) or not dominant(w, "v"):
            continue
        syn = main_synset(w, "v")
        if syn is None:
            continue
        lem = next((l for l in syn.lemmas() if l.name().lower() == w), None)
        frames = set(lem.frame_ids()) if lem else set()
        forms = {"base": w, "third": (getInflection(w, "VBZ") or [None])[0],
                 "past": (getInflection(w, "VBD") or [None])[0]}
        if not forms["third"] or not forms["past"]:
            continue
        if frames & TRANSITIVE_FRAMES:
            transitive.append(forms)
        elif frames & INTRANSITIVE_FRAMES:
            intransitive.append(forms)

    adjectives = [w for w in candidates("a")
                  if common(w, 3.6) and dominant(w, "a") and not w.endswith(("ed", "ing"))
                  and w not in NUMBERS]

    vocab = {"nouns": nouns, "locations": sorted(set(locations)), "transitive": transitive,
             "intransitive": intransitive, "adjectives": adjectives}
    counts = {k: len(v) for k, v in vocab.items()}
    vocab["source"] = {"wordnet": "nltk WordNet 3.0", "frequency": "wordfreq zipf",
                       "inflection": "lemminflect", "counts": counts,
                       "total_content_words": sum(counts.values())}
    return vocab


if __name__ == "__main__":
    v = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(v, indent=1), encoding="utf-8")
    print(json.dumps(v["source"], indent=1))
