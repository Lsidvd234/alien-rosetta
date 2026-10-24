"""Procedural language generator for the new-language tokenization study.

Two families:
- easy  (English-like): SVO, articles, few endings, Latin script.
- hard  (Tamil-like):   SOV, case/tense/agreement suffixes, sound changes at
                        morpheme boundaries, long vowels, abugida script.

Every grammar setting ("dial") and the script can be changed independently,
so effects can be measured one at a time on the easy language and then used
to predict the hard one.
"""

from .grammar import ENGLISH_LIKE, TAMIL_LIKE, GrammarConfig, Language, dial_variants
from .script import SCRIPTS, render_sentence

__all__ = [
    "ENGLISH_LIKE",
    "TAMIL_LIKE",
    "GrammarConfig",
    "Language",
    "dial_variants",
    "SCRIPTS",
    "render_sentence",
]
