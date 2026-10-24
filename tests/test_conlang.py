import sys
import unicodedata
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conlang import ENGLISH_LIKE, TAMIL_LIKE, Language, dial_variants  # noqa: E402
from conlang.dataset import TEST_SPLITS, build, heldout_pair  # noqa: E402
from conlang.evaluate import chrf, exact_match, word_accuracy  # noqa: E402
from conlang.frames import NP, VOCAB_FILE, Frame, Vocab, english  # noqa: E402
from conlang.grammar import Realization  # noqa: E402
from conlang.script import SCRIPTS, Script  # noqa: E402

SMALL = Vocab.small()


def small_build(cfg, script, n_train=600, n_test=40, seed=0):
    return build(cfg, script, n_train=n_train, n_test=n_test, seed=seed, vocab=SMALL)


# --- English side -------------------------------------------------------------

def test_english_agreement_and_tense():
    v = SMALL
    assert english(Frame(NP("dog", False), "see", NP("cat", True), None, "present", False), v) == "the dog sees the cats"
    assert english(Frame("1sg", "see", NP("man", True, "tall"), "house", "past", True), v) == \
        "I did not see the tall men in the house"
    assert english(Frame(NP("girl", True), "sleep", None, None, "future", False), v) == "the girls will sleep"
    assert english(Frame("3sgf", "catch", NP("fish", True), None, "present", True), v) == \
        "she does not catch the fish"
    assert english(Frame(NP("child", True), "run", None, None, "past", False), v) == "the children ran"


# --- grammar --------------------------------------------------------------------

def test_deterministic():
    assert small_build(TAMIL_LIKE, "abugida_tamil", seed=3)["splits"] == \
        small_build(TAMIL_LIKE, "abugida_tamil", seed=3)["splits"]


def test_word_order():
    f = Frame(NP("dog", False), "see", NP("cat", False), None, "present", False)
    easy = Language(ENGLISH_LIKE, vocab=SMALL).realize(f)
    hard = Language(TAMIL_LIKE, vocab=SMALL).realize(f)
    assert [w.gloss.split("-")[0] for w in easy.words] == ["the", "dog", "see", "the", "cat"]   # SVO
    assert [w.gloss.split("-")[0] for w in hard.words] == ["dog", "cat", "see"]                # SOV
    assert hard.words[1].gloss == "cat-ACC"
    assert hard.words[2].gloss == "see-PRS-AGR.3sgn"


def test_person_noun_agreement():
    f = Frame(NP("teacher", False), "sleep", None, None, "past", False)
    assert Language(TAMIL_LIKE, vocab=SMALL).realize(f).words[-1].gloss.endswith("AGR.3sgh")


def test_tamil_like_suffix_stacking():
    f = Frame("3pl", "find", NP("book", True), "river", "past", True)
    glosses = [w.gloss for w in Language(TAMIL_LIKE, vocab=SMALL).realize(f).words]
    assert "book-PL-ACC" in glosses and "river-LOC" in glosses
    assert glosses[-1].startswith("find") and glosses[-1].endswith("AGR.3pl-NEG")


def test_sandhi_rules():
    lang = Language(replace(TAMIL_LIKE, sandhi=True), vocab=SMALL)
    r = Realization(words=[])
    assert lang._join([("p", "a"), ("e",)], r) == ("p", "a", "v", "e")            # glide after a
    assert lang._join([("m", "i"), ("a", "l")], r) == ("m", "i", "y", "a", "l")    # glide after i
    assert lang._join([("p", "a", "t", "u"), ("e",)], r) == ("p", "a", "t", "e")   # final -u drops
    assert lang._join([("m", "a"), ("k", "u")], r) == ("m", "a", "k", "k", "u")    # doubling
    assert r.sandhi_changes == 4 and r.boundaries == 4


def test_no_sandhi_when_off():
    lang = Language(replace(TAMIL_LIKE, sandhi=False), vocab=SMALL)
    r = Realization(words=[])
    assert lang._join([("p", "a"), ("e",)], r) == ("p", "a", "e") and r.sandhi_changes == 0


# --- scripts --------------------------------------------------------------------

def test_tamil_abugida_rendering():
    s = Script("abugida_tamil", "hard")
    assert s.word(("k", "a")) == "க"
    assert s.word(("k", "aa")) == "கா"
    assert s.word(("k",)) == "க்"
    assert s.word(("a", "m", "m", "a")) == "அம்ம"
    assert s.word(("p", "i", "L", "L", "ee")) == "பிள்ளே"


def test_accent_script_is_precomposed():
    out = Script("latin_accent", "hard").word(("T", "aa", "m", "a"))
    assert out == "ṭāma" and unicodedata.normalize("NFC", out) == out


def test_pua_uses_private_use_area():
    out = Script("abugida_pua", "easy").word(("k", "a", "t"))
    assert all(0xE000 <= ord(ch) <= 0xF8FF for ch in out)


def test_same_language_different_script_same_gloss():
    a, b = small_build(ENGLISH_LIKE, "latin", seed=1), small_build(ENGLISH_LIKE, "abugida_pua", seed=1)
    assert [i["gloss"] for i in a["splits"]["train"]] == [i["gloss"] for i in b["splits"]["train"]]
    assert [i["target"] for i in a["splits"]["train"]] != [i["target"] for i in b["splits"]["train"]]


def _no_collisions(data):
    seen = {}
    for items in data["splits"].values():
        for it in items:
            prev = seen.setdefault(it["target"], it["english"])
            assert prev == it["english"], (prev, it["english"], it["target"])


def test_no_collisions_small_vocab_every_variant_and_script():
    for cfg in dial_variants().values():
        for script in SCRIPTS:
            _no_collisions(small_build(cfg, script))


# --- splits ---------------------------------------------------------------------

def test_splits_hold_out_what_they_claim():
    data = small_build(TAMIL_LIKE, "latin", n_train=1500, n_test=60)
    for it in data["splits"]["train"] + data["splits"]["test_iid"]:
        f = it["frame"]
        if f["obj"]:
            assert not heldout_pair(0, f["verb"], f["obj"]["noun"])
        plural = (f["subj"] in ("1pl", "2pl", "3pl")) if isinstance(f["subj"], str) else f["subj"]["plural"]
        assert not (f["tense"] == "past" and f["neg"] and plural)
        assert not (f["loc"] and f["obj"] and isinstance(f["subj"], dict) and f["subj"]["adj"] and f["obj"]["adj"])
    for it in data["splits"]["test_long"]:
        f = it["frame"]
        assert f["loc"] and f["subj"]["adj"] and f["obj"]["adj"]
    train_words = {w for it in data["splits"]["train"] for w in it["content"]}
    for k in TEST_SPLITS:
        assert len(data["splits"][k]) > 0
        assert all(w in train_words for it in data["splits"][k] for w in it["content"])
    eng = [it["english"] for items in data["splits"].values() for it in items]
    assert len(eng) == len(set(eng))


# --- large vocabulary ------------------------------------------------------------

@pytest.mark.skipif(not VOCAB_FILE.exists(), reason="run scripts/build_vocab.py first")
def test_large_vocab_size_and_forms():
    v = Vocab.large()
    assert v.size() >= 2000
    assert len(set(v.concepts())) == len(v.concepts())
    for lemma in v.nouns:
        assert v.noun_info[lemma][0]
    for base in v.transitive + v.intransitive:
        assert all(v.verb_info[base])


@pytest.mark.skipif(not VOCAB_FILE.exists(), reason="run scripts/build_vocab.py first")
def test_large_vocab_no_collisions():
    for cfg in (ENGLISH_LIKE, TAMIL_LIKE):
        for script in SCRIPTS:
            _no_collisions(build(cfg, script, n_train=8000, n_test=100, seed=0))


# --- metrics ---------------------------------------------------------------------

def test_metrics():
    refs = ["ab cd", "ef gh"]
    assert exact_match(refs, refs) == 1.0
    assert abs(chrf(refs, refs) - 100) < 1e-9
    assert exact_match(["ab cd", "xx"], refs) == 0.5
    assert word_accuracy(["ab zz", "ef gh"], refs) == 0.75
    assert chrf(["zzzz", "yyyy"], refs) == 0.0
