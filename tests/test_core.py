import pytest

from hoerbuchgenerator.naming import make_slug
from hoerbuchgenerator.text import chunk_text


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Jürgen Müller", "juergen_mueller"),
        ("Änne Öztürk", "aenne_oeztuerk"),
        ("Straße / Nord", "strasse_nord"),
        ("  Mehr  als... ein Name  ", "mehr_als..._ein_name"),
    ],
)
def test_make_slug_transliterates_and_sanitizes_names(name, expected):
    assert make_slug(name) == expected


@pytest.mark.parametrize("name", ["", "...", "CON", "lpt1.txt"])
def test_make_slug_rejects_empty_and_reserved_names(name):
    with pytest.raises(ValueError):
        make_slug(name)


def test_chunk_text_keeps_sentence_boundaries_and_limits_chunk_size():
    chunks = chunk_text("Erster Satz. Zweiter Satz! Dritter Satz?", max_chars=22)

    assert chunks == ["Erster Satz.", "Zweiter Satz!", "Dritter Satz?"]
    assert all(len(chunk) <= 22 for chunk in chunks)


def test_chunk_text_rejects_empty_input_and_merges_short_chunks():
    with pytest.raises(ValueError):
        chunk_text(" \n ")

    assert chunk_text("Hi. Das ist ein laengerer Satz.") == ["Hi. Das ist ein laengerer Satz."]


def test_chunk_text_splits_words_when_a_single_word_exceeds_limit():
    chunks = chunk_text("abcdefghijk", max_chars=4, min_chars=0)

    assert chunks == ["abcd", "efgh", "ijk"]