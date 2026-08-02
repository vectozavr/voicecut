from __future__ import annotations

import pytest

from voicecut.common import integer_to_words, tokenize


def test_english_tokenization_remains_backward_compatible() -> None:
    assert tokenize("Café can't 21%") == [
        "cafe",
        "can",
        "not",
        "twenty",
        "one",
        "percent",
    ]


def test_russian_tokenization_preserves_cyrillic_yo_and_short_i() -> None:
    assert tokenize("Ёлка ещё и йод", language="ru") == [
        "ёлка",
        "ещё",
        "и",
        "йод",
    ]


def test_russian_numeric_tokenization_is_deterministic_and_conservative() -> None:
    assert tokenize("21% и 3.5", language="ru") == [
        "двадцать",
        "один",
        "процент",
        "и",
        "три",
        "точка",
        "пять",
    ]
    assert integer_to_words(21_021, language="ru") == [
        "двадцать",
        "одна",
        "тысяча",
        "двадцать",
        "один",
    ]


def test_tokenization_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="unsupported tokenization language"):
        tokenize("texte", language="fr")
