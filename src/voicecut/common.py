#!/usr/bin/env python3
"""Shared data models and text helpers for the narration-cleaning pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
    "13": "thirteen",
    "14": "fourteen",
    "15": "fifteen",
    "16": "sixteen",
    "17": "seventeen",
    "18": "eighteen",
    "19": "nineteen",
    "20": "twenty",
    "30": "thirty",
    "40": "forty",
    "50": "fifty",
    "60": "sixty",
    "70": "seventy",
    "80": "eighty",
    "90": "ninety",
    "100": "hundred",
}

RUSSIAN_NUMBER_WORDS = {
    0: "ноль",
    1: "один",
    2: "два",
    3: "три",
    4: "четыре",
    5: "пять",
    6: "шесть",
    7: "семь",
    8: "восемь",
    9: "девять",
    10: "десять",
    11: "одиннадцать",
    12: "двенадцать",
    13: "тринадцать",
    14: "четырнадцать",
    15: "пятнадцать",
    16: "шестнадцать",
    17: "семнадцать",
    18: "восемнадцать",
    19: "девятнадцать",
    20: "двадцать",
    30: "тридцать",
    40: "сорок",
    50: "пятьдесят",
    60: "шестьдесят",
    70: "семьдесят",
    80: "восемьдесят",
    90: "девяносто",
    100: "сто",
    200: "двести",
    300: "триста",
    400: "четыреста",
    500: "пятьсот",
    600: "шестьсот",
    700: "семьсот",
    800: "восемьсот",
    900: "девятьсот",
}

CONTRACTIONS = {
    "arent": ["are", "not"],
    "cant": ["can", "not"],
    "couldnt": ["could", "not"],
    "didnt": ["did", "not"],
    "doesnt": ["does", "not"],
    "dont": ["do", "not"],
    "hasnt": ["has", "not"],
    "havent": ["have", "not"],
    "isnt": ["is", "not"],
    "shouldnt": ["should", "not"],
    "wasnt": ["was", "not"],
    "werent": ["were", "not"],
    "wont": ["will", "not"],
    "wouldnt": ["would", "not"],
}

APOSTROPHE_CONTRACTIONS = {
    "aren't": ["are", "not"],
    "can't": ["can", "not"],
    "couldn't": ["could", "not"],
    "didn't": ["did", "not"],
    "doesn't": ["does", "not"],
    "don't": ["do", "not"],
    "hasn't": ["has", "not"],
    "haven't": ["have", "not"],
    "i'd": ["i", "would"],
    "i'll": ["i", "will"],
    "i'm": ["i", "am"],
    "i've": ["i", "have"],
    "isn't": ["is", "not"],
    "it's": ["it", "is"],
    "let's": ["let", "us"],
    "that's": ["that", "is"],
    "they're": ["they", "are"],
    "they've": ["they", "have"],
    "wasn't": ["was", "not"],
    "we'd": ["we", "would"],
    "we'll": ["we", "will"],
    "we're": ["we", "are"],
    "we've": ["we", "have"],
    "weren't": ["were", "not"],
    "what's": ["what", "is"],
    "won't": ["will", "not"],
    "wouldn't": ["would", "not"],
    "you're": ["you", "are"],
    "you've": ["you", "have"],
}

# Whisper often renders the spoken word "once" as the numeral 1.  Those two
# forms are acoustically indistinguishable in this context, so canonicalizing
# them avoids deleting the word merely because the recognizer chose a digit.
SPOKEN_EQUIVALENTS = {
    "once": "one",
}


def _russian_below_thousand(value: int) -> list[str]:
    if not 0 <= value < 1000:
        raise ValueError("Russian number chunk must be between 0 and 999")
    if value == 0:
        return [RUSSIAN_NUMBER_WORDS[0]]
    result: list[str] = []
    hundreds, remainder = divmod(value, 100)
    if hundreds:
        result.append(RUSSIAN_NUMBER_WORDS[hundreds * 100])
    if remainder in RUSSIAN_NUMBER_WORDS:
        if remainder:
            result.append(RUSSIAN_NUMBER_WORDS[remainder])
        return result
    tens, units = divmod(remainder, 10)
    if tens:
        result.append(RUSSIAN_NUMBER_WORDS[tens * 10])
    if units:
        result.append(RUSSIAN_NUMBER_WORDS[units])
    return result


def _integer_to_russian_words(value: int) -> list[str]:
    """Return conservative nominative Russian cardinals through 999,999."""

    if value < 0:
        return ["минус", *_integer_to_russian_words(-value)]
    if value < 1000:
        return _russian_below_thousand(value)
    if value >= 1_000_000:
        # Large values have substantially more inflectional ambiguity. Keep
        # the source token literal rather than claiming an unsafe equivalence.
        return [str(value)]
    thousands, remainder = divmod(value, 1000)
    thousands_words = _russian_below_thousand(thousands)
    if thousands_words[-1] == "один":
        thousands_words[-1] = "одна"
    elif thousands_words[-1] == "два":
        thousands_words[-1] = "две"
    last_two = thousands % 100
    last_digit = thousands % 10
    if 11 <= last_two <= 14:
        thousands_noun = "тысяч"
    elif last_digit == 1:
        thousands_noun = "тысяча"
    elif 2 <= last_digit <= 4:
        thousands_noun = "тысячи"
    else:
        thousands_noun = "тысяч"
    result = [*thousands_words, thousands_noun]
    if remainder:
        result.extend(_russian_below_thousand(remainder))
    return result


def integer_to_words(value: int, language: str = "en") -> list[str]:
    if language == "ru":
        return _integer_to_russian_words(value)
    if language != "en":
        raise ValueError(f"unsupported tokenization language: {language}")
    if value < 0:
        return ["minus", *integer_to_words(-value, language=language)]
    if value <= 20 or value in {30, 40, 50, 60, 70, 80, 90}:
        return [NUMBER_WORDS[str(value)]]
    if value < 100:
        tens, remainder = divmod(value, 10)
        return [
            NUMBER_WORDS[str(tens * 10)],
            *integer_to_words(remainder, language=language),
        ]
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        result = [*integer_to_words(hundreds, language=language), "hundred"]
        return result + (
            integer_to_words(remainder, language=language) if remainder else []
        )
    if value < 1_000_000:
        thousands, remainder = divmod(value, 1000)
        result = [*integer_to_words(thousands, language=language), "thousand"]
        return result + (
            integer_to_words(remainder, language=language) if remainder else []
        )
    return [str(value)]


def numeric_tokens(token: str, language: str = "en") -> list[str]:
    if "." not in token:
        return integer_to_words(int(token), language=language)
    integer, fraction = token.split(".", 1)
    if language == "ru":
        return [
            *integer_to_words(int(integer or "0"), language=language),
            "точка",
            *(RUSSIAN_NUMBER_WORDS[int(digit)] for digit in fraction),
        ]
    if language != "en":
        raise ValueError(f"unsupported tokenization language: {language}")
    return [
        *integer_to_words(int(integer or "0"), language=language),
        "point",
        *(NUMBER_WORDS[digit] for digit in fraction),
    ]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False),
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def lightly_stem(token: str) -> str:
    if not re.fullmatch(r"[a-z]+", token):
        return token
    if len(token) <= 4:
        return token
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 6:
        return token[:-3]
    if token.endswith("ed") and len(token) > 5:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("is", "ss", "us")):
        return token[:-1]
    return token


def tokenize(
    text: str,
    aliases: dict[str, list[str]] | None = None,
    language: str = "en",
) -> list[str]:
    if language not in {"en", "ru"}:
        raise ValueError(f"unsupported tokenization language: {language}")
    aliases = aliases or {}
    if language == "ru":
        # NFKC keeps Russian letters such as ё and й intact. The English
        # path deliberately retains its historic accent-insensitive NFKD
        # behavior for backward compatibility.
        normalized = unicodedata.normalize("NFKC", text).casefold()
    else:
        normalized = "".join(
            character
            for character in unicodedata.normalize("NFKD", text).casefold()
            if not unicodedata.combining(character)
        )
    normalized = normalized.replace("’", "'").replace("–", "-").replace("—", "-")
    # ASR may emit the percent sign as a separate timed "word".  Retain that
    # timing and meaning instead of dropping it during tokenization.
    normalized = normalized.replace(
        "%",
        " процент " if language == "ru" else " percent ",
    )
    raw = re.findall(
        r"\d+(?:\.\d+)?|[^\W\d_]+(?:'[^\W\d_]+)?",
        normalized,
        flags=re.UNICODE,
    )
    result: list[str] = []
    for item in raw:
        token = item.replace("'", "")
        if language == "en" and item in APOSTROPHE_CONTRACTIONS:
            expansion = APOSTROPHE_CONTRACTIONS[item]
        elif token in aliases:
            expansion = aliases[token]
        elif language == "en" and token in CONTRACTIONS:
            expansion = CONTRACTIONS[token]
        elif re.fullmatch(r"\d+(?:\.\d+)?", token):
            expansion = numeric_tokens(token, language=language)
        else:
            expansion = [token]
        result.extend(
            (
                lightly_stem(SPOKEN_EQUIVALENTS.get(part, part))
                if language == "en"
                else part
            )
            for part in expansion
            if part
        )
    return result
