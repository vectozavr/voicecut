#!/usr/bin/env python3
"""Shared data models and text helpers for the narration-cleaning pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


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

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "then",
    "this",
    "to",
    "we",
    "when",
    "which",
    "while",
    "with",
}

FILLERS = {"ah", "eh", "erm", "hmm", "mhm", "uh", "um"}

# Whisper often renders the spoken word "once" as the numeral 1.  Those two
# forms are acoustically indistinguishable in this context, so canonicalizing
# them avoids deleting the word merely because the recognizer chose a digit.
SPOKEN_EQUIVALENTS = {
    "once": "one",
}


def integer_to_words(value: int) -> list[str]:
    if value < 0:
        return ["minus", *integer_to_words(-value)]
    if value <= 20 or value in {30, 40, 50, 60, 70, 80, 90}:
        return [NUMBER_WORDS[str(value)]]
    if value < 100:
        tens, remainder = divmod(value, 10)
        return [NUMBER_WORDS[str(tens * 10)], *integer_to_words(remainder)]
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        result = [*integer_to_words(hundreds), "hundred"]
        return result + (integer_to_words(remainder) if remainder else [])
    if value < 1_000_000:
        thousands, remainder = divmod(value, 1000)
        result = [*integer_to_words(thousands), "thousand"]
        return result + (integer_to_words(remainder) if remainder else [])
    return [str(value)]


def numeric_tokens(token: str) -> list[str]:
    if "." not in token:
        return integer_to_words(int(token))
    integer, fraction = token.split(".", 1)
    return [
        *integer_to_words(int(integer or "0")),
        "point",
        *(NUMBER_WORDS[digit] for digit in fraction),
    ]


@dataclass
class ScriptUnit:
    unit_index: int
    line_number: int
    paragraph_index: int
    cue_before: bool
    text: str
    tokens: list[str]


@dataclass
class ScriptPhrase:
    phrase_index: int
    unit_index: int
    phrase_in_unit: int
    line_number: int
    paragraph_index: int
    cue_before: bool
    text: str
    tokens: list[str]
    pause_after: str


@dataclass
class ObservedWord:
    word_index: int
    atom_index: int
    text: str
    tokens: list[str]
    start: float
    end: float
    probability: float


@dataclass
class Candidate:
    phrase_index: int
    word_start: int
    word_end: int
    score: float
    f1: float
    recall: float
    precision: float
    edit_similarity: float
    prefix_recall: float
    suffix_recall: float
    mean_probability: float
    repeated_surplus: float
    max_gap: float
    transcript: str
    source_start: float
    source_end: float


@dataclass
class SelectedPhrase:
    phrase_index: int
    unit_index: int
    word_start: int
    word_end: int
    source_start: float
    source_end: float
    transcript: str
    score: float
    score_margin: float
    f1: float
    recall: float
    precision: float
    mean_probability: float
    pause_after: str
    status: str
    ctc_start: float | None = None
    ctc_end: float | None = None
    ctc_score: float | None = None
    review_reasons: list[str] = field(default_factory=list)


@dataclass
class EditPiece:
    piece_index: int
    phrase_indices: list[int]
    unit_indices: list[int]
    source_start_sample: int
    source_end_sample: int
    first_speech_sample: int
    last_speech_sample: int
    transcript: str
    pause_after: str
    gap_after_samples: int
    fade_samples: int
    start_cut_rms_db: float
    end_cut_rms_db: float
    start_boundary_kind: str
    end_boundary_kind: str
    output_start_sample: int = 0
    output_end_sample: int = 0
    output_gap_end_sample: int = 0
    breath_attenuations: list[dict[str, Any]] = field(default_factory=list)
    interword_gap_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    alignment_diagnostics: list[str] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False),
        encoding="utf-8",
    )


def jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=capture,
    )


def probe_audio(path: Path) -> dict[str, Any]:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,bits_per_sample,bits_per_raw_sample:"
            "format=duration,size,bit_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result.stdout)


def load_aliases(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    raw = read_json(path)
    source = raw.get("aliases", raw) if isinstance(raw, dict) else {}
    aliases: dict[str, list[str]] = {}
    for key, value in source.items():
        normalized_key = simple_raw_token(str(key))
        if not normalized_key:
            continue
        if isinstance(value, list):
            expansion: list[str] = []
            for item in value:
                expansion.extend(tokenize(str(item), {}))
        else:
            expansion = tokenize(str(value), {})
        if expansion:
            aliases[normalized_key] = expansion
    return aliases


def simple_raw_token(text: str) -> str:
    text = "".join(
        character
        for character in unicodedata.normalize("NFKD", text).casefold()
        if not unicodedata.combining(character)
    )
    text = text.replace("’", "'")
    match = re.search(r"[^\W_]+(?:'[^\W_]+)?", text, flags=re.UNICODE)
    return match.group(0).replace("'", "") if match else ""


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


def tokenize(text: str, aliases: dict[str, list[str]] | None = None) -> list[str]:
    aliases = aliases or {}
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", text).casefold()
        if not unicodedata.combining(character)
    )
    normalized = normalized.replace("’", "'").replace("–", "-").replace("—", "-")
    # ASR may emit the percent sign as a separate timed "word".  Retain that
    # timing and meaning instead of dropping it during tokenization.
    normalized = normalized.replace("%", " percent ")
    raw = re.findall(
        r"\d+(?:\.\d+)?|[^\W\d_]+(?:'[^\W\d_]+)?",
        normalized,
        flags=re.UNICODE,
    )
    result: list[str] = []
    for item in raw:
        token = item.replace("'", "")
        if item in APOSTROPHE_CONTRACTIONS:
            expansion = APOSTROPHE_CONTRACTIONS[item]
        elif token in aliases:
            expansion = aliases[token]
        elif token in CONTRACTIONS:
            expansion = CONTRACTIONS[token]
        elif re.fullmatch(r"\d+(?:\.\d+)?", token):
            expansion = numeric_tokens(token)
        else:
            expansion = [token]
        result.extend(
            lightly_stem(SPOKEN_EQUIVALENTS.get(part, part))
            for part in expansion
            if part
        )
    return result


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=\S)", text.strip())
    return [part.strip() for part in parts if part.strip()]


def is_nonspoken_markdown_line(line: str) -> bool:
    """Return whether a Markdown line is presentation metadata, not narration."""

    if line.startswith("#") or (line.startswith("[") and line.endswith("]")):
        return True
    if re.fullmatch(r"!\[[^\]]*\]\([^)]+\)", line):
        return True
    if re.fullmatch(r"<?https?://\S+>?", line):
        return True
    resource_prefix = (
        r"(?:code|demo|paper|repo(?:sitory)?|slides?|website|links?|resources?)"
    )
    return bool(
        re.fullmatch(
            resource_prefix
            + r"\s*:\s*(?:<?https?://\S+>?|\[[^\]]+\]\(https?://[^)]+\))",
            line,
            flags=re.IGNORECASE,
        )
    )


def parse_script(path: Path, aliases: dict[str, list[str]] | None = None) -> list[ScriptUnit]:
    aliases = aliases or {}
    units: list[ScriptUnit] = []
    paragraph_index = 0
    cue_pending = True
    buffered_lines: list[tuple[int, str]] = []
    fenced_block: str | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph_index, cue_pending, buffered_lines
        if not buffered_lines:
            return
        paragraph_index += 1
        line_number = buffered_lines[0][0]
        paragraph = " ".join(line for _, line in buffered_lines)
        sentences = split_sentences(paragraph)
        for sentence_index, sentence in enumerate(sentences):
            tokens = tokenize(sentence, aliases)
            if not tokens:
                continue
            units.append(
                ScriptUnit(
                    unit_index=len(units),
                    line_number=line_number,
                    paragraph_index=paragraph_index,
                    cue_before=cue_pending and sentence_index == 0,
                    text=sentence,
                    tokens=tokens,
                )
            )
            cue_pending = False
        buffered_lines = []

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        fence = re.match(r"^(```+|~~~+)", line)
        if fence:
            flush_paragraph()
            marker = fence.group(1)[0]
            fenced_block = None if fenced_block == marker else marker
            cue_pending = True
            continue
        if fenced_block is not None:
            continue
        if not line:
            flush_paragraph()
            if units:
                cue_pending = True
            continue
        if is_nonspoken_markdown_line(line):
            flush_paragraph()
            cue_pending = True
            continue
        buffered_lines.append((line_number, line))
    flush_paragraph()
    return units


def split_long_clause(text: str, aliases: dict[str, list[str]], max_tokens: int) -> list[str]:
    words = text.split()
    if len(tokenize(text, aliases)) <= max_tokens or len(words) < 6:
        return [text.strip()]
    conjunctions = {
        "and",
        "but",
        "because",
        "so",
        "while",
        "whereas",
        "which",
        "that",
        "then",
    }
    midpoint = len(words) // 2
    punctuation_boundaries = [
        index + 1
        for index in range(2, len(words) - 2)
        if words[index].rstrip("\"'”’").endswith((",", ";", ":"))
    ]
    conjunction_boundaries = [
        index
        for index in range(3, len(words) - 2)
        if simple_raw_token(words[index]) in conjunctions
    ]
    if punctuation_boundaries:
        split_at = min(
            punctuation_boundaries,
            key=lambda index: abs(index - midpoint),
        )
    elif conjunction_boundaries:
        split_at = min(
            conjunction_boundaries,
            key=lambda index: abs(index - midpoint),
        )
    else:
        split_at = min(len(words) - 2, max(3, midpoint))
    left = " ".join(words[:split_at]).strip()
    right = " ".join(words[split_at:]).strip()
    return split_long_clause(left, aliases, max_tokens) + split_long_clause(
        right, aliases, max_tokens
    )


def build_phrases(
    units: list[ScriptUnit],
    aliases: dict[str, list[str]] | None = None,
    max_tokens: int = 18,
) -> list[ScriptPhrase]:
    aliases = aliases or {}
    phrases: list[ScriptPhrase] = []
    for unit in units:
        # Keep ordinary sentences intact.  Splitting every comma creates tiny
        # formula fragments ("minus W k q") that ASR cannot select without
        # overlapping their surrounding words.  Only genuinely long sentences
        # are divided, recursively, near a conjunction/midpoint.
        expanded = split_long_clause(unit.text, aliases, max_tokens)
        for local_index, part in enumerate(expanded):
            tokens = tokenize(part, aliases)
            if not tokens:
                continue
            is_last = local_index == len(expanded) - 1
            if is_last:
                pause_after = "sentence"
            elif part.rstrip().endswith((",", ";", ":")):
                pause_after = "clause"
            else:
                pause_after = "phrase"
            phrases.append(
                ScriptPhrase(
                    phrase_index=len(phrases),
                    unit_index=unit.unit_index,
                    phrase_in_unit=local_index,
                    line_number=unit.line_number,
                    paragraph_index=unit.paragraph_index,
                    cue_before=unit.cue_before and local_index == 0,
                    text=part,
                    tokens=tokens,
                    pause_after=pause_after,
                )
            )
    return phrases


def ngrams(tokens: Sequence[str], size: int) -> Iterable[tuple[str, ...]]:
    for index in range(max(0, len(tokens) - size + 1)):
        yield tuple(tokens[index : index + size])


def repeated_surplus(reference: Sequence[str], observed: Sequence[str]) -> float:
    """Return a weighted surplus repetition score relative to the script.

    Comparing against the reference is important: intentional repetitions in
    the script must not be treated as retakes.
    """

    penalty = 0.0
    for size in range(1, min(8, len(observed)) + 1):
        ref_counts = Counter(ngrams(reference, size))
        obs_counts = Counter(ngrams(observed, size))
        for gram, count in obs_counts.items():
            # A word or n-gram that differs from the script but occurs once is
            # an ASR/substitution error, not a retry.  Repetition begins only
            # with an extra copy beyond either the script's multiplicity or
            # the first observed occurrence.
            allowed_count = max(1, ref_counts.get(gram, 0))
            surplus = max(0, count - allowed_count)
            if not surplus:
                continue
            if size == 1 and gram[0] in STOPWORDS:
                weight = 0.05
            elif size == 1:
                weight = 0.12
            elif size == 2:
                weight = 0.5
            else:
                weight = min(2.0, 0.45 * size)
            penalty += weight * surplus
    return penalty


def script_text(units: Sequence[ScriptUnit], allowed_missing: set[int] | None = None) -> str:
    allowed_missing = allowed_missing or set()
    return " ".join(
        unit.text for unit in units if (unit.unit_index + 1) not in allowed_missing
    )
