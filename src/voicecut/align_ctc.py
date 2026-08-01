#!/usr/bin/env python3
"""Decode raw CTC evidence that can expose speech hidden by Whisper.

WhisperX is used only to load its language-specific torchaudio CTC model.
This module deliberately does not run WhisperX's known-text forced alignment:
the enrichment stage consumes the model's raw greedy word occurrences, retry
evidence, and source-grounded expected substitutions only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import librosa
import numpy as np
import torch
import whisperx

from .common import sha256_file


WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
MIN_ANCHOR_LEXICAL_SCORE = 0.90
MIN_ANCHOR_ACOUSTIC_SCORE = 0.45
MIN_INSERTION_ACOUSTIC_SCORE = 0.40
MAX_RETRY_WORDS = 6
MAX_SAFE_EDIT_SECONDS = 5.0
MIN_EXPECTED_SUBSTITUTION_ACOUSTIC_SCORE = 0.75
MAX_EXPECTED_SUBSTITUTION_WORDS = 2
OUTPUT_SCHEMA_VERSION = 3
OUTPUT_MODE = "raw_greedy_ctc_retry_evidence"


def serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable(item) for item in value]
    if isinstance(value, tuple):
        return [serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalized_words(value: str | Sequence[str]) -> list[str]:
    """Return the lowercase spoken words used by the acoustic retry detector."""

    if isinstance(value, str):
        text = value
    else:
        text = " ".join(str(word) for word in value)
    return WORD_PATTERN.findall(text.lower().replace("’", "'"))


def fuzzy_word_score(expected: str, observed: str) -> float:
    """Score a CTC spelling against a script word without guessing synonyms."""

    left = "".join(normalized_words(expected))
    right = "".join(normalized_words(observed))
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if min(len(left), len(right)) <= 3:
        return 0.0
    return float(SequenceMatcher(None, left, right).ratio())


def _acceptable_fuzzy_match(expected: str, observed: str, score: float) -> bool:
    if score >= 1.0:
        return True
    shortest = min(len(expected), len(observed))
    threshold = 0.75 if shortest == 4 else 0.68
    return shortest >= 4 and score >= threshold


def _alignment_key(
    matches: tuple[tuple[int, int, float], ...],
) -> tuple[Any, ...]:
    """Rank alignments, using later occurrences only as the final tie-break."""

    similarities = round(sum(item[2] for item in matches), 6)
    exact_matches = sum(item[2] >= 0.999999 for item in matches)
    greedy_indices = tuple(item[1] for item in matches)
    # Prefer the earliest expected words when equally many expected words match,
    # but prefer the latest acoustic occurrence of that same complete sequence.
    expected_indices = tuple(-item[0] for item in matches)
    return (
        len(matches),
        similarities,
        exact_matches,
        expected_indices,
        greedy_indices,
    )


def align_expected_to_greedy(
    expected: str | Sequence[str],
    greedy_words: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fuzzily align script words to raw greedy CTC words.

    Greedy CTC often preserves a false start that Whisper suppresses.  The
    alignment is deliberately lexical (not semantic), and equal complete
    occurrences resolve to the latest take.
    """

    expected_words = normalized_words(expected)
    observed_words = [
        normalized_words(str(word.get("word", ""))) for word in greedy_words
    ]
    observed = [parts[0] if len(parts) == 1 else "" for parts in observed_words]

    @lru_cache(maxsize=None)
    def solve(
        expected_index: int,
        greedy_index: int,
    ) -> tuple[tuple[int, int, float], ...]:
        if expected_index >= len(expected_words) or greedy_index >= len(observed):
            return ()
        candidates = [
            solve(expected_index + 1, greedy_index),
            solve(expected_index, greedy_index + 1),
        ]
        lexical_score = fuzzy_word_score(
            expected_words[expected_index],
            observed[greedy_index],
        )
        if _acceptable_fuzzy_match(
            expected_words[expected_index],
            observed[greedy_index],
            lexical_score,
        ):
            candidates.append(
                (
                    (expected_index, greedy_index, lexical_score),
                    *solve(expected_index + 1, greedy_index + 1),
                )
            )
        return max(candidates, key=_alignment_key)

    matches = solve(0, 0) if expected_words and observed else ()
    return [
        {
            "expected_index": expected_index,
            "expected_word": expected_words[expected_index],
            "greedy_index": greedy_index,
            "greedy_word": observed[greedy_index],
            "lexical_score": round(float(score), 6),
        }
        for expected_index, greedy_index, score in matches
    ]


def _emission_probabilities(emissions: Any) -> np.ndarray:
    if hasattr(emissions, "detach"):
        emissions = emissions.detach().cpu().numpy()
    matrix = np.asarray(emissions, dtype=np.float64)
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix[0]
    if matrix.ndim != 2:
        raise ValueError("CTC emissions must have shape [frames, vocabulary].")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return np.empty(matrix.shape, dtype=np.float64)
    row_sums = matrix.sum(axis=1)
    if (
        np.all(np.isfinite(matrix))
        and np.min(matrix) >= 0.0
        and np.max(matrix) <= 1.0
        and np.allclose(row_sums, 1.0, atol=1e-4)
    ):
        return matrix
    shifted = matrix - np.max(matrix, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    denominator = probabilities.sum(axis=1, keepdims=True)
    return np.divide(
        probabilities,
        denominator,
        out=np.zeros_like(probabilities),
        where=denominator > 0.0,
    )


def _ctc_blank_id(dictionary: dict[str, int]) -> int:
    for label in ("[pad]", "<pad>", "-"):
        if label in dictionary:
            return int(dictionary[label])
    return 0


def decode_collapsed_ctc_words(
    emissions: Any,
    dictionary: dict[str, int],
    *,
    absolute_start: float,
    absolute_end: float,
) -> list[dict[str, Any]]:
    """Greedy-decode collapsed CTC characters into absolute-time words."""

    probabilities = _emission_probabilities(emissions)
    if probabilities.shape[0] == 0 or absolute_end <= absolute_start:
        return []
    token_ids = np.argmax(probabilities, axis=1)
    inverse_dictionary: dict[int, str] = {}
    for label, token_id in dictionary.items():
        inverse_dictionary.setdefault(int(token_id), str(label).lower())
    blank_id = _ctc_blank_id(dictionary)
    separators = {"|", " ", "<space>"}
    frame_seconds = (absolute_end - absolute_start) / probabilities.shape[0]
    decoded: list[dict[str, Any]] = []
    characters: list[str] = []
    character_scores: list[float] = []
    word_start: float | None = None
    word_end: float | None = None

    def flush_word() -> None:
        nonlocal characters, character_scores, word_start, word_end
        if characters and word_start is not None and word_end is not None:
            score = float(np.mean(character_scores))
            decoded.append(
                {
                    "word": "".join(characters),
                    "start": round(word_start, 6),
                    "end": round(word_end, 6),
                    "score": round(score, 6),
                    "min_score": round(float(min(character_scores)), 6),
                    "character_count": len(characters),
                }
            )
        characters = []
        character_scores = []
        word_start = None
        word_end = None

    run_start = 0
    for frame_end in range(1, len(token_ids) + 1):
        if frame_end < len(token_ids) and token_ids[frame_end] == token_ids[run_start]:
            continue
        token_id = int(token_ids[run_start])
        label = inverse_dictionary.get(token_id, "")
        if token_id != blank_id:
            if label in separators:
                flush_word()
            elif label and not (label.startswith("<") or label.startswith("[")):
                start = absolute_start + run_start * frame_seconds
                end = absolute_start + frame_end * frame_seconds
                if word_start is None:
                    word_start = start
                word_end = end
                characters.append(label)
                character_scores.append(
                    float(np.mean(probabilities[run_start:frame_end, token_id]))
                )
        run_start = frame_end
    flush_word()
    return decoded


def decode_greedy_ctc_segment(
    model: Any,
    metadata: dict[str, Any],
    audio: np.ndarray,
    *,
    start: float,
    end: float,
    device: str,
    sample_rate: int = 16_000,
) -> list[dict[str, Any]]:
    """Run raw greedy decoding for one requested torchaudio CTC interval."""

    dictionary = metadata.get("dictionary")
    if (
        metadata.get("type") != "torchaudio"
        or not isinstance(dictionary, dict)
        or not dictionary
        or not callable(model)
    ):
        return []
    sample_start = max(0, min(len(audio), int(start * sample_rate)))
    sample_end = max(sample_start, min(len(audio), int(end * sample_rate)))
    if sample_end <= sample_start:
        return []
    waveform = torch.as_tensor(
        np.asarray(audio[sample_start:sample_end], dtype=np.float32)
    ).unsqueeze(0)
    original_samples = int(waveform.shape[-1])
    if original_samples < 400:
        lengths = torch.as_tensor([original_samples]).to(device)
        waveform = torch.nn.functional.pad(waveform, (0, 400 - original_samples))
    else:
        lengths = None
    with torch.inference_mode():
        model_result = model(waveform.to(device), lengths=lengths)
    emissions = model_result[0] if isinstance(model_result, tuple) else model_result
    return decode_collapsed_ctc_words(
        emissions[0],
        dictionary,
        absolute_start=sample_start / sample_rate,
        absolute_end=sample_end / sample_rate,
    )


def _word_acoustic_score(word: dict[str, Any]) -> float:
    value = word.get("score", word.get("confidence", 0.0))
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _finite_word_interval(word: dict[str, Any]) -> tuple[float, float] | None:
    try:
        start = float(word["start"])
        end = float(word["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return None
    return start, end


def discover_acoustic_insertions(
    expected: str | Sequence[str],
    greedy_words: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find fenced, source-grounded restarts hidden by hypothesis alignment."""

    expected_words = normalized_words(expected)
    alignment = align_expected_to_greedy(expected_words, greedy_words)
    insertions: list[dict[str, Any]] = []
    if len(alignment) < 2:
        return insertions
    for left_match, right_match in zip(alignment, alignment[1:]):
        first_unmatched = int(left_match["greedy_index"]) + 1
        right_greedy_index = int(right_match["greedy_index"])
        if first_unmatched >= right_greedy_index:
            continue
        unmatched = list(greedy_words[first_unmatched:right_greedy_index])
        if not unmatched or len(unmatched) > MAX_RETRY_WORDS:
            continue
        if (
            float(left_match["lexical_score"]) < MIN_ANCHOR_LEXICAL_SCORE
            or float(right_match["lexical_score"]) < MIN_ANCHOR_LEXICAL_SCORE
        ):
            continue
        left_anchor = greedy_words[int(left_match["greedy_index"])]
        right_anchor = greedy_words[right_greedy_index]
        if (
            _word_acoustic_score(left_anchor) < MIN_ANCHOR_ACOUSTIC_SCORE
            or _word_acoustic_score(right_anchor) < MIN_ANCHOR_ACOUSTIC_SCORE
            or any(
                _word_acoustic_score(word) < MIN_INSERTION_ACOUSTIC_SCORE
                for word in unmatched
            )
        ):
            continue
        intervals = [
            _finite_word_interval(word)
            for word in (left_anchor, *unmatched, right_anchor)
        ]
        if any(interval is None for interval in intervals):
            continue
        valid_intervals = [interval for interval in intervals if interval is not None]
        if any(
            previous[1] > current[0]
            for previous, current in zip(valid_intervals, valid_intervals[1:])
        ):
            continue
        safe_edit_start = valid_intervals[0][1]
        safe_edit_end = valid_intervals[-1][0]
        if (
            safe_edit_end <= safe_edit_start
            or safe_edit_end - safe_edit_start > MAX_SAFE_EDIT_SECONDS
        ):
            continue

        right_expected_index = int(right_match["expected_index"])
        expected_window = expected_words[
            right_expected_index : right_expected_index + len(unmatched) + 3
        ]
        retry_alignment = align_expected_to_greedy(expected_window, unmatched)
        if len(retry_alignment) != len(unmatched):
            continue
        if {int(match["greedy_index"]) for match in retry_alignment} != set(
            range(len(unmatched))
        ):
            continue
        # A conservative retry must restart on the same expected word as the
        # selected right-hand take.  This excludes fillers and paraphrases.
        if (
            int(retry_alignment[0]["expected_index"]) != 0
            or float(retry_alignment[0]["lexical_score"]) < MIN_ANCHOR_LEXICAL_SCORE
        ):
            continue
        retry_lexical_score = float(
            np.mean([float(match["lexical_score"]) for match in retry_alignment])
        )
        if retry_lexical_score < 0.78:
            continue
        acoustic_score = min(
            _word_acoustic_score(word)
            for word in (left_anchor, *unmatched, right_anchor)
        )
        source_start = valid_intervals[1][0]
        source_end = valid_intervals[-2][1]
        insertions.append(
            {
                "type": "spoken_retry",
                "reason": "greedy_ctc_restart_before_selected_take",
                "text": " ".join(
                    normalized_words(str(word.get("word", "")))[0] for word in unmatched
                ),
                "start": round(source_start, 6),
                "end": round(source_end, 6),
                "safe_edit_start": round(safe_edit_start, 6),
                "safe_edit_end": round(safe_edit_end, 6),
                "left_expected_index": int(left_match["expected_index"]),
                "right_expected_index": right_expected_index,
                "retry_expected_indices": [
                    right_expected_index + int(match["expected_index"])
                    for match in retry_alignment
                ],
                "words": [dict(word) for word in unmatched],
                "left_anchor": {
                    **left_match,
                    "start": valid_intervals[0][0],
                    "end": valid_intervals[0][1],
                },
                "right_anchor": {
                    **right_match,
                    "start": valid_intervals[-1][0],
                    "end": valid_intervals[-1][1],
                },
                "lexical_score": round(retry_lexical_score, 6),
                "acoustic_score": round(acoustic_score, 6),
                "confidence": round(
                    min(retry_lexical_score, acoustic_score),
                    6,
                ),
            }
        )
    return insertions


def _strong_lexical_equivalent(left: str, right: str) -> bool:
    score = fuzzy_word_score(left, right)
    return score >= MIN_ANCHOR_LEXICAL_SCORE


def _substitution_anchor(
    match: dict[str, Any],
    word: dict[str, Any],
    interval: tuple[float, float],
) -> dict[str, Any]:
    return {
        **match,
        "start": round(interval[0], 6),
        "end": round(interval[1], 6),
        "acoustic_score": round(_word_acoustic_score(word), 6),
    }


def discover_acoustic_expected_substitutions(
    expected: str | Sequence[str],
    greedy_words: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ground a short expected-word substitution in raw CTC evidence.

    This is evidence that a forced expected token spans real speech, not
    permission to remove that speech.  Both sides must be securely anchored,
    and insertions, deletions, reordered words, and restart-like repetitions
    remain unclassified.
    """

    expected_words = normalized_words(expected)
    alignment = align_expected_to_greedy(expected_words, greedy_words)
    substitutions: list[dict[str, Any]] = []
    if len(alignment) < 2:
        return substitutions

    for left_match, right_match in zip(alignment, alignment[1:]):
        expected_start = int(left_match["expected_index"]) + 1
        expected_end = int(right_match["expected_index"])
        greedy_start = int(left_match["greedy_index"]) + 1
        greedy_end = int(right_match["greedy_index"])
        unmatched_expected = expected_words[expected_start:expected_end]
        unmatched_greedy = list(greedy_words[greedy_start:greedy_end])
        if (
            not unmatched_expected
            or len(unmatched_expected) != len(unmatched_greedy)
            or len(unmatched_expected) > MAX_EXPECTED_SUBSTITUTION_WORDS
            or float(left_match["lexical_score"]) < MIN_ANCHOR_LEXICAL_SCORE
            or float(right_match["lexical_score"]) < MIN_ANCHOR_LEXICAL_SCORE
        ):
            continue

        left_word = greedy_words[int(left_match["greedy_index"])]
        right_word = greedy_words[int(right_match["greedy_index"])]
        grounded_words = [left_word, *unmatched_greedy, right_word]
        intervals = [_finite_word_interval(word) for word in grounded_words]
        if any(interval is None for interval in intervals):
            continue
        valid_intervals = [interval for interval in intervals if interval is not None]
        if any(
            previous[1] > current[0]
            for previous, current in zip(valid_intervals, valid_intervals[1:])
        ):
            continue
        acoustic_scores = [_word_acoustic_score(word) for word in grounded_words]
        if min(acoustic_scores) < MIN_EXPECTED_SUBSTITUTION_ACOUSTIC_SCORE:
            continue

        observed_words = [
            normalized_words(str(word.get("word", ""))) for word in unmatched_greedy
        ]
        if any(len(parts) != 1 for parts in observed_words):
            continue
        observed = [parts[0] for parts in observed_words]
        left_observed = str(left_match["greedy_word"])
        right_observed = str(right_match["greedy_word"])
        # Do not relabel a repeated anchor or a reordered expected word as a
        # harmless substitution. Those patterns remain retry/content evidence.
        if any(
            _strong_lexical_equivalent(word, left_observed)
            or _strong_lexical_equivalent(word, right_observed)
            for word in observed
        ):
            continue
        if any(
            _strong_lexical_equivalent(left, right)
            for index, left in enumerate(observed)
            for right in observed[index + 1 :]
        ):
            continue
        if any(
            _acceptable_fuzzy_match(
                expected_word,
                observed_word,
                fuzzy_word_score(expected_word, observed_word),
            )
            for expected_word in unmatched_expected
            for observed_word in observed
        ):
            continue

        left_interval = valid_intervals[0]
        right_interval = valid_intervals[-1]
        left_anchor = _substitution_anchor(
            left_match,
            left_word,
            left_interval,
        )
        right_anchor = _substitution_anchor(
            right_match,
            right_word,
            right_interval,
        )
        for offset, (expected_word, greedy_word) in enumerate(
            zip(unmatched_expected, unmatched_greedy)
        ):
            interval = valid_intervals[offset + 1]
            substitutions.append(
                {
                    "expected_index": expected_start + offset,
                    "expected_word": expected_word,
                    "greedy_index": greedy_start + offset,
                    "greedy_word": observed[offset],
                    "start": round(interval[0], 6),
                    "end": round(interval[1], 6),
                    "acoustic_score": round(
                        _word_acoustic_score(greedy_word),
                        6,
                    ),
                    "left_anchor": left_anchor,
                    "right_anchor": right_anchor,
                    "reason": ("greedy_ctc_source_grounded_expected_substitution"),
                }
            )
    return substitutions


def _requested_segments(source: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = source.get("segments", [])
    if not isinstance(raw_segments, list):
        raise ValueError("CTC input 'segments' must be a list.")
    requested: list[dict[str, Any]] = []
    seen_phrase_ids: set[int] = set()
    for position, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise ValueError(f"CTC input segment {position} must be an object.")
        phrase_index = raw.get("phrase_index")
        if type(phrase_index) is not int:
            raise ValueError(
                f"CTC input segment {position} has no integer phrase_index."
            )
        if phrase_index in seen_phrase_ids:
            raise ValueError(f"CTC input repeats phrase_index {phrase_index}.")
        try:
            start = float(raw["start"])
            end = float(raw["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"CTC input segment {phrase_index} has invalid timestamps."
            ) from error
        text = raw.get("text")
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or end <= start
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError(f"CTC input segment {phrase_index} is invalid.")
        seen_phrase_ids.add(phrase_index)
        requested.append(
            {
                "phrase_index": phrase_index,
                "start": start,
                "end": end,
                "text": text.strip(),
            }
        )
    return requested


def _requested_skipped_segments(
    source: dict[str, Any],
    *,
    decoded_phrase_ids: set[int],
) -> list[dict[str, Any]]:
    raw_segments = source.get("skipped_segments", [])
    if not isinstance(raw_segments, list):
        raise ValueError("CTC input 'skipped_segments' must be a list.")
    skipped: list[dict[str, Any]] = []
    seen_phrase_ids: set[int] = set()
    for position, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise ValueError(f"CTC skipped segment {position} must be an object.")
        phrase_index = raw.get("phrase_index")
        if type(phrase_index) is not int:
            raise ValueError(
                f"CTC skipped segment {position} has no integer phrase_index."
            )
        if phrase_index in decoded_phrase_ids or phrase_index in seen_phrase_ids:
            raise ValueError(f"CTC input repeats phrase_index {phrase_index}.")
        try:
            start = float(raw["input_start"])
            end = float(raw["input_end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"CTC skipped segment {phrase_index} has invalid timestamps."
            ) from error
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or end <= start
            or raw.get("reason") != "no_lexical_content"
        ):
            raise ValueError(f"CTC skipped segment {phrase_index} is invalid.")
        seen_phrase_ids.add(phrase_index)
        skipped.append(
            {
                "phrase_index": phrase_index,
                "input_start": start,
                "input_end": end,
                "reason": "no_lexical_content",
            }
        )
    return skipped


def _evidence_segment(
    requested: dict[str, Any],
    greedy_words: list[dict[str, Any]],
) -> dict[str, Any]:
    text = str(requested["text"])
    return {
        "phrase_index": int(requested["phrase_index"]),
        "input_start": float(requested["start"]),
        "input_end": float(requested["end"]),
        "input_text": text,
        "greedy_ctc_words": greedy_words,
        "acoustic_insertions": discover_acoustic_insertions(
            text,
            greedy_words,
        ),
        "acoustic_expected_substitutions": (
            discover_acoustic_expected_substitutions(
                text,
                greedy_words,
            )
        ),
    }


def _failed_evidence_segment(
    requested: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    """Represent one failed optional decode without losing later segments."""

    return {
        "phrase_index": int(requested["phrase_index"]),
        "input_start": float(requested["start"]),
        "input_end": float(requested["end"]),
        "input_text": str(requested["text"]),
        "greedy_ctc_words": [],
        "acoustic_insertions": [],
        "acoustic_expected_substitutions": [],
        "status": "decode_failed",
        "decode_error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def _checkpoint_identity(
    *,
    audio_path: Path,
    input_path: Path,
    language: str,
    device: str,
) -> dict[str, Any]:
    return {
        "audio": str(audio_path.resolve()),
        "audio_sha256": sha256_file(audio_path),
        "input": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "language": language,
        "device": device,
    }


def _matches_requested_segment(
    segment: dict[str, Any],
    requested: dict[str, Any],
) -> bool:
    try:
        return (
            int(segment["phrase_index"]) == int(requested["phrase_index"])
            and float(segment["input_start"]) == float(requested["start"])
            and float(segment["input_end"]) == float(requested["end"])
            and str(segment["input_text"]) == str(requested["text"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _normalized_checkpoint_segments(
    checkpoint: dict[str, Any],
    *,
    identity: dict[str, Any],
    requested_by_id: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    if (
        checkpoint.get("schema_version") != OUTPUT_SCHEMA_VERSION
        or checkpoint.get("mode") != OUTPUT_MODE
        or checkpoint.get("complete") is not False
    ):
        raise RuntimeError("CTC checkpoint has an unsupported schema.")
    for key in identity:
        if checkpoint.get(key) != identity[key]:
            raise RuntimeError("CTC checkpoint does not match this evidence run.")
    raw_segments = checkpoint.get("segments", [])
    if not isinstance(raw_segments, list):
        raise RuntimeError("CTC checkpoint has an invalid segment ledger.")

    completed: dict[int, dict[str, Any]] = {}
    for raw in raw_segments:
        if not isinstance(raw, dict) or type(raw.get("phrase_index")) is not int:
            raise RuntimeError("CTC checkpoint has an invalid phrase index.")
        phrase_index = int(raw["phrase_index"])
        if phrase_index in completed:
            raise RuntimeError(f"CTC checkpoint repeats phrase_index {phrase_index}.")
        requested = requested_by_id.get(phrase_index)
        if requested is None:
            raise RuntimeError(
                f"CTC checkpoint contains unknown phrase_index {phrase_index}."
            )
        if not _matches_requested_segment(raw, requested):
            raise RuntimeError(
                f"CTC checkpoint phrase {phrase_index} does not match its input."
            )
        greedy_words = raw.get("greedy_ctc_words")
        insertions = raw.get("acoustic_insertions")
        substitutions = raw.get("acoustic_expected_substitutions")
        if not (
            isinstance(greedy_words, list)
            and isinstance(insertions, list)
            and isinstance(substitutions, list)
        ):
            raise RuntimeError(
                f"CTC checkpoint phrase {phrase_index} has invalid evidence."
            )
        normalized = {
            "phrase_index": phrase_index,
            "input_start": float(requested["start"]),
            "input_end": float(requested["end"]),
            "input_text": str(requested["text"]),
            "greedy_ctc_words": greedy_words,
            "acoustic_insertions": insertions,
            "acoustic_expected_substitutions": substitutions,
        }
        status = raw.get("status")
        if status is not None:
            if status != "decode_failed" or not isinstance(
                raw.get("decode_error"),
                dict,
            ):
                raise RuntimeError(
                    f"CTC checkpoint phrase {phrase_index} has invalid status."
                )
            normalized["status"] = "decode_failed"
            normalized["decode_error"] = dict(raw["decode_error"])
        completed[phrase_index] = normalized
    return completed


def _write_checkpoint(
    path: Path,
    *,
    identity: dict[str, Any],
    segments: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(
            serializable(
                {
                    "schema_version": OUTPUT_SCHEMA_VERSION,
                    "mode": OUTPUT_MODE,
                    "complete": False,
                    **identity,
                    "segments": segments,
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decode raw CTC word occurrences and conservative retry evidence. "
            "No known-text forced alignment is performed."
        )
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("CTC input root must be an object.")
    segments = _requested_segments(source)
    skipped_segments = _requested_skipped_segments(
        source,
        decoded_phrase_ids={int(segment["phrase_index"]) for segment in segments},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not segments:
        args.output.write_text(
            json.dumps(
                {
                    "schema_version": OUTPUT_SCHEMA_VERSION,
                    "mode": OUTPUT_MODE,
                    "segments": [],
                    **(
                        {"skipped_segments": skipped_segments}
                        if skipped_segments
                        else {}
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    identity = _checkpoint_identity(
        audio_path=args.audio,
        input_path=args.input,
        language=args.language,
        device=args.device,
    )
    requested_by_id = {int(segment["phrase_index"]): segment for segment in segments}
    partial_path = args.output.with_name(args.output.name + ".partial")
    completed_by_id: dict[int, dict[str, Any]] = {}
    if args.resume and partial_path.exists():
        checkpoint = json.loads(partial_path.read_text(encoding="utf-8"))
        if not isinstance(checkpoint, dict):
            raise RuntimeError("CTC checkpoint root must be an object.")
        completed_by_id = _normalized_checkpoint_segments(
            checkpoint,
            identity=identity,
            requested_by_id=requested_by_id,
        )

    pending = [
        segment
        for segment in segments
        if int(segment["phrase_index"]) not in completed_by_id
    ]
    if pending:
        audio, _ = librosa.load(args.audio, sr=16000, mono=True)
        model, metadata = whisperx.load_align_model(
            language_code=args.language,
            device=args.device,
        )
        for requested in pending:
            phrase_index = int(requested["phrase_index"])
            try:
                greedy_words = decode_greedy_ctc_segment(
                    model,
                    metadata,
                    audio,
                    start=float(requested["start"]),
                    end=float(requested["end"]),
                    device=args.device,
                )
                completed_by_id[phrase_index] = _evidence_segment(
                    requested,
                    greedy_words,
                )
            except Exception as error:
                completed_by_id[phrase_index] = _failed_evidence_segment(
                    requested,
                    error,
                )
            ordered_completed = [
                completed_by_id[int(segment["phrase_index"])]
                for segment in segments
                if int(segment["phrase_index"]) in completed_by_id
            ]
            _write_checkpoint(
                partial_path,
                identity=identity,
                segments=ordered_completed,
            )
            print(
                f"decoded {len(ordered_completed)}/{len(segments)} phrases",
                file=sys.stderr,
                flush=True,
            )

    ordered_segments = [
        completed_by_id[int(segment["phrase_index"])] for segment in segments
    ]
    evidence = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "mode": OUTPUT_MODE,
        "segments": ordered_segments,
        **({"skipped_segments": skipped_segments} if skipped_segments else {}),
    }
    args.output.write_text(
        json.dumps(serializable(evidence), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if partial_path.exists():
        partial_path.unlink()
    raw_word_count = sum(
        len(segment["greedy_ctc_words"]) for segment in ordered_segments
    )
    retry_count = sum(
        len(segment["acoustic_insertions"]) for segment in ordered_segments
    )
    failed_count = sum(
        segment.get("status") == "decode_failed" for segment in ordered_segments
    )
    print(
        json.dumps(
            {
                "segments": len(ordered_segments),
                "segments_skipped_no_lexical_content": len(skipped_segments),
                "segment_decode_failures": failed_count,
                "raw_words": raw_word_count,
                "spoken_retries": retry_count,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
