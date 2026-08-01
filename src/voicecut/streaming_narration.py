#!/usr/bin/env python3
"""Streaming semantic planner for unscripted narration.

This module consumes an existing word-level Whisper transcript and produces a
source-grounded semantic edit plan.  It never reads, renders, or modifies an
audio waveform.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from functools import lru_cache
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
import unicodedata

from .common import read_json, sha256_file, tokenize, write_json
from .planner_backends import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    PlannerBackend as NarrationPlannerBackend,
    add_planner_backend_arguments,
    create_planner_backend,
)


DEFAULT_WINDOW_SECONDS = 30.0


class StreamingPlanError(RuntimeError):
    """The planner could not produce a safe, valid commitment."""


class DecisionValidationError(ValueError):
    """A model response violates the streaming-plan contract."""


class SourceGroundingValidationError(DecisionValidationError):
    """Canonical text contains material outside its declared source range."""

    def __init__(self, message: str, *, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class TranscriptWord:
    id: int
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class SourceRange:
    start_word_id: int
    end_word_id: int
    first_word_id: int
    last_word_id: int
    first_word: str
    last_word: str
    canonical_text: str
    canonical_token_count: int
    supported_token_count: int
    token_support: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FinalizedThought:
    canonical_text: str
    source_ranges: tuple[SourceRange, ...]
    grounding_validation: dict[str, Any]


@dataclass(frozen=True)
class PlannerDecision:
    finalized: tuple[FinalizedThought, ...]
    pending_start_word_id: int | None
    pending_reason: str


def streaming_response_schema() -> dict[str, Any]:
    """JSON schema passed to structured-output model backends."""

    source_range = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "first_word_id",
            "last_word_id",
            "first_word",
            "last_word",
            "canonical_text",
        ],
        "properties": {
            "first_word_id": {
                "type": "integer",
                "description": "Inclusive ID of the first retained source word.",
            },
            "last_word_id": {
                "type": "integer",
                "description": (
                    "Inclusive ID of the final retained source word. This word "
                    "is part of the selected range."
                ),
            },
            "first_word": {
                "type": "string",
                "description": (
                    "Exact boundary word copied from first_word_id, apart from "
                    "punctuation or capitalization."
                ),
            },
            "last_word": {
                "type": "string",
                "description": (
                    "Exact boundary word copied from last_word_id, apart from "
                    "punctuation or capitalization."
                ),
            },
            "canonical_text": {
                "type": "string",
                "description": (
                    "Only the canonical phrase acoustically supported by this "
                    "specific inclusive source range."
                ),
            },
        },
    }
    finalized_thought = {
        "type": "object",
        "additionalProperties": False,
        "required": ["canonical_text", "source_ranges"],
        "properties": {
            "canonical_text": {
                "type": "string",
                "description": (
                    "The intended completed thought represented faithfully by "
                    "the selected source occurrences."
                ),
            },
            "source_ranges": {
                "type": "array",
                "minItems": 1,
                "items": source_range,
                "description": (
                    "Ordered, non-overlapping inclusive source-word ranges. "
                    "Every range includes both first_word_id and last_word_id. "
                    "Prefer one contiguous complete take."
                ),
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "finalized",
            "pending_start_word_id",
            "pending_reason",
        ],
        "properties": {
            "finalized": {
                "type": "array",
                "items": finalized_thought,
            },
            "pending_start_word_id": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "null"},
                ],
                "description": (
                    "First source word still pending. Must be non-null on an "
                    "incremental call and null on the final EOF call."
                ),
            },
            "pending_reason": {
                "type": "string",
                "description": (
                    "Why the suffix remains pending, or why no pending suffix "
                    "remains on the final call."
                ),
            },
        },
    }


def load_transcript_words(
    transcript_data: dict[str, Any],
) -> list[TranscriptWord]:
    """Flatten Whisper occurrences without treating chunks as edit units."""

    raw_occurrences: list[dict[str, Any]] = []
    atoms = transcript_data.get("atoms")
    if isinstance(atoms, list):
        ordered_atoms = [atom for atom in atoms if isinstance(atom, dict)]
        ordered_atoms.sort(key=lambda atom: int(atom.get("atom_index", -1)))
        for atom in ordered_atoms:
            raw_words = atom.get("words", [])
            if isinstance(raw_words, list):
                raw_occurrences.extend(
                    word for word in raw_words if isinstance(word, dict)
                )
    if not raw_occurrences:
        whole = transcript_data.get("whole")
        if isinstance(whole, dict) and isinstance(whole.get("words"), list):
            raw_occurrences.extend(
                word for word in whole["words"] if isinstance(word, dict)
            )

    words: list[TranscriptWord] = []
    for raw in raw_occurrences:
        text = str(raw.get("word", raw.get("text", ""))).strip()
        if not text:
            continue
        if "start" not in raw or "end" not in raw:
            raise ValueError("every Whisper word requires start and end")
        start = float(raw["start"])
        end = float(raw["end"])
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError(f"invalid timestamp geometry for transcript word {text!r}")
        words.append(
            TranscriptWord(
                id=len(words),
                text=text,
                start=start,
                end=end,
            )
        )
    if not words:
        raise ValueError("the Whisper transcript contains no timed words")
    return words


def _word_payload(words: Sequence[TranscriptWord]) -> list[dict[str, Any]]:
    return [asdict(word) for word in words]


def _word_text(words: Sequence[TranscriptWord]) -> str:
    return " ".join(word.text for word in words).strip()


def _debug_word_text(words: Sequence[TranscriptWord]) -> str:
    return "\n".join(
        f"[{word.id}] {word.start:.2f}-{word.end:.2f} {word.text}" for word in words
    )


def _normalized_anchor(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", text).casefold()
        if not unicodedata.combining(character) and character.isalnum()
    )


def _compact_canonical_text(text: str) -> str:
    """Normalize punctuation, case, whitespace, and joined/split forms."""

    return _normalized_anchor(text)


def _phonetic_key(text: str) -> str:
    compact = "".join(character for character in text if character.isalpha())
    if not compact:
        return ""
    compact = (
        compact.replace("ph", "f")
        .replace("ck", "k")
        .replace("qu", "k")
        .replace("x", "ks")
    )
    first = compact[0]
    tail = "".join(character for character in compact[1:] if character not in "aeiouy")
    collapsed = [first]
    for character in tail:
        if character != collapsed[-1]:
            collapsed.append(character)
    return "".join(collapsed)


def _token_group_similarity(
    canonical_tokens: Sequence[str],
    source_tokens: Sequence[str],
) -> float | None:
    canonical = "".join(canonical_tokens)
    source = "".join(source_tokens)
    if not canonical or not source:
        return None
    if canonical == source:
        return 1.0
    lexical = SequenceMatcher(None, canonical, source).ratio()
    if canonical[0] == source[0] and lexical >= 0.78:
        return lexical
    canonical_phonetic = _phonetic_key(canonical)
    source_phonetic = _phonetic_key(source)
    if (
        canonical[0] == source[0]
        and len(canonical) >= 3
        and len(source) >= 3
        and canonical_phonetic
        and source_phonetic
        and SequenceMatcher(
            None,
            canonical_phonetic,
            source_phonetic,
        ).ratio()
        >= 0.80
    ):
        return max(0.70, lexical)
    return None


def _source_token_ledger(
    words: Sequence[TranscriptWord],
) -> tuple[list[str], list[int]]:
    source_tokens: list[str] = []
    source_owners: list[int] = []
    for word in words:
        for token in tokenize(word.text):
            source_tokens.append(token)
            source_owners.append(word.id)
    return source_tokens, source_owners


def _first_unsupported_token(
    canonical_tokens: Sequence[str],
    source_tokens: Sequence[str],
) -> str:
    source_cursor = 0
    for canonical_index, canonical_token in enumerate(canonical_tokens):
        found = False
        source_starts = (
            range(source_cursor, len(source_tokens))
            if canonical_index > 0
            else range(source_cursor, min(source_cursor + 1, len(source_tokens)))
        )
        for source_start in source_starts:
            for source_count in (1, 2):
                source_end = source_start + source_count
                if source_end > len(source_tokens):
                    continue
                if (
                    _token_group_similarity(
                        [canonical_token],
                        source_tokens[source_start:source_end],
                    )
                    is not None
                ):
                    source_cursor = source_end
                    found = True
                    break
            if found:
                break
        if not found:
            return canonical_token
    return canonical_tokens[-1] if canonical_tokens else ""


def _align_range_canonical_text(
    *,
    range_words: Sequence[TranscriptWord],
    canonical_text: str,
) -> tuple[tuple[dict[str, Any], ...], list[str]]:
    """Monotonically support every canonical token inside one fixed range."""

    canonical_tokens = tokenize(canonical_text)
    source_tokens, source_owners = _source_token_ledger(range_words)
    if not canonical_tokens:
        raise DecisionValidationError("range canonical_text has no spoken tokens")
    if not source_tokens:
        raise DecisionValidationError("selected source range has no spoken tokens")

    @lru_cache(maxsize=None)
    def solve(
        canonical_index: int,
        source_index: int,
    ) -> tuple[float, tuple[tuple[int, int, int, int, float], ...]] | None:
        if canonical_index == len(canonical_tokens):
            return (0.0, ()) if source_index == len(source_tokens) else None
        if source_index == len(source_tokens):
            return None
        best: (
            tuple[
                float,
                tuple[tuple[int, int, int, int, float], ...],
            ]
            | None
        ) = None

        # Internal Whisper filler or duplicated source tokens may remain
        # unmentioned, but canonical text must start and finish at the actual
        # range edges. Therefore the first source token cannot be skipped.
        if canonical_index > 0:
            skipped = solve(canonical_index, source_index + 1)
            if skipped is not None:
                best = (skipped[0] - 0.35, skipped[1])

        for canonical_count in (1, 2):
            canonical_end = canonical_index + canonical_count
            if canonical_end > len(canonical_tokens):
                continue
            for source_count in (1, 2):
                source_end = source_index + source_count
                if source_end > len(source_tokens):
                    continue
                similarity = _token_group_similarity(
                    canonical_tokens[canonical_index:canonical_end],
                    source_tokens[source_index:source_end],
                )
                if similarity is None:
                    continue
                remaining = solve(canonical_end, source_end)
                if remaining is None:
                    continue
                group_penalty = 0.04 * abs(canonical_count - source_count)
                candidate = (
                    remaining[0] + similarity * canonical_count - group_penalty,
                    (
                        (
                            canonical_index,
                            canonical_end,
                            source_index,
                            source_end,
                            similarity,
                        ),
                        *remaining[1],
                    ),
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate
        return best

    aligned = solve(0, 0)
    if aligned is None:
        unsupported = _first_unsupported_token(
            canonical_tokens,
            source_tokens,
        )
        return (), [unsupported]

    mappings: list[dict[str, Any]] = []
    supported_indices: set[int] = set()
    for (
        canonical_start,
        canonical_end,
        source_start,
        source_end,
        similarity,
    ) in aligned[1]:
        owner_ids = sorted(set(source_owners[source_start:source_end]))
        for canonical_index in range(canonical_start, canonical_end):
            supported_indices.add(canonical_index)
            mappings.append(
                {
                    "canonical_token_index": canonical_index,
                    "canonical_token": canonical_tokens[canonical_index],
                    "source_token_indices": list(range(source_start, source_end)),
                    "source_word_ids": owner_ids,
                    "source_tokens": source_tokens[source_start:source_end],
                    "similarity": similarity,
                }
            )
    unsupported_tokens = [
        token
        for index, token in enumerate(canonical_tokens)
        if index not in supported_indices
    ]
    return tuple(mappings), unsupported_tokens


def _nearby_source_words(
    pending_words: Sequence[TranscriptWord],
    *,
    last_word_id: int,
) -> list[TranscriptWord]:
    word_by_id = {word.id: word for word in pending_words}
    return [
        word_by_id[word_id]
        for word_id in range(last_word_id, last_word_id + 3)
        if word_id in word_by_id
    ]


def _unsupported_token_error(
    *,
    unsupported_token: str,
    start_word_id: int,
    end_word_id: int,
    range_words: Sequence[TranscriptWord],
    pending_words: Sequence[TranscriptWord],
    thought_index: int,
    range_index: int,
) -> SourceGroundingValidationError:
    final_source_word = range_words[-1]
    word_by_id = {word.id: word for word in pending_words}
    outside = word_by_id.get(end_word_id)
    outside_sentence = ""
    if (
        outside is not None
        and _token_group_similarity(
            [unsupported_token],
            tokenize(outside.text),
        )
        is not None
    ):
        outside_sentence = (
            f'; source word {outside.id} "{outside.text}" is outside the selected range'
        )
    nearby = _nearby_source_words(
        pending_words,
        last_word_id=final_source_word.id,
    )
    nearby_text = "; ".join(f"word {word.id}: {word.text}" for word in nearby)
    message = (
        f'Unsupported canonical token "{unsupported_token}": range '
        f"[{start_word_id}, {end_word_id}) ends at source word "
        f'"{final_source_word.text}"{outside_sentence}. Nearby source words: '
        f"{nearby_text}."
    )
    return SourceGroundingValidationError(
        message,
        report={
            "thought_index": thought_index,
            "range_index": range_index,
            "start_word_id": start_word_id,
            "end_word_id": end_word_id,
            "unsupported_canonical_token": unsupported_token,
            "selected_first_word": range_words[0].text,
            "selected_last_word": final_source_word.text,
            "nearby_source_words": [
                {"id": word.id, "text": word.text} for word in nearby
            ],
            "status": "invalid",
        },
    )


def _prompt(
    *,
    iteration: int,
    pending_words: Sequence[TranscriptWord],
    committed_thoughts: Sequence[dict[str, Any]],
    final_pass: bool,
) -> str:
    committed_context = [
        str(thought["canonical_text"]) for thought in committed_thoughts[-4:]
    ]
    phase = (
        "FINAL EOF PASS. Finalize every intended thought that remains and "
        "return a null pending_start_word_id."
        if final_pass
        else (
            "INCREMENTAL PASS. Keep the last currently visible thought "
            "pending, even if it sounds grammatically complete. Finalize a "
            "thought only when a distinct subsequent thought provides clear "
            "evidence that the speaker moved on."
        )
    )
    return f"""Plan a narration edit from exact Whisper word occurrences.

{phase}

The pending input is chronological source evidence, not a script. It may
contain false starts, abandoned phrases, filler, repeated takes, corrections,
or recording directions.

CORE TASK:
- Reconstruct the one clean narration a human editor would deliver.
- The finalized array contains intended thoughts, NOT every utterance,
  punctuation-delimited fragment, or attempted sentence in the recording.
- Treat nearby utterances with the same topic, beginning, ending, or semantic
  purpose as COMPETING TAKES. Keep only the final successful take. A fragment
  can be grammatical and still be an abandoned attempt.
- Never preserve alternative phrasings, self-corrections, duplicated clauses,
  or partial attempts merely because each span has valid source words.
- Whisper punctuation is unreliable and does not prove a thought boundary.
- If uncertain whether visible words are an attempt or the final take, keep
  them pending. It is safer to delay than to commit competing attempts.
- Work in this order before writing JSON: (1) group nearby competing attempts
  by semantic purpose, (2) reconstruct the one clean intended narration,
  (3) choose the longest source spans supporting it, and only then (4) assign
  inclusive boundary IDs. Do not turn each ASR fragment into its own thought.
- A repeated phrase after a pause is normally a retry or continuation, not a
  new thought, when its subject and purpose are unchanged.
- If the pending suffix begins by repeating a phrase selected in finalized,
  the earlier take was finalized too soon. Keep that thought pending and let
  the later take replace it.
- If the proposed pending suffix begins with a dependent continuation such as
  "that", "which", or "because", keep the preceding clause pending with it.

Retake example:
  source: "I want to describe... I want to explain pruning. I want to explain
  how pruning works. First, consider a weight matrix."
  correct: finalize only "I want to explain how pruning works."
  pending: "First, consider a weight matrix."
  wrong: finalize any earlier attempt as a separate thought.

Source-range selection priorities, in strict order:
1. Correctness: keep all and only the intended narration. Remove abandoned
   attempts, repetitions, incorrect replacements, filler, and directions.
2. Minimum necessary cuts: among equally correct selections, use the fewest
   discontinuous source ranges.
3. Prefer one long contiguous complete take. Use multiple ranges only when a
   unique valid prefix must be joined to a later completion.
4. Never keep unwanted speech merely to reduce the number of cuts.
5. Every discontinuity must remove something that genuinely cannot remain.

Grounding and commitment rules:
- Select the final successful take for each intended thought.
- A later full retake replaces earlier attempts at the same thought.
- Preserve unique coherent prefixes and qualifiers that were not replaced.
- Prefer one contiguous complete take whenever possible.
- Several source ranges are allowed only for a genuine preserved prefix plus
  a later completion. Never assemble a sentence from arbitrary isolated words.
- canonical_text may correct obvious ASR spelling, punctuation, and casing,
  but must not add a spoken content word unsupported by its source_ranges.
- Every finalized item needs at least one exact source range.
- Every source range uses first_word_id and last_word_id. BOTH IDs ARE
  INCLUSIVE: the words at both IDs are retained.
- Copy first_word and last_word from those exact source occurrences. They are
  lexical boundary anchors, not descriptions.
- Every source range must provide its own canonical_text containing only the
  canonical phrase supported by that range.
- canonical_text must also represent every spoken source word inside its
  range. It cannot hide filler, repetitions, abandoned words, or corrections
  that would remain audible. Split the range around unwanted occurrences.
- The thought-level canonical_text must be exactly the concatenation of its
  range-level canonical_text values, allowing only punctuation, capitalization,
  whitespace, and joined/split spelling differences.
- Use only existing word IDs from PENDING WORDS.
- Finalized ranges must be chronological, ordered, non-overlapping, and before
  pending_start_word_id.
- On an incremental pass, pending_start_word_id must identify the earliest
  source word still needed for the final visible thought. Everything from that
  word through the visible end will be reconsidered with the next look-ahead.
- Do not keep an earlier failed attempt pending when a later, more complete
  attempt is already visible.

Iteration: {iteration}
Recently committed read-only context:
{json.dumps(committed_context, ensure_ascii=False)}

PENDING WORDS:
{json.dumps(_word_payload(pending_words), ensure_ascii=False)}
"""


def _retry_prompt(
    original_prompt: str,
    *,
    invalid_raw: str,
    validation_error: str,
) -> str:
    return f"""Your previous response failed local validation.

VALIDATION ERROR:
{validation_error}

PREVIOUS INVALID RESPONSE:
{invalid_raw}

Correct the response for the exact same pending words. Do not change or invent
source IDs. Rebuild the decision from the source evidence instead of minimally
patching the invalid JSON.

CORRECTED SOURCE-RANGE CONTRACT:
- first_word_id and last_word_id are both INCLUSIVE.
- first_word and last_word must name those exact source occurrences.
- Every range must have first_word_id <= last_word_id.
- Across the whole response, each range's first_word_id must be greater than
  the previous range's last_word_id. Never overlap, repeat, or move backward.
- canonical_text inside each range may contain only words supported inside
  that inclusive range.
- canonical_text must also account for every spoken source word inside the
  range. If unwanted audio lies inside a range, split the range around it.
- If an unsupported canonical word appears immediately after last_word_id,
  correct last_word_id and last_word so the range includes it.
- Do not remove the unsupported word from canonical text merely to satisfy the
  validator unless the source audio genuinely does not contain that intended
  word.
- Do not retain abandoned attempts or repeated alternative takes as separate
  finalized thoughts. Keep only the final successful version of each thought.

ORIGINAL REQUEST:
{original_prompt}
"""


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise DecisionValidationError(
            f"{label} keys must be {sorted(expected)}; got {sorted(actual)}"
        )


def validate_decision(
    raw: str,
    *,
    pending_words: Sequence[TranscriptWord],
    final_pass: bool,
    committed_source_end: int,
) -> PlannerDecision:
    """Parse and enforce source/commitment invariants."""

    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError as error:
        raise DecisionValidationError(f"invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise DecisionValidationError("response must be a JSON object")
    _require_exact_keys(
        value,
        {"finalized", "pending_start_word_id", "pending_reason"},
        label="response",
    )
    if not pending_words:
        raise DecisionValidationError("pending input must not be empty")
    pending_ids = {word.id for word in pending_words}
    pending_word_by_id = {word.id: word for word in pending_words}
    pending_first = pending_words[0].id
    visible_end = pending_words[-1].id + 1
    if pending_ids != set(range(pending_first, visible_end)):
        raise DecisionValidationError(
            "pending input word IDs must form one contiguous source interval"
        )
    if pending_first < committed_source_end:
        raise DecisionValidationError(
            "pending input moves backward behind committed source"
        )

    raw_finalized = value["finalized"]
    if not isinstance(raw_finalized, list):
        raise DecisionValidationError("finalized must be a list")
    raw_pending_start = value["pending_start_word_id"]
    if (
        not final_pass
        and type(raw_pending_start) is int
        and raw_pending_start in pending_ids
    ):
        pending_prefix_tokens = tokenize(
            _word_text(
                [
                    pending_word_by_id[word_id]
                    for word_id in range(
                        raw_pending_start,
                        min(visible_end, raw_pending_start + 24),
                    )
                ]
            )
        )
        for thought_index, raw_thought in enumerate(raw_finalized):
            canonical_value = (
                raw_thought.get("canonical_text")
                if isinstance(raw_thought, dict)
                else None
            )
            if not isinstance(canonical_value, str):
                continue
            finalized_tokens = tokenize(canonical_value)
            match = SequenceMatcher(
                None,
                finalized_tokens,
                pending_prefix_tokens,
                autojunk=False,
            ).find_longest_match()
            if match.size >= 5:
                repeated_phrase = " ".join(
                    pending_prefix_tokens[match.b : match.b + match.size]
                )
                raise DecisionValidationError(
                    f'pending suffix repeats finalized phrase "{repeated_phrase}" '
                    f"in thought {thought_index}; keep the competing thought "
                    "pending and select only its later successful take"
                )
    finalized: list[FinalizedThought] = []
    prior_end = committed_source_end
    for thought_index, raw_thought in enumerate(raw_finalized):
        if not isinstance(raw_thought, dict):
            raise DecisionValidationError(
                f"finalized[{thought_index}] must be an object"
            )
        _require_exact_keys(
            raw_thought,
            {"canonical_text", "source_ranges"},
            label=f"finalized[{thought_index}]",
        )
        canonical_text = raw_thought["canonical_text"]
        if not isinstance(canonical_text, str) or not canonical_text.strip():
            raise DecisionValidationError(
                f"finalized[{thought_index}] has no canonical text"
            )
        raw_ranges = raw_thought["source_ranges"]
        if not isinstance(raw_ranges, list) or not raw_ranges:
            raise DecisionValidationError(
                f"finalized[{thought_index}] text has no source ranges"
            )
        ranges: list[SourceRange] = []
        range_reports: list[dict[str, Any]] = []
        for range_index, raw_range in enumerate(raw_ranges):
            if not isinstance(raw_range, dict):
                raise DecisionValidationError(
                    f"finalized[{thought_index}].source_ranges"
                    f"[{range_index}] must be an object"
                )
            _require_exact_keys(
                raw_range,
                {
                    "first_word_id",
                    "last_word_id",
                    "first_word",
                    "last_word",
                    "canonical_text",
                },
                label=(f"finalized[{thought_index}].source_ranges[{range_index}]"),
            )
            first_word_id = raw_range["first_word_id"]
            last_word_id = raw_range["last_word_id"]
            if type(first_word_id) is not int or type(last_word_id) is not int:
                raise DecisionValidationError("source range IDs must be integers")
            if first_word_id not in pending_ids:
                raise DecisionValidationError(
                    f"source first_word_id {first_word_id} is not in pending input"
                )
            if last_word_id not in pending_ids:
                raise DecisionValidationError(
                    f"source last_word_id {last_word_id} is not in pending input"
                )
            if first_word_id > last_word_id:
                raise DecisionValidationError(
                    "inclusive source range has first_word_id after last_word_id"
                )
            start = first_word_id
            end = last_word_id + 1
            if start < prior_end:
                raise DecisionValidationError("source ranges overlap or move backward")
            first_word = raw_range["first_word"]
            last_word = raw_range["last_word"]
            range_canonical_text = raw_range["canonical_text"]
            for name, text in (
                ("first_word", first_word),
                ("last_word", last_word),
                ("canonical_text", range_canonical_text),
            ):
                if not isinstance(text, str) or not text.strip():
                    raise DecisionValidationError(
                        f"source range {name} must be non-empty text"
                    )
            expected_first = pending_word_by_id[first_word_id].text
            expected_last = pending_word_by_id[last_word_id].text
            if not _normalized_anchor(first_word) or _normalized_anchor(
                first_word
            ) != _normalized_anchor(expected_first):
                raise DecisionValidationError(
                    f'first_word anchor "{first_word}" does not match source '
                    f'word {first_word_id} "{expected_first}"'
                )
            if not _normalized_anchor(last_word) or _normalized_anchor(
                last_word
            ) != _normalized_anchor(expected_last):
                raise DecisionValidationError(
                    f'last_word anchor "{last_word}" does not match source '
                    f'word {last_word_id} "{expected_last}"'
                )
            range_words = [pending_word_by_id[word_id] for word_id in range(start, end)]
            token_support, unsupported_tokens = _align_range_canonical_text(
                range_words=range_words,
                canonical_text=range_canonical_text.strip(),
            )
            if unsupported_tokens:
                raise _unsupported_token_error(
                    unsupported_token=unsupported_tokens[0],
                    start_word_id=start,
                    end_word_id=end,
                    range_words=range_words,
                    pending_words=pending_words,
                    thought_index=thought_index,
                    range_index=range_index,
                )
            canonical_token_count = len(tokenize(range_canonical_text))
            supported_token_count = len(
                {int(item["canonical_token_index"]) for item in token_support}
            )
            if supported_token_count != canonical_token_count:
                missing = next(
                    (
                        token
                        for index, token in enumerate(tokenize(range_canonical_text))
                        if index
                        not in {
                            int(item["canonical_token_index"]) for item in token_support
                        }
                    ),
                    "",
                )
                raise _unsupported_token_error(
                    unsupported_token=missing,
                    start_word_id=start,
                    end_word_id=end,
                    range_words=range_words,
                    pending_words=pending_words,
                    thought_index=thought_index,
                    range_index=range_index,
                )
            source_tokens, source_owners = _source_token_ledger(range_words)
            represented_source_indices = {
                int(source_index)
                for item in token_support
                for source_index in item["source_token_indices"]
            }
            unrepresented_source = [
                {
                    "source_token_index": source_index,
                    "source_word_id": source_owners[source_index],
                    "source_token": source_token,
                }
                for source_index, source_token in enumerate(source_tokens)
                if source_index not in represented_source_indices
            ]
            if unrepresented_source:
                first_unrepresented = unrepresented_source[0]
                source_word_id = int(first_unrepresented["source_word_id"])
                source_word = pending_word_by_id[source_word_id].text
                raise SourceGroundingValidationError(
                    "Unrepresented selected source token "
                    f'"{first_unrepresented["source_token"]}": range '
                    f"[{start}, {end}) includes source word {source_word_id} "
                    f'"{source_word}", but range canonical_text omits it. '
                    "Split the source range so unwanted audible speech is "
                    "excluded instead of hiding it in canonical text.",
                    report={
                        "thought_index": thought_index,
                        "range_index": range_index,
                        "start_word_id": start,
                        "end_word_id": end,
                        "unrepresented_source_tokens": unrepresented_source,
                        "selected_first_word": range_words[0].text,
                        "selected_last_word": range_words[-1].text,
                        "status": "invalid",
                    },
                )
            ranges.append(
                SourceRange(
                    start_word_id=start,
                    end_word_id=end,
                    first_word_id=first_word_id,
                    last_word_id=last_word_id,
                    first_word=first_word.strip(),
                    last_word=last_word.strip(),
                    canonical_text=range_canonical_text.strip(),
                    canonical_token_count=canonical_token_count,
                    supported_token_count=supported_token_count,
                    token_support=token_support,
                )
            )
            range_reports.append(
                {
                    "range_index": range_index,
                    "first_word_id": first_word_id,
                    "last_word_id": last_word_id,
                    "start_word_id": start,
                    "end_word_id": end,
                    "first_word": expected_first,
                    "last_word": expected_last,
                    "canonical_text": range_canonical_text.strip(),
                    "canonical_tokens": canonical_token_count,
                    "supported_tokens": supported_token_count,
                    "source_tokens": len(source_tokens),
                    "represented_source_tokens": len(represented_source_indices),
                    "unrepresented_source_tokens": [],
                    "unsupported_tokens": [],
                    "status": "valid",
                }
            )
            prior_end = end
        joined_range_text = " ".join(
            source_range.canonical_text for source_range in ranges
        )
        if _compact_canonical_text(canonical_text) != _compact_canonical_text(
            joined_range_text
        ):
            raise DecisionValidationError(
                f"finalized[{thought_index}].canonical_text does not equal "
                "the concatenation of its range canonical_text values after "
                "punctuation, whitespace, capitalization, and joined/split "
                "normalization"
            )
        thought_token_count = len(tokenize(canonical_text))
        finalized.append(
            FinalizedThought(
                canonical_text=canonical_text.strip(),
                source_ranges=tuple(ranges),
                grounding_validation={
                    "thought_index": thought_index,
                    "canonical_tokens": thought_token_count,
                    "supported_tokens": thought_token_count,
                    "unsupported_tokens": [],
                    "status": "valid",
                    "source_ranges": range_reports,
                },
            )
        )

    pending_start = value["pending_start_word_id"]
    if final_pass:
        if pending_start is not None:
            raise DecisionValidationError(
                "final EOF call must return pending_start_word_id=null"
            )
    else:
        if type(pending_start) is not int or pending_start not in pending_ids:
            raise DecisionValidationError(
                "incremental call requires an existing pending_start_word_id"
            )
        if pending_start < prior_end:
            raise DecisionValidationError(
                "finalized source ranges must end before pending_start_word_id"
            )
    pending_reason = value["pending_reason"]
    if not isinstance(pending_reason, str) or not pending_reason.strip():
        raise DecisionValidationError("pending_reason must be non-empty text")
    return PlannerDecision(
        finalized=tuple(finalized),
        pending_start_word_id=pending_start,
        pending_reason=pending_reason.strip(),
    )


def _serialize_decision(decision: PlannerDecision) -> dict[str, Any]:
    return {
        "finalized": [
            {
                "canonical_text": thought.canonical_text,
                "source_ranges": [
                    {
                        "start_word_id": source_range.start_word_id,
                        "end_word_id": source_range.end_word_id,
                        "first_word_id": source_range.first_word_id,
                        "last_word_id": source_range.last_word_id,
                        "first_word": source_range.first_word,
                        "last_word": source_range.last_word,
                        "canonical_text": source_range.canonical_text,
                    }
                    for source_range in thought.source_ranges
                ],
                "grounding_validation": thought.grounding_validation,
            }
            for thought in decision.finalized
        ],
        "pending_start_word_id": decision.pending_start_word_id,
        "pending_reason": decision.pending_reason,
    }


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _request_validated_decision(
    *,
    backend: NarrationPlannerBackend,
    prompt: str,
    pending_words: Sequence[TranscriptWord],
    final_pass: bool,
    committed_source_end: int,
    iteration: int,
    output_dir: Path,
    decision_validator: Callable[[PlannerDecision], None] | None = None,
) -> tuple[PlannerDecision, list[dict[str, Any]]]:
    schema = streaming_response_schema()
    attempts: list[dict[str, Any]] = []
    request_prompt = prompt
    prior_raw = ""
    prior_error = ""
    for attempt in range(1, 3):
        raw = ""
        try:
            raw = backend.generate(
                request_prompt,
                response_schema=schema,
                request_id=f"iteration-{iteration:04d}-attempt-{attempt}",
            )
            decision = validate_decision(
                raw,
                pending_words=pending_words,
                final_pass=final_pass,
                committed_source_end=committed_source_end,
            )
            if decision_validator is not None:
                decision_validator(decision)
            attempts.append(
                {
                    "attempt": attempt,
                    "raw_response": raw,
                    "validation_error": None,
                }
            )
            (
                output_dir / f"iteration_{iteration:04d}_attempt_{attempt}.raw.json"
            ).write_text(
                raw,
                encoding="utf-8",
            )
            return decision, attempts
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            grounding_failure = (
                error.report
                if isinstance(error, SourceGroundingValidationError)
                else None
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "raw_response": raw,
                    "validation_error": error_text,
                    "grounding_failure": grounding_failure,
                }
            )
            (
                output_dir / f"iteration_{iteration:04d}_attempt_{attempt}.raw.txt"
            ).write_text(
                raw,
                encoding="utf-8",
            )
            prior_raw = raw
            prior_error = error_text
            if attempt == 1:
                request_prompt = _retry_prompt(
                    prompt,
                    invalid_raw=prior_raw,
                    validation_error=prior_error,
                )
                continue
    failure = {
        "iteration": iteration,
        "final_pass": final_pass,
        "validation_error": prior_error,
        "raw_response": prior_raw,
        "attempts": attempts,
        "grounding_failure": next(
            (
                attempt["grounding_failure"]
                for attempt in reversed(attempts)
                if attempt.get("grounding_failure") is not None
            ),
            None,
        ),
    }
    write_json(output_dir / f"iteration_{iteration:04d}_failure.json", failure)
    raise StreamingPlanError(
        f"streaming planner failed iteration {iteration} after one retry: "
        f"{prior_error}; raw responses were saved in {output_dir}"
    )


def _exact_source_passthrough_thought(
    words: Sequence[TranscriptWord],
) -> FinalizedThought:
    """Create a contiguous thought whose text is copied exactly from source.

    This deliberately does not parse any model output.  Its grounding ledger
    is constructed from the source-token ledger itself, so malformed or
    unsupported model text can never leak into a delivery fallback.
    """

    if not words:
        raise StreamingPlanError("source passthrough requires at least one word")
    for left, right in zip(words, words[1:], strict=False):
        if right.id != left.id + 1:
            raise StreamingPlanError(
                "source passthrough words must form one contiguous interval"
            )

    canonical_text = _word_text(words)
    source_tokens, source_owners = _source_token_ledger(words)
    token_support = tuple(
        {
            "canonical_token_index": token_index,
            "canonical_token": token,
            "source_token_indices": [token_index],
            "source_word_ids": [source_owners[token_index]],
            "source_tokens": [token],
            "similarity": 1.0,
        }
        for token_index, token in enumerate(source_tokens)
    )
    source_range = SourceRange(
        start_word_id=words[0].id,
        end_word_id=words[-1].id + 1,
        first_word_id=words[0].id,
        last_word_id=words[-1].id,
        first_word=words[0].text,
        last_word=words[-1].text,
        canonical_text=canonical_text,
        canonical_token_count=len(source_tokens),
        supported_token_count=len(source_tokens),
        token_support=token_support,
    )
    return FinalizedThought(
        canonical_text=canonical_text,
        source_ranges=(source_range,),
        grounding_validation={
            "thought_index": 0,
            "canonical_tokens": len(source_tokens),
            "supported_tokens": len(source_tokens),
            "unsupported_tokens": [],
            "status": "valid",
            "grounding_mode": "deterministic_exact_source_passthrough",
            "source_ranges": [
                {
                    "range_index": 0,
                    "first_word_id": words[0].id,
                    "last_word_id": words[-1].id,
                    "start_word_id": words[0].id,
                    "end_word_id": words[-1].id + 1,
                    "first_word": words[0].text,
                    "last_word": words[-1].text,
                    "canonical_text": canonical_text,
                    "canonical_tokens": len(source_tokens),
                    "supported_tokens": len(source_tokens),
                    "source_tokens": len(source_tokens),
                    "represented_source_tokens": len(source_tokens),
                    "unrepresented_source_tokens": [],
                    "unsupported_tokens": [],
                    "status": "valid",
                    "grounding_mode": "deterministic_exact_source_passthrough",
                }
            ],
        },
    )


def _source_passthrough_decision(
    *,
    pending_words: Sequence[TranscriptWord],
    final_pass: bool,
    lookahead_start_word_id: int | None,
    window_seconds: float,
    iteration: int,
    attempts: Sequence[dict[str, Any]],
    trigger: str,
    reason: str,
    grounding_failure: dict[str, Any] | None,
) -> tuple[PlannerDecision, list[dict[str, Any]], dict[str, Any]]:
    """Preserve exact source while retaining one new look-ahead window."""

    if not pending_words:
        raise StreamingPlanError("source passthrough has no pending words")

    if final_pass:
        passthrough_words = list(pending_words)
        pending_start_word_id = None
        status = "eof_source_passthrough"
        pending_reason = (
            "The planner did not safely resolve the EOF input; every remaining "
            "source word was preserved exactly."
        )
    else:
        if lookahead_start_word_id is None:
            raise StreamingPlanError(
                "incremental source passthrough requires a look-ahead boundary"
            )
        older_words = [
            word for word in pending_words if word.id < lookahead_start_word_id
        ]
        # Commit at most one ordinary window.  The complete newly read window
        # remains pending so a later request can still resolve its final thought.
        prefix_end = (
            _take_new_words(older_words, 0, window_seconds=window_seconds)
            if older_words
            else 0
        )
        passthrough_words = older_words[:prefix_end]
        pending_start_word_id = (
            passthrough_words[-1].id + 1 if passthrough_words else pending_words[0].id
        )
        status = (
            "source_passthrough"
            if passthrough_words
            else "source_passthrough_deferred_for_lookahead"
        )
        if trigger == "request_failure":
            pending_reason = (
                "The planner exhausted both attempts. A bounded older prefix was "
                "preserved exactly while the newest look-ahead window remains pending."
                if passthrough_words
                else (
                    "The planner exhausted both attempts in the first visible "
                    "window; the complete window remains pending for additional "
                    "look-ahead."
                )
            )
        else:
            pending_reason = (
                "The planner repeatedly made no chronological progress. A bounded "
                "older prefix was preserved exactly while the newest look-ahead "
                "window remains pending."
            )

    finalized = (
        (_exact_source_passthrough_thought(passthrough_words),)
        if passthrough_words
        else ()
    )
    decision = PlannerDecision(
        finalized=finalized,
        pending_start_word_id=pending_start_word_id,
        pending_reason=pending_reason,
    )
    selected_ranges = (
        [
            {
                "start_word_id": passthrough_words[0].id,
                "end_word_id": passthrough_words[-1].id + 1,
            }
        ]
        if passthrough_words
        else []
    )
    remaining_words = (
        []
        if pending_start_word_id is None
        else [word for word in pending_words if word.id >= pending_start_word_id]
    )
    fallback = {
        "iteration": iteration,
        "status": status,
        "trigger": trigger,
        "reason": reason,
        "final_pass": final_pass,
        "source_ranges": selected_ranges,
        "passthrough_word_ids": [word.id for word in passthrough_words],
        "remaining_pending_range": (
            {
                "start_word_id": remaining_words[0].id,
                "end_word_id": remaining_words[-1].id + 1,
            }
            if remaining_words
            else None
        ),
        "grounding_mode": "deterministic_exact_source_passthrough",
        "rejected_model_output_accepted": False,
        "grounding_failure": grounding_failure,
    }
    return decision, list(attempts), fallback


def _passthrough_after_request_failure(
    *,
    pending_words: Sequence[TranscriptWord],
    final_pass: bool,
    lookahead_start_word_id: int | None,
    window_seconds: float,
    iteration: int,
    output_dir: Path,
) -> tuple[PlannerDecision, list[dict[str, Any]], dict[str, Any]]:
    """Fail soft after two invalid or failed backend attempts."""

    failure_path = output_dir / f"iteration_{iteration:04d}_failure.json"
    failure = read_json(failure_path)
    if not isinstance(failure, dict):
        raise StreamingPlanError(
            f"planner failure ledger is invalid for iteration {iteration}"
        )
    attempts = failure.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 2:
        raise StreamingPlanError(
            f"planner failure ledger has no two-attempt record for iteration {iteration}"
        )
    grounding_failure = failure.get("grounding_failure")
    return _source_passthrough_decision(
        pending_words=pending_words,
        final_pass=final_pass,
        lookahead_start_word_id=lookahead_start_word_id,
        window_seconds=window_seconds,
        iteration=iteration,
        attempts=attempts,
        trigger="request_failure",
        reason=str(failure.get("validation_error", "planner request failed")),
        grounding_failure=(
            grounding_failure if isinstance(grounding_failure, dict) else None
        ),
    )


def _incremental_decision_makes_progress(
    decision: PlannerDecision,
    *,
    pending_words: Sequence[TranscriptWord],
) -> bool:
    """Return whether a valid decision consumed any visible source prefix."""

    if not pending_words or decision.pending_start_word_id is None:
        raise StreamingPlanError("incremental progress check requires pending source")
    return bool(decision.finalized) or (
        decision.pending_start_word_id > pending_words[0].id
    )


def _words_from_complete_plan(plan: dict[str, Any]) -> list[TranscriptWord]:
    raw_words = plan.get("words")
    if not isinstance(raw_words, list) or not raw_words:
        raise StreamingPlanError("complete plan contains no source words")
    words: list[TranscriptWord] = []
    for expected_id, raw_word in enumerate(raw_words):
        if not isinstance(raw_word, dict) or raw_word.get("id") != expected_id:
            raise StreamingPlanError("complete plan word IDs are not contiguous")
        words.append(
            TranscriptWord(
                id=expected_id,
                text=str(raw_word["text"]),
                start=float(raw_word["start"]),
                end=float(raw_word["end"]),
            )
        )
    return words


def _thought_source_bounds(thought: dict[str, Any]) -> tuple[int, int]:
    ranges = thought.get("source_ranges")
    if not isinstance(ranges, list) or not ranges:
        raise StreamingPlanError("committed thought contains no source ranges")
    return int(ranges[0]["start_word_id"]), int(ranges[-1]["end_word_id"])


def _acoustic_repair_prompt(
    *,
    thought_index: int,
    original_thought: dict[str, Any],
    local_words: Sequence[TranscriptWord],
    unsafe_boundaries: Sequence[dict[str, Any]],
    previous_context: str | None,
    next_context: str | None,
) -> str:
    boundary_summary = []
    for boundary in unsafe_boundaries:
        boundary_summary.append(
            {
                "boundary_id": boundary.get("boundary_id"),
                "boundary_kind": boundary.get("boundary_kind"),
                "safety_status": boundary.get("safety_status"),
                "source_word_ids": boundary.get("source_word_ids"),
                "aligned_timestamps": boundary.get("aligned_timestamps"),
                "forbidden_word_ids": boundary.get("forbidden_word_ids", []),
                "forbidden_source_edges": boundary.get("forbidden_source_edges", []),
                "failure_reason": boundary.get("failure_reason"),
                "retained_word_support": boundary.get("retained_word_support"),
                "reason": boundary.get("error"),
            }
        )
    return f"""Repair one source-grounded narration thought after acoustic validation.

The original semantic decision was meaningful, but its exact occurrence layout
requires a cut where retained and omitted speech touch. The renderer correctly
refused that cut. Choose a different, acoustically separable occurrence layout
from LOCAL SOURCE WORDS.

STRICT PRIORITIES:
1. Preserve the complete intended narration and every unique content phrase.
2. Never select a source occurrence listed in forbidden_word_ids. Its acoustic
   alignment shows that this exact occurrence is incomplete or weak.
3. Do not reuse a source edge listed in forbidden_source_edges. Move the
   discontinuity or remove it by selecting a larger contiguous take.
4. Prefer one complete contiguous successful take.
5. When the speaker keeps a valid prefix and then repeats the ending, keep the
   smallest coherent prefix and the complete later retry.
6. Prefer the later complete corrected phrase over an earlier partial phrase.
7. Minimize discontinuities only after semantic correctness and acoustic
   completeness are satisfied.
8. Remove retries, repetitions, abandoned words, and recording directions.
9. Do not retain unwanted words merely to avoid a rejected source edge.
10. Return exactly ONE finalized thought and pending_start_word_id=null.
11. Every source range uses inclusive first_word_id and last_word_id, exact
   boundary words, and range-level canonical_text supported by every selected
   source word.

ACOUSTIC RETRY EXAMPLE (the IDs are illustrative; use LOCAL SOURCE WORDS):
Earlier source:
    And it might begin with familiar ...
Later correction:
    with the familiar words once upon a time.
Incorrect repair:
    And it might begin with familiar / words once upon a time.
Correct repair:
    Keep the stable prefix ending before the abandoned phrase, then use the
    complete later correction:
    And it might begin / with the familiar words once upon a time.

Read-only preceding thought:
{json.dumps(previous_context, ensure_ascii=False)}

ORIGINAL THOUGHT {thought_index}:
{json.dumps(original_thought, ensure_ascii=False)}

REJECTED ACOUSTIC BOUNDARIES:
{json.dumps(boundary_summary, ensure_ascii=False)}

Read-only following thought:
{json.dumps(next_context, ensure_ascii=False)}

LOCAL SOURCE WORDS:
{json.dumps(_word_payload(local_words), ensure_ascii=False)}
"""


def _validate_acoustic_repair_decision(
    decision: PlannerDecision,
    *,
    original_thought: dict[str, Any],
    unsafe_boundaries: Sequence[dict[str, Any]],
) -> None:
    if len(decision.finalized) != 1:
        raise DecisionValidationError(
            "acoustic repair must return exactly one replacement thought"
        )
    replacement = decision.finalized[0]
    original_tokens = tokenize(str(original_thought["canonical_text"]))
    replacement_tokens = tokenize(replacement.canonical_text)
    if not original_tokens or not replacement_tokens:
        raise DecisionValidationError("acoustic repair returned empty narration")
    matcher = SequenceMatcher(
        None,
        original_tokens,
        replacement_tokens,
        autojunk=False,
    )
    matched = sum(block.size for block in matcher.get_matching_blocks())
    if (
        matched / len(original_tokens) < 0.70
        or matcher.ratio() < 0.75
        or len(replacement_tokens) < math.ceil(0.70 * len(original_tokens))
        or len(replacement_tokens) > len(original_tokens) + 1
    ):
        raise DecisionValidationError(
            "acoustic repair changed the intended narration too substantially"
        )
    ranges = replacement.source_ranges
    forbidden_word_ids = sorted(
        {
            word_id
            for boundary in unsafe_boundaries
            for word_id in boundary.get("forbidden_word_ids", [])
            if type(word_id) is int
        }
    )
    for forbidden_word_id in forbidden_word_ids:
        if any(
            source_range.start_word_id <= forbidden_word_id < source_range.end_word_id
            for source_range in ranges
        ):
            raise DecisionValidationError(
                f"acoustic repair reused forbidden weak source word {forbidden_word_id}"
            )
    for boundary in unsafe_boundaries:
        explicit_edges = boundary.get("forbidden_source_edges")
        if isinstance(explicit_edges, list):
            for edge in explicit_edges:
                if not isinstance(edge, dict):
                    continue
                kind = edge.get("boundary_kind")
                retained_word_id = edge.get("retained_word_id")
                omitted_word_id = edge.get("omitted_word_id")
                if (
                    kind == "selected_to_omitted"
                    and type(omitted_word_id) is int
                    and any(
                        source_range.end_word_id == omitted_word_id
                        for source_range in ranges
                    )
                ):
                    raise DecisionValidationError(
                        "acoustic repair reproduced forbidden trailing source "
                        f"edge after word {retained_word_id} before word "
                        f"{omitted_word_id}"
                    )
                if (
                    kind == "omitted_to_selected"
                    and type(retained_word_id) is int
                    and any(
                        source_range.start_word_id == retained_word_id
                        for source_range in ranges
                    )
                ):
                    raise DecisionValidationError(
                        "acoustic repair reproduced forbidden leading source "
                        f"edge before word {retained_word_id} after word "
                        f"{omitted_word_id}"
                    )
        role_ids = boundary.get("source_word_ids")
        if not isinstance(role_ids, dict):
            continue
        kind = boundary.get("boundary_kind")
        if kind == "selected_to_omitted":
            forbidden_end = role_ids.get("first_omitted")
            if type(forbidden_end) is int and any(
                source_range.end_word_id == forbidden_end for source_range in ranges
            ):
                raise DecisionValidationError(
                    f"acoustic repair reproduced unsafe trailing cut before word "
                    f"{forbidden_end}"
                )
        elif kind == "omitted_to_selected":
            forbidden_start = role_ids.get("first_retained_right")
            if type(forbidden_start) is int and any(
                source_range.start_word_id == forbidden_start for source_range in ranges
            ):
                raise DecisionValidationError(
                    f"acoustic repair reproduced unsafe leading cut at word "
                    f"{forbidden_start}"
                )


def repair_plan_for_acoustic_safety(
    *,
    plan_path: Path,
    boundary_plan_path: Path,
    output_dir: Path,
    backend: NarrationPlannerBackend,
    retry_index: int,
    context_words: int = 80,
) -> dict[str, Any]:
    """Replace only thoughts implicated by fail-closed acoustic boundaries."""

    plan_path = plan_path.resolve()
    boundary_plan_path = boundary_plan_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"acoustic repair output must be empty: {output_dir}")
    if context_words <= 0:
        raise ValueError("acoustic repair context_words must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = read_json(plan_path)
    boundary_plan = read_json(boundary_plan_path)
    if not isinstance(plan, dict) or plan.get("status") != "complete":
        raise StreamingPlanError("acoustic repair requires a complete semantic plan")
    if not isinstance(boundary_plan, dict) or boundary_plan.get("status") != "unsafe":
        raise StreamingPlanError("acoustic repair requires an unsafe boundary plan")
    if boundary_plan.get("streaming_plan_sha256") not in {
        None,
        sha256_file(plan_path),
    }:
        raise StreamingPlanError(
            "unsafe boundary report belongs to a different semantic plan"
        )
    if backend.backend_name != plan.get("backend") or backend.model != plan.get(
        "model"
    ):
        raise StreamingPlanError(
            "acoustic repair backend must match the original semantic planner"
        )
    words = _words_from_complete_plan(plan)
    raw_thoughts = plan.get("committed")
    if not isinstance(raw_thoughts, list) or not raw_thoughts:
        raise StreamingPlanError("semantic plan contains no committed thoughts")
    committed = json.loads(json.dumps(raw_thoughts))
    owners: dict[int, int] = {}
    for thought_index, thought in enumerate(committed):
        for source_range in thought["source_ranges"]:
            for word_id in range(
                int(source_range["start_word_id"]),
                int(source_range["end_word_id"]),
            ):
                owners[word_id] = thought_index
    repairable_statuses = {
        "unsafe_dense_boundary",
        "weak_retained_word_alignment",
        "mfa_word_mapping_failed",
    }
    unsafe = [
        boundary
        for boundary in boundary_plan.get("boundaries", [])
        if isinstance(boundary, dict)
        and boundary.get("safety_status") in repairable_statuses
    ]
    if not unsafe:
        raise StreamingPlanError("boundary plan has no repairable acoustic boundary")
    by_thought: dict[int, list[dict[str, Any]]] = {}
    for boundary in unsafe:
        role_ids = boundary.get("source_word_ids")
        affected = {
            thought_index
            for thought_index in boundary.get("retained_thought_indices", [])
            if type(thought_index) is int and 0 <= thought_index < len(committed)
        }
        if isinstance(role_ids, dict):
            affected.update(
                owners[word_id]
                for role, word_id in role_ids.items()
                if "retained" in str(role)
                and type(word_id) is int
                and word_id in owners
            )
        if not affected:
            relevant_word_ids = {
                word_id
                for word_id in boundary.get("forbidden_word_ids", [])
                if type(word_id) is int
            }
            if isinstance(role_ids, dict):
                relevant_word_ids.update(
                    word_id for word_id in role_ids.values() if type(word_id) is int
                )
            for thought_index, thought in enumerate(committed):
                thought_start, thought_end = _thought_source_bounds(thought)
                if any(
                    thought_start <= word_id < thought_end
                    for word_id in relevant_word_ids
                ):
                    affected.add(thought_index)
        for thought_index in affected:
            by_thought.setdefault(thought_index, []).append(boundary)
    if not by_thought:
        raise StreamingPlanError("unsafe boundaries map to no committed thought")

    repair_records: list[dict[str, Any]] = []
    for thought_index in sorted(by_thought):
        original_thought = json.loads(json.dumps(committed[thought_index]))
        original_start, original_end = _thought_source_bounds(original_thought)
        lower_bound = (
            _thought_source_bounds(committed[thought_index - 1])[1]
            if thought_index > 0
            else 0
        )
        upper_bound = (
            _thought_source_bounds(committed[thought_index + 1])[0]
            if thought_index + 1 < len(committed)
            else len(words)
        )
        context_start = max(lower_bound, original_start - context_words)
        context_end = min(upper_bound, original_end + context_words)
        local_words = words[context_start:context_end]
        prompt = _acoustic_repair_prompt(
            thought_index=thought_index,
            original_thought=original_thought,
            local_words=local_words,
            unsafe_boundaries=by_thought[thought_index],
            previous_context=(
                str(committed[thought_index - 1]["canonical_text"])
                if thought_index > 0
                else None
            ),
            next_context=(
                str(committed[thought_index + 1]["canonical_text"])
                if thought_index + 1 < len(committed)
                else None
            ),
        )
        request_iteration = 10_000 + retry_index * 100 + thought_index
        decision, attempts = _request_validated_decision(
            backend=backend,
            prompt=prompt,
            pending_words=local_words,
            final_pass=True,
            committed_source_end=context_start,
            iteration=request_iteration,
            output_dir=output_dir,
            decision_validator=lambda value, original=original_thought, boundaries=(tuple(by_thought[thought_index])): (
                _validate_acoustic_repair_decision(
                    value,
                    original_thought=original,
                    unsafe_boundaries=boundaries,
                )
            ),
        )
        replacement = _serialize_decision(decision)["finalized"][0]
        replacement["committed_iteration"] = original_thought.get("committed_iteration")
        replacement["grounding_validation"]["thought_index"] = thought_index
        committed[thought_index] = replacement
        thought_boundaries = by_thought[thought_index]
        forbidden_word_ids = sorted(
            {
                word_id
                for boundary in thought_boundaries
                for word_id in boundary.get("forbidden_word_ids", [])
                if type(word_id) is int
            }
        )
        forbidden_source_edges = [
            json.loads(json.dumps(edge))
            for boundary in thought_boundaries
            for edge in boundary.get("forbidden_source_edges", [])
            if isinstance(edge, dict)
        ]
        failure_reasons = sorted(
            {
                str(reason)
                for boundary in thought_boundaries
                if (reason := boundary.get("failure_reason"))
            }
        )
        repair_records.append(
            {
                "retry_index": retry_index,
                "thought_index": thought_index,
                "original_canonical_text": original_thought["canonical_text"],
                "revised_canonical_text": replacement["canonical_text"],
                "original_source_ranges": original_thought["source_ranges"],
                "revised_source_ranges": replacement["source_ranges"],
                "unsafe_boundary_ids": [
                    boundary["boundary_id"] for boundary in by_thought[thought_index]
                ],
                "forbidden_word_ids": forbidden_word_ids,
                "forbidden_source_edges": forbidden_source_edges,
                "failure_reasons": failure_reasons,
                "acoustic_failures": [
                    {
                        "boundary_id": boundary.get("boundary_id"),
                        "safety_status": boundary.get("safety_status"),
                        "failure_reason": boundary.get("failure_reason"),
                        "retained_word_support": boundary.get("retained_word_support"),
                    }
                    for boundary in thought_boundaries
                ],
                "attempts": attempts,
            }
        )

    selected_ranges = _flatten_ranges(committed)
    committed_word_ids = [
        word_id
        for source_range in selected_ranges
        for word_id in range(
            int(source_range["start_word_id"]),
            int(source_range["end_word_id"]),
        )
    ]
    repaired = json.loads(json.dumps(plan))
    repaired.update(
        {
            "status": "complete",
            "grounding_validation": str(output_dir / "grounding_validation.json"),
            "committed": committed,
            "committed_words": committed_word_ids,
            "pending_words": [],
            "next_unread_word": len(words),
            "selected_source_ranges": selected_ranges,
            "selected_source_text": _word_text(
                [words[word_id] for word_id in committed_word_ids]
            ),
            "reconstructed_narration": " ".join(
                str(thought["canonical_text"]) for thought in committed
            ).strip(),
            "semantic_parent_plan": str(plan_path),
            "semantic_parent_plan_sha256": sha256_file(plan_path),
            "acoustic_repair_index": retry_index,
            "acoustic_repairs": [
                *plan.get("acoustic_repairs", []),
                *repair_records,
            ],
        }
    )
    grounding = _grounding_validation_document(
        committed_thoughts=committed,
        iterations=(
            plan.get("iterations", [])
            if isinstance(plan.get("iterations"), list)
            else []
        ),
        status="complete",
    )
    grounding["acoustic_repair_index"] = retry_index
    grounding["acoustic_repairs"] = repair_records
    write_json(output_dir / "grounding_validation.json", grounding)
    write_json(output_dir / "acoustic_repair.json", {"repairs": repair_records})
    write_json(output_dir / "streaming_plan.json", repaired)
    print("\nACOUSTIC SAFETY REPAIR COMPLETE")
    print(f"retry index: {retry_index}")
    print(f"thoughts revised: {len(repair_records)}")
    for record in repair_records:
        print(f"thought {record['thought_index']}: {record['revised_canonical_text']}")
    return repaired


def build_conservative_delivery_plan(
    *,
    plan_path: Path,
    boundary_plan_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Preserve source context across unresolved cuts instead of guessing.

    This is a delivery fallback, not another acoustic resolver.  It removes an
    unsafe physical discontinuity by retaining the omitted source words around
    it.  The resulting extra speech is represented explicitly in canonical
    text and revalidated by the normal bidirectional grounding validator.
    """

    plan_path = plan_path.resolve()
    boundary_plan_path = boundary_plan_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"delivery fallback output must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = read_json(plan_path)
    boundary_plan = read_json(boundary_plan_path)
    if not isinstance(plan, dict) or plan.get("status") != "complete":
        raise StreamingPlanError("delivery fallback requires a complete plan")
    if not isinstance(boundary_plan, dict) or boundary_plan.get("status") != "unsafe":
        raise StreamingPlanError("delivery fallback requires an unsafe boundary plan")
    words = _words_from_complete_plan(plan)
    committed_value = plan.get("committed")
    intervals = boundary_plan.get("source_intervals")
    boundaries = boundary_plan.get("boundaries")
    if not isinstance(committed_value, list) or not committed_value:
        raise StreamingPlanError("delivery fallback has no committed thoughts")
    if not isinstance(intervals, list) or not intervals:
        raise StreamingPlanError("delivery fallback has no source intervals")
    if not isinstance(boundaries, list):
        raise StreamingPlanError("delivery fallback has no boundary ledger")

    committed = json.loads(json.dumps(committed_value))
    selected_by_thought: dict[int, set[int]] = {}
    for thought_index, thought in enumerate(committed):
        selected_by_thought[thought_index] = {
            word_id
            for source_range in thought["source_ranges"]
            for word_id in range(
                int(source_range["start_word_id"]),
                int(source_range["end_word_id"]),
            )
        }

    additions: list[dict[str, Any]] = []
    impacted_thoughts: set[int] = set()
    handled_intervals: set[tuple[int, int, int]] = set()
    unsafe_statuses = {
        "unsafe_dense_boundary",
        "weak_retained_word_alignment",
        "completeness_alignment_failed",
        "mfa_alignment_failed",
        "mfa_runtime_failed",
        "mfa_word_mapping_failed",
        "invalid_source_interval",
    }
    for boundary in boundaries:
        if not isinstance(boundary, dict) or boundary.get("safety_status") not in (
            unsafe_statuses
        ):
            continue
        parts = str(boundary.get("boundary_id", "")).split("_")
        if len(parts) != 3 or parts[0] != "range" or not parts[1].isdigit():
            continue
        interval_index = int(parts[1])
        edge = parts[2]
        if not 0 <= interval_index < len(intervals) or edge not in {"start", "end"}:
            continue
        interval = intervals[interval_index]
        originals = interval.get("merged_original_ranges")
        if not isinstance(originals, list) or not originals:
            continue
        if edge == "start":
            end_word_id = int(interval["start_word_id"])
            if interval_index == 0:
                start_word_id = 0
                thought_index = int(originals[0]["thought_index"])
            else:
                previous = intervals[interval_index - 1]
                previous_originals = previous.get("merged_original_ranges")
                if not isinstance(previous_originals, list) or not previous_originals:
                    continue
                start_word_id = int(previous["end_word_id"])
                thought_index = int(previous_originals[-1]["thought_index"])
        else:
            start_word_id = int(interval["end_word_id"])
            thought_index = int(originals[-1]["thought_index"])
            end_word_id = (
                int(intervals[interval_index + 1]["start_word_id"])
                if interval_index + 1 < len(intervals)
                else len(words)
            )
        key = (thought_index, start_word_id, end_word_id)
        if start_word_id >= end_word_id or key in handled_intervals:
            continue
        handled_intervals.add(key)
        selected_by_thought[thought_index].update(range(start_word_id, end_word_id))
        impacted_thoughts.add(thought_index)
        additions.append(
            {
                "boundary_id": boundary.get("boundary_id"),
                "original_safety_status": boundary.get("safety_status"),
                "thought_index": thought_index,
                "start_word_id": start_word_id,
                "end_word_id": end_word_id,
                "preserved_source_text": _word_text(words[start_word_id:end_word_id]),
            }
        )
    if not additions:
        raise StreamingPlanError(
            "unsafe boundary cannot be removed by preserving adjacent source context"
        )

    raw_thoughts: list[dict[str, Any]] = []
    for thought_index, thought in enumerate(committed):
        if thought_index not in impacted_thoughts:
            raw_thoughts.append(
                {
                    "canonical_text": thought["canonical_text"],
                    "source_ranges": [
                        {
                            "first_word_id": int(source_range["start_word_id"]),
                            "last_word_id": int(source_range["end_word_id"]) - 1,
                            "first_word": words[
                                int(source_range["start_word_id"])
                            ].text,
                            "last_word": words[
                                int(source_range["end_word_id"]) - 1
                            ].text,
                            "canonical_text": source_range["canonical_text"],
                        }
                        for source_range in thought["source_ranges"]
                    ],
                }
            )
            continue
        ordered_ids = sorted(selected_by_thought[thought_index])
        spans: list[tuple[int, int]] = []
        span_start = ordered_ids[0]
        prior = ordered_ids[0]
        for word_id in ordered_ids[1:]:
            if word_id != prior + 1:
                spans.append((span_start, prior + 1))
                span_start = word_id
            prior = word_id
        spans.append((span_start, prior + 1))
        raw_ranges = []
        for start_word_id, end_word_id in spans:
            canonical = _word_text(words[start_word_id:end_word_id])
            raw_ranges.append(
                {
                    "first_word_id": start_word_id,
                    "last_word_id": end_word_id - 1,
                    "first_word": words[start_word_id].text,
                    "last_word": words[end_word_id - 1].text,
                    "canonical_text": canonical,
                }
            )
        raw_thoughts.append(
            {
                "canonical_text": " ".join(
                    source_range["canonical_text"] for source_range in raw_ranges
                ),
                "source_ranges": raw_ranges,
            }
        )

    decision = validate_decision(
        json.dumps(
            {
                "finalized": raw_thoughts,
                "pending_start_word_id": None,
                "pending_reason": "Conservative delivery plan is complete.",
            },
            ensure_ascii=False,
        ),
        pending_words=words,
        final_pass=True,
        committed_source_end=0,
    )
    serialized = _serialize_decision(decision)
    fallback_committed = serialized["finalized"]
    for thought_index, thought in enumerate(fallback_committed):
        thought["committed_iteration"] = committed[thought_index].get(
            "committed_iteration"
        )
        thought["grounding_validation"]["thought_index"] = thought_index

    selected_ranges = _flatten_ranges(fallback_committed)
    committed_word_ids = [
        word_id
        for source_range in selected_ranges
        for word_id in range(
            int(source_range["start_word_id"]),
            int(source_range["end_word_id"]),
        )
    ]
    fallback = json.loads(json.dumps(plan))
    fallback_record = {
        "status": "complete_with_preserved_source_context",
        "reason": "unresolved acoustic cuts were removed without guessing coordinates",
        "parent_plan": str(plan_path),
        "parent_plan_sha256": sha256_file(plan_path),
        "unsafe_boundary_plan": str(boundary_plan_path),
        "unsafe_boundary_plan_sha256": sha256_file(boundary_plan_path),
        "preserved_intervals": additions,
    }
    fallback.update(
        {
            "status": "complete",
            "grounding_validation": str(output_dir / "grounding_validation.json"),
            "committed": fallback_committed,
            "committed_words": committed_word_ids,
            "pending_words": [],
            "next_unread_word": len(words),
            "selected_source_ranges": selected_ranges,
            "selected_source_text": _word_text(
                [words[word_id] for word_id in committed_word_ids]
            ),
            "reconstructed_narration": " ".join(
                str(thought["canonical_text"]) for thought in fallback_committed
            ).strip(),
            "delivery_fallback": fallback_record,
        }
    )
    grounding = _grounding_validation_document(
        committed_thoughts=fallback_committed,
        iterations=(
            fallback.get("iterations", [])
            if isinstance(fallback.get("iterations"), list)
            else []
        ),
        status="complete",
    )
    grounding["delivery_fallback"] = fallback_record
    write_json(output_dir / "grounding_validation.json", grounding)
    write_json(output_dir / "delivery_fallback.json", fallback_record)
    write_json(output_dir / "streaming_plan.json", fallback)
    return fallback


def _take_new_words(
    words: Sequence[TranscriptWord],
    next_unread_word: int,
    *,
    window_seconds: float,
) -> int:
    if next_unread_word >= len(words):
        return next_unread_word
    threshold = words[next_unread_word].start + window_seconds
    end = next_unread_word + 1
    while end < len(words) and words[end].start < threshold:
        end += 1
    return end


def _flatten_ranges(
    thoughts: Sequence[dict[str, Any]],
) -> list[dict[str, int]]:
    return [
        {
            "start_word_id": int(source_range["start_word_id"]),
            "end_word_id": int(source_range["end_word_id"]),
        }
        for thought in thoughts
        for source_range in thought["source_ranges"]
    ]


def _fallback_status(fallbacks: Sequence[dict[str, Any]]) -> str:
    if any(
        isinstance(fallback.get("source_ranges"), list)
        and bool(fallback["source_ranges"])
        for fallback in fallbacks
    ):
        return "source_passthrough_used"
    if fallbacks:
        return "planner_failure_recovered_without_source_passthrough"
    return "not_used"


def _grounding_validation_document(
    *,
    committed_thoughts: Sequence[dict[str, Any]],
    iterations: Sequence[dict[str, Any]],
    status: str,
    error: str | None = None,
    grounding_failures: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    thoughts: list[dict[str, Any]] = []
    for thought_index, thought in enumerate(committed_thoughts):
        raw_validation = thought.get("grounding_validation")
        if not isinstance(raw_validation, dict):
            raise StreamingPlanError(
                f"committed thought {thought_index} has no grounding validation"
            )
        validation = json.loads(json.dumps(raw_validation))
        validation["thought_index"] = thought_index
        thoughts.append(validation)
    unsupported = [
        token
        for validation in thoughts
        for token in validation.get("unsupported_tokens", [])
    ]
    unsupported.extend(
        str(failure["unsupported_canonical_token"])
        for failure in grounding_failures
        if failure.get("unsupported_canonical_token")
    )
    unrepresented_source = [
        item
        for validation in thoughts
        for source_range in validation.get("source_ranges", [])
        for item in source_range.get("unrepresented_source_tokens", [])
    ]
    unrepresented_source.extend(
        item
        for failure in grounding_failures
        for item in failure.get("unrepresented_source_tokens", [])
    )
    finalized_thoughts = len(thoughts)
    source_ranges = sum(
        len(validation.get("source_ranges", [])) for validation in thoughts
    )
    canonical_tokens = sum(
        int(validation.get("canonical_tokens", 0)) for validation in thoughts
    )
    supported_tokens = sum(
        int(validation.get("supported_tokens", 0)) for validation in thoughts
    )
    retries = sum(
        max(0, len(iteration.get("attempts", [])) - 1) for iteration in iterations
    )
    fallbacks = [
        json.loads(json.dumps(fallback))
        for iteration in iterations
        if isinstance((fallback := iteration.get("fallback")), dict)
    ]
    return {
        "schema_version": 1,
        "validator": "strict_bidirectional_range_source_grounding_v2",
        "status": (
            "valid"
            if status == "complete"
            and not unsupported
            and not unrepresented_source
            and canonical_tokens == supported_tokens
            else status
        ),
        "finalized_thoughts": finalized_thoughts,
        "source_ranges": source_ranges,
        "canonical_tokens": canonical_tokens,
        "supported_tokens": supported_tokens,
        "unsupported_tokens": unsupported,
        "unrepresented_source_tokens": unrepresented_source,
        "planner_retries": retries,
        "fallback_status": _fallback_status(fallbacks),
        "fallback_count": len(fallbacks),
        "source_passthrough_count": sum(
            isinstance(fallback.get("source_ranges"), list)
            and bool(fallback["source_ranges"])
            for fallback in fallbacks
        ),
        "fallbacks": fallbacks,
        "plan_accepted": (
            status == "complete"
            and not unsupported
            and not unrepresented_source
            and canonical_tokens == supported_tokens
        ),
        "error": error,
        "grounding_failures": list(grounding_failures),
        "thoughts": thoughts,
    }


def _print_iteration(debug: dict[str, Any], *, backend_name: str) -> None:
    print(f"\nITERATION {debug['iteration']}")
    print("new source interval")
    print(debug["new_source_interval_text"] or "(none; final EOF flush)")
    print("complete pending input")
    print(debug["complete_pending_input_text"])
    print("newly finalized text")
    if debug["newly_finalized_text"]:
        for text in debug["newly_finalized_text"]:
            print(text)
    else:
        print("(none)")
    print("selected source ranges")
    print(
        json.dumps(
            debug["selected_source_ranges"],
            ensure_ascii=False,
            indent=2,
        )
    )
    print("remaining pending transcript")
    print(debug["remaining_pending_transcript_text"] or "(none)")
    label = "Gemini raw JSON" if backend_name == "gemini" else "Planner raw JSON"
    print(label)
    print(debug["raw_response"])
    fallback = debug.get("fallback")
    if isinstance(fallback, dict):
        print("deterministic source fallback")
        print(json.dumps(fallback, ensure_ascii=False, indent=2))


def run_streaming_planner(
    *,
    transcript_path: Path,
    output_dir: Path,
    backend: NarrationPlannerBackend,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Run chronological look-ahead planning and save its full ledger."""

    if not math.isfinite(window_seconds) or window_seconds <= 0.0:
        raise ValueError("window_seconds must be finite and positive")
    transcript_path = transcript_path.resolve()
    output_dir = output_dir.resolve()
    if not transcript_path.exists():
        raise FileNotFoundError(transcript_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory must be empty for a new plan: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_data = read_json(transcript_path)
    if not isinstance(transcript_data, dict):
        raise ValueError("Whisper transcript root must be an object")
    words = load_transcript_words(transcript_data)
    word_by_id = {word.id: word for word in words}

    committed_words: list[TranscriptWord] = []
    committed_thoughts: list[dict[str, Any]] = []
    pending_words: list[TranscriptWord] = []
    next_unread_word = 0
    committed_source_end = 0
    iterations: list[dict[str, Any]] = []
    fallback_records: list[dict[str, Any]] = []
    pending_no_progress_age = 0
    iteration = 0

    def current_plan(status: str) -> dict[str, Any]:
        selected_ranges = _flatten_ranges(committed_thoughts)
        committed_word_ids = [word.id for word in committed_words]
        return {
            "schema_version": 1,
            "planner": "streaming_narration_v1",
            "status": status,
            "backend": backend.backend_name,
            "model": backend.model,
            "transcript": str(transcript_path),
            "transcript_sha256": sha256_file(transcript_path),
            "grounding_validation": str(output_dir / "grounding_validation.json"),
            "window_seconds": window_seconds,
            "word_count": len(words),
            "words": _word_payload(words),
            "committed": committed_thoughts,
            "committed_words": committed_word_ids,
            "pending_words": [word.id for word in pending_words],
            "next_unread_word": next_unread_word,
            "selected_source_ranges": selected_ranges,
            "selected_source_text": _word_text(
                [word_by_id[word_id] for word_id in committed_word_ids]
            ),
            "reconstructed_narration": " ".join(
                str(thought["canonical_text"]) for thought in committed_thoughts
            ).strip(),
            "fallback_status": _fallback_status(fallback_records),
            "fallback_event_count": len(fallback_records),
            "source_passthrough_count": sum(
                isinstance(fallback.get("source_ranges"), list)
                and bool(fallback["source_ranges"])
                for fallback in fallback_records
            ),
            "fallbacks": fallback_records,
            "iterations": iterations,
        }

    try:
        while next_unread_word < len(words):
            new_end = _take_new_words(
                words,
                next_unread_word,
                window_seconds=window_seconds,
            )
            new_words = list(words[next_unread_word:new_end])
            next_unread_word = new_end
            pending_words = [*pending_words, *new_words]
            complete_pending_input = list(pending_words)
            iteration += 1
            prompt = _prompt(
                iteration=iteration,
                pending_words=complete_pending_input,
                committed_thoughts=committed_thoughts,
                final_pass=False,
            )
            committed_before = _fingerprint(committed_thoughts)
            fallback: dict[str, Any] | None = None
            request_failed = False
            try:
                decision, attempts = _request_validated_decision(
                    backend=backend,
                    prompt=prompt,
                    pending_words=complete_pending_input,
                    final_pass=False,
                    committed_source_end=committed_source_end,
                    iteration=iteration,
                    output_dir=output_dir,
                )
            except StreamingPlanError:
                request_failed = True
                decision, attempts, fallback = _passthrough_after_request_failure(
                    pending_words=complete_pending_input,
                    final_pass=False,
                    lookahead_start_word_id=new_words[0].id,
                    window_seconds=window_seconds,
                    iteration=iteration,
                    output_dir=output_dir,
                )
                fallback_records.append(fallback)
            if request_failed:
                # The newest look-ahead has already been exposed once. Keep
                # that age so another stalled iteration commits an old window
                # instead of allowing pending input to grow indefinitely.
                pending_no_progress_age = 1
            elif _incremental_decision_makes_progress(
                decision,
                pending_words=complete_pending_input,
            ):
                pending_no_progress_age = 0
            else:
                pending_no_progress_age += 1
                if pending_no_progress_age >= 2:
                    decision, attempts, fallback = _source_passthrough_decision(
                        pending_words=complete_pending_input,
                        final_pass=False,
                        lookahead_start_word_id=new_words[0].id,
                        window_seconds=window_seconds,
                        iteration=iteration,
                        attempts=attempts,
                        trigger="valid_no_progress",
                        reason=(
                            "The planner returned a valid decision without "
                            "consuming source input in two consecutive windows."
                        ),
                        grounding_failure=None,
                    )
                    fallback_records.append(fallback)
                    # The retained look-ahead was visible in this iteration,
                    # preserving the one-thought delay while bounding growth.
                    pending_no_progress_age = 1
            if _fingerprint(committed_thoughts) != committed_before:
                raise StreamingPlanError(
                    "committed source ranges changed during model request"
                )
            newly_committed = []
            for thought in decision.finalized:
                serialized = {
                    **_serialize_decision(
                        PlannerDecision((thought,), None, "committed")
                    )["finalized"][0],
                    "committed_iteration": iteration,
                }
                serialized["grounding_validation"]["thought_index"] = len(
                    committed_thoughts
                )
                committed_thoughts.append(serialized)
                newly_committed.append(serialized)
                for source_range in thought.source_ranges:
                    committed_words.extend(
                        word_by_id[word_id]
                        for word_id in range(
                            source_range.start_word_id,
                            source_range.end_word_id,
                        )
                    )
                    committed_source_end = source_range.end_word_id
            pending_start = decision.pending_start_word_id
            assert pending_start is not None
            pending_words = [word for word in pending_words if word.id >= pending_start]
            debug = {
                "iteration": iteration,
                "final_pass": False,
                "new_source_interval": {
                    "start_word_id": new_words[0].id,
                    "end_word_id": new_words[-1].id + 1,
                    "start": new_words[0].start,
                    "end": new_words[-1].end,
                },
                "new_source_interval_text": _debug_word_text(new_words),
                "complete_pending_input": _word_payload(complete_pending_input),
                "complete_pending_input_text": _debug_word_text(complete_pending_input),
                "newly_finalized_text": [
                    thought["canonical_text"] for thought in newly_committed
                ],
                "selected_source_ranges": _flatten_ranges(newly_committed),
                "remaining_pending_transcript": _word_payload(pending_words),
                "remaining_pending_transcript_text": _debug_word_text(pending_words),
                "pending_start_word_id": pending_start,
                "pending_reason": decision.pending_reason,
                "raw_response": attempts[-1]["raw_response"],
                "attempts": attempts,
                "fallback": fallback,
                "pending_no_progress_age": pending_no_progress_age,
                "prompt": prompt,
                "state": {
                    "committed_words": [word.id for word in committed_words],
                    "pending_words": [word.id for word in pending_words],
                    "next_unread_word": next_unread_word,
                },
                "committed_fingerprint": _fingerprint(committed_thoughts),
            }
            iterations.append(debug)
            write_json(
                output_dir / f"iteration_{iteration:04d}.json",
                debug,
            )
            write_json(
                output_dir / "streaming_plan.json",
                current_plan("in_progress"),
            )
            write_json(
                output_dir / "grounding_validation.json",
                _grounding_validation_document(
                    committed_thoughts=committed_thoughts,
                    iterations=iterations,
                    status="in_progress",
                ),
            )
            _print_iteration(debug, backend_name=backend.backend_name)

        iteration += 1
        final_pending_input = list(pending_words)
        prompt = _prompt(
            iteration=iteration,
            pending_words=final_pending_input,
            committed_thoughts=committed_thoughts,
            final_pass=True,
        )
        committed_before = _fingerprint(committed_thoughts)
        fallback = None
        request_failed = False
        try:
            decision, attempts = _request_validated_decision(
                backend=backend,
                prompt=prompt,
                pending_words=final_pending_input,
                final_pass=True,
                committed_source_end=committed_source_end,
                iteration=iteration,
                output_dir=output_dir,
            )
        except StreamingPlanError:
            request_failed = True
            decision, attempts, fallback = _passthrough_after_request_failure(
                pending_words=final_pending_input,
                final_pass=True,
                lookahead_start_word_id=None,
                window_seconds=window_seconds,
                iteration=iteration,
                output_dir=output_dir,
            )
            fallback_records.append(fallback)
        if not request_failed and not decision.finalized:
            decision, attempts, fallback = _source_passthrough_decision(
                pending_words=final_pending_input,
                final_pass=True,
                lookahead_start_word_id=None,
                window_seconds=window_seconds,
                iteration=iteration,
                attempts=attempts,
                trigger="valid_eof_no_progress",
                reason=(
                    "The planner returned a valid EOF response without finalizing "
                    "any remaining source words."
                ),
                grounding_failure=None,
            )
            fallback_records.append(fallback)
        if _fingerprint(committed_thoughts) != committed_before:
            raise StreamingPlanError(
                "committed source ranges changed during final model request"
            )
        newly_committed = []
        for thought in decision.finalized:
            serialized = {
                **_serialize_decision(PlannerDecision((thought,), None, "committed"))[
                    "finalized"
                ][0],
                "committed_iteration": iteration,
            }
            serialized["grounding_validation"]["thought_index"] = len(
                committed_thoughts
            )
            committed_thoughts.append(serialized)
            newly_committed.append(serialized)
            for source_range in thought.source_ranges:
                committed_words.extend(
                    word_by_id[word_id]
                    for word_id in range(
                        source_range.start_word_id,
                        source_range.end_word_id,
                    )
                )
                committed_source_end = source_range.end_word_id
        pending_words = []
        debug = {
            "iteration": iteration,
            "final_pass": True,
            "new_source_interval": None,
            "new_source_interval_text": "",
            "complete_pending_input": _word_payload(final_pending_input),
            "complete_pending_input_text": _debug_word_text(final_pending_input),
            "newly_finalized_text": [
                thought["canonical_text"] for thought in newly_committed
            ],
            "selected_source_ranges": _flatten_ranges(newly_committed),
            "remaining_pending_transcript": [],
            "remaining_pending_transcript_text": "",
            "pending_start_word_id": None,
            "pending_reason": decision.pending_reason,
            "raw_response": attempts[-1]["raw_response"],
            "attempts": attempts,
            "fallback": fallback,
            "pending_no_progress_age": 0,
            "prompt": prompt,
            "state": {
                "committed_words": [word.id for word in committed_words],
                "pending_words": [],
                "next_unread_word": next_unread_word,
            },
            "committed_fingerprint": _fingerprint(committed_thoughts),
        }
        iterations.append(debug)
        write_json(output_dir / f"iteration_{iteration:04d}.json", debug)
        plan = current_plan("complete")
        write_json(output_dir / "streaming_plan.json", plan)
        grounding_validation = _grounding_validation_document(
            committed_thoughts=committed_thoughts,
            iterations=iterations,
            status="complete",
        )
        write_json(
            output_dir / "grounding_validation.json",
            grounding_validation,
        )
        _print_iteration(debug, backend_name=backend.backend_name)
        print("\nCOMPLETE RECONSTRUCTED NARRATION")
        print(plan["reconstructed_narration"])
        print("\nSOURCE GROUNDING VALIDATION COMPLETE")
        print(f"finalized thoughts: {grounding_validation['finalized_thoughts']}")
        print(f"source ranges: {grounding_validation['source_ranges']}")
        print(
            "unsupported canonical tokens: "
            f"{len(grounding_validation['unsupported_tokens'])}"
        )
        print(
            "unrepresented selected source tokens: "
            f"{len(grounding_validation['unrepresented_source_tokens'])}"
        )
        print(f"planner retries: {grounding_validation['planner_retries']}")
        print(f"plan accepted: {str(grounding_validation['plan_accepted']).lower()}")
        return plan
    except Exception as error:
        failure_plan = current_plan("failed")
        error_text = f"{type(error).__name__}: {error}"
        failure_plan["error"] = error_text
        write_json(output_dir / "streaming_plan.json", failure_plan)
        report_iterations = list(iterations)
        grounding_failures: list[dict[str, Any]] = []
        for failure_path in sorted(output_dir.glob("iteration_*_failure.json")):
            failure = read_json(failure_path)
            if isinstance(failure, dict):
                report_iterations.append({"attempts": failure.get("attempts", [])})
                grounding_failure = failure.get("grounding_failure")
                if isinstance(grounding_failure, dict):
                    grounding_failures.append(grounding_failure)
        write_json(
            output_dir / "grounding_validation.json",
            _grounding_validation_document(
                committed_thoughts=committed_thoughts,
                iterations=report_iterations,
                status="failed",
                error=error_text,
                grounding_failures=grounding_failures,
            ),
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicecut --stream-plan",
        description=(
            "Create a streaming semantic edit plan from an existing Whisper "
            "word transcript; never render audio."
        ),
    )
    parser.add_argument("--stream-plan", action="store_true")
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_planner_backend_arguments(parser)
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.stream_plan:
        parser.error("streaming planner requires --stream-plan")
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    backend = create_planner_backend(
        provider=args.planner_backend,
        model=args.planner_model,
        env_file=args.env_file.resolve(),
        max_output_tokens=args.max_output_tokens,
        base_url=args.planner_base_url,
        api_key_env=args.planner_api_key_env,
        local_python=args.planner_python.absolute(),
        local_files_only=args.local_files_only,
    )
    try:
        plan = run_streaming_planner(
            transcript_path=args.transcript,
            output_dir=args.output_dir,
            backend=backend,
            window_seconds=args.window_seconds,
        )
    finally:
        backend.close()
    print(
        json.dumps(
            {
                "status": plan["status"],
                "backend": plan["backend"],
                "model": plan["model"],
                "plan": str((args.output_dir.resolve() / "streaming_plan.json")),
                "reconstructed_narration": plan["reconstructed_narration"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
