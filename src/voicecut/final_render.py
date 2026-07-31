#!/usr/bin/env python3
"""Fail-closed final rendering for an existing grounded semantic plan."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from .common import read_json, sha256_file, write_json
from .planner_backends import (
    DEFAULT_LOCAL_PYTHON,
    DEFAULT_MAX_OUTPUT_TOKENS,
    PAUSE_SYSTEM_INSTRUCTION,
    PlannerBackend as PausePlannerBackend,
    add_planner_backend_arguments,
    create_planner_backend,
)
from .rough_render import (
    MergedRange,
    PlanWord,
    flatten_selected_ranges,
    load_plan_words,
    merge_adjacent_ranges,
    timestamp_to_sample,
)
from .semantic_pause import (
    PAUSE_TARGETS_MS,
    ROOM_TONE_FADE_MS,
    SourceRoomToneAllocator,
    create_pause_plan,
    refine_eof_tail,
    validate_pause_response,
)
from .trailing_refine import (
    VerifiedQuietEvidence,
    find_verified_quiet_evidence,
)
from .streaming_narration import repair_plan_for_acoustic_safety


DEFAULT_ALIGNMENT_PYTHON = Path(sys.executable)
CONTEXT_WORDS_PER_SIDE = 2
CROP_CONTEXT_MS = 300.0
PROTECTED_SPEECH_MARGIN_MS = 20.0
MINIMUM_VERIFIED_QUIET_MS = 20.0
QUIET_FADE_MS = 5.0
ALIGNMENT_LANGUAGE = "en"
DEFAULT_MAX_ACOUSTIC_RETRIES = 3
RETAINED_WORD_EDGE_CHARACTER_COUNT = 3
RETAINED_WORD_LOCAL_CONTEXT_CHARACTER_COUNT = 8
MIN_RETAINED_WORD_SCORE = 0.45
MIN_RETAINED_EDGE_CHARACTER_SCORE = 0.48
MIN_RETAINED_EDGE_TO_CONTEXT_RATIO = 0.55
ALIGNMENT_GEOMETRY_EPSILON_SECONDS = 1e-6


class FinalRenderError(RuntimeError):
    """A safe final cut cannot be published."""


def _validate_grounded_plan(
    *,
    audio_path: Path,
    plan_path: Path,
) -> tuple[dict[str, Any], Path, int]:
    plan = read_json(plan_path)
    if not isinstance(plan, dict) or plan.get("status") != "complete":
        raise FinalRenderError("final rendering requires a complete streaming plan")
    if plan.get("planner") != "streaming_narration_v1":
        raise FinalRenderError(
            "final rendering requires a current streaming narration plan"
        )
    words = load_plan_words(plan)
    ranges = flatten_selected_ranges(plan, word_count=len(words))
    if not ranges:
        raise FinalRenderError("streaming plan selects no source ranges")
    committed = plan.get("committed")
    if not isinstance(committed, list) or not committed:
        raise FinalRenderError("streaming plan contains no committed thoughts")

    transcript_value = plan.get("transcript")
    if not isinstance(transcript_value, str) or not transcript_value:
        raise FinalRenderError(
            "streaming plan has no immutable source-transcript provenance"
        )
    transcript_path = Path(transcript_value).resolve()
    if not transcript_path.is_file():
        raise FileNotFoundError(transcript_path)
    transcript = read_json(transcript_path)
    if not isinstance(transcript, dict):
        raise FinalRenderError("source transcript root must be an object")
    expected_audio_sha = transcript.get("audio_sha256")
    if not isinstance(expected_audio_sha, str):
        raise FinalRenderError("source transcript has no audio SHA-256")
    if sha256_file(audio_path) != expected_audio_sha:
        raise FinalRenderError(
            "audio does not match the recording used by the semantic plan"
        )
    expected_transcript_sha = plan.get("transcript_sha256")
    if (
        isinstance(expected_transcript_sha, str)
        and sha256_file(transcript_path) != expected_transcript_sha
    ):
        raise FinalRenderError("source transcript changed after semantic planning")

    grounding_value = plan.get("grounding_validation")
    if not isinstance(grounding_value, str) or not grounding_value:
        raise FinalRenderError(
            "final rendering requires a strict source-grounding validation"
        )
    grounding_path = Path(grounding_value).resolve()
    if not grounding_path.is_file():
        raise FileNotFoundError(grounding_path)
    grounding = read_json(grounding_path)
    if not isinstance(grounding, dict):
        raise FinalRenderError("grounding validation root must be an object")
    if grounding.get("validator") != "strict_bidirectional_range_source_grounding_v2":
        raise FinalRenderError(
            "grounding report was not produced by the strict bidirectional "
            "source-range validator"
        )
    if (
        grounding.get("status") != "valid"
        or grounding.get("plan_accepted") is not True
        or grounding.get("unsupported_tokens") != []
        or grounding.get("unrepresented_source_tokens") != []
    ):
        raise FinalRenderError(
            "semantic plan did not pass strict source-grounding validation"
        )
    canonical_tokens = grounding.get("canonical_tokens")
    supported_tokens = grounding.get("supported_tokens")
    if (
        type(canonical_tokens) is not int
        or type(supported_tokens) is not int
        or canonical_tokens < 1
        or supported_tokens != canonical_tokens
    ):
        raise FinalRenderError("grounding validation has unsupported canonical text")
    if grounding.get("finalized_thoughts") != len(committed):
        raise FinalRenderError(
            "grounding validation does not describe every committed thought"
        )
    if grounding.get("source_ranges") != len(ranges):
        raise FinalRenderError(
            "grounding validation does not describe every selected source range"
        )

    grounding_thoughts = grounding.get("thoughts")
    if not isinstance(grounding_thoughts, list) or len(grounding_thoughts) != len(
        committed
    ):
        raise FinalRenderError(
            "grounding validation thought ledger does not match the semantic plan"
        )
    for thought_index, (thought, validation) in enumerate(
        zip(committed, grounding_thoughts, strict=True)
    ):
        if not isinstance(thought, dict) or not isinstance(validation, dict):
            raise FinalRenderError(
                f"grounding validation thought {thought_index} is malformed"
            )
        embedded = thought.get("grounding_validation")
        if not isinstance(embedded, dict):
            raise FinalRenderError(
                f"committed thought {thought_index} has no embedded grounding "
                "validation"
            )
        expected = json.loads(json.dumps(embedded))
        expected["thought_index"] = thought_index
        if validation != expected:
            raise FinalRenderError(
                f"grounding validation thought {thought_index} does not match "
                "the committed semantic plan"
            )

    return plan, grounding_path, len(ranges)


def _cache_pause_plan(
    *,
    plan_path: Path,
    output_dir: Path,
    supplied_pause_plan_path: Path | None,
    backend: PausePlannerBackend | None,
    env_file: Path,
    provider: str,
    model: str | None,
    base_url: str | None,
    api_key_env: str | None,
    local_python: Path,
    local_files_only: bool,
    max_output_tokens: int,
) -> Path:
    destination = output_dir / "pause_plan.json"
    if supplied_pause_plan_path is not None:
        supplied = supplied_pause_plan_path.resolve()
        if not supplied.is_file():
            raise FileNotFoundError(supplied)
        if supplied != destination:
            shutil.copy2(supplied, destination)
        return destination

    owns_backend = backend is None
    active_backend = (
        create_planner_backend(
            provider=provider,
            model=model,
            env_file=env_file.resolve(),
            max_output_tokens=max_output_tokens,
            system_instruction=PAUSE_SYSTEM_INSTRUCTION,
            base_url=base_url,
            api_key_env=api_key_env,
            local_python=local_python.absolute(),
            local_files_only=local_files_only,
        )
        if backend is None
        else backend
    )
    try:
        create_pause_plan(
            plan_path=plan_path,
            output_dir=output_dir,
            backend=active_backend,
        )
    finally:
        if owns_backend:
            active_backend.close()
    return destination


def _retarget_pause_plan(
    *,
    source_pause_plan_path: Path,
    repaired_plan_path: Path,
    destination: Path,
) -> Path:
    """Reuse classifications when acoustic repair preserves thought structure."""

    pause_plan = read_json(source_pause_plan_path)
    repaired_plan = read_json(repaired_plan_path)
    if not isinstance(pause_plan, dict) or not isinstance(repaired_plan, dict):
        raise FinalRenderError("cannot retarget malformed pause/semantic plans")
    committed = repaired_plan.get("committed")
    if not isinstance(committed, list):
        raise FinalRenderError("repaired semantic plan contains no thoughts")
    if pause_plan.get("thought_count") != len(committed):
        raise FinalRenderError(
            "acoustic repair changed thought count; pause classification cannot "
            "be reused"
        )
    retargeted = json.loads(json.dumps(pause_plan))
    retargeted["streaming_plan"] = str(repaired_plan_path.resolve())
    retargeted["streaming_plan_sha256"] = sha256_file(repaired_plan_path)
    retargeted["acoustic_repair_retargeted"] = True
    retargeted["parent_pause_plan"] = str(source_pause_plan_path.resolve())
    retargeted["parent_pause_plan_sha256"] = sha256_file(source_pause_plan_path)
    write_json(destination, retargeted)
    return destination


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def _finite_number(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _flatten_aligned_words(aligned: dict[str, Any]) -> list[dict[str, Any]]:
    raw_words = aligned.get("word_segments")
    if isinstance(raw_words, list):
        return [word for word in raw_words if isinstance(word, dict)]
    flattened: list[dict[str, Any]] = []
    raw_segments = aligned.get("segments")
    if not isinstance(raw_segments, list):
        return flattened
    for segment in raw_segments:
        if isinstance(segment, dict) and isinstance(segment.get("words"), list):
            flattened.extend(
                word for word in segment["words"] if isinstance(word, dict)
            )
    return flattened


def _flatten_character_word_groups(
    aligned: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    raw_segments = aligned.get("segments")
    if not isinstance(raw_segments, list):
        return groups
    for segment in raw_segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("chars"), list):
            continue
        current: list[dict[str, Any]] = []
        for character in segment["chars"]:
            if not isinstance(character, dict):
                continue
            if str(character.get("char", "")).isspace():
                if current:
                    groups.append(current)
                    current = []
            else:
                current.append(character)
        if current:
            groups.append(current)
    return groups


def _alignment_character_records(
    group: Sequence[dict[str, Any]],
    *,
    crop_start: float,
    sample_rate: int,
    total_samples: int,
) -> tuple[list[dict[str, Any]], tuple[float, float] | None]:
    records: list[dict[str, Any]] = []
    positive_timed: list[tuple[float, float]] = []
    for character in group:
        raw_character = str(character.get("char", ""))
        relative_start = _finite_number(character.get("start"))
        relative_end = _finite_number(character.get("end"))
        score = _finite_number(character.get("score"))
        absolute_start = (
            crop_start + relative_start if relative_start is not None else None
        )
        absolute_end = crop_start + relative_end if relative_end is not None else None
        start_sample = (
            timestamp_to_sample(
                absolute_start,
                sample_rate=sample_rate,
                total_samples=total_samples,
                rounding="floor",
            )
            if absolute_start is not None
            else None
        )
        end_sample = (
            timestamp_to_sample(
                absolute_end,
                sample_rate=sample_rate,
                total_samples=total_samples,
                rounding="ceil",
            )
            if absolute_end is not None
            else None
        )
        records.append(
            {
                "character": raw_character,
                "start": absolute_start,
                "end": absolute_end,
                "score": score,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "is_alphabetic": raw_character.isalpha(),
            }
        )
        if (
            relative_start is not None
            and relative_end is not None
            and relative_end > relative_start
            and score is not None
            and score > 0.0
        ):
            positive_timed.append((relative_start, relative_end))
    positive_span = (
        (positive_timed[0][0], positive_timed[-1][1]) if positive_timed else None
    )
    return records, positive_span


def _alphabetic_records(span: dict[str, Any]) -> list[dict[str, Any]]:
    characters = span.get("characters")
    if not isinstance(characters, list):
        return []
    return [
        character
        for character in characters
        if isinstance(character, dict) and str(character.get("character", "")).isalpha()
    ]


def _minimum_edge_score(
    characters: Sequence[dict[str, Any]],
    *,
    edge: str,
) -> float | None:
    if edge not in {"initial", "terminal"}:
        raise ValueError("alignment edge must be initial or terminal")
    selected = (
        list(characters[:RETAINED_WORD_EDGE_CHARACTER_COUNT])
        if edge == "initial"
        else list(characters[-RETAINED_WORD_EDGE_CHARACTER_COUNT:])
    )
    scores = [_finite_number(character.get("score")) for character in selected]
    if not scores or any(score is None for score in scores):
        return None
    return min(float(score) for score in scores if score is not None)


def _nearby_context_scores(
    *,
    span: dict[str, Any],
    context_spans: Sequence[dict[str, Any]],
    excluded_characters: Sequence[dict[str, Any]] = (),
) -> list[float]:
    excluded_ids = {id(character) for character in excluded_characters}
    target_word_id = int(span["word_id"])
    target_characters = [
        character
        for character in _alphabetic_records(span)
        if id(character) not in excluded_ids
        and _finite_number(character.get("score")) is not None
    ]
    scores = [float(character["score"]) for character in target_characters]
    if len(scores) >= RETAINED_WORD_LOCAL_CONTEXT_CHARACTER_COUNT:
        return scores[:RETAINED_WORD_LOCAL_CONTEXT_CHARACTER_COUNT]

    neighbors = sorted(
        (
            candidate
            for candidate in context_spans
            if int(candidate.get("word_id", -1)) != target_word_id
        ),
        key=lambda candidate: (
            abs(int(candidate.get("word_id", -1)) - target_word_id),
            int(candidate.get("word_id", -1)),
        ),
    )
    for neighbor in neighbors:
        for character in _alphabetic_records(neighbor):
            score = _finite_number(character.get("score"))
            if score is not None:
                scores.append(score)
                if len(scores) >= RETAINED_WORD_LOCAL_CONTEXT_CHARACTER_COUNT:
                    return scores
    return scores


def evaluate_retained_word_support(
    span: dict[str, Any],
    context_spans: Sequence[dict[str, Any]],
    *,
    edge: str,
) -> dict[str, Any]:
    """Evaluate alignment support for one retained word without changing audio."""

    if edge not in {"initial", "terminal"}:
        raise ValueError("retained-word edge must be initial or terminal")
    expected = [
        character.lower()
        for character in str(span.get("text", ""))
        if character.isalpha()
    ]
    alphabetic = _alphabetic_records(span)
    aligned_text = [
        str(character.get("character", "")).lower() for character in alphabetic
    ]
    timed = [
        character
        for character in alphabetic
        if _finite_number(character.get("start")) is not None
        and _finite_number(character.get("end")) is not None
        and _finite_number(character.get("score")) is not None
    ]
    complete_coverage = (
        bool(expected) and aligned_text == expected and len(timed) == len(expected)
    )
    character_coverage = len(timed) / len(expected) if expected else 0.0

    monotonic = complete_coverage
    previous_end: float | None = None
    if monotonic:
        for character in timed:
            start = float(character["start"])
            end = float(character["end"])
            if end <= start or (
                previous_end is not None
                and start < previous_end - ALIGNMENT_GEOMETRY_EPSILON_SECONDS
            ):
                monotonic = False
                break
            previous_end = end

    edge_characters = (
        alphabetic[:RETAINED_WORD_EDGE_CHARACTER_COUNT]
        if edge == "initial"
        else alphabetic[-RETAINED_WORD_EDGE_CHARACTER_COUNT:]
    )
    edge_scores = [
        _finite_number(character.get("score")) for character in edge_characters
    ]
    valid_edge_scores = [float(score) for score in edge_scores if score is not None]
    median_edge_score = median(valid_edge_scores) if valid_edge_scores else None
    minimum_edge_score = min(valid_edge_scores) if valid_edge_scores else None
    context_scores = _nearby_context_scores(
        span=span,
        context_spans=context_spans,
        excluded_characters=edge_characters,
    )
    word_score = _finite_number(span.get("word_score"))
    local_context_score = median(context_scores) if context_scores else word_score
    score_ratio = (
        minimum_edge_score / local_context_score
        if minimum_edge_score is not None
        and local_context_score is not None
        and local_context_score > 0.0
        else None
    )

    if not complete_coverage:
        status = "incomplete_character_coverage"
    elif not monotonic:
        status = "invalid_alignment_geometry"
    elif (
        word_score is None
        or median_edge_score is None
        or minimum_edge_score is None
        or local_context_score is None
        or score_ratio is None
    ):
        status = "incomplete_character_coverage"
    elif word_score < MIN_RETAINED_WORD_SCORE or (
        minimum_edge_score < MIN_RETAINED_EDGE_CHARACTER_SCORE
        and score_ratio < MIN_RETAINED_EDGE_TO_CONTEXT_RATIO
    ):
        status = (
            "weak_initial_word_support"
            if edge == "initial"
            else "weak_terminal_word_support"
        )
    else:
        status = "supported_complete_word"

    return {
        "word_id": int(span["word_id"]),
        "source_text": str(span.get("text", "")),
        "edge": edge,
        "status": status,
        "complete_character_coverage": complete_coverage,
        "monotonic_character_timestamps": monotonic,
        "word_score": word_score,
        "expected_alignable_character_count": len(expected),
        "aligned_character_count": len(timed),
        "character_coverage": character_coverage,
        "edge_character_scores": valid_edge_scores,
        "median_edge_score": median_edge_score,
        "minimum_edge_score": minimum_edge_score,
        "local_context_median_score": local_context_score,
        "edge_to_context_score_ratio": score_ratio,
        "character_records": [dict(character) for character in alphabetic],
        "thresholds": {
            "edge_character_count": RETAINED_WORD_EDGE_CHARACTER_COUNT,
            "local_context_character_count": RETAINED_WORD_LOCAL_CONTEXT_CHARACTER_COUNT,
            "minimum_word_score": MIN_RETAINED_WORD_SCORE,
            "minimum_edge_character_score": MIN_RETAINED_EDGE_CHARACTER_SCORE,
            "minimum_edge_to_context_ratio": MIN_RETAINED_EDGE_TO_CONTEXT_RATIO,
        },
    }


def _alignment_spans(
    *,
    job: dict[str, Any],
    worker_job: dict[str, Any],
    sample_rate: int,
    total_samples: int,
) -> dict[int, dict[str, Any]]:
    if worker_job.get("error"):
        raise ValueError(str(worker_job["error"]))
    aligned = worker_job.get("aligned")
    if not isinstance(aligned, dict):
        raise ValueError("alignment worker returned no aligned result")
    local_words = job.get("local_words")
    if not isinstance(local_words, list) or not local_words:
        raise ValueError("alignment context has no local words")
    aligned_words = _flatten_aligned_words(aligned)
    if len(aligned_words) != len(local_words):
        raise ValueError(
            "aligned word count does not match the local transcript: "
            f"{len(aligned_words)} != {len(local_words)}"
        )
    character_groups = _flatten_character_word_groups(aligned)
    use_characters = len(character_groups) == len(local_words)
    crop_start = float(job["crop_start_seconds"])
    crop_end = float(job["crop_end_seconds"])
    spans: dict[int, dict[str, Any]] = {}
    for index, (local_word, aligned_word) in enumerate(
        zip(local_words, aligned_words, strict=True)
    ):
        relative: tuple[float, float] | None = None
        granularity = "words"
        character_records: list[dict[str, Any]] = []
        if use_characters:
            character_records, relative = _alignment_character_records(
                character_groups[index],
                crop_start=crop_start,
                sample_rate=sample_rate,
                total_samples=total_samples,
            )
            if relative is not None:
                granularity = "characters"
        aligned_word_start = _finite_number(aligned_word.get("start"))
        aligned_word_end = _finite_number(aligned_word.get("end"))
        word_score = _finite_number(aligned_word.get("score"))
        if relative is None:
            if (
                aligned_word_start is None
                or aligned_word_end is None
                or word_score is None
                or word_score <= 0.0
            ):
                raise ValueError(f"local word {index} has no positive alignment")
            relative = (aligned_word_start, aligned_word_end)
        absolute_start = crop_start + relative[0]
        absolute_end = crop_start + relative[1]
        if not crop_start <= absolute_start < absolute_end <= crop_end + 1e-6:
            raise ValueError(f"local word {index} alignment falls outside its crop")
        word_id = int(local_word["id"])
        expected_characters = [
            character.lower()
            for character in str(local_word["text"])
            if character.isalpha()
        ]
        alphabetic_records = [
            character
            for character in character_records
            if bool(character["is_alphabetic"])
        ]
        aligned_character_count = sum(
            _finite_number(character.get("start")) is not None
            and _finite_number(character.get("end")) is not None
            and _finite_number(character.get("score")) is not None
            for character in alphabetic_records
        )
        matching_character_count = sum(
            str(character.get("character", "")).lower() == expected
            and _finite_number(character.get("start")) is not None
            and _finite_number(character.get("end")) is not None
            and _finite_number(character.get("score")) is not None
            for character, expected in zip(
                alphabetic_records,
                expected_characters,
                strict=False,
            )
        )
        spans[word_id] = {
            "word_id": word_id,
            "text": str(local_word["text"]),
            "source_text": str(local_word["text"]),
            "start_seconds": absolute_start,
            "end_seconds": absolute_end,
            "aligned_start": (
                crop_start + aligned_word_start
                if aligned_word_start is not None
                else absolute_start
            ),
            "aligned_end": (
                crop_start + aligned_word_end
                if aligned_word_end is not None
                else absolute_end
            ),
            "start_sample": timestamp_to_sample(
                absolute_start,
                sample_rate=sample_rate,
                total_samples=total_samples,
                rounding="floor",
            ),
            "end_sample": timestamp_to_sample(
                absolute_end,
                sample_rate=sample_rate,
                total_samples=total_samples,
                rounding="ceil",
            ),
            "granularity": granularity,
            "word_score": word_score,
            "expected_alignable_character_count": len(expected_characters),
            "aligned_character_count": aligned_character_count,
            "character_coverage": (
                matching_character_count / len(expected_characters)
                if expected_characters
                else 0.0
            ),
            "characters": character_records,
            "initial_edge_score": _minimum_edge_score(
                alphabetic_records,
                edge="initial",
            ),
            "terminal_edge_score": _minimum_edge_score(
                alphabetic_records,
                edge="terminal",
            ),
            "local_context_score": None,
        }
    context_spans = list(spans.values())
    for span in context_spans:
        context_scores = _nearby_context_scores(
            span=span,
            context_spans=context_spans,
        )
        span["local_context_score"] = (
            median(context_scores)
            if context_scores
            else _finite_number(span.get("word_score"))
        )
    return spans


def _thought_bounds(plan: dict[str, Any]) -> list[tuple[int, int]]:
    thoughts = plan.get("committed")
    if not isinstance(thoughts, list) or not thoughts:
        raise FinalRenderError("semantic plan contains no committed thoughts")
    bounds: list[tuple[int, int]] = []
    for thought_index, thought in enumerate(thoughts):
        if not isinstance(thought, dict) or not isinstance(
            thought.get("source_ranges"), list
        ):
            raise FinalRenderError(f"committed thought {thought_index} is malformed")
        ranges = thought["source_ranges"]
        if not ranges:
            raise FinalRenderError(f"committed thought {thought_index} has no ranges")
        bounds.append(
            (
                int(ranges[0]["start_word_id"]),
                int(ranges[-1]["end_word_id"]),
            )
        )
    return bounds


def _word_owners(plan: dict[str, Any]) -> dict[int, int]:
    owners: dict[int, int] = {}
    for thought_index, thought in enumerate(plan["committed"]):
        for source_range in thought["source_ranges"]:
            for word_id in range(
                int(source_range["start_word_id"]),
                int(source_range["end_word_id"]),
            ):
                if word_id in owners:
                    raise FinalRenderError(
                        f"selected word {word_id} belongs to multiple thoughts"
                    )
                owners[word_id] = thought_index
    return owners


def _range_index_for_word(ranges: Sequence[MergedRange], word_id: int) -> int:
    for index, source_range in enumerate(ranges):
        if source_range.start_word_id <= word_id < source_range.end_word_id:
            return index
    raise FinalRenderError(f"selected word {word_id} belongs to no merged range")


def _alignment_event_specs(
    *,
    words: Sequence[PlanWord],
    ranges: Sequence[MergedRange],
    thought_bounds: Sequence[tuple[int, int]],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if ranges[0].start_word_id > 0:
        specs.append(
            {
                "event_key": "leading_source_cut",
                "event_kind": "leading_source_cut",
                "range_index": 0,
                "role_word_ids": {
                    "previous_omitted": ranges[0].start_word_id - 1,
                    "first_retained_right": ranges[0].start_word_id,
                },
            }
        )
    for gap_index, (left, right) in enumerate(zip(ranges, ranges[1:])):
        if right.start_word_id <= left.end_word_id:
            raise FinalRenderError("merged source ranges are not disjoint")
        specs.append(
            {
                "event_key": f"source_gap_{gap_index:04d}",
                "event_kind": "source_gap",
                "left_range_index": gap_index,
                "right_range_index": gap_index + 1,
                "role_word_ids": {
                    "last_retained_left": left.end_word_id - 1,
                    "first_omitted": left.end_word_id,
                    "last_omitted": right.start_word_id - 1,
                    "first_retained_right": right.start_word_id,
                },
            }
        )
    if ranges[-1].end_word_id < len(words):
        specs.append(
            {
                "event_key": "trailing_source_cut",
                "event_kind": "trailing_source_cut",
                "range_index": len(ranges) - 1,
                "role_word_ids": {
                    "last_retained_left": ranges[-1].end_word_id - 1,
                    "first_omitted": ranges[-1].end_word_id,
                },
            }
        )
    for transition_index, (left, right) in enumerate(
        zip(thought_bounds, thought_bounds[1:])
    ):
        left_word = left[1] - 1
        right_word = right[0]
        if right_word != left_word + 1:
            continue
        left_range = _range_index_for_word(ranges, left_word)
        right_range = _range_index_for_word(ranges, right_word)
        if left_range != right_range:
            continue
        specs.append(
            {
                "event_key": f"internal_thought_gap_{transition_index:04d}",
                "event_kind": "internal_thought_gap",
                "transition_index": transition_index,
                "range_index": left_range,
                "role_word_ids": {
                    "previous_retained": left_word,
                    "next_retained": right_word,
                },
            }
        )
    return specs


def _prepare_alignment_jobs(
    *,
    specs: Sequence[dict[str, Any]],
    words: Sequence[PlanWord],
    source_audio: np.ndarray,
    sample_rate: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    context_samples = round(CROP_CONTEXT_MS * sample_rate / 1000.0)
    total_samples = len(source_audio)
    contexts_dir = output_dir / "alignment_contexts"
    if specs:
        contexts_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for context_index, spec in enumerate(specs):
        role_ids = [int(value) for value in spec["role_word_ids"].values()]
        context_start_id = max(0, min(role_ids) - CONTEXT_WORDS_PER_SIDE)
        context_end_id = min(len(words), max(role_ids) + CONTEXT_WORDS_PER_SIDE + 1)
        crop_start = max(
            0,
            math.floor(words[context_start_id].start * sample_rate) - context_samples,
        )
        crop_end = min(
            total_samples,
            math.ceil(words[context_end_id - 1].end * sample_rate) + context_samples,
        )
        while context_start_id > 0:
            previous = words[context_start_id - 1]
            if math.ceil(previous.end * sample_rate) <= crop_start:
                break
            context_start_id -= 1
            crop_start = min(crop_start, math.floor(previous.start * sample_rate))
        while context_end_id < len(words):
            following = words[context_end_id]
            if math.floor(following.start * sample_rate) >= crop_end:
                break
            crop_end = max(crop_end, math.ceil(following.end * sample_rate))
            context_end_id += 1
        if crop_end <= crop_start:
            raise FinalRenderError(f"alignment context {context_index} is empty")
        context_path = contexts_dir / f"context_{context_index:04d}.wav"
        sf.write(
            context_path,
            source_audio[crop_start:crop_end],
            sample_rate,
            subtype="FLOAT",
        )
        local_words = [
            {
                "id": words[word_id].id,
                "text": words[word_id].text,
                "start": words[word_id].start,
                "end": words[word_id].end,
            }
            for word_id in range(context_start_id, context_end_id)
        ]
        jobs.append(
            {
                **spec,
                "clip_index": context_index,
                "context_index": context_index,
                "context_start_word_id": context_start_id,
                "context_end_word_id": context_end_id,
                "role_local_indices": {
                    role: word_id - context_start_id
                    for role, word_id in spec["role_word_ids"].items()
                },
                "local_words": local_words,
                "local_source_text": " ".join(
                    str(word["text"]) for word in local_words
                ),
                "crop_start_sample": crop_start,
                "crop_end_sample": crop_end,
                "crop_start_seconds": crop_start / sample_rate,
                "crop_end_seconds": crop_end / sample_rate,
                "crop_duration_seconds": (crop_end - crop_start) / sample_rate,
                "crop_wav": str(context_path.resolve()),
            }
        )
    return jobs


def _run_alignment_worker(
    *,
    jobs_path: Path,
    result_path: Path,
    alignment_python: Path,
    log_path: Path,
) -> dict[str, Any]:
    executable = _absolute_without_resolving_symlinks(alignment_python)
    if not executable.is_file():
        raise FileNotFoundError(
            "alignment Python does not exist; install VoiceCut's audio "
            f"dependencies or pass --alignment-python: {executable}"
        )
    command = [
        str(executable),
        "-m",
        "voicecut.forced_align_worker",
        "--jobs",
        str(jobs_path),
        "--output",
        str(result_path),
        "--language",
        ALIGNMENT_LANGUAGE,
        "--device",
        "cpu",
    ]
    environment = os.environ.copy()
    source_dir = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = source_dir + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    process = subprocess.run(command, text=True, capture_output=True, env=environment)
    log_path.write_text(
        "$ "
        + " ".join(command)
        + "\n\n[stdout]\n"
        + process.stdout
        + "\n[stderr]\n"
        + process.stderr,
        encoding="utf-8",
    )
    if not result_path.is_file():
        raise FinalRenderError(
            f"WhisperX alignment worker produced no result; see {log_path}"
        )
    result = read_json(result_path)
    if not isinstance(result, dict):
        raise FinalRenderError("WhisperX result root must be an object")
    return result


def _protected_intervals(
    *,
    spec: dict[str, Any],
    spans: dict[int, dict[str, Any]],
    sample_rate: int,
    total_samples: int,
) -> list[dict[str, Any]]:
    margin = round(PROTECTED_SPEECH_MARGIN_MS * sample_rate / 1000.0)
    ordered_spans = sorted(spans.values(), key=lambda item: int(item["start_sample"]))
    protected: list[dict[str, Any]] = []
    for role, raw_word_id in spec["role_word_ids"].items():
        word_id = int(raw_word_id)
        span = spans.get(word_id)
        if span is None:
            raise ValueError(f"required role {role} word {word_id} was not aligned")
        start = int(span["start_sample"])
        end = int(span["end_sample"])
        margin_start = max(0, start - margin)
        margin_end = min(total_samples, end + margin)
        for neighbor in ordered_spans:
            neighbor_start = int(neighbor["start_sample"])
            neighbor_end = int(neighbor["end_sample"])
            if neighbor_end <= start and neighbor_end > margin_start:
                margin_start = neighbor_end
            if neighbor_start >= end and neighbor_start < margin_end:
                margin_end = neighbor_start
        protected.append(
            {
                "role": role,
                "word_id": word_id,
                "text": span["text"],
                "start_sample": start,
                "end_sample": end,
                "start_seconds": start / sample_rate,
                "end_seconds": end / sample_rate,
                "margin_start_sample": margin_start,
                "margin_end_sample": margin_end,
                "protected_margin_ms": PROTECTED_SPEECH_MARGIN_MS,
                "alignment_granularity": span["granularity"],
            }
        )
    return protected


def _protected_by_role(
    protected: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {str(item["role"]): item for item in protected}


def _quiet_payload(evidence: VerifiedQuietEvidence | None) -> dict[str, Any] | None:
    if evidence is None:
        return None
    return {
        "start_sample": evidence.quiet_start_sample,
        "end_sample": evidence.quiet_end_sample,
        "local_noise_floor_db": evidence.local_noise_floor_db,
        "silence_threshold_db": evidence.silence_threshold_db,
        "minimum_quiet_ms": evidence.minimum_quiet_ms,
        "rms_frame_ms": evidence.frame_ms,
        "rms_hop_ms": evidence.hop_ms,
        "verification": "forced_alignment_gap_plus_stable_rms_quiet",
    }


def _resolve_aligned_cut(
    *,
    boundary_id: str,
    boundary_kind: str,
    spec: dict[str, Any],
    spans: dict[int, dict[str, Any]] | None,
    alignment_error: str | None,
    words: Sequence[PlanWord],
    mono: np.ndarray,
    sample_rate: int,
    total_samples: int,
    retained_role: str,
    omitted_role: str,
    direction: str,
) -> dict[str, Any]:
    role_ids = {key: int(value) for key, value in spec["role_word_ids"].items()}
    retained_id = role_ids[retained_role]
    omitted_id = role_ids[omitted_role]
    whisper = {
        "retained_start_seconds": words[retained_id].start,
        "retained_end_seconds": words[retained_id].end,
        "omitted_start_seconds": words[omitted_id].start,
        "omitted_end_seconds": words[omitted_id].end,
    }
    common = {
        "boundary_id": boundary_id,
        "boundary_kind": boundary_kind,
        "alignment_context_id": spec["event_key"],
        "source_word_ids": role_ids,
        "whisper_timestamps": whisper,
        "protected_speech_intervals": [],
        "aligned_timestamps": None,
        "verified_quiet_interval": None,
        "selected_source_sample": None,
        "selected_source_seconds": None,
        "fade_intervals": [],
        "retained_word_support": None,
        "forbidden_word_ids": [],
        "forbidden_source_edges": [],
        "failure_reason": None,
    }
    if spans is None:
        return {
            **common,
            "boundary_method": "forced_alignment_failed",
            "safety_status": "alignment_failed",
            "failure_reason": "alignment_failed",
            "error": alignment_error or "alignment evidence is missing",
        }
    try:
        protected = _protected_intervals(
            spec=spec,
            spans=spans,
            sample_rate=sample_rate,
            total_samples=total_samples,
        )
        protected_roles = _protected_by_role(protected)
        retained = protected_roles[retained_role]
        omitted = protected_roles[omitted_role]
        retained_span = spans[retained_id]
        omitted_span = spans[omitted_id]
        support_edge = "terminal" if direction == "trailing" else "initial"
        retained_support = evaluate_retained_word_support(
            retained_span,
            list(spans.values()),
            edge=support_edge,
        )
        aligned_timestamps = {
            "retained_start_seconds": retained_span["start_seconds"],
            "retained_end_seconds": retained_span["end_seconds"],
            "omitted_start_seconds": omitted_span["start_seconds"],
            "omitted_end_seconds": omitted_span["end_seconds"],
        }
        if retained_support["status"] != "supported_complete_word":
            return {
                **common,
                "protected_speech_intervals": protected,
                "aligned_timestamps": aligned_timestamps,
                "retained_word_support": retained_support,
                "forbidden_word_ids": [retained_id],
                "boundary_method": "retained_word_acoustic_support_failed",
                "safety_status": "weak_retained_word_alignment",
                "failure_reason": retained_support["status"],
                "character_scores": [
                    character.get("score")
                    for character in retained_support["character_records"]
                ],
                "edge_score": retained_support["minimum_edge_score"],
                "local_context_score": retained_support["local_context_median_score"],
                "score_ratio": retained_support["edge_to_context_score_ratio"],
                "error": (
                    f"retained word {retained_id} "
                    f"{words[retained_id].text!r} failed acoustic support: "
                    f"{retained_support['status']}"
                ),
            }
        if direction == "trailing":
            search_start = int(retained["margin_end_sample"])
            search_end = int(omitted["start_sample"])
            preference = "left"
        elif direction == "leading":
            search_start = int(omitted["end_sample"])
            search_end = int(retained["margin_start_sample"])
            preference = "right"
        else:
            raise ValueError("boundary direction must be leading or trailing")
        evidence = find_verified_quiet_evidence(
            mono,
            search_start_sample=search_start,
            search_end_sample=search_end,
            sample_rate=sample_rate,
            alignment_interval_verified=True,
            minimum_quiet_ms=MINIMUM_VERIFIED_QUIET_MS,
            preference=preference,
        )
        if evidence is None:
            return {
                **common,
                "protected_speech_intervals": protected,
                "aligned_timestamps": aligned_timestamps,
                "retained_word_support": retained_support,
                "forbidden_source_edges": [
                    {
                        "boundary_kind": boundary_kind,
                        "retained_word_id": retained_id,
                        "omitted_word_id": omitted_id,
                    }
                ],
                "boundary_method": "unsafe_dense_boundary",
                "safety_status": "unsafe_dense_boundary",
                "failure_reason": "no_verified_non_speech_interval",
                "error": (
                    "forced alignment exposes no verified non-speech interval "
                    "outside the retained-word protection margin"
                ),
            }
        fade_samples = round(QUIET_FADE_MS * sample_rate / 1000.0)
        if direction == "trailing":
            selected = evidence.quiet_end_sample
            fade_start = max(evidence.quiet_start_sample, selected - fade_samples)
            fade_end = selected
            curve = "fade_out"
        else:
            selected = evidence.quiet_start_sample
            fade_start = selected
            fade_end = min(evidence.quiet_end_sample, selected + fade_samples)
            curve = "fade_in"
        fades = (
            [
                {
                    "source_start_sample": fade_start,
                    "source_end_sample": fade_end,
                    "curve": curve,
                    "verified_quiet": True,
                }
            ]
            if fade_end > fade_start
            else []
        )
        return {
            **common,
            "protected_speech_intervals": protected,
            "aligned_timestamps": aligned_timestamps,
            "verified_quiet_interval": _quiet_payload(evidence),
            "selected_source_sample": selected,
            "selected_source_seconds": selected / sample_rate,
            "fade_intervals": fades,
            "boundary_method": "forced_alignment_verified_quiet",
            "safety_status": "safe",
            "retained_word_support": retained_support,
            "error": None,
        }
    except Exception as error:
        return {
            **common,
            "boundary_method": "forced_alignment_failed",
            "safety_status": "alignment_failed",
            "failure_reason": "alignment_failed",
            "error": f"{type(error).__name__}: {error}",
        }


def _source_edge_boundary(
    *,
    boundary_id: str,
    boundary_kind: str,
    selected_sample: int,
    sample_rate: int,
) -> dict[str, Any]:
    return {
        "boundary_id": boundary_id,
        "boundary_kind": boundary_kind,
        "alignment_context_id": None,
        "source_word_ids": {},
        "whisper_timestamps": None,
        "aligned_timestamps": None,
        "protected_speech_intervals": [],
        "verified_quiet_interval": None,
        "selected_source_sample": selected_sample,
        "selected_source_seconds": selected_sample / sample_rate,
        "fade_intervals": [],
        "retained_word_support": None,
        "forbidden_word_ids": [],
        "forbidden_source_edges": [],
        "failure_reason": None,
        "boundary_method": boundary_kind,
        "safety_status": "safe",
        "error": None,
    }


def _pause_plan_transitions(
    *,
    pause_plan: dict[str, Any],
    thought_count: int,
) -> list[dict[str, Any]]:
    if pause_plan.get("planner") != "semantic_pause_planner_v1":
        raise FinalRenderError("pause plan has an unsupported planner")
    transitions = pause_plan.get("transitions")
    if not isinstance(transitions, list):
        raise FinalRenderError("pause plan contains no transitions")
    validate_pause_response(
        json.dumps({"transitions": transitions}),
        thought_count=thought_count,
    )
    return transitions


def _ranges_from_selection(
    plan: dict[str, Any],
    *,
    word_count: int,
) -> list[MergedRange]:
    return merge_adjacent_ranges(flatten_selected_ranges(plan, word_count=word_count))


def _collect_alignment_evidence(
    *,
    jobs: Sequence[dict[str, Any]],
    worker_result: dict[str, Any],
    sample_rate: int,
    total_samples: int,
) -> list[dict[str, Any]]:
    raw_worker_jobs = worker_result.get("jobs")
    if not isinstance(raw_worker_jobs, list):
        raise FinalRenderError("alignment worker returned no jobs list")
    worker_by_index = {
        int(item["clip_index"]): item
        for item in raw_worker_jobs
        if isinstance(item, dict) and type(item.get("clip_index")) is int
    }
    evidence: list[dict[str, Any]] = []
    for job in jobs:
        context_index = int(job["context_index"])
        worker_job = worker_by_index.get(
            context_index,
            {
                "clip_index": context_index,
                "aligned": None,
                "error": "alignment worker omitted this context",
            },
        )
        try:
            spans = _alignment_spans(
                job=job,
                worker_job=worker_job,
                sample_rate=sample_rate,
                total_samples=total_samples,
            )
            error = None
            status = "aligned"
        except Exception as alignment_error:
            spans = None
            error = f"{type(alignment_error).__name__}: {alignment_error}"
            status = "alignment_failed"
        evidence.append(
            {
                "context_index": context_index,
                "context_id": job["event_key"],
                "context_kind": job["event_kind"],
                "crop_wav": job["crop_wav"],
                "crop_start_sample": job["crop_start_sample"],
                "crop_end_sample": job["crop_end_sample"],
                "crop_start_seconds": job["crop_start_seconds"],
                "crop_end_seconds": job["crop_end_seconds"],
                "local_source_text": job["local_source_text"],
                "local_words": job["local_words"],
                "role_word_ids": job["role_word_ids"],
                "aligned_word_spans": (
                    [spans[word_id] for word_id in sorted(spans)]
                    if spans is not None
                    else []
                ),
                "status": status,
                "error": error,
                "_spans": spans,
                "_spec": job,
            }
        )
    return evidence


def _public_alignment_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in context.items() if key not in {"_spans", "_spec"}
    }


def _room_tone_ranges(
    *,
    allocator: SourceRoomToneAllocator,
    frame_count: int,
    reference_sample: int,
) -> tuple[list[dict[str, int]], int]:
    if frame_count <= 0:
        return [], 0
    _, selection = allocator.allocate(
        frame_count=frame_count,
        reference_sample=reference_sample,
    )
    if selection is None:
        raise FinalRenderError("positive room-tone allocation returned no source")
    ranges = [
        {"source_start_sample": start, "source_end_sample": end}
        for start, end in selection.source_ranges
    ]
    if (
        sum(item["source_end_sample"] - item["source_start_sample"] for item in ranges)
        != frame_count
    ):
        raise FinalRenderError("room-tone source ranges have the wrong duration")
    return ranges, allocator.fade_samples


def _append_source_segment(
    segments: list[dict[str, Any]],
    *,
    source_start: int,
    source_end: int,
    output_cursor: int,
    fades: Sequence[dict[str, Any]],
    source_interval_index: int,
) -> int:
    if source_end <= source_start:
        return output_cursor
    local_fades = [
        dict(fade)
        for fade in fades
        if int(fade["source_start_sample"]) >= source_start
        and int(fade["source_end_sample"]) <= source_end
    ]
    output_end = output_cursor + source_end - source_start
    segments.append(
        {
            "segment_index": len(segments),
            "kind": "source",
            "source_interval_index": source_interval_index,
            "source_start_sample": source_start,
            "source_end_sample": source_end,
            "output_start_sample": output_cursor,
            "output_end_sample": output_end,
            "gain_envelopes": local_fades,
        }
    )
    return output_end


def _append_room_tone_segments(
    segments: list[dict[str, Any]],
    *,
    source_ranges: Sequence[dict[str, int]],
    output_cursor: int,
    fade_samples: int,
    join_id: str,
) -> int:
    for source_range in source_ranges:
        source_start = int(source_range["source_start_sample"])
        source_end = int(source_range["source_end_sample"])
        frame_count = source_end - source_start
        if frame_count <= 0:
            raise FinalRenderError("room-tone source range is empty")
        actual_fade = min(fade_samples, frame_count // 2)
        output_end = output_cursor + frame_count
        segments.append(
            {
                "segment_index": len(segments),
                "kind": "room_tone",
                "join_id": join_id,
                "source_start_sample": source_start,
                "source_end_sample": source_end,
                "output_start_sample": output_cursor,
                "output_end_sample": output_end,
                "fade_in_samples": actual_fade,
                "fade_out_samples": actual_fade,
            }
        )
        output_cursor = output_end
    return output_cursor


def _interval_fades(
    *,
    interval: dict[str, Any],
    boundary_by_id: dict[str, dict[str, Any]],
    internal_joins: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    fades: list[dict[str, Any]] = []
    for key in ("start_boundary_id", "end_boundary_id"):
        boundary = boundary_by_id[str(interval[key])]
        fades.extend(boundary["fade_intervals"])
    for join in internal_joins:
        fades.extend(join.get("fade_intervals", []))
    return sorted(fades, key=lambda item: int(item["source_start_sample"]))


def _build_output_segments(
    *,
    source_intervals: list[dict[str, Any]],
    boundaries: Sequence[dict[str, Any]],
    joins: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    boundary_by_id = {str(boundary["boundary_id"]): boundary for boundary in boundaries}
    internal_by_interval: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(len(source_intervals))
    }
    clip_join_by_left: dict[int, dict[str, Any]] = {}
    for join in joins:
        if join["join_kind"] == "internal_thought_pause":
            internal_by_interval[int(join["source_interval_index"])].append(join)
        elif join["join_kind"] == "source_discontinuity":
            clip_join_by_left[int(join["left_source_interval_index"])] = join

    output_segments: list[dict[str, Any]] = []
    output_cursor = 0
    for interval_index, interval in enumerate(source_intervals):
        interval["output_start_sample"] = output_cursor
        internal = sorted(
            internal_by_interval[interval_index],
            key=lambda item: int(item.get("source_insertion_sample") or -1),
        )
        fades = _interval_fades(
            interval=interval,
            boundary_by_id=boundary_by_id,
            internal_joins=internal,
        )
        source_cursor = int(interval["source_start_sample"])
        for join in internal:
            if int(join["inserted_pause_samples"]) <= 0:
                continue
            insertion = int(join["source_insertion_sample"])
            output_cursor = _append_source_segment(
                output_segments,
                source_start=source_cursor,
                source_end=insertion,
                output_cursor=output_cursor,
                fades=fades,
                source_interval_index=interval_index,
            )
            pause_start = output_cursor
            output_cursor = _append_room_tone_segments(
                output_segments,
                source_ranges=join["room_tone_source_ranges"],
                output_cursor=output_cursor,
                fade_samples=int(join["room_tone_fade_samples"]),
                join_id=str(join["join_id"]),
            )
            join["output_pause_start_sample"] = pause_start
            join["output_pause_end_sample"] = output_cursor
            source_cursor = insertion
        output_cursor = _append_source_segment(
            output_segments,
            source_start=source_cursor,
            source_end=int(interval["source_end_sample"]),
            output_cursor=output_cursor,
            fades=fades,
            source_interval_index=interval_index,
        )
        interval["output_end_sample"] = output_cursor
        if interval_index < len(source_intervals) - 1:
            join = clip_join_by_left[interval_index]
            pause_start = output_cursor
            output_cursor = _append_room_tone_segments(
                output_segments,
                source_ranges=join["room_tone_source_ranges"],
                output_cursor=output_cursor,
                fade_samples=int(join["room_tone_fade_samples"]),
                join_id=str(join["join_id"]),
            )
            join["output_pause_start_sample"] = pause_start
            join["output_pause_end_sample"] = output_cursor
    return output_segments


def build_final_boundary_plan(
    *,
    audio_path: Path,
    semantic_plan: dict[str, Any],
    semantic_plan_path: Path,
    pause_plan: dict[str, Any],
    pause_plan_path: Path,
    output_dir: Path,
    alignment_python: Path,
    alignment_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve every source boundary before rendering any output waveform."""

    source_audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )
    sample_rate = int(sample_rate)
    total_samples, channel_count = source_audio.shape
    mono = np.mean(source_audio, axis=1, dtype=np.float64).astype(np.float32)
    words = load_plan_words(semantic_plan)
    ranges = _ranges_from_selection(semantic_plan, word_count=len(words))
    thought_bounds = _thought_bounds(semantic_plan)
    transitions = _pause_plan_transitions(
        pause_plan=pause_plan,
        thought_count=len(thought_bounds),
    )
    if sha256_file(semantic_plan_path) != pause_plan.get("streaming_plan_sha256"):
        raise FinalRenderError("pause plan belongs to a different semantic plan")

    specs = _alignment_event_specs(
        words=words,
        ranges=ranges,
        thought_bounds=thought_bounds,
    )
    jobs = _prepare_alignment_jobs(
        specs=specs,
        words=words,
        source_audio=source_audio,
        sample_rate=sample_rate,
        output_dir=output_dir,
    )
    jobs_path = output_dir / "alignment_jobs.json"
    write_json(
        jobs_path,
        {
            "schema_version": 1,
            "source_audio": str(audio_path),
            "source_audio_sha256": sha256_file(audio_path),
            "language": ALIGNMENT_LANGUAGE,
            "device": "cpu",
            "jobs": jobs,
        },
    )
    worker_path = output_dir / "alignment_worker_result.json"
    if alignment_payload is not None:
        worker_result = alignment_payload
        write_json(worker_path, worker_result)
    elif jobs:
        worker_result = _run_alignment_worker(
            jobs_path=jobs_path,
            result_path=worker_path,
            alignment_python=alignment_python,
            log_path=output_dir / "alignment_worker.log",
        )
    else:
        worker_result = {
            "schema_version": 1,
            "backend": "whisperx_alignment",
            "language": ALIGNMENT_LANGUAGE,
            "device": "cpu",
            "jobs": [],
            "model_load_skipped": "no source cuts or internal pauses need alignment",
        }
        write_json(worker_path, worker_result)
    contexts = _collect_alignment_evidence(
        jobs=jobs,
        worker_result=worker_result,
        sample_rate=sample_rate,
        total_samples=total_samples,
    )
    context_by_key = {str(context["context_id"]): context for context in contexts}

    boundaries: list[dict[str, Any]] = []
    source_intervals: list[dict[str, Any]] = []
    for range_index, source_range in enumerate(ranges):
        if range_index == 0 and source_range.start_word_id == 0:
            start_boundary = _source_edge_boundary(
                boundary_id="source_start",
                boundary_kind="source_start",
                selected_sample=0,
                sample_rate=sample_rate,
            )
        else:
            if range_index == 0:
                context_key = "leading_source_cut"
                retained_role = "first_retained_right"
                omitted_role = "previous_omitted"
            else:
                context_key = f"source_gap_{range_index - 1:04d}"
                retained_role = "first_retained_right"
                omitted_role = "last_omitted"
            context = context_by_key[context_key]
            start_boundary = _resolve_aligned_cut(
                boundary_id=f"range_{range_index:04d}_start",
                boundary_kind="omitted_to_selected",
                spec=context["_spec"],
                spans=context["_spans"],
                alignment_error=context["error"],
                words=words,
                mono=mono,
                sample_rate=sample_rate,
                total_samples=total_samples,
                retained_role=retained_role,
                omitted_role=omitted_role,
                direction="leading",
            )
        boundaries.append(start_boundary)

        if range_index == len(ranges) - 1 and source_range.end_word_id == len(words):
            raw_end = words[source_range.end_word_id - 1].end
            raw_end_sample = timestamp_to_sample(
                raw_end,
                sample_rate=sample_rate,
                total_samples=total_samples,
                rounding="ceil",
            )
            eof = refine_eof_tail(
                mono,
                sample_rate=sample_rate,
                raw_end_seconds=raw_end,
                previous_end_sample=raw_end_sample,
                fade_ms=QUIET_FADE_MS,
            )
            eof_fades: list[dict[str, Any]] = []
            if eof.fade_out_samples:
                fade_start = eof.new_end_sample - eof.fade_out_samples
                if (
                    eof.stable_silence_start_sample is None
                    or fade_start < eof.stable_silence_start_sample
                ):
                    raise FinalRenderError("EOF fade is not inside verified silence")
                eof_fades.append(
                    {
                        "source_start_sample": fade_start,
                        "source_end_sample": eof.new_end_sample,
                        "curve": "fade_out",
                        "verified_quiet": True,
                    }
                )
            end_boundary = {
                **_source_edge_boundary(
                    boundary_id="end_of_file",
                    boundary_kind="end_of_file",
                    selected_sample=eof.new_end_sample,
                    sample_rate=sample_rate,
                ),
                "source_word_ids": {"last_retained_left": source_range.end_word_id - 1},
                "whisper_timestamps": {"retained_end_seconds": raw_end},
                "verified_quiet_interval": (
                    {
                        "start_sample": eof.stable_silence_start_sample,
                        "end_sample": eof.new_end_sample,
                        "local_noise_floor_db": eof.local_noise_floor_db,
                        "silence_threshold_db": eof.silence_threshold_db,
                        "verification": "safe_eof_tail",
                    }
                    if eof.stable_silence_start_sample is not None
                    else None
                ),
                "fade_intervals": eof_fades,
                "boundary_method": "eof_safe_tail",
            }
        else:
            if range_index < len(ranges) - 1:
                context_key = f"source_gap_{range_index:04d}"
            else:
                context_key = "trailing_source_cut"
            context = context_by_key[context_key]
            end_boundary = _resolve_aligned_cut(
                boundary_id=f"range_{range_index:04d}_end",
                boundary_kind="selected_to_omitted",
                spec=context["_spec"],
                spans=context["_spans"],
                alignment_error=context["error"],
                words=words,
                mono=mono,
                sample_rate=sample_rate,
                total_samples=total_samples,
                retained_role="last_retained_left",
                omitted_role="first_omitted",
                direction="trailing",
            )
        boundaries.append(end_boundary)
        source_intervals.append(
            {
                "source_interval_index": range_index,
                "start_word_id": source_range.start_word_id,
                "end_word_id": source_range.end_word_id,
                "start_boundary_id": start_boundary["boundary_id"],
                "end_boundary_id": end_boundary["boundary_id"],
                "source_start_sample": start_boundary["selected_source_sample"],
                "source_end_sample": end_boundary["selected_source_sample"],
                "merged_original_ranges": [
                    {
                        "start_word_id": original.start_word_id,
                        "end_word_id": original.end_word_id,
                        "thought_index": original.thought_index,
                    }
                    for original in source_range.original_ranges
                ],
            }
        )

    unsafe_boundaries = [
        boundary for boundary in boundaries if boundary["safety_status"] != "safe"
    ]
    for interval in source_intervals:
        start = interval["source_start_sample"]
        end = interval["source_end_sample"]
        if (
            start is not None
            and end is not None
            and not 0 <= start < end <= total_samples
        ):
            unsafe_boundaries.append(
                {
                    "boundary_id": f"interval_{interval['source_interval_index']}",
                    "safety_status": "invalid_source_interval",
                }
            )

    owners = _word_owners(semantic_plan)
    for boundary in boundaries:
        role_ids = boundary.get("source_word_ids")
        retained_thought_indices: set[int] = set()
        if isinstance(role_ids, dict):
            for role, raw_word_id in role_ids.items():
                if "retained" not in str(role) or type(raw_word_id) is not int:
                    continue
                thought_index = owners.get(raw_word_id)
                if thought_index is not None:
                    retained_thought_indices.add(thought_index)
        boundary["retained_thought_indices"] = sorted(retained_thought_indices)
    transition_by_pair = {
        (
            int(transition["after_thought_index"]),
            int(transition["before_thought_index"]),
        ): transition
        for transition in transitions
    }
    boundary_by_id = {str(boundary["boundary_id"]): boundary for boundary in boundaries}
    joins: list[dict[str, Any]] = []
    for left_index in range(len(source_intervals) - 1):
        left_interval = source_intervals[left_index]
        right_interval = source_intervals[left_index + 1]
        left_boundary = boundary_by_id[str(left_interval["end_boundary_id"])]
        right_boundary = boundary_by_id[str(right_interval["start_boundary_id"])]
        left_word_id = int(left_interval["end_word_id"]) - 1
        right_word_id = int(right_interval["start_word_id"])
        left_thought = owners[left_word_id]
        right_thought = owners[right_word_id]
        if left_thought == right_thought:
            pause_type = "continuation"
        else:
            transition = transition_by_pair.get((left_thought, right_thought))
            if transition is None:
                raise FinalRenderError(
                    "source join has no semantic pause classification"
                )
            pause_type = str(transition["pause_type"])
        target_samples = round(PAUSE_TARGETS_MS[pause_type] * sample_rate / 1000.0)
        if (
            left_boundary["safety_status"] == "safe"
            and right_boundary["safety_status"] == "safe"
        ):
            left_retained = next(
                item
                for item in left_boundary["protected_speech_intervals"]
                if item["role"] == "last_retained_left"
            )
            right_retained = next(
                item
                for item in right_boundary["protected_speech_intervals"]
                if item["role"] == "first_retained_right"
            )
            existing_samples = max(
                0,
                int(left_boundary["selected_source_sample"])
                - int(left_retained["end_sample"]),
            ) + max(
                0,
                int(right_retained["start_sample"])
                - int(right_boundary["selected_source_sample"]),
            )
            inserted_samples = max(0, target_samples - existing_samples)
            safety_status = "safe"
        else:
            existing_samples = 0
            inserted_samples = 0
            safety_status = "unsafe_dense_boundary"
        joins.append(
            {
                "join_id": f"source_join_{left_index:04d}",
                "join_index": len(joins),
                "join_kind": "source_discontinuity",
                "left_source_interval_index": left_index,
                "right_source_interval_index": left_index + 1,
                "left_word_id": left_word_id,
                "right_word_id": right_word_id,
                "pause_type": pause_type,
                "target_pause_samples": target_samples,
                "target_pause_ms": target_samples * 1000.0 / sample_rate,
                "estimated_existing_pause_samples": existing_samples,
                "estimated_existing_pause_ms": existing_samples * 1000.0 / sample_rate,
                "inserted_pause_samples": inserted_samples,
                "inserted_pause_ms": inserted_samples * 1000.0 / sample_rate,
                "left_boundary_id": left_boundary["boundary_id"],
                "right_boundary_id": right_boundary["boundary_id"],
                "protected_speech_intervals": [
                    *left_boundary["protected_speech_intervals"],
                    *right_boundary["protected_speech_intervals"],
                ],
                "verified_quiet_intervals": [
                    left_boundary["verified_quiet_interval"],
                    right_boundary["verified_quiet_interval"],
                ],
                "fade_intervals": [
                    *left_boundary["fade_intervals"],
                    *right_boundary["fade_intervals"],
                ],
                "safety_status": safety_status,
                "insertion_method": "resolved_source_join",
                "room_tone_source_ranges": [],
                "room_tone_fade_samples": 0,
            }
        )

    for transition_index, transition in enumerate(transitions):
        context_key = f"internal_thought_gap_{transition_index:04d}"
        context = context_by_key.get(context_key)
        if context is None:
            continue
        spec = context["_spec"]
        spans = context["_spans"]
        pause_type = str(transition["pause_type"])
        target_samples = round(PAUSE_TARGETS_MS[pause_type] * sample_rate / 1000.0)
        common = {
            "join_id": context_key,
            "join_index": len(joins),
            "join_kind": "internal_thought_pause",
            "source_interval_index": int(spec["range_index"]),
            "after_thought_index": int(transition["after_thought_index"]),
            "before_thought_index": int(transition["before_thought_index"]),
            "pause_type": pause_type,
            "target_pause_samples": target_samples,
            "target_pause_ms": target_samples * 1000.0 / sample_rate,
            "source_insertion_sample": None,
            "verified_quiet_interval": None,
            "protected_speech_intervals": [],
            "fade_intervals": [],
            "room_tone_source_ranges": [],
            "room_tone_fade_samples": 0,
        }
        if spans is None:
            joins.append(
                {
                    **common,
                    "estimated_existing_pause_samples": 0,
                    "estimated_existing_pause_ms": 0.0,
                    "inserted_pause_samples": 0,
                    "inserted_pause_ms": 0.0,
                    "safety_status": "pause_not_inserted_no_safe_point",
                    "insertion_method": "none",
                    "error": context["error"],
                }
            )
            continue
        try:
            protected = _protected_intervals(
                spec=spec,
                spans=spans,
                sample_rate=sample_rate,
                total_samples=total_samples,
            )
            roles = _protected_by_role(protected)
            previous_span = spans[int(spec["role_word_ids"]["previous_retained"])]
            next_span = spans[int(spec["role_word_ids"]["next_retained"])]
            evidence = find_verified_quiet_evidence(
                mono,
                search_start_sample=int(
                    roles["previous_retained"]["margin_end_sample"]
                ),
                search_end_sample=int(roles["next_retained"]["margin_start_sample"]),
                sample_rate=sample_rate,
                alignment_interval_verified=True,
                minimum_quiet_ms=MINIMUM_VERIFIED_QUIET_MS,
                preference="longest",
            )
            existing_samples = max(
                0,
                int(next_span["start_sample"]) - int(previous_span["end_sample"]),
            )
            if evidence is None:
                joins.append(
                    {
                        **common,
                        "protected_speech_intervals": protected,
                        "estimated_existing_pause_samples": existing_samples,
                        "estimated_existing_pause_ms": (
                            existing_samples * 1000.0 / sample_rate
                        ),
                        "inserted_pause_samples": 0,
                        "inserted_pause_ms": 0.0,
                        "safety_status": "pause_not_inserted_no_safe_point",
                        "insertion_method": "none",
                        "error": "no aligned and verified natural inter-word gap",
                    }
                )
                continue
            inserted_samples = max(0, target_samples - existing_samples)
            insertion_sample = (
                evidence.quiet_start_sample + evidence.quiet_end_sample
            ) // 2
            fade_samples = round(QUIET_FADE_MS * sample_rate / 1000.0)
            fade_intervals = []
            if inserted_samples:
                fade_intervals = [
                    {
                        "source_start_sample": max(
                            evidence.quiet_start_sample,
                            insertion_sample - fade_samples,
                        ),
                        "source_end_sample": insertion_sample,
                        "curve": "fade_out",
                        "verified_quiet": True,
                    },
                    {
                        "source_start_sample": insertion_sample,
                        "source_end_sample": min(
                            evidence.quiet_end_sample,
                            insertion_sample + fade_samples,
                        ),
                        "curve": "fade_in",
                        "verified_quiet": True,
                    },
                ]
                fade_intervals = [
                    fade
                    for fade in fade_intervals
                    if fade["source_end_sample"] > fade["source_start_sample"]
                ]
            joins.append(
                {
                    **common,
                    "source_insertion_sample": insertion_sample,
                    "verified_quiet_interval": _quiet_payload(evidence),
                    "protected_speech_intervals": protected,
                    "fade_intervals": fade_intervals,
                    "estimated_existing_pause_samples": existing_samples,
                    "estimated_existing_pause_ms": (
                        existing_samples * 1000.0 / sample_rate
                    ),
                    "inserted_pause_samples": inserted_samples,
                    "inserted_pause_ms": inserted_samples * 1000.0 / sample_rate,
                    "safety_status": "safe",
                    "insertion_method": (
                        "aligned_verified_interword_gap"
                        if inserted_samples
                        else "none_existing_pause"
                    ),
                    "error": None,
                }
            )
        except Exception as error:
            joins.append(
                {
                    **common,
                    "estimated_existing_pause_samples": 0,
                    "estimated_existing_pause_ms": 0.0,
                    "inserted_pause_samples": 0,
                    "inserted_pause_ms": 0.0,
                    "safety_status": "pause_not_inserted_no_safe_point",
                    "insertion_method": "none",
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    plan_status = "unsafe" if unsafe_boundaries else "safe"
    output_segments: list[dict[str, Any]] = []
    if plan_status == "safe":
        allocator = SourceRoomToneAllocator(
            source_audio=source_audio,
            mono=mono,
            words=words,
            sample_rate=sample_rate,
        )
        for join in sorted(joins, key=lambda item: int(item["join_index"])):
            frame_count = int(join["inserted_pause_samples"])
            if frame_count <= 0:
                continue
            reference = (
                int(join["source_insertion_sample"])
                if join["join_kind"] == "internal_thought_pause"
                else int(
                    source_intervals[int(join["left_source_interval_index"])][
                        "source_end_sample"
                    ]
                )
            )
            ranges_payload, room_fade = _room_tone_ranges(
                allocator=allocator,
                frame_count=frame_count,
                reference_sample=reference,
            )
            join["room_tone_source_ranges"] = ranges_payload
            join["room_tone_fade_samples"] = room_fade
        output_segments = _build_output_segments(
            source_intervals=source_intervals,
            boundaries=boundaries,
            joins=joins,
        )

    expected_frames = (
        int(output_segments[-1]["output_end_sample"]) if output_segments else 0
    )
    final_boundary = boundaries[-1]
    return {
        "schema_version": 1,
        "planner": "authoritative_single_pass_boundary_plan_v1",
        "status": plan_status,
        "source_audio": str(audio_path),
        "source_audio_sha256": sha256_file(audio_path),
        "source_sample_rate": sample_rate,
        "source_channel_count": channel_count,
        "source_frame_count": total_samples,
        "streaming_plan": str(semantic_plan_path),
        "streaming_plan_sha256": sha256_file(semantic_plan_path),
        "pause_plan": str(pause_plan_path),
        "pause_plan_sha256": sha256_file(pause_plan_path),
        "configuration": {
            "alignment_backend": "whisperx_alignment",
            "alignment_language": ALIGNMENT_LANGUAGE,
            "alignment_device": "cpu",
            "protected_speech_margin_ms": PROTECTED_SPEECH_MARGIN_MS,
            "minimum_verified_quiet_ms": MINIMUM_VERIFIED_QUIET_MS,
            "quiet_fade_ms": QUIET_FADE_MS,
            "room_tone_fade_ms": ROOM_TONE_FADE_MS,
            "pause_targets_ms": PAUSE_TARGETS_MS,
            "retained_word_support": {
                "edge_character_count": RETAINED_WORD_EDGE_CHARACTER_COUNT,
                "local_context_character_count": (
                    RETAINED_WORD_LOCAL_CONTEXT_CHARACTER_COUNT
                ),
                "minimum_word_score": MIN_RETAINED_WORD_SCORE,
                "minimum_edge_character_score": (MIN_RETAINED_EDGE_CHARACTER_SCORE),
                "minimum_edge_to_context_ratio": (MIN_RETAINED_EDGE_TO_CONTEXT_RATIO),
            },
        },
        "alignment_jobs": str(jobs_path.resolve()),
        "alignment_worker_result": str(worker_path.resolve()),
        "alignment_contexts": [
            _public_alignment_context(context) for context in contexts
        ],
        "source_intervals": source_intervals,
        "boundaries": boundaries,
        "joins": joins,
        "output_segments": output_segments,
        "expected_output_frame_count": expected_frames,
        "alignment_context_count": len(contexts),
        "alignment_resolved_boundaries": sum(
            boundary["safety_status"] == "safe"
            and boundary["alignment_context_id"] is not None
            for boundary in boundaries
        ),
        "unsafe_dense_boundaries": sum(
            boundary["safety_status"] == "unsafe_dense_boundary"
            for boundary in boundaries
        ),
        "weak_retained_word_alignments": sum(
            boundary["safety_status"] == "weak_retained_word_alignment"
            for boundary in boundaries
        ),
        "alignment_failures": sum(
            boundary["safety_status"] == "alignment_failed" for boundary in boundaries
        ),
        "final_boundary": final_boundary,
    }


def _intervals_overlap(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> bool:
    return max(left_start, right_start) < min(left_end, right_end)


def _assert_boundary_plan_invariants(boundary_plan: dict[str, Any]) -> None:
    boundaries = boundary_plan.get("boundaries")
    joins = boundary_plan.get("joins")
    segments = boundary_plan.get("output_segments")
    if not isinstance(boundaries, list) or not isinstance(joins, list):
        raise FinalRenderError("boundary plan has no boundary/join ledger")
    if not isinstance(segments, list) or not segments:
        raise FinalRenderError("safe boundary plan has no output trace")
    boundary_ids: set[str] = set()
    for boundary in boundaries:
        boundary_id = str(boundary.get("boundary_id"))
        if boundary_id in boundary_ids:
            raise FinalRenderError(f"boundary {boundary_id} has multiple decisions")
        boundary_ids.add(boundary_id)
        if boundary.get("safety_status") != "safe":
            raise FinalRenderError(
                f"{boundary.get('safety_status')}: boundary {boundary_id} is unsafe"
            )
        selected = boundary.get("selected_source_sample")
        if type(selected) is not int:
            raise FinalRenderError(f"boundary {boundary_id} has no final sample")
        protected = boundary.get("protected_speech_intervals", [])
        fades = boundary.get("fade_intervals", [])
        for interval in protected:
            if "retained" not in str(interval.get("role", "")):
                continue
            start = int(interval["start_sample"])
            end = int(interval["end_sample"])
            if start < selected < end:
                raise FinalRenderError(
                    f"boundary {boundary_id} cuts aligned retained speech"
                )
            for fade in fades:
                if _intervals_overlap(
                    start,
                    end,
                    int(fade["source_start_sample"]),
                    int(fade["source_end_sample"]),
                ):
                    raise FinalRenderError(
                        f"boundary {boundary_id} fade overlaps retained speech"
                    )
    for join in joins:
        insertion = join.get("source_insertion_sample")
        if insertion is None:
            continue
        insertion = int(insertion)
        for interval in join.get("protected_speech_intervals", []):
            if int(interval["start_sample"]) <= insertion < int(interval["end_sample"]):
                raise FinalRenderError(
                    f"semantic pause {join.get('join_id')} lies inside aligned speech"
                )
        for fade in join.get("fade_intervals", []):
            for interval in join.get("protected_speech_intervals", []):
                if _intervals_overlap(
                    int(interval["start_sample"]),
                    int(interval["end_sample"]),
                    int(fade["source_start_sample"]),
                    int(fade["source_end_sample"]),
                ):
                    raise FinalRenderError(
                        f"semantic pause {join.get('join_id')} fade overlaps speech"
                    )
    cursor = 0
    for expected_index, segment in enumerate(segments):
        if segment.get("segment_index") != expected_index:
            raise FinalRenderError("output trace segment indices are not immutable")
        if segment.get("kind") not in {"source", "room_tone"}:
            raise FinalRenderError("output trace contains an untraceable segment")
        source_start = int(segment["source_start_sample"])
        source_end = int(segment["source_end_sample"])
        output_start = int(segment["output_start_sample"])
        output_end = int(segment["output_end_sample"])
        if (
            output_start != cursor
            or source_end - source_start != output_end - output_start
        ):
            raise FinalRenderError("output trace geometry is inconsistent")
        cursor = output_end
    if cursor != int(boundary_plan["expected_output_frame_count"]):
        raise FinalRenderError("output trace does not match planned frame count")


def _apply_source_gain_envelopes(
    chunk: np.ndarray,
    *,
    source_start: int,
    envelopes: Sequence[dict[str, Any]],
) -> np.ndarray:
    rendered = np.array(chunk, dtype=np.float32, copy=True)
    for envelope in envelopes:
        start = int(envelope["source_start_sample"]) - source_start
        end = int(envelope["source_end_sample"]) - source_start
        if not 0 <= start < end <= len(rendered):
            raise FinalRenderError("planned gain envelope leaves its source segment")
        if envelope["curve"] == "fade_in":
            gain = np.linspace(0.0, 1.0, end - start, dtype=np.float32)
        elif envelope["curve"] == "fade_out":
            gain = np.linspace(1.0, 0.0, end - start, dtype=np.float32)
        else:
            raise FinalRenderError("boundary plan contains an unknown fade curve")
        rendered[start:end] *= gain[:, None]
    return rendered


def render_boundary_plan(
    *,
    audio_path: Path,
    boundary_plan_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Materialize an immutable boundary plan once from canonical samples."""

    boundary_plan = read_json(boundary_plan_path)
    if not isinstance(boundary_plan, dict):
        raise FinalRenderError("final boundary plan root must be an object")
    if boundary_plan.get("planner") != "authoritative_single_pass_boundary_plan_v1":
        raise FinalRenderError("unsupported final boundary plan")
    if boundary_plan.get("status") != "safe":
        unsafe = boundary_plan.get("unsafe_dense_boundaries", 0)
        raise FinalRenderError(
            f"unsafe_dense_boundary: refusing to render {unsafe} unsafe boundaries"
        )
    if sha256_file(audio_path) != boundary_plan.get("source_audio_sha256"):
        raise FinalRenderError("canonical source changed after boundary planning")
    _assert_boundary_plan_invariants(boundary_plan)

    source_audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )
    sample_rate = int(sample_rate)
    if (
        sample_rate != int(boundary_plan["source_sample_rate"])
        or source_audio.shape[1] != int(boundary_plan["source_channel_count"])
        or len(source_audio) != int(boundary_plan["source_frame_count"])
    ):
        raise FinalRenderError("canonical source geometry changed after planning")

    parts: list[np.ndarray] = []
    for segment in boundary_plan["output_segments"]:
        source_start = int(segment["source_start_sample"])
        source_end = int(segment["source_end_sample"])
        if not 0 <= source_start < source_end <= len(source_audio):
            raise FinalRenderError("output trace references invalid source samples")
        chunk = source_audio[source_start:source_end]
        if segment["kind"] == "source":
            rendered = _apply_source_gain_envelopes(
                chunk,
                source_start=source_start,
                envelopes=segment.get("gain_envelopes", []),
            )
        else:
            rendered = np.array(chunk, dtype=np.float32, copy=True)
            fade_in = int(segment.get("fade_in_samples", 0))
            fade_out = int(segment.get("fade_out_samples", 0))
            if fade_in:
                rendered[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[
                    :, None
                ]
            if fade_out:
                rendered[-fade_out:] *= np.linspace(
                    1.0, 0.0, fade_out, dtype=np.float32
                )[:, None]
        parts.append(rendered)
    rendered_audio = np.concatenate(parts, axis=0)
    if len(rendered_audio) != int(boundary_plan["expected_output_frame_count"]):
        raise FinalRenderError("single-pass render length differs from boundary plan")

    # Every aligned retained span must map through an unmodified source segment.
    protected_retained = {
        (
            int(interval["word_id"]),
            int(interval["start_sample"]),
            int(interval["end_sample"]),
        )
        for boundary in boundary_plan["boundaries"]
        for interval in boundary.get("protected_speech_intervals", [])
        if "retained" in str(interval.get("role", ""))
    }
    for _, protected_start, protected_end in protected_retained:
        mapped = False
        for segment in boundary_plan["output_segments"]:
            if segment["kind"] != "source":
                continue
            source_start = int(segment["source_start_sample"])
            source_end = int(segment["source_end_sample"])
            if source_start <= protected_start and source_end >= protected_end:
                output_start = int(segment["output_start_sample"]) + (
                    protected_start - source_start
                )
                output_end = output_start + protected_end - protected_start
                if not np.array_equal(
                    rendered_audio[output_start:output_end],
                    source_audio[protected_start:protected_end],
                ):
                    raise FinalRenderError(
                        "retained aligned speech changed during single-pass render"
                    )
                mapped = True
                break
        if not mapped:
            raise FinalRenderError(
                "retained aligned speech is absent from output trace"
            )

    if output_path.exists():
        raise RuntimeError(f"final output already exists: {output_path}")
    sf.write(output_path, rendered_audio, sample_rate, subtype="FLOAT")
    written, written_rate = sf.read(output_path, dtype="float32", always_2d=True)
    if int(written_rate) != sample_rate or not np.array_equal(written, rendered_audio):
        raise FinalRenderError(
            "written final WAV differs from the single render buffer"
        )
    return {
        "frame_count": len(rendered_audio),
        "sample_rate": sample_rate,
        "channel_count": source_audio.shape[1],
        "duration_seconds": len(rendered_audio) / sample_rate,
        "output_sha256": sha256_file(output_path),
    }


def render_final_cut(
    *,
    audio_path: Path,
    plan_path: Path,
    output_dir: Path,
    pause_plan_path: Path | None = None,
    alignment_python: Path = DEFAULT_ALIGNMENT_PYTHON,
    env_file: Path = Path(".env"),
    planner_backend: str = "gemini",
    planner_model: str | None = None,
    planner_base_url: str | None = None,
    planner_api_key_env: str | None = None,
    planner_python: Path = DEFAULT_LOCAL_PYTHON,
    local_files_only: bool = False,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    pause_backend: PausePlannerBackend | None = None,
    repair_backend: PausePlannerBackend | None = None,
    alignment_payload: dict[str, Any] | None = None,
    alignment_payloads: Sequence[dict[str, Any]] | None = None,
    max_acoustic_retries: int = DEFAULT_MAX_ACOUSTIC_RETRIES,
    write_debug_artifacts: bool = False,
) -> dict[str, Any]:
    """Build one immutable boundary plan, then render canonical samples once."""

    audio_path = audio_path.resolve()
    plan_path = plan_path.resolve()
    output_dir = output_dir.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"output directory must be empty for final rendering: {output_dir}"
        )
    if max_acoustic_retries < 0:
        raise ValueError("max_acoustic_retries must be non-negative")
    original_plan_path = plan_path
    plan, grounding_path, selected_range_count = _validate_grounded_plan(
        audio_path=audio_path,
        plan_path=plan_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_pause_plan = _cache_pause_plan(
        plan_path=plan_path,
        output_dir=output_dir,
        supplied_pause_plan_path=pause_plan_path,
        backend=pause_backend,
        env_file=env_file,
        provider=planner_backend,
        model=planner_model,
        base_url=planner_base_url,
        api_key_env=planner_api_key_env,
        local_python=planner_python,
        local_files_only=local_files_only,
        max_output_tokens=max_output_tokens,
    )
    effective_plan = plan
    effective_plan_path = plan_path
    effective_grounding_path = grounding_path
    effective_selected_range_count = selected_range_count
    effective_pause_path = cached_pause_plan
    acoustic_repair_records: list[dict[str, Any]] = []
    rejected_boundary_history: list[dict[str, Any]] = []
    active_repair_backend = repair_backend
    owns_repair_backend = False
    boundary_plan: dict[str, Any]
    try:
        for acoustic_attempt in range(max_acoustic_retries + 1):
            pause_plan = read_json(effective_pause_path)
            if not isinstance(pause_plan, dict):
                raise FinalRenderError("pause plan root must be an object")
            evidence_dir = (
                output_dir
                if acoustic_attempt == 0
                else output_dir
                / "acoustic_retries"
                / f"retry_{acoustic_attempt:02d}"
                / "boundary_evidence"
            )
            evidence_dir.mkdir(parents=True, exist_ok=True)
            attempt_payload = (
                alignment_payloads[acoustic_attempt]
                if alignment_payloads is not None
                and acoustic_attempt < len(alignment_payloads)
                else alignment_payload
                if acoustic_attempt == 0
                else None
            )
            boundary_plan = build_final_boundary_plan(
                audio_path=audio_path,
                semantic_plan=effective_plan,
                semantic_plan_path=effective_plan_path,
                pause_plan=pause_plan,
                pause_plan_path=effective_pause_path,
                output_dir=evidence_dir,
                alignment_python=alignment_python,
                alignment_payload=attempt_payload,
            )
            if boundary_plan["status"] == "safe":
                break
            dense_count = int(boundary_plan["unsafe_dense_boundaries"])
            weak_word_count = int(boundary_plan.get("weak_retained_word_alignments", 0))
            alignment_failures = int(boundary_plan["alignment_failures"])
            if (
                dense_count + weak_word_count == 0
                or alignment_failures > 0
                or acoustic_attempt >= max_acoustic_retries
            ):
                break
            retry_index = acoustic_attempt + 1
            retry_dir = output_dir / "acoustic_retries" / f"retry_{retry_index:02d}"
            retry_dir.mkdir(parents=True, exist_ok=True)
            rejected_path = retry_dir / "rejected_boundary_plan.json"
            current_rejections = [
                json.loads(json.dumps(boundary))
                for boundary in boundary_plan["boundaries"]
                if boundary.get("safety_status")
                in {"unsafe_dense_boundary", "weak_retained_word_alignment"}
            ]
            repair_constraints = json.loads(json.dumps(boundary_plan))
            repair_constraints["boundaries"] = [
                *current_rejections,
                *rejected_boundary_history,
            ]
            repair_constraints["acoustic_rejection_history"] = [
                boundary.get("boundary_id") for boundary in rejected_boundary_history
            ]
            write_json(rejected_path, repair_constraints)
            for rejection in current_rejections:
                historical = json.loads(json.dumps(rejection))
                historical["boundary_id"] = (
                    f"retry_{retry_index:02d}:"
                    f"{historical.get('boundary_id', 'unsafe_boundary')}"
                )
                rejected_boundary_history.append(historical)
            if active_repair_backend is None:
                active_repair_backend = create_planner_backend(
                    provider=planner_backend,
                    model=planner_model,
                    env_file=env_file.resolve(),
                    max_output_tokens=max_output_tokens,
                    base_url=planner_base_url,
                    api_key_env=planner_api_key_env,
                    local_python=planner_python.absolute(),
                    local_files_only=local_files_only,
                )
                owns_repair_backend = True
            semantic_retry_dir = retry_dir / "semantic_plan"
            repaired = repair_plan_for_acoustic_safety(
                plan_path=effective_plan_path,
                boundary_plan_path=rejected_path,
                output_dir=semantic_retry_dir,
                backend=active_repair_backend,
                retry_index=retry_index,
            )
            repaired_path = semantic_retry_dir / "streaming_plan.json"
            (
                effective_plan,
                effective_grounding_path,
                effective_selected_range_count,
            ) = _validate_grounded_plan(
                audio_path=audio_path,
                plan_path=repaired_path,
            )
            if effective_plan != repaired:
                raise FinalRenderError(
                    "saved acoustic repair differs from its in-memory plan"
                )
            effective_plan_path = repaired_path
            effective_pause_path = _retarget_pause_plan(
                source_pause_plan_path=effective_pause_path,
                repaired_plan_path=effective_plan_path,
                destination=retry_dir / "pause_plan.json",
            )
            acoustic_repair_records.append(
                {
                    "retry_index": retry_index,
                    "rejected_boundary_plan": str(rejected_path.resolve()),
                    "rejected_boundary_plan_sha256": sha256_file(rejected_path),
                    "repaired_streaming_plan": str(effective_plan_path.resolve()),
                    "repaired_streaming_plan_sha256": sha256_file(effective_plan_path),
                }
            )
    finally:
        if owns_repair_backend and active_repair_backend is not None:
            active_repair_backend.close()
    boundary_plan_path = output_dir / "final_boundary_plan.json"
    write_json(boundary_plan_path, boundary_plan)
    boundary_plan_sha = sha256_file(boundary_plan_path)
    if boundary_plan["status"] != "safe":
        unsafe = int(boundary_plan["unsafe_dense_boundaries"])
        weak = int(boundary_plan.get("weak_retained_word_alignments", 0))
        failures = int(boundary_plan["alignment_failures"])
        if weak:
            reason = "weak_retained_word_alignment"
        elif unsafe:
            reason = "unsafe_dense_boundary"
        else:
            reason = "forced_alignment_failed"
        raise FinalRenderError(
            f"{reason}: boundary plan has {unsafe} dense boundaries and "
            f"{weak} weak retained-word alignments and {failures} alignment "
            "failures after "
            f"{len(acoustic_repair_records)} acoustic repair attempts; "
            "no audio was rendered"
        )

    final_path = output_dir / "final_cut.wav"
    render_result = render_boundary_plan(
        audio_path=audio_path,
        boundary_plan_path=boundary_plan_path,
        output_path=final_path,
    )
    if sha256_file(boundary_plan_path) != boundary_plan_sha:
        final_path.unlink(missing_ok=True)
        raise FinalRenderError("final boundary plan changed during rendering")

    manifest = {
        "schema_version": 2,
        "renderer": "authoritative_single_pass_final_render_v2",
        "status": "complete",
        "source_audio": str(audio_path),
        "source_audio_sha256": sha256_file(audio_path),
        "streaming_plan": str(original_plan_path),
        "streaming_plan_sha256": sha256_file(original_plan_path),
        "effective_streaming_plan": str(effective_plan_path),
        "effective_streaming_plan_sha256": sha256_file(effective_plan_path),
        "grounding_validation": str(effective_grounding_path),
        "grounding_validation_sha256": sha256_file(effective_grounding_path),
        "pause_plan": str(effective_pause_path),
        "pause_plan_sha256": sha256_file(effective_pause_path),
        "pause_planner_backend": pause_plan.get("backend"),
        "pause_planner_model": pause_plan.get("model"),
        "final_boundary_plan": str(boundary_plan_path.resolve()),
        "final_boundary_plan_sha256": boundary_plan_sha,
        "final_cut_wav": str(final_path.resolve()),
        "final_cut_wav_sha256": render_result["output_sha256"],
        "sample_rate": render_result["sample_rate"],
        "channel_count": render_result["channel_count"],
        "frame_count": render_result["frame_count"],
        "duration_seconds": render_result["duration_seconds"],
        "semantic_thoughts": len(effective_plan["committed"]),
        "selected_source_ranges": effective_selected_range_count,
        "acoustic_repair_attempts": len(acoustic_repair_records),
        "acoustic_repairs": acoustic_repair_records,
        "rendered_clips": len(boundary_plan["source_intervals"]),
        "debug_artifacts_requested": write_debug_artifacts,
        "debug_artifacts_written": False,
        "alignment_contexts": boundary_plan["alignment_context_count"],
        "alignment_resolved_boundaries": boundary_plan["alignment_resolved_boundaries"],
        "unsafe_dense_boundaries": boundary_plan["unsafe_dense_boundaries"],
        "weak_retained_word_alignments": boundary_plan.get(
            "weak_retained_word_alignments", 0
        ),
        "unresolved_boundaries": 0,
        "clip_joins": [
            join
            for join in boundary_plan["joins"]
            if join["join_kind"] == "source_discontinuity"
        ],
        "final_boundary": boundary_plan["final_boundary"],
    }
    write_json(output_dir / "final_render_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicecut --render-plan",
        description=(
            "Resolve one immutable alignment-protected boundary plan, then "
            "render it once from the canonical source WAV."
        ),
    )
    parser.add_argument("--render-plan", action="store_true")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pause-plan", type=Path)
    parser.add_argument(
        "--alignment-python",
        type=Path,
        default=DEFAULT_ALIGNMENT_PYTHON,
    )
    add_planner_backend_arguments(parser)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--debug-artifacts",
        action="store_true",
        help=(
            "Request optional diagnostics without changing the single-pass "
            "production render graph."
        ),
    )
    parser.add_argument(
        "--max-acoustic-retries",
        type=int,
        default=DEFAULT_MAX_ACOUSTIC_RETRIES,
        help=(
            "Maximum source-grounded semantic retries after a fail-closed "
            "dense boundary."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.render_plan:
        raise SystemExit("final plan rendering requires --render-plan")
    if args.max_acoustic_retries < 0:
        raise SystemExit("--max-acoustic-retries must be non-negative")
    manifest = render_final_cut(
        audio_path=args.audio,
        plan_path=args.plan,
        output_dir=args.output_dir,
        pause_plan_path=args.pause_plan,
        alignment_python=args.alignment_python,
        env_file=args.env_file,
        planner_backend=args.planner_backend,
        planner_model=args.planner_model,
        planner_base_url=args.planner_base_url,
        planner_api_key_env=args.planner_api_key_env,
        planner_python=args.planner_python,
        local_files_only=args.local_files_only,
        max_output_tokens=args.max_output_tokens,
        max_acoustic_retries=args.max_acoustic_retries,
        write_debug_artifacts=args.debug_artifacts,
    )
    print("\nFINAL CUT CREATED")
    print(f"semantic thoughts: {manifest['semantic_thoughts']}")
    print(f"rendered clips: {manifest['rendered_clips']}")
    print(f"alignment contexts: {manifest['alignment_contexts']}")
    print(f"acoustic repair attempts: {manifest['acoustic_repair_attempts']}")
    print(f"alignment-resolved boundaries: {manifest['alignment_resolved_boundaries']}")
    print(f"unsafe dense boundaries: {manifest['unsafe_dense_boundaries']}")
    print(f"unresolved boundaries: {manifest['unresolved_boundaries']}")
    print(f"duration: {manifest['duration_seconds']:.3f} s")
    print(f"output path: {manifest['final_cut_wav']}")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "final_cut": manifest["final_cut_wav"],
                "manifest": str(
                    (args.output_dir.resolve() / "final_render_manifest.json")
                ),
                "cached_pause_plan": manifest["pause_plan"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
