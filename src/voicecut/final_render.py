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
import warnings
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from .ambience import (
    DEFAULT_AMBIENCE_CROSSFADE_MS,
    DEFAULT_AMBIENCE_THRESHOLDS,
    build_clean_ambience_bank,
    evaluate_ambience_candidate,
    plan_ambience_assembly,
)
from .breath_cleanup import (
    BREATH_EVENT_GUARD_MS,
    BREATH_TRANSITION_MS,
    breath_room_tone_exclusions,
    plan_breath_replacements,
)
from .breath_detection import (
    DEFAULT_BREATH_MIN_DURATION_MS,
    DEFAULT_BREATH_THRESHOLD,
    DEFAULT_RESPIRO_CACHE_ROOT,
    RESPIRO_CHECKPOINT_SHA256,
    RESPIRO_FRAME_HOP_MS,
    RESPIRO_UPSTREAM_COMMIT,
    BreathDetectionError,
    analyze_breath_evidence,
)
from .common import read_json, sha256_file, write_json
from .mfa_alignment import (
    DEFAULT_MFA_CACHE_ROOT,
    DEFAULT_MFA_PREFIX,
    MFA_MODEL_ID,
    MFA_VERSION,
    MFAAlignmentError,
    align_mfa_contexts,
    seconds_to_sample,
    source_word_alignment,
)
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
    create_pause_plan,
    refine_eof_tail,
    validate_pause_response,
)
from .streaming_narration import (
    StreamingPlanError,
    build_conservative_delivery_plan,
    repair_plan_for_acoustic_safety,
)


DEFAULT_ALIGNMENT_PYTHON = Path(sys.executable)
CONTEXT_WORDS_PER_SIDE = 3
CROP_CONTEXT_MS = 400.0
PROTECTED_SPEECH_MARGIN_MS = 10.0
MINIMUM_VERIFIED_QUIET_MS = 20.0
QUIET_FADE_MS = 5.0
MFA_ZERO_CROSSING_SNAP_MS = 2.0
ALIGNMENT_LANGUAGE = "en"
DEFAULT_MAX_ACOUSTIC_RETRIES = 3
RETAINED_WORD_EDGE_CHARACTER_COUNT = 3
RETAINED_WORD_LOCAL_CONTEXT_CHARACTER_COUNT = 8
RETAINED_WORD_NEARBY_RETRY_DISTANCE = 4
RETAINED_WORD_MIN_RETRY_TOKEN_LENGTH = 5
MIN_RETAINED_WORD_SCORE = 0.45
MIN_RETAINED_EDGE_CHARACTER_SCORE = 0.48
MIN_RETAINED_EDGE_TO_CONTEXT_RATIO = 0.55
ALIGNMENT_GEOMETRY_EPSILON_SECONDS = 1e-6
COMPLETENESS_CROP_ROUNDING_TOLERANCE_SECONDS = 0.001
MFA_SAMPLE_ROUNDING_OVERLAP = 1
BREATH_CLEANUP_MODES = ("off", "replace")
BREATH_ALIGNMENT_MIN_GAP_MS = 20.0
PAUSE_POLICIES = ("semantic", "cuts")
CUTS_PAUSE_BACKEND = "deterministic_video_cuts"
AMBIENCE_BANK_SCHEMA_VERSION = 1
AMBIENCE_ARTIFACT_GUARD_MS = 0.0


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


def _write_cut_pause_plan(*, plan_path: Path, destination: Path) -> Path:
    """Write a no-model, zero-insertion transition ledger for video cuts."""

    plan = read_json(plan_path)
    committed = plan.get("committed") if isinstance(plan, dict) else None
    if not isinstance(committed, list) or not committed:
        raise FinalRenderError("cut pause policy requires committed thoughts")
    transitions = [
        {
            "after_thought_index": index,
            "before_thought_index": index + 1,
            "pause_type": "continuation",
        }
        for index in range(len(committed) - 1)
    ]
    write_json(
        destination,
        {
            "schema_version": 1,
            "planner": "semantic_pause_planner_v1",
            "backend": CUTS_PAUSE_BACKEND,
            "model": None,
            "pause_policy": "cuts",
            "streaming_plan": str(plan_path.resolve()),
            "streaming_plan_sha256": sha256_file(plan_path),
            "thought_count": len(committed),
            "transition_count": len(transitions),
            "transitions": transitions,
            "attempts": [
                {
                    "attempt": 0,
                    "raw_response": None,
                    "validation_error": None,
                    "model_call_skipped": "video output uses clear source cuts",
                }
            ],
        },
    )
    return destination


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
    pause_policy: str,
) -> Path:
    destination = output_dir / "pause_plan.json"
    if pause_policy == "cuts":
        if supplied_pause_plan_path is not None:
            raise FinalRenderError(
                "pause_plan_path cannot be supplied when pause_policy='cuts'"
            )
        return _write_cut_pause_plan(
            plan_path=plan_path,
            destination=destination,
        )
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


def _has_nearby_same_word_retry(
    *,
    span: dict[str, Any],
    context_spans: Sequence[dict[str, Any]],
    edge: str,
) -> bool:
    """Detect a nearby repeated occurrence that corroborates a weak edge.

    WhisperX character scores are not calibrated phoneme confidences: a
    single orthographic character can score poorly even for a complete word
    (notably silent letters).  A nearby repetition in the retry direction is
    independent source evidence that a weak edge belongs to an abandoned
    occurrence.  This keeps the veto conservative without globally lowering
    its confidence thresholds.
    """

    target_id = int(span["word_id"])
    target_text = "".join(
        character.casefold()
        for character in str(span.get("text", ""))
        if character.isalpha()
    )
    if not target_text:
        return False
    if len(target_text) < RETAINED_WORD_MIN_RETRY_TOKEN_LENGTH:
        # Short function words repeat naturally inside correct phrases. Their
        # repetition is not independent evidence of an abandoned take.
        return False
    for candidate in context_spans:
        candidate_id = int(candidate.get("word_id", -1))
        distance = candidate_id - target_id
        if edge == "terminal":
            in_retry_direction = 0 < distance <= RETAINED_WORD_NEARBY_RETRY_DISTANCE
        else:
            in_retry_direction = -RETAINED_WORD_NEARBY_RETRY_DISTANCE <= distance < 0
        if not in_retry_direction:
            continue
        candidate_text = "".join(
            character.casefold()
            for character in str(candidate.get("text", ""))
            if character.isalpha()
        )
        if candidate_text == target_text:
            return True
    return False


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
    median_score_ratio = (
        median_edge_score / local_context_score
        if median_edge_score is not None
        and local_context_score is not None
        and local_context_score > 0.0
        else None
    )
    nearby_same_word_retry = _has_nearby_same_word_retry(
        span=span,
        context_spans=context_spans,
        edge=edge,
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
    ):
        status = "incomplete_character_coverage"
    # WhisperX scores are not calibrated completeness probabilities. On long
    # recordings, a low score by itself otherwise turns inevitable alignment
    # outliers into destructive false vetoes of unique narration. Apply the
    # score gate only when a nearby repeated occurrence independently supports
    # the abandoned-retry hypothesis. Coverage and timestamp geometry remain
    # unconditional requirements; MFA remains the coordinate authority.
    elif nearby_same_word_retry and (
        word_score < MIN_RETAINED_WORD_SCORE
        or (
            median_edge_score < MIN_RETAINED_EDGE_CHARACTER_SCORE
            and (
                median_score_ratio is None
                or median_score_ratio < MIN_RETAINED_EDGE_TO_CONTEXT_RATIO
            )
        )
        or (
            minimum_edge_score < MIN_RETAINED_EDGE_CHARACTER_SCORE
            and (
                score_ratio is None or score_ratio < MIN_RETAINED_EDGE_TO_CONTEXT_RATIO
            )
        )
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
        "median_edge_to_context_score_ratio": median_score_ratio,
        "nearby_same_word_retry": nearby_same_word_retry,
        "character_records": [dict(character) for character in alphabetic],
        "thresholds": {
            "edge_character_count": RETAINED_WORD_EDGE_CHARACTER_COUNT,
            "local_context_character_count": RETAINED_WORD_LOCAL_CONTEXT_CHARACTER_COUNT,
            "nearby_retry_word_distance": RETAINED_WORD_NEARBY_RETRY_DISTANCE,
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
                # Neighboring context words can be unaligned without making
                # the retained boundary word unusable.  The caller evaluates
                # the required retained occurrence explicitly and fails
                # closed when that specific word is absent.
                continue
            relative = (aligned_word_start, aligned_word_end)
        absolute_start = crop_start + relative[0]
        absolute_end = crop_start + relative[1]
        if not crop_start <= absolute_start < absolute_end <= crop_end:
            overshoot = max(
                crop_start - absolute_start,
                absolute_end - crop_end,
                0.0,
            )
            if overshoot <= COMPLETENESS_CROP_ROUNDING_TOLERANCE_SECONDS:
                absolute_start = max(crop_start, absolute_start)
                absolute_end = min(crop_end, absolute_end)
            else:
                # Neighboring context words can fall outside the decoded crop
                # without invalidating its retained boundary word. A required
                # occurrence still fails closed when its support is absent.
                continue
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


def _next_lexical_word_id(
    words: Sequence[PlanWord],
    *,
    start_word_id: int,
    end_word_id: int,
) -> int:
    for word_id in range(start_word_id, end_word_id):
        if any(character.isalnum() for character in words[word_id].text):
            return word_id
    raise FinalRenderError(
        f"source interval [{start_word_id}, {end_word_id}) has no alignable word"
    )


def _previous_lexical_word_id(
    words: Sequence[PlanWord],
    *,
    start_word_id: int,
    end_word_id: int,
) -> int:
    for word_id in range(end_word_id - 1, start_word_id - 1, -1):
        if any(character.isalnum() for character in words[word_id].text):
            return word_id
    raise FinalRenderError(
        f"source interval [{start_word_id}, {end_word_id}) has no alignable word"
    )


def _alignment_event_specs(
    *,
    words: Sequence[PlanWord],
    ranges: Sequence[MergedRange],
    thought_bounds: Sequence[tuple[int, int]],
    include_breath_gaps: bool = False,
) -> list[dict[str, Any]]:
    """Describe every MFA context needed before any coordinate is selected.

    The two physical endpoints of a source gap deliberately use separate local
    contexts.  This keeps a long omitted region out of a single forced-
    alignment crop while all contexts still run in one MFA batch.
    """

    specs: list[dict[str, Any]] = []
    if ranges[0].start_word_id > 0:
        first_retained = _next_lexical_word_id(
            words,
            start_word_id=ranges[0].start_word_id,
            end_word_id=ranges[0].end_word_id,
        )
        previous_omitted = _previous_lexical_word_id(
            words,
            start_word_id=0,
            end_word_id=ranges[0].start_word_id,
        )
        specs.append(
            {
                "event_key": "leading_source_cut",
                "event_kind": "leading_source_cut",
                "range_index": 0,
                "role_word_ids": {
                    "previous_omitted": previous_omitted,
                    "first_retained_right": first_retained,
                },
            }
        )
    for gap_index, (left, right) in enumerate(zip(ranges, ranges[1:])):
        if right.start_word_id <= left.end_word_id:
            raise FinalRenderError("merged source ranges are not disjoint")
        last_retained_left = _previous_lexical_word_id(
            words,
            start_word_id=left.start_word_id,
            end_word_id=left.end_word_id,
        )
        first_omitted = _next_lexical_word_id(
            words,
            start_word_id=left.end_word_id,
            end_word_id=right.start_word_id,
        )
        last_omitted = _previous_lexical_word_id(
            words,
            start_word_id=left.end_word_id,
            end_word_id=right.start_word_id,
        )
        first_retained_right = _next_lexical_word_id(
            words,
            start_word_id=right.start_word_id,
            end_word_id=right.end_word_id,
        )
        specs.append(
            {
                "event_key": f"source_gap_{gap_index:04d}_left",
                "event_kind": "source_gap_left",
                "source_gap_index": gap_index,
                "left_range_index": gap_index,
                "role_word_ids": {
                    "last_retained_left": last_retained_left,
                    "first_omitted": first_omitted,
                },
            }
        )
        specs.append(
            {
                "event_key": f"source_gap_{gap_index:04d}_right",
                "event_kind": "source_gap_right",
                "source_gap_index": gap_index,
                "right_range_index": gap_index + 1,
                "role_word_ids": {
                    "last_omitted": last_omitted,
                    "first_retained_right": first_retained_right,
                },
            }
        )
    if ranges[-1].end_word_id < len(words):
        last_retained = _previous_lexical_word_id(
            words,
            start_word_id=ranges[-1].start_word_id,
            end_word_id=ranges[-1].end_word_id,
        )
        first_omitted = _next_lexical_word_id(
            words,
            start_word_id=ranges[-1].end_word_id,
            end_word_id=len(words),
        )
        specs.append(
            {
                "event_key": "trailing_source_cut",
                "event_kind": "trailing_source_cut",
                "range_index": len(ranges) - 1,
                "role_word_ids": {
                    "last_retained_left": last_retained,
                    "first_omitted": first_omitted,
                },
            }
        )
    if ranges[-1].end_word_id == len(words):
        final_retained = _previous_lexical_word_id(
            words,
            start_word_id=ranges[-1].start_word_id,
            end_word_id=ranges[-1].end_word_id,
        )
        specs.append(
            {
                "event_key": "eof_tail",
                "event_kind": "eof_tail",
                "range_index": len(ranges) - 1,
                "role_word_ids": {
                    "final_retained": final_retained,
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
    if include_breath_gaps:
        existing_pairs = {
            (
                int(spec["role_word_ids"]["previous_retained"]),
                int(spec["role_word_ids"]["next_retained"]),
            )
            for spec in specs
            if spec["event_kind"] == "internal_thought_gap"
        }
        gap_index = 0
        for range_index, source_range in enumerate(ranges):
            for left_word_id in range(
                source_range.start_word_id,
                source_range.end_word_id - 1,
            ):
                right_word_id = left_word_id + 1
                if (left_word_id, right_word_id) in existing_pairs:
                    continue
                approximate_gap_ms = max(
                    0.0,
                    (words[right_word_id].start - words[left_word_id].end) * 1000.0,
                )
                if approximate_gap_ms < BREATH_ALIGNMENT_MIN_GAP_MS:
                    continue
                specs.append(
                    {
                        "event_key": f"breath_retained_gap_{gap_index:04d}",
                        "event_kind": "breath_retained_gap",
                        "range_index": range_index,
                        "approximate_gap_ms": approximate_gap_ms,
                        "completeness_required": False,
                        "role_word_ids": {
                            "previous_retained": left_word_id,
                            "next_retained": right_word_id,
                        },
                    }
                )
                gap_index += 1
    return specs


def _prepare_alignment_jobs(
    *,
    specs: Sequence[dict[str, Any]],
    words: Sequence[PlanWord],
    ranges: Sequence[MergedRange],
    source_audio: np.ndarray,
    sample_rate: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    context_samples = round(CROP_CONTEXT_MS * sample_rate / 1000.0)
    total_samples = len(source_audio)
    contexts_dir = output_dir / "completeness_contexts"
    if specs:
        contexts_dir.mkdir(parents=True, exist_ok=True)
    selected_word_ids = {
        word_id
        for source_range in ranges
        for word_id in range(
            source_range.start_word_id,
            source_range.end_word_id,
        )
    }
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
        # Expanding the transcript to cover words that overlap the requested
        # handles used to collapse the crop back onto the first/last word. MFA
        # then received no acoustic lead-in or tail and could omit otherwise
        # valid utterances in large batches. Restore only the portion of each
        # handle that does not cross a neighboring, untranscribed word.
        first_word_start = math.floor(words[context_start_id].start * sample_rate)
        previous_word_end = (
            math.ceil(words[context_start_id - 1].end * sample_rate)
            if context_start_id > 0
            else 0
        )
        crop_start = min(
            crop_start,
            max(previous_word_end, first_word_start - context_samples),
        )
        last_word_end = math.ceil(words[context_end_id - 1].end * sample_rate)
        next_word_start = (
            math.floor(words[context_end_id].start * sample_rate)
            if context_end_id < len(words)
            else total_samples
        )
        crop_end = max(
            crop_end,
            min(next_word_start, last_word_end + context_samples),
        )
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
                "selected": word_id in selected_word_ids,
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
                "boundary_ids": [str(spec["event_key"])],
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


def _run_completeness_worker(
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
            f"WhisperX completeness worker produced no result; see {log_path}"
        )
    result = read_json(result_path)
    if not isinstance(result, dict):
        raise FinalRenderError("WhisperX completeness result must be an object")
    return result


def _mfa_protected_intervals(
    *,
    spec: dict[str, Any],
    context: dict[str, Any],
    sample_rate: int,
    total_samples: int,
) -> list[dict[str, Any]]:
    """Protect required words using MFA phones, never Whisper timestamps."""

    margin = round(PROTECTED_SPEECH_MARGIN_MS * sample_rate / 1000.0)
    all_context_phones = [
        phone for phone in context.get("phones", []) if isinstance(phone, dict)
    ]
    previous_end = -1
    previous_end_seconds = -math.inf
    for phone in all_context_phones:
        phone_start = int(phone["start_sample"])
        phone_end = int(phone["end_sample"])
        phone_start_seconds = float(phone["start_seconds"])
        phone_end_seconds = float(phone["end_seconds"])
        if not 0 <= phone_start < phone_end <= total_samples:
            raise MFAAlignmentError(
                "mfa_word_mapping_failed",
                f"phone {phone.get('phone')!r} has invalid source samples",
            )
        if (
            phone_start_seconds
            < previous_end_seconds - ALIGNMENT_GEOMETRY_EPSILON_SECONDS
            or phone_start < previous_end - MFA_SAMPLE_ROUNDING_OVERLAP
        ):
            raise MFAAlignmentError(
                "mfa_word_mapping_failed",
                "MFA context contains overlapping phone intervals",
            )
        previous_end = phone_end
        previous_end_seconds = phone_end_seconds
    context_phones = [
        phone for phone in all_context_phones if not bool(phone.get("is_silence"))
    ]
    protected: list[dict[str, Any]] = []
    for role, raw_word_id in spec["role_word_ids"].items():
        word_id = int(raw_word_id)
        aligned = source_word_alignment(context, word_id)
        word_start_sample = int(aligned["start_sample"])
        word_end_sample = int(aligned["end_sample"])
        nested_previous_end = word_start_sample
        nested_previous_end_seconds = float(aligned["start_seconds"])
        for phone in aligned["phones"]:
            phone_start = int(phone["start_sample"])
            phone_end = int(phone["end_sample"])
            phone_start_seconds = float(phone["start_seconds"])
            phone_end_seconds = float(phone["end_seconds"])
            if (
                phone_start < word_start_sample
                or phone_end > word_end_sample
                or phone_start < nested_previous_end - MFA_SAMPLE_ROUNDING_OVERLAP
                or phone_start_seconds
                < nested_previous_end_seconds - ALIGNMENT_GEOMETRY_EPSILON_SECONDS
                or phone_end <= phone_start
            ):
                raise MFAAlignmentError(
                    "mfa_word_mapping_failed",
                    f"word {word_id} has invalid nested phone geometry",
                )
            nested_previous_end = phone_end
            nested_previous_end_seconds = phone_end_seconds
        first_phone = dict(aligned["first_non_silence_phone"])
        last_phone = dict(aligned["last_non_silence_phone"])
        start = int(first_phone["start_sample"])
        end = int(last_phone["end_sample"])
        boundary_start = seconds_to_sample(
            float(first_phone["start_seconds"]),
            sample_rate,
            boundary="ceil",
        )
        boundary_end = seconds_to_sample(
            float(last_phone["end_seconds"]),
            sample_rate,
            boundary="ceil",
        )
        if not (0 <= start <= boundary_start < boundary_end <= end <= total_samples):
            raise MFAAlignmentError(
                "mfa_word_mapping_failed",
                f"word {word_id} has invalid absolute phone samples",
            )
        # MFA intervals are continuous half-open times.  Mapping both edges
        # with ceil yields a discrete half-open sample span without assigning
        # the shared boundary sample to both adjacent phones.  The original
        # floor/ceil coordinates remain preserved in the phone evidence.
        start = boundary_start
        end = boundary_end
        margin_start = max(0, start - margin)
        margin_end = min(total_samples, end + margin)
        for phone in context_phones:
            phone_start = seconds_to_sample(
                float(phone["start_seconds"]),
                sample_rate,
                boundary="ceil",
            )
            phone_end = seconds_to_sample(
                float(phone["end_seconds"]),
                sample_rate,
                boundary="ceil",
            )
            if phone_end <= start and phone_end > margin_start:
                margin_start = phone_end
            if phone_start >= end and phone_start < margin_end:
                margin_end = phone_start
        protected.append(
            {
                "role": role,
                "word_id": word_id,
                "text": aligned["source_text"],
                "start_sample": start,
                "end_sample": end,
                "start_seconds": start / sample_rate,
                "end_seconds": end / sample_rate,
                "boundary_start_sample": boundary_start,
                "boundary_end_sample": boundary_end,
                "boundary_start_seconds": boundary_start / sample_rate,
                "boundary_end_seconds": boundary_end / sample_rate,
                "margin_start_sample": margin_start,
                "margin_end_sample": margin_end,
                "protected_margin_ms": PROTECTED_SPEECH_MARGIN_MS,
                "alignment_backend": "mfa",
                "mfa_word_interval": {
                    "start_seconds": aligned["start_seconds"],
                    "end_seconds": aligned["end_seconds"],
                    "start_sample": aligned["start_sample"],
                    "end_sample": aligned["end_sample"],
                    "mfa_tokens": list(aligned["mfa_tokens"]),
                },
                "mfa_phone_intervals": [dict(phone) for phone in aligned["phones"]],
                "first_non_silence_phone": first_phone,
                "last_non_silence_phone": last_phone,
            }
        )
    return protected


def _protected_by_role(
    protected: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {str(item["role"]): item for item in protected}


def _mfa_non_speech_interval(
    *,
    context: dict[str, Any],
    start_sample: int,
    end_sample: int,
    sample_rate: int,
) -> dict[str, Any] | None:
    if end_sample <= start_sample:
        return None
    for phone in context.get("phones", []):
        if not isinstance(phone, dict) or bool(phone.get("is_silence")):
            continue
        phone_start = seconds_to_sample(
            float(phone["start_seconds"]),
            sample_rate,
            boundary="ceil",
        )
        phone_end = seconds_to_sample(
            float(phone["end_seconds"]),
            sample_rate,
            boundary="ceil",
        )
        if _intervals_overlap(
            start_sample,
            end_sample,
            phone_start,
            phone_end,
        ):
            raise MFAAlignmentError(
                "mfa_word_mapping_failed",
                "candidate non-speech interval overlaps an MFA speech phone",
            )
    silence_phones = [
        dict(phone)
        for phone in context.get("phones", [])
        if isinstance(phone, dict)
        and bool(phone.get("is_silence"))
        and _intervals_overlap(
            start_sample,
            end_sample,
            int(phone["start_sample"]),
            int(phone["end_sample"]),
        )
    ]
    return {
        "start_sample": start_sample,
        "end_sample": end_sample,
        "start_seconds": start_sample / sample_rate,
        "end_seconds": end_sample / sample_rate,
        "duration_ms": (end_sample - start_sample) * 1000.0 / sample_rate,
        "silence_phone_intervals": silence_phones,
        "verification": "mfa_phone_free_interval",
    }


def _snap_zero_crossing_in_mfa_gap(
    mono: np.ndarray,
    *,
    candidate: int,
    interval_start: int,
    interval_end: int,
    sample_rate: int,
) -> int:
    """Snap at most 2 ms, strictly inside an MFA-confirmed non-speech gap."""

    candidate = max(interval_start, min(interval_end, candidate))
    radius = round(MFA_ZERO_CROSSING_SNAP_MS * sample_rate / 1000.0)
    search_start = max(interval_start, candidate - radius, 1)
    search_end = min(interval_end, candidate + radius, len(mono) - 1)
    crossings = [
        index
        for index in range(search_start, search_end + 1)
        if (mono[index - 1] <= 0.0 < mono[index])
        or (mono[index - 1] >= 0.0 > mono[index])
    ]
    return (
        min(crossings, key=lambda index: (abs(index - candidate), index))
        if crossings
        else candidate
    )


def _resolve_mfa_cut(
    *,
    boundary_id: str,
    boundary_kind: str,
    spec: dict[str, Any],
    mfa_context: dict[str, Any] | None,
    mfa_error: str | None,
    retained_support: dict[str, Any] | None,
    completeness_error: str | None,
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
        "approximate": True,
        "retained_start_seconds": words[retained_id].start,
        "retained_end_seconds": words[retained_id].end,
        "omitted_start_seconds": words[omitted_id].start,
        "omitted_end_seconds": words[omitted_id].end,
    }
    common = {
        "boundary_id": boundary_id,
        "boundary_kind": boundary_kind,
        "alignment_context_id": spec["event_key"],
        "mfa_context_id": spec["event_key"],
        "source_word_ids": role_ids,
        "whisper_anchors": whisper,
        "whisper_timestamps": whisper,
        "protected_speech_intervals": [],
        "mfa_word_interval": None,
        "mfa_phone_intervals": [],
        "final_retained_phone": None,
        "aligned_timestamps": None,
        "verified_silence_interval": None,
        "verified_quiet_interval": None,
        "selected_source_sample": None,
        "selected_source_seconds": None,
        "fade_intervals": [],
        "retained_word_support": retained_support,
        "forbidden_word_ids": [],
        "forbidden_source_edges": [],
        "failure_reason": None,
    }
    if completeness_error is not None or retained_support is None:
        return {
            **common,
            "boundary_method": "retained_word_completeness_failed",
            "safety_status": "completeness_alignment_failed",
            "failure_reason": "completeness_alignment_failed",
            "error": completeness_error or "retained-word support is missing",
        }
    if retained_support["status"] != "supported_complete_word":
        return {
            **common,
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
                f"retained word {retained_id} {words[retained_id].text!r} "
                f"failed acoustic support: {retained_support['status']}"
            ),
        }
    if mfa_context is None:
        if mfa_error == "mfa_not_run_due_to_weak_retained_word":
            status = "mfa_not_run_due_to_weak_retained_word"
        elif mfa_error and mfa_error.startswith("mfa_word_mapping_failed"):
            status = "mfa_word_mapping_failed"
        else:
            status = "mfa_alignment_failed"
        forbidden_edge = {
            "boundary_kind": boundary_kind,
            "retained_word_id": retained_id,
            "omitted_word_id": omitted_id,
        }
        return {
            **common,
            "forbidden_source_edges": [forbidden_edge],
            "boundary_method": status,
            "safety_status": status,
            "failure_reason": status,
            "error": mfa_error or "MFA evidence is missing",
        }
    try:
        protected = _mfa_protected_intervals(
            spec=spec,
            context=mfa_context,
            sample_rate=sample_rate,
            total_samples=total_samples,
        )
        roles = _protected_by_role(protected)
        retained = roles[retained_role]
        omitted = roles[omitted_role]
        if direction == "trailing":
            speech_boundary = int(retained["boundary_end_sample"])
            gap_start = int(retained["end_sample"])
            gap_end = int(omitted["start_sample"])
            candidate = min(
                gap_end,
                int(retained["margin_end_sample"]),
            )
            curve = "fade_out"
        elif direction == "leading":
            speech_boundary = int(retained["boundary_start_sample"])
            gap_start = int(omitted["end_sample"])
            gap_end = int(retained["start_sample"])
            candidate = max(
                gap_start,
                int(retained["margin_start_sample"]),
            )
            curve = "fade_in"
        else:
            raise ValueError("boundary direction must be leading or trailing")
        silence = _mfa_non_speech_interval(
            context=mfa_context,
            start_sample=gap_start,
            end_sample=gap_end,
            sample_rate=sample_rate,
        )
        if silence is None:
            selected = speech_boundary
            method = "mfa_dense_phone_boundary"
            fades: list[dict[str, Any]] = []
        else:
            selected = _snap_zero_crossing_in_mfa_gap(
                mono,
                candidate=candidate,
                interval_start=gap_start,
                interval_end=gap_end,
                sample_rate=sample_rate,
            )
            method = "mfa_verified_silence"
            fade_samples = round(QUIET_FADE_MS * sample_rate / 1000.0)
            if direction == "trailing":
                fade_start = max(gap_start, selected - fade_samples)
                fade_end = selected
            else:
                fade_start = selected
                fade_end = min(gap_end, selected + fade_samples)
            fades = (
                [
                    {
                        "source_start_sample": fade_start,
                        "source_end_sample": fade_end,
                        "curve": curve,
                        "verified_quiet": True,
                        "verification": "mfa_phone_free_interval",
                    }
                ]
                if fade_end > fade_start
                else []
            )
        retained_word = retained["mfa_word_interval"]
        retained_phones = retained["mfa_phone_intervals"]
        retained_final_phone = (
            retained["last_non_silence_phone"]
            if direction == "trailing"
            else retained["first_non_silence_phone"]
        )
        return {
            **common,
            "protected_speech_intervals": protected,
            "mfa_word_interval": retained_word,
            "mfa_phone_intervals": retained_phones,
            "final_retained_phone": retained_final_phone,
            "aligned_timestamps": {
                "backend": "mfa",
                "retained_start_seconds": retained["start_seconds"],
                "retained_end_seconds": retained["end_seconds"],
                "omitted_start_seconds": omitted["start_seconds"],
                "omitted_end_seconds": omitted["end_seconds"],
            },
            "verified_silence_interval": silence,
            "verified_quiet_interval": silence,
            "selected_source_sample": selected,
            "selected_source_seconds": selected / sample_rate,
            "fade_intervals": fades,
            "boundary_method": method,
            "safety_status": "safe",
            "error": None,
        }
    except (MFAAlignmentError, KeyError, TypeError, ValueError) as error:
        status = (
            error.code
            if isinstance(error, MFAAlignmentError)
            else "mfa_word_mapping_failed"
        )
        return {
            **common,
            "boundary_method": status,
            "safety_status": status,
            "failure_reason": status,
            "error": f"{type(error).__name__}: {error}",
        }


def _resolve_mfa_eof_boundary(
    *,
    spec: dict[str, Any],
    mfa_context: dict[str, Any] | None,
    mfa_error: str | None,
    retained_support: dict[str, Any] | None,
    completeness_error: str | None,
    words: Sequence[PlanWord],
    mono: np.ndarray,
    sample_rate: int,
    total_samples: int,
) -> dict[str, Any]:
    word_id = int(spec["role_word_ids"]["final_retained"])
    whisper = {
        "approximate": True,
        "retained_start_seconds": words[word_id].start,
        "retained_end_seconds": words[word_id].end,
    }
    common = {
        "boundary_id": "end_of_file",
        "boundary_kind": "end_of_file",
        "alignment_context_id": spec["event_key"],
        "mfa_context_id": spec["event_key"],
        "source_word_ids": {"final_retained": word_id},
        "whisper_anchors": whisper,
        "whisper_timestamps": whisper,
        "protected_speech_intervals": [],
        "mfa_word_interval": None,
        "mfa_phone_intervals": [],
        "final_retained_phone": None,
        "aligned_timestamps": None,
        "verified_silence_interval": None,
        "verified_quiet_interval": None,
        "selected_source_sample": None,
        "selected_source_seconds": None,
        "fade_intervals": [],
        "retained_word_support": retained_support,
        "forbidden_word_ids": [],
        "forbidden_source_edges": [],
        "failure_reason": None,
    }
    if completeness_error is not None or retained_support is None:
        return {
            **common,
            "boundary_method": "retained_word_completeness_failed",
            "safety_status": "completeness_alignment_failed",
            "failure_reason": "completeness_alignment_failed",
            "error": completeness_error or "retained-word support is missing",
        }
    if retained_support["status"] != "supported_complete_word":
        return {
            **common,
            "forbidden_word_ids": [word_id],
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
                f"retained EOF word {word_id} {words[word_id].text!r} failed "
                f"acoustic support: {retained_support['status']}"
            ),
        }
    if mfa_context is None:
        if mfa_error == "mfa_not_run_due_to_weak_retained_word":
            status = "mfa_not_run_due_to_weak_retained_word"
        elif mfa_error and mfa_error.startswith("mfa_word_mapping_failed"):
            status = "mfa_word_mapping_failed"
        else:
            status = "mfa_alignment_failed"
        return {
            **common,
            "boundary_method": status,
            "safety_status": status,
            "failure_reason": status,
            "error": mfa_error or "MFA EOF evidence is missing",
        }
    try:
        protected = _mfa_protected_intervals(
            spec=spec,
            context=mfa_context,
            sample_rate=sample_rate,
            total_samples=total_samples,
        )
        retained = _protected_by_role(protected)["final_retained"]
        final_phone = dict(retained["last_non_silence_phone"])
        phone_end = int(final_phone["end_sample"])
        eof = refine_eof_tail(
            mono,
            sample_rate=sample_rate,
            raw_end_seconds=float(final_phone["end_seconds"]),
            previous_end_sample=phone_end,
            fade_ms=QUIET_FADE_MS,
        )
        if eof.new_end_sample < phone_end:
            raise MFAAlignmentError(
                "mfa_word_mapping_failed",
                "EOF tail ends before the retained final MFA phone",
            )
        eof_fades: list[dict[str, Any]] = []
        if eof.fade_out_samples:
            fade_start = eof.new_end_sample - eof.fade_out_samples
            if (
                fade_start < phone_end
                or eof.stable_silence_start_sample is None
                or fade_start < eof.stable_silence_start_sample
            ):
                raise MFAAlignmentError(
                    "mfa_word_mapping_failed",
                    "EOF fade overlaps the retained final MFA phone",
                )
            eof_fades.append(
                {
                    "source_start_sample": fade_start,
                    "source_end_sample": eof.new_end_sample,
                    "curve": "fade_out",
                    "verified_quiet": True,
                    "verification": "eof_stable_silence_after_mfa_phone",
                }
            )
        silence = (
            {
                "start_sample": eof.stable_silence_start_sample,
                "end_sample": eof.new_end_sample,
                "start_seconds": eof.stable_silence_start_sample / sample_rate,
                "end_seconds": eof.new_end_sample / sample_rate,
                "local_noise_floor_db": eof.local_noise_floor_db,
                "silence_threshold_db": eof.silence_threshold_db,
                "verification": "safe_eof_tail_after_mfa_phone",
            }
            if eof.stable_silence_start_sample is not None
            else None
        )
        return {
            **common,
            "protected_speech_intervals": protected,
            "mfa_word_interval": retained["mfa_word_interval"],
            "mfa_phone_intervals": retained["mfa_phone_intervals"],
            "final_retained_phone": final_phone,
            "aligned_timestamps": {
                "backend": "mfa",
                "retained_start_seconds": retained["start_seconds"],
                "retained_end_seconds": retained["end_seconds"],
            },
            "verified_silence_interval": silence,
            "verified_quiet_interval": silence,
            "selected_source_sample": eof.new_end_sample,
            "selected_source_seconds": eof.new_end_sample / sample_rate,
            "fade_intervals": eof_fades,
            "boundary_method": "eof_safe_tail",
            "safety_status": "safe",
            "last_word": words[word_id].text,
            "final_word_id": word_id,
            "final_word_text": words[word_id].text,
            "final_phone": final_phone["phone"],
            "mfa_phone_end": final_phone["end_seconds"],
            "retained_tail_end": eof.new_end_sample / sample_rate,
            "fade_interval": eof_fades[0] if eof_fades else None,
            "error": None,
        }
    except (MFAAlignmentError, KeyError, TypeError, ValueError) as error:
        status = (
            error.code
            if isinstance(error, MFAAlignmentError)
            else "mfa_word_mapping_failed"
        )
        return {
            **common,
            "boundary_method": status,
            "safety_status": status,
            "failure_reason": status,
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
        "mfa_context_id": None,
        "source_word_ids": {},
        "whisper_anchors": None,
        "whisper_timestamps": None,
        "mfa_word_interval": None,
        "mfa_phone_intervals": [],
        "final_retained_phone": None,
        "aligned_timestamps": None,
        "protected_speech_intervals": [],
        "verified_silence_interval": None,
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


def _collect_completeness_evidence(
    *,
    jobs: Sequence[dict[str, Any]],
    worker_result: dict[str, Any],
    sample_rate: int,
    total_samples: int,
) -> list[dict[str, Any]]:
    raw_worker_jobs = worker_result.get("jobs")
    if not isinstance(raw_worker_jobs, list):
        raise FinalRenderError("completeness worker returned no jobs list")
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
                "purpose": "retained_word_completeness_veto",
                "coordinate_authority": False,
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


def _completeness_support_by_context(
    contexts: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    support: dict[str, dict[str, Any]] = {}
    for context in contexts:
        kind = str(context["context_kind"])
        if kind in {"leading_source_cut", "source_gap_right"}:
            retained_role = "first_retained_right"
            edge = "initial"
        elif kind in {"source_gap_left", "trailing_source_cut"}:
            retained_role = "last_retained_left"
            edge = "terminal"
        elif kind == "eof_tail":
            retained_role = "final_retained"
            edge = "terminal"
        else:
            continue
        spans = context.get("_spans")
        spec = context["_spec"]
        retained_id = int(spec["role_word_ids"][retained_role])
        if not isinstance(spans, dict):
            support[str(context["context_id"])] = {
                "support": None,
                "error": context.get("error") or "completeness evidence is missing",
            }
            continue
        retained_span = spans.get(retained_id)
        if not isinstance(retained_span, dict):
            support[str(context["context_id"])] = {
                "support": None,
                "error": f"retained word {retained_id} was not completeness-aligned",
            }
            continue
        support[str(context["context_id"])] = {
            "support": evaluate_retained_word_support(
                retained_span,
                list(spans.values()),
                edge=edge,
            ),
            "error": None,
        }
    return support


def _mfa_context_requests(jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "context_id": str(job["event_key"]),
            "crop_source_start_seconds": float(job["crop_start_seconds"]),
            "crop_source_end_seconds": float(job["crop_end_seconds"]),
            "words": [
                {
                    "word_id": int(word["id"]),
                    "text": str(word["text"]),
                    "start_seconds": float(word["start"]),
                    "end_seconds": float(word["end"]),
                    "selected": bool(word["selected"]),
                }
                for word in job["local_words"]
            ],
            "boundary_ids": list(job["boundary_ids"]),
        }
        for job in jobs
    ]


def _validated_mfa_contexts(
    *,
    payload: dict[str, Any],
    jobs: Sequence[dict[str, Any]],
    source_audio_sha256: str,
    sample_rate: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if (
        payload.get("backend") != "mfa"
        or payload.get("mfa_version") != MFA_VERSION
        or payload.get("model_id") != MFA_MODEL_ID
        or payload.get("fine_tune") is not True
    ):
        raise FinalRenderError("MFA payload has incompatible runtime provenance")
    if payload.get("source_audio_sha256") != source_audio_sha256:
        raise FinalRenderError("MFA payload belongs to a different canonical source")
    if payload.get("sample_rate") != sample_rate:
        raise FinalRenderError("MFA payload has the wrong canonical sample rate")
    raw_contexts = payload.get("contexts")
    if not isinstance(raw_contexts, list):
        raise FinalRenderError("MFA payload has no contexts")
    context_by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_contexts:
        if not isinstance(raw, dict):
            raise FinalRenderError("MFA payload contains a malformed context")
        context_id = str(raw.get("context_id", ""))
        if not context_id or context_id in context_by_id:
            raise FinalRenderError("MFA context IDs are empty or duplicated")
        context_by_id[context_id] = raw
    raw_errors = payload.get("context_errors", [])
    if not isinstance(raw_errors, list):
        raise FinalRenderError("MFA payload context errors are malformed")
    errors_by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_errors:
        if not isinstance(raw, dict):
            raise FinalRenderError("MFA payload contains a malformed context error")
        context_id = str(raw.get("context_id", ""))
        if not context_id or context_id in errors_by_id or context_id in context_by_id:
            raise FinalRenderError(
                "MFA context error IDs are empty, duplicated, or already aligned"
            )
        code = str(raw.get("code", ""))
        error = str(raw.get("error", ""))
        if code != "mfa_word_mapping_failed" or not error:
            raise FinalRenderError("MFA context error has invalid failure evidence")
        errors_by_id[context_id] = raw
    expected = {str(job["event_key"]) for job in jobs}
    if set(context_by_id) | set(errors_by_id) != expected:
        raise FinalRenderError(
            "MFA payload contexts do not exactly match this boundary attempt"
        )
    for job in jobs:
        context_id = str(job["event_key"])
        if context_id in errors_by_id:
            if errors_by_id[context_id].get("boundary_ids") != job["boundary_ids"]:
                raise FinalRenderError(
                    f"MFA context error {context_id} has stale boundary ownership"
                )
            continue
        context = context_by_id[context_id]
        if (
            context.get("crop_source_start_sample") != job["crop_start_sample"]
            or context.get("crop_source_end_sample") != job["crop_end_sample"]
        ):
            raise FinalRenderError(
                f"MFA context {context_id} has stale crop sample coordinates"
            )
        expected_ids = [int(word["id"]) for word in job["local_words"]]
        if context.get("ordered_source_word_ids") != expected_ids:
            raise FinalRenderError(
                f"MFA context {context_id} has stale source-word ordering"
            )
        if context.get("boundary_ids") != job["boundary_ids"]:
            raise FinalRenderError(
                f"MFA context {context_id} has stale boundary ownership"
            )
        originals = context.get("original_source_words")
        if not isinstance(originals, list) or len(originals) != len(job["local_words"]):
            raise FinalRenderError(
                f"MFA context {context_id} has stale source-word metadata"
            )
        for expected_word, actual_word in zip(
            job["local_words"], originals, strict=True
        ):
            if not isinstance(actual_word, dict) or (
                actual_word.get("word_id", actual_word.get("id")) != expected_word["id"]
                or actual_word.get("text") != expected_word["text"]
                or actual_word.get("selected") is not expected_word["selected"]
                or abs(
                    float(
                        actual_word.get(
                            "start_seconds",
                            actual_word.get("start"),
                        )
                    )
                    - float(expected_word["start"])
                )
                > ALIGNMENT_GEOMETRY_EPSILON_SECONDS
                or abs(
                    float(
                        actual_word.get(
                            "end_seconds",
                            actual_word.get("end"),
                        )
                    )
                    - float(expected_word["end"])
                )
                > ALIGNMENT_GEOMETRY_EPSILON_SECONDS
            ):
                raise FinalRenderError(
                    f"MFA context {context_id} has stale source-word metadata"
                )
    return context_by_id, errors_by_id


def _selected_word_ids(ranges: Sequence[MergedRange]) -> set[int]:
    return {
        word_id
        for source_range in ranges
        for word_id in range(
            source_range.start_word_id,
            source_range.end_word_id,
        )
    }


def _retained_mfa_phone_mask(
    *,
    mfa_contexts: Sequence[dict[str, Any]],
    source_intervals: Sequence[dict[str, Any]],
    selected_word_ids: set[int],
    sample_rate: int,
    total_samples: int,
) -> list[dict[str, Any]]:
    """Protect every MFA non-silence phone intersecting retained source."""

    margin = round(PROTECTED_SPEECH_MARGIN_MS * sample_rate / 1000.0)
    mask: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[int, ...], str, int, int, int, int]] = set()
    for context in mfa_contexts:
        context_id = str(context.get("context_id", ""))
        aligned_words = [
            word for word in context.get("words", []) if isinstance(word, dict)
        ]
        for phone in context.get("phones", []):
            if not isinstance(phone, dict) or bool(phone.get("is_silence")):
                continue
            phone_start = seconds_to_sample(
                float(phone["start_seconds"]),
                sample_rate,
                boundary="ceil",
            )
            phone_end = seconds_to_sample(
                float(phone["end_seconds"]),
                sample_rate,
                boundary="ceil",
            )
            if not 0 <= phone_start < phone_end <= total_samples:
                raise FinalRenderError("retained MFA phone has invalid geometry")
            mapped_ids: set[int] = set()
            for aligned_word in aligned_words:
                word_ids = {
                    int(value)
                    for value in aligned_word.get("source_word_ids", [])
                    if type(value) is int and int(value) in selected_word_ids
                }
                if not word_ids:
                    continue
                for word_phone in aligned_word.get("phones", []):
                    if not isinstance(word_phone, dict):
                        continue
                    word_phone_start = seconds_to_sample(
                        float(word_phone["start_seconds"]),
                        sample_rate,
                        boundary="ceil",
                    )
                    word_phone_end = seconds_to_sample(
                        float(word_phone["end_seconds"]),
                        sample_rate,
                        boundary="ceil",
                    )
                    if (
                        str(word_phone.get("phone", "")) == str(phone.get("phone", ""))
                        and abs(word_phone_start - phone_start)
                        <= MFA_SAMPLE_ROUNDING_OVERLAP
                        and abs(word_phone_end - phone_end)
                        <= MFA_SAMPLE_ROUNDING_OVERLAP
                    ):
                        mapped_ids.update(word_ids)
            retained_ids = tuple(sorted(mapped_ids))
            for interval in source_intervals:
                interval_start = int(interval["source_start_sample"])
                interval_end = int(interval["source_end_sample"])
                if not _intervals_overlap(
                    phone_start,
                    phone_end,
                    interval_start,
                    interval_end,
                ):
                    continue
                protected_start = max(interval_start, phone_start - margin)
                protected_end = min(interval_end, phone_end + margin)
                interval_index = int(interval["source_interval_index"])
                key = (
                    interval_index,
                    retained_ids,
                    str(phone.get("phone", "")),
                    phone_start,
                    phone_end,
                    protected_start,
                    protected_end,
                )
                if key in seen:
                    continue
                seen.add(key)
                mask.append(
                    {
                        "source_interval_index": interval_index,
                        "source_word_ids": list(retained_ids),
                        "phone": str(phone.get("phone", "")),
                        "phone_start_sample": phone_start,
                        "phone_end_sample": phone_end,
                        "start_sample": protected_start,
                        "end_sample": protected_end,
                        "protected_margin_ms": PROTECTED_SPEECH_MARGIN_MS,
                        "alignment_backend": "mfa",
                        "context_ids": [context_id],
                        "source_word_mapping": (
                            "mapped_retained_word"
                            if retained_ids
                            else "context_non_silence_phone"
                        ),
                    }
                )
    return sorted(
        mask,
        key=lambda item: (
            int(item["start_sample"]),
            int(item["end_sample"]),
            str(item["phone"]),
        ),
    )


def _collect_mfa_non_silence_source_spans(
    *,
    mfa_contexts: Sequence[dict[str, Any]],
    sample_rate: int,
    total_samples: int,
) -> list[dict[str, Any]]:
    """Union every context-level MFA non-silence phone in source time."""

    raw: list[tuple[int, int, str, str]] = []
    for context in mfa_contexts:
        context_id = str(context.get("context_id", ""))
        for phone in context.get("phones", []):
            if not isinstance(phone, dict) or bool(phone.get("is_silence")):
                continue
            start = max(
                0,
                seconds_to_sample(
                    float(phone["start_seconds"]),
                    sample_rate,
                    boundary="ceil",
                ),
            )
            end = min(
                total_samples,
                seconds_to_sample(
                    float(phone["end_seconds"]),
                    sample_rate,
                    boundary="ceil",
                ),
            )
            if end > start:
                raw.append((start, end, context_id, str(phone.get("phone", ""))))
    merged: list[dict[str, Any]] = []
    for start, end, context_id, phone in sorted(raw):
        if not merged or start > int(merged[-1]["end_sample"]):
            merged.append(
                {
                    "start_sample": start,
                    "end_sample": end,
                    "context_ids": [context_id],
                    "phones": [phone],
                    "verification": "mfa_non_silence_phone",
                }
            )
            continue
        merged[-1]["end_sample"] = max(int(merged[-1]["end_sample"]), end)
        if context_id not in merged[-1]["context_ids"]:
            merged[-1]["context_ids"].append(context_id)
        if phone not in merged[-1]["phones"]:
            merged[-1]["phones"].append(phone)
    return merged


def _mfa_gap_evidence(
    *,
    jobs: Sequence[dict[str, Any]],
    mfa_context_by_key: dict[str, dict[str, Any]],
    source_intervals: Sequence[dict[str, Any]],
    sample_rate: int,
) -> list[dict[str, Any]]:
    """Return retained gaps that MFA proves contain no non-silence phone."""

    evidence: list[dict[str, Any]] = []
    for job in jobs:
        if job["event_kind"] not in {
            "breath_retained_gap",
            "internal_thought_gap",
        }:
            continue
        context_id = str(job["event_key"])
        context = mfa_context_by_key.get(context_id)
        if context is None:
            continue
        role_ids = job["role_word_ids"]
        previous_id = int(role_ids["previous_retained"])
        next_id = int(role_ids["next_retained"])
        try:
            previous = source_word_alignment(context, previous_id)
            following = source_word_alignment(context, next_id)
            previous_end = seconds_to_sample(
                float(previous["last_non_silence_phone"]["end_seconds"]),
                sample_rate,
                boundary="ceil",
            )
            following_start = seconds_to_sample(
                float(following["first_non_silence_phone"]["start_seconds"]),
                sample_rate,
                boundary="ceil",
            )
            non_speech = _mfa_non_speech_interval(
                context=context,
                start_sample=previous_end,
                end_sample=following_start,
                sample_rate=sample_rate,
            )
        except (MFAAlignmentError, KeyError, TypeError, ValueError):
            continue
        if non_speech is None:
            continue
        interval = source_intervals[int(job["range_index"])]
        start = max(
            int(non_speech["start_sample"]),
            int(interval["source_start_sample"]),
        )
        end = min(
            int(non_speech["end_sample"]),
            int(interval["source_end_sample"]),
        )
        if end <= start:
            continue
        evidence.append(
            {
                "start_sample": start,
                "end_sample": end,
                "source_interval_index": int(interval["source_interval_index"]),
                "context_id": context_id,
                "previous_word_id": previous_id,
                "next_word_id": next_id,
                "verification": "mfa_phone_free_interval",
            }
        )
    return evidence


def _mfa_silence_source_spans(
    *,
    mfa_contexts: Sequence[dict[str, Any]],
    non_silence_spans: Sequence[dict[str, Any]],
    sample_rate: int,
    total_samples: int,
) -> list[dict[str, Any]]:
    """Collect explicit MFA silence phones as verified room-tone candidates."""

    raw: list[tuple[int, int, str]] = []
    for context in mfa_contexts:
        context_id = str(context.get("context_id", ""))
        for phone in context.get("phones", []):
            if not isinstance(phone, dict) or not bool(phone.get("is_silence")):
                continue
            start = max(
                0,
                seconds_to_sample(
                    float(phone["start_seconds"]),
                    sample_rate,
                    boundary="ceil",
                ),
            )
            end = min(
                total_samples,
                seconds_to_sample(
                    float(phone["end_seconds"]),
                    sample_rate,
                    boundary="ceil",
                ),
            )
            if end > start:
                raw.append((start, end, context_id))
    safe_raw: list[tuple[int, int, str]] = []
    exclusions = [
        (int(item["start_sample"]), int(item["end_sample"]))
        for item in non_silence_spans
    ]
    for start, end, context_id in raw:
        safe_raw.extend(
            (safe_start, safe_end, context_id)
            for safe_start, safe_end in _subtract_sample_intervals(
                [(start, end)],
                exclusions,
            )
        )
    merged: list[dict[str, Any]] = []
    for start, end, context_id in sorted(safe_raw):
        if not merged or start > int(merged[-1]["end_sample"]):
            merged.append(
                {
                    "start_sample": start,
                    "end_sample": end,
                    "context_ids": [context_id],
                    "verification": "mfa_silence_phone",
                }
            )
            continue
        merged[-1]["end_sample"] = max(int(merged[-1]["end_sample"]), end)
        if context_id not in merged[-1]["context_ids"]:
            merged[-1]["context_ids"].append(context_id)
    return merged


def _subtract_sample_intervals(
    spans: Sequence[tuple[int, int]],
    exclusions: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    remaining = list(spans)
    for excluded_start, excluded_end in sorted(exclusions):
        next_remaining: list[tuple[int, int]] = []
        for start, end in remaining:
            overlap_start = max(start, excluded_start)
            overlap_end = min(end, excluded_end)
            if overlap_end <= overlap_start:
                next_remaining.append((start, end))
                continue
            if start < overlap_start:
                next_remaining.append((start, overlap_start))
            if overlap_end < end:
                next_remaining.append((overlap_end, end))
        remaining = next_remaining
    return remaining


def _subtract_artifact_replacements_from_breath_events(
    *,
    events: Sequence[dict[str, Any]],
    artifact_replacements: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prevent two cleanup decisions from owning the same canonical samples."""

    artifact_targets = [
        (
            int(replacement["target_start_sample"]),
            int(replacement["target_end_sample"]),
        )
        for replacement in artifact_replacements
    ]
    fragments: list[dict[str, Any]] = []
    covered_records: list[dict[str, Any]] = []
    for source_event_index, event in enumerate(events):
        event_start = int(event["start_sample"])
        event_end = int(event["end_sample"])
        uncovered = _subtract_sample_intervals(
            [(event_start, event_end)],
            artifact_targets,
        )
        if not uncovered:
            covered_records.append(
                {
                    **dict(event),
                    "event_index": source_event_index,
                    "source_event_index": source_event_index,
                    "editable_intersection": [],
                    "protected_phone_intersections": [],
                    "replacements": [
                        replacement["replacement_id"]
                        for replacement in artifact_replacements
                        if _intervals_overlap(
                            event_start,
                            event_end,
                            int(replacement["target_start_sample"]),
                            int(replacement["target_end_sample"]),
                        )
                    ],
                    "status": "breath_cleanup_covered_by_artifact_replacement",
                }
            )
            continue
        for fragment_index, (fragment_start, fragment_end) in enumerate(uncovered):
            fragments.append(
                {
                    **dict(event),
                    "start_sample": fragment_start,
                    "end_sample": fragment_end,
                    "source_event_index": source_event_index,
                    "fragment_index": fragment_index,
                }
            )
    return fragments, covered_records


def _aggregate_breath_fragment_records(
    *,
    source_events: Sequence[dict[str, Any]],
    fragment_records: Sequence[dict[str, Any]],
    covered_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Restore one stable diagnostic record per detector event."""

    covered_by_index = {
        int(record["source_event_index"]): dict(record) for record in covered_records
    }
    aggregated: list[dict[str, Any]] = []
    for source_event_index, source_event in enumerate(source_events):
        covered = covered_by_index.get(source_event_index)
        if covered is not None:
            aggregated.append(covered)
            continue
        fragments = [
            dict(record)
            for record in fragment_records
            if int(record.get("source_event_index", record.get("event_index", -1)))
            == source_event_index
        ]
        replacements = [
            str(replacement_id)
            for fragment in fragments
            for replacement_id in fragment.get("replacements", [])
        ]
        protected = [
            dict(interval)
            for fragment in fragments
            for interval in fragment.get("protected_phone_intersections", [])
        ]
        editable = [
            dict(interval)
            for fragment in fragments
            for interval in fragment.get("editable_intersection", [])
        ]
        statuses = [str(fragment.get("status", "")) for fragment in fragments]
        if replacements:
            status = "breath_replaced_with_verified_clean_ambience"
        elif "breath_cleanup_skipped_phone_overlap" in statuses:
            status = "breath_cleanup_skipped_phone_overlap"
        elif "clean_ambience_unavailable" in statuses:
            status = "clean_ambience_unavailable"
        else:
            status = "breath_cleanup_skipped_no_mfa_non_speech"
        aggregated.append(
            {
                **dict(source_event),
                "event_index": source_event_index,
                "source_event_index": source_event_index,
                "fragments": fragments,
                "editable_intersection": editable,
                "protected_phone_intersections": protected,
                "replacements": replacements,
                "status": status,
            }
        )
    return aggregated


def _suppress_fades_over_protected_speech(
    *,
    boundaries: Sequence[dict[str, Any]],
    joins: Sequence[dict[str, Any]],
    protected_speech_mask: Sequence[dict[str, Any]],
) -> None:
    """Remove any fade that conflicts with the authoritative MFA mask.

    Boundary-local alignment contexts can expose a slightly larger quiet
    handle than another context for the same retained phone.  The union mask
    is authoritative.  Shortening a ramp would change its gain geometry, so
    the conservative operation is to suppress the whole fade before the plan
    becomes immutable.
    """

    for owner in [*boundaries, *joins]:
        fades = owner.get("fade_intervals", [])
        if not isinstance(fades, list) or not fades:
            continue
        safe: list[dict[str, Any]] = []
        suppressed: list[dict[str, Any]] = []
        for raw_fade in fades:
            fade = dict(raw_fade)
            overlaps = [
                {
                    "source_word_ids": list(interval.get("source_word_ids", [])),
                    "phone": interval.get("phone"),
                    "start_sample": int(interval["start_sample"]),
                    "end_sample": int(interval["end_sample"]),
                }
                for interval in protected_speech_mask
                if _intervals_overlap(
                    int(fade["source_start_sample"]),
                    int(fade["source_end_sample"]),
                    int(interval["start_sample"]),
                    int(interval["end_sample"]),
                )
            ]
            if overlaps:
                suppressed.append(
                    {
                        **fade,
                        "reason": "mfa_protected_speech_or_margin_overlap",
                        "protected_intersections": overlaps,
                    }
                )
            else:
                safe.append(fade)
        owner["fade_intervals"] = safe
        if suppressed:
            owner.setdefault("suppressed_fade_intervals", []).extend(suppressed)


def _editable_non_speech_intervals(
    *,
    gap_evidence: Sequence[dict[str, Any]],
    source_intervals: Sequence[dict[str, Any]],
    boundaries: Sequence[dict[str, Any]],
    joins: Sequence[dict[str, Any]],
    protected_speech_mask: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Union only MFA-confirmed gaps that are physically retained in output."""

    boundary_by_id = {str(item["boundary_id"]): item for item in boundaries}
    candidates: list[dict[str, Any]] = [dict(item) for item in gap_evidence]
    for interval in source_intervals:
        interval_start = int(interval["source_start_sample"])
        interval_end = int(interval["source_end_sample"])
        for boundary_key in ("start_boundary_id", "end_boundary_id"):
            boundary = boundary_by_id[str(interval[boundary_key])]
            quiet = boundary.get("verified_quiet_interval")
            if not isinstance(quiet, dict):
                continue
            start = max(interval_start, int(quiet["start_sample"]))
            end = min(interval_end, int(quiet["end_sample"]))
            if end > start:
                candidates.append(
                    {
                        "start_sample": start,
                        "end_sample": end,
                        "source_interval_index": int(interval["source_interval_index"]),
                        "context_id": boundary.get("mfa_context_id"),
                        "verification": "mfa_boundary_phone_free_interval",
                    }
                )

    split_points_by_interval: dict[int, set[int]] = {}
    fade_exclusions_by_interval: dict[int, list[tuple[int, int]]] = {}
    for join in joins:
        if join.get("join_kind") != "internal_thought_pause":
            continue
        interval_index = int(join["source_interval_index"])
        insertion = join.get("source_insertion_sample")
        if type(insertion) is int:
            split_points_by_interval.setdefault(interval_index, set()).add(insertion)
        for fade in join.get("fade_intervals", []):
            fade_exclusions_by_interval.setdefault(interval_index, []).append(
                (
                    int(fade["source_start_sample"]),
                    int(fade["source_end_sample"]),
                )
            )
    for interval in source_intervals:
        interval_index = int(interval["source_interval_index"])
        for boundary_key in ("start_boundary_id", "end_boundary_id"):
            boundary = boundary_by_id[str(interval[boundary_key])]
            for fade in boundary.get("fade_intervals", []):
                fade_exclusions_by_interval.setdefault(interval_index, []).append(
                    (
                        int(fade["source_start_sample"]),
                        int(fade["source_end_sample"]),
                    )
                )

    result: list[dict[str, Any]] = []
    for interval in source_intervals:
        interval_index = int(interval["source_interval_index"])
        raw_spans = sorted(
            (
                max(
                    int(item["start_sample"]),
                    int(interval["source_start_sample"]),
                ),
                min(
                    int(item["end_sample"]),
                    int(interval["source_end_sample"]),
                ),
            )
            for item in candidates
            if int(item["source_interval_index"]) == interval_index
        )
        merged: list[list[int]] = []
        for start, end in raw_spans:
            if end <= start:
                continue
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        exclusions = [
            (int(item["start_sample"]), int(item["end_sample"]))
            for item in protected_speech_mask
            if int(item["source_interval_index"]) == interval_index
        ]
        exclusions.extend(fade_exclusions_by_interval.get(interval_index, []))
        safe_spans = _subtract_sample_intervals(
            [(start, end) for start, end in merged],
            exclusions,
        )
        for split_point in sorted(split_points_by_interval.get(interval_index, set())):
            safe_spans = [
                piece
                for start, end in safe_spans
                for piece in (
                    ((start, split_point), (split_point, end))
                    if start < split_point < end
                    else ((start, end),)
                )
                if piece[1] > piece[0]
            ]
        result.extend(
            {
                "start_sample": start,
                "end_sample": end,
                "source_interval_index": interval_index,
                "verification": "mfa_confirmed_editable_non_speech",
            }
            for start, end in safe_spans
        )
    return result


def _baseline_room_tone_exclusions(
    *,
    protected_speech_mask: Sequence[dict[str, Any]],
    mfa_non_silence_spans: Sequence[dict[str, Any]],
    boundaries: Sequence[dict[str, Any]],
    joins: Sequence[dict[str, Any]],
    words: Sequence[PlanWord],
    sample_rate: int,
    total_samples: int,
) -> list[dict[str, Any]]:
    exclusions = [
        {
            "start_sample": int(item["start_sample"]),
            "end_sample": int(item["end_sample"]),
            "reason": "mfa_non_silence_phone",
        }
        for item in mfa_non_silence_spans
    ]
    exclusions.extend(
        {
            "start_sample": int(item["start_sample"]),
            "end_sample": int(item["end_sample"]),
            "reason": "mfa_protected_phone_margin",
        }
        for item in protected_speech_mask
    )
    for boundary in boundaries:
        selected = boundary.get("selected_source_sample")
        if type(selected) is int:
            exclusions.append(
                {
                    "start_sample": max(0, selected - 1),
                    "end_sample": min(total_samples, selected + 1),
                    "reason": "cut_transition",
                }
            )
        for word_id in boundary.get("forbidden_word_ids", []):
            if type(word_id) is not int or not 0 <= word_id < len(words):
                continue
            exclusions.append(
                {
                    "start_sample": max(
                        0,
                        math.floor(words[word_id].start * sample_rate),
                    ),
                    "end_sample": min(
                        total_samples,
                        math.ceil(words[word_id].end * sample_rate),
                    ),
                    "reason": "forbidden_or_weak_word",
                }
            )
    for join in joins:
        insertion = join.get("source_insertion_sample")
        if type(insertion) is int:
            exclusions.append(
                {
                    "start_sample": max(0, insertion - 1),
                    "end_sample": min(total_samples, insertion + 1),
                    "reason": "cut_transition",
                }
            )
    return exclusions


def _merge_verified_ambience_spans(
    spans: Sequence[dict[str, Any]],
    *,
    total_samples: int,
) -> list[dict[str, Any]]:
    """Union MFA-confirmed non-speech while retaining its provenance."""

    merged: list[dict[str, Any]] = []
    for raw in sorted(
        spans,
        key=lambda item: (int(item["start_sample"]), int(item["end_sample"])),
    ):
        start = max(0, min(total_samples, int(raw["start_sample"])))
        end = max(start, min(total_samples, int(raw["end_sample"])))
        if end <= start:
            continue
        raw_verification = raw.get("verification", "mfa_confirmed_non_speech")
        verifications = (
            [str(value) for value in raw_verification]
            if isinstance(raw_verification, list)
            else [str(raw_verification)]
        )
        context_ids = {str(value) for value in raw.get("context_ids", []) if str(value)}
        context_id = raw.get("context_id")
        if context_id:
            context_ids.add(str(context_id))
        if not merged or start > int(merged[-1]["end_sample"]):
            merged.append(
                {
                    "start_sample": start,
                    "end_sample": end,
                    "verification": verifications,
                    "context_ids": sorted(context_ids),
                }
            )
            continue
        merged[-1]["end_sample"] = max(int(merged[-1]["end_sample"]), end)
        for verification in verifications:
            if verification not in merged[-1]["verification"]:
                merged[-1]["verification"].append(verification)
        merged[-1]["context_ids"] = sorted({*merged[-1]["context_ids"], *context_ids})
    return merged


def _ambience_candidate_pieces(
    *,
    verified_spans: Sequence[dict[str, Any]],
    exclusions: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split verified spans so every accepted/rejected sample is inspectable."""

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    candidate_index = 0
    rejected_index = 0
    for span in verified_spans:
        span_start = int(span["start_sample"])
        span_end = int(span["end_sample"])
        overlapping = [
            exclusion
            for exclusion in exclusions
            if _intervals_overlap(
                span_start,
                span_end,
                int(exclusion["start_sample"]),
                int(exclusion["end_sample"]),
            )
        ]
        cut_points = {span_start, span_end}
        for exclusion in overlapping:
            cut_points.add(max(span_start, int(exclusion["start_sample"])))
            cut_points.add(min(span_end, int(exclusion["end_sample"])))
        ordered = sorted(cut_points)
        for start, end in zip(ordered, ordered[1:]):
            if end <= start:
                continue
            reasons = sorted(
                {
                    str(exclusion.get("reason", "excluded_source_interval"))
                    for exclusion in overlapping
                    if _intervals_overlap(
                        start,
                        end,
                        int(exclusion["start_sample"]),
                        int(exclusion["end_sample"]),
                    )
                }
            )
            common = {
                "source_start_sample": start,
                "source_end_sample": end,
                "duration_samples": end - start,
                "mfa_verification": list(span["verification"]),
                "mfa_context_ids": list(span["context_ids"]),
            }
            if reasons:
                rejected.append(
                    {
                        "candidate_id": f"ambience_rejected_{rejected_index:05d}",
                        **common,
                        "accepted": False,
                        "status": "rejected",
                        "rejection_reasons": [
                            {"code": reason, "source": "source_mask"}
                            for reason in reasons
                        ],
                    }
                )
                rejected_index += 1
                continue
            candidates.append(
                {
                    "candidate_id": f"ambience_{candidate_index:05d}",
                    **common,
                }
            )
            candidate_index += 1
    return candidates, rejected


def _build_verified_ambience_bank(
    *,
    source_audio: np.ndarray,
    sample_rate: int,
    total_samples: int,
    verified_spans: Sequence[dict[str, Any]],
    exclusions: Sequence[dict[str, Any]],
    detector_status: str,
    detector_error: str | None,
) -> dict[str, Any]:
    """Build the sole production bank from untouched canonical samples."""

    merged_spans = _merge_verified_ambience_spans(
        verified_spans,
        total_samples=total_samples,
    )
    candidates, mask_rejections = _ambience_candidate_pieces(
        verified_spans=merged_spans,
        exclusions=exclusions,
    )
    bank = build_clean_ambience_bank(
        source_audio,
        candidates=candidates,
        sample_rate=sample_rate,
        thresholds=DEFAULT_AMBIENCE_THRESHOLDS,
    )
    accepted = list(bank["accepted_candidates"])
    if accepted:
        target_rms_db = float(
            median(float(item["metrics"]["rms_db"]) for item in accepted)
        )
        for item in bank["candidates"]:
            item["target_rms_db"] = target_rms_db
            item["noise_level_delta_db"] = abs(
                float(item["metrics"]["rms_db"]) - target_rms_db
            )
        bank["target_rms_db"] = target_rms_db
    deterministic_rejections = list(bank["rejected_candidates"])
    all_rejections = [*mask_rejections, *deterministic_rejections]
    for item in all_rejections:
        reasons = item.get("rejection_reasons", [])
        item["reason"] = (
            str(reasons[0].get("code", "rejected"))
            if isinstance(reasons, list) and reasons
            else "rejected"
        )
    bank["schema_version"] = AMBIENCE_BANK_SCHEMA_VERSION
    bank["detector_status"] = detector_status
    bank["detector_error"] = detector_error
    bank["verified_source_spans"] = merged_spans
    bank["candidates"] = [*bank["candidates"], *mask_rejections]
    bank["rejected_candidates"] = all_rejections
    if detector_status not in {"complete", "no_relevant_regions"}:
        # Respiro failure must not allow unverified ambience into inserted or
        # cleaned pauses.  The edit can still render with source gaps intact.
        rejection_code = (
            "nuisance_detector_disabled"
            if detector_status == "disabled_by_user"
            else "breath_detector_unavailable"
        )
        invalidated = [
            {
                **dict(item),
                "accepted": False,
                "status": "rejected",
                "rejection_reasons": [
                    *list(item.get("rejection_reasons", [])),
                    {"code": rejection_code, "source": "source_mask"},
                ],
                "reason": rejection_code,
            }
            for item in accepted
        ]
        invalidated_by_id = {str(item["candidate_id"]): item for item in invalidated}
        bank["candidates"] = [
            invalidated_by_id.get(str(item.get("candidate_id")), item)
            for item in bank["candidates"]
        ]
        bank["rejected_candidates"] = [*all_rejections, *invalidated]
        bank["accepted_candidates"] = []
        bank["status"] = "clean_ambience_unavailable"
        bank["failure_reason"] = rejection_code
    bank["accepted_candidate_count"] = len(bank["accepted_candidates"])
    bank["rejected_candidate_count"] = len(bank["rejected_candidates"])
    return bank


def _internal_gap_nuisance_evidence(
    *,
    joins: Sequence[dict[str, Any]],
    source_audio: np.ndarray,
    sample_rate: int,
    detected_events: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flag deterministic artifacts only inside retained MFA-confirmed gaps."""

    guard = round(AMBIENCE_ARTIFACT_GUARD_MS * sample_rate / 1000.0)
    artifact_codes = {
        "clipping",
        "excessive_crest_factor",
        "excessive_spectral_flux",
        "unstable_log_band_energy",
        "sudden_rms_burst",
        "sample_discontinuity",
    }
    artifact_events: list[dict[str, Any]] = []
    for join in joins:
        pause_content = join.get("pause_content")
        if not isinstance(pause_content, dict):
            continue
        if join.get("join_kind") != "internal_thought_pause":
            pause_content["original_gap_content"] = "omitted_gap_not_used"
            continue
        quiet = join.get("verified_quiet_interval")
        if not isinstance(quiet, dict):
            pause_content["original_gap_content"] = "preserved_unverified"
            continue
        gap_start = int(quiet["start_sample"])
        gap_end = int(quiet["end_sample"])
        breath_intersections = [
            {
                "kind": "breath",
                "start_sample": max(gap_start, int(event["start_sample"])),
                "end_sample": min(gap_end, int(event["end_sample"])),
                "maximum_probability": event.get("maximum_probability"),
                "mean_probability": event.get("mean_probability"),
            }
            for event in detected_events
            if _intervals_overlap(
                gap_start,
                gap_end,
                int(event["start_sample"]),
                int(event["end_sample"]),
            )
        ]
        evaluation_start = gap_start + guard
        evaluation_end = gap_end - guard
        artifact_evidence: dict[str, Any] | None = None
        if evaluation_end > evaluation_start:
            artifact_evidence = evaluate_ambience_candidate(
                source_audio,
                candidate_id=f"existing_gap_{join['join_id']}",
                start_sample=evaluation_start,
                end_sample=evaluation_end,
                sample_rate=sample_rate,
                thresholds=DEFAULT_AMBIENCE_THRESHOLDS,
            )
        artifact_reasons = (
            [
                dict(reason)
                for reason in artifact_evidence["rejection_reasons"]
                if str(reason.get("code")) in artifact_codes
            ]
            if artifact_evidence is not None
            else []
        )
        artifact_intersections: list[dict[str, Any]] = []
        if artifact_reasons:
            artifact = {
                "event_type": "deterministic_nonstationary_artifact",
                "start_sample": gap_start,
                "end_sample": gap_end,
                "analysis_start_sample": evaluation_start,
                "analysis_end_sample": evaluation_end,
                "join_id": str(join["join_id"]),
                "metrics": artifact_evidence["metrics"],
                "rejection_reasons": artifact_reasons,
                "local_noise_floor_db": _local_source_noise_floor_db(
                    source_audio,
                    sample_rate=sample_rate,
                    reference_sample=(gap_start + gap_end) // 2,
                ),
            }
            artifact_events.append(artifact)
            artifact_intersections.append(artifact)
        pause_content["nuisance_mask_intersections"] = [
            *breath_intersections,
            *artifact_intersections,
        ]
        pause_content["existing_gap_stationarity"] = (
            {
                "analysis_start_sample": evaluation_start,
                "analysis_end_sample": evaluation_end,
                "metrics": artifact_evidence["metrics"],
                "status": (
                    "artifact_detected" if artifact_reasons else "verified_clean"
                ),
                "rejection_reasons": artifact_reasons,
            }
            if artifact_evidence is not None
            else {
                "status": "not_evaluated_insufficient_duration",
                "analysis_start_sample": evaluation_start,
                "analysis_end_sample": evaluation_end,
            }
        )
        if breath_intersections or artifact_intersections:
            pause_content["original_gap_content"] = "nuisance_detected"
        elif artifact_evidence is not None:
            pause_content["original_gap_content"] = "preserved_verified_clean"
        else:
            pause_content["original_gap_content"] = (
                "preserved_unverified_insufficient_duration"
            )
    return artifact_events


def _validate_breath_payload(
    payload: dict[str, Any],
    *,
    threshold: float,
    minimum_duration_ms: int,
    total_samples: int,
) -> dict[str, Any]:
    if (
        payload.get("backend") != "respiro-en"
        or payload.get("upstream_commit") != RESPIRO_UPSTREAM_COMMIT
        or payload.get("checkpoint_sha256") != RESPIRO_CHECKPOINT_SHA256
        or payload.get("frame_hop_ms") != RESPIRO_FRAME_HOP_MS
    ):
        raise BreathDetectionError("breath payload has incompatible provenance")
    if (
        float(payload.get("threshold", -1.0)) != threshold
        or int(payload.get("minimum_duration_ms", -1)) != minimum_duration_ms
    ):
        raise BreathDetectionError("breath payload has stale detector settings")
    if payload.get("status") not in {"complete", "no_relevant_regions"}:
        raise BreathDetectionError("breath payload is incomplete")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise BreathDetectionError("breath payload contains no event list")
    previous_end = 0
    for event in raw_events:
        if not isinstance(event, dict):
            raise BreathDetectionError("breath payload contains a malformed event")
        start = event.get("start_sample")
        end = event.get("end_sample")
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= total_samples
            or start < previous_end
        ):
            raise BreathDetectionError("breath event geometry is invalid or overlaps")
        previous_end = end
    return json.loads(json.dumps(payload))


def _room_tone_ranges(
    *,
    ambience_bank: dict[str, Any],
    frame_count: int,
    reference_sample: int,
    reference_rms_db: float,
    candidate_usage_counts: dict[str, int],
) -> dict[str, Any]:
    if frame_count <= 0:
        return plan_ambience_assembly(
            ambience_bank,
            required_samples=0,
            sample_rate=int(ambience_bank["sample_rate"]),
        )
    assembly = plan_ambience_assembly(
        ambience_bank,
        required_samples=frame_count,
        sample_rate=int(ambience_bank["sample_rate"]),
        reference_sample=reference_sample,
        reference_rms_db=reference_rms_db,
        crossfade_ms=DEFAULT_AMBIENCE_CROSSFADE_MS,
        candidate_usage_counts=candidate_usage_counts,
    )
    if assembly["status"] == "complete":
        for candidate_id in assembly["candidate_ids"]:
            key = str(candidate_id)
            candidate_usage_counts[key] = candidate_usage_counts.get(key, 0) + 1
    return assembly


def _local_source_noise_floor_db(
    source_audio: np.ndarray,
    *,
    sample_rate: int,
    reference_sample: int,
    radius_ms: float = 750.0,
) -> float:
    """Estimate the nearby canonical noise floor for ambience ranking only."""

    radius = max(1, round(radius_ms * sample_rate / 1000.0))
    start = max(0, reference_sample - radius)
    end = min(len(source_audio), reference_sample + radius)
    frame_samples = max(1, round(20.0 * sample_rate / 1000.0))
    hop_samples = max(1, round(10.0 * sample_rate / 1000.0))
    window = np.asarray(source_audio[start:end], dtype=np.float64)
    if not len(window):
        return -120.0
    starts = list(range(0, max(1, len(window) - frame_samples + 1), hop_samples))
    if not starts:
        starts = [0]
    frame_rms = []
    for frame_start in starts:
        frame = window[frame_start : frame_start + frame_samples]
        if len(frame):
            frame_rms.append(float(np.sqrt(np.mean(np.square(frame)))))
    if not frame_rms:
        return -120.0
    rms = float(np.percentile(np.asarray(frame_rms), 20.0))
    return 20.0 * math.log10(max(rms, 1.0e-12))


def _initialize_pause_content(joins: Sequence[dict[str, Any]]) -> None:
    """Create the pause provenance ledger before nuisance/bank analysis."""

    for join in joins:
        join["room_tone_source_ranges"] = []
        join["room_tone_fade_samples"] = 0
        join["room_tone_crossfades"] = []
        join["ambience_assembly"] = None
        join["requested_inserted_pause_samples"] = int(join["inserted_pause_samples"])
        join["pause_content"] = {
            "pause_type": join["pause_type"],
            "target_pause_samples": int(join["target_pause_samples"]),
            "original_gap_samples": int(join["estimated_existing_pause_samples"]),
            "original_content_preserved": (
                join["join_kind"] == "internal_thought_pause"
            ),
            "original_content_replaced": False,
            "nuisance_mask_intersections": [],
            "clean_ambience_candidate_ids": [],
            "source_to_output_sample_mapping": [],
            "crossfades": [],
            "stationarity_metrics": [],
            "status": (
                "original_gap_satisfies_target"
                if int(join["inserted_pause_samples"]) <= 0
                else "not_planned"
            ),
        }


def _allocate_semantic_pause_room_tone(
    *,
    ambience_bank: dict[str, Any],
    joins: Sequence[dict[str, Any]],
    source_intervals: Sequence[dict[str, Any]],
    source_audio: np.ndarray,
    candidate_usage_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Allocate pause content without changing any resolved source endpoint."""

    usage_counts = candidate_usage_counts if candidate_usage_counts is not None else {}
    allocation_ledger: list[dict[str, Any]] = []
    if any(not isinstance(join.get("pause_content"), dict) for join in joins):
        _initialize_pause_content(joins)
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
        reference_rms_db = _local_source_noise_floor_db(
            source_audio,
            sample_rate=int(ambience_bank["sample_rate"]),
            reference_sample=reference,
        )
        assembly = _room_tone_ranges(
            ambience_bank=ambience_bank,
            frame_count=frame_count,
            reference_sample=reference,
            reference_rms_db=reference_rms_db,
            candidate_usage_counts=usage_counts,
        )
        ledger_entry = {
            "join_id": str(join["join_id"]),
            "reference_sample": reference,
            "reference_noise_floor_db": reference_rms_db,
            **json.loads(json.dumps(assembly)),
        }
        allocation_ledger.append(ledger_entry)
        join["ambience_assembly"] = json.loads(json.dumps(assembly))
        join["pause_content"]["surrounding_noise_floor_db"] = reference_rms_db
        if assembly["status"] != "complete":
            join["inserted_pause_samples"] = 0
            join["inserted_pause_ms"] = 0.0
            join["safety_status"] = "safe"
            join["pause_content"]["status"] = "clean_ambience_unavailable"
            continue
        trace = list(assembly["source_trace"])
        join["room_tone_source_ranges"] = [
            {
                "candidate_id": item["candidate_id"],
                "source_start_sample": int(item["source_start_sample"]),
                "source_end_sample": int(item["source_end_sample"]),
            }
            for item in trace
        ]
        join["room_tone_crossfades"] = list(assembly["crossfades"])
        join["pause_content"].update(
            {
                "clean_ambience_candidate_ids": list(assembly["candidate_ids"]),
                "source_to_output_sample_mapping": trace,
                "crossfades": list(assembly["crossfades"]),
                "stationarity_metrics": [
                    {
                        "candidate_id": item["candidate_id"],
                        "stationarity_score": item["stationarity_score"],
                        "noise_level_delta_db": item["noise_level_delta_db"],
                    }
                    for item in trace
                ],
                "status": "verified_clean_ambience",
            }
        )
    return allocation_ledger


def _append_source_segment(
    segments: list[dict[str, Any]],
    *,
    source_start: int,
    source_end: int,
    output_cursor: int,
    fades: Sequence[dict[str, Any]],
    source_interval_index: int,
    breath_replacements: Sequence[dict[str, Any]],
) -> int:
    if source_end <= source_start:
        return output_cursor
    local_fades = [
        dict(fade)
        for fade in fades
        if int(fade["source_start_sample"]) >= source_start
        and int(fade["source_end_sample"]) <= source_end
    ]
    local_replacements = [
        json.loads(json.dumps(replacement))
        for replacement in breath_replacements
        if int(replacement["target_start_sample"]) >= source_start
        and int(replacement["target_end_sample"]) <= source_end
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
            "sample_replacements": local_replacements,
        }
    )
    return output_end


def _append_ambience_segment(
    segments: list[dict[str, Any]],
    *,
    assembly: dict[str, Any],
    output_cursor: int,
    join_id: str,
) -> int:
    if assembly.get("status") != "complete":
        raise FinalRenderError("planned pause has no verified clean ambience")
    frame_count = int(assembly["planned_output_samples"])
    if frame_count <= 0:
        raise FinalRenderError("positive ambience segment is empty")
    output_end = output_cursor + frame_count
    trace = []
    for item in assembly["source_trace"]:
        relative_start = int(item["output_start_sample"])
        relative_end = int(item["output_end_sample"])
        trace.append(
            {
                **dict(item),
                "output_start_sample": output_cursor + relative_start,
                "output_end_sample": output_cursor + relative_end,
            }
        )
    crossfades = []
    for item in assembly["crossfades"]:
        relative_start = int(item["output_start_sample"])
        relative_end = int(item["output_end_sample"])
        crossfades.append(
            {
                **dict(item),
                "output_start_sample": output_cursor + relative_start,
                "output_end_sample": output_cursor + relative_end,
            }
        )
    segments.append(
        {
            "segment_index": len(segments),
            "kind": "ambience",
            "join_id": join_id,
            "output_start_sample": output_cursor,
            "output_end_sample": output_end,
            "source_trace": trace,
            "equal_power_crossfades": crossfades,
            "source_reuse": bool(assembly.get("source_reuse", False)),
        }
    )
    return output_end


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
    breath_replacements: Sequence[dict[str, Any]] = (),
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
                breath_replacements=breath_replacements,
            )
            pause_start = output_cursor
            output_cursor = _append_ambience_segment(
                output_segments,
                assembly=join["ambience_assembly"],
                output_cursor=output_cursor,
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
            breath_replacements=breath_replacements,
        )
        interval["output_end_sample"] = output_cursor
        if interval_index < len(source_intervals) - 1:
            join = clip_join_by_left[interval_index]
            pause_start = output_cursor
            if int(join["inserted_pause_samples"]) > 0:
                output_cursor = _append_ambience_segment(
                    output_segments,
                    assembly=join["ambience_assembly"],
                    output_cursor=output_cursor,
                    join_id=str(join["join_id"]),
                )
            join["output_pause_start_sample"] = pause_start
            join["output_pause_end_sample"] = output_cursor
    planned_replacements = {str(item["replacement_id"]) for item in breath_replacements}
    attached_replacements = {
        str(item["replacement_id"])
        for segment in output_segments
        for item in segment.get("sample_replacements", [])
    }
    if attached_replacements != planned_replacements:
        raise FinalRenderError("breath replacement does not map to one source segment")
    ambience_segment_by_join = {
        str(segment["join_id"]): segment
        for segment in output_segments
        if segment["kind"] == "ambience"
    }
    for join in joins:
        segment = ambience_segment_by_join.get(str(join["join_id"]))
        if segment is None:
            continue
        pause_content = join["pause_content"]
        pause_content["source_to_output_sample_mapping"] = list(segment["source_trace"])
        pause_content["crossfades"] = list(segment["equal_power_crossfades"])
    return output_segments


def _plan_ambience_edge_transitions(
    *,
    segments: Sequence[dict[str, Any]],
    joins: Sequence[dict[str, Any]],
    source_audio: np.ndarray,
    sample_rate: int,
) -> None:
    """Taper ambience only when adjacent source already reaches verified quiet."""

    join_by_id = {str(join["join_id"]): join for join in joins}
    requested = max(1, round(DEFAULT_AMBIENCE_CROSSFADE_MS * sample_rate / 1000.0))
    for segment_index, segment in enumerate(segments):
        if segment.get("kind") != "ambience":
            continue
        if segment_index == 0 or segment_index + 1 >= len(segments):
            raise FinalRenderError("ambience segment has no adjacent retained source")
        left = segments[segment_index - 1]
        right = segments[segment_index + 1]
        if left.get("kind") != "source" or right.get("kind") != "source":
            raise FinalRenderError("ambience is not bounded by retained source")
        duration = int(segment["output_end_sample"]) - int(
            segment["output_start_sample"]
        )
        transition_samples = min(requested, duration // 2)
        left_source_end = int(left["source_end_sample"])
        right_source_start = int(right["source_start_sample"])
        left_fade = next(
            (
                fade
                for fade in left.get("gain_envelopes", [])
                if fade.get("curve") == "fade_out"
                and fade.get("verified_quiet") is True
                and int(fade["source_end_sample"]) == left_source_end
                and int(fade["source_end_sample"]) - int(fade["source_start_sample"])
                >= 2
            ),
            None,
        )
        right_fade = next(
            (
                fade
                for fade in right.get("gain_envelopes", [])
                if fade.get("curve") == "fade_in"
                and fade.get("verified_quiet") is True
                and int(fade["source_start_sample"]) == right_source_start
                and int(fade["source_end_sample"]) - int(fade["source_start_sample"])
                >= 2
            ),
            None,
        )
        first_trace = segment["source_trace"][0]
        last_trace = segment["source_trace"][-1]
        raw_left = source_audio[left_source_end - 1]
        raw_ambience_start = source_audio[int(first_trace["source_start_sample"])]
        raw_ambience_end = source_audio[int(last_trace["source_end_sample"]) - 1]
        raw_right = source_audio[right_source_start]
        edge_envelopes: list[dict[str, Any]] = []
        transition_records: list[dict[str, Any]] = []
        for side, source_fade, raw_source, raw_ambience in (
            ("left", left_fade, raw_left, raw_ambience_start),
            ("right", right_fade, raw_right, raw_ambience_end),
        ):
            applied = source_fade is not None and transition_samples >= 2
            if side == "left":
                ambience_start = int(segment["output_start_sample"])
                ambience_end = ambience_start + transition_samples
                curve = "equal_power_fade_in"
            else:
                ambience_end = int(segment["output_end_sample"])
                ambience_start = ambience_end - transition_samples
                curve = "equal_power_fade_out"
            if applied:
                edge_envelopes.append(
                    {
                        "side": side,
                        "curve": curve,
                        "output_start_sample": ambience_start,
                        "output_end_sample": ambience_end,
                        "verified_ambience": True,
                    }
                )
            raw_discontinuity = float(np.max(np.abs(raw_ambience - raw_source)))
            transition_records.append(
                {
                    "side": side,
                    "status": (
                        "ambience_edge_taper_applied"
                        if applied
                        else "not_applied_no_verified_source_handle"
                    ),
                    "source_fade_interval": (
                        dict(source_fade) if source_fade is not None else None
                    ),
                    "ambience_transition_interval": (
                        {
                            "output_start_sample": ambience_start,
                            "output_end_sample": ambience_end,
                            "curve": curve,
                        }
                        if applied
                        else None
                    ),
                    "raw_maximum_sample_discontinuity": raw_discontinuity,
                    "planned_maximum_sample_discontinuity": (
                        0.0 if applied else raw_discontinuity
                    ),
                }
            )
        segment["edge_gain_envelopes"] = edge_envelopes
        segment["outer_transitions"] = transition_records
        join = join_by_id[str(segment["join_id"])]
        join["pause_content"]["outer_transitions"] = transition_records
        join["pause_content"]["maximum_sample_discontinuity"] = max(
            float(record["planned_maximum_sample_discontinuity"])
            for record in transition_records
        )


def _finalize_pause_output_provenance(
    *,
    joins: Sequence[dict[str, Any]],
    segments: Sequence[dict[str, Any]],
) -> None:
    """Attach direct canonical-source-to-final-output mappings to each pause."""

    source_segments = [
        segment for segment in segments if segment.get("kind") == "source"
    ]
    replacement_location: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for segment in source_segments:
        for replacement in segment.get("sample_replacements", []):
            replacement_location[str(replacement["replacement_id"])] = (
                segment,
                replacement,
            )
    for join in joins:
        pause_content = join.get("pause_content")
        if not isinstance(pause_content, dict):
            continue
        raw_gaps: list[dict[str, Any]] = []
        quiet = join.get("verified_quiet_interval")
        if isinstance(quiet, dict):
            raw_gaps.append(quiet)
        for item in join.get("verified_quiet_intervals", []):
            if isinstance(item, dict):
                raw_gaps.append(item)
        original_mappings: list[dict[str, int]] = []
        for gap in raw_gaps:
            gap_start = int(gap["start_sample"])
            gap_end = int(gap["end_sample"])
            for segment in source_segments:
                source_start = int(segment["source_start_sample"])
                source_end = int(segment["source_end_sample"])
                retained_start = max(gap_start, source_start)
                retained_end = min(gap_end, source_end)
                if retained_start >= retained_end:
                    continue
                output_start = int(segment["output_start_sample"]) + (
                    retained_start - source_start
                )
                original_mappings.append(
                    {
                        "source_start_sample": retained_start,
                        "source_end_sample": retained_end,
                        "output_start_sample": output_start,
                        "output_end_sample": output_start
                        + retained_end
                        - retained_start,
                    }
                )
        pause_content["original_gap_source_to_output_mapping"] = original_mappings
        for nuisance in pause_content.get("nuisance_replacements", []):
            replacement_id = str(nuisance["replacement_id"])
            location = replacement_location.get(replacement_id)
            if location is None:
                continue
            segment, replacement = location
            source_segment_start = int(segment["source_start_sample"])
            output_segment_start = int(segment["output_start_sample"])
            target_start = int(replacement["target_start_sample"])
            target_end = int(replacement["target_end_sample"])
            final_target_start = (
                output_segment_start + target_start - source_segment_start
            )
            nuisance["output_target_start_sample"] = final_target_start
            nuisance["output_target_end_sample"] = (
                final_target_start + target_end - target_start
            )
            nuisance["source_to_output_sample_mapping"] = [
                {
                    **dict(contribution),
                    "output_start_sample": final_target_start
                    + int(contribution["output_start_sample"]),
                    "output_end_sample": final_target_start
                    + int(contribution["output_end_sample"]),
                }
                for contribution in replacement.get("source_trace", [])
            ]
            nuisance["crossfades"] = [
                {
                    **dict(crossfade),
                    "output_start_sample": final_target_start
                    + int(crossfade["output_start_sample"]),
                    "output_end_sample": final_target_start
                    + int(crossfade["output_end_sample"]),
                }
                for crossfade in replacement.get("ambience_assembly", {}).get(
                    "crossfades", []
                )
            ]


def build_final_boundary_plan(
    *,
    audio_path: Path,
    semantic_plan: dict[str, Any],
    semantic_plan_path: Path,
    pause_plan: dict[str, Any],
    pause_plan_path: Path,
    output_dir: Path,
    alignment_python: Path,
    alignment_backend: str = "mfa",
    mfa_prefix: Path = DEFAULT_MFA_PREFIX,
    mfa_cache_root: Path = DEFAULT_MFA_CACHE_ROOT,
    mfa_micromamba: str | Path = "micromamba",
    mfa_num_jobs: int = 1,
    alignment_payload: dict[str, Any] | None = None,
    mfa_payload: dict[str, Any] | None = None,
    breath_cleanup: str = "off",
    breath_threshold: float = DEFAULT_BREATH_THRESHOLD,
    breath_min_duration_ms: int = DEFAULT_BREATH_MIN_DURATION_MS,
    respiro_cache_root: Path = DEFAULT_RESPIRO_CACHE_ROOT,
    breath_payload: dict[str, Any] | None = None,
    pause_policy: str = "semantic",
) -> dict[str, Any]:
    """Resolve every source boundary before rendering any output waveform."""

    if alignment_backend != "mfa":
        raise FinalRenderError("production cut coordinates require MFA")
    if breath_cleanup not in BREATH_CLEANUP_MODES:
        raise ValueError(f"unsupported breath cleanup mode: {breath_cleanup}")
    if pause_policy not in PAUSE_POLICIES:
        raise ValueError(f"unsupported pause policy: {pause_policy}")
    if not 0.0 <= breath_threshold <= 1.0:
        raise ValueError("breath threshold must be inside [0, 1]")
    if breath_min_duration_ms <= 0:
        raise ValueError("breath minimum duration must be positive")
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
        include_breath_gaps=breath_cleanup == "replace",
    )
    jobs = _prepare_alignment_jobs(
        specs=specs,
        words=words,
        ranges=ranges,
        source_audio=source_audio,
        sample_rate=sample_rate,
        output_dir=output_dir,
    )
    completeness_jobs = [
        job for job in jobs if bool(job.get("completeness_required", True))
    ]
    jobs_path = output_dir / "completeness_jobs.json"
    write_json(
        jobs_path,
        {
            "schema_version": 1,
            "source_audio": str(audio_path),
            "source_audio_sha256": sha256_file(audio_path),
            "language": ALIGNMENT_LANGUAGE,
            "device": "cpu",
            "purpose": "retained_word_completeness_veto",
            "coordinate_authority": False,
            "jobs": completeness_jobs,
        },
    )
    worker_path = output_dir / "completeness_worker_result.json"
    if alignment_payload is not None:
        worker_result = alignment_payload
        write_json(worker_path, worker_result)
    elif completeness_jobs:
        worker_result = _run_completeness_worker(
            jobs_path=jobs_path,
            result_path=worker_path,
            alignment_python=alignment_python,
            log_path=output_dir / "completeness_worker.log",
        )
    else:
        worker_result = {
            "schema_version": 1,
            "backend": "whisperx_alignment",
            "language": ALIGNMENT_LANGUAGE,
            "device": "cpu",
            "purpose": "retained_word_completeness_veto",
            "coordinate_authority": False,
            "jobs": [],
            "model_load_skipped": "no boundary contexts need completeness evidence",
        }
        write_json(worker_path, worker_result)
    completeness_contexts = _collect_completeness_evidence(
        jobs=completeness_jobs,
        worker_result=worker_result,
        sample_rate=sample_rate,
        total_samples=total_samples,
    )
    completeness_by_key = {
        str(context["context_id"]): context for context in completeness_contexts
    }
    completeness_support = _completeness_support_by_context(completeness_contexts)
    completeness_blocks_mfa = any(
        item["error"] is not None
        or item["support"] is None
        or item["support"]["status"] != "supported_complete_word"
        for item in completeness_support.values()
    )
    mfa_alignment_path = (
        output_dir / "mfa_alignment" / "metadata" / "mfa_alignment.json"
    )
    mfa_result: dict[str, Any] | None = None
    mfa_context_by_key: dict[str, dict[str, Any]] = {}
    mfa_context_errors: dict[str, dict[str, Any]] = {}
    mfa_global_error: str | None = None
    if completeness_blocks_mfa:
        mfa_global_error = "mfa_not_run_due_to_weak_retained_word"
    else:
        try:
            if mfa_payload is not None:
                mfa_result = mfa_payload
                mfa_alignment_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(mfa_alignment_path, mfa_result)
            else:
                mfa_result = align_mfa_contexts(
                    audio_path=audio_path,
                    contexts=_mfa_context_requests(jobs),
                    work_dir=output_dir,
                    prefix=mfa_prefix,
                    cache_root=mfa_cache_root,
                    micromamba=mfa_micromamba,
                    num_jobs=mfa_num_jobs,
                )
            mfa_context_by_key, mfa_context_errors = _validated_mfa_contexts(
                payload=mfa_result,
                jobs=jobs,
                source_audio_sha256=sha256_file(audio_path),
                sample_rate=sample_rate,
            )
        except (
            FinalRenderError,
            MFAAlignmentError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            mfa_global_error = f"{type(error).__name__}: {error}"

    def mfa_error_for(context_key: str) -> str | None:
        context_error = mfa_context_errors.get(context_key)
        if context_error is not None:
            return f"{context_error['code']}: {context_error['error']}"
        return mfa_global_error

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
                context_key = f"source_gap_{range_index - 1:04d}_right"
                retained_role = "first_retained_right"
                omitted_role = "last_omitted"
            completeness = completeness_support.get(
                context_key,
                {"support": None, "error": "completeness context is missing"},
            )
            start_boundary = _resolve_mfa_cut(
                boundary_id=f"range_{range_index:04d}_start",
                boundary_kind="omitted_to_selected",
                spec=completeness_by_key[context_key]["_spec"],
                mfa_context=mfa_context_by_key.get(context_key),
                mfa_error=mfa_error_for(context_key),
                retained_support=completeness["support"],
                completeness_error=completeness["error"],
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
            context_key = "eof_tail"
            completeness = completeness_support.get(
                context_key,
                {"support": None, "error": "completeness context is missing"},
            )
            end_boundary = _resolve_mfa_eof_boundary(
                spec=completeness_by_key[context_key]["_spec"],
                mfa_context=mfa_context_by_key.get(context_key),
                mfa_error=mfa_error_for(context_key),
                retained_support=completeness["support"],
                completeness_error=completeness["error"],
                words=words,
                mono=mono,
                sample_rate=sample_rate,
                total_samples=total_samples,
            )
        else:
            if range_index < len(ranges) - 1:
                context_key = f"source_gap_{range_index:04d}_left"
            else:
                context_key = "trailing_source_cut"
            completeness = completeness_support.get(
                context_key,
                {"support": None, "error": "completeness context is missing"},
            )
            end_boundary = _resolve_mfa_cut(
                boundary_id=f"range_{range_index:04d}_end",
                boundary_kind="selected_to_omitted",
                spec=completeness_by_key[context_key]["_spec"],
                mfa_context=mfa_context_by_key.get(context_key),
                mfa_error=mfa_error_for(context_key),
                retained_support=completeness["support"],
                completeness_error=completeness["error"],
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
        target_samples = (
            0
            if pause_policy == "cuts"
            else round(PAUSE_TARGETS_MS[pause_type] * sample_rate / 1000.0)
        )
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
            safety_status = "unsafe_source_boundary"
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
        completeness_context = completeness_by_key.get(context_key)
        if completeness_context is None:
            continue
        spec = completeness_context["_spec"]
        mfa_context = mfa_context_by_key.get(context_key)
        pause_type = str(transition["pause_type"])
        target_samples = (
            0
            if pause_policy == "cuts"
            else round(PAUSE_TARGETS_MS[pause_type] * sample_rate / 1000.0)
        )
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
        if mfa_context is None:
            joins.append(
                {
                    **common,
                    "estimated_existing_pause_samples": 0,
                    "estimated_existing_pause_ms": 0.0,
                    "inserted_pause_samples": 0,
                    "inserted_pause_ms": 0.0,
                    "safety_status": "pause_not_inserted_no_safe_point",
                    "insertion_method": "none",
                    "error": (
                        mfa_error_for(context_key) or "MFA pause context is missing"
                    ),
                }
            )
            continue
        try:
            protected = _mfa_protected_intervals(
                spec=spec,
                context=mfa_context,
                sample_rate=sample_rate,
                total_samples=total_samples,
            )
            roles = _protected_by_role(protected)
            gap_start = int(roles["previous_retained"]["end_sample"])
            gap_end = int(roles["next_retained"]["start_sample"])
            evidence = _mfa_non_speech_interval(
                context=mfa_context,
                start_sample=gap_start,
                end_sample=gap_end,
                sample_rate=sample_rate,
            )
            existing_samples = max(0, gap_end - gap_start)
            minimum_gap_samples = round(
                MINIMUM_VERIFIED_QUIET_MS * sample_rate / 1000.0
            )
            if evidence is None or existing_samples < minimum_gap_samples:
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
                        "error": "no MFA-confirmed natural inter-word gap",
                    }
                )
                continue
            inserted_samples = max(0, target_samples - existing_samples)
            insertion_sample = _snap_zero_crossing_in_mfa_gap(
                mono,
                candidate=(gap_start + gap_end) // 2,
                interval_start=gap_start,
                interval_end=gap_end,
                sample_rate=sample_rate,
            )
            fade_samples = round(QUIET_FADE_MS * sample_rate / 1000.0)
            fade_intervals = []
            if inserted_samples:
                fade_intervals = [
                    {
                        "source_start_sample": max(
                            gap_start,
                            insertion_sample - fade_samples,
                        ),
                        "source_end_sample": insertion_sample,
                        "curve": "fade_out",
                        "verified_quiet": True,
                    },
                    {
                        "source_start_sample": insertion_sample,
                        "source_end_sample": min(
                            gap_end,
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
                    "verified_quiet_interval": evidence,
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
                        "mfa_confirmed_interword_gap"
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
    protected_speech_mask: list[dict[str, Any]] = []
    editable_non_speech: list[dict[str, Any]] = []
    mfa_non_silence_source_spans: list[dict[str, Any]] = []
    room_tone_mfa_source_spans: list[dict[str, Any]] = []
    breath_replacements: list[dict[str, Any]] = []
    artifact_replacements: list[dict[str, Any]] = []
    artifact_cleanup_plan: dict[str, Any] = {
        "status": "not_run_unsafe_boundary_plan",
        "events": [],
        "replacements": [],
    }
    breath_cleanup_plan: dict[str, Any] = {
        "mode": breath_cleanup,
        "backend": "respiro-en" if breath_cleanup == "replace" else None,
        "upstream_commit": RESPIRO_UPSTREAM_COMMIT,
        "checkpoint_sha256": RESPIRO_CHECKPOINT_SHA256,
        "threshold": breath_threshold,
        "minimum_duration_ms": breath_min_duration_ms,
        "event_guard_ms": BREATH_EVENT_GUARD_MS,
        "transition_ms": BREATH_TRANSITION_MS,
        "status": (
            "disabled" if breath_cleanup == "off" else "not_run_unsafe_boundary_plan"
        ),
        "detected_events": [],
        "events": [],
        "replacements": [],
        "room_tone_exclusions": [],
        "room_tone_candidate_rejections": [],
        "room_tone_allocations": [],
        "error": None,
    }
    clean_ambience_bank: dict[str, Any] = {
        "schema_version": AMBIENCE_BANK_SCHEMA_VERSION,
        "status": "not_built_unsafe_boundary_plan",
        "sample_rate": sample_rate,
        "accepted_candidates": [],
        "rejected_candidates": [],
        "candidates": [],
    }
    if plan_status == "safe":
        protected_speech_mask = _retained_mfa_phone_mask(
            mfa_contexts=list(mfa_context_by_key.values()),
            source_intervals=source_intervals,
            selected_word_ids=_selected_word_ids(ranges),
            sample_rate=sample_rate,
            total_samples=total_samples,
        )
        _suppress_fades_over_protected_speech(
            boundaries=boundaries,
            joins=joins,
            protected_speech_mask=protected_speech_mask,
        )
        gap_evidence = _mfa_gap_evidence(
            jobs=jobs,
            mfa_context_by_key=mfa_context_by_key,
            source_intervals=source_intervals,
            sample_rate=sample_rate,
        )
        editable_non_speech = _editable_non_speech_intervals(
            gap_evidence=gap_evidence,
            source_intervals=source_intervals,
            boundaries=boundaries,
            joins=joins,
            protected_speech_mask=protected_speech_mask,
        )
        mfa_non_silence_source_spans = _collect_mfa_non_silence_source_spans(
            mfa_contexts=list(mfa_context_by_key.values()),
            sample_rate=sample_rate,
            total_samples=total_samples,
        )
        room_tone_mfa_source_spans = _mfa_silence_source_spans(
            mfa_contexts=list(mfa_context_by_key.values()),
            non_silence_spans=mfa_non_silence_source_spans,
            sample_rate=sample_rate,
            total_samples=total_samples,
        )
        verified_ambience_spans = _merge_verified_ambience_spans(
            [*editable_non_speech, *room_tone_mfa_source_spans],
            total_samples=total_samples,
        )
        baseline_exclusions = _baseline_room_tone_exclusions(
            protected_speech_mask=protected_speech_mask,
            mfa_non_silence_spans=mfa_non_silence_source_spans,
            boundaries=boundaries,
            joins=joins,
            words=words,
            sample_rate=sample_rate,
            total_samples=total_samples,
        )
        breath_evidence: dict[str, Any] | None = None
        detector_error: str | None = None
        detector_status = "disabled_by_user"
        if breath_cleanup == "replace":
            relevant_ranges = [
                (int(item["start_sample"]), int(item["end_sample"]))
                for item in verified_ambience_spans
            ]
            relevant_ranges.extend(
                (selected, min(total_samples, selected + 1))
                for boundary in boundaries
                if type(boundary.get("selected_source_sample")) is int
                for selected in [int(boundary["selected_source_sample"])]
                if selected < total_samples
            )
            try:
                raw_breath_evidence = (
                    breath_payload
                    if breath_payload is not None
                    else analyze_breath_evidence(
                        source_audio=source_audio,
                        source_sample_rate=sample_rate,
                        relevant_ranges=relevant_ranges,
                        threshold=breath_threshold,
                        minimum_duration_ms=breath_min_duration_ms,
                        cache_root=respiro_cache_root,
                    )
                )
                breath_evidence = _validate_breath_payload(
                    raw_breath_evidence,
                    threshold=breath_threshold,
                    minimum_duration_ms=breath_min_duration_ms,
                    total_samples=total_samples,
                )
                detector_status = str(breath_evidence["status"])
            except Exception as error:
                warning = (
                    "Respiro-en breath cleanup was skipped; preserving the "
                    f"otherwise valid MFA edit: {type(error).__name__}: {error}"
                )
                warnings.warn(warning, RuntimeWarning, stacklevel=2)
                breath_cleanup_plan["status"] = (
                    "breath_cleanup_skipped_detector_failure"
                )
                breath_cleanup_plan["error"] = warning
                detector_status = "breath_cleanup_skipped_detector_failure"
                detector_error = warning

        raw_detected_events = (
            list(breath_evidence["events"]) if breath_evidence is not None else []
        )
        detected_events = [
            {
                **dict(event),
                "local_noise_floor_db": _local_source_noise_floor_db(
                    source_audio,
                    sample_rate=sample_rate,
                    reference_sample=(
                        int(event["start_sample"]) + int(event["end_sample"])
                    )
                    // 2,
                ),
            }
            for event in raw_detected_events
        ]
        detector_exclusions = breath_room_tone_exclusions(
            detected_events,
            sample_rate=sample_rate,
            total_samples=total_samples,
        )
        _initialize_pause_content(joins)
        artifact_events = _internal_gap_nuisance_evidence(
            joins=joins,
            source_audio=source_audio,
            sample_rate=sample_rate,
            detected_events=detected_events,
        )
        artifact_exclusions = [
            {
                "start_sample": int(event["start_sample"]),
                "end_sample": int(event["end_sample"]),
                "reason": "deterministic_nonstationary_artifact",
                "join_id": str(event["join_id"]),
            }
            for event in artifact_events
        ]
        room_tone_exclusions = [
            *baseline_exclusions,
            *detector_exclusions,
            *artifact_exclusions,
        ]
        clean_ambience_bank = _build_verified_ambience_bank(
            source_audio=source_audio,
            sample_rate=sample_rate,
            total_samples=total_samples,
            verified_spans=verified_ambience_spans,
            exclusions=room_tone_exclusions,
            detector_status=detector_status,
            detector_error=detector_error,
        )
        ambience_usage_counts: dict[str, int] = {}
        room_tone_allocations = _allocate_semantic_pause_room_tone(
            ambience_bank=clean_ambience_bank,
            joins=joins,
            source_intervals=source_intervals,
            source_audio=source_audio,
            candidate_usage_counts=ambience_usage_counts,
        )
        if artifact_events:
            artifact_replacements, artifact_records = plan_breath_replacements(
                events=artifact_events,
                editable_non_speech=editable_non_speech,
                protected_speech_mask=protected_speech_mask,
                ambience_bank=clean_ambience_bank,
                sample_rate=sample_rate,
                total_samples=total_samples,
                guard_ms=0.0,
                transition_ms=BREATH_TRANSITION_MS,
                candidate_usage_counts=ambience_usage_counts,
            )
            replacement_id_map: dict[str, str] = {}
            for replacement_index, replacement in enumerate(artifact_replacements):
                old_id = str(replacement["replacement_id"])
                new_id = f"artifact_replacement_{replacement_index:04d}"
                replacement_id_map[old_id] = new_id
                replacement["replacement_id"] = new_id
                replacement["event_type"] = "deterministic_nonstationary_artifact"
                replacement["status"] = "artifact_replaced_with_verified_clean_ambience"
            for record in artifact_records:
                record["event_type"] = "deterministic_nonstationary_artifact"
                record["replacements"] = [
                    replacement_id_map.get(str(value), str(value))
                    for value in record.get("replacements", [])
                ]
                if record["replacements"]:
                    record["status"] = "artifact_replaced_with_verified_clean_ambience"
            artifact_cleanup_plan = {
                "status": (
                    "complete"
                    if artifact_records
                    and all(record.get("replacements") for record in artifact_records)
                    else "clean_ambience_unavailable"
                ),
                "events": artifact_records,
                "replacements": artifact_replacements,
            }
        else:
            artifact_cleanup_plan = {
                "status": "no_artifacts_detected",
                "events": [],
                "replacements": [],
            }
        if breath_cleanup == "replace" and breath_evidence is not None:
            breath_fragments, covered_event_records = (
                _subtract_artifact_replacements_from_breath_events(
                    events=detected_events,
                    artifact_replacements=artifact_replacements,
                )
            )
            breath_replacements, event_records = plan_breath_replacements(
                events=breath_fragments,
                editable_non_speech=editable_non_speech,
                protected_speech_mask=protected_speech_mask,
                ambience_bank=clean_ambience_bank,
                sample_rate=sample_rate,
                total_samples=total_samples,
                candidate_usage_counts=ambience_usage_counts,
            )
            breath_cleanup_plan["events"] = _aggregate_breath_fragment_records(
                source_events=detected_events,
                fragment_records=event_records,
                covered_records=covered_event_records,
            )
            breath_cleanup_plan["replacements"] = breath_replacements
            breath_cleanup_plan["status"] = "complete"
        all_nuisance_replacements = sorted(
            [*breath_replacements, *artifact_replacements],
            key=lambda item: (
                int(item["target_start_sample"]),
                int(item["target_end_sample"]),
            ),
        )
        for join in joins:
            pause_content = join.get("pause_content")
            quiet = join.get("verified_quiet_interval")
            if not isinstance(pause_content, dict) or not isinstance(quiet, dict):
                continue
            gap_start = int(quiet["start_sample"])
            gap_end = int(quiet["end_sample"])
            replacements = [
                replacement
                for replacement in all_nuisance_replacements
                if _intervals_overlap(
                    gap_start,
                    gap_end,
                    int(replacement["target_start_sample"]),
                    int(replacement["target_end_sample"]),
                )
            ]
            pause_content["nuisance_replacements"] = [
                {
                    "replacement_id": replacement["replacement_id"],
                    "target_start_sample": replacement["target_start_sample"],
                    "target_end_sample": replacement["target_end_sample"],
                    "candidate_ids": list(replacement.get("candidate_ids", [])),
                    "source_trace": list(replacement.get("source_trace", [])),
                    "crossfades": list(replacement.get("equal_power_crossfades", [])),
                    "status": replacement["status"],
                }
                for replacement in replacements
            ]
            if replacements:
                covered = _subtract_sample_intervals(
                    [(gap_start, gap_end)],
                    [
                        (
                            int(replacement["target_start_sample"]),
                            int(replacement["target_end_sample"]),
                        )
                        for replacement in replacements
                    ],
                )
                pause_content["original_gap_content"] = (
                    "replaced_with_verified_clean_ambience"
                    if not covered
                    else "partially_replaced_with_verified_clean_ambience"
                )
                pause_content["original_content_preserved"] = bool(covered)
                pause_content["original_content_replaced"] = True
            elif pause_content["nuisance_mask_intersections"]:
                pause_content["status"] = "clean_ambience_unavailable"
        breath_cleanup_plan["detector_evidence"] = breath_evidence
        breath_cleanup_plan["detected_events"] = detected_events
        breath_cleanup_plan["room_tone_exclusions"] = room_tone_exclusions
        breath_cleanup_plan["room_tone_candidate_rejections"] = list(
            clean_ambience_bank["rejected_candidates"]
        )
        breath_cleanup_plan["room_tone_allocations"] = room_tone_allocations
        clean_ambience_bank["usage_policy"] = (
            "rank stationarity, local noise-floor match, and duration first; "
            "then prefer the least-used candidate; never repeat a candidate "
            "within one pause bed; allow fully traced reuse across distinct pauses"
        )
        clean_ambience_bank["usage_ledger"] = [
            {
                "candidate_id": str(candidate["candidate_id"]),
                "use_count": int(
                    ambience_usage_counts.get(str(candidate["candidate_id"]), 0)
                ),
                "reused_across_distinct_pauses": int(
                    ambience_usage_counts.get(str(candidate["candidate_id"]), 0)
                )
                > 1,
            }
            for candidate in clean_ambience_bank["accepted_candidates"]
        ]
        output_segments = _build_output_segments(
            source_intervals=source_intervals,
            boundaries=boundaries,
            joins=joins,
            breath_replacements=all_nuisance_replacements,
        )
        _plan_ambience_edge_transitions(
            segments=output_segments,
            joins=joins,
            source_audio=source_audio,
            sample_rate=sample_rate,
        )
        _finalize_pause_output_provenance(
            joins=joins,
            segments=output_segments,
        )

    expected_frames = (
        int(output_segments[-1]["output_end_sample"]) if output_segments else 0
    )
    final_boundary = boundaries[-1]
    semantic_fallbacks = (
        [
            json.loads(json.dumps(item))
            for item in semantic_plan.get("fallbacks", [])
            if isinstance(item, dict)
        ]
        if isinstance(semantic_plan.get("fallbacks"), list)
        else []
    )
    semantic_preserved_source_fallbacks = [
        json.loads(json.dumps(item))
        for item in semantic_fallbacks
        if isinstance(item.get("source_ranges"), list) and item["source_ranges"]
    ]
    pause_degraded_batches = (
        [
            json.loads(json.dumps(item))
            for item in pause_plan.get("degraded_batches", [])
            if isinstance(item, dict)
        ]
        if isinstance(pause_plan.get("degraded_batches"), list)
        else []
    )
    if (
        isinstance(semantic_plan.get("delivery_fallback"), dict)
        or semantic_preserved_source_fallbacks
    ):
        delivery_status = "complete_with_preserved_source_context"
    elif pause_degraded_batches:
        delivery_status = "complete_with_deterministic_pauses"
    else:
        delivery_status = "complete"
    return {
        "schema_version": 2,
        "planner": "authoritative_single_pass_boundary_plan_v2",
        "status": plan_status,
        "alignment_backend": "mfa",
        "mfa_version": MFA_VERSION,
        "mfa_model": MFA_MODEL_ID,
        "mfa_fine_tune": True,
        "source_audio": str(audio_path),
        "source_audio_sha256": sha256_file(audio_path),
        "source_sample_rate": sample_rate,
        "source_channel_count": channel_count,
        "source_frame_count": total_samples,
        "streaming_plan": str(semantic_plan_path),
        "streaming_plan_sha256": sha256_file(semantic_plan_path),
        "pause_plan": str(pause_plan_path),
        "pause_plan_sha256": sha256_file(pause_plan_path),
        "pause_policy": pause_policy,
        "delivery_status": delivery_status,
        "delivery_fallback": semantic_plan.get("delivery_fallback"),
        "semantic_planner_fallbacks": semantic_fallbacks,
        "semantic_preserved_source_fallbacks": (semantic_preserved_source_fallbacks),
        "pause_plan_degraded": bool(pause_degraded_batches),
        "pause_degraded_batches": pause_degraded_batches,
        "configuration": {
            "alignment_backend": "mfa",
            "mfa_version": MFA_VERSION,
            "mfa_model": MFA_MODEL_ID,
            "mfa_fine_tune": True,
            "mfa_prefix": str(mfa_prefix.resolve()),
            "mfa_cache_root": str(mfa_cache_root.resolve()),
            "mfa_num_jobs": mfa_num_jobs,
            "whisperx_purpose": "retained_word_completeness_veto",
            "whisperx_coordinate_authority": False,
            "protected_speech_margin_ms": PROTECTED_SPEECH_MARGIN_MS,
            "minimum_verified_quiet_ms": MINIMUM_VERIFIED_QUIET_MS,
            "quiet_fade_ms": QUIET_FADE_MS,
            "mfa_zero_crossing_snap_ms": MFA_ZERO_CROSSING_SNAP_MS,
            "ambience_crossfade_ms": DEFAULT_AMBIENCE_CROSSFADE_MS,
            "ambience_thresholds": dict(clean_ambience_bank.get("thresholds", {})),
            "pause_targets_ms": PAUSE_TARGETS_MS,
            "pause_policy": pause_policy,
            "breath_cleanup": breath_cleanup,
            "breath_threshold": breath_threshold,
            "breath_min_duration_ms": breath_min_duration_ms,
            "respiro_upstream_commit": RESPIRO_UPSTREAM_COMMIT,
            "respiro_checkpoint_sha256": RESPIRO_CHECKPOINT_SHA256,
            "respiro_cache_root": str(respiro_cache_root.resolve()),
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
        "completeness_jobs": str(jobs_path.resolve()),
        "completeness_worker_result": str(worker_path.resolve()),
        "completeness_contexts": [
            _public_alignment_context(context) for context in completeness_contexts
        ],
        "mfa_alignment": (
            str(mfa_alignment_path.resolve()) if mfa_alignment_path.is_file() else None
        ),
        "alignment_contexts": (
            list(mfa_result.get("contexts", [])) if mfa_result is not None else []
        ),
        "mfa_context_errors": list(mfa_context_errors.values()),
        "mfa_error": mfa_global_error,
        "source_intervals": source_intervals,
        "boundaries": boundaries,
        "joins": joins,
        "protected_speech_mask": protected_speech_mask,
        "editable_non_speech": editable_non_speech,
        "mfa_non_silence_source_spans": mfa_non_silence_source_spans,
        "room_tone_mfa_source_spans": room_tone_mfa_source_spans,
        "clean_ambience_bank": clean_ambience_bank,
        "breath_cleanup": breath_cleanup_plan,
        "artifact_cleanup": artifact_cleanup_plan,
        "output_segments": output_segments,
        "expected_output_frame_count": expected_frames,
        "alignment_context_count": len(jobs),
        "alignment_resolved_boundaries": sum(
            boundary["safety_status"] == "safe"
            and boundary["alignment_context_id"] is not None
            for boundary in boundaries
        ),
        "unsafe_dense_boundaries": sum(
            boundary["safety_status"] == "unsafe_dense_boundary"
            for boundary in boundaries
        ),
        "mfa_dense_phone_boundaries": sum(
            boundary["boundary_method"] == "mfa_dense_phone_boundary"
            for boundary in boundaries
        ),
        "weak_retained_word_alignments": sum(
            boundary["safety_status"] == "weak_retained_word_alignment"
            for boundary in boundaries
        ),
        "mfa_word_mapping_failures": sum(
            boundary["safety_status"] == "mfa_word_mapping_failed"
            for boundary in boundaries
        ),
        "alignment_failures": sum(
            boundary["safety_status"]
            in {
                "completeness_alignment_failed",
                "mfa_alignment_failed",
                "mfa_runtime_failed",
                "mfa_word_mapping_failed",
            }
            for boundary in boundaries
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
    if (
        boundary_plan.get("alignment_backend") != "mfa"
        or boundary_plan.get("mfa_version") != MFA_VERSION
        or boundary_plan.get("mfa_model") != MFA_MODEL_ID
        or boundary_plan.get("mfa_fine_tune") is not True
    ):
        raise FinalRenderError("boundary plan has no authoritative MFA provenance")
    boundaries = boundary_plan.get("boundaries")
    joins = boundary_plan.get("joins")
    segments = boundary_plan.get("output_segments")
    if not isinstance(boundaries, list) or not isinstance(joins, list):
        raise FinalRenderError("boundary plan has no boundary/join ledger")
    if not isinstance(segments, list) or not segments:
        raise FinalRenderError("safe boundary plan has no output trace")
    protected_mask = boundary_plan.get("protected_speech_mask", [])
    if not isinstance(protected_mask, list):
        raise FinalRenderError("boundary plan has a malformed protected-speech mask")
    ambience_bank = boundary_plan.get("clean_ambience_bank")
    ambience_required = any(
        segment.get("kind") == "ambience" or bool(segment.get("sample_replacements"))
        for segment in segments
    )
    if not isinstance(ambience_bank, dict):
        if ambience_required:
            raise FinalRenderError("boundary plan has no clean ambience bank")
        ambience_bank = {"accepted_candidates": []}
    accepted_candidates = ambience_bank.get("accepted_candidates", [])
    if not isinstance(accepted_candidates, list):
        raise FinalRenderError("clean ambience bank is malformed")
    accepted_by_id = {
        str(candidate.get("candidate_id")): candidate
        for candidate in accepted_candidates
        if candidate.get("accepted") is True
    }
    if len(accepted_by_id) != len(accepted_candidates):
        raise FinalRenderError("clean ambience bank candidate IDs are invalid")
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
    replacement_ids: set[str] = set()
    actual_ambience_usage_counts: dict[str, int] = {}
    room_tone_exclusions = boundary_plan.get("breath_cleanup", {}).get(
        "room_tone_exclusions",
        [],
    )
    for expected_index, segment in enumerate(segments):
        if segment.get("segment_index") != expected_index:
            raise FinalRenderError("output trace segment indices are not immutable")
        if segment.get("kind") not in {"source", "ambience"}:
            raise FinalRenderError("output trace contains an untraceable segment")
        output_start = int(segment["output_start_sample"])
        output_end = int(segment["output_end_sample"])
        if output_start != cursor or output_end <= output_start:
            raise FinalRenderError("output trace geometry is inconsistent")
        if segment.get("kind") == "ambience":
            trace = segment.get("source_trace")
            if not isinstance(trace, list) or not trace:
                raise FinalRenderError("ambience segment has no source trace")
            if bool(segment.get("source_reuse")):
                raise FinalRenderError("ambience segment repeats canonical source")
            previous_trace_end = output_start
            local_candidate_ids: set[str] = set()
            for trace_index, contribution in enumerate(trace):
                candidate_id = str(contribution.get("candidate_id", ""))
                if candidate_id not in accepted_by_id:
                    raise FinalRenderError(
                        "ambience segment references a rejected bank candidate"
                    )
                if candidate_id in local_candidate_ids:
                    raise FinalRenderError(
                        "ambience segment tiles one candidate more than once"
                    )
                local_candidate_ids.add(candidate_id)
                actual_ambience_usage_counts[candidate_id] = (
                    actual_ambience_usage_counts.get(candidate_id, 0) + 1
                )
                source_start = int(contribution["source_start_sample"])
                source_end = int(contribution["source_end_sample"])
                accepted_candidate = accepted_by_id[candidate_id]
                if not (
                    int(accepted_candidate["source_start_sample"])
                    <= source_start
                    < source_end
                    <= int(accepted_candidate["source_end_sample"])
                ):
                    raise FinalRenderError(
                        "ambience trace leaves its accepted bank candidate"
                    )
                trace_start = int(contribution["output_start_sample"])
                trace_end = int(contribution["output_end_sample"])
                if not (
                    0
                    <= source_start
                    < source_end
                    <= int(boundary_plan["source_frame_count"])
                    and output_start <= trace_start < trace_end <= output_end
                    and source_end - source_start == trace_end - trace_start
                ):
                    raise FinalRenderError("ambience source trace geometry is invalid")
                if trace_index == 0 and trace_start != output_start:
                    raise FinalRenderError(
                        "ambience trace does not start at its segment"
                    )
                if trace_start > previous_trace_end:
                    raise FinalRenderError("ambience trace contains an unplanned hole")
                previous_trace_end = trace_end
                for exclusion in room_tone_exclusions:
                    if _intervals_overlap(
                        source_start,
                        source_end,
                        int(exclusion["start_sample"]),
                        int(exclusion["end_sample"]),
                    ):
                        raise FinalRenderError(
                            "semantic pause uses forbidden ambience source"
                        )
                for protected in protected_mask:
                    if _intervals_overlap(
                        source_start,
                        source_end,
                        int(protected["start_sample"]),
                        int(protected["end_sample"]),
                    ):
                        raise FinalRenderError(
                            "semantic pause ambience overlaps retained MFA speech"
                        )
            if previous_trace_end != output_end:
                raise FinalRenderError("ambience trace does not fill its segment")
            crossfades = segment.get("equal_power_crossfades", [])
            if len(crossfades) != len(trace) - 1:
                raise FinalRenderError("ambience crossfade ledger is incomplete")
            for left, right, crossfade in zip(trace, trace[1:], crossfades):
                if (
                    crossfade.get("curve") != "equal_power"
                    or int(crossfade["output_start_sample"])
                    != int(right["output_start_sample"])
                    or int(crossfade["output_end_sample"])
                    != int(left["output_end_sample"])
                ):
                    raise FinalRenderError("ambience crossfade geometry is invalid")
            edge_envelopes = segment.get("edge_gain_envelopes", [])
            if not isinstance(edge_envelopes, list) or len(edge_envelopes) > 2:
                raise FinalRenderError("ambience edge fade ledger is malformed")
            edge_sides: set[str] = set()
            for envelope in edge_envelopes:
                side = str(envelope.get("side"))
                curve = str(envelope.get("curve"))
                edge_start = int(envelope["output_start_sample"])
                edge_end = int(envelope["output_end_sample"])
                if (
                    side in edge_sides
                    or side not in {"left", "right"}
                    or curve != f"equal_power_fade_{'in' if side == 'left' else 'out'}"
                    or envelope.get("verified_ambience") is not True
                    or not output_start <= edge_start < edge_end <= output_end
                    or (side == "left" and edge_start != output_start)
                    or (side == "right" and edge_end != output_end)
                ):
                    raise FinalRenderError("ambience edge fade geometry is invalid")
                edge_sides.add(side)
            outer_transitions = segment.get("outer_transitions", [])
            if not isinstance(outer_transitions, list) or len(outer_transitions) != 2:
                raise FinalRenderError("ambience outer-transition ledger is incomplete")
            transition_by_side = {
                str(record.get("side")): record for record in outer_transitions
            }
            if set(transition_by_side) != {"left", "right"}:
                raise FinalRenderError("ambience outer-transition sides are invalid")
            envelope_by_side = {
                str(envelope["side"]): envelope for envelope in edge_envelopes
            }
            for side, record in transition_by_side.items():
                applied = record.get("status") == "ambience_edge_taper_applied"
                transition = record.get("ambience_transition_interval")
                envelope = envelope_by_side.get(side)
                if applied != (envelope is not None) or (
                    applied
                    and (
                        not isinstance(transition, dict)
                        or int(transition["output_start_sample"])
                        != int(envelope["output_start_sample"])
                        or int(transition["output_end_sample"])
                        != int(envelope["output_end_sample"])
                        or str(transition["curve"]) != str(envelope["curve"])
                    )
                ):
                    raise FinalRenderError(
                        "ambience outer-transition decision is inconsistent"
                    )
                if float(record["planned_maximum_sample_discontinuity"]) < 0.0:
                    raise FinalRenderError(
                        "ambience seam discontinuity cannot be negative"
                    )
            cursor = output_end
            continue

        source_start = int(segment["source_start_sample"])
        source_end = int(segment["source_end_sample"])
        if source_end - source_start != output_end - output_start:
            raise FinalRenderError("source output trace geometry is inconsistent")
        gain_envelopes = segment.get("gain_envelopes", [])
        for envelope in gain_envelopes:
            for protected in protected_mask:
                if _intervals_overlap(
                    int(envelope["source_start_sample"]),
                    int(envelope["source_end_sample"]),
                    int(protected["start_sample"]),
                    int(protected["end_sample"]),
                ):
                    raise FinalRenderError(
                        "source fade overlaps the retained MFA phone mask"
                    )
        previous_replacement_end = source_start
        for replacement in segment.get("sample_replacements", []):
            replacement_id = str(replacement.get("replacement_id", ""))
            if not replacement_id or replacement_id in replacement_ids:
                raise FinalRenderError("breath replacement IDs are empty or duplicated")
            replacement_ids.add(replacement_id)
            target_start = int(replacement["target_start_sample"])
            target_end = int(replacement["target_end_sample"])
            if not (
                source_start
                <= previous_replacement_end
                <= target_start
                < target_end
                <= source_end
            ):
                raise FinalRenderError("breath replacement target leaves its segment")
            previous_replacement_end = target_end
            replacement_ranges = replacement.get("source_trace")
            assembly = replacement.get("ambience_assembly")
            if (
                not isinstance(replacement_ranges, list)
                or not replacement_ranges
                or not isinstance(assembly, dict)
                or assembly.get("status") != "complete"
            ):
                raise FinalRenderError("breath replacement has no ambience trace")
            replacement_length = int(assembly["planned_output_samples"])
            if replacement_length != target_end - target_start:
                raise FinalRenderError("breath replacement changes output duration")
            local_candidate_ids: set[str] = set()
            for source_range in replacement_ranges:
                candidate_id = str(source_range.get("candidate_id", ""))
                if candidate_id not in accepted_by_id:
                    raise FinalRenderError(
                        "breath replacement references a rejected ambience candidate"
                    )
                if candidate_id in local_candidate_ids:
                    raise FinalRenderError(
                        "nuisance replacement tiles one candidate more than once"
                    )
                local_candidate_ids.add(candidate_id)
                actual_ambience_usage_counts[candidate_id] = (
                    actual_ambience_usage_counts.get(candidate_id, 0) + 1
                )
                replacement_start = int(source_range["source_start_sample"])
                replacement_end = int(source_range["source_end_sample"])
                accepted_candidate = accepted_by_id[candidate_id]
                if not (
                    int(accepted_candidate["source_start_sample"])
                    <= replacement_start
                    < replacement_end
                    <= int(accepted_candidate["source_end_sample"])
                ):
                    raise FinalRenderError(
                        "nuisance replacement leaves its accepted bank candidate"
                    )
                if not (
                    0
                    <= replacement_start
                    < replacement_end
                    <= int(boundary_plan["source_frame_count"])
                ):
                    raise FinalRenderError(
                        "breath room-tone source has invalid canonical geometry"
                    )
                for exclusion in room_tone_exclusions:
                    if _intervals_overlap(
                        replacement_start,
                        replacement_end,
                        int(exclusion["start_sample"]),
                        int(exclusion["end_sample"]),
                    ):
                        raise FinalRenderError(
                            "breath replacement uses forbidden room-tone source"
                        )
                for protected in protected_mask:
                    if _intervals_overlap(
                        replacement_start,
                        replacement_end,
                        int(protected["start_sample"]),
                        int(protected["end_sample"]),
                    ):
                        raise FinalRenderError(
                            "breath replacement source overlaps retained MFA speech"
                        )
            replacement_crossfades = replacement.get("equal_power_crossfades", [])
            if len(replacement_crossfades) != len(replacement_ranges) - 1:
                raise FinalRenderError(
                    "breath replacement crossfade ledger is incomplete"
                )
            for transition in replacement_crossfades:
                if transition.get("curve") != "equal_power":
                    raise FinalRenderError(
                        "breath replacement ambience crossfade is not equal-power"
                    )
            for protected in protected_mask:
                if _intervals_overlap(
                    target_start,
                    target_end,
                    int(protected["start_sample"]),
                    int(protected["end_sample"]),
                ):
                    raise FinalRenderError(
                        "breath replacement overlaps a retained MFA phone"
                    )
            for transition in replacement.get("transition_ranges", []):
                transition_start = int(transition["target_start_sample"])
                transition_end = int(transition["target_end_sample"])
                if not target_start <= transition_start < transition_end <= target_end:
                    raise FinalRenderError("breath transition leaves its target")
                for protected in protected_mask:
                    if _intervals_overlap(
                        transition_start,
                        transition_end,
                        int(protected["start_sample"]),
                        int(protected["end_sample"]),
                    ):
                        raise FinalRenderError(
                            "breath transition overlaps a retained MFA phone"
                        )
        cursor = output_end
    if cursor != int(boundary_plan["expected_output_frame_count"]):
        raise FinalRenderError("output trace does not match planned frame count")
    usage_ledger = ambience_bank.get("usage_ledger", [])
    if usage_ledger:
        expected_usage = {
            str(item["candidate_id"]): int(item["use_count"]) for item in usage_ledger
        }
        if expected_usage != {
            candidate_id: actual_ambience_usage_counts.get(candidate_id, 0)
            for candidate_id in expected_usage
        }:
            raise FinalRenderError("clean ambience usage ledger is inconsistent")


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


def _render_ambience_source_trace(
    *,
    canonical_source: np.ndarray,
    source_trace: Sequence[dict[str, Any]],
    expected_samples: int,
    output_coordinate_base: int = 0,
) -> np.ndarray:
    """Assemble distinct clean candidates with ambience-only equal-power joins."""

    if expected_samples <= 0 or not source_trace:
        raise FinalRenderError("ambience trace has no output capacity")
    first = source_trace[0]
    first_start = int(first["source_start_sample"])
    first_end = int(first["source_end_sample"])
    rendered = np.array(
        canonical_source[first_start:first_end],
        dtype=np.float32,
        copy=True,
    )
    current_output_end = int(first["output_end_sample"]) - output_coordinate_base
    if int(first["output_start_sample"]) != output_coordinate_base:
        raise FinalRenderError("ambience trace starts at the wrong output coordinate")
    for item in source_trace[1:]:
        source_start = int(item["source_start_sample"])
        source_end = int(item["source_end_sample"])
        output_start = int(item["output_start_sample"]) - output_coordinate_base
        output_end = int(item["output_end_sample"]) - output_coordinate_base
        overlap = current_output_end - output_start
        chunk = np.array(
            canonical_source[source_start:source_end],
            dtype=np.float32,
            copy=True,
        )
        if not 0 < overlap <= min(len(rendered), len(chunk)):
            raise FinalRenderError("ambience trace has invalid overlap geometry")
        theta = np.linspace(
            0.0,
            np.pi / 2.0,
            overlap,
            endpoint=True,
            dtype=np.float32,
        )[:, None]
        blended = rendered[-overlap:] * np.cos(theta) + chunk[:overlap] * np.sin(theta)
        rendered = np.concatenate(
            [rendered[:-overlap], blended, chunk[overlap:]],
            axis=0,
        )
        current_output_end = output_end
    if len(rendered) != expected_samples or current_output_end != expected_samples:
        raise FinalRenderError("ambience trace rendered the wrong duration")
    return rendered


def _apply_ambience_edge_gain_envelopes(
    rendered: np.ndarray,
    *,
    output_start_sample: int,
    envelopes: Sequence[dict[str, Any]],
) -> np.ndarray:
    output = np.array(rendered, dtype=np.float32, copy=True)
    for envelope in envelopes:
        start = int(envelope["output_start_sample"]) - output_start_sample
        end = int(envelope["output_end_sample"]) - output_start_sample
        if not 0 <= start < end <= len(output):
            raise FinalRenderError("ambience edge fade leaves its output segment")
        theta = np.linspace(
            0.0,
            np.pi / 2.0,
            end - start,
            endpoint=True,
            dtype=np.float32,
        )[:, None]
        if envelope["curve"] == "equal_power_fade_in":
            gain = np.sin(theta)
        elif envelope["curve"] == "equal_power_fade_out":
            gain = np.cos(theta)
        else:
            raise FinalRenderError("ambience edge fade has an unknown curve")
        output[start:end] *= gain
    return output


def _apply_breath_replacements(
    rendered: np.ndarray,
    *,
    canonical_chunk: np.ndarray,
    canonical_source: np.ndarray,
    source_start: int,
    replacements: Sequence[dict[str, Any]],
) -> np.ndarray:
    output = np.array(rendered, dtype=np.float32, copy=True)
    for replacement in replacements:
        target_start = int(replacement["target_start_sample"])
        target_end = int(replacement["target_end_sample"])
        local_start = target_start - source_start
        local_end = target_end - source_start
        if not 0 <= local_start < local_end <= len(output):
            raise FinalRenderError("breath replacement leaves its source chunk")
        room_tone = _render_ambience_source_trace(
            canonical_source=canonical_source,
            source_trace=replacement["source_trace"],
            expected_samples=local_end - local_start,
        )
        original = canonical_chunk[local_start:local_end]
        replacement_audio = np.array(room_tone, dtype=np.float32, copy=True)
        transition = int(replacement.get("transition_samples", 0))
        if transition:
            if transition * 2 > len(replacement_audio):
                raise FinalRenderError("breath transition is longer than its target")
            theta = np.linspace(
                0.0,
                np.pi / 2.0,
                transition,
                endpoint=True,
                dtype=np.float32,
            )[:, None]
            replacement_audio[:transition] = original[:transition] * np.cos(
                theta
            ) + room_tone[:transition] * np.sin(theta)
            replacement_audio[-transition:] = room_tone[-transition:] * np.cos(
                theta
            ) + original[-transition:] * np.sin(theta)
        output[local_start:local_end] = replacement_audio
    return output


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
    if boundary_plan.get("planner") != "authoritative_single_pass_boundary_plan_v2":
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
        if segment["kind"] == "ambience":
            rendered = _render_ambience_source_trace(
                canonical_source=source_audio,
                source_trace=segment["source_trace"],
                expected_samples=(
                    int(segment["output_end_sample"])
                    - int(segment["output_start_sample"])
                ),
                output_coordinate_base=int(segment["output_start_sample"]),
            )
            rendered = _apply_ambience_edge_gain_envelopes(
                rendered,
                output_start_sample=int(segment["output_start_sample"]),
                envelopes=segment.get("edge_gain_envelopes", []),
            )
            parts.append(rendered)
            continue
        source_start = int(segment["source_start_sample"])
        source_end = int(segment["source_end_sample"])
        if not 0 <= source_start < source_end <= len(source_audio):
            raise FinalRenderError("output trace references invalid source samples")
        chunk = source_audio[source_start:source_end]
        if segment["kind"] == "source":
            rendered = _apply_breath_replacements(
                np.asarray(chunk, dtype=np.float32),
                canonical_chunk=chunk,
                canonical_source=source_audio,
                source_start=source_start,
                replacements=segment.get("sample_replacements", []),
            )
            # Source-edge gain is applied last and only inside MFA-confirmed
            # non-speech.  If a nuisance replacement reaches the same quiet
            # edge, the authoritative edge taper still meets inserted
            # ambience at zero without touching retained phones.
            rendered = _apply_source_gain_envelopes(
                rendered,
                source_start=source_start,
                envelopes=segment.get("gain_envelopes", []),
            )
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
    protected_retained.update(
        (
            -1,
            int(interval["start_sample"]),
            int(interval["end_sample"]),
        )
        for interval in boundary_plan.get("protected_speech_mask", [])
    )
    for _, protected_start, protected_end in protected_retained:
        mapped = False
        for segment in boundary_plan["output_segments"]:
            if segment["kind"] != "source":
                continue
            source_start = int(segment["source_start_sample"])
            source_end = int(segment["source_end_sample"])
            # The same retained word can be aligned in independent contexts at
            # its leading and trailing cuts.  MFA emits float32 interval times,
            # so conservative sample rounding can make those contexts disagree
            # by one sample even though their continuous-time phone edge is the
            # same.  Accept only that documented rounding tolerance; larger
            # omissions still prove that retained speech disappeared.
            if (
                source_start <= protected_start + MFA_SAMPLE_ROUNDING_OVERLAP
                and source_end >= protected_end - MFA_SAMPLE_ROUNDING_OVERLAP
            ):
                mapped_start = max(source_start, protected_start)
                mapped_end = min(source_end, protected_end)
                if mapped_end <= mapped_start:
                    continue
                output_start = int(segment["output_start_sample"]) + (
                    mapped_start - source_start
                )
                output_end = output_start + mapped_end - mapped_start
                if not np.array_equal(
                    rendered_audio[output_start:output_end],
                    source_audio[mapped_start:mapped_end],
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
    alignment_backend: str = "mfa",
    mfa_prefix: Path = DEFAULT_MFA_PREFIX,
    mfa_cache_root: Path = DEFAULT_MFA_CACHE_ROOT,
    mfa_micromamba: str | Path = "micromamba",
    mfa_num_jobs: int = 1,
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
    mfa_payload: dict[str, Any] | None = None,
    mfa_payloads: Sequence[dict[str, Any]] | None = None,
    breath_cleanup: str = "off",
    breath_threshold: float = DEFAULT_BREATH_THRESHOLD,
    breath_min_duration_ms: int = DEFAULT_BREATH_MIN_DURATION_MS,
    respiro_cache_root: Path = DEFAULT_RESPIRO_CACHE_ROOT,
    breath_payload: dict[str, Any] | None = None,
    max_acoustic_retries: int = DEFAULT_MAX_ACOUSTIC_RETRIES,
    write_debug_artifacts: bool = False,
    pause_policy: str = "semantic",
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
    if alignment_backend != "mfa":
        raise ValueError("MFA is the only production alignment backend")
    if mfa_num_jobs <= 0:
        raise ValueError("mfa_num_jobs must be positive")
    if breath_cleanup not in BREATH_CLEANUP_MODES:
        raise ValueError(f"unsupported breath cleanup mode: {breath_cleanup}")
    if pause_policy not in PAUSE_POLICIES:
        raise ValueError(f"unsupported pause policy: {pause_policy}")
    if not 0.0 <= breath_threshold <= 1.0:
        raise ValueError("breath threshold must be inside [0, 1]")
    if breath_min_duration_ms <= 0:
        raise ValueError("breath minimum duration must be positive")
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
        pause_policy=pause_policy,
    )
    effective_plan = plan
    effective_plan_path = plan_path
    effective_grounding_path = grounding_path
    effective_selected_range_count = selected_range_count
    effective_pause_path = cached_pause_plan
    acoustic_repair_records: list[dict[str, Any]] = []
    acoustic_repair_failures: list[dict[str, Any]] = []
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
            attempt_mfa_payload = (
                mfa_payloads[acoustic_attempt]
                if mfa_payloads is not None and acoustic_attempt < len(mfa_payloads)
                else mfa_payload
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
                alignment_backend=alignment_backend,
                mfa_prefix=mfa_prefix,
                mfa_cache_root=mfa_cache_root,
                mfa_micromamba=mfa_micromamba,
                mfa_num_jobs=mfa_num_jobs,
                alignment_payload=attempt_payload,
                mfa_payload=attempt_mfa_payload,
                breath_cleanup=breath_cleanup,
                breath_threshold=breath_threshold,
                breath_min_duration_ms=breath_min_duration_ms,
                respiro_cache_root=respiro_cache_root,
                breath_payload=breath_payload,
                pause_policy=pause_policy,
            )
            if boundary_plan["status"] == "safe":
                break
            dense_count = int(boundary_plan["unsafe_dense_boundaries"])
            weak_word_count = int(boundary_plan.get("weak_retained_word_alignments", 0))
            mapping_failure_count = int(
                boundary_plan.get("mfa_word_mapping_failures", 0)
            )
            alignment_failures = int(boundary_plan["alignment_failures"])
            if (
                dense_count + weak_word_count + mapping_failure_count == 0
                or alignment_failures > mapping_failure_count
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
                in {
                    "unsafe_dense_boundary",
                    "weak_retained_word_alignment",
                    "mfa_word_mapping_failed",
                }
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
            try:
                repaired = repair_plan_for_acoustic_safety(
                    plan_path=effective_plan_path,
                    boundary_plan_path=rejected_path,
                    output_dir=semantic_retry_dir,
                    backend=active_repair_backend,
                    retry_index=retry_index,
                )
            except StreamingPlanError as error:
                acoustic_repair_failures.append(
                    {
                        "retry_index": retry_index,
                        "rejected_boundary_plan": str(rejected_path.resolve()),
                        "rejected_boundary_plan_sha256": sha256_file(rejected_path),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                break
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

    delivery_fallback_record: dict[str, Any] | None = None
    if boundary_plan["status"] != "safe":
        fallback_dir = output_dir / "conservative_delivery_fallback"
        unsafe_plan_path = fallback_dir / "unsafe_boundary_plan.json"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        write_json(unsafe_plan_path, boundary_plan)
        fallback_plan_dir = fallback_dir / "semantic_plan"
        try:
            fallback_plan = build_conservative_delivery_plan(
                plan_path=effective_plan_path,
                boundary_plan_path=unsafe_plan_path,
                output_dir=fallback_plan_dir,
            )
            fallback_plan_path = fallback_plan_dir / "streaming_plan.json"
            (
                effective_plan,
                effective_grounding_path,
                effective_selected_range_count,
            ) = _validate_grounded_plan(
                audio_path=audio_path,
                plan_path=fallback_plan_path,
            )
            if effective_plan != fallback_plan:
                raise FinalRenderError(
                    "saved conservative delivery plan differs from memory"
                )
            effective_plan_path = fallback_plan_path
            effective_pause_path = _retarget_pause_plan(
                source_pause_plan_path=effective_pause_path,
                repaired_plan_path=effective_plan_path,
                destination=fallback_dir / "pause_plan.json",
            )
            pause_plan = read_json(effective_pause_path)
            if not isinstance(pause_plan, dict):
                raise FinalRenderError("fallback pause plan root must be an object")
            fallback_payload_index = max_acoustic_retries + 1
            fallback_alignment_payload = (
                alignment_payloads[fallback_payload_index]
                if alignment_payloads is not None
                and fallback_payload_index < len(alignment_payloads)
                else None
            )
            fallback_mfa_payload = (
                mfa_payloads[fallback_payload_index]
                if mfa_payloads is not None
                and fallback_payload_index < len(mfa_payloads)
                else None
            )
            boundary_plan = build_final_boundary_plan(
                audio_path=audio_path,
                semantic_plan=effective_plan,
                semantic_plan_path=effective_plan_path,
                pause_plan=pause_plan,
                pause_plan_path=effective_pause_path,
                output_dir=fallback_dir / "boundary_evidence",
                alignment_python=alignment_python,
                alignment_backend=alignment_backend,
                mfa_prefix=mfa_prefix,
                mfa_cache_root=mfa_cache_root,
                mfa_micromamba=mfa_micromamba,
                mfa_num_jobs=mfa_num_jobs,
                alignment_payload=fallback_alignment_payload,
                mfa_payload=fallback_mfa_payload,
                breath_cleanup=breath_cleanup,
                breath_threshold=breath_threshold,
                breath_min_duration_ms=breath_min_duration_ms,
                respiro_cache_root=respiro_cache_root,
                breath_payload=breath_payload,
                pause_policy=pause_policy,
            )
            delivery_fallback_record = effective_plan.get("delivery_fallback")
        except (FinalRenderError, StreamingPlanError, OSError, ValueError) as error:
            delivery_fallback_record = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            write_json(
                fallback_dir / "delivery_fallback_failure.json",
                delivery_fallback_record,
            )
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
            reason = next(
                (
                    str(boundary["safety_status"])
                    for boundary in boundary_plan["boundaries"]
                    if boundary["safety_status"]
                    not in {"safe", "mfa_not_run_due_to_weak_retained_word"}
                ),
                "mfa_alignment_failed",
            )
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

    debug_manifest: dict[str, Any] | None = None
    if write_debug_artifacts:
        try:
            from .breath_debug import write_breath_debug_artifacts

            debug_manifest = write_breath_debug_artifacts(
                canonical_audio_path=audio_path,
                rendered_audio_path=final_path,
                boundary_plan_path=boundary_plan_path,
                output_dir=output_dir / "breath_debug",
            )
        except Exception as error:
            debug_manifest = {
                "status": "failed",
                "errors": [f"{type(error).__name__}: {error}"],
            }
            warnings.warn(
                "Breath diagnostics could not be written; the frozen plan "
                f"and final render remain valid: {type(error).__name__}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    manifest = {
        "schema_version": 3,
        "renderer": "authoritative_single_pass_final_render_v3",
        "status": "complete",
        "delivery_status": boundary_plan.get("delivery_status", "complete"),
        "delivery_fallback": (
            delivery_fallback_record or boundary_plan.get("delivery_fallback")
        ),
        "semantic_planner_fallbacks": list(
            boundary_plan.get("semantic_planner_fallbacks", [])
        ),
        "semantic_planner_request_failure_count": len(
            boundary_plan.get("semantic_planner_fallbacks", [])
        ),
        "semantic_planner_fallback_count": len(
            boundary_plan.get("semantic_preserved_source_fallbacks", [])
        ),
        "semantic_preserved_source_fallbacks": list(
            boundary_plan.get("semantic_preserved_source_fallbacks", [])
        ),
        "pause_plan_degraded": bool(boundary_plan.get("pause_plan_degraded")),
        "pause_degraded_batches": list(boundary_plan.get("pause_degraded_batches", [])),
        "pause_degraded_batch_count": len(
            boundary_plan.get("pause_degraded_batches", [])
        ),
        "alignment_backend": "mfa",
        "mfa_version": MFA_VERSION,
        "mfa_model": MFA_MODEL_ID,
        "mfa_fine_tune": True,
        "pause_policy": pause_policy,
        "breath_cleanup_mode": breath_cleanup,
        "breath_threshold": breath_threshold,
        "breath_min_duration_ms": breath_min_duration_ms,
        "respiro_upstream_commit": RESPIRO_UPSTREAM_COMMIT,
        "respiro_checkpoint_sha256": RESPIRO_CHECKPOINT_SHA256,
        "clean_ambience_bank_status": boundary_plan["clean_ambience_bank"]["status"],
        "clean_ambience_candidates": len(
            boundary_plan["clean_ambience_bank"].get("accepted_candidates", [])
        ),
        "ambience_candidates_rejected": len(
            boundary_plan["clean_ambience_bank"].get("rejected_candidates", [])
        ),
        "pauses_rendered_with_clean_ambience": sum(
            join.get("pause_content", {}).get("status") == "verified_clean_ambience"
            for join in boundary_plan["joins"]
        ),
        "original_gaps_preserved": sum(
            str(join.get("pause_content", {}).get("original_gap_content", ""))
            == "preserved_verified_clean"
            for join in boundary_plan["joins"]
        ),
        "original_gaps_replaced": sum(
            bool(join.get("pause_content", {}).get("original_content_replaced"))
            for join in boundary_plan["joins"]
        ),
        "pauses_clean_ambience_unavailable": sum(
            join.get("pause_content", {}).get("status") == "clean_ambience_unavailable"
            for join in boundary_plan["joins"]
        ),
        "breath_cleanup_status": boundary_plan["breath_cleanup"]["status"],
        "breaths_detected": len(
            boundary_plan["breath_cleanup"].get("detected_events", [])
        ),
        "breath_replacement_intervals": len(
            boundary_plan["breath_cleanup"].get("replacements", [])
        ),
        "breaths_replaced": sum(
            event.get("status")
            in {
                "breath_replaced_with_verified_room_tone",
                "breath_replaced_with_verified_clean_ambience",
            }
            for event in boundary_plan["breath_cleanup"].get("events", [])
        ),
        "breaths_skipped_phone_overlap": sum(
            event.get("status") == "breath_cleanup_skipped_phone_overlap"
            for event in boundary_plan["breath_cleanup"].get("events", [])
        ),
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
        "acoustic_repair_failures": acoustic_repair_failures,
        "rendered_clips": len(boundary_plan["source_intervals"]),
        "debug_artifacts_requested": write_debug_artifacts,
        "debug_artifacts_written": (
            debug_manifest is not None
            and debug_manifest.get("status") in {"complete", "complete_with_failures"}
        ),
        "breath_debug_manifest": (
            debug_manifest.get("manifest_path") if debug_manifest is not None else None
        ),
        "breath_debug_status": (
            debug_manifest.get("status") if debug_manifest is not None else None
        ),
        "breath_debug_errors": (
            list(debug_manifest.get("errors", [])) if debug_manifest is not None else []
        ),
        "alignment_contexts": boundary_plan["alignment_context_count"],
        "alignment_resolved_boundaries": boundary_plan["alignment_resolved_boundaries"],
        "unsafe_dense_boundaries": boundary_plan["unsafe_dense_boundaries"],
        "mfa_dense_phone_boundaries": boundary_plan["mfa_dense_phone_boundaries"],
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
    if manifest["delivery_status"] != "complete":
        warning_parts: list[str] = []
        semantic_fallback_count = int(manifest["semantic_planner_fallback_count"])
        if semantic_fallback_count:
            warning_parts.append(
                f"preserved exact source in {semantic_fallback_count} semantic "
                "planner window(s)"
            )
        delivery_record = manifest.get("delivery_fallback")
        preserved = (
            delivery_record.get("preserved_intervals", [])
            if isinstance(delivery_record, dict)
            else []
        )
        if preserved:
            warning_parts.append(
                f"preserved source context at {len(preserved)} unresolved cut(s)"
            )
        pause_fallback_count = int(manifest["pause_degraded_batch_count"])
        if pause_fallback_count:
            warning_parts.append(
                f"used deterministic pauses for {pause_fallback_count} "
                "planner batch(es)"
            )
        if not warning_parts:
            warning_parts.append("preserved conservative source context")
        warnings.warn(
            "VoiceCut completed with conservative source preservation and a "
            "playable fallback: "
            + "; ".join(warning_parts)
            + ". Review those localized regions; no invalid model range or "
            "guessed acoustic cut was accepted.",
            RuntimeWarning,
            stacklevel=2,
        )
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
        "--pause-policy",
        choices=PAUSE_POLICIES,
        default="semantic",
        help=(
            "Use semantic room-tone pauses for audio or zero-insertion clear "
            "cuts for video (default: semantic)."
        ),
    )
    parser.add_argument(
        "--alignment-python",
        type=Path,
        default=DEFAULT_ALIGNMENT_PYTHON,
        help=(
            "Python containing WhisperX, used only for the retained-word "
            "completeness veto; it never supplies cut coordinates."
        ),
    )
    parser.add_argument(
        "--alignment-backend",
        choices=("mfa",),
        default="mfa",
        help="Production coordinate backend (MFA is fail-closed and mandatory).",
    )
    parser.add_argument("--mfa-prefix", type=Path, default=DEFAULT_MFA_PREFIX)
    parser.add_argument(
        "--mfa-cache-root",
        type=Path,
        default=DEFAULT_MFA_CACHE_ROOT,
    )
    parser.add_argument("--mfa-micromamba", default="micromamba")
    parser.add_argument("--mfa-num-jobs", type=int, default=1)
    parser.add_argument(
        "--breath-cleanup",
        choices=BREATH_CLEANUP_MODES,
        default="replace",
        help=(
            "Respiro-en cleanup and ambience screening inside MFA-confirmed "
            "non-speech; off suppresses unverified inserted ambience "
            "(default: replace)."
        ),
    )
    parser.add_argument(
        "--breath-threshold",
        type=float,
        default=DEFAULT_BREATH_THRESHOLD,
        help="Respiro-en frame-probability threshold (default: 0.5).",
    )
    parser.add_argument(
        "--breath-min-duration-ms",
        type=int,
        default=DEFAULT_BREATH_MIN_DURATION_MS,
        help="Minimum consecutive breath-positive duration (default: 80 ms).",
    )
    parser.add_argument(
        "--respiro-cache-root",
        type=Path,
        default=DEFAULT_RESPIRO_CACHE_ROOT,
        help="Verified pinned Respiro-en runtime-model cache.",
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
    if args.mfa_num_jobs <= 0:
        raise SystemExit("--mfa-num-jobs must be positive")
    if not 0.0 <= args.breath_threshold <= 1.0:
        raise SystemExit("--breath-threshold must be inside [0, 1]")
    if args.breath_min_duration_ms <= 0:
        raise SystemExit("--breath-min-duration-ms must be positive")
    manifest = render_final_cut(
        audio_path=args.audio,
        plan_path=args.plan,
        output_dir=args.output_dir,
        pause_plan_path=args.pause_plan,
        alignment_python=args.alignment_python,
        alignment_backend=args.alignment_backend,
        mfa_prefix=args.mfa_prefix,
        mfa_cache_root=args.mfa_cache_root,
        mfa_micromamba=args.mfa_micromamba,
        mfa_num_jobs=args.mfa_num_jobs,
        breath_cleanup=args.breath_cleanup,
        breath_threshold=args.breath_threshold,
        breath_min_duration_ms=args.breath_min_duration_ms,
        respiro_cache_root=args.respiro_cache_root,
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
        pause_policy=args.pause_policy,
    )
    print("\nFINAL CUT CREATED")
    print(f"delivery status: {manifest['delivery_status']}")
    print(f"semantic thoughts: {manifest['semantic_thoughts']}")
    print(f"rendered clips: {manifest['rendered_clips']}")
    print(f"alignment contexts: {manifest['alignment_contexts']}")
    print(f"acoustic repair attempts: {manifest['acoustic_repair_attempts']}")
    print(f"alignment-resolved boundaries: {manifest['alignment_resolved_boundaries']}")
    print(f"unsafe dense boundaries: {manifest['unsafe_dense_boundaries']}")
    print(f"unresolved boundaries: {manifest['unresolved_boundaries']}")
    print(f"breaths detected: {manifest['breaths_detected']}")
    print(f"breaths replaced: {manifest['breaths_replaced']}")
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
