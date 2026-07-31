#!/usr/bin/env python3
"""Semantic pause classification plus a debug-only compatibility preview.

Production consumes the classification as evidence in
:mod:`voicecut.final_render`; this module's WAV renderer is not in the
production dependency chain.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from .common import read_json, sha256_file, write_json
from .rough_render import (
    flatten_selected_ranges,
    load_plan_words,
    merge_adjacent_ranges,
)
from .planner_backends import PlannerBackend as PausePlannerBackend
from .trailing_refine import (
    RMS_FRAME_MS,
    _first_stable_silence,
    _local_threshold,
    rms_envelope_db,
)


PAUSE_TYPES = ("continuation", "short", "thought", "section")
PAUSE_TARGETS_MS = {
    "continuation": 80,
    "short": 250,
    "thought": 650,
    "section": 1000,
}
MIN_QUIET_INSERTION_MS = 20.0
QUIET_SEARCH_MARGIN_MS = 120.0
EOF_MIN_EXTENSION_MS = 180.0
EOF_TRAILING_SILENCE_MS = 300.0
ROOM_TONE_WORD_GUARD_MS = 40.0
ROOM_TONE_FADE_MS = 5.0


class PausePlanError(RuntimeError):
    """The pause plan or pause render cannot be produced safely."""


class PausePlanValidationError(ValueError):
    """A model response violates the pause-planning contract."""


@dataclass(frozen=True)
class QuietInsertionPoint:
    source_sample: int
    quiet_start_sample: int
    quiet_end_sample: int
    existing_pause_samples: int
    local_noise_floor_db: float
    silence_threshold_db: float


@dataclass(frozen=True)
class EofTailDecision:
    raw_end_sample: int
    previous_end_sample: int
    search_start_sample: int
    stable_silence_start_sample: int | None
    new_end_sample: int
    fade_out_samples: int
    local_noise_floor_db: float
    silence_threshold_db: float
    boundary_method: str


@dataclass(frozen=True)
class ClipQuietHandles:
    leading_samples: int
    trailing_samples: int
    leading_method: str
    trailing_method: str
    leading_threshold_db: float
    trailing_threshold_db: float


@dataclass(frozen=True)
class RoomToneSelection:
    start_sample: int
    end_sample: int
    source_ranges: tuple[tuple[int, int], ...]
    reference_sample: int
    local_noise_floor_db: float
    silence_threshold_db: float


def pause_response_schema() -> dict[str, Any]:
    """Structured-output schema for transition classifications."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["transitions"],
        "properties": {
            "transitions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "after_thought_index",
                        "before_thought_index",
                        "pause_type",
                    ],
                    "properties": {
                        "after_thought_index": {"type": "integer"},
                        "before_thought_index": {"type": "integer"},
                        "pause_type": {
                            "type": "string",
                            "enum": list(PAUSE_TYPES),
                        },
                    },
                },
            }
        },
    }


def _load_committed_thoughts(plan: dict[str, Any]) -> list[dict[str, Any]]:
    if plan.get("status") != "complete":
        raise PausePlanError("semantic pauses require a complete streaming plan")
    raw_thoughts = plan.get("committed")
    if not isinstance(raw_thoughts, list) or not raw_thoughts:
        raise PausePlanError("streaming plan has no committed thoughts")
    thoughts: list[dict[str, Any]] = []
    for index, thought in enumerate(raw_thoughts):
        if not isinstance(thought, dict):
            raise PausePlanError(f"committed thought {index} is not an object")
        canonical = thought.get("canonical_text")
        ranges = thought.get("source_ranges")
        if not isinstance(canonical, str) or not canonical.strip():
            raise PausePlanError(f"committed thought {index} has no canonical text")
        if not isinstance(ranges, list) or not ranges:
            raise PausePlanError(f"committed thought {index} has no source ranges")
        thoughts.append(thought)
    return thoughts


def _pause_prompt(thoughts: Sequence[dict[str, Any]]) -> str:
    readable = "\n".join(
        f"thought {index}: {thought['canonical_text']}"
        for index, thought in enumerate(thoughts)
    )
    return f"""Classify every transition between adjacent committed narration
thoughts. The thoughts are already final and source-grounded. This is only a
pause-classification task.

Use exactly one pause_type for every transition:

continuation
The second thought grammatically or semantically continues the first.
Do not create a noticeable new pause.

short
A small clause or closely connected sentence boundary.

thought
The previous thought is complete and the next sentence starts a new
statement. This should receive a clearly audible pause.

section
There is a stronger topic, paragraph, or presentation-section transition.

Rules:
- Return exactly {len(thoughts) - 1} transitions.
- Transition i must be after_thought_index=i and before_thought_index=i+1.
- Keep transitions ordered.
- Do not modify or reproduce any narration in the JSON.
- Do not return milliseconds, explanations, source ranges, or extra fields.

COMMITTED THOUGHTS:
{readable}
"""


def _retry_prompt(
    original_prompt: str,
    *,
    invalid_raw: str,
    validation_error: str,
) -> str:
    return f"""Your previous pause classification failed local validation.

VALIDATION ERROR:
{validation_error}

PREVIOUS INVALID RESPONSE:
{invalid_raw}

Correct the JSON for the exact same transitions. Return no extra fields.

ORIGINAL REQUEST:
{original_prompt}
"""


def validate_pause_response(
    raw: str,
    *,
    thought_count: int,
) -> list[dict[str, Any]]:
    """Validate exact coverage, ordering, indices, keys, and pause labels."""

    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError as error:
        raise PausePlanValidationError(f"invalid JSON: {error}") from error
    if not isinstance(value, dict) or set(value) != {"transitions"}:
        raise PausePlanValidationError(
            "response must contain exactly the key 'transitions'"
        )
    transitions = value["transitions"]
    if not isinstance(transitions, list):
        raise PausePlanValidationError("transitions must be a list")
    expected_count = max(0, thought_count - 1)
    if len(transitions) != expected_count:
        raise PausePlanValidationError(
            f"expected {expected_count} transitions; got {len(transitions)}"
        )
    validated: list[dict[str, Any]] = []
    expected_keys = {
        "after_thought_index",
        "before_thought_index",
        "pause_type",
    }
    for index, transition in enumerate(transitions):
        if not isinstance(transition, dict) or set(transition) != expected_keys:
            raise PausePlanValidationError(
                f"transition {index} must contain exactly {sorted(expected_keys)}"
            )
        after = transition["after_thought_index"]
        before = transition["before_thought_index"]
        pause_type = transition["pause_type"]
        if type(after) is not int or type(before) is not int:
            raise PausePlanValidationError(
                f"transition {index} indices must be integers"
            )
        if after != index or before != index + 1:
            raise PausePlanValidationError(
                f"transition {index} must use indices {index} -> {index + 1}"
            )
        if pause_type not in PAUSE_TYPES:
            raise PausePlanValidationError(
                f"transition {index} has invalid pause_type {pause_type!r}"
            )
        validated.append(dict(transition))
    return validated


def create_pause_plan(
    *,
    plan_path: Path,
    output_dir: Path,
    backend: PausePlannerBackend,
) -> dict[str, Any]:
    """Call the pause classifier once, retry malformed output once, and save."""

    plan_path = plan_path.resolve()
    output_dir = output_dir.resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    pause_plan_path = output_dir / "pause_plan.json"
    if pause_plan_path.exists():
        raise RuntimeError(f"pause plan already exists: {pause_plan_path}")

    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise PausePlanError("streaming plan root must be an object")
    thoughts = _load_committed_thoughts(plan)
    prompt = _pause_prompt(thoughts)
    schema = pause_response_schema()
    attempts: list[dict[str, Any]] = []

    if len(thoughts) == 1:
        transitions: list[dict[str, Any]] = []
        raw = '{"transitions":[]}'
        attempts.append(
            {
                "attempt": 0,
                "raw_response": raw,
                "validation_error": None,
                "model_call_skipped": "there are no thought transitions",
            }
        )
    else:
        request_prompt = prompt
        last_raw = ""
        last_error = ""
        transitions = []
        for attempt in range(1, 3):
            raw = ""
            try:
                raw = backend.generate(
                    request_prompt,
                    response_schema=schema,
                    request_id=f"pause-plan-attempt-{attempt}",
                )
                transitions = validate_pause_response(
                    raw,
                    thought_count=len(thoughts),
                )
                attempts.append(
                    {
                        "attempt": attempt,
                        "raw_response": raw,
                        "validation_error": None,
                    }
                )
                (output_dir / f"pause_plan_attempt_{attempt}.raw.json").write_text(
                    raw,
                    encoding="utf-8",
                )
                break
            except Exception as error:
                last_raw = raw
                last_error = f"{type(error).__name__}: {error}"
                attempts.append(
                    {
                        "attempt": attempt,
                        "raw_response": raw,
                        "validation_error": last_error,
                    }
                )
                (output_dir / f"pause_plan_attempt_{attempt}.raw.txt").write_text(
                    raw,
                    encoding="utf-8",
                )
                if attempt == 1:
                    request_prompt = _retry_prompt(
                        prompt,
                        invalid_raw=last_raw,
                        validation_error=last_error,
                    )
        else:
            failure = {
                "schema_version": 1,
                "streaming_plan": str(plan_path),
                "streaming_plan_sha256": sha256_file(plan_path),
                "validation_error": last_error,
                "raw_response": last_raw,
                "attempts": attempts,
            }
            write_json(output_dir / "pause_plan_failure.json", failure)
            raise PausePlanError(
                f"{backend.backend_name} pause planner failed after one retry: "
                f"{last_error}; raw responses were saved in {output_dir}"
            )

    pause_plan = {
        "schema_version": 1,
        "planner": "semantic_pause_planner_v1",
        "backend": backend.backend_name,
        "model": backend.model,
        "streaming_plan": str(plan_path),
        "streaming_plan_sha256": sha256_file(plan_path),
        "thought_count": len(thoughts),
        "transition_count": len(transitions),
        "transitions": transitions,
        "attempts": attempts,
    }
    write_json(pause_plan_path, pause_plan)
    return pause_plan


def _validate_render_inputs(
    *,
    render_manifest_path: Path,
    pause_plan_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
    Path,
]:
    render_manifest = read_json(render_manifest_path)
    pause_plan = read_json(pause_plan_path)
    if not isinstance(render_manifest, dict):
        raise PausePlanError("render manifest root must be an object")
    if render_manifest.get("renderer") != "streaming_plan_full_boundary_alignment_v1":
        raise PausePlanError(
            "semantic pauses require a full-boundary alignment manifest"
        )
    if not isinstance(pause_plan, dict):
        raise PausePlanError("pause plan root must be an object")
    if pause_plan.get("planner") != "semantic_pause_planner_v1":
        raise PausePlanError("pause plan has an unsupported planner")

    audio_path = Path(str(render_manifest.get("source_audio", ""))).resolve()
    plan_path = Path(str(render_manifest.get("streaming_plan", ""))).resolve()
    previous_wav_path = Path(
        str(render_manifest.get("full_boundary_aligned_wav", ""))
    ).resolve()
    for path in (audio_path, plan_path, previous_wav_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(audio_path) != render_manifest.get("source_audio_sha256"):
        raise PausePlanError("source audio changed after forced alignment")
    if sha256_file(plan_path) != render_manifest.get("streaming_plan_sha256"):
        raise PausePlanError("streaming plan changed after forced alignment")
    if sha256_file(previous_wav_path) != render_manifest.get(
        "full_boundary_aligned_wav_sha256"
    ):
        raise PausePlanError("full-boundary aligned preview changed on disk")
    if sha256_file(plan_path) != pause_plan.get("streaming_plan_sha256"):
        raise PausePlanError("pause plan belongs to a different semantic plan")

    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise PausePlanError("streaming plan root must be an object")
    thoughts = _load_committed_thoughts(plan)
    transitions = pause_plan.get("transitions")
    if not isinstance(transitions, list):
        raise PausePlanError("pause plan transitions must be a list")
    validate_pause_response(
        json.dumps({"transitions": transitions}),
        thought_count=len(thoughts),
    )

    words = load_plan_words(plan)
    merged = merge_adjacent_ranges(flatten_selected_ranges(plan, word_count=len(words)))
    clips = render_manifest.get("clips")
    if not isinstance(clips, list) or len(clips) != len(merged):
        raise PausePlanError("forced-aligned clips differ from frozen semantic ranges")
    for clip, source_range in zip(clips, merged, strict=True):
        if (
            clip.get("source_word_start") != source_range.start_word_id
            or clip.get("source_word_end") != source_range.end_word_id
        ):
            raise PausePlanError(
                "forced-aligned clip boundaries differ from semantic ranges"
            )
    return (
        render_manifest,
        pause_plan,
        plan,
        audio_path,
        plan_path,
        previous_wav_path,
    )


def find_quiet_insertion_point(
    mono: np.ndarray,
    *,
    sample_rate: int,
    previous_word_end_seconds: float,
    next_word_start_seconds: float,
    clip_start_sample: int,
    clip_end_sample: int,
    min_quiet_ms: float = MIN_QUIET_INSERTION_MS,
    search_margin_ms: float = QUIET_SEARCH_MARGIN_MS,
) -> QuietInsertionPoint | None:
    """Find a low-energy run around an internal selected-word boundary."""

    if mono.ndim != 1 or not len(mono):
        raise ValueError("mono analysis waveform must be non-empty")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if min_quiet_ms <= 0.0 or search_margin_ms < 0.0:
        raise ValueError("quiet search durations are invalid")
    total_samples = len(mono)
    nominal_left = round(previous_word_end_seconds * sample_rate)
    nominal_right = round(next_word_start_seconds * sample_rate)
    center = round((nominal_left + nominal_right) / 2.0)
    margin = round(search_margin_ms * sample_rate / 1000.0)
    search_start = max(clip_start_sample, min(nominal_left, nominal_right) - margin)
    search_end = min(clip_end_sample, max(nominal_left, nominal_right) + margin)
    if not 0 <= search_start < search_end <= total_samples:
        return None
    noise_floor_db, threshold_db = _local_threshold(
        mono,
        raw_end_sample=max(0, min(total_samples, center)),
        sample_rate=sample_rate,
    )
    starts, rms_db = rms_envelope_db(
        mono,
        start_sample=search_start,
        end_sample=search_end,
        sample_rate=sample_rate,
    )
    frame_samples = max(1, round(RMS_FRAME_MS * sample_rate / 1000.0))
    required_samples = max(
        frame_samples,
        round(min_quiet_ms * sample_rate / 1000.0),
    )
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    run_last_start: int | None = None
    for frame_start, level_db in zip(starts, rms_db, strict=True):
        frame_start = int(frame_start)
        if level_db < threshold_db:
            if run_start is None:
                run_start = frame_start
            run_last_start = frame_start
        elif run_start is not None and run_last_start is not None:
            run_end = run_last_start + frame_samples
            if run_end - run_start >= required_samples:
                runs.append((run_start, run_end))
            run_start = None
            run_last_start = None
    if run_start is not None and run_last_start is not None:
        run_end = run_last_start + frame_samples
        if run_end - run_start >= required_samples:
            runs.append((run_start, run_end))
    if not runs:
        return None

    quiet_start, quiet_end = min(
        runs,
        key=lambda run: (
            abs(((run[0] + run[1]) // 2) - center),
            -(run[1] - run[0]),
        ),
    )
    half_required = max(1, required_samples // 2)
    safe_start = quiet_start + half_required
    safe_end = quiet_end - half_required
    if safe_start > safe_end:
        source_sample = (quiet_start + quiet_end) // 2
    else:
        candidate = mono[safe_start : safe_end + 1]
        source_sample = safe_start + int(np.argmin(np.abs(candidate)))
    return QuietInsertionPoint(
        source_sample=source_sample,
        quiet_start_sample=quiet_start,
        quiet_end_sample=quiet_end,
        existing_pause_samples=quiet_end - quiet_start,
        local_noise_floor_db=noise_floor_db,
        silence_threshold_db=threshold_db,
    )


def _first_active_frame(
    mono: np.ndarray,
    *,
    start_sample: int,
    end_sample: int,
    sample_rate: int,
    threshold_db: float,
) -> int | None:
    if end_sample <= start_sample:
        return None
    starts, levels = rms_envelope_db(
        mono,
        start_sample=start_sample,
        end_sample=end_sample,
        sample_rate=sample_rate,
    )
    return next(
        (
            int(frame_start)
            for frame_start, level in zip(starts, levels, strict=True)
            if level >= threshold_db
        ),
        None,
    )


def _last_active_frame_end(
    mono: np.ndarray,
    *,
    start_sample: int,
    end_sample: int,
    sample_rate: int,
    threshold_db: float,
) -> int | None:
    if end_sample <= start_sample:
        return None
    starts, levels = rms_envelope_db(
        mono,
        start_sample=start_sample,
        end_sample=end_sample,
        sample_rate=sample_rate,
    )
    frame_samples = max(1, round(RMS_FRAME_MS * sample_rate / 1000.0))
    active = [
        int(frame_start) + frame_samples
        for frame_start, level in zip(starts, levels, strict=True)
        if level >= threshold_db
    ]
    return min(end_sample, active[-1]) if active else None


def estimate_clip_quiet_handles(
    mono: np.ndarray,
    *,
    words: Sequence[Any],
    clip: dict[str, Any],
    sample_rate: int,
    source_end_sample: int | None = None,
    eof_stable_silence_start_sample: int | None = None,
) -> ClipQuietHandles:
    """Measure retained quiet head/tail used toward a target total pause."""

    if mono.ndim != 1 or not len(mono):
        raise ValueError("mono analysis waveform must be non-empty")
    start = int(clip["final_source_start_sample"])
    end = (
        int(clip["final_source_end_sample"])
        if source_end_sample is None
        else int(source_end_sample)
    )
    first_word_id = int(clip["source_word_start"])
    last_word_id = int(clip["source_word_end"]) - 1
    if not (
        0 <= first_word_id <= last_word_id < len(words)
        and 0 <= start < end <= len(mono)
    ):
        raise PausePlanError("clip has invalid quiet-handle geometry")

    raw_start = max(
        start,
        min(end, round(float(words[first_word_id].start) * sample_rate)),
    )
    raw_end = max(
        start,
        min(end, round(float(words[last_word_id].end) * sample_rate)),
    )
    _, leading_threshold = _local_threshold(
        mono,
        raw_end_sample=raw_start,
        sample_rate=sample_rate,
    )
    _, trailing_threshold = _local_threshold(
        mono,
        raw_end_sample=raw_end,
        sample_rate=sample_rate,
    )
    search_handle = round(500.0 * sample_rate / 1000.0)
    first_active = _first_active_frame(
        mono,
        start_sample=start,
        end_sample=min(end, raw_start + search_handle),
        sample_rate=sample_rate,
        threshold_db=leading_threshold,
    )
    if first_active is None:
        leading_samples = 0
        leading_method = "conservative_no_active_frame"
    else:
        leading_samples = max(0, first_active - start)
        leading_method = "waveform_first_active_frame"

    aligned_kept_end = clip.get("forced_aligned_kept_end_seconds")
    stable_start = clip.get("stable_silence_start_seconds")
    if eof_stable_silence_start_sample is not None:
        speech_end = int(eof_stable_silence_start_sample)
        trailing_method = "eof_stable_silence"
    elif type(aligned_kept_end) in {int, float} and math.isfinite(
        float(aligned_kept_end)
    ):
        speech_end = round(float(aligned_kept_end) * sample_rate)
        trailing_method = "forced_aligned_kept_end"
    elif type(stable_start) in {int, float} and math.isfinite(float(stable_start)):
        speech_end = round(float(stable_start) * sample_rate)
        trailing_method = "stable_silence_start"
    else:
        last_active = _last_active_frame_end(
            mono,
            start_sample=max(start, raw_end - search_handle),
            end_sample=end,
            sample_rate=sample_rate,
            threshold_db=trailing_threshold,
        )
        if last_active is None:
            speech_end = end
            trailing_method = "conservative_no_active_frame"
        else:
            speech_end = last_active
            trailing_method = "waveform_last_active_frame"
    speech_end = max(start, min(end, speech_end))
    trailing_samples = max(0, end - speech_end)
    return ClipQuietHandles(
        leading_samples=leading_samples,
        trailing_samples=trailing_samples,
        leading_method=leading_method,
        trailing_method=trailing_method,
        leading_threshold_db=leading_threshold,
        trailing_threshold_db=trailing_threshold,
    )


class SourceRoomToneAllocator:
    """Allocate unique, waveform-verified quiet source spans for pauses."""

    def __init__(
        self,
        *,
        source_audio: np.ndarray,
        mono: np.ndarray,
        words: Sequence[Any],
        sample_rate: int,
        word_guard_ms: float = ROOM_TONE_WORD_GUARD_MS,
        fade_ms: float = ROOM_TONE_FADE_MS,
        allowed_source_spans: Sequence[tuple[int, int]] | None = None,
        exclusion_intervals: Sequence[dict[str, Any]] = (),
        allow_reuse: bool = False,
    ) -> None:
        if source_audio.ndim != 2 or len(source_audio) != len(mono):
            raise ValueError("source audio and mono analysis must share geometry")
        if sample_rate <= 0 or word_guard_ms < 0.0 or fade_ms < 0.0:
            raise ValueError("invalid room-tone allocator configuration")
        self.source_audio = source_audio
        self.mono = mono
        self.sample_rate = sample_rate
        self.fade_samples = round(fade_ms * sample_rate / 1000.0)
        self.allow_reuse = bool(allow_reuse)
        self._allocated: list[tuple[int, int]] = []
        self._rejection_ledger: list[dict[str, Any]] = []
        self._allocation_ledger: list[dict[str, Any]] = []
        raw_runs = self._discover_runs(
            words=words,
            word_guard_samples=round(word_guard_ms * sample_rate / 1000.0),
        )
        self._runs = self._filter_runs(
            raw_runs,
            allowed_source_spans=allowed_source_spans,
            exclusion_intervals=exclusion_intervals,
        )

    @property
    def candidate_ranges(self) -> list[tuple[int, int]]:
        return [(start, end) for start, end, _, _ in self._runs]

    @property
    def rejection_ledger(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._rejection_ledger]

    @property
    def allocation_ledger(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._allocation_ledger]

    @staticmethod
    def _subtract_interval(
        spans: Sequence[tuple[int, int]],
        *,
        excluded_start: int,
        excluded_end: int,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        remaining: list[tuple[int, int]] = []
        removed: list[tuple[int, int]] = []
        for start, end in spans:
            overlap_start = max(start, excluded_start)
            overlap_end = min(end, excluded_end)
            if overlap_end <= overlap_start:
                remaining.append((start, end))
                continue
            removed.append((overlap_start, overlap_end))
            if start < overlap_start:
                remaining.append((start, overlap_start))
            if overlap_end < end:
                remaining.append((overlap_end, end))
        return remaining, removed

    @staticmethod
    def _merged_spans(
        spans: Sequence[tuple[int, int]],
        *,
        total_samples: int,
    ) -> list[tuple[int, int]]:
        normalized = sorted(
            (max(0, int(start)), min(total_samples, int(end)))
            for start, end in spans
            if int(end) > int(start)
        )
        merged: list[list[int]] = []
        for start, end in normalized:
            if end <= start:
                continue
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return [(start, end) for start, end in merged]

    def _filter_runs(
        self,
        runs: Sequence[tuple[int, int, float, float]],
        *,
        allowed_source_spans: Sequence[tuple[int, int]] | None,
        exclusion_intervals: Sequence[dict[str, Any]],
    ) -> list[tuple[int, int, float, float]]:
        total_samples = len(self.mono)
        allowed = (
            self._merged_spans(
                allowed_source_spans,
                total_samples=total_samples,
            )
            if allowed_source_spans is not None
            else None
        )
        exclusions: list[tuple[int, int, str]] = []
        for raw in exclusion_intervals:
            start = max(0, int(raw["start_sample"]))
            end = min(total_samples, int(raw["end_sample"]))
            reason = str(raw.get("reason") or "unspecified_exclusion")
            if end > start:
                exclusions.append((start, end, reason))

        filtered: list[tuple[int, int, float, float]] = []
        for run_start, run_end, noise_floor, threshold in runs:
            spans = [(run_start, run_end)]
            for excluded_start, excluded_end, reason in sorted(exclusions):
                spans, removed = self._subtract_interval(
                    spans,
                    excluded_start=excluded_start,
                    excluded_end=excluded_end,
                )
                self._rejection_ledger.extend(
                    {
                        "candidate_start_sample": run_start,
                        "candidate_end_sample": run_end,
                        "rejected_start_sample": start,
                        "rejected_end_sample": end,
                        "reason": reason,
                    }
                    for start, end in removed
                )
            if allowed is not None:
                allowed_parts = [
                    (max(start, allowed_start), min(end, allowed_end))
                    for start, end in spans
                    for allowed_start, allowed_end in allowed
                    if min(end, allowed_end) > max(start, allowed_start)
                ]
                disallowed = list(spans)
                for allowed_start, allowed_end in allowed:
                    disallowed, _ = self._subtract_interval(
                        disallowed,
                        excluded_start=allowed_start,
                        excluded_end=allowed_end,
                    )
                self._rejection_ledger.extend(
                    {
                        "candidate_start_sample": run_start,
                        "candidate_end_sample": run_end,
                        "rejected_start_sample": start,
                        "rejected_end_sample": end,
                        "reason": "not_mfa_verified_non_speech",
                    }
                    for start, end in disallowed
                )
                spans = self._merged_spans(
                    allowed_parts,
                    total_samples=total_samples,
                )
            filtered.extend(
                (start, end, noise_floor, threshold)
                for start, end in spans
                if end > start
            )
        return filtered

    def _discover_runs(
        self,
        *,
        words: Sequence[Any],
        word_guard_samples: int,
    ) -> list[tuple[int, int, float, float]]:
        total_samples = len(self.mono)
        timed = [
            (
                max(
                    0,
                    min(
                        total_samples,
                        math.floor(float(word.start) * self.sample_rate),
                    ),
                ),
                max(
                    0,
                    min(
                        total_samples,
                        math.ceil(float(word.end) * self.sample_rate),
                    ),
                ),
            )
            for word in words
        ]
        gaps: list[tuple[int, int]] = []
        cursor = 0
        for start, end in timed:
            gap_start = cursor + word_guard_samples
            gap_end = start - word_guard_samples
            if gap_end > gap_start:
                gaps.append((gap_start, gap_end))
            cursor = max(cursor, end)
        tail_start = cursor + word_guard_samples
        if total_samples > tail_start:
            gaps.append((tail_start, total_samples))

        frame_samples = max(
            1,
            round(RMS_FRAME_MS * self.sample_rate / 1000.0),
        )
        runs: list[tuple[int, int, float, float]] = []
        for gap_start, gap_end in gaps:
            if gap_end - gap_start < frame_samples:
                continue
            midpoint = (gap_start + gap_end) // 2
            noise_floor, threshold = _local_threshold(
                self.mono,
                raw_end_sample=midpoint,
                sample_rate=self.sample_rate,
            )
            starts, levels = rms_envelope_db(
                self.mono,
                start_sample=gap_start,
                end_sample=gap_end,
                sample_rate=self.sample_rate,
            )
            run_start: int | None = None
            run_end: int | None = None
            for frame_start, level in zip(starts, levels, strict=True):
                frame_start = int(frame_start)
                if level < threshold:
                    if run_start is None:
                        run_start = frame_start
                    run_end = min(gap_end, frame_start + frame_samples)
                elif run_start is not None and run_end is not None:
                    runs.append((run_start, run_end, noise_floor, threshold))
                    run_start = None
                    run_end = None
            if run_start is not None and run_end is not None:
                runs.append((run_start, run_end, noise_floor, threshold))
        return runs

    def _available_spans(
        self,
        start: int,
        end: int,
    ) -> list[tuple[int, int]]:
        spans = [(start, end)]
        for used_start, used_end in sorted(self._allocated):
            next_spans: list[tuple[int, int]] = []
            for span_start, span_end in spans:
                if used_end <= span_start or used_start >= span_end:
                    next_spans.append((span_start, span_end))
                    continue
                if used_start > span_start:
                    next_spans.append((span_start, used_start))
                if used_end < span_end:
                    next_spans.append((used_end, span_end))
            spans = next_spans
        return spans

    def allocate(
        self,
        *,
        frame_count: int,
        reference_sample: int,
    ) -> tuple[np.ndarray, RoomToneSelection | None]:
        if frame_count < 0:
            raise ValueError("room-tone frame count cannot be negative")
        if frame_count == 0:
            return (
                np.zeros(
                    (0, self.source_audio.shape[1]),
                    dtype=np.float32,
                ),
                None,
            )
        candidates: list[tuple[int, int, float, float]] = []
        for run_start, run_end, noise_floor, threshold in self._runs:
            for start, end in self._available_spans(run_start, run_end):
                if end > start:
                    candidates.append((start, end, noise_floor, threshold))
        unique_capacity = sum(end - start for start, end, _, _ in candidates)
        if unique_capacity < frame_count and not self.allow_reuse:
            raise PausePlanError(
                "verified source room tone cannot satisfy a required pause "
                f"of {frame_count * 1000.0 / self.sample_rate:.1f} ms"
            )
        ranked = sorted(
            candidates,
            key=lambda item: (
                abs(((item[0] + item[1]) // 2) - reference_sample),
                -(item[1] - item[0]),
            ),
        )
        remaining = frame_count
        chunks: list[np.ndarray] = []
        selected_ranges: list[tuple[int, int]] = []
        selected_noise_floors: list[float] = []
        selected_thresholds: list[float] = []
        reusable = sorted(
            self._runs,
            key=lambda item: (
                abs(((item[0] + item[1]) // 2) - reference_sample),
                -(item[1] - item[0]),
            ),
        )
        candidate_batches = [ranked]
        if self.allow_reuse:
            if not reusable:
                raise PausePlanError("no verified source room tone is available")
            repeated_capacity = sum(end - start for start, end, _, _ in reusable)
            repeat_count = max(
                1, math.ceil(max(0, remaining - unique_capacity) / repeated_capacity)
            )
            candidate_batches.extend(reusable for _ in range(repeat_count))
        for batch in candidate_batches:
            for available_start, available_end, noise_floor, threshold in batch:
                if remaining <= 0:
                    break
                take = min(remaining, available_end - available_start)
                desired_start = reference_sample - take // 2
                start = max(
                    available_start,
                    min(available_end - take, desired_start),
                )
                end = start + take
                reused = any(
                    max(start, used_start) < min(end, used_end)
                    for used_start, used_end in self._allocated
                )
                chunk = np.array(
                    self.source_audio[start:end],
                    dtype=np.float32,
                    copy=True,
                )
                fade = min(self.fade_samples, len(chunk) // 2)
                if fade:
                    chunk[:fade] *= np.linspace(
                        0.0,
                        1.0,
                        fade,
                        endpoint=True,
                        dtype=np.float32,
                    )[:, None]
                    chunk[-fade:] *= np.linspace(
                        1.0,
                        0.0,
                        fade,
                        endpoint=True,
                        dtype=np.float32,
                    )[:, None]
                chunks.append(chunk)
                selected_ranges.append((start, end))
                selected_noise_floors.append(noise_floor)
                selected_thresholds.append(threshold)
                self._allocated.append((start, end))
                self._allocation_ledger.append(
                    {
                        "source_start_sample": start,
                        "source_end_sample": end,
                        "reference_sample": reference_sample,
                        "requested_frame_count": frame_count,
                        "reused": reused,
                    }
                )
                remaining -= take
            if remaining <= 0:
                break
        if remaining:
            raise PausePlanError("room-tone allocation accounting failed")
        audio = np.concatenate(chunks, axis=0)
        if len(audio) != frame_count:
            raise PausePlanError("room-tone allocation has wrong duration")
        return (
            audio,
            RoomToneSelection(
                start_sample=selected_ranges[0][0],
                end_sample=selected_ranges[-1][1],
                source_ranges=tuple(selected_ranges),
                reference_sample=reference_sample,
                local_noise_floor_db=float(np.mean(selected_noise_floors)),
                silence_threshold_db=float(np.mean(selected_thresholds)),
            ),
        )


def refine_eof_tail(
    mono: np.ndarray,
    *,
    sample_rate: int,
    raw_end_seconds: float,
    previous_end_sample: int,
    fade_ms: float,
    eof_min_extension_ms: float = EOF_MIN_EXTENSION_MS,
    eof_trailing_silence_ms: float = EOF_TRAILING_SILENCE_MS,
) -> EofTailDecision:
    """Retain a safe final-word decay and tail because no later word is at risk."""

    if mono.ndim != 1 or not len(mono):
        raise ValueError("mono analysis waveform must be non-empty")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    total_samples = len(mono)
    raw_end = max(
        0,
        min(total_samples, math.ceil(raw_end_seconds * sample_rate)),
    )
    previous_end = max(0, min(total_samples, int(previous_end_sample)))
    search_start = min(
        total_samples,
        raw_end + round(eof_min_extension_ms * sample_rate / 1000.0),
    )
    noise_floor_db, threshold_db = _local_threshold(
        mono,
        raw_end_sample=raw_end,
        sample_rate=sample_rate,
    )
    stable_start = _first_stable_silence(
        mono,
        raw_end_sample=search_start,
        search_start_sample=search_start,
        search_limit_sample=total_samples,
        sample_rate=sample_rate,
        threshold_db=threshold_db,
        require_recent_active=False,
    )
    if stable_start is None:
        new_end = total_samples
        fade_out = 0
    else:
        candidate_end = min(
            total_samples,
            stable_start + round(eof_trailing_silence_ms * sample_rate / 1000.0),
        )
        new_end = max(previous_end, candidate_end)
        requested_fade = max(0, round(fade_ms * sample_rate / 1000.0))
        fade_out = min(requested_fade, new_end)
        if new_end - fade_out < stable_start:
            fade_out = 0
    return EofTailDecision(
        raw_end_sample=raw_end,
        previous_end_sample=previous_end,
        search_start_sample=search_start,
        stable_silence_start_sample=stable_start,
        new_end_sample=new_end,
        fade_out_samples=fade_out,
        local_noise_floor_db=noise_floor_db,
        silence_threshold_db=threshold_db,
        boundary_method="eof_safe_tail",
    )


def _apply_edge_fades(
    samples: np.ndarray,
    *,
    fade_in_samples: int,
    fade_out_samples: int,
) -> np.ndarray:
    rendered = np.array(samples, dtype=np.float32, copy=True)
    fade_in = min(max(0, fade_in_samples), len(rendered))
    fade_out = min(max(0, fade_out_samples), len(rendered))
    if fade_in:
        ramp = np.linspace(
            0.0,
            1.0,
            fade_in,
            endpoint=True,
            dtype=np.float32,
        )
        rendered[:fade_in] *= ramp[:, None]
    if fade_out:
        ramp = np.linspace(
            1.0,
            0.0,
            fade_out,
            endpoint=True,
            dtype=np.float32,
        )
        rendered[-fade_out:] *= ramp[:, None]
    return rendered


def _thought_source_bounds(
    thoughts: Sequence[dict[str, Any]],
) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    previous_end = 0
    for index, thought in enumerate(thoughts):
        ranges = thought["source_ranges"]
        first = ranges[0]
        last = ranges[-1]
        start = first.get("start_word_id")
        end = last.get("end_word_id")
        if type(start) is not int or type(end) is not int or not start < end:
            raise PausePlanError(f"thought {index} has invalid source boundaries")
        if start < previous_end:
            raise PausePlanError("committed thought ranges move backward")
        bounds.append((start, end))
        previous_end = end
    return bounds


def _clip_index_for_word(
    clips: Sequence[dict[str, Any]],
    word_id: int,
) -> int:
    matches = [
        int(clip["clip_index"])
        for clip in clips
        if int(clip["source_word_start"]) <= word_id < int(clip["source_word_end"])
    ]
    if len(matches) != 1:
        raise PausePlanError(
            f"selected word {word_id} does not belong to exactly one clip"
        )
    return matches[0]


def _insert_audio_segments(
    audio: np.ndarray,
    *,
    insertions: Sequence[tuple[int, np.ndarray]],
) -> np.ndarray:
    """Insert ordered native-channel source segments without removing audio."""

    if audio.ndim != 2:
        raise ValueError("audio must have shape (frames, channels)")
    cursor = 0
    parts: list[np.ndarray] = []
    for local_sample, inserted in insertions:
        if not cursor <= local_sample <= len(audio):
            raise PausePlanError("audio insertion points move backward")
        if inserted.ndim != 2 or inserted.shape[1] != audio.shape[1]:
            raise PausePlanError("inserted room tone has incompatible channels")
        parts.append(audio[cursor:local_sample])
        if len(inserted):
            parts.append(inserted)
        cursor = local_sample
    parts.append(audio[cursor:])
    return np.concatenate(parts, axis=0) if parts else np.array(audio, copy=True)


def _write_final_sentence_previews(
    *,
    output_dir: Path,
    source_audio: np.ndarray,
    sample_rate: int,
    words: Sequence[Any],
    thoughts: Sequence[dict[str, Any]],
    old_end_sample: int,
    new_end_sample: int,
) -> tuple[Path, Path]:
    final_start_id = int(thoughts[-1]["source_ranges"][0]["start_word_id"])
    start = max(
        0,
        math.floor(words[final_start_id].start * sample_rate)
        - round(30.0 * sample_rate / 1000.0),
    )
    fade = round(5.0 * sample_rate / 1000.0)
    old_audio = _apply_edge_fades(
        source_audio[start:old_end_sample],
        fade_in_samples=fade,
        fade_out_samples=fade,
    )
    new_audio = _apply_edge_fades(
        source_audio[start:new_end_sample],
        fade_in_samples=fade,
        fade_out_samples=fade,
    )
    old_path = output_dir / "final_sentence_old.wav"
    new_path = output_dir / "final_sentence_new.wav"
    sf.write(old_path, old_audio, sample_rate, subtype="FLOAT")
    sf.write(new_path, new_audio, sample_rate, subtype="FLOAT")
    return old_path, new_path


def render_semantic_pauses(
    *,
    render_manifest_path: Path,
    pause_plan_path: Path,
    output_dir: Path,
    pause_targets_ms: dict[str, int] | None = None,
    min_quiet_insertion_ms: float = MIN_QUIET_INSERTION_MS,
    quiet_search_margin_ms: float = QUIET_SEARCH_MARGIN_MS,
    eof_min_extension_ms: float = EOF_MIN_EXTENSION_MS,
    eof_trailing_silence_ms: float = EOF_TRAILING_SILENCE_MS,
    write_debug_artifacts: bool = True,
) -> dict[str, Any]:
    """Render semantic pauses onto the frozen forced-aligned preview."""

    render_manifest_path = render_manifest_path.resolve()
    pause_plan_path = pause_plan_path.resolve()
    output_dir = output_dir.resolve()
    if not render_manifest_path.is_file():
        raise FileNotFoundError(render_manifest_path)
    if not pause_plan_path.is_file():
        raise FileNotFoundError(pause_plan_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rough_cut_with_semantic_pauses.wav"
    manifest_path = output_dir / "pause_render_manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise RuntimeError("semantic-pause render outputs already exist")

    targets = dict(PAUSE_TARGETS_MS if pause_targets_ms is None else pause_targets_ms)
    if set(targets) != set(PAUSE_TYPES):
        raise ValueError(f"pause targets must define exactly {list(PAUSE_TYPES)}")
    for pause_type, milliseconds in targets.items():
        if type(milliseconds) is not int or milliseconds < 0:
            raise ValueError(f"invalid target for {pause_type}: {milliseconds}")

    (
        previous,
        pause_plan,
        plan,
        audio_path,
        plan_path,
        previous_wav_path,
    ) = _validate_render_inputs(
        render_manifest_path=render_manifest_path,
        pause_plan_path=pause_plan_path,
    )
    thoughts = _load_committed_thoughts(plan)
    words = load_plan_words(plan)
    clips = previous["clips"]
    transitions = pause_plan["transitions"]
    thought_bounds = _thought_source_bounds(thoughts)

    source_audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )
    sample_rate = int(sample_rate)
    total_samples, channel_count = source_audio.shape
    previous_audio, previous_rate = sf.read(
        previous_wav_path,
        dtype="float32",
        always_2d=True,
    )
    if (
        int(previous_rate) != sample_rate
        or previous_audio.shape[1] != channel_count
        or len(previous_audio)
        != int(previous["full_boundary_aligned_expected_output_frame_count"])
    ):
        raise PausePlanError("previous forced-aligned WAV geometry changed")
    if (
        sample_rate != int(previous["source_sample_rate"])
        or total_samples != int(previous["source_frame_count"])
        or channel_count != int(previous["source_channel_count"])
    ):
        raise PausePlanError("source audio geometry changed")
    mono = np.mean(source_audio, axis=1, dtype=np.float64).astype(np.float32)

    # Slice the verified previous preview so all ordinary and forced-aligned
    # boundaries remain byte-for-byte unchanged inside every non-final clip.
    clip_audio: list[np.ndarray] = []
    for index, clip in enumerate(clips):
        if int(clip["clip_index"]) != index:
            raise PausePlanError("clip indices must be contiguous and ordered")
        start = int(clip["final_output_start_sample"])
        end = int(clip["final_output_end_sample"])
        expected = int(clip["final_frame_count"])
        if not 0 <= start < end <= len(previous_audio) or end - start != expected:
            raise PausePlanError(f"clip {index} has invalid previous geometry")
        clip_audio.append(np.array(previous_audio[start:end], copy=True))

    fade_ms = float(previous["configuration"]["clip_fade_ms"])
    final_clip = clips[-1]
    final_raw_end_seconds = words[int(final_clip["source_word_end"]) - 1].end
    previous_final_end = int(final_clip["final_source_end_sample"])
    final_source_start = int(final_clip["final_source_start_sample"])
    final_selected_end = int(final_clip["source_word_end"])
    if final_selected_end == len(words):
        eof = refine_eof_tail(
            mono,
            sample_rate=sample_rate,
            raw_end_seconds=final_raw_end_seconds,
            previous_end_sample=previous_final_end,
            fade_ms=fade_ms,
            eof_min_extension_ms=eof_min_extension_ms,
            eof_trailing_silence_ms=eof_trailing_silence_ms,
        )
        if not 0 <= final_source_start < eof.new_end_sample <= total_samples:
            raise PausePlanError("safe EOF tail produced invalid source geometry")
        clip_audio[-1] = _apply_edge_fades(
            source_audio[final_source_start : eof.new_end_sample],
            fade_in_samples=int(final_clip["final_fade_in_samples"]),
            fade_out_samples=eof.fade_out_samples,
        )
    else:
        raw_end_sample = max(
            0,
            min(
                total_samples,
                math.ceil(final_raw_end_seconds * sample_rate),
            ),
        )
        noise_floor_db, threshold_db = _local_threshold(
            mono,
            raw_end_sample=raw_end_sample,
            sample_rate=sample_rate,
        )
        eof = EofTailDecision(
            raw_end_sample=raw_end_sample,
            previous_end_sample=previous_final_end,
            search_start_sample=raw_end_sample,
            stable_silence_start_sample=None,
            new_end_sample=previous_final_end,
            fade_out_samples=int(final_clip["final_fade_out_samples"]),
            local_noise_floor_db=noise_floor_db,
            silence_threshold_db=threshold_db,
            boundary_method="not_end_of_file",
        )

    configured_join_ms = float(previous["configuration"]["inter_clip_silence_ms"])
    clip_handles = [
        estimate_clip_quiet_handles(
            mono,
            words=words,
            clip=clip,
            sample_rate=sample_rate,
            source_end_sample=(eof.new_end_sample if index == len(clips) - 1 else None),
            eof_stable_silence_start_sample=(
                eof.stable_silence_start_sample if index == len(clips) - 1 else None
            ),
        )
        for index, clip in enumerate(clips)
    ]

    thought_by_word: dict[int, int] = {}
    for thought_index, thought in enumerate(thoughts):
        for source_range in thought["source_ranges"]:
            start = int(source_range["start_word_id"])
            end = int(source_range["end_word_id"])
            for word_id in range(start, end):
                if word_id in thought_by_word:
                    raise PausePlanError(
                        f"selected word {word_id} belongs to multiple thoughts"
                    )
                thought_by_word[word_id] = thought_index
    transition_by_pair = {
        (
            int(transition["after_thought_index"]),
            int(transition["before_thought_index"]),
        ): transition
        for transition in transitions
    }
    join_samples: list[int] = []
    clip_join_manifest: list[dict[str, Any]] = []
    for join_index, (left_clip, right_clip) in enumerate(zip(clips, clips[1:])):
        left_word_id = int(left_clip["source_word_end"]) - 1
        right_word_id = int(right_clip["source_word_start"])
        left_thought = thought_by_word.get(left_word_id)
        right_thought = thought_by_word.get(right_word_id)
        if left_thought is None or right_thought is None:
            raise PausePlanError("clip join contains an unowned selected word")
        if left_thought == right_thought:
            pause_type = "continuation"
            transition_kind = "intra_thought_source_cut"
        else:
            transition = transition_by_pair.get((left_thought, right_thought))
            if transition is None:
                raise PausePlanError(
                    "clip join crosses thoughts without a classified transition"
                )
            pause_type = str(transition["pause_type"])
            transition_kind = "thought_transition"
        target_samples = round(targets[pause_type] * sample_rate / 1000.0)
        retained_tail = clip_handles[join_index].trailing_samples
        retained_head = clip_handles[join_index + 1].leading_samples
        existing_samples = retained_tail + retained_head
        inserted_samples = max(0, target_samples - existing_samples)
        join_samples.append(inserted_samples)
        clip_join_manifest.append(
            {
                "join_index": join_index,
                "left_clip_index": int(left_clip["clip_index"]),
                "right_clip_index": int(right_clip["clip_index"]),
                "left_thought_index": left_thought,
                "right_thought_index": right_thought,
                "transition_kind": transition_kind,
                "pause_type": pause_type,
                "target_pause_ms": target_samples * 1000.0 / sample_rate,
                "retained_tail_ms": retained_tail * 1000.0 / sample_rate,
                "retained_head_ms": retained_head * 1000.0 / sample_rate,
                "estimated_existing_pause_ms": (
                    existing_samples * 1000.0 / sample_rate
                ),
                "inserted_pause_ms": (inserted_samples * 1000.0 / sample_rate),
                "estimated_total_pause_ms": (
                    (existing_samples + inserted_samples) * 1000.0 / sample_rate
                ),
                "left_handle_method": (clip_handles[join_index].trailing_method),
                "right_handle_method": (clip_handles[join_index + 1].leading_method),
                "status": (
                    "pause_inserted"
                    if inserted_samples
                    else "existing_pause_satisfies_target"
                ),
            }
        )
    room_tone_allocator = SourceRoomToneAllocator(
        source_audio=source_audio,
        mono=mono,
        words=words,
        sample_rate=sample_rate,
    )
    join_audio: list[np.ndarray] = []
    for join_index, (inserted_samples, join) in enumerate(
        zip(join_samples, clip_join_manifest, strict=True)
    ):
        filler, selection = room_tone_allocator.allocate(
            frame_count=inserted_samples,
            reference_sample=int(clips[join_index]["final_source_end_sample"]),
        )
        join_audio.append(filler)
        join.update(
            {
                "pause_fill_method": (
                    "verified_source_room_tone" if selection is not None else "none"
                ),
                "room_tone_source_start_seconds": (
                    selection.start_sample / sample_rate
                    if selection is not None
                    else None
                ),
                "room_tone_source_end_seconds": (
                    selection.end_sample / sample_rate
                    if selection is not None
                    else None
                ),
                "room_tone_source_ranges_seconds": (
                    [
                        {
                            "start": start / sample_rate,
                            "end": end / sample_rate,
                        }
                        for start, end in selection.source_ranges
                    ]
                    if selection is not None
                    else []
                ),
                "room_tone_noise_floor_db": (
                    selection.local_noise_floor_db if selection is not None else None
                ),
                "room_tone_silence_threshold_db": (
                    selection.silence_threshold_db if selection is not None else None
                ),
            }
        )
    internal_insertions: dict[
        int,
        list[tuple[int, np.ndarray, int, RoomToneSelection]],
    ] = {index: [] for index in range(len(clips))}
    transition_manifest: list[dict[str, Any]] = []

    for transition_index, transition in enumerate(transitions):
        after = int(transition["after_thought_index"])
        before = int(transition["before_thought_index"])
        pause_type = str(transition["pause_type"])
        target_ms = int(targets[pause_type])
        previous_start, previous_end = thought_bounds[after]
        next_start, next_end = thought_bounds[before]
        del previous_start, next_end
        last_word_id = previous_end - 1
        first_word_id = next_start
        previous_clip_index = _clip_index_for_word(clips, last_word_id)
        next_clip_index = _clip_index_for_word(clips, first_word_id)
        common = {
            "after_thought_index": after,
            "before_thought_index": before,
            "pause_type": pause_type,
            "target_pause_ms": target_ms,
            "last_word_id": last_word_id,
            "last_word": words[last_word_id].text,
            "first_word_id": first_word_id,
            "first_word": words[first_word_id].text,
            "previous_clip_index": previous_clip_index,
            "next_clip_index": next_clip_index,
        }

        if previous_clip_index == next_clip_index:
            if previous_end != next_start:
                raise PausePlanError(
                    "non-adjacent thought ranges unexpectedly share one clip"
                )
            clip = clips[previous_clip_index]
            quiet = find_quiet_insertion_point(
                mono,
                sample_rate=sample_rate,
                previous_word_end_seconds=words[last_word_id].end,
                next_word_start_seconds=words[first_word_id].start,
                clip_start_sample=int(clip["final_source_start_sample"]),
                clip_end_sample=(
                    eof.new_end_sample
                    if previous_clip_index == len(clips) - 1
                    else int(clip["final_source_end_sample"])
                ),
                min_quiet_ms=min_quiet_insertion_ms,
                search_margin_ms=quiet_search_margin_ms,
            )
            if quiet is None:
                record = {
                    **common,
                    "boundary_location": "inside_continuous_source_clip",
                    "estimated_existing_pause_ms": 0.0,
                    "inserted_pause_ms": 0.0,
                    "insertion_method": "none",
                    "status": "pause_not_inserted_no_safe_point",
                    "source_insertion_seconds": None,
                    "quiet_region_start_seconds": None,
                    "quiet_region_end_seconds": None,
                    "silence_threshold_db": None,
                    "local_noise_floor_db": None,
                }
            else:
                existing_ms = quiet.existing_pause_samples * 1000.0 / sample_rate
                inserted_samples = max(
                    0,
                    round((target_ms - existing_ms) * sample_rate / 1000.0),
                )
                if inserted_samples:
                    method = "quiet_waveform_point"
                    status = "pause_inserted"
                    filler, selection = room_tone_allocator.allocate(
                        frame_count=inserted_samples,
                        reference_sample=quiet.source_sample,
                    )
                    if selection is None:
                        raise PausePlanError(
                            "positive pause allocation returned no source"
                        )
                    internal_insertions[previous_clip_index].append(
                        (
                            quiet.source_sample
                            - int(clip["final_source_start_sample"]),
                            filler,
                            transition_index,
                            selection,
                        )
                    )
                else:
                    method = "none_existing_pause"
                    status = "existing_pause_satisfies_target"
                record = {
                    **common,
                    "boundary_location": "inside_continuous_source_clip",
                    "estimated_existing_pause_ms": existing_ms,
                    "inserted_pause_ms": inserted_samples * 1000.0 / sample_rate,
                    "insertion_method": method,
                    "status": status,
                    "source_insertion_seconds": quiet.source_sample / sample_rate,
                    "quiet_region_start_seconds": (
                        quiet.quiet_start_sample / sample_rate
                    ),
                    "quiet_region_end_seconds": quiet.quiet_end_sample / sample_rate,
                    "silence_threshold_db": quiet.silence_threshold_db,
                    "local_noise_floor_db": quiet.local_noise_floor_db,
                    "pause_fill_method": (
                        "verified_source_room_tone" if inserted_samples else "none"
                    ),
                    "room_tone_source_start_seconds": (
                        selection.start_sample / sample_rate
                        if inserted_samples
                        else None
                    ),
                    "room_tone_source_end_seconds": (
                        selection.end_sample / sample_rate if inserted_samples else None
                    ),
                    "room_tone_source_ranges_seconds": (
                        [
                            {
                                "start": start / sample_rate,
                                "end": end / sample_rate,
                            }
                            for start, end in selection.source_ranges
                        ]
                        if inserted_samples
                        else []
                    ),
                }
        else:
            if next_clip_index != previous_clip_index + 1:
                raise PausePlanError(
                    "a thought transition skips an unexpected rendered clip"
                )
            join_index = previous_clip_index
            join = clip_join_manifest[join_index]
            if (
                join["left_thought_index"] != after
                or join["right_thought_index"] != before
            ):
                raise PausePlanError(
                    "classified thought transition does not match clip join"
                )
            inserted_ms = float(join["inserted_pause_ms"])
            record = {
                **common,
                "boundary_location": "between_non_contiguous_source_clips",
                "estimated_existing_pause_ms": float(
                    join["estimated_existing_pause_ms"]
                ),
                "retained_tail_ms": float(join["retained_tail_ms"]),
                "retained_head_ms": float(join["retained_head_ms"]),
                "inserted_pause_ms": inserted_ms,
                "estimated_total_pause_ms": float(join["estimated_total_pause_ms"]),
                "insertion_method": (
                    "non_contiguous_join" if inserted_ms else "none_existing_pause"
                ),
                "status": str(join["status"]),
                "source_insertion_seconds": None,
                "quiet_region_start_seconds": None,
                "quiet_region_end_seconds": None,
                "silence_threshold_db": None,
                "local_noise_floor_db": None,
                "pause_fill_method": str(join["pause_fill_method"]),
                "room_tone_source_start_seconds": (
                    join["room_tone_source_start_seconds"]
                ),
                "room_tone_source_end_seconds": (join["room_tone_source_end_seconds"]),
                "room_tone_source_ranges_seconds": (
                    join["room_tone_source_ranges_seconds"]
                ),
            }
        transition_manifest.append(record)

    modified_clips: list[np.ndarray] = []
    for clip_index, audio in enumerate(clip_audio):
        ordered = sorted(internal_insertions[clip_index], key=lambda item: item[0])
        insertions = [(local, filler) for local, filler, _, _ in ordered]
        modified_clips.append(_insert_audio_segments(audio, insertions=insertions))

    output_parts: list[np.ndarray] = []
    output_cursor = 0
    output_clips: list[dict[str, Any]] = []
    transition_by_index = {
        index: record for index, record in enumerate(transition_manifest)
    }
    for clip_index, audio in enumerate(modified_clips):
        if clip_index:
            pause_audio = join_audio[clip_index - 1]
            silence_count = len(pause_audio)
            join_start = output_cursor
            output_parts.append(pause_audio)
            output_cursor += silence_count
            join_record = clip_join_manifest[clip_index - 1]
            join_record["output_inserted_pause_start_seconds"] = (
                join_start / sample_rate
            )
            join_record["output_inserted_pause_end_seconds"] = (
                output_cursor / sample_rate
            )
            for record in transition_manifest:
                if (
                    record["boundary_location"] == "between_non_contiguous_source_clips"
                    and record["previous_clip_index"] == clip_index - 1
                    and record["next_clip_index"] == clip_index
                ):
                    record["output_pause_start_seconds"] = join_start / sample_rate
                    record["output_pause_end_seconds"] = output_cursor / sample_rate
        clip_output_start = output_cursor
        output_parts.append(audio)
        output_cursor += len(audio)
        clip_output_end = output_cursor

        inserted_before = 0
        for local_sample, filler, transition_index, _ in sorted(
            internal_insertions[clip_index],
            key=lambda item: item[0],
        ):
            count = len(filler)
            insertion_output = clip_output_start + local_sample + inserted_before
            record = transition_by_index[transition_index]
            record["output_pause_start_seconds"] = insertion_output / sample_rate
            record["output_pause_end_seconds"] = (
                insertion_output + count
            ) / sample_rate
            inserted_before += count
        output_clips.append(
            {
                **clips[clip_index],
                "semantic_output_start_sample": clip_output_start,
                "semantic_output_end_sample": clip_output_end,
                "semantic_output_start_seconds": clip_output_start / sample_rate,
                "semantic_output_end_seconds": clip_output_end / sample_rate,
                "semantic_frame_count": len(audio),
                "retained_leading_quiet_ms": (
                    clip_handles[clip_index].leading_samples * 1000.0 / sample_rate
                ),
                "retained_trailing_quiet_ms": (
                    clip_handles[clip_index].trailing_samples * 1000.0 / sample_rate
                ),
                "leading_quiet_method": (clip_handles[clip_index].leading_method),
                "trailing_quiet_method": (clip_handles[clip_index].trailing_method),
                "internal_semantic_pause_samples": (
                    len(audio) - len(clip_audio[clip_index])
                ),
                "semantic_source_end_sample": (
                    eof.new_end_sample
                    if clip_index == len(clips) - 1
                    else int(clips[clip_index]["final_source_end_sample"])
                ),
            }
        )

    rendered = np.concatenate(output_parts, axis=0)
    expected_frames = sum(len(audio) for audio in modified_clips) + sum(
        len(audio) for audio in join_audio
    )
    if len(rendered) != expected_frames:
        raise PausePlanError("semantic-pause duration does not equal clips plus pauses")
    sf.write(output_path, rendered, sample_rate, subtype="FLOAT")
    output_info = sf.info(output_path)
    if (
        int(output_info.frames) != expected_frames
        or int(output_info.samplerate) != sample_rate
        or int(output_info.channels) != channel_count
    ):
        raise PausePlanError("semantic-pause WAV has unexpected geometry")

    final_old_path: Path | None = None
    final_new_path: Path | None = None
    if write_debug_artifacts:
        final_old_path, final_new_path = _write_final_sentence_previews(
            output_dir=output_dir,
            source_audio=source_audio,
            sample_rate=sample_rate,
            words=words,
            thoughts=thoughts,
            old_end_sample=previous_final_end,
            new_end_sample=eof.new_end_sample,
        )
    final_boundary = {
        "last_word": words[int(final_clip["source_word_end"]) - 1].text,
        "last_word_id": int(final_clip["source_word_end"]) - 1,
        "raw_end_seconds": eof.raw_end_sample / sample_rate,
        "previous_refined_end_seconds": previous_final_end / sample_rate,
        "eof_search_start_seconds": eof.search_start_sample / sample_rate,
        "stable_silence_start_seconds": (
            eof.stable_silence_start_sample / sample_rate
            if eof.stable_silence_start_sample is not None
            else None
        ),
        "new_final_end_seconds": eof.new_end_sample / sample_rate,
        "end_extension_from_previous_ms": (
            (eof.new_end_sample - previous_final_end) * 1000.0 / sample_rate
        ),
        "boundary_method": eof.boundary_method,
        "fade_out_samples": eof.fade_out_samples,
        "local_noise_floor_db": eof.local_noise_floor_db,
        "silence_threshold_db": eof.silence_threshold_db,
        "old_final_sentence_wav": (
            str(final_old_path.resolve()) if final_old_path is not None else None
        ),
        "new_final_sentence_wav": (
            str(final_new_path.resolve()) if final_new_path is not None else None
        ),
    }
    inserted_count = sum(
        record["status"] == "pause_inserted"
        for record in transition_manifest
        if record["boundary_location"] == "inside_continuous_source_clip"
    ) + sum(float(record["inserted_pause_ms"]) > 0.0 for record in clip_join_manifest)
    unsafe_count = sum(
        record["status"] == "pause_not_inserted_no_safe_point"
        for record in transition_manifest
    )
    manifest = {
        "schema_version": 2,
        "renderer": "streaming_plan_semantic_pause_render_v2",
        "source_audio": str(audio_path),
        "source_audio_sha256": sha256_file(audio_path),
        "streaming_plan": str(plan_path),
        "streaming_plan_sha256": sha256_file(plan_path),
        "input_render_manifest": str(render_manifest_path),
        "input_render_manifest_sha256": sha256_file(render_manifest_path),
        "pause_plan": str(pause_plan_path),
        "pause_plan_sha256": sha256_file(pause_plan_path),
        "previous_preview_wav": str(previous_wav_path),
        "previous_preview_wav_sha256": sha256_file(previous_wav_path),
        "rough_cut_with_semantic_pauses_wav": str(output_path),
        "rough_cut_with_semantic_pauses_wav_sha256": sha256_file(output_path),
        "source_sample_rate": sample_rate,
        "source_channel_count": channel_count,
        "source_frame_count": total_samples,
        "semantic_pause_output_frame_count": expected_frames,
        "semantic_pause_output_duration_seconds": expected_frames / sample_rate,
        "debug_artifacts_written": write_debug_artifacts,
        "configuration": {
            "pause_targets_ms": targets,
            "input_preview_join_ms": configured_join_ms,
            "non_contiguous_join_policy": ("target_total_minus_retained_quiet_handles"),
            "minimum_quiet_insertion_ms": min_quiet_insertion_ms,
            "quiet_search_margin_ms": quiet_search_margin_ms,
            "eof_min_extension_ms": eof_min_extension_ms,
            "eof_trailing_silence_ms": eof_trailing_silence_ms,
            "output_subtype": "FLOAT",
        },
        "thought_transition_count": len(transition_manifest),
        "pause_type_counts": {
            pause_type: sum(
                record["pause_type"] == pause_type for record in transition_manifest
            )
            for pause_type in PAUSE_TYPES
        },
        "pauses_inserted": inserted_count,
        "unsafe_boundaries_skipped": unsafe_count,
        "transitions": transition_manifest,
        "clip_joins": clip_join_manifest,
        "final_boundary": final_boundary,
        "clips": output_clips,
    }
    write_json(manifest_path, manifest)
    return manifest
