#!/usr/bin/env python3
"""Debug-only compatibility preview for selected clip starts.

Production resolves both sides of every discontinuity symmetrically in
:mod:`voicecut.final_render` and slices the canonical source only once. This
module remains solely for inspecting legacy preview behavior.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from .common import read_json, sha256_file, write_json
from .hard_align import (
    CONTEXT_WORDS_PER_SIDE,
    CROP_CONTEXT_MS,
    DEFAULT_ALIGNMENT_PYTHON,
    MAX_BOUNDARY_SHIFT_MS,
    _aligned_character_end,
    _aligned_character_start,
    _flatten_aligned_words,
    _flatten_character_word_groups,
    _run_alignment_worker,
)
from .rough_render import (
    RoughRenderError,
    flatten_selected_ranges,
    load_plan_words,
    merge_adjacent_ranges,
    timestamp_to_sample,
)
from .trailing_refine import RMS_FRAME_MS, _local_threshold, rms_envelope_db


LEADING_QUIET_MS = 30.0
DENSE_BOUNDARY_FADE_MS = 2.0
LOW_AMPLITUDE_SNAP_MS = 5.0
WAVEFORM_STABLE_QUIET_MS = 70.0
WAVEFORM_SUSTAINED_ACTIVE_MS = 50.0
WAVEFORM_SEARCH_LEFT_MS = 200.0
WAVEFORM_SEARCH_RIGHT_MS = 750.0


@dataclass(frozen=True)
class LeadingWaveformCandidate:
    quiet_start_sample: int
    active_onset_sample: int
    start_sample: int
    search_start_sample: int
    search_end_sample: int
    quiet_duration_samples: int
    local_noise_floor_db: float
    silence_threshold_db: float


@dataclass(frozen=True)
class LeadingWaveformAssessment:
    candidate: LeadingWaveformCandidate | None
    error: str


@dataclass(frozen=True)
class LeadingBoundaryDecision:
    status: str
    error: str | None
    alignment_granularity: str | None
    omitted_end_seconds: float | None
    kept_start_seconds: float | None
    start_seconds: float
    shift_ms: float
    retained_leading_quiet_ms: float | None
    dense_boundary: bool | None
    fade_in_samples: int
    snap_method: str | None
    low_amplitude_threshold_db: float | None
    waveform_quiet_start_seconds: float | None
    waveform_active_onset_seconds: float | None
    waveform_quiet_duration_ms: float | None
    waveform_search_start_seconds: float | None
    waveform_search_end_seconds: float | None


def _finite_number(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive_character_boundary_score(
    group: Sequence[dict[str, Any]],
    *,
    boundary: str,
) -> float | None:
    scored = [
        score
        for character in group
        if _finite_number(character.get(boundary)) is not None
        and (score := _finite_number(character.get("score"))) is not None
    ]
    if not scored:
        return None
    score = scored[-1] if boundary == "end" else scored[0]
    return score if score > 0.0 else None


def leading_alignment_positions(
    *,
    job: dict[str, Any],
    worker_job: dict[str, Any],
) -> tuple[float, float, str]:
    """Map the omitted/kept pair by local sequence position.

    Text lookup is intentionally forbidden here because repeated words are
    common around narration retakes.
    """

    if worker_job.get("error"):
        raise ValueError(str(worker_job["error"]))
    aligned = worker_job.get("aligned")
    if not isinstance(aligned, dict):
        raise ValueError("worker did not return an aligned result")
    local_words = job.get("local_words")
    if not isinstance(local_words, list) or not local_words:
        raise ValueError("alignment job has no local word ledger")
    omitted_index = job.get("omitted_local_index")
    kept_index = job.get("kept_local_index")
    if type(omitted_index) is not int or type(kept_index) is not int:
        raise ValueError("alignment job has invalid local boundary indices")

    aligned_words = _flatten_aligned_words(aligned)
    if len(aligned_words) != len(local_words):
        raise ValueError(
            "aligned word count does not match local transcript positions: "
            f"{len(aligned_words)} != {len(local_words)}"
        )
    if not 0 <= omitted_index < kept_index < len(aligned_words):
        raise ValueError("local boundary indices do not fit aligned words")

    character_groups = _flatten_character_word_groups(aligned)
    if len(character_groups) == len(local_words):
        omitted_group = character_groups[omitted_index]
        kept_group = character_groups[kept_index]
        omitted_end = _aligned_character_end(omitted_group)
        kept_start = _aligned_character_start(kept_group)
        omitted_score = _positive_character_boundary_score(
            omitted_group,
            boundary="end",
        )
        kept_score = _positive_character_boundary_score(
            kept_group,
            boundary="start",
        )
        if (
            omitted_end is not None
            and kept_start is not None
            and omitted_score is not None
            and kept_score is not None
        ):
            return omitted_end, kept_start, "characters"

    omitted_word = aligned_words[omitted_index]
    kept_word = aligned_words[kept_index]
    omitted_end = _finite_number(omitted_word.get("end"))
    kept_start = _finite_number(kept_word.get("start"))
    omitted_score = _finite_number(omitted_word.get("score"))
    kept_score = _finite_number(kept_word.get("score"))
    if (
        omitted_end is None
        or kept_start is None
        or omitted_score is None
        or kept_score is None
        or omitted_score <= 0.0
        or kept_score <= 0.0
    ):
        raise ValueError(
            "omitted or kept boundary word was not aligned with positive confidence"
        )
    return omitted_end, kept_start, "words"


def find_leading_waveform_candidate(
    mono: np.ndarray,
    *,
    job: dict[str, Any],
    sample_rate: int,
    leading_quiet_ms: float = LEADING_QUIET_MS,
    stable_quiet_ms: float = WAVEFORM_STABLE_QUIET_MS,
    sustained_active_ms: float = WAVEFORM_SUSTAINED_ACTIVE_MS,
    search_left_ms: float = WAVEFORM_SEARCH_LEFT_MS,
    search_right_ms: float = WAVEFORM_SEARCH_RIGHT_MS,
) -> LeadingWaveformCandidate | None:
    """Find a stable quiet run containing the old start, then speech onset.

    Requiring the existing start to lie inside the quiet run ties the
    waveform decision to the already selected semantic boundary.  It avoids
    accidentally choosing an earlier pause elsewhere in the local crop.
    """

    if mono.ndim != 1 or not len(mono):
        raise ValueError("mono analysis waveform must be non-empty")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    for name, value in (
        ("leading_quiet_ms", leading_quiet_ms),
        ("stable_quiet_ms", stable_quiet_ms),
        ("sustained_active_ms", sustained_active_ms),
        ("search_left_ms", search_left_ms),
        ("search_right_ms", search_right_ms),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    total_samples = len(mono)
    crop_start = int(job["crop_start_sample"])
    crop_end = int(job["crop_end_sample"])
    previous_start = int(job["previous_source_start_sample"])
    first_word_start = timestamp_to_sample(
        float(job["first_kept_word_start_seconds"]),
        sample_rate=sample_rate,
        total_samples=total_samples,
        rounding="nearest",
    )
    first_word_end = timestamp_to_sample(
        float(job["first_kept_word_end_seconds"]),
        sample_rate=sample_rate,
        total_samples=total_samples,
        rounding="ceil",
    )
    left = round(search_left_ms * sample_rate / 1000.0)
    right = round(search_right_ms * sample_rate / 1000.0)
    search_start = max(
        crop_start,
        min(previous_start, first_word_start) - left,
    )
    search_end = min(
        crop_end,
        max(
            first_word_end,
            previous_start + right,
            first_word_start + right,
        ),
    )
    if search_end <= search_start:
        return None

    noise_floor_db, threshold_db = _local_threshold(
        mono,
        raw_end_sample=previous_start,
        sample_rate=sample_rate,
    )
    frame_starts, levels_db = rms_envelope_db(
        mono,
        start_sample=search_start,
        end_sample=search_end,
        sample_rate=sample_rate,
    )
    if not len(frame_starts):
        return None
    frame_samples = max(
        1,
        round(RMS_FRAME_MS * sample_rate / 1000.0),
    )
    stable_quiet_samples = max(
        frame_samples,
        round(stable_quiet_ms * sample_rate / 1000.0),
    )
    sustained_active_samples = max(
        frame_samples,
        round(sustained_active_ms * sample_rate / 1000.0),
    )
    target_quiet_samples = max(
        0,
        round(leading_quiet_ms * sample_rate / 1000.0),
    )

    quiet_start: int | None = None
    for index, (frame_start_value, level_db) in enumerate(
        zip(frame_starts, levels_db, strict=True)
    ):
        frame_start = int(frame_start_value)
        if level_db < threshold_db:
            if quiet_start is None:
                quiet_start = frame_start
            continue
        if quiet_start is None:
            continue

        active_onset = frame_start
        quiet_duration = active_onset - quiet_start
        old_start_is_in_quiet = quiet_start <= previous_start <= active_onset
        if quiet_duration < stable_quiet_samples or not old_start_is_in_quiet:
            quiet_start = None
            continue

        active_end = active_onset
        for active_index in range(index, len(frame_starts)):
            candidate_start = int(frame_starts[active_index])
            if levels_db[active_index] < threshold_db:
                break
            active_end = candidate_start + frame_samples
            if active_end - active_onset >= sustained_active_samples:
                natural_start = max(
                    quiet_start,
                    active_onset - target_quiet_samples,
                )
                start_sample, _, _ = _snap_leading_start(
                    mono,
                    target_sample=natural_start,
                    interval_start_sample=quiet_start,
                    interval_end_sample=active_onset,
                    sample_rate=sample_rate,
                )
                return LeadingWaveformCandidate(
                    quiet_start_sample=quiet_start,
                    active_onset_sample=active_onset,
                    start_sample=start_sample,
                    search_start_sample=search_start,
                    search_end_sample=search_end,
                    quiet_duration_samples=quiet_duration,
                    local_noise_floor_db=noise_floor_db,
                    silence_threshold_db=threshold_db,
                )
        quiet_start = None
    return None


def assess_leading_waveform(
    mono: np.ndarray,
    *,
    job: dict[str, Any],
    sample_rate: int,
    leading_quiet_ms: float = LEADING_QUIET_MS,
) -> LeadingWaveformAssessment:
    """Run the deterministic waveform preclassification exactly once."""

    try:
        candidate = find_leading_waveform_candidate(
            mono,
            job=job,
            sample_rate=sample_rate,
            leading_quiet_ms=leading_quiet_ms,
        )
    except Exception as error:
        return LeadingWaveformAssessment(
            candidate=None,
            error=f"{type(error).__name__}: {error}",
        )
    return LeadingWaveformAssessment(
        candidate=candidate,
        error=(
            ""
            if candidate is not None
            else "no stable quiet-to-sustained-speech transition found"
        ),
    )


def _snap_leading_start(
    mono: np.ndarray,
    *,
    target_sample: int,
    interval_start_sample: int,
    interval_end_sample: int,
    sample_rate: int,
) -> tuple[int, str, float]:
    """Snap near the 30 ms handle while staying inside the aligned gap."""

    _, threshold_db = _local_threshold(
        mono,
        raw_end_sample=interval_end_sample,
        sample_rate=sample_rate,
    )
    if interval_end_sample <= interval_start_sample:
        return interval_end_sample, "touching_words", threshold_db
    radius = max(
        0,
        round(LOW_AMPLITUDE_SNAP_MS * sample_rate / 1000.0),
    )
    search_start = max(interval_start_sample, target_sample - radius)
    search_end = min(interval_end_sample, target_sample + radius)
    if search_end < search_start:
        return target_sample, "leading_quiet_target", threshold_db
    candidate = mono[search_start : search_end + 1]
    if not len(candidate):
        return target_sample, "leading_quiet_target", threshold_db
    snapped = search_start + int(np.argmin(np.abs(candidate)))
    return snapped, "low_amplitude_near_leading_target", threshold_db


def decide_leading_boundary(
    *,
    job: dict[str, Any],
    worker_job: dict[str, Any],
    mono: np.ndarray,
    sample_rate: int,
    leading_quiet_ms: float = LEADING_QUIET_MS,
    dense_boundary_fade_ms: float = DENSE_BOUNDARY_FADE_MS,
    max_shift_ms: float = MAX_BOUNDARY_SHIFT_MS,
    waveform_assessment: LeadingWaveformAssessment | None = None,
) -> LeadingBoundaryDecision:
    """Choose a waveform-safe start, falling back to forced alignment."""

    previous_start_sample = int(job["previous_source_start_sample"])
    previous_fade_in_samples = int(job["previous_fade_in_samples"])
    crop_start = float(job["crop_start_seconds"])
    crop_duration = float(job["crop_duration_seconds"])
    crop_end = crop_start + crop_duration
    total_samples = len(mono)
    if waveform_assessment is None:
        waveform_assessment = assess_leading_waveform(
            mono,
            job=job,
            sample_rate=sample_rate,
            leading_quiet_ms=leading_quiet_ms,
        )
    waveform = waveform_assessment.candidate
    waveform_error = waveform_assessment.error
    if waveform is not None:
        quiet_samples = waveform.active_onset_sample - waveform.start_sample
        fade_in_samples = min(
            previous_fade_in_samples,
            quiet_samples,
        )
        return LeadingBoundaryDecision(
            status="leading_waveform_silence",
            error=None,
            alignment_granularity=None,
            omitted_end_seconds=None,
            kept_start_seconds=None,
            start_seconds=waveform.start_sample / sample_rate,
            shift_ms=(
                (waveform.start_sample - previous_start_sample) * 1000.0 / sample_rate
            ),
            retained_leading_quiet_ms=(quiet_samples * 1000.0 / sample_rate),
            dense_boundary=False,
            fade_in_samples=fade_in_samples,
            snap_method="low_amplitude_before_sustained_onset",
            low_amplitude_threshold_db=waveform.silence_threshold_db,
            waveform_quiet_start_seconds=(waveform.quiet_start_sample / sample_rate),
            waveform_active_onset_seconds=(waveform.active_onset_sample / sample_rate),
            waveform_quiet_duration_ms=(
                waveform.quiet_duration_samples * 1000.0 / sample_rate
            ),
            waveform_search_start_seconds=(waveform.search_start_sample / sample_rate),
            waveform_search_end_seconds=(waveform.search_end_sample / sample_rate),
        )
    try:
        omitted_relative, kept_relative, granularity = leading_alignment_positions(
            job=job,
            worker_job=worker_job,
        )
        if not (
            0.0 <= omitted_relative <= crop_duration
            and 0.0 <= kept_relative <= crop_duration
        ):
            raise ValueError("aligned timestamps fall outside the local crop")
        omitted_end = crop_start + omitted_relative
        kept_start = crop_start + kept_relative
        if not (
            crop_start <= omitted_end <= crop_end
            and crop_start <= kept_start <= crop_end
        ):
            raise ValueError("absolute aligned timestamps fall outside crop")

        omitted_end_sample = timestamp_to_sample(
            omitted_end,
            sample_rate=sample_rate,
            total_samples=total_samples,
            rounding="ceil",
        )
        kept_start_sample = timestamp_to_sample(
            kept_start,
            sample_rate=sample_rate,
            total_samples=total_samples,
            rounding="floor",
        )
        if kept_start_sample < omitted_end_sample:
            raise ValueError("aligned omitted-word end occurs after kept-word start")

        maximum_shift_samples = math.floor(max_shift_ms * sample_rate / 1000.0 + 1e-9)
        allowed_start = max(
            omitted_end_sample,
            previous_start_sample - maximum_shift_samples,
        )
        allowed_end = min(
            kept_start_sample,
            previous_start_sample + maximum_shift_samples,
        )
        if allowed_end < allowed_start:
            raise ValueError(
                "aligned leading interval lies outside the permitted "
                f"{max_shift_ms:.3f} ms boundary shift"
            )

        desired_quiet_samples = max(
            0,
            round(leading_quiet_ms * sample_rate / 1000.0),
        )
        natural_target = max(
            omitted_end_sample,
            kept_start_sample - desired_quiet_samples,
        )
        target = max(allowed_start, min(allowed_end, natural_target))
        start_sample, snap_method, threshold_db = _snap_leading_start(
            mono,
            target_sample=target,
            interval_start_sample=allowed_start,
            interval_end_sample=allowed_end,
            sample_rate=sample_rate,
        )
        if not allowed_start <= start_sample <= allowed_end:
            raise ValueError("chosen start leaves the aligned interword interval")
        if start_sample < omitted_end_sample:
            raise ValueError("chosen start includes the aligned omitted word")
        if start_sample > kept_start_sample:
            raise ValueError("chosen start falls inside the selected word")

        quiet_samples = kept_start_sample - start_sample
        dense_boundary = quiet_samples < desired_quiet_samples
        dense_fade_limit = round(dense_boundary_fade_ms * sample_rate / 1000.0)
        fade_in_samples = min(
            previous_fade_in_samples,
            quiet_samples,
            dense_fade_limit if dense_boundary else previous_fade_in_samples,
        )
        shift_ms = (start_sample - previous_start_sample) * 1000.0 / sample_rate
        if abs(shift_ms) > max_shift_ms + 1e-6:
            raise ValueError(
                f"leading shift {shift_ms:.3f} ms exceeds {max_shift_ms:.3f} ms"
            )
        return LeadingBoundaryDecision(
            status="leading_forced_alignment",
            error=None,
            alignment_granularity=granularity,
            omitted_end_seconds=omitted_end,
            kept_start_seconds=kept_start,
            start_seconds=start_sample / sample_rate,
            shift_ms=shift_ms,
            retained_leading_quiet_ms=(quiet_samples * 1000.0 / sample_rate),
            dense_boundary=dense_boundary,
            fade_in_samples=fade_in_samples,
            snap_method=snap_method,
            low_amplitude_threshold_db=threshold_db,
            waveform_quiet_start_seconds=None,
            waveform_active_onset_seconds=None,
            waveform_quiet_duration_ms=None,
            waveform_search_start_seconds=None,
            waveform_search_end_seconds=None,
        )
    except Exception as error:
        return LeadingBoundaryDecision(
            status="leading_forced_alignment_failed",
            error=(
                f"waveform: {waveform_error}; forced alignment: "
                f"{type(error).__name__}: {error}"
            ),
            alignment_granularity=None,
            omitted_end_seconds=None,
            kept_start_seconds=None,
            start_seconds=previous_start_sample / sample_rate,
            shift_ms=0.0,
            retained_leading_quiet_ms=None,
            dense_boundary=None,
            fade_in_samples=previous_fade_in_samples,
            snap_method=None,
            low_amplitude_threshold_db=None,
            waveform_quiet_start_seconds=None,
            waveform_active_onset_seconds=None,
            waveform_quiet_duration_ms=None,
            waveform_search_start_seconds=None,
            waveform_search_end_seconds=None,
        )


def _apply_edge_fades(
    samples: np.ndarray,
    *,
    fade_in_samples: int,
    fade_out_samples: int,
) -> np.ndarray:
    rendered = np.array(samples, dtype=np.float32, copy=True)
    fade_in = min(max(0, fade_in_samples), len(rendered) // 2)
    if fade_in:
        rendered[:fade_in] *= np.linspace(
            0.0,
            1.0,
            fade_in,
            endpoint=True,
            dtype=np.float32,
        )[:, None]
    fade_out = min(max(0, fade_out_samples), len(rendered) // 2)
    if fade_out:
        rendered[-fade_out:] *= np.linspace(
            1.0,
            0.0,
            fade_out,
            endpoint=True,
            dtype=np.float32,
        )[:, None]
    return rendered


def _prepare_leading_alignment_jobs(
    *,
    eligible_clips: Sequence[dict[str, Any]],
    words: Sequence[Any],
    source_audio: np.ndarray,
    previous_audio: np.ndarray,
    sample_rate: int,
    debug_root: Path,
    crop_context_ms: float,
) -> list[dict[str, Any]]:
    total_samples = len(source_audio)
    context_samples = round(crop_context_ms * sample_rate / 1000.0)
    jobs: list[dict[str, Any]] = []
    for clip in eligible_clips:
        clip_index = int(clip["clip_index"])
        first_kept_id = int(clip["source_word_start"])
        previous_omitted_id = first_kept_id - 1
        if not 0 <= previous_omitted_id < first_kept_id < len(words):
            raise RoughRenderError(
                f"clip {clip_index} has no preceding omitted source word"
            )

        requested_context_start_id = max(
            0,
            previous_omitted_id - CONTEXT_WORDS_PER_SIDE,
        )
        requested_context_end_id = min(
            len(words),
            first_kept_id + CONTEXT_WORDS_PER_SIDE + 1,
        )
        crop_start = max(
            0,
            math.floor(words[requested_context_start_id].start * sample_rate)
            - context_samples,
        )
        crop_end = min(
            total_samples,
            math.ceil(words[requested_context_end_id - 1].end * sample_rate)
            + context_samples,
        )

        # The complete local transcript must describe words intersecting the
        # requested acoustic handles, not just the nominal context words.
        context_start_id = requested_context_start_id
        while context_start_id > 0:
            previous_word = words[context_start_id - 1]
            if math.ceil(previous_word.end * sample_rate) <= crop_start:
                break
            context_start_id -= 1
            crop_start = min(
                crop_start,
                math.floor(previous_word.start * sample_rate),
            )
        context_end_id = requested_context_end_id
        while context_end_id < len(words):
            following_word = words[context_end_id]
            if math.floor(following_word.start * sample_rate) >= crop_end:
                break
            crop_end = max(
                crop_end,
                math.ceil(following_word.end * sample_rate),
            )
            context_end_id += 1
        if crop_end <= crop_start:
            raise RoughRenderError(
                f"clip {clip_index} produced an empty leading context crop"
            )

        output_start = int(clip["final_output_start_sample"])
        output_end = int(clip["final_output_end_sample"])
        if not 0 <= output_start < output_end <= len(previous_audio):
            raise RoughRenderError(
                f"clip {clip_index} has invalid previous output geometry"
            )
        old_clip = previous_audio[output_start:output_end]
        if len(old_clip) != int(clip["final_frame_count"]):
            raise RoughRenderError(
                f"clip {clip_index} frame count differs from previous preview"
            )

        boundary_dir = debug_root / f"clip_{clip_index:03d}_start"
        context_path = boundary_dir / "context.wav"

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
                "clip_index": clip_index,
                "previous_omitted_word_id": previous_omitted_id,
                "first_kept_word_id": first_kept_id,
                "previous_omitted_word": words[previous_omitted_id].text,
                "first_kept_word": words[first_kept_id].text,
                "previous_omitted_word_start_seconds": words[previous_omitted_id].start,
                "previous_omitted_word_end_seconds": words[previous_omitted_id].end,
                "first_kept_word_start_seconds": words[first_kept_id].start,
                "first_kept_word_end_seconds": words[first_kept_id].end,
                "requested_context_start_word_id": requested_context_start_id,
                "requested_context_end_word_id": requested_context_end_id,
                "context_start_word_id": context_start_id,
                "context_end_word_id": context_end_id,
                "omitted_local_index": (previous_omitted_id - context_start_id),
                "kept_local_index": first_kept_id - context_start_id,
                "local_words": local_words,
                "local_source_text": " ".join(word["text"] for word in local_words),
                "crop_start_sample": crop_start,
                "crop_end_sample": crop_end,
                "crop_start_seconds": crop_start / sample_rate,
                "crop_end_seconds": crop_end / sample_rate,
                "crop_duration_seconds": ((crop_end - crop_start) / sample_rate),
                "crop_wav": str(context_path.resolve()),
                "previous_source_start_sample": int(clip["final_source_start_sample"]),
                "previous_source_start_seconds": (
                    int(clip["final_source_start_sample"]) / sample_rate
                ),
                "previous_fade_in_samples": int(clip["final_fade_in_samples"]),
                "previous_clip_wav": None,
            }
        )
    return jobs


def _save_leading_alignment_plot(
    *,
    path: Path,
    mono: np.ndarray,
    sample_rate: int,
    job: dict[str, Any],
    decision: LeadingBoundaryDecision,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    crop_start = int(job["crop_start_sample"])
    crop_end = int(job["crop_end_sample"])
    time = np.arange(crop_start, crop_end) / sample_rate
    figure = Figure(figsize=(12, 4.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots(1, 1)
    axis.plot(
        time,
        mono[crop_start:crop_end],
        color="#345995",
        linewidth=0.65,
        label="mono waveform",
    )
    lines: list[tuple[float | None, str, str, str]] = [
        (
            float(job["previous_source_start_seconds"]),
            "#e45756",
            "--",
            "previous clip start",
        ),
        (
            decision.omitted_end_seconds,
            "#f3a712",
            "-",
            "omitted-word aligned end",
        ),
        (
            decision.kept_start_seconds,
            "#54a24b",
            "-",
            "kept-word aligned start",
        ),
        (
            decision.start_seconds,
            "#7a5195",
            ":",
            "chosen clip start",
        ),
        (
            decision.waveform_quiet_start_seconds,
            "#72b7b2",
            "--",
            "stable quiet start",
        ),
        (
            decision.waveform_active_onset_seconds,
            "#eeca3b",
            "-",
            "sustained active onset",
        ),
    ]
    for position, color, style, label in lines:
        if position is not None:
            axis.axvline(
                position,
                color=color,
                linestyle=style,
                linewidth=1.5,
                label=label,
            )
    axis.set_xlabel("absolute source time (seconds)")
    axis.set_ylabel("amplitude")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=8)
    axis.set_title(
        f"Clip {int(job['clip_index']):03d} start: "
        f"{job['previous_omitted_word']} → {job['first_kept_word']} "
        f"({decision.status})"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)


def _validate_inputs(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path]:
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise RoughRenderError("forced-aligned manifest root must be an object")
    if manifest.get("renderer") != ("streaming_plan_hard_boundary_alignment_v1"):
        raise RoughRenderError(
            "leading alignment requires a hard-boundary alignment v1 manifest"
        )
    audio_path = Path(str(manifest.get("source_audio", ""))).resolve()
    plan_path = Path(str(manifest.get("streaming_plan", ""))).resolve()
    previous_wav = Path(str(manifest.get("hard_boundary_aligned_wav", ""))).resolve()
    for path in (audio_path, plan_path, previous_wav):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(audio_path) != manifest.get("source_audio_sha256"):
        raise RoughRenderError("source audio changed after hard alignment")
    if sha256_file(plan_path) != manifest.get("streaming_plan_sha256"):
        raise RoughRenderError("streaming plan changed after hard alignment")
    if sha256_file(previous_wav) != manifest.get("hard_boundary_aligned_wav_sha256"):
        raise RoughRenderError("hard-aligned preview changed on disk")

    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise RoughRenderError("streaming plan root must be an object")
    words = load_plan_words(plan)
    merged_ranges = merge_adjacent_ranges(
        flatten_selected_ranges(plan, word_count=len(words))
    )
    clips = manifest.get("clips")
    if not isinstance(clips, list) or len(clips) != len(merged_ranges):
        raise RoughRenderError("hard-aligned clips differ from frozen semantic plan")
    for clip, source_range in zip(clips, merged_ranges, strict=True):
        if (
            clip.get("source_word_start") != source_range.start_word_id
            or clip.get("source_word_end") != source_range.end_word_id
        ):
            raise RoughRenderError(
                "hard-aligned clip ranges differ from frozen semantic plan"
            )
    return manifest, plan, audio_path, plan_path, previous_wav


def render_leading_aligned_preview(
    *,
    aligned_manifest_path: Path,
    output_dir: Path,
    alignment_python: Path = DEFAULT_ALIGNMENT_PYTHON,
    language: str = "en",
    crop_context_ms: float = CROP_CONTEXT_MS,
    leading_quiet_ms: float = LEADING_QUIET_MS,
    dense_boundary_fade_ms: float = DENSE_BOUNDARY_FADE_MS,
    max_shift_ms: float = MAX_BOUNDARY_SHIFT_MS,
    alignment_payload: dict[str, Any] | None = None,
    write_debug_artifacts: bool = True,
) -> dict[str, Any]:
    """Resolve every selected clip start that follows omitted source words."""

    if language != "en":
        raise ValueError("this experiment requires the English align model")
    for name, value in (
        ("crop_context_ms", crop_context_ms),
        ("leading_quiet_ms", leading_quiet_ms),
        ("dense_boundary_fade_ms", dense_boundary_fade_ms),
        ("max_shift_ms", max_shift_ms),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if dense_boundary_fade_ms > 2.0:
        raise ValueError("dense_boundary_fade_ms must not exceed 2 ms")

    forced_manifest_path = aligned_manifest_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"output directory must be empty for leading alignment: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_root = output_dir / "leading_alignment_debug"
    clips_root = output_dir / "clips"
    if write_debug_artifacts:
        clips_root.mkdir()

    previous, plan, audio_path, plan_path, previous_wav_path = _validate_inputs(
        forced_manifest_path
    )
    words = load_plan_words(plan)
    source_audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )
    previous_audio, previous_rate = sf.read(
        previous_wav_path,
        dtype="float32",
        always_2d=True,
    )
    sample_rate = int(sample_rate)
    total_samples, channel_count = source_audio.shape
    expected_previous_frames = int(
        previous["hard_boundary_aligned_expected_output_frame_count"]
    )
    if (
        int(previous_rate) != sample_rate
        or previous_audio.shape[1] != channel_count
        or len(previous_audio) != expected_previous_frames
    ):
        raise RoughRenderError("hard-aligned preview has unexpected geometry")
    if (
        sample_rate != int(previous["source_sample_rate"])
        or total_samples != int(previous["source_frame_count"])
        or channel_count != int(previous["source_channel_count"])
    ):
        raise RoughRenderError("source audio geometry changed")
    mono = np.mean(source_audio, axis=1, dtype=np.float64).astype(np.float32)

    previous_clips = previous["clips"]
    eligible_clips = [
        clip for clip in previous_clips if int(clip["source_word_start"]) > 0
    ]
    boundary_jobs = _prepare_leading_alignment_jobs(
        eligible_clips=eligible_clips,
        words=words,
        source_audio=source_audio,
        previous_audio=previous_audio,
        sample_rate=sample_rate,
        debug_root=debug_root,
        crop_context_ms=crop_context_ms,
    )
    waveform_assessment_by_clip = {
        int(job["clip_index"]): assess_leading_waveform(
            mono,
            job=job,
            sample_rate=sample_rate,
            leading_quiet_ms=leading_quiet_ms,
        )
        for job in boundary_jobs
    }
    alignment_jobs = [
        job
        for job in boundary_jobs
        if waveform_assessment_by_clip[int(job["clip_index"])].candidate is None
    ]
    alignment_clip_indices = {int(job["clip_index"]) for job in alignment_jobs}
    for job in boundary_jobs:
        clip_index = int(job["clip_index"])
        needs_alignment_context = clip_index in alignment_clip_indices
        if needs_alignment_context or write_debug_artifacts:
            boundary_dir = debug_root / f"clip_{clip_index:03d}_start"
            boundary_dir.mkdir(parents=True, exist_ok=False)
            context_path = boundary_dir / "context.wav"
            sf.write(
                context_path,
                source_audio[
                    int(job["crop_start_sample"]) : int(job["crop_end_sample"])
                ],
                sample_rate,
                subtype="FLOAT",
            )
            job["crop_wav"] = str(context_path.resolve())
            if write_debug_artifacts:
                output_start = int(
                    previous_clips[clip_index]["final_output_start_sample"]
                )
                output_end = int(previous_clips[clip_index]["final_output_end_sample"])
                old_clip_path = boundary_dir / "old_clip.wav"
                sf.write(
                    old_clip_path,
                    previous_audio[output_start:output_end],
                    sample_rate,
                    subtype="FLOAT",
                )
                job["previous_clip_wav"] = str(old_clip_path.resolve())
        else:
            job["crop_wav"] = None
    boundary_jobs_path = output_dir / "leading_boundary_jobs.json"
    write_json(
        boundary_jobs_path,
        {
            "schema_version": 1,
            "source_audio": str(audio_path),
            "source_audio_sha256": previous["source_audio_sha256"],
            "routing": "waveform_silence_then_whisperx",
            "jobs": [
                {
                    **job,
                    "waveform_assessment": asdict(
                        waveform_assessment_by_clip[int(job["clip_index"])]
                    ),
                    "routed_to": (
                        "waveform"
                        if waveform_assessment_by_clip[int(job["clip_index"])].candidate
                        is not None
                        else "whisperx"
                    ),
                }
                for job in boundary_jobs
            ],
        },
    )
    jobs_path = output_dir / "leading_alignment_jobs.json"
    write_json(
        jobs_path,
        {
            "schema_version": 1,
            "source_audio": str(audio_path),
            "source_audio_sha256": previous["source_audio_sha256"],
            "language": language,
            "device": "cpu",
            "jobs": alignment_jobs,
        },
    )
    worker_result_path = output_dir / "leading_alignment_worker_result.json"
    if not alignment_jobs:
        worker_result = {
            "schema_version": 1,
            "backend": "whisperx_alignment",
            "language": language,
            "device": "cpu",
            "model_call_skipped": (
                "every leading boundary was resolved from stable waveform "
                "silence; WhisperX was not loaded"
            ),
            "jobs": [],
        }
        write_json(worker_result_path, worker_result)
    elif alignment_payload is None:
        worker_result = _run_alignment_worker(
            jobs_path=jobs_path,
            result_path=worker_result_path,
            alignment_python=alignment_python,
            log_path=output_dir / "leading_alignment_worker.log",
            language=language,
        )
    else:
        worker_result = alignment_payload
        write_json(worker_result_path, worker_result)
    worker_jobs = worker_result.get("jobs")
    if not isinstance(worker_jobs, list):
        raise RoughRenderError("alignment worker returned no jobs list")
    worker_by_clip = {
        int(item["clip_index"]): item
        for item in worker_jobs
        if isinstance(item, dict) and type(item.get("clip_index")) is int
    }
    job_by_clip = {int(job["clip_index"]): job for job in boundary_jobs}

    silence_ms = float(previous["configuration"]["inter_clip_silence_ms"])
    silence_samples = round(silence_ms * sample_rate / 1000.0)
    output_parts: list[np.ndarray] = []
    final_clips: list[dict[str, Any]] = []
    decisions: list[LeadingBoundaryDecision] = []
    output_cursor = 0
    for clip in previous_clips:
        clip_index = int(clip["clip_index"])
        if clip_index:
            output_parts.append(
                np.zeros(
                    (silence_samples, channel_count),
                    dtype=np.float32,
                )
            )
            output_cursor += silence_samples

        old_output_start = int(clip["final_output_start_sample"])
        old_output_end = int(clip["final_output_end_sample"])
        old_audio = previous_audio[old_output_start:old_output_end]
        if len(old_audio) != int(clip["final_frame_count"]):
            raise RoughRenderError(f"hard-aligned clip {clip_index} changed geometry")
        previous_source_start = int(clip["final_source_start_sample"])
        source_end = int(clip["final_source_end_sample"])
        previous_fade_in = int(clip["final_fade_in_samples"])
        fade_out = int(clip["final_fade_out_samples"])
        final_audio = old_audio
        final_source_start = previous_source_start
        final_fade_in = previous_fade_in
        enhanced = {
            **clip,
            "previous_final_source_start_sample": previous_source_start,
            "previous_final_source_start_seconds": (
                previous_source_start / sample_rate
            ),
            "leading_previous_omitted_word_id": None,
            "leading_previous_omitted_word": None,
            "leading_first_kept_word_id": int(clip["source_word_start"]),
            "leading_first_kept_word": words[int(clip["source_word_start"])].text,
            "leading_forced_aligned_omitted_end_seconds": None,
            "leading_forced_aligned_kept_start_seconds": None,
            "leading_final_start_seconds": (previous_source_start / sample_rate),
            "leading_shift_ms": 0.0,
            "leading_retained_quiet_ms": None,
            "leading_boundary_method": "start_of_file",
            "leading_alignment_status": "not_applicable",
            "leading_alignment_error": None,
            "leading_alignment_granularity": None,
            "leading_alignment_snap_method": None,
            "leading_alignment_low_amplitude_threshold_db": None,
            "leading_waveform_quiet_start_seconds": None,
            "leading_waveform_active_onset_seconds": None,
            "leading_waveform_quiet_duration_ms": None,
            "leading_waveform_search_start_seconds": None,
            "leading_waveform_search_end_seconds": None,
            "leading_dense_boundary": None,
            "leading_boundary_fade_ms": (previous_fade_in * 1000.0 / sample_rate),
        }

        if clip_index in job_by_clip:
            job = job_by_clip[clip_index]
            assessment = waveform_assessment_by_clip[clip_index]
            if assessment.candidate is not None:
                worker_job = {
                    "clip_index": clip_index,
                    "model_call_skipped": "boundary resolved from waveform",
                    "error": None,
                    "aligned": None,
                }
            else:
                worker_job = worker_by_clip.get(
                    clip_index,
                    {
                        "clip_index": clip_index,
                        "error": "alignment worker omitted this boundary",
                        "aligned": None,
                    },
                )
            decision = decide_leading_boundary(
                job=job,
                worker_job=worker_job,
                mono=mono,
                sample_rate=sample_rate,
                leading_quiet_ms=leading_quiet_ms,
                dense_boundary_fade_ms=dense_boundary_fade_ms,
                max_shift_ms=max_shift_ms,
                waveform_assessment=assessment,
            )
            decisions.append(decision)
            if decision.status in {
                "leading_waveform_silence",
                "leading_forced_alignment",
            }:
                final_source_start = round(decision.start_seconds * sample_rate)
                final_fade_in = decision.fade_in_samples
                if not 0 <= final_source_start < source_end <= total_samples:
                    raise RoughRenderError(
                        f"leading-aligned clip {clip_index} has invalid geometry"
                    )
                final_audio = _apply_edge_fades(
                    source_audio[final_source_start:source_end],
                    fade_in_samples=final_fade_in,
                    fade_out_samples=fade_out,
                )

            boundary_dir = debug_root / f"clip_{clip_index:03d}_start"
            if write_debug_artifacts:
                sf.write(
                    boundary_dir / "new_clip.wav",
                    final_audio,
                    sample_rate,
                    subtype="FLOAT",
                )
                _save_leading_alignment_plot(
                    path=boundary_dir / "alignment_plot.png",
                    mono=mono,
                    sample_rate=sample_rate,
                    job=job,
                    decision=decision,
                )
                write_json(
                    boundary_dir / "alignment.json",
                    {
                        "schema_version": 1,
                        "job": job,
                        "worker_result": worker_job,
                        "decision": asdict(decision),
                    },
                )
            enhanced.update(
                {
                    "leading_previous_omitted_word_id": (
                        job["previous_omitted_word_id"]
                    ),
                    "leading_previous_omitted_word": (job["previous_omitted_word"]),
                    "leading_first_kept_word_id": job["first_kept_word_id"],
                    "leading_first_kept_word": job["first_kept_word"],
                    "leading_forced_aligned_omitted_end_seconds": (
                        decision.omitted_end_seconds
                    ),
                    "leading_forced_aligned_kept_start_seconds": (
                        decision.kept_start_seconds
                    ),
                    "leading_final_start_seconds": decision.start_seconds,
                    "leading_shift_ms": decision.shift_ms,
                    "leading_retained_quiet_ms": (decision.retained_leading_quiet_ms),
                    "leading_boundary_method": decision.status,
                    "leading_alignment_status": decision.status,
                    "leading_alignment_error": decision.error,
                    "leading_alignment_granularity": (decision.alignment_granularity),
                    "leading_alignment_snap_method": decision.snap_method,
                    "leading_alignment_low_amplitude_threshold_db": (
                        decision.low_amplitude_threshold_db
                    ),
                    "leading_waveform_quiet_start_seconds": (
                        decision.waveform_quiet_start_seconds
                    ),
                    "leading_waveform_active_onset_seconds": (
                        decision.waveform_active_onset_seconds
                    ),
                    "leading_waveform_quiet_duration_ms": (
                        decision.waveform_quiet_duration_ms
                    ),
                    "leading_waveform_search_start_seconds": (
                        decision.waveform_search_start_seconds
                    ),
                    "leading_waveform_search_end_seconds": (
                        decision.waveform_search_end_seconds
                    ),
                    "leading_dense_boundary": decision.dense_boundary,
                    "leading_boundary_fade_ms": (
                        decision.fade_in_samples * 1000.0 / sample_rate
                    ),
                    "leading_alignment_debug_dir": (
                        str(boundary_dir.resolve()) if boundary_dir.exists() else None
                    ),
                    "leading_alignment_context_wav": job["crop_wav"],
                    "leading_alignment_old_clip_wav": (
                        str((boundary_dir / "old_clip.wav").resolve())
                        if write_debug_artifacts
                        else None
                    ),
                    "leading_alignment_new_clip_wav": (
                        str((boundary_dir / "new_clip.wav").resolve())
                        if write_debug_artifacts
                        else None
                    ),
                    "leading_alignment_json": (
                        str((boundary_dir / "alignment.json").resolve())
                        if write_debug_artifacts
                        else None
                    ),
                    "leading_alignment_plot": (
                        str((boundary_dir / "alignment_plot.png").resolve())
                        if write_debug_artifacts
                        else None
                    ),
                }
            )

        clip_path = (
            clips_root / f"clip_{clip_index:03d}.wav" if write_debug_artifacts else None
        )
        if clip_path is not None:
            sf.write(clip_path, final_audio, sample_rate, subtype="FLOAT")
        output_start = output_cursor
        output_end = output_start + len(final_audio)
        output_parts.append(final_audio)
        output_cursor = output_end
        enhanced.update(
            {
                # Endpoint fields are intentionally copied from the previous
                # stage.  Only the source/output start and frame count change.
                "final_source_start_sample": final_source_start,
                "final_source_end_sample": source_end,
                "final_fade_in_samples": final_fade_in,
                "final_fade_out_samples": fade_out,
                "final_frame_count": len(final_audio),
                "final_output_start_sample": output_start,
                "final_output_end_sample": output_end,
                "final_output_start_seconds": output_start / sample_rate,
                "final_output_end_seconds": output_end / sample_rate,
                "final_clip_wav": (
                    str(clip_path.resolve()) if clip_path is not None else None
                ),
                "final_clip_wav_sha256": (
                    sha256_file(clip_path) if clip_path is not None else None
                ),
            }
        )
        final_clips.append(enhanced)

    full_audio = (
        np.concatenate(output_parts, axis=0)
        if output_parts
        else np.zeros((0, channel_count), dtype=np.float32)
    )
    expected_frames = sum(
        int(clip["final_frame_count"]) for clip in final_clips
    ) + silence_samples * max(0, len(final_clips) - 1)
    if len(full_audio) != expected_frames:
        raise RoughRenderError(
            "full-boundary duration does not equal clips plus fixed silences"
        )
    full_path = output_dir / "full_boundary_aligned.wav"
    sf.write(full_path, full_audio, sample_rate, subtype="FLOAT")
    full_info = sf.info(full_path)
    if (
        int(full_info.frames) != expected_frames
        or int(full_info.samplerate) != sample_rate
        or int(full_info.channels) != channel_count
    ):
        raise RoughRenderError("full-boundary WAV has unexpected audio geometry")

    successful = [
        decision
        for decision in decisions
        if decision.status in {"leading_waveform_silence", "leading_forced_alignment"}
    ]
    waveform_successful = [
        decision
        for decision in decisions
        if decision.status == "leading_waveform_silence"
    ]
    forced_successful = [
        decision
        for decision in decisions
        if decision.status == "leading_forced_alignment"
    ]
    failures = [
        decision
        for decision in decisions
        if decision.status == "leading_forced_alignment_failed"
    ]
    unresolved = [
        {
            "clip_index": int(clip["clip_index"]),
            "previous_omitted_word": clip["leading_previous_omitted_word"],
            "first_kept_word": clip["leading_first_kept_word"],
            "error": clip["leading_alignment_error"],
        }
        for clip in final_clips
        if clip["leading_alignment_status"] == "leading_forced_alignment_failed"
    ]
    manifest = {
        **previous,
        "schema_version": 1,
        "renderer": "streaming_plan_full_boundary_alignment_v1",
        "previous_hard_alignment_manifest": str(forced_manifest_path),
        "previous_hard_alignment_manifest_sha256": sha256_file(forced_manifest_path),
        "previous_hard_boundary_aligned_wav": str(previous_wav_path),
        "previous_hard_boundary_aligned_wav_sha256": (
            previous["hard_boundary_aligned_wav_sha256"]
        ),
        "full_boundary_aligned_wav": str(full_path.resolve()),
        "full_boundary_aligned_wav_sha256": sha256_file(full_path),
        "full_boundary_aligned_duration_seconds": (expected_frames / sample_rate),
        "full_boundary_aligned_expected_output_frame_count": expected_frames,
        "debug_artifacts_written": write_debug_artifacts,
        "leading_alignment_configuration": {
            "strategy": "waveform_silence_then_whisperx",
            "backend": "whisperx_alignment",
            "waveform_analysis": "adaptive_rms",
            "language": language,
            "device": "cpu",
            "alignment_python": str(alignment_python.expanduser()),
            "context_words_before_omitted": CONTEXT_WORDS_PER_SIDE,
            "context_words_after_kept": CONTEXT_WORDS_PER_SIDE,
            "crop_context_ms": crop_context_ms,
            "target_leading_quiet_ms": leading_quiet_ms,
            "waveform_stable_quiet_ms": WAVEFORM_STABLE_QUIET_MS,
            "waveform_sustained_active_ms": WAVEFORM_SUSTAINED_ACTIVE_MS,
            "waveform_search_left_ms": WAVEFORM_SEARCH_LEFT_MS,
            "waveform_search_right_ms": WAVEFORM_SEARCH_RIGHT_MS,
            "dense_boundary_fade_ms": dense_boundary_fade_ms,
            "maximum_boundary_shift_ms": max_shift_ms,
            "inter_clip_silence_ms": silence_ms,
        },
        "leading_boundary_jobs": str(boundary_jobs_path.resolve()),
        "leading_alignment_jobs": str(jobs_path.resolve()),
        "leading_alignment_worker_result": str(worker_result_path.resolve()),
        "leading_boundaries_found": len(eligible_clips),
        "leading_boundaries_sent_to_whisperx": len(alignment_jobs),
        "leading_boundaries_successfully_aligned": len(successful),
        "leading_waveform_silence_boundaries": len(waveform_successful),
        "leading_forced_alignment_boundaries": len(forced_successful),
        "leading_alignment_failures": len(failures),
        "all_leading_boundaries_resolved": not failures,
        "unresolved_leading_boundaries": unresolved,
        "average_leading_boundary_shift_ms": (
            float(np.mean([decision.shift_ms for decision in successful]))
            if successful
            else 0.0
        ),
        "clips": final_clips,
    }
    manifest_path = output_dir / "render_manifest_full_boundary_aligned.json"
    write_json(manifest_path, manifest)
    return manifest
