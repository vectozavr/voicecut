#!/usr/bin/env python3
"""Refine rough-cut clip endings using only the local source waveform."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from .common import read_json, sha256_file, write_json
from .rough_render import (
    CLIP_FADE_MS,
    EDGE_PADDING_MS,
    INTER_CLIP_SILENCE_MS,
    MergedRange,
    PlanWord,
    RoughRenderError,
    flatten_selected_ranges,
    load_plan_words,
    merge_adjacent_ranges,
    render_rough_cut,
    timestamp_to_sample,
)


SEARCH_LEFT_MS = 50.0
SEARCH_RIGHT_MS = 450.0
NEXT_WORD_GUARD_MS = 10.0
RMS_FRAME_MS = 10.0
RMS_HOP_MS = 5.0
NOISE_CONTEXT_SECONDS = 1.0
NOISE_PERCENTILE = 20.0
SILENCE_OFFSET_DB = 12.0
MIN_SILENCE_THRESHOLD_DB = -55.0
MAX_SILENCE_THRESHOLD_DB = -30.0
STABLE_SILENCE_MS = 70.0
SILENCE_ENDPOINT_OFFSET_MS = 20.0
LOW_AMPLITUDE_SNAP_MS = 5.0
MAX_ACTIVE_TO_SILENCE_GAP_MS = 30.0


@dataclass(frozen=True)
class BoundaryRefinement:
    raw_end_sample: int
    refined_end_sample: int
    search_start_sample: int
    search_limit_sample: int
    next_omitted_word_start_sample: int | None
    boundary_method: str
    local_noise_floor_db: float
    silence_threshold_db: float
    stable_silence_start_sample: int | None


def rms_envelope_db(
    mono: np.ndarray,
    *,
    start_sample: int,
    end_sample: int,
    sample_rate: int,
    frame_ms: float = RMS_FRAME_MS,
    hop_ms: float = RMS_HOP_MS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute frame starts and short-time RMS in dBFS."""

    if mono.ndim != 1:
        raise ValueError("analysis waveform must be mono")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if frame_ms <= 0.0 or hop_ms <= 0.0:
        raise ValueError("RMS frame and hop durations must be positive")
    start = max(0, min(len(mono), int(start_sample)))
    end = max(start, min(len(mono), int(end_sample)))
    frame_samples = max(1, round(frame_ms * sample_rate / 1000.0))
    hop_samples = max(1, round(hop_ms * sample_rate / 1000.0))
    if end - start < frame_samples:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    frame_starts = np.arange(
        start,
        end - frame_samples + 1,
        hop_samples,
        dtype=np.int64,
    )
    rms = np.empty(len(frame_starts), dtype=np.float64)
    for index, frame_start in enumerate(frame_starts):
        frame = mono[frame_start : frame_start + frame_samples].astype(
            np.float64,
            copy=False,
        )
        rms[index] = math.sqrt(float(np.mean(frame * frame)))
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-8))
    return frame_starts, rms_db


def _local_threshold(
    mono: np.ndarray,
    *,
    raw_end_sample: int,
    sample_rate: int,
) -> tuple[float, float]:
    half_context = round(NOISE_CONTEXT_SECONDS * sample_rate / 2.0)
    starts, rms_db = rms_envelope_db(
        mono,
        start_sample=raw_end_sample - half_context,
        end_sample=raw_end_sample + half_context,
        sample_rate=sample_rate,
    )
    del starts
    if len(rms_db):
        noise_floor_db = float(np.percentile(rms_db, NOISE_PERCENTILE))
    else:
        single = mono[max(0, raw_end_sample - 1) : min(len(mono), raw_end_sample + 1)]
        rms = (
            math.sqrt(float(np.mean(single.astype(np.float64) ** 2)))
            if len(single)
            else 0.0
        )
        noise_floor_db = 20.0 * math.log10(max(rms, 1e-8))
    threshold = float(
        np.clip(
            noise_floor_db + SILENCE_OFFSET_DB,
            MIN_SILENCE_THRESHOLD_DB,
            MAX_SILENCE_THRESHOLD_DB,
        )
    )
    return noise_floor_db, threshold


def _first_stable_silence(
    mono: np.ndarray,
    *,
    raw_end_sample: int,
    search_start_sample: int,
    search_limit_sample: int,
    sample_rate: int,
    threshold_db: float,
    require_recent_active: bool = True,
) -> int | None:
    frame_samples = max(1, round(RMS_FRAME_MS * sample_rate / 1000.0))
    stable_samples = max(
        frame_samples,
        round(STABLE_SILENCE_MS * sample_rate / 1000.0),
    )
    starts, rms_db = rms_envelope_db(
        mono,
        start_sample=search_start_sample,
        end_sample=search_limit_sample,
        sample_rate=sample_rate,
    )
    max_active_gap_samples = max(
        frame_samples,
        round(MAX_ACTIVE_TO_SILENCE_GAP_MS * sample_rate / 1000.0),
    )
    last_active_end: int | None = None
    run_start: int | None = None
    for frame_start, level_db in zip(starts, rms_db, strict=True):
        frame_start = int(frame_start)
        frame_end = frame_start + frame_samples
        if level_db >= threshold_db:
            last_active_end = frame_end
            run_start = None
            continue
        if frame_start < raw_end_sample:
            continue
        if run_start is None:
            run_start = frame_start
        if frame_end - run_start >= stable_samples:
            # A low-energy run is evidence of a speech ending only when the
            # waveform crossed into it from recent active speech.  Without
            # this guard, a raw timestamp already sitting in background
            # quiet can be mistaken for the end even when speech resumes
            # later in the search window.
            has_recent_active = (
                last_active_end is not None
                and run_start - last_active_end <= max_active_gap_samples
            )
            if not require_recent_active or has_recent_active:
                return run_start
    return None


def refine_trailing_boundary(
    mono: np.ndarray,
    *,
    words: Sequence[PlanWord],
    source_range: MergedRange,
    sample_rate: int,
) -> BoundaryRefinement:
    """Find the first stable post-word silence without crossing omitted speech."""

    if mono.ndim != 1 or not len(mono):
        raise ValueError("mono analysis waveform must be non-empty")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    end_id = source_range.end_word_id
    if not 0 < end_id <= len(words):
        raise RoughRenderError("selected range end does not fit the word ledger")
    total_samples = len(mono)
    raw_end_seconds = words[end_id - 1].end
    raw_end_sample = timestamp_to_sample(
        raw_end_seconds,
        sample_rate=sample_rate,
        total_samples=total_samples,
        rounding="ceil",
    )
    search_start = max(
        0,
        raw_end_sample - round(SEARCH_LEFT_MS * sample_rate / 1000.0),
    )
    desired_limit = min(
        total_samples,
        raw_end_sample + round(SEARCH_RIGHT_MS * sample_rate / 1000.0),
    )
    next_word_start: int | None = None
    if end_id < len(words):
        next_word_start = timestamp_to_sample(
            words[end_id].start,
            sample_rate=sample_rate,
            total_samples=total_samples,
            rounding="floor",
        )
        desired_limit = min(
            desired_limit,
            next_word_start - round(NEXT_WORD_GUARD_MS * sample_rate / 1000.0),
        )
    # When the guarded limit falls before raw_end, there is no searchable gap.
    # Keeping the raw endpoint is the required hard-boundary fallback.
    search_limit = max(raw_end_sample, desired_limit)
    noise_floor_db, threshold_db = _local_threshold(
        mono,
        raw_end_sample=raw_end_sample,
        sample_rate=sample_rate,
    )

    if end_id == len(words) and raw_end_sample >= total_samples:
        return BoundaryRefinement(
            raw_end_sample=raw_end_sample,
            refined_end_sample=raw_end_sample,
            search_start_sample=search_start,
            search_limit_sample=search_limit,
            next_omitted_word_start_sample=None,
            boundary_method="end_of_file",
            local_noise_floor_db=noise_floor_db,
            silence_threshold_db=threshold_db,
            stable_silence_start_sample=None,
        )
    if desired_limit <= raw_end_sample:
        return BoundaryRefinement(
            raw_end_sample=raw_end_sample,
            refined_end_sample=raw_end_sample,
            search_start_sample=search_start,
            search_limit_sample=search_limit,
            next_omitted_word_start_sample=next_word_start,
            boundary_method="hard_boundary",
            local_noise_floor_db=noise_floor_db,
            silence_threshold_db=threshold_db,
            stable_silence_start_sample=None,
        )

    stable_start = _first_stable_silence(
        mono,
        raw_end_sample=raw_end_sample,
        search_start_sample=search_start,
        search_limit_sample=search_limit,
        sample_rate=sample_rate,
        threshold_db=threshold_db,
    )
    if stable_start is None:
        if end_id == len(words):
            # There is no later source word to protect at EOF.  Keeping the
            # entire remaining source is safer than trusting a timestamp
            # when the waveform never establishes a real speech-to-silence
            # transition inside the search window.
            return BoundaryRefinement(
                raw_end_sample=raw_end_sample,
                refined_end_sample=total_samples,
                search_start_sample=search_start,
                search_limit_sample=search_limit,
                next_omitted_word_start_sample=None,
                boundary_method="end_of_file",
                local_noise_floor_db=noise_floor_db,
                silence_threshold_db=threshold_db,
                stable_silence_start_sample=None,
            )
        return BoundaryRefinement(
            raw_end_sample=raw_end_sample,
            refined_end_sample=raw_end_sample,
            search_start_sample=search_start,
            search_limit_sample=search_limit,
            next_omitted_word_start_sample=next_word_start,
            boundary_method="hard_boundary",
            local_noise_floor_db=noise_floor_db,
            silence_threshold_db=threshold_db,
            stable_silence_start_sample=None,
        )

    target = min(
        search_limit,
        stable_start + round(SILENCE_ENDPOINT_OFFSET_MS * sample_rate / 1000.0),
    )
    snap_radius = round(LOW_AMPLITUDE_SNAP_MS * sample_rate / 1000.0)
    snap_start = max(raw_end_sample, stable_start, target - snap_radius)
    snap_end = min(search_limit, target + snap_radius)
    if snap_end > snap_start:
        relative = int(np.argmin(np.abs(mono[snap_start : snap_end + 1])))
        refined_end = snap_start + relative
    else:
        refined_end = target
    refined_end = max(raw_end_sample, min(search_limit, refined_end))
    if next_word_start is not None and refined_end > next_word_start:
        raise RoughRenderError("refined boundary crosses the next omitted source word")
    return BoundaryRefinement(
        raw_end_sample=raw_end_sample,
        refined_end_sample=refined_end,
        search_start_sample=search_start,
        search_limit_sample=search_limit,
        next_omitted_word_start_sample=next_word_start,
        boundary_method="stable_silence",
        local_noise_floor_db=noise_floor_db,
        silence_threshold_db=threshold_db,
        stable_silence_start_sample=stable_start,
    )


def _apply_refined_fades(
    samples: np.ndarray,
    *,
    requested_fade_samples: int,
    fade_out: bool,
    stable_silence_offset_samples: int | None,
) -> tuple[np.ndarray, int, int]:
    """Keep the existing fade-in; fade out only inside verified silence."""

    rendered = np.array(samples, dtype=np.float32, copy=True)
    fade_in_samples = min(requested_fade_samples, len(rendered) // 2)
    if fade_in_samples:
        ramp = np.linspace(
            0.0,
            1.0,
            fade_in_samples,
            endpoint=True,
            dtype=np.float32,
        )
        rendered[:fade_in_samples] *= ramp[:, None]
    fade_out_samples = 0
    if fade_out:
        fade_out_samples = min(requested_fade_samples, len(rendered) // 2)
        if fade_out_samples:
            fade_start = len(rendered) - fade_out_samples
            if (
                stable_silence_offset_samples is None
                or fade_start < stable_silence_offset_samples
            ):
                raise RoughRenderError(
                    "refined fade-out would begin before stable silence"
                )
            ramp = np.linspace(
                1.0,
                0.0,
                fade_out_samples,
                endpoint=True,
                dtype=np.float32,
            )
            rendered[-fade_out_samples:] *= ramp[:, None]
    return rendered, fade_in_samples, fade_out_samples


def _save_boundary_plot(
    *,
    path: Path,
    mono: np.ndarray,
    sample_rate: int,
    boundary: BoundaryRefinement,
    clip_index: int,
) -> None:
    """Save a two-panel waveform/RMS diagnostic around one endpoint."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    plot_start = max(
        0,
        boundary.raw_end_sample - round(0.5 * sample_rate),
    )
    plot_end = min(
        len(mono),
        boundary.raw_end_sample + round(0.5 * sample_rate),
    )
    waveform_time = np.arange(plot_start, plot_end) / sample_rate
    frame_starts, rms_db = rms_envelope_db(
        mono,
        start_sample=plot_start,
        end_sample=plot_end,
        sample_rate=sample_rate,
    )
    frame_time = (
        frame_starts + round(RMS_FRAME_MS * sample_rate / 2000.0)
    ) / sample_rate

    figure = Figure(figsize=(11, 6), constrained_layout=True)
    FigureCanvasAgg(figure)
    waveform_axis, rms_axis = figure.subplots(2, 1, sharex=True)
    waveform_axis.plot(
        waveform_time,
        mono[plot_start:plot_end],
        linewidth=0.7,
        color="#345995",
        label="mono waveform",
    )
    rms_axis.plot(
        frame_time,
        rms_db,
        linewidth=1.2,
        color="#1b998b",
        label="10 ms RMS / 5 ms hop",
    )
    rms_axis.axhline(
        boundary.silence_threshold_db,
        color="#7a5195",
        linestyle="--",
        label=f"silence threshold {boundary.silence_threshold_db:.1f} dB",
    )

    verticals = [
        (
            boundary.raw_end_sample / sample_rate,
            "#e45756",
            "-",
            "raw Whisper end",
        ),
        (
            boundary.refined_end_sample / sample_rate,
            "#2e8b57",
            "-",
            f"refined end ({boundary.boundary_method})",
        ),
    ]
    if boundary.next_omitted_word_start_sample is not None:
        verticals.append(
            (
                boundary.next_omitted_word_start_sample / sample_rate,
                "#f3a712",
                ":",
                "next omitted word",
            )
        )
    else:
        waveform_axis.text(
            0.99,
            0.96,
            "next omitted word: none (EOF)",
            ha="right",
            va="top",
            transform=waveform_axis.transAxes,
            fontsize=8,
            color="#8a5a00",
        )
    for axis in (waveform_axis, rms_axis):
        for position, color, style, label in verticals:
            axis.axvline(
                position,
                color=color,
                linestyle=style,
                linewidth=1.4,
                label=label,
            )
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize=8)
    waveform_axis.set_ylabel("amplitude")
    rms_axis.set_ylabel("RMS dBFS")
    rms_axis.set_xlabel("source time (seconds)")
    figure.suptitle(f"Clip {clip_index:03d} trailing boundary")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)


def render_trailing_refined_preview(
    *,
    audio_path: Path,
    plan_path: Path,
    output_dir: Path,
    edge_padding_ms: float = EDGE_PADDING_MS,
    clip_fade_ms: float = CLIP_FADE_MS,
    inter_clip_silence_ms: float = INTER_CLIP_SILENCE_MS,
    write_debug_artifacts: bool = True,
) -> dict[str, Any]:
    """Create baseline and trailing-refined previews in a fresh directory."""

    audio_path = audio_path.resolve()
    plan_path = plan_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"output directory must be empty for trailing refinement: {output_dir}"
        )

    baseline = render_rough_cut(
        audio_path=audio_path,
        plan_path=plan_path,
        output_dir=output_dir,
        edge_padding_ms=edge_padding_ms,
        clip_fade_ms=clip_fade_ms,
        inter_clip_silence_ms=inter_clip_silence_ms,
        write_debug_artifacts=write_debug_artifacts,
    )
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise RoughRenderError("streaming plan root must be an object")
    words = load_plan_words(plan)
    merged_ranges = merge_adjacent_ranges(
        flatten_selected_ranges(plan, word_count=len(words))
    )
    if len(merged_ranges) != len(baseline["clips"]):
        raise RoughRenderError("baseline clip list differs from merged plan ranges")

    source_audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )
    sample_rate = int(sample_rate)
    mono = np.mean(source_audio, axis=1, dtype=np.float64).astype(np.float32)
    total_samples, channel_count = source_audio.shape
    fade_samples = round(clip_fade_ms * sample_rate / 1000.0)
    silence_samples = round(inter_clip_silence_ms * sample_rate / 1000.0)
    debug_dir = output_dir / "boundary_debug"
    refined_clips_dir = output_dir / "clips_refined"
    if write_debug_artifacts:
        debug_dir.mkdir()
        refined_clips_dir.mkdir()

    output_parts: list[np.ndarray] = []
    refined_clips: list[dict[str, Any]] = []
    output_cursor = 0
    for clip_index, (merged_range, baseline_clip) in enumerate(
        zip(merged_ranges, baseline["clips"], strict=True)
    ):
        if clip_index:
            output_parts.append(
                np.zeros(
                    (silence_samples, channel_count),
                    dtype=np.float32,
                )
            )
            output_cursor += silence_samples
        boundary = refine_trailing_boundary(
            mono,
            words=words,
            source_range=merged_range,
            sample_rate=sample_rate,
        )
        source_start = int(baseline_clip["source_start_sample"])
        source_end = boundary.refined_end_sample
        if not 0 <= source_start < source_end <= total_samples:
            raise RoughRenderError(
                f"refined clip {clip_index} has invalid source geometry"
            )
        stable_offset = (
            boundary.stable_silence_start_sample - source_start
            if boundary.stable_silence_start_sample is not None
            else None
        )
        rendered_clip, fade_in_count, fade_out_count = _apply_refined_fades(
            source_audio[source_start:source_end],
            requested_fade_samples=fade_samples,
            fade_out=boundary.boundary_method == "stable_silence",
            stable_silence_offset_samples=stable_offset,
        )
        refined_clip_path = (
            refined_clips_dir / f"clip_{clip_index:03d}.wav"
            if write_debug_artifacts
            else None
        )
        if refined_clip_path is not None:
            sf.write(
                refined_clip_path,
                rendered_clip,
                sample_rate,
                subtype="FLOAT",
            )
        output_start = output_cursor
        output_end = output_start + len(rendered_clip)
        output_parts.append(rendered_clip)
        output_cursor = output_end

        raw_end_seconds = words[merged_range.end_word_id - 1].end
        refined_end_seconds = boundary.refined_end_sample / sample_rate
        extension_ms = max(
            0.0,
            (refined_end_seconds - raw_end_seconds) * 1000.0,
        )
        debug_path = (
            debug_dir / f"clip_{clip_index:03d}_end.png"
            if write_debug_artifacts
            else None
        )
        if debug_path is not None:
            _save_boundary_plot(
                path=debug_path,
                mono=mono,
                sample_rate=sample_rate,
                boundary=boundary,
                clip_index=clip_index,
            )
        refined_clips.append(
            {
                **baseline_clip,
                "last_selected_word": words[merged_range.end_word_id - 1].text,
                "raw_end_seconds": raw_end_seconds,
                "refined_end_seconds": refined_end_seconds,
                "end_extension_ms": extension_ms,
                "boundary_method": boundary.boundary_method,
                "local_noise_floor_db": boundary.local_noise_floor_db,
                "silence_threshold_db": boundary.silence_threshold_db,
                "search_start_seconds": (boundary.search_start_sample / sample_rate),
                "search_limit_seconds": (boundary.search_limit_sample / sample_rate),
                "next_omitted_word_start_seconds": (
                    boundary.next_omitted_word_start_sample / sample_rate
                    if boundary.next_omitted_word_start_sample is not None
                    else None
                ),
                "stable_silence_start_seconds": (
                    boundary.stable_silence_start_sample / sample_rate
                    if boundary.stable_silence_start_sample is not None
                    else None
                ),
                "refined_source_start_sample": source_start,
                "refined_source_end_sample": source_end,
                "refined_frame_count": len(rendered_clip),
                "refined_output_start_sample": output_start,
                "refined_output_end_sample": output_end,
                "refined_output_start_seconds": output_start / sample_rate,
                "refined_output_end_seconds": output_end / sample_rate,
                "refined_fade_in_samples": fade_in_count,
                "refined_fade_out_samples": fade_out_count,
                "refined_clip_wav": (
                    str(refined_clip_path.resolve())
                    if refined_clip_path is not None
                    else None
                ),
                "refined_clip_wav_sha256": (
                    sha256_file(refined_clip_path)
                    if refined_clip_path is not None
                    else None
                ),
                "boundary_debug_plot": (
                    str(debug_path.resolve()) if debug_path is not None else None
                ),
                "boundary_debug": asdict(boundary),
            }
        )

    refined_audio = np.concatenate(output_parts, axis=0)
    expected_frames = sum(
        int(clip["refined_frame_count"]) for clip in refined_clips
    ) + silence_samples * (len(refined_clips) - 1)
    if len(refined_audio) != expected_frames:
        raise RoughRenderError(
            "refined duration does not equal clips plus fixed silences"
        )
    refined_path = output_dir / "rough_cut_refined.wav"
    sf.write(
        refined_path,
        refined_audio,
        sample_rate,
        subtype="FLOAT",
    )
    refined_info = sf.info(refined_path)
    if (
        int(refined_info.frames) != expected_frames
        or int(refined_info.samplerate) != sample_rate
        or int(refined_info.channels) != channel_count
    ):
        raise RoughRenderError("written refined WAV has unexpected audio geometry")

    extensions = [float(clip["end_extension_ms"]) for clip in refined_clips]
    refined_count = sum(
        clip["boundary_method"] == "stable_silence" for clip in refined_clips
    )
    hard_count = sum(
        clip["boundary_method"] == "hard_boundary" for clip in refined_clips
    )
    manifest = {
        **baseline,
        "schema_version": 1,
        "renderer": "streaming_plan_trailing_refinement_v1",
        "baseline_render_manifest": str(
            (output_dir / "render_manifest.json").resolve()
        ),
        "rough_cut_refined_wav": str(refined_path.resolve()),
        "rough_cut_refined_wav_sha256": sha256_file(refined_path),
        "rough_cut_refined_duration_seconds": (expected_frames / sample_rate),
        "refined_expected_output_frame_count": expected_frames,
        "debug_artifacts_written": write_debug_artifacts,
        "trailing_boundary_configuration": {
            "search_left_ms": SEARCH_LEFT_MS,
            "search_right_ms": SEARCH_RIGHT_MS,
            "next_word_guard_ms": NEXT_WORD_GUARD_MS,
            "rms_frame_ms": RMS_FRAME_MS,
            "rms_hop_ms": RMS_HOP_MS,
            "noise_context_seconds": NOISE_CONTEXT_SECONDS,
            "noise_percentile": NOISE_PERCENTILE,
            "silence_offset_db": SILENCE_OFFSET_DB,
            "silence_threshold_clip_db": [
                MIN_SILENCE_THRESHOLD_DB,
                MAX_SILENCE_THRESHOLD_DB,
            ],
            "stable_silence_ms": STABLE_SILENCE_MS,
            "max_active_to_silence_gap_ms": MAX_ACTIVE_TO_SILENCE_GAP_MS,
            "silence_endpoint_offset_ms": SILENCE_ENDPOINT_OFFSET_MS,
            "low_amplitude_snap_ms": LOW_AMPLITUDE_SNAP_MS,
        },
        "number_of_refined_boundaries": refined_count,
        "number_of_hard_boundaries": hard_count,
        "number_of_end_of_file_boundaries": sum(
            clip["boundary_method"] == "end_of_file" for clip in refined_clips
        ),
        "average_extension_ms": (float(np.mean(extensions)) if extensions else 0.0),
        "maximum_extension_ms": max(extensions, default=0.0),
        "clips": refined_clips,
    }
    write_json(output_dir / "render_manifest_refined.json", manifest)
    return manifest
