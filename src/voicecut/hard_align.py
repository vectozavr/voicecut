#!/usr/bin/env python3
"""Resolve only rough-preview hard boundaries with local forced alignment."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from .common import read_json, sha256_file, write_json
from .rough_render import (
    RoughRenderError,
    flatten_selected_ranges,
    load_plan_words,
    merge_adjacent_ranges,
    timestamp_to_sample,
)
from .trailing_refine import _local_threshold


CONTEXT_WORDS_PER_SIDE = 2
CROP_CONTEXT_MS = 300.0
MAX_BOUNDARY_SHIFT_MS = 350.0
HARD_BOUNDARY_FADE_MS = 2.0
LOW_AMPLITUDE_FRAME_MS = 4.0
LOW_AMPLITUDE_HOP_MS = 1.0

DEFAULT_ALIGNMENT_PYTHON = Path(sys.executable)


@dataclass(frozen=True)
class ForcedBoundaryDecision:
    status: str
    error: str | None
    alignment_granularity: str | None
    kept_end_seconds: float | None
    omitted_start_seconds: float | None
    cut_seconds: float
    shift_ms: float
    snap_method: str | None
    low_amplitude_threshold_db: float | None


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
    words: list[dict[str, Any]] = []
    raw_segments = aligned.get("segments")
    if not isinstance(raw_segments, list):
        return words
    for segment in raw_segments:
        if isinstance(segment, dict) and isinstance(segment.get("words"), list):
            words.extend(word for word in segment["words"] if isinstance(word, dict))
    return words


def _flatten_character_word_groups(
    aligned: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    raw_segments = aligned.get("segments")
    if not isinstance(raw_segments, list):
        return groups
    for segment in raw_segments:
        if not isinstance(segment, dict):
            continue
        raw_chars = segment.get("chars")
        if not isinstance(raw_chars, list):
            continue
        current: list[dict[str, Any]] = []
        for character in raw_chars:
            if not isinstance(character, dict):
                continue
            text = str(character.get("char", ""))
            if text.isspace():
                if current:
                    groups.append(current)
                    current = []
            else:
                current.append(character)
        if current:
            groups.append(current)
    return groups


def _aligned_character_end(group: Sequence[dict[str, Any]]) -> float | None:
    values = [
        value
        for character in group
        if (value := _finite_number(character.get("end"))) is not None
        and _finite_number(character.get("score")) is not None
    ]
    return values[-1] if values else None


def _aligned_character_start(
    group: Sequence[dict[str, Any]],
) -> float | None:
    values = [
        value
        for character in group
        if (value := _finite_number(character.get("start"))) is not None
        and _finite_number(character.get("score")) is not None
    ]
    return values[0] if values else None


def alignment_positions(
    *,
    job: dict[str, Any],
    worker_job: dict[str, Any],
) -> tuple[float, float, str]:
    """Map local alignment positions by sequence, never by word text."""

    if worker_job.get("error"):
        raise ValueError(str(worker_job["error"]))
    aligned = worker_job.get("aligned")
    if not isinstance(aligned, dict):
        raise ValueError("worker did not return an aligned result")
    local_words = job.get("local_words")
    if not isinstance(local_words, list) or not local_words:
        raise ValueError("alignment job has no local word ledger")
    kept_index = job.get("kept_local_index")
    omitted_index = job.get("omitted_local_index")
    if type(kept_index) is not int or type(omitted_index) is not int:
        raise ValueError("alignment job has invalid local boundary indices")

    aligned_words = _flatten_aligned_words(aligned)
    if len(aligned_words) != len(local_words):
        raise ValueError(
            "aligned word count does not match local transcript positions: "
            f"{len(aligned_words)} != {len(local_words)}"
        )
    if not 0 <= kept_index < omitted_index < len(aligned_words):
        raise ValueError("local boundary indices do not fit aligned words")

    character_groups = _flatten_character_word_groups(aligned)
    if len(character_groups) == len(local_words):
        kept_end = _aligned_character_end(character_groups[kept_index])
        omitted_start = _aligned_character_start(character_groups[omitted_index])
        if kept_end is not None and omitted_start is not None:
            return kept_end, omitted_start, "characters"

    kept_word = aligned_words[kept_index]
    omitted_word = aligned_words[omitted_index]
    kept_end = _finite_number(kept_word.get("end"))
    omitted_start = _finite_number(omitted_word.get("start"))
    kept_score = _finite_number(kept_word.get("score"))
    omitted_score = _finite_number(omitted_word.get("score"))
    if (
        kept_end is None
        or omitted_start is None
        or kept_score is None
        or omitted_score is None
    ):
        raise ValueError("kept or omitted boundary word was not aligned")
    return kept_end, omitted_start, "words"


def _snap_between_aligned_words(
    mono: np.ndarray,
    *,
    sample_rate: int,
    kept_end_sample: int,
    omitted_start_sample: int,
    raw_end_sample: int,
) -> tuple[int, str, float]:
    if omitted_start_sample <= kept_end_sample:
        _, threshold_db = _local_threshold(
            mono,
            raw_end_sample=raw_end_sample,
            sample_rate=sample_rate,
        )
        return kept_end_sample, "touching_words", threshold_db

    midpoint = round((kept_end_sample + omitted_start_sample) / 2.0)
    _, threshold_db = _local_threshold(
        mono,
        raw_end_sample=raw_end_sample,
        sample_rate=sample_rate,
    )
    frame_samples = max(
        1,
        round(LOW_AMPLITUDE_FRAME_MS * sample_rate / 1000.0),
    )
    hop_samples = max(
        1,
        round(LOW_AMPLITUDE_HOP_MS * sample_rate / 1000.0),
    )
    gap_samples = omitted_start_sample - kept_end_sample
    if gap_samples < frame_samples:
        return midpoint, "forced_alignment_midpoint", threshold_db

    starts = np.arange(
        kept_end_sample,
        omitted_start_sample - frame_samples + 1,
        hop_samples,
        dtype=np.int64,
    )
    levels: list[float] = []
    for frame_start in starts:
        frame = mono[int(frame_start) : int(frame_start) + frame_samples].astype(
            np.float64, copy=False
        )
        rms = math.sqrt(float(np.mean(frame * frame)))
        levels.append(20.0 * math.log10(max(rms, 1e-8)))
    quietest_index = int(np.argmin(levels))
    if levels[quietest_index] >= threshold_db:
        return midpoint, "forced_alignment_midpoint", threshold_db

    quiet_start = int(starts[quietest_index])
    quiet_end = quiet_start + frame_samples
    relative = int(np.argmin(np.abs(mono[quiet_start : quiet_end + 1])))
    snapped = quiet_start + relative
    snapped = max(kept_end_sample, min(omitted_start_sample, snapped))
    return snapped, "low_amplitude_between_words", threshold_db


def decide_forced_boundary(
    *,
    job: dict[str, Any],
    worker_job: dict[str, Any],
    mono: np.ndarray,
    sample_rate: int,
    max_shift_ms: float = MAX_BOUNDARY_SHIFT_MS,
) -> ForcedBoundaryDecision:
    raw_end = float(job["raw_end_seconds"])
    previous_end = float(job["silence_refined_end_seconds"])
    crop_start = float(job["crop_start_seconds"])
    crop_duration = float(job["crop_duration_seconds"])
    crop_end = crop_start + crop_duration
    try:
        kept_relative, omitted_relative, granularity = alignment_positions(
            job=job,
            worker_job=worker_job,
        )
        if not (
            0.0 <= kept_relative <= crop_duration
            and 0.0 <= omitted_relative <= crop_duration
        ):
            raise ValueError("aligned timestamps fall outside the local crop")
        kept_end = crop_start + kept_relative
        omitted_start = crop_start + omitted_relative
        if not (
            crop_start <= kept_end <= crop_end
            and crop_start <= omitted_start <= crop_end
        ):
            raise ValueError("absolute aligned timestamps fall outside crop")

        total_samples = len(mono)
        kept_sample = timestamp_to_sample(
            kept_end,
            sample_rate=sample_rate,
            total_samples=total_samples,
            rounding="ceil",
        )
        omitted_sample = timestamp_to_sample(
            omitted_start,
            sample_rate=sample_rate,
            total_samples=total_samples,
            rounding="floor",
        )
        raw_sample = timestamp_to_sample(
            raw_end,
            sample_rate=sample_rate,
            total_samples=total_samples,
            rounding="ceil",
        )
        maximum_shift_samples = math.floor(max_shift_ms * sample_rate / 1000.0 + 1e-9)
        allowed_start = max(kept_sample, raw_sample - maximum_shift_samples)
        natural_end = omitted_sample if omitted_sample > kept_sample else kept_sample
        allowed_end = min(
            natural_end,
            raw_sample + maximum_shift_samples,
        )
        if allowed_end < allowed_start:
            required_shift_ms = (kept_sample - raw_sample) * 1000.0 / sample_rate
            raise ValueError(
                "aligned kept-word end lies outside the permitted boundary "
                f"shift: {required_shift_ms:.3f} ms exceeds "
                f"{max_shift_ms:.3f} ms"
            )
        cut_sample, snap_method, threshold_db = _snap_between_aligned_words(
            mono,
            sample_rate=sample_rate,
            kept_end_sample=allowed_start,
            omitted_start_sample=allowed_end,
            raw_end_sample=raw_sample,
        )
        if cut_sample < kept_sample:
            raise ValueError("chosen cut precedes aligned kept-word end")
        if omitted_sample > kept_sample and cut_sample > omitted_sample:
            raise ValueError("chosen cut passes aligned omitted-word start")
        cut_seconds = cut_sample / sample_rate
        shift_ms = (cut_seconds - raw_end) * 1000.0
        if abs(shift_ms) > max_shift_ms + 1e-6:
            raise ValueError(
                f"boundary shift {shift_ms:.3f} ms exceeds {max_shift_ms:.3f} ms"
            )
        return ForcedBoundaryDecision(
            status="forced_alignment",
            error=None,
            alignment_granularity=granularity,
            kept_end_seconds=kept_end,
            omitted_start_seconds=omitted_start,
            cut_seconds=cut_seconds,
            shift_ms=shift_ms,
            snap_method=snap_method,
            low_amplitude_threshold_db=threshold_db,
        )
    except Exception as error:
        return ForcedBoundaryDecision(
            status="forced_alignment_failed",
            error=f"{type(error).__name__}: {error}",
            alignment_granularity=None,
            kept_end_seconds=None,
            omitted_start_seconds=None,
            cut_seconds=previous_end,
            shift_ms=(previous_end - raw_end) * 1000.0,
            snap_method=None,
            low_amplitude_threshold_db=None,
        )


def _apply_forced_boundary_fades(
    samples: np.ndarray,
    *,
    fade_in_samples: int,
    fade_out_samples: int,
) -> np.ndarray:
    rendered = np.array(samples, dtype=np.float32, copy=True)
    fade_in = min(fade_in_samples, len(rendered) // 2)
    if fade_in:
        rendered[:fade_in] *= np.linspace(
            0.0,
            1.0,
            fade_in,
            endpoint=True,
            dtype=np.float32,
        )[:, None]
    fade_out = min(fade_out_samples, len(rendered) // 2)
    if fade_out:
        rendered[-fade_out:] *= np.linspace(
            1.0,
            0.0,
            fade_out,
            endpoint=True,
            dtype=np.float32,
        )[:, None]
    return rendered


def _save_alignment_plot(
    *,
    path: Path,
    mono: np.ndarray,
    sample_rate: int,
    job: dict[str, Any],
    decision: ForcedBoundaryDecision,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    crop_start_sample = int(job["crop_start_sample"])
    crop_end_sample = int(job["crop_end_sample"])
    time = np.arange(crop_start_sample, crop_end_sample) / sample_rate
    figure = Figure(figsize=(12, 4.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots(1, 1)
    axis.plot(
        time,
        mono[crop_start_sample:crop_end_sample],
        color="#345995",
        linewidth=0.65,
        label="mono waveform",
    )
    lines: list[tuple[float | None, str, str, str]] = [
        (
            float(job["raw_end_seconds"]),
            "#e45756",
            "--",
            "original Whisper boundary",
        ),
        (
            decision.kept_end_seconds,
            "#54a24b",
            "-",
            "kept-word aligned end",
        ),
        (
            decision.omitted_start_seconds,
            "#f3a712",
            "-",
            "omitted-word aligned start",
        ),
        (
            decision.cut_seconds,
            "#7a5195",
            ":",
            "chosen cut",
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
        f"Clip {int(job['clip_index']):03d}: "
        f"{job['last_kept_word']} → {job['first_omitted_word']} "
        f"({decision.status})"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)


def _prepare_alignment_jobs(
    *,
    hard_clips: Sequence[dict[str, Any]],
    words: Sequence[Any],
    source_audio: np.ndarray,
    previous_audio: np.ndarray,
    sample_rate: int,
    debug_root: Path,
    crop_context_ms: float,
    write_debug_artifacts: bool,
) -> list[dict[str, Any]]:
    total_samples = len(source_audio)
    context_samples = round(crop_context_ms * sample_rate / 1000.0)
    jobs: list[dict[str, Any]] = []
    for clip in hard_clips:
        clip_index = int(clip["clip_index"])
        first_omitted_id = int(clip["source_word_end"])
        last_kept_id = first_omitted_id - 1
        if not 0 <= last_kept_id < first_omitted_id < len(words):
            raise RoughRenderError(
                f"hard clip {clip_index} has no following omitted word"
            )
        requested_context_start_id = max(
            0,
            last_kept_id - CONTEXT_WORDS_PER_SIDE,
        )
        requested_context_end_id = min(
            len(words),
            first_omitted_id + CONTEXT_WORDS_PER_SIDE + 1,
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
        # A requested 300 ms handle can intersect an adjacent spoken word.
        # Include every such word in full so the local transcript describes
        # all speech actually present in the crop.
        context_start_id = requested_context_start_id
        while context_start_id > 0:
            previous = words[context_start_id - 1]
            if math.ceil(previous.end * sample_rate) <= crop_start:
                break
            context_start_id -= 1
            crop_start = min(
                crop_start,
                math.floor(previous.start * sample_rate),
            )
        context_end_id = requested_context_end_id
        while context_end_id < len(words):
            following = words[context_end_id]
            if math.floor(following.start * sample_rate) >= crop_end:
                break
            crop_end = max(
                crop_end,
                math.ceil(following.end * sample_rate),
            )
            context_end_id += 1
        if crop_end <= crop_start:
            raise RoughRenderError(
                f"hard clip {clip_index} produced an empty context crop"
            )
        boundary_dir = debug_root / f"clip_{clip_index:03d}"
        boundary_dir.mkdir(parents=True, exist_ok=False)
        context_path = boundary_dir / "context.wav"
        sf.write(
            context_path,
            source_audio[crop_start:crop_end],
            sample_rate,
            subtype="FLOAT",
        )
        output_start = int(clip["refined_output_start_sample"])
        output_end = int(clip["refined_output_end_sample"])
        if not 0 <= output_start < output_end <= len(previous_audio):
            raise RoughRenderError(
                f"hard clip {clip_index} has invalid previous output geometry"
            )
        old_clip = previous_audio[output_start:output_end]
        if len(old_clip) != int(clip["refined_frame_count"]):
            raise RoughRenderError(
                f"hard clip {clip_index} frame count differs from refined render"
            )
        if write_debug_artifacts:
            sf.write(
                boundary_dir / "old_clip.wav",
                old_clip,
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
                "clip_index": clip_index,
                "last_kept_word_id": last_kept_id,
                "first_omitted_word_id": first_omitted_id,
                "last_kept_word": words[last_kept_id].text,
                "first_omitted_word": words[first_omitted_id].text,
                "requested_context_start_word_id": (requested_context_start_id),
                "requested_context_end_word_id": requested_context_end_id,
                "context_start_word_id": context_start_id,
                "context_end_word_id": context_end_id,
                "kept_local_index": last_kept_id - context_start_id,
                "omitted_local_index": first_omitted_id - context_start_id,
                "local_words": local_words,
                "local_source_text": " ".join(word["text"] for word in local_words),
                "crop_start_sample": crop_start,
                "crop_end_sample": crop_end,
                "crop_start_seconds": crop_start / sample_rate,
                "crop_end_seconds": crop_end / sample_rate,
                "crop_duration_seconds": (crop_end - crop_start) / sample_rate,
                "crop_wav": str(context_path.resolve()),
                "raw_end_seconds": float(clip["raw_end_seconds"]),
                "silence_refined_end_seconds": float(clip["refined_end_seconds"]),
            }
        )
    return jobs


def _run_alignment_worker(
    *,
    jobs_path: Path,
    result_path: Path,
    alignment_python: Path,
    log_path: Path,
    language: str,
) -> dict[str, Any]:
    alignment_python = _absolute_without_resolving_symlinks(alignment_python)
    if not alignment_python.is_file():
        raise FileNotFoundError(
            "alignment Python does not exist; install VoiceCut's audio "
            f"dependencies or pass --alignment-python: {alignment_python}"
        )
    command = [
        str(alignment_python),
        "-m",
        "voicecut.forced_align_worker",
        "--jobs",
        str(jobs_path),
        "--output",
        str(result_path),
        "--language",
        language,
        "--device",
        "cpu",
    ]
    environment = os.environ.copy()
    source_dir = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = source_dir + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=environment,
    )
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
        raise RuntimeError(
            f"WhisperX alignment worker produced no result; see {log_path}"
        )
    result = read_json(result_path)
    if not isinstance(result, dict):
        raise RuntimeError("WhisperX worker result root is not an object")
    return result


def _validate_inputs(
    manifest_path: Path,
) -> tuple[dict[str, Any], Path, Path]:
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise RoughRenderError("refined manifest root must be an object")
    if manifest.get("renderer") != "streaming_plan_trailing_refinement_v1":
        raise RoughRenderError(
            "forced alignment requires a trailing-refinement v1 manifest"
        )
    audio_path = Path(str(manifest.get("source_audio", ""))).resolve()
    plan_path = Path(str(manifest.get("streaming_plan", ""))).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    if sha256_file(audio_path) != manifest.get("source_audio_sha256"):
        raise RoughRenderError("source audio changed after trailing refinement")
    if sha256_file(plan_path) != manifest.get("streaming_plan_sha256"):
        raise RoughRenderError("streaming plan changed after trailing refinement")
    return manifest, audio_path, plan_path


def render_forced_aligned_preview(
    *,
    refined_manifest_path: Path,
    output_dir: Path,
    alignment_python: Path = DEFAULT_ALIGNMENT_PYTHON,
    language: str = "en",
    crop_context_ms: float = CROP_CONTEXT_MS,
    max_shift_ms: float = MAX_BOUNDARY_SHIFT_MS,
    hard_boundary_fade_ms: float = HARD_BOUNDARY_FADE_MS,
    alignment_payload: dict[str, Any] | None = None,
    write_debug_artifacts: bool = True,
) -> dict[str, Any]:
    """Apply forced alignment only to clips marked ``hard_boundary``."""

    if language != "en":
        raise ValueError("this experiment requires the English align model")
    for name, value in (
        ("crop_context_ms", crop_context_ms),
        ("max_shift_ms", max_shift_ms),
        ("hard_boundary_fade_ms", hard_boundary_fade_ms),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    refined_manifest_path = refined_manifest_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"output directory must be empty for hard-boundary alignment: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_root = output_dir / "forced_alignment_debug"

    previous, audio_path, plan_path = _validate_inputs(refined_manifest_path)
    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise RoughRenderError("streaming plan root must be an object")
    words = load_plan_words(plan)
    merged_ranges = merge_adjacent_ranges(
        flatten_selected_ranges(plan, word_count=len(words))
    )
    previous_clips = previous.get("clips")
    if not isinstance(previous_clips, list):
        raise RoughRenderError("refined manifest clips must be a list")
    if len(previous_clips) != len(merged_ranges):
        raise RoughRenderError(
            "refined manifest clip count differs from frozen semantic plan"
        )
    for clip, source_range in zip(
        previous_clips,
        merged_ranges,
        strict=True,
    ):
        if (
            clip.get("source_word_start") != source_range.start_word_id
            or clip.get("source_word_end") != source_range.end_word_id
        ):
            raise RoughRenderError(
                "refined clip ranges differ from frozen semantic plan"
            )

    source_audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )
    sample_rate = int(sample_rate)
    total_samples, channel_count = source_audio.shape
    if (
        sample_rate != int(previous["source_sample_rate"])
        or total_samples != int(previous["source_frame_count"])
        or channel_count != int(previous["source_channel_count"])
    ):
        raise RoughRenderError("source audio geometry changed")
    mono = np.mean(source_audio, axis=1, dtype=np.float64).astype(np.float32)
    refined_wav_value = previous.get("rough_cut_refined_wav")
    if not isinstance(refined_wav_value, str):
        raise RoughRenderError("trailing manifest has no refined WAV")
    refined_wav_path = Path(refined_wav_value).resolve()
    if not refined_wav_path.is_file():
        raise FileNotFoundError(refined_wav_path)
    if sha256_file(refined_wav_path) != previous.get("rough_cut_refined_wav_sha256"):
        raise RoughRenderError("refined WAV changed after trailing refinement")
    refined_audio, refined_rate = sf.read(
        refined_wav_path,
        dtype="float32",
        always_2d=True,
    )
    if (
        int(refined_rate) != sample_rate
        or refined_audio.shape[1] != channel_count
        or len(refined_audio) != int(previous["refined_expected_output_frame_count"])
    ):
        raise RoughRenderError("refined WAV has unexpected geometry")

    hard_clips = [
        clip
        for clip in previous_clips
        if clip.get("boundary_method") == "hard_boundary"
    ]
    jobs = _prepare_alignment_jobs(
        hard_clips=hard_clips,
        words=words,
        source_audio=source_audio,
        previous_audio=refined_audio,
        sample_rate=sample_rate,
        debug_root=debug_root,
        crop_context_ms=crop_context_ms,
        write_debug_artifacts=write_debug_artifacts,
    )
    jobs_path = output_dir / "alignment_jobs.json"
    write_json(
        jobs_path,
        {
            "schema_version": 1,
            "source_audio": str(audio_path),
            "source_audio_sha256": previous["source_audio_sha256"],
            "language": language,
            "device": "cpu",
            "jobs": jobs,
        },
    )
    worker_result_path = output_dir / "alignment_worker_result.json"
    if alignment_payload is None and jobs:
        worker_result = _run_alignment_worker(
            jobs_path=jobs_path,
            result_path=worker_result_path,
            alignment_python=alignment_python,
            log_path=output_dir / "alignment_worker.log",
            language=language,
        )
    elif alignment_payload is not None:
        worker_result = alignment_payload
        write_json(worker_result_path, worker_result)
    else:
        worker_result = {
            "schema_version": 1,
            "backend": "whisperx_alignment",
            "language": language,
            "device": "cpu",
            "jobs": [],
            "model_load_skipped": "no hard boundaries require alignment",
        }
        write_json(worker_result_path, worker_result)
    worker_jobs = worker_result.get("jobs")
    if not isinstance(worker_jobs, list):
        raise RoughRenderError("alignment worker returned no jobs list")
    worker_by_clip = {
        int(item["clip_index"]): item
        for item in worker_jobs
        if isinstance(item, dict) and type(item.get("clip_index")) is int
    }
    job_by_clip = {int(job["clip_index"]): job for job in jobs}

    silence_ms = float(previous["configuration"]["inter_clip_silence_ms"])
    silence_samples = round(silence_ms * sample_rate / 1000.0)
    normal_fade_samples = round(
        float(previous["configuration"]["clip_fade_ms"]) * sample_rate / 1000.0
    )
    hard_fade_samples = round(hard_boundary_fade_ms * sample_rate / 1000.0)

    output_parts: list[np.ndarray] = []
    final_clips: list[dict[str, Any]] = []
    output_cursor = 0
    decisions: list[ForcedBoundaryDecision] = []
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

        previous_start = int(clip["refined_output_start_sample"])
        previous_end = int(clip["refined_output_end_sample"])
        if not 0 <= previous_start < previous_end <= len(refined_audio):
            raise RoughRenderError(
                f"previous refined clip {clip_index} has invalid output geometry"
            )
        previous_clip_audio = np.array(
            refined_audio[previous_start:previous_end],
            copy=True,
        )
        if len(previous_clip_audio) != int(clip["refined_frame_count"]):
            raise RoughRenderError(
                f"previous refined clip {clip_index} has changed geometry"
            )

        enhanced = {
            **clip,
            "silence_refined_end_seconds": float(clip["refined_end_seconds"]),
            "forced_aligned_kept_end_seconds": None,
            "forced_aligned_omitted_start_seconds": None,
            "final_cut_seconds": float(clip["refined_end_seconds"]),
            "final_shift_ms": 0.0,
            "first_omitted_word": (
                words[int(clip["source_word_end"])].text
                if int(clip["source_word_end"]) < len(words)
                else None
            ),
            "alignment_status": "not_applicable",
            "alignment_error": None,
            "alignment_granularity": None,
            "alignment_snap_method": None,
            "hard_boundary_fade_ms": None,
            "final_source_start_sample": int(clip["refined_source_start_sample"]),
            "final_source_end_sample": int(clip["refined_source_end_sample"]),
            "final_fade_in_samples": int(clip["refined_fade_in_samples"]),
            "final_fade_out_samples": int(clip["refined_fade_out_samples"]),
            "final_clip_wav": (
                clip.get("refined_clip_wav") if write_debug_artifacts else None
            ),
        }
        final_audio = previous_clip_audio
        if clip.get("boundary_method") == "hard_boundary":
            job = job_by_clip[clip_index]
            worker_job = worker_by_clip.get(
                clip_index,
                {
                    "clip_index": clip_index,
                    "error": "alignment worker omitted this boundary",
                    "aligned": None,
                },
            )
            decision = decide_forced_boundary(
                job=job,
                worker_job=worker_job,
                mono=mono,
                sample_rate=sample_rate,
                max_shift_ms=max_shift_ms,
            )
            decisions.append(decision)
            boundary_dir = debug_root / f"clip_{clip_index:03d}"
            if decision.status == "forced_alignment":
                source_start = int(clip["refined_source_start_sample"])
                source_end = round(decision.cut_seconds * sample_rate)
                if not 0 <= source_start < source_end <= total_samples:
                    raise RoughRenderError(
                        f"forced clip {clip_index} has invalid source geometry"
                    )
                final_audio = _apply_forced_boundary_fades(
                    source_audio[source_start:source_end],
                    fade_in_samples=normal_fade_samples,
                    fade_out_samples=hard_fade_samples,
                )
            if write_debug_artifacts:
                sf.write(
                    boundary_dir / "new_clip.wav",
                    final_audio,
                    sample_rate,
                    subtype="FLOAT",
                )
                _save_alignment_plot(
                    path=boundary_dir / "alignment_plot.png",
                    mono=mono,
                    sample_rate=sample_rate,
                    job=job,
                    decision=decision,
                )
                alignment_debug = {
                    "schema_version": 1,
                    "job": job,
                    "worker_result": worker_job,
                    "decision": asdict(decision),
                }
                write_json(boundary_dir / "alignment.json", alignment_debug)
            enhanced.update(
                {
                    "last_kept_word": job["last_kept_word"],
                    "first_omitted_word": job["first_omitted_word"],
                    "forced_aligned_kept_end_seconds": (decision.kept_end_seconds),
                    "forced_aligned_omitted_start_seconds": (
                        decision.omitted_start_seconds
                    ),
                    "final_cut_seconds": decision.cut_seconds,
                    "final_shift_ms": decision.shift_ms,
                    "boundary_method": decision.status,
                    "alignment_status": decision.status,
                    "alignment_error": decision.error,
                    "alignment_granularity": (decision.alignment_granularity),
                    "alignment_snap_method": decision.snap_method,
                    "alignment_low_amplitude_threshold_db": (
                        decision.low_amplitude_threshold_db
                    ),
                    "hard_boundary_fade_ms": (
                        hard_boundary_fade_ms
                        if decision.status == "forced_alignment"
                        else None
                    ),
                    "final_source_end_sample": (
                        round(decision.cut_seconds * sample_rate)
                        if decision.status == "forced_alignment"
                        else int(clip["refined_source_end_sample"])
                    ),
                    "final_fade_out_samples": (
                        hard_fade_samples
                        if decision.status == "forced_alignment"
                        else int(clip["refined_fade_out_samples"])
                    ),
                    "final_clip_wav": (
                        str((boundary_dir / "new_clip.wav").resolve())
                        if write_debug_artifacts
                        else None
                    ),
                    "forced_alignment_debug_dir": str(boundary_dir.resolve()),
                    "forced_alignment_context_wav": str(
                        (boundary_dir / "context.wav").resolve()
                    ),
                    "forced_alignment_old_clip_wav": (
                        str((boundary_dir / "old_clip.wav").resolve())
                        if write_debug_artifacts
                        else None
                    ),
                    "forced_alignment_new_clip_wav": (
                        str((boundary_dir / "new_clip.wav").resolve())
                        if write_debug_artifacts
                        else None
                    ),
                    "forced_alignment_json": (
                        str((boundary_dir / "alignment.json").resolve())
                        if write_debug_artifacts
                        else None
                    ),
                    "forced_alignment_plot": (
                        str((boundary_dir / "alignment_plot.png").resolve())
                        if write_debug_artifacts
                        else None
                    ),
                }
            )

        output_start = output_cursor
        output_end = output_start + len(final_audio)
        output_parts.append(final_audio)
        output_cursor = output_end
        enhanced.update(
            {
                "final_frame_count": len(final_audio),
                "final_output_start_sample": output_start,
                "final_output_end_sample": output_end,
                "final_output_start_seconds": output_start / sample_rate,
                "final_output_end_seconds": output_end / sample_rate,
            }
        )
        final_clips.append(enhanced)

    forced_audio = np.concatenate(output_parts, axis=0)
    expected_frames = sum(
        int(clip["final_frame_count"]) for clip in final_clips
    ) + silence_samples * (len(final_clips) - 1)
    if len(forced_audio) != expected_frames:
        raise RoughRenderError(
            "forced-aligned duration does not equal clips plus fixed silences"
        )
    forced_path = output_dir / "hard_boundary_aligned.wav"
    sf.write(forced_path, forced_audio, sample_rate, subtype="FLOAT")
    forced_info = sf.info(forced_path)
    if (
        int(forced_info.frames) != expected_frames
        or int(forced_info.samplerate) != sample_rate
        or int(forced_info.channels) != channel_count
    ):
        raise RoughRenderError("forced-aligned WAV has unexpected audio geometry")

    successful = [
        decision for decision in decisions if decision.status == "forced_alignment"
    ]
    manifest = {
        **previous,
        "schema_version": 1,
        "renderer": "streaming_plan_hard_boundary_alignment_v1",
        "previous_refined_manifest": str(refined_manifest_path),
        "previous_refined_manifest_sha256": sha256_file(refined_manifest_path),
        "previous_rough_cut_refined_wav": previous["rough_cut_refined_wav"],
        "hard_boundary_aligned_wav": str(forced_path.resolve()),
        "hard_boundary_aligned_wav_sha256": sha256_file(forced_path),
        "hard_boundary_aligned_duration_seconds": (expected_frames / sample_rate),
        "hard_boundary_aligned_expected_output_frame_count": expected_frames,
        "debug_artifacts_written": write_debug_artifacts,
        "forced_alignment_configuration": {
            "backend": "whisperx_alignment",
            "language": language,
            "device": "cpu",
            "alignment_python": str(
                _absolute_without_resolving_symlinks(alignment_python)
            ),
            "context_words_before": CONTEXT_WORDS_PER_SIDE,
            "context_words_after_omitted": CONTEXT_WORDS_PER_SIDE,
            "crop_context_ms": crop_context_ms,
            "maximum_boundary_shift_ms": max_shift_ms,
            "hard_boundary_fade_ms": hard_boundary_fade_ms,
            "inter_clip_silence_ms": silence_ms,
        },
        "alignment_jobs": str(jobs_path.resolve()),
        "alignment_worker_result": str(worker_result_path.resolve()),
        "hard_boundaries_found": len(hard_clips),
        "successfully_aligned": len(successful),
        "alignment_failures": len(decisions) - len(successful),
        "average_boundary_shift_ms": (
            float(np.mean([decision.shift_ms for decision in successful]))
            if successful
            else 0.0
        ),
        "clips": final_clips,
    }
    manifest_path = output_dir / "render_manifest_forced_aligned.json"
    write_json(manifest_path, manifest)
    return manifest
