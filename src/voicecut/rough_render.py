#!/usr/bin/env python3
"""Render a plan-grounded rough WAV without making semantic decisions.

This intentionally simple renderer consumes only immutable word IDs selected
by an existing streaming narration plan. It does not transcribe, call an LLM,
inspect canonical text, or perform advanced audio processing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from .common import read_json, sha256_file, write_json


EDGE_PADDING_MS = 30.0
CLIP_FADE_MS = 5.0
INTER_CLIP_SILENCE_MS = 80.0


class RoughRenderError(RuntimeError):
    """The semantic plan cannot be rendered without violating its ranges."""


@dataclass(frozen=True)
class PlanWord:
    id: int
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class SelectedRange:
    start_word_id: int
    end_word_id: int
    thought_index: int
    thought_range_index: int


@dataclass(frozen=True)
class MergedRange:
    start_word_id: int
    end_word_id: int
    original_ranges: tuple[SelectedRange, ...]


def load_plan_words(plan: dict[str, Any]) -> list[PlanWord]:
    """Load the immutable word ledger and require ID == list position."""

    raw_words = plan.get("words")
    if not isinstance(raw_words, list) or not raw_words:
        raise RoughRenderError("plan.words must be a non-empty list")
    words: list[PlanWord] = []
    for position, raw_word in enumerate(raw_words):
        if not isinstance(raw_word, dict):
            raise RoughRenderError(f"plan.words[{position}] must be an object")
        word_id = raw_word.get("id")
        text = raw_word.get("text")
        start = raw_word.get("start")
        end = raw_word.get("end")
        if type(word_id) is not int or word_id != position:
            raise RoughRenderError(
                "plan word IDs must be unique, contiguous, and equal to their "
                f"list positions; position {position} has ID {word_id!r}"
            )
        if not isinstance(text, str) or not text.strip():
            raise RoughRenderError(f"plan word {word_id} has no text")
        if type(start) not in {int, float} or type(end) not in {int, float}:
            raise RoughRenderError(f"plan word {word_id} has invalid timestamps")
        start_value = float(start)
        end_value = float(end)
        if (
            not math.isfinite(start_value)
            or not math.isfinite(end_value)
            or end_value < start_value
        ):
            raise RoughRenderError(
                f"plan word {word_id} has invalid timestamp geometry"
            )
        words.append(
            PlanWord(
                id=word_id,
                text=text.strip(),
                start=start_value,
                end=end_value,
            )
        )
    return words


def flatten_selected_ranges(
    plan: dict[str, Any],
    *,
    word_count: int,
) -> list[SelectedRange]:
    """Read committed source ranges in their declared chronological order."""

    committed = plan.get("committed")
    if not isinstance(committed, list):
        raise RoughRenderError("plan.committed must be a list")
    flattened: list[SelectedRange] = []
    previous_end = 0
    for thought_index, thought in enumerate(committed):
        if not isinstance(thought, dict):
            raise RoughRenderError(f"plan.committed[{thought_index}] must be an object")
        raw_ranges = thought.get("source_ranges")
        if not isinstance(raw_ranges, list):
            raise RoughRenderError(
                f"plan.committed[{thought_index}].source_ranges must be a list"
            )
        for range_index, raw_range in enumerate(raw_ranges):
            if not isinstance(raw_range, dict):
                raise RoughRenderError(
                    f"committed range {thought_index}:{range_index} must be an object"
                )
            start = raw_range.get("start_word_id")
            end = raw_range.get("end_word_id")
            if type(start) is not int or type(end) is not int:
                raise RoughRenderError("source-range IDs must be integers")
            if not 0 <= start < end <= word_count:
                raise RoughRenderError(
                    f"invalid source range [{start}, {end}) for {word_count} words"
                )
            if start < previous_end:
                raise RoughRenderError(
                    "committed source ranges overlap or move backward: "
                    f"[{start}, {end}) follows a range ending at {previous_end}"
                )
            flattened.append(
                SelectedRange(
                    start_word_id=start,
                    end_word_id=end,
                    thought_index=thought_index,
                    thought_range_index=range_index,
                )
            )
            previous_end = end
    if not flattened:
        raise RoughRenderError("the semantic plan contains no selected ranges")

    declared = plan.get("selected_source_ranges")
    if declared is not None:
        expected = [
            {
                "start_word_id": source_range.start_word_id,
                "end_word_id": source_range.end_word_id,
            }
            for source_range in flattened
        ]
        if declared != expected:
            raise RoughRenderError(
                "plan.selected_source_ranges does not match committed ranges"
            )
    return flattened


def merge_adjacent_ranges(
    ranges: Sequence[SelectedRange],
) -> list[MergedRange]:
    """Merge only ranges whose half-open word boundaries touch exactly."""

    merged: list[MergedRange] = []
    for source_range in ranges:
        if not source_range.start_word_id < source_range.end_word_id:
            raise RoughRenderError("cannot merge an empty or reversed range")
        if merged and source_range.start_word_id < merged[-1].end_word_id:
            raise RoughRenderError("cannot merge overlapping or backward ranges")
        if merged and source_range.start_word_id == merged[-1].end_word_id:
            previous = merged[-1]
            merged[-1] = MergedRange(
                start_word_id=previous.start_word_id,
                end_word_id=source_range.end_word_id,
                original_ranges=(
                    *previous.original_ranges,
                    source_range,
                ),
            )
        else:
            merged.append(
                MergedRange(
                    start_word_id=source_range.start_word_id,
                    end_word_id=source_range.end_word_id,
                    original_ranges=(source_range,),
                )
            )
    return merged


def omitted_word_intervals(
    *,
    word_count: int,
    merged_ranges: Sequence[MergedRange],
) -> list[tuple[int, int]]:
    """Return the complete complement of the selected word IDs."""

    omitted: list[tuple[int, int]] = []
    cursor = 0
    for source_range in merged_ranges:
        if cursor < source_range.start_word_id:
            omitted.append((cursor, source_range.start_word_id))
        cursor = source_range.end_word_id
    if cursor < word_count:
        omitted.append((cursor, word_count))
    return omitted


def timestamp_to_sample(
    seconds: float,
    *,
    sample_rate: int,
    total_samples: int,
    rounding: str,
) -> int:
    """Convert a timestamp without inventing one-sample floating-point gaps.

    Whisper timestamps often represent one shared word boundary with the same
    binary float. Multiplication by the sample rate can nevertheless produce
    a value such as ``476160.00000000006``. Treat values already within one
    millionth of a sample of an integer as that exact sample before applying
    the requested conservative rounding direction.
    """

    if (
        type(seconds) not in {int, float}
        or not math.isfinite(float(seconds))
        or float(seconds) < 0.0
    ):
        raise ValueError("timestamp must be finite and non-negative")
    if sample_rate <= 0 or total_samples <= 0:
        raise ValueError("sample_rate and total_samples must be positive")
    if rounding not in {"floor", "ceil", "nearest"}:
        raise ValueError("rounding must be floor, ceil, or nearest")
    scaled = float(seconds) * sample_rate
    nearest = round(scaled)
    if math.isclose(scaled, nearest, rel_tol=0.0, abs_tol=1e-6):
        sample = nearest
    elif rounding == "floor":
        sample = math.floor(scaled)
    elif rounding == "ceil":
        sample = math.ceil(scaled)
    else:
        sample = nearest
    return max(0, min(total_samples, int(sample)))


def range_to_sample_bounds(
    *,
    words: Sequence[PlanWord],
    source_range: MergedRange,
    sample_rate: int,
    total_samples: int,
    edge_padding_ms: float = EDGE_PADDING_MS,
) -> tuple[int, int, int, int]:
    """Map one selected range to bounded source sample indices.

    Returns ``clip_start, clip_end, selected_start, selected_end``. Starts use
    floor and ends use ceil so the selected timestamp interval is preserved.
    Excluded-neighbor fences use ceil/floor in the safe direction.
    """

    if sample_rate <= 0 or total_samples <= 0:
        raise ValueError("sample_rate and total_samples must be positive")
    if not math.isfinite(edge_padding_ms) or edge_padding_ms < 0.0:
        raise ValueError("edge_padding_ms must be finite and non-negative")
    start_id = source_range.start_word_id
    end_id = source_range.end_word_id
    if not 0 <= start_id < end_id <= len(words):
        raise RoughRenderError(
            f"range [{start_id}, {end_id}) does not fit the word ledger"
        )

    selected_start = timestamp_to_sample(
        words[start_id].start,
        sample_rate=sample_rate,
        total_samples=total_samples,
        rounding="floor",
    )
    selected_end = timestamp_to_sample(
        words[end_id - 1].end,
        sample_rate=sample_rate,
        total_samples=total_samples,
        rounding="ceil",
    )
    if selected_end <= selected_start:
        raise RoughRenderError(
            f"selected range [{start_id}, {end_id}) collapses in source audio"
        )

    padding_samples = round(edge_padding_ms * sample_rate / 1000.0)
    earliest_start = 0
    if start_id > 0:
        earliest_start = timestamp_to_sample(
            words[start_id - 1].end,
            sample_rate=sample_rate,
            total_samples=total_samples,
            rounding="ceil",
        )
        if earliest_start > selected_start:
            # ASR timestamps can genuinely overlap. Preserving the selected
            # word takes priority; there is no safe leading padding in this
            # case, and the ambiguity remains visible in the manifest.
            earliest_start = selected_start

    latest_end = total_samples
    if end_id < len(words):
        latest_end = timestamp_to_sample(
            words[end_id].start,
            sample_rate=sample_rate,
            total_samples=total_samples,
            rounding="floor",
        )
        if latest_end < selected_end:
            # Do not shorten a selected word to obey an approximate timestamp
            # on an excluded neighbor. Keep the full selected interval and
            # let trailing refinement classify this as a hard boundary.
            latest_end = selected_end

    clip_start = max(earliest_start, selected_start - padding_samples)
    clip_end = min(latest_end, selected_end + padding_samples)
    if not clip_start <= selected_start < selected_end <= clip_end:
        raise RoughRenderError(
            f"padding fences do not contain selected range [{start_id}, {end_id})"
        )
    return clip_start, clip_end, selected_start, selected_end


def apply_linear_edge_fades(
    samples: np.ndarray,
    *,
    requested_fade_samples: int,
) -> tuple[np.ndarray, int]:
    """Apply simple linear fades while leaving the clip interior untouched."""

    if samples.ndim != 2:
        raise ValueError("audio samples must have shape (frames, channels)")
    if requested_fade_samples < 0:
        raise ValueError("requested_fade_samples must be non-negative")
    rendered = np.array(samples, dtype=np.float32, copy=True)
    fade_samples = min(requested_fade_samples, len(rendered) // 2)
    if fade_samples:
        ramp = np.linspace(
            0.0,
            1.0,
            fade_samples,
            endpoint=True,
            dtype=np.float32,
        )
        rendered[:fade_samples] *= ramp[:, None]
        rendered[-fade_samples:] *= ramp[::-1, None]
    return rendered, fade_samples


def _text(words: Sequence[PlanWord], start: int, end: int) -> str:
    return " ".join(word.text for word in words[start:end]).strip()


def render_rough_cut(
    *,
    audio_path: Path,
    plan_path: Path,
    output_dir: Path,
    edge_padding_ms: float = EDGE_PADDING_MS,
    clip_fade_ms: float = CLIP_FADE_MS,
    inter_clip_silence_ms: float = INTER_CLIP_SILENCE_MS,
    write_debug_artifacts: bool = True,
) -> dict[str, Any]:
    """Render an inspectable WAV directly from committed plan word ranges."""

    for name, value in (
        ("edge_padding_ms", edge_padding_ms),
        ("clip_fade_ms", clip_fade_ms),
        ("inter_clip_silence_ms", inter_clip_silence_ms),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    audio_path = audio_path.resolve()
    plan_path = plan_path.resolve()
    output_dir = output_dir.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        entries = list(output_dir.iterdir())
        abandoned_clips = output_dir / "clips"
        if (
            entries == [abandoned_clips]
            and abandoned_clips.is_dir()
            and not any(abandoned_clips.iterdir())
        ):
            # A prior pre-render validation failure may have created only the
            # empty directory skeleton. It contains no user data or artifact.
            abandoned_clips.rmdir()
            output_dir.rmdir()
        else:
            raise RuntimeError(
                f"output directory must be empty for a rough render: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    if write_debug_artifacts:
        clips_dir.mkdir()

    plan = read_json(plan_path)
    if not isinstance(plan, dict):
        raise RoughRenderError("streaming plan root must be an object")
    if plan.get("status") != "complete":
        raise RoughRenderError("only a complete streaming plan can be rendered")
    words = load_plan_words(plan)
    original_ranges = flatten_selected_ranges(plan, word_count=len(words))
    merged_ranges = merge_adjacent_ranges(original_ranges)
    omitted_ranges = omitted_word_intervals(
        word_count=len(words),
        merged_ranges=merged_ranges,
    )

    selected_ids = {
        word_id
        for source_range in original_ranges
        for word_id in range(
            source_range.start_word_id,
            source_range.end_word_id,
        )
    }
    for merged_range in merged_ranges:
        merged_ids = set(
            range(
                merged_range.start_word_id,
                merged_range.end_word_id,
            )
        )
        if not merged_ids.issubset(selected_ids):
            raise RoughRenderError(
                "an omitted source word appears inside a merged range"
            )
    omitted_ids = {
        word_id for start, end in omitted_ranges for word_id in range(start, end)
    }
    if selected_ids & omitted_ids or selected_ids | omitted_ids != set(
        range(len(words))
    ):
        raise RoughRenderError(
            "selected and omitted word intervals are not complementary"
        )

    source_info = sf.info(audio_path)
    source_audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )
    sample_rate = int(sample_rate)
    if sample_rate != int(source_info.samplerate):
        raise RoughRenderError("source sample-rate changed while reading")
    total_samples, channel_count = source_audio.shape
    if total_samples != int(source_info.frames):
        raise RoughRenderError("source frame count changed while reading")

    requested_fade_samples = round(clip_fade_ms * sample_rate / 1000.0)
    silence_samples = round(inter_clip_silence_ms * sample_rate / 1000.0)
    output_parts: list[np.ndarray] = []
    clip_manifest: list[dict[str, Any]] = []
    output_cursor = 0

    for clip_index, merged_range in enumerate(merged_ranges):
        if clip_index:
            output_parts.append(
                np.zeros(
                    (silence_samples, channel_count),
                    dtype=np.float32,
                )
            )
            output_cursor += silence_samples
        (
            source_start,
            source_end,
            selected_start,
            selected_end,
        ) = range_to_sample_bounds(
            words=words,
            source_range=merged_range,
            sample_rate=sample_rate,
            total_samples=total_samples,
            edge_padding_ms=edge_padding_ms,
        )
        source_clip = source_audio[source_start:source_end]
        rendered_clip, actual_fade_samples = apply_linear_edge_fades(
            source_clip,
            requested_fade_samples=requested_fade_samples,
        )
        clip_path = (
            clips_dir / f"clip_{clip_index:03d}.wav" if write_debug_artifacts else None
        )
        if clip_path is not None:
            sf.write(
                clip_path,
                rendered_clip,
                sample_rate,
                subtype="FLOAT",
            )
        output_start = output_cursor
        output_end = output_start + len(rendered_clip)
        output_parts.append(rendered_clip)
        output_cursor = output_end
        clip_manifest.append(
            {
                "clip_index": clip_index,
                "source_word_start": merged_range.start_word_id,
                "source_word_end": merged_range.end_word_id,
                "source_start_seconds": source_start / sample_rate,
                "source_end_seconds": source_end / sample_rate,
                "source_text": _text(
                    words,
                    merged_range.start_word_id,
                    merged_range.end_word_id,
                ),
                "output_start_seconds": output_start / sample_rate,
                "output_end_seconds": output_end / sample_rate,
                "merged_original_ranges": [
                    {
                        "start_word_id": source_range.start_word_id,
                        "end_word_id": source_range.end_word_id,
                    }
                    for source_range in merged_range.original_ranges
                ],
                "source_start_sample": source_start,
                "source_end_sample": source_end,
                "selected_start_sample": selected_start,
                "selected_end_sample": selected_end,
                "output_start_sample": output_start,
                "output_end_sample": output_end,
                "frame_count": len(rendered_clip),
                "leading_padding_samples": selected_start - source_start,
                "trailing_padding_samples": source_end - selected_end,
                "previous_excluded_word_overlap_ms": (
                    max(
                        0.0,
                        words[merged_range.start_word_id - 1].end
                        - words[merged_range.start_word_id].start,
                    )
                    * 1000.0
                    if merged_range.start_word_id > 0
                    else 0.0
                ),
                "next_excluded_word_overlap_ms": (
                    max(
                        0.0,
                        words[merged_range.end_word_id - 1].end
                        - words[merged_range.end_word_id].start,
                    )
                    * 1000.0
                    if merged_range.end_word_id < len(words)
                    else 0.0
                ),
                "start_boundary_requires_alignment": (
                    merged_range.start_word_id > 0
                    and words[merged_range.start_word_id - 1].end
                    > words[merged_range.start_word_id].start
                ),
                "end_boundary_requires_alignment": (
                    merged_range.end_word_id < len(words)
                    and words[merged_range.end_word_id - 1].end
                    > words[merged_range.end_word_id].start
                ),
                "fade_samples": actual_fade_samples,
                "clip_wav": (
                    str(clip_path.resolve()) if clip_path is not None else None
                ),
                "clip_wav_sha256": (
                    sha256_file(clip_path) if clip_path is not None else None
                ),
            }
        )

    rough_audio = np.concatenate(output_parts, axis=0)
    expected_output_samples = sum(
        int(clip["frame_count"]) for clip in clip_manifest
    ) + silence_samples * (len(clip_manifest) - 1)
    if len(rough_audio) != expected_output_samples:
        raise RoughRenderError(
            "rough-cut duration does not equal clips plus inserted silences"
        )
    rough_cut_path = output_dir / "rough_cut.wav"
    if write_debug_artifacts:
        sf.write(
            rough_cut_path,
            rough_audio,
            sample_rate,
            subtype="FLOAT",
        )
        rough_info = sf.info(rough_cut_path)
        if int(rough_info.frames) != expected_output_samples:
            raise RoughRenderError(
                "written rough-cut duration differs by more than one sample"
            )
        if (
            int(rough_info.samplerate) != sample_rate
            or int(rough_info.channels) != channel_count
        ):
            raise RoughRenderError(
                "rough cut did not preserve source sample rate and channel count"
            )

    omitted_manifest = [
        {
            "start_word_id": start,
            "end_word_id": end,
            "source_start_seconds": words[start].start,
            "source_end_seconds": words[end - 1].end,
            "source_text": _text(words, start, end),
        }
        for start, end in omitted_ranges
    ]
    manifest = {
        "schema_version": 1,
        "renderer": "streaming_plan_rough_cut_v1",
        "source_audio": str(audio_path),
        "source_audio_sha256": sha256_file(audio_path),
        "source_sample_rate": sample_rate,
        "source_channel_count": channel_count,
        "source_frame_count": total_samples,
        "source_duration_seconds": total_samples / sample_rate,
        "streaming_plan": str(plan_path),
        "streaming_plan_sha256": sha256_file(plan_path),
        "configuration": {
            "edge_padding_ms": edge_padding_ms,
            "clip_fade_ms": clip_fade_ms,
            "inter_clip_silence_ms": inter_clip_silence_ms,
            "fade_shape": "linear",
            "output_subtype": "FLOAT",
        },
        "original_selected_range_count": len(original_ranges),
        "merged_rendered_clip_count": len(merged_ranges),
        "selected_word_count": len(selected_ids),
        "omitted_word_count": len(omitted_ids),
        "inserted_silence_samples": (silence_samples * (len(clip_manifest) - 1)),
        "expected_output_frame_count": expected_output_samples,
        "rough_cut_duration_seconds": expected_output_samples / sample_rate,
        "debug_artifacts_written": write_debug_artifacts,
        "rough_cut_wav": (
            str(rough_cut_path.resolve()) if write_debug_artifacts else None
        ),
        "rough_cut_wav_sha256": (
            sha256_file(rough_cut_path) if write_debug_artifacts else None
        ),
        "clips": clip_manifest,
        "omitted_word_intervals": omitted_manifest,
    }
    write_json(output_dir / "render_manifest.json", manifest)
    return manifest
