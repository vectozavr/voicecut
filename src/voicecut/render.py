#!/usr/bin/env python3
"""Refine edit boundaries against the waveform and render the narration EDL."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .common import EditPiece, load_aliases, read_json, tokenize, write_json
from .plan import load_observed_words


START_BOUNDARY_REASONS = {
    "active_start_without_strong_ctc",
    "ctc_asr_start_disagreement",
    "low_ctc_first_word_score",
}
END_BOUNDARY_REASONS = {
    "active_end_without_strong_ctc",
    "ctc_asr_end_disagreement",
    "low_ctc_last_word_score",
}


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def rms_db(audio: np.ndarray) -> float:
    if not len(audio):
        return -200.0
    value = math.sqrt(float(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * math.log10(max(value, 1e-10))


def local_rms_db(audio: np.ndarray, center: int, radius: int) -> float:
    start = max(0, center - radius)
    end = min(len(audio), center + radius)
    return rms_db(audio[start:end])


def snap_zero_crossing(audio: np.ndarray, proposed: int, radius: int) -> int:
    start = max(0, proposed - radius)
    end = min(len(audio), proposed + radius + 1)
    local = audio[start:end]
    if len(local) < 2:
        return max(0, min(len(audio), proposed))
    crossings = np.flatnonzero(np.signbit(local[:-1]) != np.signbit(local[1:])) + 1
    if len(crossings):
        return start + int(crossings[np.argmin(np.abs(start + crossings - proposed))])
    return start + int(np.argmin(np.abs(local)))


def clamp_snapped_boundaries(
    snapped_start: int,
    snapped_end: int,
    *,
    lower_fence: int,
    speech_start: int,
    speech_end: int,
    upper_fence: int,
) -> tuple[int, int]:
    """Keep zero-crossing refinement outside speech and inside hard fences."""

    source_start = max(lower_fence, min(speech_start, snapped_start))
    source_end = max(speech_end, min(upper_fence, snapped_end))
    return source_start, source_end


def boundary_before(
    *,
    target: int,
    lower: int,
    rms_frames: np.ndarray,
    frame_samples: int,
    sample_rate: int,
    threshold_db: float,
) -> tuple[int, str]:
    target_frame = min(len(rms_frames), math.ceil(target / frame_samples))
    lower_frame = max(0, lower // frame_samples)
    local = rms_frames[lower_frame:target_frame] < threshold_db
    minimum_frames = max(2, round(0.020 * sample_rate / frame_samples))
    runs = [run for run in true_runs(local) if run[1] - run[0] >= minimum_frames]
    if runs:
        start, end = runs[-1]
        absolute_start = (lower_frame + start) * frame_samples
        absolute_end = min(target, (lower_frame + end) * frame_samples)
        pre_roll = round(0.038 * sample_rate)
        return max(absolute_start, absolute_end - pre_roll), "quiet_valley"
    return max(lower, target - round(0.048 * sample_rate)), "active_fallback"


def boundary_after(
    *,
    target: int,
    upper: int,
    rms_frames: np.ndarray,
    frame_samples: int,
    sample_rate: int,
    threshold_db: float,
) -> tuple[int, str]:
    target_frame = max(0, target // frame_samples)
    upper_frame = min(len(rms_frames), math.ceil(upper / frame_samples))
    local = rms_frames[target_frame:upper_frame] < threshold_db
    minimum_frames = max(2, round(0.025 * sample_rate / frame_samples))
    runs = [run for run in true_runs(local) if run[1] - run[0] >= minimum_frames]
    if runs:
        start, end = runs[0]
        absolute_start = max(target, (target_frame + start) * frame_samples)
        absolute_end = min(upper, (target_frame + end) * frame_samples)
        post_roll = round(0.080 * sample_rate)
        return min(absolute_end, absolute_start + post_roll), "quiet_valley"
    return min(upper, target + round(0.090 * sample_rate)), "active_fallback"


def ctc_boundaries(
    ctc_data: dict[str, Any] | None,
    plan: dict[str, Any],
) -> dict[int, dict[str, float | None]]:
    if not ctc_data:
        return {}
    output: dict[int, dict[str, float | None]] = {}
    segments = list(ctc_data.get("segments", []))
    explicit_ids = all("phrase_index" in segment for segment in segments)
    if explicit_ids:
        identified = [
            (int(segment["phrase_index"]), segment) for segment in segments
        ]
    else:
        phrase_indices = list(plan.get("ctc_segment_phrase_indices", []))
        # Legacy positional output is safe only when the aligner returned
        # exactly one segment per input.  Otherwise discard it rather than
        # shifting timestamps onto unrelated phrases.
        if len(phrase_indices) != len(segments):
            return {}
        identified = list(zip((int(value) for value in phrase_indices), segments))
    for phrase_index, segment in identified:
        words = [
            word
            for word in segment.get("words", [])
            if "start" in word and "end" in word
        ]
        if not words:
            continue
        scores = [
            float(word["score"])
            for word in words
            if word.get("score") is not None and math.isfinite(float(word["score"]))
        ]
        first_score = (
            float(words[0]["score"])
            if words[0].get("score") is not None
            and math.isfinite(float(words[0]["score"]))
            else None
        )
        last_score = (
            float(words[-1]["score"])
            if words[-1].get("score") is not None
            and math.isfinite(float(words[-1]["score"]))
            else None
        )
        output[int(phrase_index)] = {
            "start": float(words[0]["start"]),
            "end": float(words[-1]["end"]),
            "mean_score": sum(scores) / len(scores) if scores else None,
            "first_score": first_score,
            "last_score": last_score,
        }
    return output


def map_ctc_words_to_observed(
    ctc_data: dict[str, Any] | None,
    observed_words: list[Any],
    aliases: dict[str, list[str]],
    *,
    maximum_midpoint_distance: float = 0.80,
) -> dict[int, dict[str, Any]]:
    """Map selected CTC words back to the matching source-ASR word indices.

    The CTC file contains only the selected script phrases, while the source
    transcript intentionally contains every retry.  Matching by normalized
    token and nearby source time therefore gives retained words precise
    forced-alignment fences without accidentally mapping them onto a discarded
    take.
    """

    if not ctc_data:
        return {}
    rows: list[dict[str, Any]] = []
    for segment in ctc_data.get("segments", []):
        for raw in segment.get("words", []):
            if "start" not in raw or "end" not in raw:
                continue
            normalized = tokenize(str(raw.get("word", "")), aliases)
            if len(normalized) != 1:
                continue
            start = float(raw["start"])
            end = float(raw["end"])
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                continue
            rows.append(
                {
                    "token": normalized[0],
                    "word": str(raw.get("word", "")),
                    "start": start,
                    "end": end,
                    "score": (
                        float(raw["score"])
                        if raw.get("score") is not None
                        and math.isfinite(float(raw["score"]))
                        else None
                    ),
                    "phrase_index": (
                        int(segment["phrase_index"])
                        if "phrase_index" in segment
                        else None
                    ),
                }
            )
    rows.sort(key=lambda row: (float(row["start"]), float(row["end"])))

    mapping: dict[int, dict[str, Any]] = {}
    used_observed: set[int] = set()
    for row in rows:
        midpoint = (float(row["start"]) + float(row["end"])) / 2.0
        candidates: list[tuple[float, int]] = []
        for word in observed_words:
            word_index = int(word.word_index)
            if word_index in used_observed or list(word.tokens) != [row["token"]]:
                continue
            word_midpoint = (float(word.start) + float(word.end)) / 2.0
            distance = abs(word_midpoint - midpoint)
            if distance <= maximum_midpoint_distance:
                candidates.append((distance, word_index))
        if not candidates:
            continue
        _, word_index = min(candidates)
        mapping[word_index] = row
        used_observed.add(word_index)
    return mapping


def corroborated_inward_ctc_start(
    source: np.ndarray,
    *,
    sample_rate: int,
    asr_start: float,
    ctc_start: float | None,
    ctc_score: float | None,
    noise_floor_db: float,
) -> tuple[float, bool]:
    """Use a later CTC word onset only when the waveform confirms the onset."""

    if (
        ctc_start is None
        or ctc_score is None
        or ctc_score < 0.60
        or not 0.035 <= ctc_start - asr_start <= 0.30
    ):
        return asr_start, False
    pre_start = max(0, round((ctc_start - 0.100) * sample_rate))
    pre_end = max(pre_start, round((ctc_start - 0.035) * sample_rate))
    post_start = max(0, round(ctc_start * sample_rate))
    post_end = min(len(source), round((ctc_start + 0.080) * sample_rate))
    before = rms_db(source[pre_start:pre_end])
    after = rms_db(source[post_start:post_end])
    corroborated = (
        after >= noise_floor_db + 12.0
        and after >= before + 5.0
    )
    return (ctc_start, True) if corroborated else (asr_start, False)


def fuse_asr_ctc_interval(
    asr_start: float,
    asr_end: float,
    ctc_value: dict[str, float | None] | None,
    *,
    maximum_extension: float = 0.30,
    minimum_edge_score: float = 0.45,
) -> tuple[float, float]:
    """Conservatively fuse CTC edges without ever shortening the ASR span."""

    if not ctc_value:
        return asr_start, asr_end

    def finite_value(key: str) -> float | None:
        value = ctc_value.get(key)
        if value is None:
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None

    ctc_start = finite_value("start")
    ctc_end = finite_value("end")
    first_score = finite_value("first_score")
    last_score = finite_value("last_score")
    speech_start = asr_start
    speech_end = asr_end
    if (
        ctc_start is not None
        and ctc_start < asr_start
        and asr_start - ctc_start <= maximum_extension
        and first_score is not None
        and first_score >= minimum_edge_score
    ):
        speech_start = ctc_start
    if (
        ctc_end is not None
        and ctc_end > asr_end
        and ctc_end - asr_end <= maximum_extension
        and last_score is not None
        and last_score >= minimum_edge_score
    ):
        speech_end = ctc_end
    return speech_start, speech_end


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return np.mean(audio, axis=1, dtype=np.float32), int(sample_rate)


def choose_pause_seconds(
    current: dict[str, Any],
    following: dict[str, Any] | None,
) -> float:
    if following is None:
        return 0.55
    if int(current["unit_index"]) == int(following["unit_index"]):
        pause_class = str(current["pause_after"])
        return {"phrase": 0.105, "clause": 0.165, "sentence": 0.245}.get(
            pause_class, 0.12
        )
    if bool(following.get("cue_before")):
        return 0.58
    if int(current["paragraph_index"]) != int(following["paragraph_index"]):
        return 0.36
    return 0.27


def equal_power_crossfade(left: np.ndarray, right: np.ndarray, length: int) -> np.ndarray:
    length = min(length, len(left), len(right))
    if length <= 0:
        return np.empty(0, dtype=np.float32)
    theta = np.linspace(0.0, math.pi / 2.0, length, endpoint=True)
    return (
        left[-length:] * np.cos(theta).astype(np.float32)
        + right[:length] * np.sin(theta).astype(np.float32)
    )


def room_tone_segment(
    sources: list[np.ndarray],
    length: int,
    index: int,
    sample_rate: int = 48000,
) -> np.ndarray:
    if not sources:
        return np.zeros(length, dtype=np.float32)
    source = sources[index % len(sources)]
    if not len(source):
        return np.zeros(length, dtype=np.float32)
    if length <= len(source):
        available = len(source) - length
        offset = (index * 7919) % (available + 1) if available else 0
        return source[offset : offset + length].copy()
    fade = min(round(0.015 * sample_rate), len(source) // 4)
    result = source.copy()
    while len(result) < length:
        # Crossfade the true end and beginning of the captured region.  Rolling
        # a tile moves its wrap discontinuity inside the tile, beyond the
        # crossfade, which creates a click on the following loop.
        tile = source
        overlap = min(fade, len(result), len(tile))
        if overlap:
            result = np.concatenate(
                (
                    result[:-overlap],
                    equal_power_crossfade(result, tile, overlap),
                    tile[overlap:],
                )
            )
        else:
            result = np.concatenate((result, tile))
    return result[:length].astype(np.float32, copy=False)


def synthetic_room_tones(
    *,
    sample_rate: int,
    noise_floor_db: float,
    count: int = 4,
) -> list[np.ndarray]:
    """Create deterministic, low-level shaped noise when no safe room exists."""

    target_db = float(np.clip(noise_floor_db, -80.0, -58.0))
    target_rms = 10.0 ** (target_db / 20.0)
    length = max(1, round(2.0 * sample_rate))
    sources: list[np.ndarray] = []
    for index in range(count):
        rng = np.random.default_rng(271828 + index)
        white = rng.standard_normal(length + 8).astype(np.float32)
        # A short decorrelation filter avoids harsh, perfectly white silence
        # fill while remaining neutral enough for arbitrary microphones.
        kernel = np.array([0.18, 0.34, 0.48, 0.34, 0.18], dtype=np.float32)
        shaped = np.convolve(white, kernel, mode="valid")[:length].astype(np.float32)
        current_rms = math.sqrt(float(np.mean(np.square(shaped, dtype=np.float64))))
        shaped *= target_rms / max(current_rms, 1e-12)
        sources.append(shaped)
    return sources


def breath_like(
    segment: np.ndarray,
    sample_rate: int,
    noise_floor_db: float,
) -> tuple[bool, dict[str, float]]:
    level = rms_db(segment)
    peak = float(np.max(np.abs(segment))) if len(segment) else 0.0
    crest_db = 20.0 * math.log10(max(peak, 1e-10)) - level
    if len(segment) < max(32, round(0.07 * sample_rate)):
        return False, {"rms_db": level, "crest_db": crest_db, "hf_ratio_db": -200.0}
    windowed = segment * np.hanning(len(segment)).astype(np.float32)
    spectrum = np.square(np.abs(np.fft.rfft(windowed)), dtype=np.float64)
    frequencies = np.fft.rfftfreq(len(segment), 1.0 / sample_rate)
    total = float(np.sum(spectrum[(frequencies >= 80) & (frequencies <= 8000)]))
    high = float(np.sum(spectrum[(frequencies >= 500) & (frequencies <= 8000)]))
    ratio_db = 10.0 * math.log10(max(high, 1e-20) / max(total, 1e-20))
    accepted = (
        noise_floor_db + 4.0 <= level <= -30.0
        and ratio_db > -5.7
        and crest_db < 18.0
    )
    return accepted, {
        "rms_db": level,
        "crest_db": crest_db,
        "hf_ratio_db": ratio_db,
    }


def _frame_periodicity(frame: np.ndarray, sample_rate: int) -> float:
    centered = frame.astype(np.float64) - float(np.mean(frame))
    energy = float(np.dot(centered, centered))
    if energy <= 1e-16:
        return 0.0
    minimum_lag = max(1, round(sample_rate / 400.0))
    maximum_lag = min(len(centered) - 1, round(sample_rate / 60.0))
    if maximum_lag < minimum_lag:
        return 0.0
    return float(
        max(
            np.dot(centered[:-lag], centered[lag:]) / energy
            for lag in range(minimum_lag, maximum_lag + 1)
        )
    )


def _close_short_false_runs(mask: np.ndarray, maximum_hole: int) -> np.ndarray:
    closed = np.asarray(mask, dtype=bool).copy()
    inverse_runs = true_runs(~closed)
    for start, end in inverse_runs:
        if (
            start > 0
            and end < len(closed)
            and end - start <= maximum_hole
        ):
            closed[start:end] = True
    return closed


def classify_interword_gap_core(
    audio: np.ndarray,
    sample_rate: int,
    room_floor_db: float,
) -> tuple[bool, dict[str, Any]]:
    """Find a sustained aperiodic breath inside an already CTC-safe gap.

    This helper deliberately does not decide where words end.  Its caller must
    first remove the aligned phone/word guards.  Classification is framewise:
    a breath must remain above the room floor, broadband, low-periodicity, and
    non-impulsive for at least 80 ms.  Isolated clicks and ordinary room tone
    are reported but never passed to the breath gain stage.
    """

    core = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    level = rms_db(core)
    peak = float(np.max(np.abs(core))) if len(core) else 0.0
    peak_db = 20.0 * math.log10(max(peak, 1e-10))
    base: dict[str, Any] = {
        "core_samples": len(core),
        "core_duration_seconds": len(core) / sample_rate,
        "room_floor_db": float(room_floor_db),
        "core_rms_db": level,
        "core_peak_db": peak_db,
        "core_crest_db": peak_db - level,
    }
    minimum_event = round(0.080 * sample_rate)
    if len(core) < minimum_event:
        return False, {**base, "status": "too_short"}
    if level <= room_floor_db + 3.0:
        return False, {**base, "status": "room_tone"}

    frame_length = max(32, round(0.020 * sample_rate))
    hop = max(1, round(0.010 * sample_rate))
    if len(core) < frame_length:
        return False, {**base, "status": "too_short"}
    starts = np.arange(0, len(core) - frame_length + 1, hop, dtype=np.int64)
    frame_rows: list[dict[str, float]] = []
    for start in starts:
        frame = core[int(start) : int(start) + frame_length]
        frame_level = rms_db(frame)
        frame_peak = float(np.max(np.abs(frame))) if len(frame) else 0.0
        frame_peak_db = 20.0 * math.log10(max(frame_peak, 1e-10))
        windowed = frame * np.hanning(len(frame)).astype(np.float32)
        spectrum = np.square(np.abs(np.fft.rfft(windowed)), dtype=np.float64)
        frequencies = np.fft.rfftfreq(len(frame), 1.0 / sample_rate)
        band = (frequencies >= 80.0) & (frequencies <= 8000.0)
        high = (frequencies >= 500.0) & (frequencies <= 8000.0)
        band_energy = float(np.sum(spectrum[band]))
        high_energy = float(np.sum(spectrum[high]))
        hf_ratio_db = 10.0 * math.log10(
            max(high_energy, 1e-20) / max(band_energy, 1e-20)
        )
        positive_band = spectrum[band] + 1e-20
        flatness = float(
            np.exp(np.mean(np.log(positive_band)))
            / max(np.mean(positive_band), 1e-20)
        )
        frame_rows.append(
            {
                "rms_db": frame_level,
                "crest_db": frame_peak_db - frame_level,
                "hf_ratio_db": hf_ratio_db,
                "flatness": flatness,
                "periodicity": _frame_periodicity(frame, sample_rate),
            }
        )

    candidate_mask = np.array(
        [
            room_floor_db + 7.5 <= row["rms_db"] <= -30.0
            and row["hf_ratio_db"] > -5.5
            and row["periodicity"] < 0.48
            and row["crest_db"] < 18.0
            for row in frame_rows
        ],
        dtype=bool,
    )
    candidate_mask = _close_short_false_runs(candidate_mask, maximum_hole=2)
    proposals: list[tuple[float, int, int]] = []
    for start_frame, end_frame in true_runs(candidate_mask):
        candidate_start = int(starts[start_frame])
        candidate_end = int(starts[end_frame - 1]) + frame_length
        duration = candidate_end - candidate_start
        if not minimum_event <= duration <= round(0.700 * sample_rate):
            continue
        rows = frame_rows[start_frame:end_frame]
        excess = float(
            np.mean([row["rms_db"] - room_floor_db for row in rows])
        )
        proposals.append((excess + duration / sample_rate, candidate_start, candidate_end))

    if proposals:
        _, candidate_start, candidate_end = max(proposals)
        # A long aperiodic event attached to the left fence may be a fricative
        # whose ASR word end was early.  Forced-alignment callers normally keep
        # it outside this core; this final guard makes fallback use fail closed.
        if candidate_start <= hop:
            return False, {
                **base,
                "status": "edge_attached_activity",
                "candidate_start_sample": candidate_start,
                "candidate_end_sample": candidate_end,
            }
        selected_frames = [
            row
            for row, start in zip(frame_rows, starts)
            if candidate_start <= start < candidate_end
        ]
        return True, {
            **base,
            "status": "breath_candidate",
            "candidate_start_sample": candidate_start,
            "candidate_end_sample": candidate_end,
            "candidate_duration_seconds": (
                candidate_end - candidate_start
            ) / sample_rate,
            "candidate_rms_db": rms_db(core[candidate_start:candidate_end]),
            "candidate_hf_ratio_db": float(
                np.mean([row["hf_ratio_db"] for row in selected_frames])
            ),
            "candidate_periodicity": float(
                np.mean([row["periodicity"] for row in selected_frames])
            ),
            "candidate_flatness": float(
                np.mean([row["flatness"] for row in selected_frames])
            ),
        }

    energetic = np.array(
        [row["rms_db"] >= room_floor_db + 7.5 for row in frame_rows],
        dtype=bool,
    )
    energetic_runs = [
        (
            int(starts[start]),
            int(starts[end - 1]) + frame_length,
        )
        for start, end in true_runs(energetic)
    ]
    longest_energetic = max(
        (end - start for start, end in energetic_runs),
        default=0,
    )
    if base["core_crest_db"] >= 18.0 and longest_energetic < minimum_event:
        status = "high_crest_transient"
    elif longest_energetic:
        status = "voiced_or_fricative"
    else:
        status = "room_tone"
    return False, {
        **base,
        "status": status,
        "longest_energetic_seconds": longest_energetic / sample_rate,
    }


def attenuate_interword_gap(
    audio: np.ndarray,
    *,
    sample_rate: int,
    left_word_end_sample: int,
    right_word_start_sample: int,
    room_tone: np.ndarray,
    guard_seconds: float = 0.070,
    minimum_editable_seconds: float = 0.080,
    attenuation_db: float = -36.0,
    fade_seconds: float = 0.025,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Strongly clean only the protected interior of an inter-word gap.

    Word timestamps define hard fences.  Samples inside the guard on either
    side remain bit-identical; the editable center is smoothly reduced and
    replaced by captured room tone, which avoids the unnatural digital-black
    silence produced by a hard noise gate.
    """

    result = np.asarray(audio, dtype=np.float32).copy()
    if result.ndim != 1 or sample_rate <= 0:
        raise ValueError("audio must be a mono vector with a positive sample rate")
    left = max(0, min(len(result), int(left_word_end_sample)))
    right = max(left, min(len(result), int(right_word_start_sample)))
    guard = max(0, round(float(guard_seconds) * sample_rate))
    edit_start = left + guard
    edit_end = right - guard
    minimum = max(1, round(float(minimum_editable_seconds) * sample_rate))
    if edit_end - edit_start < minimum:
        return result, None

    length = edit_end - edit_start
    fade = min(
        max(1, round(float(fade_seconds) * sample_rate)),
        max(1, length // 3),
    )
    target_gain = float(10.0 ** (float(attenuation_db) / 20.0))
    target_gain = float(np.clip(target_gain, 0.0, 1.0))
    gain = np.full(length, target_gain, dtype=np.float32)
    theta = np.linspace(0.0, math.pi / 2.0, fade, endpoint=True)
    transition = (
        target_gain
        + (1.0 - target_gain) * np.square(np.cos(theta))
    ).astype(np.float32)
    gain[:fade] = transition
    gain[-fade:] = transition[::-1]

    tone = np.asarray(room_tone, dtype=np.float32).reshape(-1)
    if len(tone) != length:
        tone = room_tone_segment(
            [tone],
            length,
            index=0,
            sample_rate=sample_rate,
        )
    original = result[edit_start:edit_end].copy()
    original_level = rms_db(original)
    tone_level = rms_db(tone)
    # Replacing one realization of ordinary room noise with another is not an
    # improvement and needlessly changes otherwise valid samples.
    if original_level <= tone_level + 3.0:
        return result, None
    result[edit_start:edit_end] = (
        original * gain
        + tone * np.sqrt(np.maximum(0.0, 1.0 - np.square(gain)))
    )
    processed_level = rms_db(result[edit_start:edit_end])
    center_start = fade
    center_end = length - fade
    if center_end <= center_start:
        center_start = 0
        center_end = length
    central_original = original[center_start:center_end]
    central_processed = result[
        edit_start + center_start : edit_start + center_end
    ]
    central_tone = tone[center_start:center_end]
    original_excess = max(
        0.0,
        float(np.mean(np.square(central_original, dtype=np.float64)))
        - float(np.mean(np.square(central_tone, dtype=np.float64))),
    )
    processed_excess = max(
        0.0,
        float(np.mean(np.square(central_processed, dtype=np.float64)))
        - float(np.mean(np.square(central_tone, dtype=np.float64))),
    )
    excess_reduction = 10.0 * math.log10(
        max(original_excess, 1e-20) / max(processed_excess, 1e-20)
    )
    metadata: dict[str, Any] = {
        "edit_start_sample": edit_start,
        "edit_end_sample": edit_end,
        "left_word_end_sample": left,
        "right_word_start_sample": right,
        "guard_samples": guard,
        "fade_samples": fade,
        "target_attenuation_db": float(attenuation_db),
        "original_rms_db": original_level,
        "room_tone_rms_db": tone_level,
        "processed_rms_db": processed_level,
        "measured_reduction_db": original_level - processed_level,
        "central_original_rms_db": rms_db(central_original),
        "central_processed_rms_db": rms_db(central_processed),
        "central_room_tone_rms_db": rms_db(central_tone),
        "central_excess_reduction_db": excess_reduction,
        "original_peak": float(np.max(np.abs(original))) if len(original) else 0.0,
        "processed_peak": (
            float(np.max(np.abs(result[edit_start:edit_end])))
            if length
            else 0.0
        ),
    }
    return result, metadata


def attenuate_breaths(
    clip: np.ndarray,
    *,
    piece: EditPiece,
    selected_word_indices: list[int],
    observed_words: list[Any],
    ctc_word_map: dict[int, dict[str, Any]],
    vad_mask: np.ndarray,
    frame_samples: int,
    source_sample_rate: int,
    source_start_sample: int,
    room_tones: list[np.ndarray],
    noise_floor_db: float,
) -> np.ndarray:
    if len(selected_word_indices) < 2:
        return clip
    result = clip.copy()
    selected_set = set(selected_word_indices)
    ordered = sorted(selected_set)
    room_floor_db = (
        float(np.median([rms_db(tone) for tone in room_tones if len(tone)]))
        if any(len(tone) for tone in room_tones)
        else float(noise_floor_db)
    )
    for proposal_index, (left_index, right_index) in enumerate(
        zip(ordered, ordered[1:])
    ):
        if right_index != left_index + 1:
            continue
        left = observed_words[left_index]
        right = observed_words[right_index]
        left_ctc = ctc_word_map.get(left_index)
        right_ctc = ctc_word_map.get(right_index)
        diagnostic_base: dict[str, Any] = {
            "left_word_index": left_index,
            "right_word_index": right_index,
            "left_word": left.text,
            "right_word": right.text,
        }
        if not left_ctc or not right_ctc:
            asr_gap = float(right.start) - float(left.end)
            if asr_gap >= 0.14:
                piece.interword_gap_diagnostics.append(
                    {
                        **diagnostic_base,
                        "gap_seconds": asr_gap,
                        "status": "forced_alignment_unavailable",
                    }
                )
            continue
        left_score = left_ctc.get("score")
        right_score = right_ctc.get("score")
        if (
            left_score is None
            or right_score is None
            or float(left_score) < 0.55
            or float(right_score) < 0.55
        ):
            aligned_gap = float(right_ctc["start"]) - float(left_ctc["end"])
            if aligned_gap >= 0.14:
                piece.interword_gap_diagnostics.append(
                    {
                        **diagnostic_base,
                        "gap_seconds": aligned_gap,
                        "status": "weak_forced_alignment",
                        "left_ctc_score": left_score,
                        "right_ctc_score": right_score,
                    }
                )
            continue

        gap = float(right_ctc["start"]) - float(left_ctc["end"])
        if not 0.14 <= gap <= 2.00:
            continue
        left_end = max(
            source_start_sample,
            round(float(left_ctc["end"]) * source_sample_rate),
        )
        right_start = min(
            source_start_sample + len(result),
            round(float(right_ctc["start"]) * source_sample_rate),
        )
        local_left = left_end - source_start_sample
        local_right = right_start - source_start_sample
        guard_samples = round(0.070 * source_sample_rate)
        safe_start = local_left + guard_samples
        safe_end = local_right - guard_samples
        if safe_end - safe_start < round(0.080 * source_sample_rate):
            piece.interword_gap_diagnostics.append(
                {
                    **diagnostic_base,
                    "gap_seconds": gap,
                    "status": "too_short_after_guards",
                    "left_ctc_score": left_score,
                    "right_ctc_score": right_score,
                }
            )
            continue

        safe_core = result[safe_start:safe_end]
        accepted, classification = classify_interword_gap_core(
            safe_core,
            source_sample_rate,
            room_floor_db,
        )
        absolute_safe_start = source_start_sample + safe_start
        absolute_safe_end = source_start_sample + safe_end
        start_frame = max(0, absolute_safe_start // frame_samples)
        end_frame = min(
            len(vad_mask),
            math.ceil(absolute_safe_end / frame_samples),
        )
        vad_fraction = (
            float(np.mean(vad_mask[start_frame:end_frame]))
            if end_frame > start_frame
            else 1.0
        )
        common_diagnostic = {
            **diagnostic_base,
            "gap_seconds": gap,
            "left_ctc_score": left_score,
            "right_ctc_score": right_score,
            "source_left_word_end_sample": left_end,
            "source_right_word_start_sample": right_start,
            "source_safe_start_sample": absolute_safe_start,
            "source_safe_end_sample": absolute_safe_end,
            "guard_samples": guard_samples,
            # Silero's padded binary mask is retained only as evidence.  It is
            # not a veto because it labels the supplied breaths as speech.
            "padded_vad_fraction": vad_fraction,
            **classification,
        }
        if not accepted:
            piece.interword_gap_diagnostics.append(
                {
                    **common_diagnostic,
                    "status": str(classification["status"]),
                }
            )
            continue

        candidate_start = safe_start + int(
            classification["candidate_start_sample"]
        )
        candidate_end = safe_start + int(
            classification["candidate_end_sample"]
        )
        if (
            candidate_start < safe_start
            or candidate_end > safe_end
            or candidate_end - candidate_start
            < round(0.080 * source_sample_rate)
        ):
            piece.interword_gap_diagnostics.append(
                {
                    **common_diagnostic,
                    "status": "invalid_candidate_bounds",
                }
            )
            continue

        room = room_tone_segment(
            room_tones,
            candidate_end - candidate_start,
            piece.piece_index * 997 + proposal_index,
            sample_rate=source_sample_rate,
        )
        cleaned, edit = attenuate_interword_gap(
            result,
            sample_rate=source_sample_rate,
            left_word_end_sample=candidate_start,
            right_word_start_sample=candidate_end,
            room_tone=room,
            guard_seconds=0.0,
            minimum_editable_seconds=0.080,
            attenuation_db=-36.0,
            fade_seconds=0.035,
        )
        if edit is None:
            piece.interword_gap_diagnostics.append(
                {
                    **common_diagnostic,
                    "status": "candidate_at_room_tone",
                }
            )
            continue
        result = cleaned
        source_edit_start = source_start_sample + int(edit["edit_start_sample"])
        source_edit_end = source_start_sample + int(edit["edit_end_sample"])
        record = {
            **edit,
            **common_diagnostic,
            "status": "cleaned",
            "source_start_sample": source_edit_start,
            "source_end_sample": source_edit_end,
            "source_candidate_start_sample": (
                source_start_sample + candidate_start
            ),
            "source_candidate_end_sample": (
                source_start_sample + candidate_end
            ),
        }
        piece.breath_attenuations.append(record)
        piece.interword_gap_diagnostics.append(record)
    return result


def can_merge_adjacent_pieces(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    sample_rate: int,
    maximum_source_gap: float = 0.19,
) -> bool:
    """Return true only when no observed word is skipped across the join."""

    word_skip = int(current["word_start"]) - int(previous["word_end"])
    acoustic_gap = (
        int(current["first_speech_sample"]) - int(previous["last_speech_sample"])
    ) / sample_rate
    previous_ctc_end = previous.get("ctc_end")
    current_ctc_start = current.get("ctc_start")
    previous_ctc_score = previous.get("ctc_last_score")
    current_ctc_score = current.get("ctc_first_score")
    if (
        previous_ctc_end is not None
        and current_ctc_start is not None
        and previous_ctc_score is not None
        and current_ctc_score is not None
        and float(previous_ctc_score) >= 0.55
        and float(current_ctc_score) >= 0.55
    ):
        # ASR commonly stretches the next word backward across an inhale.  A
        # reliable CTC gap is therefore the safer merge decision.
        acoustic_gap = float(current_ctc_start) - float(previous_ctc_end)
    consecutive_phrase = (
        int(current["phrase_indices"][0])
        == int(previous["phrase_indices"][-1]) + 1
    )
    return (
        consecutive_phrase
        and word_skip == 0
        and acoustic_gap <= maximum_source_gap
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--edit-plan", type=Path, required=True)
    parser.add_argument("--ctc-alignment", type=Path)
    parser.add_argument("--aliases", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--disable-breath-attenuation", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis = read_json(args.analysis)
    plan = read_json(args.edit_plan)
    aliases = load_aliases(args.aliases)
    observed = load_observed_words(args.transcript, aliases)
    ctc_data = read_json(args.ctc_alignment) if args.ctc_alignment else None
    ctc = ctc_boundaries(ctc_data, plan)
    ctc_word_map = map_ctc_words_to_observed(ctc_data, observed, aliases)
    source, sample_rate = load_mono(args.audio)
    if sample_rate != int(analysis["sample_rate"]):
        raise RuntimeError("Analysis sample rate does not match source audio.")
    features = np.load(analysis["features"])
    rms_frames = features["rms_db"]
    vad_mask = features["vad_mask"].astype(bool)
    frame_samples = int(analysis["frame_samples"])
    noise_floor_db = float(analysis["noise_floor_db"])
    quiet_threshold = float(np.clip(noise_floor_db + 13.0, -52.0, -37.0))
    atoms = {int(atom["atom_index"]): atom for atom in analysis["atoms"]}

    selected = [
        item
        for item in plan["selections"]
        if item["status"] in {"selected", "review"}
    ]
    selected.sort(key=lambda item: int(item["phrase_index"]))
    if not selected:
        raise RuntimeError(
            "The edit plan contains no matched script phrases; no audio can be "
            "rendered. Check that the recording and script correspond."
        )
    raw_pieces: list[dict[str, Any]] = []
    for item in selected:
        candidate = item["candidate"]
        word_start = int(candidate["word_start"])
        word_end = int(candidate["word_end"])
        first_word = observed[word_start]
        last_word = observed[word_end - 1]
        ctc_value = ctc.get(int(item["phrase_index"]))
        asr_start = first_word.start
        asr_end = last_word.end
        ctc_start = (
            float(ctc_value["start"])
            if ctc_value and ctc_value.get("start") is not None
            else None
        )
        ctc_end = (
            float(ctc_value["end"])
            if ctc_value and ctc_value.get("end") is not None
            else None
        )
        ctc_score = (
            float(ctc_value["mean_score"])
            if ctc_value and ctc_value.get("mean_score") is not None
            else None
        )

        # CTC is only a conservative extension hint.  A weak or under-aligned
        # CTC edge must never cut into the ASR-selected words.
        speech_start, speech_end = fuse_asr_ctc_interval(
            asr_start,
            asr_end,
            ctc_value,
        )
        ctc_first_score = (
            float(ctc_value["first_score"])
            if ctc_value and ctc_value.get("first_score") is not None
            else None
        )
        ctc_last_score = (
            float(ctc_value["last_score"])
            if ctc_value and ctc_value.get("last_score") is not None
            else None
        )
        # Whole-atom ASR often starts the following word at the inhale.  A
        # later forced-aligned onset may move the cut inward only when a clear
        # waveform onset independently corroborates it.
        inward_start, used_inward_ctc_start = corroborated_inward_ctc_start(
            source,
            sample_rate=sample_rate,
            asr_start=asr_start,
            ctc_start=ctc_start,
            ctc_score=ctc_first_score,
            noise_floor_db=noise_floor_db,
        )
        if used_inward_ctc_start:
            speech_start = inward_start

        first_atom = atoms[first_word.atom_index]
        last_atom = atoms[last_word.atom_index]
        lower_seconds = max(float(first_atom["start"]), speech_start - 0.42)
        upper_seconds = min(float(last_atom["end"]), speech_end + 0.46)
        if word_start > 0 and observed[word_start - 1].end <= speech_start:
            lower_seconds = max(lower_seconds, observed[word_start - 1].end)
        if word_end < len(observed) and observed[word_end].start >= speech_end:
            upper_seconds = min(upper_seconds, observed[word_end].start)
        speech_start = max(lower_seconds, min(upper_seconds, speech_start))
        speech_end = max(speech_start, min(upper_seconds, speech_end))
        target_start = round(speech_start * sample_rate)
        target_end = round(speech_end * sample_rate)
        lower_sample = max(0, round(lower_seconds * sample_rate))
        upper_sample = min(len(source), round(upper_seconds * sample_rate))
        proposed_start, start_kind = boundary_before(
            target=target_start,
            lower=lower_sample,
            rms_frames=rms_frames,
            frame_samples=frame_samples,
            sample_rate=sample_rate,
            threshold_db=quiet_threshold,
        )
        proposed_end, end_kind = boundary_after(
            target=target_end,
            upper=upper_sample,
            rms_frames=rms_frames,
            frame_samples=frame_samples,
            sample_rate=sample_rate,
            threshold_db=quiet_threshold,
        )
        snapped_start = snap_zero_crossing(
            source, proposed_start, round(0.004 * sample_rate)
        )
        snapped_end = snap_zero_crossing(
            source, proposed_end, round(0.004 * sample_rate)
        )
        # Zero-crossing snapping must not cross the hard neighboring-word
        # fences or move onto the speech side of the ASR/CTC anchor.
        source_start, source_end = clamp_snapped_boundaries(
            snapped_start,
            snapped_end,
            lower_fence=lower_sample,
            speech_start=target_start,
            speech_end=target_end,
            upper_fence=upper_sample,
        )
        forced_minimum = source_end <= source_start
        if forced_minimum:
            source_end = min(
                upper_sample,
                source_start + round(0.08 * sample_rate),
            )
            end_kind = "forced_minimum"
        reasons = list(item.get("review_reasons", []))
        if forced_minimum:
            reasons.append("degenerate_boundary_interval")
        alignment_diagnostics: list[str] = []
        if ctc_start is not None and abs(ctc_start - asr_start) > 0.30:
            alignment_diagnostics.append("ctc_asr_start_disagreement")
        if ctc_end is not None and abs(ctc_end - asr_end) > 0.30:
            alignment_diagnostics.append("ctc_asr_end_disagreement")
        if ctc_value and (
            ctc_value.get("first_score") is None
            or float(ctc_value["first_score"]) < 0.45
        ):
            alignment_diagnostics.append("low_ctc_first_word_score")
        if ctc_value and (
            ctc_value.get("last_score") is None
            or float(ctc_value["last_score"]) < 0.45
        ):
            alignment_diagnostics.append("low_ctc_last_word_score")
        start_edge_score = ctc_first_score
        end_edge_score = ctc_last_score
        if start_kind != "quiet_valley" and (
            start_edge_score is None or start_edge_score < 0.55
        ):
            reasons.append("active_start_without_strong_ctc")
        if end_kind != "quiet_valley" and (
            end_edge_score is None or end_edge_score < 0.55
        ):
            reasons.append("active_end_without_strong_ctc")
        raw_pieces.append(
            {
                "phrase_indices": [int(item["phrase_index"])],
                "unit_indices": [int(item["unit_index"])],
                "word_start": word_start,
                "word_end": word_end,
                "selected_word_indices": list(range(word_start, word_end)),
                "source_start_sample": source_start,
                "source_end_sample": source_end,
                "first_speech_sample": target_start,
                "last_speech_sample": target_end,
                "transcript": str(candidate["transcript"]),
                "pause_after": str(item["pause_after"]),
                "start_boundary_kind": start_kind,
                "end_boundary_kind": end_kind,
                "ctc_score": ctc_score,
                "ctc_start": ctc_start,
                "ctc_end": ctc_end,
                "ctc_first_score": ctc_first_score,
                "ctc_last_score": ctc_last_score,
                "used_inward_ctc_start": used_inward_ctc_start,
                "alignment_diagnostics": alignment_diagnostics,
                "review_reasons": reasons,
                "first_plan_item": item,
                "last_plan_item": item,
            }
        )

    merged: list[dict[str, Any]] = []
    for raw in raw_pieces:
        if merged:
            previous = merged[-1]
            # Even one deliberately skipped ASR word can be an "uh" or a
            # duplicate.  Merging across it would put the discarded sound back.
            if can_merge_adjacent_pieces(
                previous,
                raw,
                sample_rate=sample_rate,
            ):
                previous["review_reasons"] = [
                    reason
                    for reason in previous["review_reasons"]
                    if reason not in END_BOUNDARY_REASONS
                ]
                retained_raw_reasons = [
                    reason
                    for reason in raw["review_reasons"]
                    if reason not in START_BOUNDARY_REASONS
                ]
                previous["phrase_indices"].extend(raw["phrase_indices"])
                previous["unit_indices"].extend(raw["unit_indices"])
                previous["word_end"] = raw["word_end"]
                previous["selected_word_indices"].extend(raw["selected_word_indices"])
                previous["source_end_sample"] = max(
                    previous["source_end_sample"], raw["source_end_sample"]
                )
                previous["last_speech_sample"] = raw["last_speech_sample"]
                previous["transcript"] += " " + raw["transcript"]
                previous["pause_after"] = raw["pause_after"]
                previous["end_boundary_kind"] = raw["end_boundary_kind"]
                previous["ctc_end"] = raw.get("ctc_end")
                previous["ctc_last_score"] = raw.get("ctc_last_score")
                previous["review_reasons"].extend(retained_raw_reasons)
                previous["alignment_diagnostics"].extend(
                    raw["alignment_diagnostics"]
                )
                previous["last_plan_item"] = raw["last_plan_item"]
                continue
        merged.append(raw)

    room_tones = [
        source[int(region["start_sample"]) : int(region["end_sample"])].copy()
        for region in analysis.get("room_tone_regions", [])
        if int(region["end_sample"]) > int(region["start_sample"])
    ]
    synthetic_tone = not room_tones
    if synthetic_tone:
        room_tones = synthetic_room_tones(
            sample_rate=sample_rate,
            noise_floor_db=noise_floor_db,
        )
    pieces: list[EditPiece] = []
    for index, raw in enumerate(merged):
        following = (
            merged[index + 1]["first_plan_item"]
            if index + 1 < len(merged)
            else None
        )
        current = raw["last_plan_item"]
        gap_seconds = choose_pause_seconds(current, following)
        piece = EditPiece(
            piece_index=index,
            phrase_indices=list(raw["phrase_indices"]),
            unit_indices=list(raw["unit_indices"]),
            source_start_sample=int(raw["source_start_sample"]),
            source_end_sample=int(raw["source_end_sample"]),
            first_speech_sample=int(raw["first_speech_sample"]),
            last_speech_sample=int(raw["last_speech_sample"]),
            transcript=str(raw["transcript"]),
            pause_after=str(raw["pause_after"]),
            gap_after_samples=round(gap_seconds * sample_rate),
            fade_samples=round(0.015 * sample_rate),
            start_cut_rms_db=local_rms_db(
                source, int(raw["source_start_sample"]), round(0.010 * sample_rate)
            ),
            end_cut_rms_db=local_rms_db(
                source, int(raw["source_end_sample"]), round(0.010 * sample_rate)
            ),
            start_boundary_kind=str(raw["start_boundary_kind"]),
            end_boundary_kind=str(raw["end_boundary_kind"]),
            alignment_diagnostics=sorted(set(raw["alignment_diagnostics"])),
            review_reasons=sorted(set(raw["review_reasons"])),
        )
        if synthetic_tone:
            piece.review_reasons.append("synthetic_room_tone_no_valid_source")
        raw["piece"] = piece
        pieces.append(piece)

    output_path = args.output_dir / "edited_unmastered.wav"
    with sf.SoundFile(
        output_path,
        mode="w",
        samplerate=sample_rate,
        channels=1,
        format="WAV",
        subtype="FLOAT",
    ) as output:
        pending: np.ndarray | None = None
        cursor = 0

        def append(segment: np.ndarray, fade_samples: int) -> int:
            nonlocal pending, cursor
            segment = segment.astype(np.float32, copy=False)
            if pending is None:
                pending = segment
                return cursor
            overlap = min(fade_samples, len(pending), len(segment))
            segment_start = cursor + len(pending) - overlap
            if overlap:
                if len(pending) > overlap:
                    output.write(pending[:-overlap])
                    cursor += len(pending) - overlap
                output.write(equal_power_crossfade(pending, segment, overlap))
                cursor += overlap
                pending = segment[overlap:]
            else:
                output.write(pending)
                cursor += len(pending)
                pending = segment
            return segment_start

        for raw, piece in zip(merged, pieces):
            clip = source[piece.source_start_sample : piece.source_end_sample].copy()
            if not args.disable_breath_attenuation:
                clip = attenuate_breaths(
                    clip,
                    piece=piece,
                    selected_word_indices=list(raw["selected_word_indices"]),
                    observed_words=observed,
                    ctc_word_map=ctc_word_map,
                    vad_mask=vad_mask,
                    frame_samples=frame_samples,
                    source_sample_rate=sample_rate,
                    source_start_sample=piece.source_start_sample,
                    room_tones=room_tones,
                    noise_floor_db=noise_floor_db,
                )
            piece.output_start_sample = append(clip, piece.fade_samples)
            piece.output_end_sample = piece.output_start_sample + len(clip)
            tone = room_tone_segment(
                room_tones,
                piece.gap_after_samples,
                piece.piece_index,
                sample_rate=sample_rate,
            )
            tone_start = append(tone, piece.fade_samples)
            piece.output_gap_end_sample = tone_start + len(tone)
        if pending is not None and len(pending):
            output.write(pending)

    edl = {
        "schema_version": 1,
        "sample_rate": sample_rate,
        "audio": str(args.audio.resolve()),
        "output": str(output_path.resolve()),
        "room_tone_regions": analysis.get("room_tone_regions", []),
        "pieces": [asdict(piece) for piece in pieces],
    }
    write_json(args.output_dir / "edit_decision_list.json", edl)
    with (args.output_dir / "edit_decision_list.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = list(asdict(pieces[0]).keys()) if pieces else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for piece in pieces:
            writer.writerow(asdict(piece))
    render_report = {
        "output": str(output_path.resolve()),
        "sample_rate": sample_rate,
        "pieces": len(pieces),
        "duration": sf.info(output_path).duration,
        "breaths_attenuated": sum(len(piece.breath_attenuations) for piece in pieces),
        "interword_gaps_examined": sum(
            len(piece.interword_gap_diagnostics) for piece in pieces
        ),
        "interword_gaps_cleaned": sum(
            diagnostic.get("status") == "cleaned"
            for piece in pieces
            for diagnostic in piece.interword_gap_diagnostics
        ),
        "interword_gaps_protected_as_speech": sum(
            diagnostic.get("status")
            in {
                "voiced_or_fricative",
                "high_crest_transient",
                "edge_attached_activity",
            }
            for piece in pieces
            for diagnostic in piece.interword_gap_diagnostics
        ),
        "room_tone_status": (
            "synthetic_review_required" if synthetic_tone else "validated_source"
        ),
        "active_fallback_starts": sum(
            piece.start_boundary_kind != "quiet_valley" for piece in pieces
        ),
        "active_fallback_ends": sum(
            piece.end_boundary_kind != "quiet_valley" for piece in pieces
        ),
        "review_piece_count": sum(bool(piece.review_reasons) for piece in pieces),
    }
    write_json(args.output_dir / "render_report.json", render_report)
    print(json.dumps(render_report, indent=2))


if __name__ == "__main__":
    main()
