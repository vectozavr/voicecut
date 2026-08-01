"""Pure evaluation and assembly planning for verified source ambience.

This module never changes source samples.  It measures candidate regions,
records explicit acceptance or rejection reasons, and produces a traceable
plan for assembling an ambience bed from distinct canonical-source ranges.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AmbienceThresholds:
    """Conservative deterministic gates for source ambience candidates."""

    minimum_duration_ms: float = 80.0
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    clipping_amplitude: float = 0.999
    maximum_clipping_samples: int = 0
    minimum_rms_amplitude: float = 1.0e-8
    maximum_crest_factor_db: float = 18.0
    maximum_spectral_flux: float = 0.65
    maximum_log_band_variance_db2: float = 25.0
    maximum_rms_burst_db: float = 8.0
    maximum_sample_discontinuity: float = 0.05
    maximum_discontinuity_to_rms_ratio: float = 12.0
    spectral_band_count: int = 8

    def validate(self) -> None:
        if self.minimum_duration_ms <= 0.0:
            raise ValueError("minimum ambience duration must be positive")
        if self.frame_ms <= 0.0 or self.hop_ms <= 0.0:
            raise ValueError("ambience frame geometry must be positive")
        if not 0.0 < self.clipping_amplitude <= 1.0:
            raise ValueError("clipping amplitude must be inside (0, 1]")
        if self.maximum_clipping_samples < 0:
            raise ValueError("maximum clipping count cannot be negative")
        if (
            self.maximum_crest_factor_db <= 0.0
            or self.minimum_rms_amplitude <= 0.0
            or self.maximum_spectral_flux < 0.0
            or self.maximum_log_band_variance_db2 < 0.0
            or self.maximum_rms_burst_db < 0.0
            or self.maximum_sample_discontinuity < 0.0
            or self.maximum_discontinuity_to_rms_ratio <= 0.0
        ):
            raise ValueError("ambience rejection thresholds are invalid")
        if self.spectral_band_count < 2:
            raise ValueError("at least two spectral bands are required")


DEFAULT_AMBIENCE_THRESHOLDS = AmbienceThresholds()
DEFAULT_AMBIENCE_CROSSFADE_MS = 10.0
_EPSILON = np.finfo(np.float64).tiny


def _mono(source_audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(source_audio)
    if audio.ndim == 1:
        return np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2 and audio.shape[1] > 0:
        return np.mean(audio, axis=1, dtype=np.float64)
    raise ValueError("source audio must be mono or frames-by-channels")


def _frames(
    mono: np.ndarray,
    *,
    frame_samples: int,
    hop_samples: int,
) -> np.ndarray:
    if len(mono) <= frame_samples:
        padded = np.zeros(frame_samples, dtype=np.float64)
        padded[: len(mono)] = mono
        return padded[None, :]
    starts = list(range(0, len(mono) - frame_samples + 1, hop_samples))
    final_start = len(mono) - frame_samples
    if starts[-1] != final_start:
        starts.append(final_start)
    return np.stack([mono[start : start + frame_samples] for start in starts])


def _log_band_energies(
    power: np.ndarray,
    *,
    sample_rate: int,
    fft_size: int,
    band_count: int,
) -> np.ndarray:
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    positive = frequencies[frequencies > 0.0]
    if not len(positive):
        return np.zeros((len(power), band_count), dtype=np.float64)
    minimum = max(float(positive[0]), 20.0)
    maximum = max(minimum * 1.01, sample_rate / 2.0)
    edges = np.geomspace(minimum, maximum, num=band_count + 1)
    bands: list[np.ndarray] = []
    for index in range(band_count):
        if index == band_count - 1:
            mask = (frequencies >= edges[index]) & (frequencies <= edges[index + 1])
        else:
            mask = (frequencies >= edges[index]) & (frequencies < edges[index + 1])
        if not np.any(mask):
            nearest = int(
                np.argmin(
                    np.abs(frequencies - math.sqrt(edges[index] * edges[index + 1]))
                )
            )
            band_power = power[:, nearest]
        else:
            band_power = np.mean(power[:, mask], axis=1)
        bands.append(10.0 * np.log10(np.maximum(band_power, _EPSILON)))
    return np.stack(bands, axis=1)


def measure_ambience_candidate(
    samples: np.ndarray,
    *,
    sample_rate: int,
    thresholds: AmbienceThresholds = DEFAULT_AMBIENCE_THRESHOLDS,
) -> dict[str, int | float]:
    """Return deterministic nuisance and stationarity metrics for one region."""

    thresholds.validate()
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    raw_audio = np.asarray(samples)
    if raw_audio.ndim == 1:
        channels = raw_audio[:, None]
    elif raw_audio.ndim == 2 and raw_audio.shape[1] > 0:
        channels = raw_audio
    else:
        raise ValueError("source audio must be mono or frames-by-channels")
    if not len(channels):
        raise ValueError("ambience candidate is empty")
    finite = np.isfinite(channels)
    finite_channels = np.where(finite, channels, 0.0).astype(np.float64, copy=False)
    finite_mono = np.mean(finite_channels, axis=1)
    frame_samples = max(4, round(thresholds.frame_ms * sample_rate / 1000.0))
    hop_samples = max(1, round(thresholds.hop_ms * sample_rate / 1000.0))
    framed = _frames(
        finite_mono,
        frame_samples=frame_samples,
        hop_samples=hop_samples,
    )
    channel_frame_rms = np.stack(
        [
            np.sqrt(
                np.mean(
                    np.square(
                        _frames(
                            finite_channels[:, channel_index],
                            frame_samples=frame_samples,
                            hop_samples=hop_samples,
                        )
                    ),
                    axis=1,
                )
            )
            for channel_index in range(finite_channels.shape[1])
        ],
        axis=1,
    )
    channel_frame_rms_db = 20.0 * np.log10(np.maximum(channel_frame_rms, _EPSILON))
    channel_rms = np.sqrt(np.mean(np.square(finite_channels), axis=0))
    channel_peak = np.max(np.abs(finite_channels), axis=0)
    rms = float(np.sqrt(np.mean(np.square(finite_channels))))
    peak = float(np.max(channel_peak))
    crest_factor = float(np.max(channel_peak / np.maximum(channel_rms, _EPSILON)))
    crest_factor_db = 20.0 * math.log10(max(crest_factor, _EPSILON))

    # Keep enough frequency resolution for low-rate test/telephony audio.  A
    # tiny FFT makes a stationary low-frequency tone appear to have large
    # frame-to-frame flux solely because of phase leakage.
    fft_size = max(512, 1 << (frame_samples - 1).bit_length())
    window = np.hanning(frame_samples)
    spectra = np.fft.rfft(framed * window[None, :], n=fft_size, axis=1)
    power = np.square(np.abs(spectra))
    normalized = power / np.maximum(np.sum(power, axis=1, keepdims=True), _EPSILON)
    if len(normalized) > 1:
        spectral_flux = float(
            np.max(np.linalg.norm(np.diff(normalized, axis=0), axis=1))
        )
    else:
        spectral_flux = 0.0
    log_bands = _log_band_energies(
        power,
        sample_rate=sample_rate,
        fft_size=fft_size,
        band_count=thresholds.spectral_band_count,
    )
    log_band_variance = float(
        np.median(np.var(log_bands, axis=0)) if len(log_bands) > 1 else 0.0
    )
    if len(channel_frame_rms_db) > 1:
        level_above_median = float(
            np.max(
                channel_frame_rms_db
                - np.median(channel_frame_rms_db, axis=0, keepdims=True)
            )
        )
        adjacent_jump = float(np.max(np.abs(np.diff(channel_frame_rms_db, axis=0))))
        rms_burst_db = max(level_above_median, adjacent_jump)
    else:
        rms_burst_db = 0.0
    differences = np.abs(np.diff(finite_channels, axis=0))
    maximum_discontinuity = float(np.max(differences)) if len(differences) else 0.0

    return {
        "sample_count": len(channels),
        "channel_count": int(finite_channels.shape[1]),
        "duration_ms": float(len(channels) * 1000.0 / sample_rate),
        "non_finite_sample_count": int(np.count_nonzero(~finite)),
        "clipping_count": int(
            np.count_nonzero(np.abs(finite_channels) >= thresholds.clipping_amplitude)
        ),
        "peak_amplitude": peak,
        "rms_amplitude": rms,
        "rms_db": float(20.0 * math.log10(max(rms, _EPSILON))),
        "crest_factor_db": float(crest_factor_db),
        "maximum_spectral_flux": spectral_flux,
        "log_band_variance_db2": log_band_variance,
        "maximum_rms_burst_db": rms_burst_db,
        "maximum_sample_discontinuity": maximum_discontinuity,
        "discontinuity_to_rms_ratio": float(maximum_discontinuity / max(rms, _EPSILON)),
    }


def _reason(
    code: str,
    *,
    metric: str,
    value: int | float,
    limit: int | float,
) -> dict[str, str | int | float]:
    return {
        "code": code,
        "metric": metric,
        "value": value,
        "limit": limit,
    }


def evaluate_ambience_candidate(
    source_audio: np.ndarray,
    *,
    candidate_id: str,
    start_sample: int,
    end_sample: int,
    sample_rate: int,
    target_rms_db: float | None = None,
    thresholds: AmbienceThresholds = DEFAULT_AMBIENCE_THRESHOLDS,
) -> dict[str, Any]:
    """Evaluate one canonical-source range without modifying any samples."""

    thresholds.validate()
    if not candidate_id:
        raise ValueError("ambience candidate ID cannot be empty")
    total_samples = len(source_audio)
    if not 0 <= start_sample < end_sample <= total_samples:
        raise ValueError("ambience candidate leaves canonical source")
    metrics = measure_ambience_candidate(
        source_audio[start_sample:end_sample],
        sample_rate=sample_rate,
        thresholds=thresholds,
    )
    reasons: list[dict[str, str | int | float]] = []
    if metrics["duration_ms"] < thresholds.minimum_duration_ms:
        reasons.append(
            _reason(
                "insufficient_duration",
                metric="duration_ms",
                value=metrics["duration_ms"],
                limit=thresholds.minimum_duration_ms,
            )
        )
    if metrics["non_finite_sample_count"]:
        reasons.append(
            _reason(
                "non_finite_samples",
                metric="non_finite_sample_count",
                value=metrics["non_finite_sample_count"],
                limit=0,
            )
        )
    if metrics["rms_amplitude"] < thresholds.minimum_rms_amplitude:
        reasons.append(
            _reason(
                "digital_or_near_silence",
                metric="rms_amplitude",
                value=metrics["rms_amplitude"],
                limit=thresholds.minimum_rms_amplitude,
            )
        )
    gates = (
        (
            "clipping",
            "clipping_count",
            thresholds.maximum_clipping_samples,
        ),
        (
            "excessive_crest_factor",
            "crest_factor_db",
            thresholds.maximum_crest_factor_db,
        ),
        (
            "excessive_spectral_flux",
            "maximum_spectral_flux",
            thresholds.maximum_spectral_flux,
        ),
        (
            "unstable_log_band_energy",
            "log_band_variance_db2",
            thresholds.maximum_log_band_variance_db2,
        ),
        (
            "sudden_rms_burst",
            "maximum_rms_burst_db",
            thresholds.maximum_rms_burst_db,
        ),
        (
            "sample_discontinuity",
            "maximum_sample_discontinuity",
            thresholds.maximum_sample_discontinuity,
        ),
        (
            "sample_discontinuity",
            "discontinuity_to_rms_ratio",
            thresholds.maximum_discontinuity_to_rms_ratio,
        ),
    )
    for code, metric, limit in gates:
        if metrics[metric] > limit:
            reasons.append(
                _reason(code, metric=metric, value=metrics[metric], limit=limit)
            )
    ratios = [
        float(metrics["crest_factor_db"]) / thresholds.maximum_crest_factor_db,
        float(metrics["maximum_spectral_flux"])
        / max(thresholds.maximum_spectral_flux, _EPSILON),
        float(metrics["log_band_variance_db2"])
        / max(thresholds.maximum_log_band_variance_db2, _EPSILON),
        float(metrics["maximum_rms_burst_db"])
        / max(thresholds.maximum_rms_burst_db, _EPSILON),
        float(metrics["discontinuity_to_rms_ratio"])
        / thresholds.maximum_discontinuity_to_rms_ratio,
    ]
    stationarity_score = float(np.mean(ratios))
    noise_level_delta = (
        abs(float(metrics["rms_db"]) - target_rms_db)
        if target_rms_db is not None
        else 0.0
    )
    return {
        "candidate_id": candidate_id,
        "source_start_sample": start_sample,
        "source_end_sample": end_sample,
        "duration_samples": end_sample - start_sample,
        "duration_ms": (end_sample - start_sample) * 1000.0 / sample_rate,
        "metrics": metrics,
        "stationarity_score": stationarity_score,
        "target_rms_db": target_rms_db,
        "noise_level_delta_db": noise_level_delta,
        "accepted": not reasons,
        "status": "accepted" if not reasons else "rejected",
        "rejection_reasons": reasons,
    }


def build_clean_ambience_bank(
    source_audio: np.ndarray,
    *,
    candidates: Sequence[Mapping[str, Any]],
    sample_rate: int,
    target_rms_db: float | None = None,
    thresholds: AmbienceThresholds = DEFAULT_AMBIENCE_THRESHOLDS,
) -> dict[str, Any]:
    """Evaluate source-coordinate candidates into an inspectable clean bank."""

    evaluations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in candidates:
        candidate_id = str(raw.get("candidate_id", ""))
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate ambience candidate ID: {candidate_id}")
        seen_ids.add(candidate_id)
        evaluation = evaluate_ambience_candidate(
            source_audio,
            candidate_id=candidate_id,
            start_sample=int(raw["source_start_sample"]),
            end_sample=int(raw["source_end_sample"]),
            sample_rate=sample_rate,
            target_rms_db=target_rms_db,
            thresholds=thresholds,
        )
        # Candidate provenance is immutable source-coordinate evidence.  Keep
        # it beside the deterministic metrics so later pause traces can name
        # the MFA context and external masks that admitted the range.
        for key, value in raw.items():
            if key not in evaluation:
                evaluation[key] = value
        evaluations.append(evaluation)
    accepted = [item for item in evaluations if item["accepted"]]
    rejected = [item for item in evaluations if not item["accepted"]]
    return {
        "schema_version": 1,
        "status": "complete" if accepted else "clean_ambience_unavailable",
        "sample_rate": sample_rate,
        "thresholds": asdict(thresholds),
        "target_rms_db": target_rms_db,
        "candidates": evaluations,
        "accepted_candidates": accepted,
        "rejected_candidates": rejected,
    }


def _candidate_rank(
    candidate: Mapping[str, Any],
    *,
    reference_sample: int | None,
    reference_rms_db: float | None = None,
    prior_use_count: int = 0,
) -> tuple[float, float, int, int, int, int, str]:
    start = int(candidate["source_start_sample"])
    end = int(candidate["source_end_sample"])
    midpoint = (start + end) // 2
    metrics = candidate.get("metrics")
    candidate_rms_db = (
        float(metrics["rms_db"])
        if isinstance(metrics, Mapping) and "rms_db" in metrics
        else None
    )
    level_delta = (
        abs(candidate_rms_db - reference_rms_db)
        if candidate_rms_db is not None and reference_rms_db is not None
        else float(candidate["noise_level_delta_db"])
    )
    return (
        float(candidate["stationarity_score"]),
        level_delta,
        -(end - start),
        prior_use_count,
        abs(midpoint - reference_sample) if reference_sample is not None else 0,
        start,
        str(candidate["candidate_id"]),
    )


def _overlaps_selected(
    candidate: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> bool:
    start = int(candidate["source_start_sample"])
    end = int(candidate["source_end_sample"])
    return any(
        max(start, int(other["source_start_sample"]))
        < min(end, int(other["source_end_sample"]))
        for other in selected
    )


def _centered_source_range(
    candidate: Mapping[str, Any],
    *,
    frame_count: int,
) -> tuple[int, int]:
    start = int(candidate["source_start_sample"])
    end = int(candidate["source_end_sample"])
    if not 0 < frame_count <= end - start:
        raise ValueError("ambience source take has invalid duration")
    selected_start = start + ((end - start) - frame_count) // 2
    return selected_start, selected_start + frame_count


def plan_ambience_assembly(
    bank: Mapping[str, Any],
    *,
    required_samples: int,
    sample_rate: int,
    crossfade_ms: float = DEFAULT_AMBIENCE_CROSSFADE_MS,
    reference_sample: int | None = None,
    reference_rms_db: float | None = None,
    excluded_candidate_ids: Sequence[str] = (),
    candidate_usage_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Plan a traceable ambience bed without repeating source inside one bed.

    ``candidate_usage_counts`` rotates clean candidates across independent
    pauses.  Reuse between separate pauses is permitted only after less-used
    candidates have been preferred; a single ambience bed never tiles or
    repeats a candidate.
    """

    if required_samples < 0:
        raise ValueError("required ambience duration cannot be negative")
    if sample_rate <= 0 or crossfade_ms < 0.0:
        raise ValueError("ambience assembly geometry is invalid")
    if required_samples == 0:
        return {
            "status": "complete",
            "required_output_samples": 0,
            "planned_output_samples": 0,
            "source_trace": [],
            "crossfades": [],
            "source_reuse": False,
        }
    raw_accepted = bank.get("accepted_candidates", [])
    if not isinstance(raw_accepted, list):
        raise ValueError("ambience bank has no accepted candidate list")
    excluded = set(excluded_candidate_ids)
    accepted = [
        dict(item)
        for item in raw_accepted
        if item.get("accepted") is True
        and str(item.get("candidate_id", "")) not in excluded
    ]
    candidate_ids = [str(item.get("candidate_id", "")) for item in accepted]
    if any(not candidate_id for candidate_id in candidate_ids) or len(
        set(candidate_ids)
    ) != len(candidate_ids):
        raise ValueError("accepted ambience candidates have invalid IDs")
    usage_counts = candidate_usage_counts or {}
    if any(int(value) < 0 for value in usage_counts.values()):
        raise ValueError("ambience candidate usage counts cannot be negative")
    ranked = sorted(
        accepted,
        key=lambda item: _candidate_rank(
            item,
            reference_sample=reference_sample,
            reference_rms_db=reference_rms_db,
            prior_use_count=int(usage_counts.get(str(item["candidate_id"]), 0)),
        ),
    )

    sufficiently_long = [
        item
        for item in ranked
        if int(item["source_end_sample"]) - int(item["source_start_sample"])
        >= required_samples
    ]
    if sufficiently_long:
        candidate = sufficiently_long[0]
        source_start, source_end = _centered_source_range(
            candidate,
            frame_count=required_samples,
        )
        return {
            "status": "complete",
            "required_output_samples": required_samples,
            "planned_output_samples": required_samples,
            "candidate_ids": [str(candidate["candidate_id"])],
            "source_trace": [
                {
                    "trace_index": 0,
                    "candidate_id": str(candidate["candidate_id"]),
                    "source_start_sample": source_start,
                    "source_end_sample": source_end,
                    "output_start_sample": 0,
                    "output_end_sample": required_samples,
                    "stationarity_score": candidate["stationarity_score"],
                    "noise_level_delta_db": (
                        abs(float(candidate["metrics"]["rms_db"]) - reference_rms_db)
                        if reference_rms_db is not None
                        and isinstance(candidate.get("metrics"), Mapping)
                        and "rms_db" in candidate["metrics"]
                        else candidate["noise_level_delta_db"]
                    ),
                    "reference_noise_floor_db": reference_rms_db,
                    "prior_use_count": int(
                        usage_counts.get(str(candidate["candidate_id"]), 0)
                    ),
                }
            ],
            "crossfades": [],
            "source_reuse": False,
        }

    requested_crossfade = round(crossfade_ms * sample_rate / 1000.0)
    selected: list[dict[str, Any]] = []
    output_capacity = 0
    selected_crossfades: list[int] = []
    for candidate in ranked:
        if _overlaps_selected(candidate, selected):
            continue
        duration = int(candidate["source_end_sample"]) - int(
            candidate["source_start_sample"]
        )
        crossfade = (
            min(
                requested_crossfade,
                duration // 2,
                int(selected[-1]["duration_samples"]) // 2,
            )
            if selected
            else 0
        )
        effective = duration - crossfade
        if effective <= 0:
            continue
        selected.append(candidate)
        selected_crossfades.append(crossfade)
        output_capacity += effective
        if output_capacity >= required_samples:
            break
    if output_capacity < required_samples:
        return {
            "status": "clean_ambience_unavailable",
            "required_output_samples": required_samples,
            "planned_output_samples": 0,
            "available_unique_output_samples": output_capacity,
            "candidate_ids": [str(item["candidate_id"]) for item in selected],
            "source_trace": [],
            "crossfades": [],
            "source_reuse": False,
        }

    trace: list[dict[str, Any]] = []
    crossfades: list[dict[str, Any]] = []
    output_cursor = 0
    for index, candidate in enumerate(selected):
        crossfade = selected_crossfades[index]
        if index == len(selected) - 1:
            take = required_samples - output_cursor + crossfade
        else:
            take = int(candidate["duration_samples"])
        source_start, source_end = _centered_source_range(candidate, frame_count=take)
        output_start = output_cursor - crossfade
        output_end = output_start + take
        trace.append(
            {
                "trace_index": index,
                "candidate_id": str(candidate["candidate_id"]),
                "source_start_sample": source_start,
                "source_end_sample": source_end,
                "output_start_sample": output_start,
                "output_end_sample": output_end,
                "stationarity_score": candidate["stationarity_score"],
                "noise_level_delta_db": (
                    abs(float(candidate["metrics"]["rms_db"]) - reference_rms_db)
                    if reference_rms_db is not None
                    and isinstance(candidate.get("metrics"), Mapping)
                    and "rms_db" in candidate["metrics"]
                    else candidate["noise_level_delta_db"]
                ),
                "reference_noise_floor_db": reference_rms_db,
                "prior_use_count": int(
                    usage_counts.get(str(candidate["candidate_id"]), 0)
                ),
            }
        )
        if index:
            crossfades.append(
                {
                    "crossfade_index": index - 1,
                    "left_candidate_id": str(selected[index - 1]["candidate_id"]),
                    "right_candidate_id": str(candidate["candidate_id"]),
                    "output_start_sample": output_start,
                    "output_end_sample": output_start + crossfade,
                    "duration_samples": crossfade,
                    "curve": "equal_power",
                    "left_gain": "cosine",
                    "right_gain": "sine",
                }
            )
        output_cursor = output_end
    if output_cursor != required_samples:
        raise RuntimeError("ambience assembly accounting failed")
    source_ranges = [
        (int(item["source_start_sample"]), int(item["source_end_sample"]))
        for item in trace
    ]
    if any(
        max(left_start, right_start) < min(left_end, right_end)
        for index, (left_start, left_end) in enumerate(source_ranges)
        for right_start, right_end in source_ranges[index + 1 :]
    ):
        raise RuntimeError("ambience assembly reused canonical source samples")
    return {
        "status": "complete",
        "required_output_samples": required_samples,
        "planned_output_samples": output_cursor,
        "candidate_ids": [str(item["candidate_id"]) for item in selected],
        "source_trace": trace,
        "crossfades": crossfades,
        "source_reuse": False,
    }


__all__ = [
    "DEFAULT_AMBIENCE_CROSSFADE_MS",
    "DEFAULT_AMBIENCE_THRESHOLDS",
    "AmbienceThresholds",
    "build_clean_ambience_bank",
    "evaluate_ambience_candidate",
    "measure_ambience_candidate",
    "plan_ambience_assembly",
]
