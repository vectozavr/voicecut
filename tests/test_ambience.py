from __future__ import annotations

import numpy as np
import pytest

from voicecut.ambience import (
    AmbienceThresholds,
    build_clean_ambience_bank,
    evaluate_ambience_candidate,
    measure_ambience_candidate,
    plan_ambience_assembly,
)


SAMPLE_RATE = 16_000


def _stationary(frame_count: int, *, frequency: float = 80.0) -> np.ndarray:
    timeline = np.arange(frame_count, dtype=np.float64) / SAMPLE_RATE
    mono = 0.0007 * np.sin(2.0 * np.pi * frequency * timeline)
    mono += 0.0003 * np.sin(2.0 * np.pi * 337.0 * timeline)
    return mono.astype(np.float32)[:, None]


def _reason_codes(evaluation: dict[str, object]) -> set[str]:
    return {
        str(reason["code"])
        for reason in evaluation["rejection_reasons"]  # type: ignore[index]
    }


def _accepted_candidate(
    candidate_id: str,
    start: int,
    end: int,
    *,
    stationarity: float,
    level_delta: float = 0.0,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source_start_sample": start,
        "source_end_sample": end,
        "duration_samples": end - start,
        "stationarity_score": stationarity,
        "noise_level_delta_db": level_delta,
        "accepted": True,
        "status": "accepted",
        "rejection_reasons": [],
    }


def test_stationary_candidate_is_accepted_without_mutating_source() -> None:
    source = _stationary(SAMPLE_RATE)
    before = source.copy()

    evaluation = evaluate_ambience_candidate(
        source,
        candidate_id="clean",
        start_sample=0,
        end_sample=len(source),
        sample_rate=SAMPLE_RATE,
        target_rms_db=-65.0,
    )

    assert evaluation["accepted"] is True
    assert evaluation["status"] == "accepted"
    assert evaluation["rejection_reasons"] == []
    assert evaluation["noise_level_delta_db"] < 1.0
    assert np.array_equal(source, before)


def test_metrics_include_every_required_deterministic_feature() -> None:
    metrics = measure_ambience_candidate(
        _stationary(SAMPLE_RATE),
        sample_rate=SAMPLE_RATE,
    )

    assert {
        "clipping_count",
        "crest_factor_db",
        "maximum_spectral_flux",
        "log_band_variance_db2",
        "maximum_rms_burst_db",
        "maximum_sample_discontinuity",
        "discontinuity_to_rms_ratio",
    } <= metrics.keys()


@pytest.mark.parametrize(
    ("artifact", "expected_reason"),
    [
        ("clipping", "clipping"),
        ("click", "excessive_crest_factor"),
        ("creak", "unstable_log_band_energy"),
        ("unstable", "unstable_log_band_energy"),
        ("burst", "sudden_rms_burst"),
        ("step", "sample_discontinuity"),
    ],
)
def test_extreme_nuisance_artifacts_are_rejected(
    artifact: str,
    expected_reason: str,
) -> None:
    source = _stationary(SAMPLE_RATE)
    midpoint = len(source) // 2
    if artifact == "clipping":
        source[midpoint, 0] = 1.0
    elif artifact == "click":
        source[midpoint, 0] = 0.2
    elif artifact == "creak":
        duration = 1_000
        timeline = np.arange(duration, dtype=np.float64) / SAMPLE_RATE
        source[5_000:6_000, 0] += (0.02 * np.sin(2.0 * np.pi * 42.0 * timeline)).astype(
            np.float32
        )
    elif artifact == "unstable":
        source[len(source) // 3 : 2 * len(source) // 3] *= 20.0
    elif artifact == "burst":
        source[midpoint - 200 : midpoint + 200] *= 30.0
    elif artifact == "step":
        source[midpoint:, 0] += 0.08
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(artifact)

    evaluation = evaluate_ambience_candidate(
        source,
        candidate_id=artifact,
        start_sample=0,
        end_sample=len(source),
        sample_rate=SAMPLE_RATE,
    )

    assert evaluation["accepted"] is False
    assert evaluation["status"] == "rejected"
    assert expected_reason in _reason_codes(evaluation)


def test_low_rms_alone_cannot_make_click_candidate_valid() -> None:
    source = _stationary(SAMPLE_RATE) * 0.01
    source[len(source) // 2, 0] = 0.002

    evaluation = evaluate_ambience_candidate(
        source,
        candidate_id="quiet-click",
        start_sample=0,
        end_sample=len(source),
        sample_rate=SAMPLE_RATE,
    )

    assert evaluation["metrics"]["rms_db"] < -75.0
    assert evaluation["accepted"] is False
    assert _reason_codes(evaluation) & {
        "excessive_crest_factor",
        "sudden_rms_burst",
        "sample_discontinuity",
    }


def test_digital_silence_is_not_accepted_as_room_ambience() -> None:
    source = np.zeros((SAMPLE_RATE, 1), dtype=np.float32)

    evaluation = evaluate_ambience_candidate(
        source,
        candidate_id="digital-silence",
        start_sample=0,
        end_sample=len(source),
        sample_rate=SAMPLE_RATE,
    )

    assert evaluation["accepted"] is False
    assert "digital_or_near_silence" in _reason_codes(evaluation)


def test_opposite_polarity_stereo_click_cannot_cancel_out_of_rejection() -> None:
    mono = _stationary(SAMPLE_RATE)
    source = np.repeat(mono, 2, axis=1)
    midpoint = len(source) // 2
    source[midpoint, 0] = 0.2
    source[midpoint, 1] = -0.2

    evaluation = evaluate_ambience_candidate(
        source,
        candidate_id="stereo-cancelling-click",
        start_sample=0,
        end_sample=len(source),
        sample_rate=SAMPLE_RATE,
    )

    assert evaluation["accepted"] is False
    assert _reason_codes(evaluation) & {
        "excessive_crest_factor",
        "sudden_rms_burst",
        "sample_discontinuity",
    }


def test_minimum_duration_is_configurable_and_recorded() -> None:
    source = _stationary(round(0.08 * SAMPLE_RATE))
    thresholds = AmbienceThresholds(minimum_duration_ms=100.0)

    evaluation = evaluate_ambience_candidate(
        source,
        candidate_id="too-short",
        start_sample=0,
        end_sample=len(source),
        sample_rate=SAMPLE_RATE,
        thresholds=thresholds,
    )

    assert _reason_codes(evaluation) == {"insufficient_duration"}
    reason = evaluation["rejection_reasons"][0]
    assert reason["value"] == pytest.approx(80.0)
    assert reason["limit"] == 100.0


def test_bank_preserves_complete_accepted_and_rejected_candidate_table() -> None:
    source = np.concatenate(
        [_stationary(SAMPLE_RATE), _stationary(SAMPLE_RATE, frequency=113.0)]
    )
    source[SAMPLE_RATE + SAMPLE_RATE // 2, 0] = 0.2
    before = source.copy()

    bank = build_clean_ambience_bank(
        source,
        candidates=[
            {
                "candidate_id": "clean",
                "source_start_sample": 0,
                "source_end_sample": SAMPLE_RATE,
            },
            {
                "candidate_id": "click",
                "source_start_sample": SAMPLE_RATE,
                "source_end_sample": 2 * SAMPLE_RATE,
            },
        ],
        sample_rate=SAMPLE_RATE,
    )

    assert bank["status"] == "complete"
    assert [item["candidate_id"] for item in bank["accepted_candidates"]] == ["clean"]
    assert [item["candidate_id"] for item in bank["rejected_candidates"]] == ["click"]
    assert len(bank["candidates"]) == 2
    assert np.array_equal(source, before)


def test_assembly_prefers_one_long_accepted_candidate() -> None:
    bank = {
        "accepted_candidates": [
            _accepted_candidate("short", 0, 600, stationarity=0.01),
            _accepted_candidate("long", 1_000, 2_500, stationarity=0.20),
            _accepted_candidate("best-long", 3_000, 4_400, stationarity=0.05),
        ]
    }

    plan = plan_ambience_assembly(
        bank,
        required_samples=1_200,
        sample_rate=1_000,
    )

    assert plan["status"] == "complete"
    assert plan["candidate_ids"] == ["best-long"]
    assert plan["planned_output_samples"] == 1_200
    assert len(plan["source_trace"]) == 1
    assert plan["crossfades"] == []
    assert plan["source_reuse"] is False


def test_assembly_uses_distinct_candidates_with_equal_power_crossfades() -> None:
    bank = {
        "accepted_candidates": [
            _accepted_candidate("a", 0, 600, stationarity=0.01),
            _accepted_candidate("b", 1_000, 1_600, stationarity=0.02),
            _accepted_candidate("c", 2_000, 2_600, stationarity=0.03),
        ]
    }

    plan = plan_ambience_assembly(
        bank,
        required_samples=1_500,
        sample_rate=1_000,
        crossfade_ms=100.0,
    )

    assert plan["status"] == "complete"
    assert plan["candidate_ids"] == ["a", "b", "c"]
    assert plan["planned_output_samples"] == 1_500
    assert len(set(plan["candidate_ids"])) == 3
    assert [item["duration_samples"] for item in plan["crossfades"]] == [100, 100]
    assert all(item["curve"] == "equal_power" for item in plan["crossfades"])
    assert all(item["left_gain"] == "cosine" for item in plan["crossfades"])
    assert all(item["right_gain"] == "sine" for item in plan["crossfades"])
    source_ranges = [
        (item["source_start_sample"], item["source_end_sample"])
        for item in plan["source_trace"]
    ]
    assert all(
        max(left_start, right_start) >= min(left_end, right_end)
        for index, (left_start, left_end) in enumerate(source_ranges)
        for right_start, right_end in source_ranges[index + 1 :]
    )
    assert plan["source_reuse"] is False


def test_overlapping_candidates_are_never_used_as_distinct_source() -> None:
    bank = {
        "accepted_candidates": [
            _accepted_candidate("a", 0, 600, stationarity=0.01),
            _accepted_candidate("overlap", 500, 1_100, stationarity=0.02),
            _accepted_candidate("c", 1_200, 1_800, stationarity=0.03),
        ]
    }

    plan = plan_ambience_assembly(
        bank,
        required_samples=1_100,
        sample_rate=1_000,
        crossfade_ms=100.0,
    )

    assert plan["status"] == "complete"
    assert plan["candidate_ids"] == ["a", "c"]
    assert "overlap" not in plan["candidate_ids"]
    assert plan["source_reuse"] is False


def test_insufficient_unique_capacity_returns_clean_ambience_unavailable() -> None:
    bank = {
        "accepted_candidates": [
            _accepted_candidate("a", 0, 600, stationarity=0.01),
            _accepted_candidate("b", 1_000, 1_600, stationarity=0.02),
            {
                **_accepted_candidate("rejected", 2_000, 4_000, stationarity=0.0),
                "accepted": False,
                "status": "rejected",
            },
        ]
    }

    plan = plan_ambience_assembly(
        bank,
        required_samples=1_200,
        sample_rate=1_000,
        crossfade_ms=100.0,
    )

    assert plan["status"] == "clean_ambience_unavailable"
    assert plan["available_unique_output_samples"] == 1_100
    assert plan["planned_output_samples"] == 0
    assert plan["source_trace"] == []
    assert plan["crossfades"] == []
    assert "rejected" not in plan["candidate_ids"]
    assert plan["source_reuse"] is False


def test_assembly_rotates_candidates_used_by_earlier_pauses() -> None:
    bank = {
        "accepted_candidates": [
            _accepted_candidate("best", 0, 2_000, stationarity=0.01),
            _accepted_candidate("unused", 3_000, 5_000, stationarity=0.01),
        ]
    }

    plan = plan_ambience_assembly(
        bank,
        required_samples=1_000,
        sample_rate=1_000,
        candidate_usage_counts={"best": 1, "unused": 0},
    )

    assert plan["status"] == "complete"
    assert plan["candidate_ids"] == ["unused"]
    assert plan["source_trace"][0]["prior_use_count"] == 0


def test_assembly_prefers_noise_level_matching_the_local_pause_context() -> None:
    close = _accepted_candidate("close", 0, 2_000, stationarity=0.01)
    close["metrics"] = {"rms_db": -58.0}
    far = _accepted_candidate("far", 3_000, 5_000, stationarity=0.01)
    far["metrics"] = {"rms_db": -78.0}
    bank = {"accepted_candidates": [far, close]}

    plan = plan_ambience_assembly(
        bank,
        required_samples=1_000,
        sample_rate=1_000,
        reference_rms_db=-60.0,
    )

    assert plan["candidate_ids"] == ["close"]
    assert plan["source_trace"][0]["noise_level_delta_db"] == pytest.approx(2.0)
    assert plan["source_trace"][0]["reference_noise_floor_db"] == -60.0
