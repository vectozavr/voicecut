from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import voicecut.breath_detection as breath_detection
from voicecut.breath_detection import (
    BreathDetectionError,
    RESPIRO_FRAME_HOP_SAMPLES,
    RESPIRO_SAMPLE_RATE,
    analyze_breath_evidence,
    load_respiro_runtime,
    merge_analysis_crops,
    split_long_analysis_crops,
    verify_respiro_runtime,
)


def test_threshold_runs_use_half_open_ten_ms_frames_and_minimum_duration() -> None:
    probabilities = np.array(
        [0.0, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.0],
        dtype=np.float32,
    )

    events = breath_detection._events_for_crop(
        probabilities=probabilities,
        crop_start_sample=0,
        crop_end_sample=2_000,
        source_sample_rate=RESPIRO_SAMPLE_RATE,
        threshold=0.5,
        minimum_duration_ms=80,
        crop_id="crop",
    )

    assert len(events) == 1
    assert events[0]["first_frame"] == 1
    assert events[0]["end_frame"] == 9
    assert events[0]["start_sample"] == RESPIRO_FRAME_HOP_SAMPLES
    assert events[0]["end_sample"] == 9 * RESPIRO_FRAME_HOP_SAMPLES
    assert events[0]["end_sample"] - events[0]["start_sample"] == 1_280
    assert (
        breath_detection._event_frame_runs(
            probabilities,
            threshold=0.5,
            minimum_duration_ms=81,
        )
        == []
    )


def test_relevant_crop_context_is_clamped_and_overlaps_are_merged() -> None:
    assert merge_analysis_crops(
        [(1_000, 1_200), (1_800, 2_000), (5_000, 5_100)],
        total_samples=6_000,
        sample_rate=1_000,
        context_ms=500.0,
    ) == [(500, 2_500), (4_500, 5_600)]
    assert merge_analysis_crops(
        [(100, 200), (300, 400)],
        total_samples=500,
        sample_rate=1_000,
        context_ms=50.0,
    ) == [(50, 450)]


def test_max_crop_splitting_deduplicates_events_from_overlap() -> None:
    total_samples = 10 * RESPIRO_SAMPLE_RATE
    expected_chunks = [
        (0, 4 * RESPIRO_SAMPLE_RATE),
        (3 * RESPIRO_SAMPLE_RATE, 7 * RESPIRO_SAMPLE_RATE),
        (6 * RESPIRO_SAMPLE_RATE, 10 * RESPIRO_SAMPLE_RATE),
    ]
    assert (
        split_long_analysis_crops(
            [(0, total_samples)],
            sample_rate=RESPIRO_SAMPLE_RATE,
            max_crop_seconds=4.0,
            context_ms=500.0,
        )
        == expected_chunks
    )

    source = (np.arange(total_samples, dtype=np.float32) / float(total_samples))[
        :, None
    ]
    event_start = 56_000
    event_end = 59_200
    inferred_crop_starts: list[int] = []

    def probabilities_for_crop(waveform: np.ndarray) -> np.ndarray:
        crop_start = round(float(waveform[0]) * total_samples)
        inferred_crop_starts.append(crop_start)
        probabilities = np.zeros(
            len(waveform) // RESPIRO_FRAME_HOP_SAMPLES + 1,
            dtype=np.float32,
        )
        overlap_start = max(crop_start, event_start)
        overlap_end = min(crop_start + len(waveform), event_end)
        if overlap_end > overlap_start:
            first_frame = (overlap_start - crop_start) // RESPIRO_FRAME_HOP_SAMPLES
            end_frame = (overlap_end - crop_start) // RESPIRO_FRAME_HOP_SAMPLES
            probabilities[first_frame:end_frame] = 0.9
        return probabilities

    evidence = analyze_breath_evidence(
        source_audio=source,
        source_sample_rate=RESPIRO_SAMPLE_RATE,
        relevant_ranges=[(0, total_samples)],
        threshold=0.5,
        minimum_duration_ms=80,
        context_ms=500.0,
        max_crop_seconds=4.0,
        probability_provider=probabilities_for_crop,
    )

    assert inferred_crop_starts == [start for start, _ in expected_chunks]
    assert [
        (crop["source_start_sample"], crop["source_end_sample"])
        for crop in evidence["analysis_crops"]
    ] == expected_chunks
    assert len(evidence["events"]) == 1
    event = evidence["events"][0]
    assert (event["start_sample"], event["end_sample"]) == (
        event_start,
        event_end,
    )
    assert event["crop_ids"] == ["breath_crop_0000", "breath_crop_0001"]
    assert len(event["detection_fragments"]) == 2


def test_frame_events_map_outward_to_non_16khz_canonical_samples() -> None:
    source_sample_rate = 44_117
    crop_start = 123
    probabilities = np.array(
        [0.0, 0.0, 0.9, 0.9, 0.9, 0.0],
        dtype=np.float32,
    )

    events = breath_detection._events_for_crop(
        probabilities=probabilities,
        crop_start_sample=crop_start,
        crop_end_sample=5_000,
        source_sample_rate=source_sample_rate,
        threshold=0.5,
        minimum_duration_ms=30,
        crop_id="non_16khz",
    )

    assert len(events) == 1
    expected_start = crop_start + (2 * source_sample_rate) // 100
    expected_end = crop_start + (5 * source_sample_rate + 99) // 100
    assert events[0]["start_sample"] == expected_start
    assert events[0]["end_sample"] == expected_end
    assert events[0]["start_sample"] <= crop_start + 0.020 * source_sample_rate
    assert events[0]["end_sample"] >= crop_start + 0.050 * source_sample_rate


def test_inference_receives_only_expanded_relevant_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.zeros((10 * RESPIRO_SAMPLE_RATE, 2), dtype=np.float32)
    received_lengths: list[int] = []

    def runtime_must_not_load(**_: object) -> object:
        raise AssertionError("injected probabilities must bypass runtime loading")

    def zero_probabilities(waveform: np.ndarray) -> np.ndarray:
        received_lengths.append(len(waveform))
        return np.zeros(
            len(waveform) // RESPIRO_FRAME_HOP_SAMPLES + 1,
            dtype=np.float32,
        )

    monkeypatch.setattr(
        breath_detection,
        "load_respiro_runtime",
        runtime_must_not_load,
    )
    evidence = analyze_breath_evidence(
        source_audio=source,
        source_sample_rate=RESPIRO_SAMPLE_RATE,
        relevant_ranges=[(2 * RESPIRO_SAMPLE_RATE, 3 * RESPIRO_SAMPLE_RATE)],
        context_ms=250.0,
        probability_provider=zero_probabilities,
    )

    assert received_lengths == [24_000]
    assert len(evidence["analysis_crops"]) == 1
    assert evidence["analysis_crops"][0]["source_start_sample"] == 28_000
    assert evidence["analysis_crops"][0]["source_end_sample"] == 52_000
    assert evidence["events"] == []


def test_runtime_verification_fails_closed_on_hash_mismatch(tmp_path: Path) -> None:
    (tmp_path / "modules.py").write_bytes(b"not the pinned module")
    (tmp_path / "respiro-en.pt").write_bytes(b"not the pinned checkpoint")
    (tmp_path / "LICENSE").write_bytes(b"not the pinned license")

    with pytest.raises(BreathDetectionError, match="hash mismatch"):
        verify_respiro_runtime(tmp_path)


@pytest.mark.skipif(
    os.environ.get("VOICECUT_RUN_BREATH_INTEGRATION") != "1",
    reason="set VOICECUT_RUN_BREATH_INTEGRATION=1 to load pinned Respiro-en",
)
def test_pinned_respiro_runtime_opt_in_integration() -> None:
    runtime = load_respiro_runtime(device="cpu")
    evidence = analyze_breath_evidence(
        source_audio=np.zeros((2 * RESPIRO_SAMPLE_RATE, 1), dtype=np.float32),
        source_sample_rate=RESPIRO_SAMPLE_RATE,
        relevant_ranges=[(0, 2 * RESPIRO_SAMPLE_RATE)],
        context_ms=0.0,
        runtime=runtime,
    )

    assert evidence["status"] == "complete"
    assert evidence["execution_device"] == "cpu"
    assert len(evidence["analysis_crops"]) == 1
    assert evidence["analysis_crops"][0]["frame_count"] > 0
