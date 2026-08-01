from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from voicecut.breath_cleanup import (
    breath_room_tone_exclusions,
    plan_breath_replacements,
)


@dataclass(frozen=True)
class _Selection:
    source_ranges: list[tuple[int, int]]


class _RecordingAllocator:
    def __init__(self, *, source_cursor: int = 1_000) -> None:
        self.source_cursor = source_cursor
        self.requests: list[tuple[int, int]] = []

    def allocate(
        self,
        *,
        frame_count: int,
        reference_sample: int,
    ) -> tuple[np.ndarray, _Selection]:
        self.requests.append((frame_count, reference_sample))
        start = self.source_cursor
        end = start + frame_count
        self.source_cursor = end + 10
        return (
            np.zeros((frame_count, 1), dtype=np.float32),
            _Selection(source_ranges=[(start, end)]),
        )


def _ambience_candidate(
    candidate_id: str,
    start: int,
    end: int,
    *,
    stationarity_score: float = 0.1,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source_start_sample": start,
        "source_end_sample": end,
        "duration_samples": end - start,
        "stationarity_score": stationarity_score,
        "noise_level_delta_db": 0.0,
        "accepted": True,
        "status": "accepted",
        "rejection_reasons": [],
    }


def _ambience_bank(*candidates: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "complete",
        "sample_rate": 1_000,
        "accepted_candidates": list(candidates),
        "rejected_candidates": [],
    }


def test_room_tone_exclusions_preserve_event_and_guard_reasons() -> None:
    exclusions = breath_room_tone_exclusions(
        [
            {"start_sample": 100, "end_sample": 130},
            {"start_sample": 140, "end_sample": 170},
        ],
        sample_rate=1_000,
        total_samples=300,
        guard_ms=30.0,
    )

    assert exclusions == [
        {
            "start_sample": 100,
            "end_sample": 130,
            "reason": "breath_event",
            "event_index": 0,
        },
        {
            "start_sample": 70,
            "end_sample": 100,
            "reason": "breath_guard",
            "event_index": 0,
        },
        {
            "start_sample": 130,
            "end_sample": 160,
            "reason": "breath_guard",
            "event_index": 0,
        },
        {
            "start_sample": 140,
            "end_sample": 170,
            "reason": "breath_event",
            "event_index": 1,
        },
        {
            "start_sample": 110,
            "end_sample": 140,
            "reason": "breath_guard",
            "event_index": 1,
        },
        {
            "start_sample": 170,
            "end_sample": 200,
            "reason": "breath_guard",
            "event_index": 1,
        },
    ]
    assert exclusions[2]["start_sample"] < exclusions[4]["end_sample"]
    assert {item["reason"] for item in exclusions} == {
        "breath_event",
        "breath_guard",
    }


def test_replacement_uses_exact_duration_and_half_open_transitions() -> None:
    allocator = _RecordingAllocator()
    replacements, events = plan_breath_replacements(
        events=[{"start_sample": 200, "end_sample": 260}],
        editable_non_speech=[
            {
                "start_sample": 150,
                "end_sample": 350,
                "verification": "mfa_confirmed_editable_non_speech",
            }
        ],
        protected_speech_mask=[],
        allocator=allocator,
        sample_rate=1_000,
        total_samples=2_000,
        guard_ms=30.0,
        transition_ms=10.0,
    )

    assert len(replacements) == 1
    replacement = replacements[0]
    assert (replacement["target_start_sample"], replacement["target_end_sample"]) == (
        170,
        290,
    )
    assert replacement["target_duration_samples"] == 120
    assert replacement["replacement_duration_samples"] == 120
    assert replacement["replacement_room_tone_source_ranges"] == [
        {"source_start_sample": 1_000, "source_end_sample": 1_120}
    ]
    assert replacement["transition_ranges"] == [
        {
            "kind": "crossfade_in",
            "curve": "equal_power",
            "target_start_sample": 170,
            "target_end_sample": 180,
        },
        {
            "kind": "crossfade_out",
            "curve": "equal_power",
            "target_start_sample": 280,
            "target_end_sample": 290,
        },
    ]
    assert allocator.requests == [(120, 230)]
    assert events[0]["status"] == "breath_replaced_with_verified_room_tone"


def test_overlapping_event_guards_never_create_overlapping_targets() -> None:
    allocator = _RecordingAllocator()
    replacements, events = plan_breath_replacements(
        events=[
            {"start_sample": 200, "end_sample": 240},
            {"start_sample": 260, "end_sample": 300},
        ],
        editable_non_speech=[{"start_sample": 100, "end_sample": 400}],
        protected_speech_mask=[],
        allocator=allocator,
        sample_rate=1_000,
        total_samples=2_000,
        guard_ms=30.0,
        transition_ms=10.0,
    )

    assert [
        (item["target_start_sample"], item["target_end_sample"])
        for item in replacements
    ] == [(170, 270), (270, 330)]
    assert (
        replacements[0]["target_end_sample"] == replacements[1]["target_start_sample"]
    )
    assert [event["status"] for event in events] == [
        "breath_replaced_with_verified_room_tone",
        "breath_replaced_with_verified_room_tone",
    ]


def test_verified_bank_replacement_records_trace_and_candidate_id() -> None:
    replacements, events = plan_breath_replacements(
        events=[{"start_sample": 200, "end_sample": 260}],
        editable_non_speech=[{"start_sample": 150, "end_sample": 350}],
        protected_speech_mask=[],
        ambience_bank=_ambience_bank(_ambience_candidate("clean_000", 1_000, 1_300)),
        sample_rate=1_000,
        total_samples=2_000,
        guard_ms=30.0,
        transition_ms=10.0,
    )

    assert events[0]["status"] == "breath_replaced_with_verified_clean_ambience"
    assert len(replacements) == 1
    replacement = replacements[0]
    assert replacement["candidate_ids"] == ["clean_000"]
    assert replacement["source_reuse"] is False
    assert replacement["replacement_duration_samples"] == 120
    assert replacement["ambience_assembly"]["planned_output_samples"] == 120
    assert len(replacement["source_trace"]) == 1
    assert {
        key: replacement["source_trace"][0][key]
        for key in (
            "trace_index",
            "candidate_id",
            "source_start_sample",
            "source_end_sample",
            "output_start_sample",
            "output_end_sample",
        )
    } == {
        "trace_index": 0,
        "candidate_id": "clean_000",
        "source_start_sample": 1_090,
        "source_end_sample": 1_210,
        "output_start_sample": 0,
        "output_end_sample": 120,
    }
    assert replacement["source_trace"][0]["stationarity_score"] == 0.1
    assert replacement["replacement_room_tone_source_ranges"] == [
        {
            "candidate_id": "clean_000",
            "source_start_sample": 1_090,
            "source_end_sample": 1_210,
        }
    ]
    assert replacement["equal_power_crossfades"] == []
    assert {item["curve"] for item in replacement["transition_ranges"]} == {
        "equal_power"
    }


def test_verified_bank_uses_distinct_candidates_with_equal_power_crossfade() -> None:
    replacements, _ = plan_breath_replacements(
        events=[{"start_sample": 200, "end_sample": 350}],
        editable_non_speech=[{"start_sample": 200, "end_sample": 350}],
        protected_speech_mask=[],
        ambience_bank=_ambience_bank(
            _ambience_candidate("clean_000", 1_000, 1_080),
            _ambience_candidate("clean_001", 1_100, 1_180),
        ),
        sample_rate=1_000,
        total_samples=2_000,
        guard_ms=0.0,
        transition_ms=0.0,
    )

    replacement = replacements[0]
    assert replacement["candidate_ids"] == ["clean_000", "clean_001"]
    assert replacement["replacement_duration_samples"] == 150
    assert (
        sum(
            item["source_end_sample"] - item["source_start_sample"]
            for item in replacement["replacement_room_tone_source_ranges"]
        )
        == 160
    )
    assert replacement["ambience_assembly"]["planned_output_samples"] == 150
    assert replacement["equal_power_crossfades"] == [
        {
            "crossfade_index": 0,
            "left_candidate_id": "clean_000",
            "right_candidate_id": "clean_001",
            "output_start_sample": 70,
            "output_end_sample": 80,
            "duration_samples": 10,
            "curve": "equal_power",
            "left_gain": "cosine",
            "right_gain": "sine",
            "target_start_sample": 270,
            "target_end_sample": 280,
        }
    ]


def test_verified_bank_does_not_reuse_source_across_replacements() -> None:
    replacements, events = plan_breath_replacements(
        events=[
            {"start_sample": 200, "end_sample": 240},
            {"start_sample": 300, "end_sample": 340},
        ],
        editable_non_speech=[{"start_sample": 150, "end_sample": 400}],
        protected_speech_mask=[],
        ambience_bank=_ambience_bank(
            _ambience_candidate("clean_000", 1_000, 1_040),
            _ambience_candidate("clean_001", 1_100, 1_140),
        ),
        sample_rate=1_000,
        total_samples=2_000,
        guard_ms=0.0,
        transition_ms=0.0,
    )

    assert [item["candidate_ids"] for item in replacements] == [
        ["clean_000"],
        ["clean_001"],
    ]
    assert [event["status"] for event in events] == [
        "breath_replaced_with_verified_clean_ambience",
        "breath_replaced_with_verified_clean_ambience",
    ]
    first_range = replacements[0]["replacement_room_tone_source_ranges"][0]
    second_range = replacements[1]["replacement_room_tone_source_ranges"][0]
    assert first_range["source_end_sample"] <= second_range["source_start_sample"]


def test_verified_bank_reuse_across_distinct_events_is_explicitly_traced() -> None:
    usage_counts: dict[str, int] = {}
    replacements, events = plan_breath_replacements(
        events=[
            {"start_sample": 200, "end_sample": 240},
            {"start_sample": 300, "end_sample": 340},
        ],
        editable_non_speech=[{"start_sample": 150, "end_sample": 400}],
        protected_speech_mask=[],
        ambience_bank=_ambience_bank(_ambience_candidate("clean_000", 1_000, 1_040)),
        sample_rate=1_000,
        total_samples=2_000,
        guard_ms=0.0,
        transition_ms=0.0,
        candidate_usage_counts=usage_counts,
    )

    assert len(replacements) == 2
    assert events[0]["status"] == "breath_replaced_with_verified_clean_ambience"
    assert events[1]["status"] == "breath_replaced_with_verified_clean_ambience"
    assert replacements[0]["candidate_ids"] == ["clean_000"]
    assert replacements[1]["candidate_ids"] == ["clean_000"]
    assert replacements[0]["source_reuse"] is False
    assert replacements[1]["source_reuse"] is False
    assert usage_counts == {"clean_000": 2}


def test_bank_and_legacy_allocator_are_mutually_exclusive() -> None:
    with np.testing.assert_raises_regex(ValueError, "exactly one"):
        plan_breath_replacements(
            events=[],
            editable_non_speech=[],
            protected_speech_mask=[],
            ambience_bank=_ambience_bank(),
            allocator=_RecordingAllocator(),
            sample_rate=1_000,
            total_samples=2_000,
        )


@pytest.mark.parametrize("phone", ["S", "Z", "F", "TH", "SH", "SH AH N"])
def test_false_breath_detection_cannot_replace_a_retained_fricative_phone(
    phone: str,
) -> None:
    replacements, events = plan_breath_replacements(
        events=[{"start_sample": 100, "end_sample": 200}],
        editable_non_speech=[{"start_sample": 200, "end_sample": 300}],
        protected_speech_mask=[
            {
                "phone": phone,
                "start_sample": 100,
                "end_sample": 200,
            }
        ],
        ambience_bank=_ambience_bank(
            _ambience_candidate("clean", 500, 700),
        ),
        sample_rate=1_000,
        total_samples=1_000,
        guard_ms=0.0,
        transition_ms=0.0,
    )

    assert replacements == []
    assert events[0]["status"] == "breath_cleanup_skipped_phone_overlap"
