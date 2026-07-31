from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
            "target_start_sample": 170,
            "target_end_sample": 180,
        },
        {
            "kind": "crossfade_out",
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
