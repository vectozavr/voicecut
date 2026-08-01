"""Pure breath-replacement planning inside MFA-confirmed non-speech."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .ambience import plan_ambience_assembly
from .semantic_pause import PausePlanError

BREATH_EVENT_GUARD_MS = 30.0
BREATH_TRANSITION_MS = 10.0


def intervals_overlap(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> bool:
    return max(left_start, right_start) < min(left_end, right_end)


def breath_room_tone_exclusions(
    events: Sequence[dict[str, Any]],
    *,
    sample_rate: int,
    total_samples: int,
    guard_ms: float = BREATH_EVENT_GUARD_MS,
) -> list[dict[str, Any]]:
    """Separate event and guard exclusions so rejection reasons remain inspectable."""

    if sample_rate <= 0 or total_samples < 0 or guard_ms < 0.0:
        raise ValueError("invalid breath-exclusion geometry")
    guard = round(guard_ms * sample_rate / 1000.0)
    exclusions: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        start = max(0, min(total_samples, int(event["start_sample"])))
        end = max(start, min(total_samples, int(event["end_sample"])))
        if end <= start:
            continue
        exclusions.append(
            {
                "start_sample": start,
                "end_sample": end,
                "reason": "breath_event",
                "event_index": event_index,
            }
        )
        if start > 0 and guard:
            exclusions.append(
                {
                    "start_sample": max(0, start - guard),
                    "end_sample": start,
                    "reason": "breath_guard",
                    "event_index": event_index,
                }
            )
        if end < total_samples and guard:
            exclusions.append(
                {
                    "start_sample": end,
                    "end_sample": min(total_samples, end + guard),
                    "reason": "breath_guard",
                    "event_index": event_index,
                }
            )
    return exclusions


def plan_breath_replacements(
    *,
    events: Sequence[dict[str, Any]],
    editable_non_speech: Sequence[dict[str, Any]],
    protected_speech_mask: Sequence[dict[str, Any]],
    ambience_bank: Mapping[str, Any] | None = None,
    allocator: Any | None = None,
    sample_rate: int,
    total_samples: int,
    guard_ms: float = BREATH_EVENT_GUARD_MS,
    transition_ms: float = BREATH_TRANSITION_MS,
    candidate_usage_counts: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Plan equal-duration replacements without moving a cut or touching speech.

    Production callers pass the immutable verified ``ambience_bank``.  The
    optional allocator is retained only for compatibility with legacy/debug
    callers and is mutually exclusive with the bank.
    """

    if sample_rate <= 0 or total_samples <= 0:
        raise ValueError("invalid breath-replacement source geometry")
    if guard_ms < 0.0 or transition_ms < 0.0:
        raise ValueError("breath guard/transition cannot be negative")
    if (ambience_bank is None) == (allocator is None):
        raise ValueError("provide exactly one of ambience_bank or allocator")
    if ambience_bank is not None and int(ambience_bank.get("sample_rate", 0)) != int(
        sample_rate
    ):
        raise ValueError("ambience bank sample rate does not match the source")
    guard = round(guard_ms * sample_rate / 1000.0)
    requested_transition = round(transition_ms * sample_rate / 1000.0)
    replacements: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    previous_target_end = 0
    usage_counts = candidate_usage_counts if candidate_usage_counts is not None else {}

    ordered_events = sorted(events, key=lambda item: int(item["start_sample"]))
    for event_index, event in enumerate(ordered_events):
        event_start = max(0, min(total_samples, int(event["start_sample"])))
        event_end = max(event_start, min(total_samples, int(event["end_sample"])))
        protected_intersections = [
            dict(interval)
            for interval in protected_speech_mask
            if intervals_overlap(
                event_start,
                event_end,
                int(interval["start_sample"]),
                int(interval["end_sample"]),
            )
        ]
        editable_intersections = [
            {
                **dict(interval),
                "intersection_start_sample": max(
                    event_start,
                    int(interval["start_sample"]),
                ),
                "intersection_end_sample": min(
                    event_end,
                    int(interval["end_sample"]),
                ),
            }
            for interval in editable_non_speech
            if intervals_overlap(
                event_start,
                event_end,
                int(interval["start_sample"]),
                int(interval["end_sample"]),
            )
        ]
        source_event_index = int(event.get("source_event_index", event_index))
        common = {
            **dict(event),
            "event_index": source_event_index,
            "planner_event_index": event_index,
            "editable_intersection": editable_intersections,
            "protected_phone_intersections": protected_intersections,
            "replacements": [],
        }
        if not editable_intersections:
            event_records.append(
                {
                    **common,
                    "status": (
                        "breath_cleanup_skipped_phone_overlap"
                        if protected_intersections
                        else "breath_cleanup_skipped_no_mfa_non_speech"
                    ),
                }
            )
            continue

        event_replacements: list[dict[str, Any]] = []
        ambience_attempts: list[dict[str, Any]] = []
        for editable in editable_intersections:
            editable_start = int(editable["start_sample"])
            editable_end = int(editable["end_sample"])
            target_start = max(
                editable_start,
                event_start - guard,
                previous_target_end,
            )
            target_end = min(editable_end, event_end + guard)
            if target_end <= target_start:
                continue
            frame_count = target_end - target_start
            reference_sample = (target_start + target_end) // 2
            if ambience_bank is not None:
                assembly = plan_ambience_assembly(
                    ambience_bank,
                    required_samples=frame_count,
                    sample_rate=sample_rate,
                    reference_sample=reference_sample,
                    reference_rms_db=(
                        float(event["local_noise_floor_db"])
                        if event.get("local_noise_floor_db") is not None
                        else None
                    ),
                    candidate_usage_counts=usage_counts,
                )
                ambience_attempts.append(
                    {
                        "target_start_sample": target_start,
                        "target_end_sample": target_end,
                        "required_output_samples": frame_count,
                        "status": assembly["status"],
                        "candidate_ids": list(assembly.get("candidate_ids", [])),
                        "available_unique_output_samples": int(
                            assembly.get("available_unique_output_samples", frame_count)
                        ),
                    }
                )
                if assembly["status"] != "complete":
                    continue
                source_trace = [dict(item) for item in assembly["source_trace"]]
                replacement_ranges = [
                    {
                        "candidate_id": str(item["candidate_id"]),
                        "source_start_sample": int(item["source_start_sample"]),
                        "source_end_sample": int(item["source_end_sample"]),
                    }
                    for item in source_trace
                ]
                equal_power_crossfades = [
                    {
                        **dict(item),
                        "target_start_sample": target_start
                        + int(item["output_start_sample"]),
                        "target_end_sample": target_start
                        + int(item["output_end_sample"]),
                    }
                    for item in assembly.get("crossfades", [])
                ]
                for candidate_id in assembly["candidate_ids"]:
                    key = str(candidate_id)
                    usage_counts[key] = usage_counts.get(key, 0) + 1
                replacement_status = "breath_replaced_with_verified_clean_ambience"
            else:
                try:
                    _, selection = allocator.allocate(
                        frame_count=frame_count,
                        reference_sample=reference_sample,
                    )
                except PausePlanError:
                    continue
                if selection is None:
                    continue
                replacement_ranges = [
                    {
                        "source_start_sample": start,
                        "source_end_sample": end,
                    }
                    for start, end in selection.source_ranges
                ]
                if (
                    sum(
                        item["source_end_sample"] - item["source_start_sample"]
                        for item in replacement_ranges
                    )
                    != frame_count
                ):
                    raise PausePlanError("breath room-tone allocation changed duration")
                source_trace = [
                    {
                        "trace_index": index,
                        "candidate_id": f"legacy_allocator_{index:04d}",
                        "source_start_sample": item["source_start_sample"],
                        "source_end_sample": item["source_end_sample"],
                        "output_start_sample": sum(
                            int(previous["source_end_sample"])
                            - int(previous["source_start_sample"])
                            for previous in replacement_ranges[:index]
                        ),
                        "output_end_sample": sum(
                            int(previous["source_end_sample"])
                            - int(previous["source_start_sample"])
                            for previous in replacement_ranges[: index + 1]
                        ),
                    }
                    for index, item in enumerate(replacement_ranges)
                ]
                assembly = {
                    "status": "complete",
                    "required_output_samples": frame_count,
                    "planned_output_samples": frame_count,
                    "source_trace": source_trace,
                    "crossfades": [],
                    "source_reuse": None,
                    "compatibility_mode": "legacy_allocator",
                }
                equal_power_crossfades = []
                replacement_status = "breath_replaced_with_verified_room_tone"
            transition = min(requested_transition, frame_count // 2)
            replacement = {
                "replacement_id": f"breath_replacement_{len(replacements):04d}",
                "event_index": source_event_index,
                "planner_event_index": event_index,
                "fragment_index": event.get("fragment_index"),
                "target_start_sample": target_start,
                "target_end_sample": target_end,
                "target_duration_samples": frame_count,
                "replacement_room_tone_source_ranges": replacement_ranges,
                "replacement_duration_samples": frame_count,
                "ambience_assembly": assembly,
                "candidate_ids": list(assembly.get("candidate_ids", [])),
                "source_trace": source_trace,
                "equal_power_crossfades": equal_power_crossfades,
                "source_reuse": assembly.get("source_reuse"),
                "transition_samples": transition,
                "transition_ranges": (
                    [
                        {
                            "kind": "crossfade_in",
                            "curve": "equal_power",
                            "target_start_sample": target_start,
                            "target_end_sample": target_start + transition,
                        },
                        {
                            "kind": "crossfade_out",
                            "curve": "equal_power",
                            "target_start_sample": target_end - transition,
                            "target_end_sample": target_end,
                        },
                    ]
                    if transition
                    else []
                ),
                "editable_non_speech": dict(editable),
                "status": replacement_status,
            }
            replacements.append(replacement)
            event_replacements.append(replacement)
            previous_target_end = target_end
        event_records.append(
            {
                **common,
                "replacements": [
                    replacement["replacement_id"] for replacement in event_replacements
                ],
                "ambience_attempts": ambience_attempts,
                "status": (
                    event_replacements[0]["status"]
                    if event_replacements
                    else "clean_ambience_unavailable"
                ),
            }
        )

    return replacements, event_records


__all__ = [
    "BREATH_EVENT_GUARD_MS",
    "BREATH_TRANSITION_MS",
    "breath_room_tone_exclusions",
    "intervals_overlap",
    "plan_breath_replacements",
]
