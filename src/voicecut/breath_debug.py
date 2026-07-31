"""Debug-only artifacts for inspecting planned breath cleanup.

This module is loaded only when the explicit debug flag is enabled.  It reads an
already frozen boundary plan and the already rendered final WAV, then writes only
short diagnostic excerpts and plots.  It never renders narration or feeds data
back into boundary planning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from .common import read_json, sha256_file, write_json


DEFAULT_EXCERPT_CONTEXT_MS = 250.0


class BreathDebugError(RuntimeError):
    """A debug artifact could not be produced safely or unambiguously."""


def _integer(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise BreathDebugError(f"{field} must be an integer")
    return value


def _validate_output_segments(
    segments: Any,
    *,
    source_frames: int,
    output_frames: int,
) -> list[dict[str, Any]]:
    if not isinstance(segments, list) or not segments:
        raise BreathDebugError("boundary plan has no output segments")
    validated: list[dict[str, Any]] = []
    output_cursor = 0
    for index, raw_segment in enumerate(segments):
        if not isinstance(raw_segment, dict):
            raise BreathDebugError("boundary plan contains a malformed output segment")
        kind = str(raw_segment.get("kind", ""))
        if kind not in {"source", "room_tone"}:
            raise BreathDebugError(f"unsupported output segment kind: {kind!r}")
        source_start = _integer(
            raw_segment.get("source_start_sample"),
            field=f"output_segments[{index}].source_start_sample",
        )
        source_end = _integer(
            raw_segment.get("source_end_sample"),
            field=f"output_segments[{index}].source_end_sample",
        )
        output_start = _integer(
            raw_segment.get("output_start_sample"),
            field=f"output_segments[{index}].output_start_sample",
        )
        output_end = _integer(
            raw_segment.get("output_end_sample"),
            field=f"output_segments[{index}].output_end_sample",
        )
        if not 0 <= source_start < source_end <= source_frames:
            raise BreathDebugError("output segment leaves the canonical source")
        if (
            output_start != output_cursor
            or not output_start < output_end <= output_frames
        ):
            raise BreathDebugError(
                "output segments are not contiguous final-output traces"
            )
        if source_end - source_start != output_end - output_start:
            raise BreathDebugError("output segment changes source duration")
        validated.append(raw_segment)
        output_cursor = output_end
    if output_cursor != output_frames:
        raise BreathDebugError("output segment trace does not cover the final WAV")
    return validated


def _write_audio_excerpt(
    *,
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
    full_audio_frames: int,
) -> None:
    if audio.ndim != 2 or not len(audio):
        raise BreathDebugError("refusing to write an empty diagnostic excerpt")
    if len(audio) >= full_audio_frames:
        raise BreathDebugError("refusing to write a full-length intermediate WAV")
    sf.write(path, audio, sample_rate, subtype="FLOAT")


def _protected_phone_spans(plan: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for raw in plan.get("protected_speech_mask", []):
        if not isinstance(raw, dict):
            continue
        start = raw.get("phone_start_sample", raw.get("start_sample"))
        end = raw.get("phone_end_sample", raw.get("end_sample"))
        if type(start) is int and type(end) is int and start < end:
            spans.append(
                {
                    "start_sample": start,
                    "end_sample": end,
                    "phone": str(raw.get("phone", "phone")),
                    "source_word_ids": list(raw.get("source_word_ids", [])),
                }
            )

    protected_containers = [
        *plan.get("boundaries", []),
        *plan.get("joins", []),
    ]
    for container in protected_containers:
        if not isinstance(container, dict):
            continue
        for protected in container.get("protected_speech_intervals", []):
            if not isinstance(protected, dict):
                continue
            for phone in protected.get("mfa_phone_intervals", []):
                if not isinstance(phone, dict) or bool(phone.get("is_silence")):
                    continue
                start = phone.get("start_sample")
                end = phone.get("end_sample")
                if type(start) is int and type(end) is int and start < end:
                    spans.append(
                        {
                            "start_sample": start,
                            "end_sample": end,
                            "phone": str(phone.get("phone", "phone")),
                            "source_word_ids": [protected.get("word_id")],
                        }
                    )

    for context in plan.get("alignment_contexts", []):
        if not isinstance(context, dict):
            continue
        for word in context.get("words", []):
            if not isinstance(word, dict):
                continue
            source_word_ids = list(word.get("source_word_ids", []))
            for phone in word.get("phones", []):
                if not isinstance(phone, dict) or bool(phone.get("is_silence")):
                    continue
                start = phone.get("start_sample")
                end = phone.get("end_sample")
                if type(start) is int and type(end) is int and start < end:
                    spans.append(
                        {
                            "start_sample": start,
                            "end_sample": end,
                            "phone": str(phone.get("phone", "phone")),
                            "source_word_ids": source_word_ids,
                        }
                    )

    unique: dict[tuple[int, int, str], dict[str, Any]] = {}
    for span in spans:
        key = (
            int(span["start_sample"]),
            int(span["end_sample"]),
            str(span["phone"]),
        )
        unique.setdefault(key, span)
    return sorted(
        unique.values(),
        key=lambda item: (int(item["start_sample"]), int(item["end_sample"])),
    )


def _event_phone_spans(
    phone_spans: Sequence[dict[str, Any]],
    *,
    excerpt_start: int,
    excerpt_end: int,
) -> list[dict[str, Any]]:
    return [
        dict(span)
        for span in phone_spans
        if max(excerpt_start, int(span["start_sample"]))
        < min(excerpt_end, int(span["end_sample"]))
    ]


def _event_probability_crops(
    event: dict[str, Any],
    detector_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_crops = detector_evidence.get("analysis_crops", [])
    if not isinstance(raw_crops, list):
        return []
    requested_ids = {
        str(value) for value in event.get("crop_ids", []) if isinstance(value, str)
    }
    event_start = int(event["start_sample"])
    event_end = int(event["end_sample"])
    selected: list[dict[str, Any]] = []
    for crop in raw_crops:
        if not isinstance(crop, dict):
            continue
        crop_id = str(crop.get("crop_id", ""))
        start = crop.get("source_start_sample")
        end = crop.get("source_end_sample")
        if type(start) is not int or type(end) is not int or start >= end:
            continue
        if requested_ids:
            include = crop_id in requested_ids
        else:
            include = max(start, event_start) < min(end, event_end)
        probabilities = crop.get("frame_probabilities")
        if include and isinstance(probabilities, list):
            selected.append(crop)
    return selected


def _write_probability_plot(
    *,
    path: Path,
    event: dict[str, Any],
    crops: Sequence[dict[str, Any]],
    phone_spans: Sequence[dict[str, Any]],
    source_sample_rate: int,
    frame_hop_ms: float,
    threshold: float,
    excerpt_start: int,
    excerpt_end: int,
) -> None:
    if not crops:
        raise BreathDebugError("event has no stored frame probabilities")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - depends on optional runtime
        raise BreathDebugError(f"matplotlib is unavailable: {error}") from error

    figure, axis = plt.subplots(figsize=(10.0, 3.5), constrained_layout=True)
    try:
        for crop in crops:
            probabilities = np.asarray(crop["frame_probabilities"], dtype=np.float64)
            crop_start = int(crop["source_start_sample"])
            frame_offsets = (
                np.arange(len(probabilities), dtype=np.float64)
                * source_sample_rate
                * frame_hop_ms
                / 1000.0
            )
            seconds = (crop_start + frame_offsets) / source_sample_rate
            axis.plot(
                seconds,
                probabilities,
                linewidth=1.0,
                label=str(crop.get("crop_id", "probability")),
            )
        axis.axhline(threshold, color="black", linestyle="--", linewidth=0.8)
        axis.axvspan(
            int(event["start_sample"]) / source_sample_rate,
            int(event["end_sample"]) / source_sample_rate,
            color="tab:orange",
            alpha=0.22,
            label="detected breath",
        )
        phone_color_index = 0
        phone_colors = ("tab:red", "tab:purple", "tab:brown", "tab:pink")
        for phone in phone_spans:
            color = phone_colors[phone_color_index % len(phone_colors)]
            phone_color_index += 1
            start_seconds = int(phone["start_sample"]) / source_sample_rate
            end_seconds = int(phone["end_sample"]) / source_sample_rate
            axis.axvspan(start_seconds, end_seconds, color=color, alpha=0.16)
            axis.text(
                (start_seconds + end_seconds) / 2.0,
                1.01,
                str(phone["phone"]),
                color=color,
                fontsize=8,
                ha="center",
                va="bottom",
                clip_on=False,
            )
        axis.set_xlim(
            excerpt_start / source_sample_rate, excerpt_end / source_sample_rate
        )
        axis.set_ylim(0.0, 1.05)
        axis.set_xlabel("Canonical source time (seconds)")
        axis.set_ylabel("Breath probability")
        axis.set_title("Respiro-en frame probabilities and protected MFA phones")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=7)
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)


def _retained_output_mappings(
    *,
    event_start: int,
    event_end: int,
    excerpt_start: int,
    excerpt_end: int,
    segments: Sequence[dict[str, Any]],
) -> list[dict[str, int]]:
    mappings: list[dict[str, int]] = []
    for segment in segments:
        if segment.get("kind") != "source":
            continue
        source_start = int(segment["source_start_sample"])
        source_end = int(segment["source_end_sample"])
        retained_start = max(event_start, source_start)
        retained_end = min(event_end, source_end)
        if retained_start >= retained_end:
            continue
        source_excerpt_start = max(excerpt_start, source_start)
        source_excerpt_end = min(excerpt_end, source_end)
        output_segment_start = int(segment["output_start_sample"])
        mappings.append(
            {
                "segment_index": int(segment["segment_index"]),
                "source_event_start_sample": retained_start,
                "source_event_end_sample": retained_end,
                "output_event_start_sample": output_segment_start
                + retained_start
                - source_start,
                "output_event_end_sample": output_segment_start
                + retained_end
                - source_start,
                "source_excerpt_start_sample": source_excerpt_start,
                "source_excerpt_end_sample": source_excerpt_end,
                "output_excerpt_start_sample": output_segment_start
                + source_excerpt_start
                - source_start,
                "output_excerpt_end_sample": output_segment_start
                + source_excerpt_end
                - source_start,
            }
        )
    return mappings


def _replacement_ids(event: dict[str, Any]) -> list[str]:
    raw = event.get("replacements", [])
    if not isinstance(raw, list):
        return []
    return [str(value) for value in raw if isinstance(value, str)]


def _write_replacement_excerpts(
    *,
    event_dir: Path,
    event: dict[str, Any],
    replacement_by_id: dict[str, dict[str, Any]],
    canonical_audio: np.ndarray,
    sample_rate: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    replacement_ids = _replacement_ids(event)
    for index, replacement_id in enumerate(replacement_ids):
        replacement = replacement_by_id.get(replacement_id)
        if replacement is None:
            raise BreathDebugError(f"event references unknown {replacement_id}")
        raw_ranges = replacement.get("replacement_room_tone_source_ranges")
        if not isinstance(raw_ranges, list) or not raw_ranges:
            raise BreathDebugError(f"{replacement_id} has no room-tone source ranges")
        parts: list[np.ndarray] = []
        ranges: list[dict[str, int]] = []
        excerpt_cursor = 0
        for raw_range in raw_ranges:
            if not isinstance(raw_range, dict):
                raise BreathDebugError("replacement has a malformed room-tone range")
            start = _integer(
                raw_range.get("source_start_sample"), field="room-tone start"
            )
            end = _integer(raw_range.get("source_end_sample"), field="room-tone end")
            if not 0 <= start < end <= len(canonical_audio):
                raise BreathDebugError("replacement room tone leaves canonical source")
            parts.append(canonical_audio[start:end])
            ranges.append(
                {
                    "source_start_sample": start,
                    "source_end_sample": end,
                    "excerpt_start_sample": excerpt_cursor,
                    "excerpt_end_sample": excerpt_cursor + end - start,
                }
            )
            excerpt_cursor += end - start
        filename = (
            "replacement_room_tone.wav"
            if len(replacement_ids) == 1
            else f"replacement_room_tone_{index:03d}.wav"
        )
        path = event_dir / filename
        _write_audio_excerpt(
            path=path,
            audio=np.concatenate(parts, axis=0),
            sample_rate=sample_rate,
            full_audio_frames=len(canonical_audio),
        )
        records.append(
            {
                "replacement_id": replacement_id,
                "path": str(path.resolve()),
                "source_ranges": ranges,
                "target_start_sample": replacement.get("target_start_sample"),
                "target_end_sample": replacement.get("target_end_sample"),
                "transition_ranges": replacement.get("transition_ranges", []),
            }
        )
    return records


def _detected_events(breath_cleanup: dict[str, Any]) -> list[dict[str, Any]]:
    planned = breath_cleanup.get("events")
    if isinstance(planned, list) and planned:
        return [dict(event) for event in planned if isinstance(event, dict)]
    detected = breath_cleanup.get("detected_events", [])
    if not isinstance(detected, list):
        return []
    return [dict(event) for event in detected if isinstance(event, dict)]


def write_breath_debug_artifacts(
    *,
    canonical_audio_path: Path,
    rendered_audio_path: Path,
    boundary_plan_path: Path,
    output_dir: Path,
    excerpt_context_ms: float = DEFAULT_EXCERPT_CONTEXT_MS,
) -> dict[str, Any]:
    """Write per-event diagnostics from an immutable, already rendered plan.

    Failures are reported in the returned debug manifest instead of escaping
    into production.  The function writes only event-sized WAV excerpts.
    """

    canonical_audio_path = Path(canonical_audio_path)
    rendered_audio_path = Path(rendered_audio_path)
    boundary_plan_path = Path(boundary_plan_path)
    output_dir = Path(output_dir)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": "breath_cleanup_debug_diagnostics",
        "diagnostics_only": True,
        "production_input": False,
        "full_intermediate_audio_written": False,
        "canonical_audio": str(canonical_audio_path.resolve()),
        "rendered_final_audio": str(rendered_audio_path.resolve()),
        "final_boundary_plan": str(boundary_plan_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "excerpt_context_ms": excerpt_context_ms,
        "status": "failed",
        "detected_event_count": 0,
        "event_diagnostics_written": 0,
        "event_diagnostics_failed": 0,
        "probability_plot_failures": 0,
        "retained_event_count": 0,
        "not_retained_event_count": 0,
        "replacement_excerpt_count": 0,
        "events": [],
        "errors": [],
    }
    try:
        if excerpt_context_ms < 0.0:
            raise BreathDebugError("excerpt context cannot be negative")
        output_dir.mkdir(parents=True, exist_ok=True)
        plan = read_json(boundary_plan_path)
        if not isinstance(plan, dict):
            raise BreathDebugError("final boundary plan root must be an object")
        if plan.get("status") != "safe":
            raise BreathDebugError("debug diagnostics require a safe frozen plan")
        planned_source_hash = plan.get("source_audio_sha256")
        if (
            planned_source_hash
            and sha256_file(canonical_audio_path) != planned_source_hash
        ):
            raise BreathDebugError("canonical source differs from the frozen plan")

        canonical_audio, canonical_rate = sf.read(
            canonical_audio_path,
            dtype="float32",
            always_2d=True,
        )
        rendered_audio, rendered_rate = sf.read(
            rendered_audio_path,
            dtype="float32",
            always_2d=True,
        )
        canonical_rate = int(canonical_rate)
        rendered_rate = int(rendered_rate)
        if canonical_rate != rendered_rate:
            raise BreathDebugError("canonical and final WAV sample rates differ")
        if canonical_audio.shape[1] != rendered_audio.shape[1]:
            raise BreathDebugError("canonical and final WAV channel counts differ")
        if canonical_rate != int(plan.get("source_sample_rate", -1)):
            raise BreathDebugError("canonical sample rate differs from the frozen plan")
        if len(canonical_audio) != int(plan.get("source_frame_count", -1)):
            raise BreathDebugError("canonical frame count differs from the frozen plan")
        if len(rendered_audio) != int(plan.get("expected_output_frame_count", -1)):
            raise BreathDebugError("final frame count differs from the frozen plan")
        segments = _validate_output_segments(
            plan.get("output_segments"),
            source_frames=len(canonical_audio),
            output_frames=len(rendered_audio),
        )

        breath_cleanup = plan.get("breath_cleanup", {})
        if not isinstance(breath_cleanup, dict):
            raise BreathDebugError("frozen plan has malformed breath-cleanup evidence")
        events = _detected_events(breath_cleanup)
        detector_evidence = breath_cleanup.get("detector_evidence", {})
        if not isinstance(detector_evidence, dict):
            detector_evidence = {}
        replacements = breath_cleanup.get("replacements", [])
        if not isinstance(replacements, list):
            replacements = []
        replacement_by_id = {
            str(replacement["replacement_id"]): replacement
            for replacement in replacements
            if isinstance(replacement, dict) and "replacement_id" in replacement
        }
        phone_spans = _protected_phone_spans(plan)
        context_samples = round(excerpt_context_ms * canonical_rate / 1000.0)
        frame_hop_ms = float(detector_evidence.get("frame_hop_ms", 10.0))
        threshold = float(
            detector_evidence.get(
                "threshold",
                breath_cleanup.get("threshold", 0.5),
            )
        )
        manifest["final_boundary_plan_sha256"] = sha256_file(boundary_plan_path)
        manifest["canonical_audio_sha256"] = sha256_file(canonical_audio_path)
        manifest["rendered_final_audio_sha256"] = sha256_file(rendered_audio_path)
        manifest["detected_event_count"] = len(events)

        for fallback_index, event in enumerate(events):
            event_index = event.get("event_index", fallback_index)
            if type(event_index) is not int:
                event_index = fallback_index
            event_dir = output_dir / f"event_{event_index:04d}"
            event_dir.mkdir(parents=True, exist_ok=True)
            record: dict[str, Any] = {
                "event_index": event_index,
                "event": event,
                "status": "failed",
                "errors": [],
            }
            try:
                event_start = _integer(event.get("start_sample"), field="event start")
                event_end = _integer(event.get("end_sample"), field="event end")
                if not 0 <= event_start < event_end <= len(canonical_audio):
                    raise BreathDebugError("event leaves the canonical source")
                excerpt_start = max(0, event_start - context_samples)
                excerpt_end = min(len(canonical_audio), event_end + context_samples)
                original_path = event_dir / "original_source.wav"
                _write_audio_excerpt(
                    path=original_path,
                    audio=canonical_audio[excerpt_start:excerpt_end],
                    sample_rate=canonical_rate,
                    full_audio_frames=len(canonical_audio),
                )
                record["original_source_excerpt"] = {
                    "path": str(original_path.resolve()),
                    "source_start_sample": excerpt_start,
                    "source_end_sample": excerpt_end,
                    "event_start_sample_in_excerpt": event_start - excerpt_start,
                    "event_end_sample_in_excerpt": event_end - excerpt_start,
                }

                mappings = _retained_output_mappings(
                    event_start=event_start,
                    event_end=event_end,
                    excerpt_start=excerpt_start,
                    excerpt_end=excerpt_end,
                    segments=segments,
                )
                cleaned_records: list[dict[str, Any]] = []
                for mapping_index, mapping in enumerate(mappings):
                    output_start = mapping["output_excerpt_start_sample"]
                    output_end = mapping["output_excerpt_end_sample"]
                    filename = (
                        "cleaned_output.wav"
                        if len(mappings) == 1
                        else f"cleaned_output_{mapping_index:03d}.wav"
                    )
                    cleaned_path = event_dir / filename
                    _write_audio_excerpt(
                        path=cleaned_path,
                        audio=rendered_audio[output_start:output_end],
                        sample_rate=canonical_rate,
                        full_audio_frames=len(rendered_audio),
                    )
                    cleaned_records.append(
                        {
                            **mapping,
                            "path": str(cleaned_path.resolve()),
                        }
                    )
                if cleaned_records:
                    record["retained_output"] = {
                        "status": "retained",
                        "mappings": cleaned_records,
                    }
                    manifest["retained_event_count"] += 1
                else:
                    record["retained_output"] = {
                        "status": "not_retained",
                        "reason": (
                            "event source range is absent from retained source "
                            "output_segments"
                        ),
                        "mappings": [],
                    }
                    manifest["not_retained_event_count"] += 1

                overlapping_phones = _event_phone_spans(
                    phone_spans,
                    excerpt_start=excerpt_start,
                    excerpt_end=excerpt_end,
                )
                probability_crops = _event_probability_crops(event, detector_evidence)
                plot_path = event_dir / "probabilities_and_mfa.png"
                try:
                    _write_probability_plot(
                        path=plot_path,
                        event=event,
                        crops=probability_crops,
                        phone_spans=overlapping_phones,
                        source_sample_rate=canonical_rate,
                        frame_hop_ms=frame_hop_ms,
                        threshold=threshold,
                        excerpt_start=excerpt_start,
                        excerpt_end=excerpt_end,
                    )
                    record["probability_plot"] = {
                        "status": "written",
                        "path": str(plot_path.resolve()),
                        "crop_ids": [
                            str(crop.get("crop_id", "")) for crop in probability_crops
                        ],
                        "mfa_phone_spans": overlapping_phones,
                    }
                except Exception as error:
                    record["probability_plot"] = {
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "mfa_phone_spans": overlapping_phones,
                    }
                    record["errors"].append(record["probability_plot"]["error"])
                    manifest["probability_plot_failures"] += 1

                replacement_records = _write_replacement_excerpts(
                    event_dir=event_dir,
                    event=event,
                    replacement_by_id=replacement_by_id,
                    canonical_audio=canonical_audio,
                    sample_rate=canonical_rate,
                )
                record["replacement_room_tone_excerpts"] = replacement_records
                manifest["replacement_excerpt_count"] += len(replacement_records)
                record["status"] = (
                    "complete" if not record["errors"] else "complete_with_plot_failure"
                )
                manifest["event_diagnostics_written"] += 1
            except Exception as error:
                record["errors"].append(f"{type(error).__name__}: {error}")
                manifest["event_diagnostics_failed"] += 1
            finally:
                event_json_path = event_dir / "event.json"
                record["event_json"] = str(event_json_path.resolve())
                try:
                    write_json(event_json_path, record)
                except Exception as error:
                    manifest["errors"].append(
                        f"could not write event {event_index} JSON: "
                        f"{type(error).__name__}: {error}"
                    )
                manifest["events"].append(record)

        manifest["status"] = (
            "complete"
            if manifest["event_diagnostics_failed"] == 0
            and manifest["probability_plot_failures"] == 0
            and not manifest["errors"]
            else "complete_with_failures"
        )
    except Exception as error:
        manifest["errors"].append(f"{type(error).__name__}: {error}")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "breath_debug_manifest.json"
        manifest["manifest_path"] = str(manifest_path.resolve())
        write_json(manifest_path, manifest)
    except Exception as error:
        manifest["errors"].append(
            f"could not write debug manifest: {type(error).__name__}: {error}"
        )
        manifest["status"] = "failed"
    return manifest


__all__ = [
    "BreathDebugError",
    "DEFAULT_EXCERPT_CONTEXT_MS",
    "write_breath_debug_artifacts",
]
