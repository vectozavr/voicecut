from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from voicecut.common import read_json, sha256_file, write_json
from voicecut.final_render import FinalRenderError, render_final_cut


SAMPLE_RATE = 1000


def _speech(audio: np.ndarray, start: int, end: int, frequency: float) -> None:
    time = np.arange(end - start, dtype=np.float32) / SAMPLE_RATE
    audio[start:end, 0] += 0.20 * np.sin(2.0 * np.pi * frequency * time)


def _source(length: int) -> np.ndarray:
    time = np.arange(length, dtype=np.float32) / SAMPLE_RATE
    return (0.0005 * np.sin(2.0 * np.pi * 11.0 * time))[:, None]


def _range_payload(
    words: list[dict[str, Any]],
    start: int,
    end: int,
) -> dict[str, Any]:
    canonical = " ".join(str(word["text"]) for word in words[start:end])
    return {
        "start_word_id": start,
        "end_word_id": end,
        "first_word_id": start,
        "last_word_id": end - 1,
        "first_word": words[start]["text"],
        "last_word": words[end - 1]["text"],
        "canonical_text": canonical,
    }


def _write_grounded_case(
    tmp_path: Path,
    *,
    source: np.ndarray,
    words: list[dict[str, Any]],
    thought_ranges: list[list[tuple[int, int]]],
    pause_types: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    audio_path = tmp_path / "source.wav"
    sf.write(audio_path, source, SAMPLE_RATE, subtype="FLOAT")
    transcript_path = tmp_path / "source_transcript.json"
    write_json(
        transcript_path,
        {
            "schema_version": 1,
            "artifact_role": "source_transcript",
            "audio": str(audio_path),
            "audio_sha256": sha256_file(audio_path),
            "atoms": [],
        },
    )

    committed: list[dict[str, Any]] = []
    canonical_total = 0
    selected_ranges: list[dict[str, int]] = []
    for thought_index, ranges in enumerate(thought_ranges):
        payloads = [_range_payload(words, start, end) for start, end in ranges]
        validation_ranges: list[dict[str, Any]] = []
        thought_tokens = 0
        for range_index, payload in enumerate(payloads):
            token_count = int(payload["end_word_id"]) - int(payload["start_word_id"])
            thought_tokens += token_count
            validation_ranges.append(
                {
                    "range_index": range_index,
                    **payload,
                    "canonical_tokens": token_count,
                    "supported_tokens": token_count,
                    "source_tokens": token_count,
                    "represented_source_tokens": token_count,
                    "unrepresented_source_tokens": [],
                    "unsupported_tokens": [],
                    "status": "valid",
                }
            )
            selected_ranges.append(
                {
                    "start_word_id": int(payload["start_word_id"]),
                    "end_word_id": int(payload["end_word_id"]),
                }
            )
        canonical_total += thought_tokens
        validation = {
            "thought_index": thought_index,
            "canonical_tokens": thought_tokens,
            "supported_tokens": thought_tokens,
            "unsupported_tokens": [],
            "status": "valid",
            "source_ranges": validation_ranges,
        }
        committed.append(
            {
                "canonical_text": " ".join(
                    str(payload["canonical_text"]) for payload in payloads
                ),
                "source_ranges": payloads,
                "grounding_validation": validation,
                "committed_iteration": 1,
            }
        )

    grounding_path = tmp_path / "grounding_validation.json"
    plan_path = tmp_path / "streaming_plan.json"
    plan = {
        "schema_version": 1,
        "planner": "streaming_narration_v1",
        "status": "complete",
        "backend": "gemini",
        "model": "cached-test-model",
        "transcript": str(transcript_path),
        "transcript_sha256": sha256_file(transcript_path),
        "grounding_validation": str(grounding_path),
        "word_count": len(words),
        "words": words,
        "committed": committed,
        "selected_source_ranges": selected_ranges,
        "reconstructed_narration": " ".join(
            str(thought["canonical_text"]) for thought in committed
        ),
    }
    write_json(plan_path, plan)
    write_json(
        grounding_path,
        {
            "schema_version": 1,
            "validator": "strict_bidirectional_range_source_grounding_v2",
            "status": "valid",
            "finalized_thoughts": len(committed),
            "source_ranges": len(selected_ranges),
            "canonical_tokens": canonical_total,
            "supported_tokens": canonical_total,
            "unsupported_tokens": [],
            "unrepresented_source_tokens": [],
            "planner_retries": 0,
            "plan_accepted": True,
            "error": None,
            "thoughts": [
                copy.deepcopy(thought["grounding_validation"]) for thought in committed
            ],
        },
    )

    transitions = []
    selected_pause_types = pause_types or ["continuation"] * max(0, len(committed) - 1)
    for index, pause_type in enumerate(selected_pause_types):
        transitions.append(
            {
                "after_thought_index": index,
                "before_thought_index": index + 1,
                "pause_type": pause_type,
            }
        )
    pause_path = tmp_path / "pause_plan.json"
    write_json(
        pause_path,
        {
            "schema_version": 1,
            "planner": "semantic_pause_planner_v1",
            "backend": "gemini",
            "model": "cached-test-model",
            "streaming_plan": str(plan_path),
            "streaming_plan_sha256": sha256_file(plan_path),
            "thought_count": len(committed),
            "transition_count": len(transitions),
            "transitions": transitions,
            "attempts": [],
        },
    )
    return audio_path, plan_path, pause_path


def _aligned_payload(
    words: list[dict[str, Any]],
    spans: list[tuple[float, float]],
) -> dict[str, Any]:
    aligned_words = [
        {
            "word": word["text"],
            "start": start,
            "end": end,
            "score": 0.99,
        }
        for word, (start, end) in zip(words, spans, strict=True)
    ]
    characters: list[dict[str, Any]] = []
    for index, (word, (start, end)) in enumerate(zip(words, spans, strict=True)):
        alphabetic = [
            character for character in str(word["text"]) if character.isalpha()
        ]
        duration = (end - start) / len(alphabetic)
        for character_index, character in enumerate(alphabetic):
            character_start = start + character_index * duration
            character_end = start + (character_index + 1) * duration
            characters.append(
                {
                    "char": character,
                    "start": character_start,
                    "end": character_end,
                    "score": 0.99,
                }
            )
        if index < len(words) - 1:
            characters.append({"char": " "})
    return {
        "schema_version": 1,
        "backend": "whisperx_alignment",
        "language": "en",
        "device": "cpu",
        "jobs": [
            {
                "clip_index": 0,
                "error": None,
                "aligned": {
                    "word_segments": aligned_words,
                    "segments": [
                        {
                            "text": " ".join(str(word["text"]) for word in words),
                            "words": aligned_words,
                            "chars": characters,
                        }
                    ],
                },
            }
        ],
    }


def _render(
    tmp_path: Path,
    *,
    source: np.ndarray,
    words: list[dict[str, Any]],
    thought_ranges: list[list[tuple[int, int]]],
    spans: list[tuple[float, float]] | None,
    pause_types: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, Path]:
    audio_path, plan_path, pause_path = _write_grounded_case(
        tmp_path,
        source=source,
        words=words,
        thought_ranges=thought_ranges,
        pause_types=pause_types,
    )
    output_dir = tmp_path / "final"
    manifest = render_final_cut(
        audio_path=audio_path,
        plan_path=plan_path,
        output_dir=output_dir,
        pause_plan_path=pause_path,
        alignment_python=tmp_path / "model-must-not-run",
        alignment_payload=(
            _aligned_payload(words, spans) if spans is not None else {"jobs": []}
        ),
    )
    boundary_plan = read_json(Path(manifest["final_boundary_plan"]))
    rendered, _ = sf.read(
        Path(manifest["final_cut_wav"]), dtype="float32", always_2d=True
    )
    return manifest, boundary_plan, rendered, audio_path


def _boundary(plan: dict[str, Any], kind: str) -> dict[str, Any]:
    return next(
        boundary for boundary in plan["boundaries"] if boundary["boundary_kind"] == kind
    )


def test_final_s_fricative_is_never_cut_or_faded(tmp_path: Path) -> None:
    source = _source(2300)
    _speech(source, 100, 640, 173.0)
    source[500:640, 0] += np.linspace(0.08, 0.01, 140, dtype=np.float32)
    _speech(source, 720, 940, 191.0)
    _speech(source, 1200, 1500, 211.0)
    words = [
        {"id": 0, "text": "examples", "start": 0.10, "end": 0.50},
        {"id": 1, "text": "wrong", "start": 0.49, "end": 0.94},
        {"id": 2, "text": "continue", "start": 1.20, "end": 1.50},
    ]

    _, plan, rendered, _ = _render(
        tmp_path,
        source=source,
        words=words,
        thought_ranges=[[(0, 1), (2, 3)]],
        spans=[(0.10, 0.64), (0.72, 0.94), (1.20, 1.50)],
    )

    end = _boundary(plan, "selected_to_omitted")
    assert end["selected_source_sample"] >= 640
    assert all(fade["source_start_sample"] >= 640 for fade in end["fade_intervals"])
    protected = next(
        interval
        for interval in end["protected_speech_intervals"]
        if interval["role"] == "last_retained_left"
    )
    mapping = next(
        segment
        for segment in plan["output_segments"]
        if segment["kind"] == "source"
        and segment["source_start_sample"] <= protected["start_sample"]
        and segment["source_end_sample"] >= protected["end_sample"]
    )
    output_start = mapping["output_start_sample"] + (
        protected["start_sample"] - mapping["source_start_sample"]
    )
    output_end = output_start + protected["end_sample"] - protected["start_sample"]
    assert np.array_equal(
        rendered[output_start:output_end],
        source[protected["start_sample"] : protected["end_sample"]],
    )


def test_retained_word_onset_close_to_removed_material_is_preserved(
    tmp_path: Path,
) -> None:
    source = _source(2100)
    _speech(source, 100, 350, 173.0)
    _speech(source, 500, 900, 191.0)
    _speech(source, 970, 1400, 211.0)
    words = [
        {"id": 0, "text": "keep", "start": 0.10, "end": 0.35},
        {"id": 1, "text": "discard", "start": 0.50, "end": 1.05},
        {"id": 2, "text": "starts", "start": 1.04, "end": 1.40},
    ]

    _, plan, rendered, _ = _render(
        tmp_path,
        source=source,
        words=words,
        thought_ranges=[[(0, 1), (2, 3)]],
        spans=[(0.10, 0.35), (0.50, 0.90), (0.97, 1.40)],
    )

    start = _boundary(plan, "omitted_to_selected")
    assert start["selected_source_sample"] <= 950
    retained = next(
        interval
        for interval in start["protected_speech_intervals"]
        if interval["role"] == "first_retained_right"
    )
    assert all(
        fade["source_end_sample"] <= retained["start_sample"]
        for fade in start["fade_intervals"]
    )
    source_segment = next(
        segment
        for segment in plan["output_segments"]
        if segment["kind"] == "source"
        and segment["source_start_sample"] <= retained["start_sample"]
        and segment["source_end_sample"] >= retained["end_sample"]
    )
    offset = retained["start_sample"] - source_segment["source_start_sample"]
    output_start = source_segment["output_start_sample"] + offset
    output_end = output_start + retained["end_sample"] - retained["start_sample"]
    assert np.array_equal(
        rendered[output_start:output_end],
        source[retained["start_sample"] : retained["end_sample"]],
    )


def test_overlapping_whisper_timestamps_are_not_clamped_into_speech(
    tmp_path: Path,
) -> None:
    source = _source(2200)
    _speech(source, 100, 500, 173.0)
    _speech(source, 650, 900, 191.0)
    _speech(source, 1100, 1450, 211.0)
    words = [
        {"id": 0, "text": "left", "start": 0.10, "end": 0.70},
        {"id": 1, "text": "removed", "start": 0.45, "end": 1.20},
        {"id": 2, "text": "right", "start": 1.00, "end": 1.45},
    ]

    _, plan, _, _ = _render(
        tmp_path,
        source=source,
        words=words,
        thought_ranges=[[(0, 1), (2, 3)]],
        spans=[(0.10, 0.50), (0.65, 0.90), (1.10, 1.45)],
    )

    left = _boundary(plan, "selected_to_omitted")
    right = _boundary(plan, "omitted_to_selected")
    assert left["selected_source_sample"] >= 520
    assert left["selected_source_sample"] <= 650
    assert right["selected_source_sample"] >= 900
    assert right["selected_source_sample"] <= 1080
    assert left["whisper_timestamps"]["retained_end_seconds"] == 0.70
    assert left["aligned_timestamps"]["retained_end_seconds"] == 0.50


def test_semantic_pause_inside_contiguous_range_uses_aligned_quiet_gap(
    tmp_path: Path,
) -> None:
    source = _source(2500)
    _speech(source, 100, 500, 173.0)
    _speech(source, 800, 1200, 211.0)
    words = [
        {"id": 0, "text": "first", "start": 0.10, "end": 0.58},
        {"id": 1, "text": "second", "start": 0.70, "end": 1.20},
    ]

    _, plan, rendered, _ = _render(
        tmp_path,
        source=source,
        words=words,
        thought_ranges=[[(0, 1)], [(1, 2)]],
        spans=[(0.10, 0.50), (0.80, 1.20)],
        pause_types=["thought"],
    )

    assert len(plan["source_intervals"]) == 1
    insertion = next(
        join for join in plan["joins"] if join["join_kind"] == "internal_thought_pause"
    )
    assert insertion["safety_status"] == "safe"
    assert 500 <= insertion["source_insertion_sample"] <= 800
    assert insertion["inserted_pause_samples"] > 0
    for protected in insertion["protected_speech_intervals"]:
        assert not (
            protected["start_sample"]
            <= insertion["source_insertion_sample"]
            < protected["end_sample"]
        )
    assert len(rendered) == plan["expected_output_frame_count"]


def test_dense_boundary_fails_closed_instead_of_guessing(tmp_path: Path) -> None:
    source = _source(1800)
    _speech(source, 100, 600, 173.0)
    _speech(source, 600, 900, 191.0)
    _speech(source, 980, 1300, 211.0)
    words = [
        {"id": 0, "text": "keeps", "start": 0.10, "end": 0.55},
        {"id": 1, "text": "removed", "start": 0.55, "end": 0.90},
        {"id": 2, "text": "after", "start": 0.98, "end": 1.30},
    ]
    audio_path, plan_path, pause_path = _write_grounded_case(
        tmp_path,
        source=source,
        words=words,
        thought_ranges=[[(0, 1), (2, 3)]],
    )
    output_dir = tmp_path / "final"

    with pytest.raises(FinalRenderError, match="unsafe_dense_boundary"):
        render_final_cut(
            audio_path=audio_path,
            plan_path=plan_path,
            output_dir=output_dir,
            pause_plan_path=pause_path,
            alignment_python=tmp_path / "model-must-not-run",
            alignment_payload=_aligned_payload(
                words,
                [(0.10, 0.60), (0.60, 0.90), (0.98, 1.30)],
            ),
            max_acoustic_retries=0,
        )

    boundary_plan = read_json(output_dir / "final_boundary_plan.json")
    assert boundary_plan["status"] == "unsafe"
    assert boundary_plan["unsafe_dense_boundaries"] >= 1
    assert any(
        boundary["safety_status"] == "unsafe_dense_boundary"
        for boundary in boundary_plan["boundaries"]
    )
    assert not (output_dir / "final_cut.wav").exists()


def test_final_word_at_eof_keeps_decay_and_has_no_speech_fade(tmp_path: Path) -> None:
    source = _source(1700)
    _speech(source, 100, 780, 173.0)
    source[600:780, 0] += np.linspace(0.08, 0.01, 180, dtype=np.float32)
    words = [
        {"id": 0, "text": "finals", "start": 0.10, "end": 0.55},
    ]

    _, plan, rendered, _ = _render(
        tmp_path,
        source=source,
        words=words,
        thought_ranges=[[(0, 1)]],
        spans=None,
    )

    final_boundary = plan["final_boundary"]
    assert final_boundary["boundary_method"] == "eof_safe_tail"
    assert final_boundary["selected_source_sample"] >= 780
    for fade in final_boundary["fade_intervals"]:
        assert fade["source_start_sample"] >= 780
    assert np.array_equal(rendered[100:780], source[100:780])
    assert not list((tmp_path / "final").glob("work/**/*.wav"))
