from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import voicecut.final_render as final_render_module
from voicecut.common import read_json, sha256_file, write_json
from voicecut.final_render import FinalRenderError, render_final_cut


SAMPLE_RATE = 1000


class ExplodingPauseBackend:
    """Proves that a supplied pause plan prevents a Gemini call."""

    backend_name = "gemini"
    model = "must-not-run"

    def generate(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any],
        request_id: str,
    ) -> str:
        del prompt, response_schema, request_id
        raise AssertionError("cached pause plan should bypass the backend")

    def close(self) -> None:
        raise AssertionError("a caller-owned, unused backend must not be closed")


class StaticRepairBackend:
    backend_name = "gemini"
    model = "cached-test-model"

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any],
        request_id: str,
    ) -> str:
        del request_id
        if not response_schema:
            raise AssertionError("repair backend received no JSON schema")
        self.prompts.append(prompt)
        return json.dumps(self.response)

    def close(self) -> None:
        raise AssertionError("a caller-owned repair backend must not be closed")


def _speech(
    audio: np.ndarray,
    start: int,
    end: int,
    *,
    frequency: float,
) -> None:
    time = np.arange(end - start, dtype=np.float32) / SAMPLE_RATE
    audio[start:end, 0] += 0.20 * np.sin(2.0 * np.pi * frequency * time)


def _range_payload(
    *,
    first_word_id: int,
    last_word_id: int,
    first_word: str,
    last_word: str,
    canonical_text: str,
) -> dict[str, Any]:
    return {
        "start_word_id": first_word_id,
        "end_word_id": last_word_id + 1,
        "first_word_id": first_word_id,
        "last_word_id": last_word_id,
        "first_word": first_word,
        "last_word": last_word,
        "canonical_text": canonical_text,
    }


def _thought(
    *,
    thought_index: int,
    source_range: dict[str, Any],
) -> dict[str, Any]:
    validation_range = {
        "range_index": 0,
        **source_range,
        "canonical_tokens": 1,
        "supported_tokens": 1,
        "source_tokens": 1,
        "represented_source_tokens": 1,
        "unrepresented_source_tokens": [],
        "unsupported_tokens": [],
        "status": "valid",
    }
    validation = {
        "thought_index": thought_index,
        "canonical_tokens": 1,
        "supported_tokens": 1,
        "unsupported_tokens": [],
        "status": "valid",
        "source_ranges": [validation_range],
    }
    return {
        "canonical_text": source_range["canonical_text"],
        "source_ranges": [source_range],
        "grounding_validation": validation,
        "committed_iteration": 1,
    }


def _aligned_job(
    *,
    clip_index: int,
) -> dict[str, Any]:
    words = ["selected", "omitted", "last"]
    times = [(0.10, 0.62), (0.72, 0.95), (1.20, 1.52)]
    word_segments = [
        {
            "word": word,
            "start": start,
            "end": end,
            "score": 0.95,
        }
        for word, (start, end) in zip(words, times, strict=True)
    ]
    chars: list[dict[str, Any]] = []
    for index, (word, (start, end)) in enumerate(zip(words, times, strict=True)):
        duration = (end - start) / len(word)
        for character_index, character in enumerate(word):
            character_start = start + character_index * duration
            character_end = start + (character_index + 1) * duration
            chars.append(
                {
                    "char": character,
                    "start": character_start,
                    "end": character_end,
                    "score": 0.95,
                }
            )
        if index < len(words) - 1:
            chars.append({"char": " "})
    return {
        "clip_index": clip_index,
        "error": None,
        "aligned": {
            "word_segments": word_segments,
            "segments": [
                {
                    "text": " ".join(words),
                    "words": word_segments,
                    "chars": chars,
                }
            ],
        },
    }


def _grounded_fixture(
    tmp_path: Path,
    *,
    dense_leading_boundary: bool = False,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    source = np.empty((2300, 1), dtype=np.float32)
    time = np.arange(len(source), dtype=np.float32) / SAMPLE_RATE
    source[:, 0] = 0.001 * np.sin(2.0 * np.pi * 13.0 * time)
    _speech(source, 100, 620, frequency=173.0)
    _speech(source, 720, 950, frequency=191.0)
    if dense_leading_boundary:
        _speech(source, 950, 1200, frequency=191.0)
    _speech(source, 1200, 1520, frequency=211.0)
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

    selected_range = _range_payload(
        first_word_id=0,
        last_word_id=0,
        first_word="selected",
        last_word="selected",
        canonical_text="Selected.",
    )
    final_range = _range_payload(
        first_word_id=2,
        last_word_id=2,
        first_word="last",
        last_word="last",
        canonical_text="Last.",
    )
    committed = [
        _thought(thought_index=0, source_range=selected_range),
        _thought(thought_index=1, source_range=final_range),
    ]
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
        "word_count": 3,
        "words": [
            {
                "id": 0,
                "text": "selected",
                "start": 0.10,
                "end": 0.50,
            },
            {
                "id": 1,
                "text": "omitted",
                "start": 0.50,
                "end": 0.95,
            },
            {
                "id": 2,
                "text": "last",
                "start": 1.20,
                "end": 1.50,
            },
        ],
        "committed": committed,
        "selected_source_ranges": [
            {"start_word_id": 0, "end_word_id": 1},
            {"start_word_id": 2, "end_word_id": 3},
        ],
        "reconstructed_narration": "Selected. Last.",
    }
    write_json(plan_path, plan)

    grounding = {
        "schema_version": 1,
        "validator": "strict_bidirectional_range_source_grounding_v2",
        "status": "valid",
        "finalized_thoughts": 2,
        "source_ranges": 2,
        "canonical_tokens": 2,
        "supported_tokens": 2,
        "unsupported_tokens": [],
        "unrepresented_source_tokens": [],
        "planner_retries": 0,
        "plan_accepted": True,
        "error": None,
        "thoughts": [
            copy.deepcopy(thought["grounding_validation"]) for thought in committed
        ],
    }
    write_json(grounding_path, grounding)

    pause_plan_path = tmp_path / "cached_pause_plan.json"
    write_json(
        pause_plan_path,
        {
            "schema_version": 1,
            "planner": "semantic_pause_planner_v1",
            "backend": "gemini",
            "model": "cached-test-model",
            "streaming_plan": str(plan_path),
            "streaming_plan_sha256": sha256_file(plan_path),
            "thought_count": 2,
            "transition_count": 1,
            "transitions": [
                {
                    "after_thought_index": 0,
                    "before_thought_index": 1,
                    "pause_type": "continuation",
                }
            ],
            "attempts": [],
        },
    )
    return audio_path, plan_path, pause_plan_path, grounding


def _familiar_thought(
    *,
    words: list[dict[str, Any]],
    ranges: list[tuple[int, int]],
    canonical_text: str,
) -> dict[str, Any]:
    source_ranges: list[dict[str, Any]] = []
    validation_ranges: list[dict[str, Any]] = []
    token_count = 0
    for range_index, (start, end) in enumerate(ranges):
        range_text = " ".join(str(word["text"]) for word in words[start:end])
        source_range = _range_payload(
            first_word_id=start,
            last_word_id=end - 1,
            first_word=str(words[start]["text"]),
            last_word=str(words[end - 1]["text"]),
            canonical_text=range_text,
        )
        source_ranges.append(source_range)
        range_tokens = end - start
        token_count += range_tokens
        validation_ranges.append(
            {
                "range_index": range_index,
                **source_range,
                "canonical_tokens": range_tokens,
                "supported_tokens": range_tokens,
                "source_tokens": range_tokens,
                "represented_source_tokens": range_tokens,
                "unrepresented_source_tokens": [],
                "unsupported_tokens": [],
                "status": "valid",
            }
        )
    validation = {
        "thought_index": 0,
        "canonical_tokens": token_count,
        "supported_tokens": token_count,
        "unsupported_tokens": [],
        "status": "valid",
        "source_ranges": validation_ranges,
    }
    return {
        "canonical_text": canonical_text,
        "source_ranges": source_ranges,
        "grounding_validation": validation,
        "committed_iteration": 1,
    }


def _familiar_repair_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, list[dict[str, Any]]]:
    word_specs = [
        ("And", 0.02, 0.05),
        ("it", 0.06, 0.09),
        ("might", 0.10, 0.14),
        ("begin", 0.15, 0.25),
        ("with", 0.35, 0.42),
        ("familiar", 0.45, 0.70),
        ("with", 0.90, 0.98),
        ("the", 1.00, 1.05),
        ("familiar", 1.08, 1.38),
        ("words", 1.60, 1.72),
        ("once", 1.75, 1.84),
        ("upon", 1.87, 1.98),
        ("a", 2.01, 2.05),
        ("time.", 2.08, 2.30),
    ]
    words = [
        {"id": word_id, "text": text, "start": start, "end": end}
        for word_id, (text, start, end) in enumerate(word_specs)
    ]
    source = np.empty((3000, 1), dtype=np.float32)
    timeline = np.arange(len(source), dtype=np.float32) / SAMPLE_RATE
    source[:, 0] = 0.0005 * np.sin(2.0 * np.pi * 11.0 * timeline)
    for word in words:
        _speech(
            source,
            round(float(word["start"]) * SAMPLE_RATE),
            round(float(word["end"]) * SAMPLE_RATE),
            frequency=131.0 + 7.0 * int(word["id"]),
        )
    audio_path = tmp_path / "familiar_source.wav"
    sf.write(audio_path, source, SAMPLE_RATE, subtype="FLOAT")

    transcript_path = tmp_path / "familiar_transcript.json"
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

    canonical_text = "And it might begin with familiar words once upon a time."
    thought = _familiar_thought(
        words=words,
        ranges=[(0, 6), (9, 14)],
        canonical_text=canonical_text,
    )
    grounding_path = tmp_path / "familiar_grounding_validation.json"
    plan_path = tmp_path / "familiar_streaming_plan.json"
    write_json(
        grounding_path,
        {
            "schema_version": 1,
            "validator": "strict_bidirectional_range_source_grounding_v2",
            "status": "valid",
            "finalized_thoughts": 1,
            "source_ranges": 2,
            "canonical_tokens": 11,
            "supported_tokens": 11,
            "unsupported_tokens": [],
            "unrepresented_source_tokens": [],
            "planner_retries": 0,
            "plan_accepted": True,
            "error": None,
            "thoughts": [copy.deepcopy(thought["grounding_validation"])],
        },
    )
    write_json(
        plan_path,
        {
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
            "committed": [thought],
            "committed_words": [*range(0, 6), *range(9, 14)],
            "pending_words": [],
            "next_unread_word": len(words),
            "selected_source_ranges": [
                {"start_word_id": 0, "end_word_id": 6},
                {"start_word_id": 9, "end_word_id": 14},
            ],
            "selected_source_text": canonical_text,
            "reconstructed_narration": canonical_text,
        },
    )

    pause_path = tmp_path / "familiar_pause_plan.json"
    write_json(
        pause_path,
        {
            "schema_version": 1,
            "planner": "semantic_pause_planner_v1",
            "backend": "gemini",
            "model": "cached-test-model",
            "streaming_plan": str(plan_path),
            "streaming_plan_sha256": sha256_file(plan_path),
            "thought_count": 1,
            "transition_count": 0,
            "transitions": [],
            "attempts": [],
        },
    )
    return audio_path, plan_path, pause_path, words


def _scored_alignment_payload(
    *,
    words: list[dict[str, Any]],
    local_word_ids: range,
    weak_terminal_word_id: int | None = None,
) -> dict[str, Any]:
    aligned_words: list[dict[str, Any]] = []
    characters: list[dict[str, Any]] = []
    selected_words = [words[word_id] for word_id in local_word_ids]
    for index, word in enumerate(selected_words):
        text = str(word["text"])
        alphabetic = [character for character in text if character.isalpha()]
        start = float(word["start"])
        end = float(word["end"])
        scores = [0.95] * len(alphabetic)
        word_score = 0.95
        if int(word["id"]) == weak_terminal_word_id:
            scores[-3:] = [0.08, 0.05, 0.03]
            word_score = 0.35
        aligned_words.append(
            {
                "word": text,
                "start": start,
                "end": end,
                "score": word_score,
            }
        )
        character_duration = (end - start) / len(alphabetic)
        for character_index, (character, score) in enumerate(
            zip(alphabetic, scores, strict=True)
        ):
            characters.append(
                {
                    "char": character,
                    "start": start + character_index * character_duration,
                    "end": start + (character_index + 1) * character_duration,
                    "score": score,
                }
            )
        if index < len(selected_words) - 1:
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
                            "text": " ".join(
                                str(word["text"]) for word in selected_words
                            ),
                            "words": aligned_words,
                            "chars": characters,
                        }
                    ],
                },
            }
        ],
    }


def test_final_render_builds_one_boundary_plan_and_renders_source_once(
    tmp_path: Path,
) -> None:
    audio_path, plan_path, pause_plan_path, _ = _grounded_fixture(tmp_path)
    output_dir = tmp_path / "final"

    manifest = render_final_cut(
        audio_path=audio_path,
        plan_path=plan_path,
        output_dir=output_dir,
        pause_plan_path=pause_plan_path,
        alignment_python=tmp_path / "model-must-not-run",
        pause_backend=ExplodingPauseBackend(),
        alignment_payload={"jobs": [_aligned_job(clip_index=0)]},
    )

    assert manifest["status"] == "complete"
    assert manifest["semantic_thoughts"] == 2
    assert manifest["selected_source_ranges"] == 2
    assert manifest["acoustic_repair_attempts"] == 0
    assert manifest["effective_streaming_plan"] == str(plan_path.resolve())
    assert manifest["rendered_clips"] == 2
    assert manifest["alignment_contexts"] == 1
    assert manifest["alignment_resolved_boundaries"] == 2
    assert manifest["unsafe_dense_boundaries"] == 0
    assert manifest["unresolved_boundaries"] == 0
    assert manifest["final_boundary"]["boundary_method"] == "eof_safe_tail"

    final_path = output_dir / "final_cut.wav"
    assert Path(manifest["final_cut_wav"]) == final_path
    assert manifest["final_cut_wav_sha256"] == sha256_file(final_path)
    boundary_plan_path = Path(manifest["final_boundary_plan"])
    assert manifest["final_boundary_plan_sha256"] == sha256_file(boundary_plan_path)
    assert sf.info(final_path).samplerate == SAMPLE_RATE
    assert sf.info(final_path).channels == 1
    assert (output_dir / "pause_plan.json").read_bytes() == (
        pause_plan_path.read_bytes()
    )

    boundary_plan = read_json(boundary_plan_path)
    assert boundary_plan["planner"] == "authoritative_single_pass_boundary_plan_v1"
    assert boundary_plan["status"] == "safe"
    assert len(boundary_plan["source_intervals"]) == 2
    assert len(boundary_plan["output_segments"]) >= 2
    assert all(
        boundary["safety_status"] == "safe" for boundary in boundary_plan["boundaries"]
    )

    assert manifest["debug_artifacts_written"] is False
    assert (output_dir / "alignment_contexts/context_0000.wav").is_file()
    assert not (output_dir / "work").exists()
    rendered_stage_wavs = [
        path
        for path in output_dir.rglob("*.wav")
        if path.name != "final_cut.wav" and "alignment_contexts" not in path.parts
    ]
    assert rendered_stage_wavs == []

    saved = read_json(output_dir / "final_render_manifest.json")
    assert saved == manifest


def test_debug_flag_does_not_reintroduce_staged_production_wavs(
    tmp_path: Path,
) -> None:
    audio_path, plan_path, pause_plan_path, _ = _grounded_fixture(tmp_path)
    output_dir = tmp_path / "final-debug"

    manifest = render_final_cut(
        audio_path=audio_path,
        plan_path=plan_path,
        output_dir=output_dir,
        pause_plan_path=pause_plan_path,
        alignment_python=tmp_path / "model-must-not-run",
        pause_backend=ExplodingPauseBackend(),
        alignment_payload={"jobs": [_aligned_job(clip_index=0)]},
        write_debug_artifacts=True,
    )

    assert manifest["debug_artifacts_requested"] is True
    assert manifest["debug_artifacts_written"] is False
    assert not (output_dir / "work").exists()
    assert not list(output_dir.rglob("rough_cut*.wav"))
    assert not list(output_dir.rglob("*aligned.wav"))


def test_final_render_rejects_stale_grounding_ledger_before_audio_work(
    tmp_path: Path,
) -> None:
    audio_path, plan_path, pause_plan_path, grounding = _grounded_fixture(tmp_path)
    grounding["thoughts"][0]["source_ranges"][0]["last_word_id"] = 1
    grounding_path = Path(read_json(plan_path)["grounding_validation"])
    write_json(grounding_path, grounding)
    output_dir = tmp_path / "rejected"

    with pytest.raises(
        FinalRenderError,
        match="does not match the committed semantic plan",
    ):
        render_final_cut(
            audio_path=audio_path,
            plan_path=plan_path,
            output_dir=output_dir,
            pause_plan_path=pause_plan_path,
            alignment_payload={"jobs": []},
        )

    assert not output_dir.exists()


def test_final_render_fails_closed_on_alignment_failure(
    tmp_path: Path,
) -> None:
    audio_path, plan_path, pause_plan_path, _ = _grounded_fixture(tmp_path)
    output_dir = tmp_path / "failed"

    with pytest.raises(
        FinalRenderError,
        match="forced_alignment_failed",
    ):
        render_final_cut(
            audio_path=audio_path,
            plan_path=plan_path,
            output_dir=output_dir,
            pause_plan_path=pause_plan_path,
            alignment_python=tmp_path / "model-must-not-run",
            alignment_payload={
                "jobs": [
                    {
                        "clip_index": 0,
                        "error": "selected word did not align",
                        "aligned": None,
                    }
                ]
            },
        )

    assert not (output_dir / "final_cut.wav").exists()
    assert (output_dir / "pause_plan.json").is_file()
    boundary_plan = read_json(output_dir / "final_boundary_plan.json")
    assert boundary_plan["status"] == "unsafe"
    assert boundary_plan["alignment_failures"] == 2


def test_final_render_fails_closed_on_dense_leading_boundary(
    tmp_path: Path,
) -> None:
    audio_path, plan_path, pause_plan_path, _ = _grounded_fixture(
        tmp_path,
        dense_leading_boundary=True,
    )
    output_dir = tmp_path / "failed_leading"

    with pytest.raises(
        FinalRenderError,
        match="unsafe_dense_boundary",
    ):
        render_final_cut(
            audio_path=audio_path,
            plan_path=plan_path,
            output_dir=output_dir,
            pause_plan_path=pause_plan_path,
            alignment_python=tmp_path / "model-must-not-run",
            alignment_payload={"jobs": [_aligned_job(clip_index=0)]},
            max_acoustic_retries=0,
        )

    assert not (output_dir / "final_cut.wav").exists()
    boundary_plan = read_json(output_dir / "final_boundary_plan.json")
    assert boundary_plan["unsafe_dense_boundaries"] == 1


def test_final_boundary_plan_cannot_change_during_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path, plan_path, pause_plan_path, _ = _grounded_fixture(tmp_path)
    output_dir = tmp_path / "mutated"
    original_render = final_render_module.render_boundary_plan

    def mutate_after_render(**kwargs: Any) -> dict[str, Any]:
        result = original_render(**kwargs)
        boundary_path = Path(kwargs["boundary_plan_path"])
        changed = read_json(boundary_path)
        changed["boundaries"][1]["selected_source_sample"] += 1
        write_json(boundary_path, changed)
        return result

    monkeypatch.setattr(
        final_render_module,
        "render_boundary_plan",
        mutate_after_render,
    )
    with pytest.raises(FinalRenderError, match="changed during rendering"):
        render_final_cut(
            audio_path=audio_path,
            plan_path=plan_path,
            output_dir=output_dir,
            pause_plan_path=pause_plan_path,
            alignment_python=tmp_path / "model-must-not-run",
            alignment_payload={"jobs": [_aligned_job(clip_index=0)]},
        )

    assert not (output_dir / "final_cut.wav").exists()
    assert not (output_dir / "final_render_manifest.json").exists()


def test_weak_first_familiar_is_repaired_to_complete_later_occurrence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path, plan_path, pause_path, words = _familiar_repair_fixture(tmp_path)
    output_dir = tmp_path / "familiar_final"
    repair_backend = StaticRepairBackend(
        {
            "finalized": [
                {
                    "canonical_text": (
                        "And it might begin with the familiar words once upon a time."
                    ),
                    "source_ranges": [
                        {
                            "first_word_id": 0,
                            "last_word_id": 3,
                            "first_word": "And",
                            "last_word": "begin",
                            "canonical_text": "And it might begin",
                        },
                        {
                            "first_word_id": 6,
                            "last_word_id": 13,
                            "first_word": "with",
                            "last_word": "time.",
                            "canonical_text": (
                                "with the familiar words once upon a time."
                            ),
                        },
                    ],
                }
            ],
            "pending_start_word_id": None,
            "pending_reason": "The complete later correction is retained.",
        }
    )
    render_calls = 0
    original_render = final_render_module.render_boundary_plan

    def count_authoritative_render(**kwargs: Any) -> dict[str, Any]:
        nonlocal render_calls
        render_calls += 1
        return original_render(**kwargs)

    monkeypatch.setattr(
        final_render_module,
        "render_boundary_plan",
        count_authoritative_render,
    )

    manifest = render_final_cut(
        audio_path=audio_path,
        plan_path=plan_path,
        output_dir=output_dir,
        pause_plan_path=pause_path,
        alignment_python=tmp_path / "model-must-not-run",
        repair_backend=repair_backend,
        alignment_payloads=[
            _scored_alignment_payload(
                words=words,
                local_word_ids=range(0, 14),
                weak_terminal_word_id=5,
            ),
            _scored_alignment_payload(
                words=words,
                local_word_ids=range(0, 10),
            ),
        ],
        max_acoustic_retries=1,
    )

    assert render_calls == 1
    assert manifest["acoustic_repair_attempts"] == 1
    assert len(repair_backend.prompts) == 1

    rejected_path = Path(manifest["acoustic_repairs"][0]["rejected_boundary_plan"])
    rejected_plan = read_json(rejected_path)
    weak_boundary = next(
        boundary
        for boundary in rejected_plan["boundaries"]
        if boundary.get("safety_status") == "weak_retained_word_alignment"
    )
    assert weak_boundary["source_word_ids"]["last_retained_left"] == 5
    assert 5 in weak_boundary["forbidden_word_ids"]
    assert weak_boundary["retained_word_support"]["status"] == (
        "weak_terminal_word_support"
    )

    effective_plan = read_json(Path(manifest["effective_streaming_plan"]))
    selected_ranges = effective_plan["selected_source_ranges"]
    assert selected_ranges == [
        {"start_word_id": 0, "end_word_id": 4},
        {"start_word_id": 6, "end_word_id": 14},
    ]
    assert not any(
        source_range["start_word_id"] <= 5 < source_range["end_word_id"]
        for source_range in selected_ranges
    )
    assert any(
        source_range["start_word_id"] <= 8 < source_range["end_word_id"]
        for source_range in selected_ranges
    )
    assert not (
        any(source_range["end_word_id"] == 6 for source_range in selected_ranges)
        and any(source_range["start_word_id"] == 9 for source_range in selected_ranges)
    )

    boundary_plan = read_json(Path(manifest["final_boundary_plan"]))
    assert boundary_plan["status"] == "safe"
    later_familiar = next(
        span
        for context in boundary_plan["alignment_contexts"]
        for span in context["aligned_word_spans"]
        if span["word_id"] == 8
    )
    familiar_start = int(later_familiar["start_sample"])
    familiar_end = int(later_familiar["end_sample"])
    assert familiar_start < familiar_end
    for boundary in boundary_plan["boundaries"]:
        selected_sample = boundary.get("selected_source_sample")
        if selected_sample is not None:
            assert not familiar_start < int(selected_sample) < familiar_end
        for fade in boundary.get("fade_intervals", []):
            assert max(familiar_start, int(fade["source_start_sample"])) >= min(
                familiar_end,
                int(fade["source_end_sample"]),
            )
    for join in boundary_plan["joins"]:
        for fade in join.get("fade_intervals", []):
            assert max(familiar_start, int(fade["source_start_sample"])) >= min(
                familiar_end,
                int(fade["source_end_sample"]),
            )

    source_segment = next(
        segment
        for segment in boundary_plan["output_segments"]
        if segment["kind"] == "source"
        and int(segment["source_start_sample"]) <= familiar_start
        and int(segment["source_end_sample"]) >= familiar_end
    )
    output_start = int(source_segment["output_start_sample"]) + (
        familiar_start - int(source_segment["source_start_sample"])
    )
    output_end = output_start + familiar_end - familiar_start
    source_audio, _ = sf.read(audio_path, dtype="float32", always_2d=True)
    rendered_audio, _ = sf.read(
        Path(manifest["final_cut_wav"]),
        dtype="float32",
        always_2d=True,
    )
    assert np.array_equal(
        rendered_audio[output_start:output_end],
        source_audio[familiar_start:familiar_end],
    )

    rendered_wavs = sorted(
        path
        for path in output_dir.rglob("*.wav")
        if "alignment_contexts" not in path.parts
    )
    assert rendered_wavs == [Path(manifest["final_cut_wav"])]
