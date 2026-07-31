from __future__ import annotations

import copy
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
        chars.append(
            {
                "char": word[0],
                "start": start,
                "end": end,
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
