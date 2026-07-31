from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from voicecut.common import read_json, sha256_file, write_json
from voicecut.semantic_pause import (
    PausePlanValidationError,
    _insert_audio_segments,
    create_pause_plan,
    find_quiet_insertion_point,
    refine_eof_tail,
    render_semantic_pauses,
    validate_pause_response,
)


class FakePauseBackend:
    backend_name = "gemini"
    model = "fake-gemini"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any],
        request_id: str,
    ) -> str:
        del response_schema, request_id
        self.prompts.append(prompt)
        return self.responses.pop(0)

    def close(self) -> None:
        return None


def _speech(audio: np.ndarray, start: int, end: int, *, rate: int) -> None:
    time = np.arange(end - start, dtype=np.float32) / rate
    audio[start:end, 0] = 0.25 * np.sin(2.0 * np.pi * 170.0 * time)


def _synthetic_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    sample_rate = 1000
    source = np.zeros((2200, 1), dtype=np.float32)
    room_time = np.arange(len(source), dtype=np.float32) / sample_rate
    source[:, 0] = 0.001 * np.sin(2.0 * np.pi * 17.0 * room_time)
    _speech(source, 100, 300, rate=sample_rate)
    _speech(source, 500, 700, rate=sample_rate)
    _speech(source, 900, 1000, rate=sample_rate)  # omitted
    _speech(source, 1200, 1500, rate=sample_rate)
    audio_path = tmp_path / "source.wav"
    sf.write(audio_path, source, sample_rate, subtype="FLOAT")

    words = [
        {"id": 0, "text": "alpha", "start": 0.10, "end": 0.30},
        {"id": 1, "text": "beta", "start": 0.50, "end": 0.70},
        {"id": 2, "text": "wrong", "start": 0.90, "end": 1.00},
        {"id": 3, "text": "omega", "start": 1.20, "end": 1.40},
    ]
    plan = {
        "schema_version": 1,
        "planner": "streaming_narration_v1",
        "status": "complete",
        "words": words,
        "word_count": len(words),
        "committed": [
            {
                "canonical_text": "Alpha.",
                "source_ranges": [{"start_word_id": 0, "end_word_id": 1}],
            },
            {
                "canonical_text": "Beta.",
                "source_ranges": [{"start_word_id": 1, "end_word_id": 2}],
            },
            {
                "canonical_text": "Omega.",
                "source_ranges": [{"start_word_id": 3, "end_word_id": 4}],
            },
        ],
        "selected_source_ranges": [
            {"start_word_id": 0, "end_word_id": 1},
            {"start_word_id": 1, "end_word_id": 2},
            {"start_word_id": 3, "end_word_id": 4},
        ],
    }
    plan_path = tmp_path / "streaming_plan.json"
    write_json(plan_path, plan)

    clip_0 = source[50:800].copy()
    clip_1 = source[1150:1500].copy()
    previous = np.concatenate(
        [clip_0, np.zeros((80, 1), dtype=np.float32), clip_1],
        axis=0,
    )
    previous_path = tmp_path / "full_boundary_aligned.wav"
    sf.write(previous_path, previous, sample_rate, subtype="FLOAT")
    manifest = {
        "schema_version": 1,
        "renderer": "streaming_plan_full_boundary_alignment_v1",
        "source_audio": str(audio_path),
        "source_audio_sha256": sha256_file(audio_path),
        "streaming_plan": str(plan_path),
        "streaming_plan_sha256": sha256_file(plan_path),
        "source_sample_rate": sample_rate,
        "source_channel_count": 1,
        "source_frame_count": len(source),
        "configuration": {
            "edge_padding_ms": 30.0,
            "clip_fade_ms": 5.0,
            "inter_clip_silence_ms": 80.0,
        },
        "full_boundary_aligned_wav": str(previous_path),
        "full_boundary_aligned_wav_sha256": sha256_file(previous_path),
        "full_boundary_aligned_expected_output_frame_count": len(previous),
        "clips": [
            {
                "clip_index": 0,
                "source_word_start": 0,
                "source_word_end": 2,
                "final_source_start_sample": 50,
                "final_source_end_sample": 800,
                "final_output_start_sample": 0,
                "final_output_end_sample": 750,
                "final_frame_count": 750,
                "final_fade_in_samples": 5,
                "final_fade_out_samples": 5,
            },
            {
                "clip_index": 1,
                "source_word_start": 3,
                "source_word_end": 4,
                "final_source_start_sample": 1150,
                "final_source_end_sample": 1500,
                "final_output_start_sample": 830,
                "final_output_end_sample": 1180,
                "final_frame_count": 350,
                "final_fade_in_samples": 5,
                "final_fade_out_samples": 5,
            },
        ],
    }
    manifest_path = tmp_path / "render_manifest_forced_aligned.json"
    write_json(manifest_path, manifest)
    return audio_path, plan_path, manifest_path


def test_pause_response_requires_exact_order_and_count() -> None:
    raw = json.dumps(
        {
            "transitions": [
                {
                    "after_thought_index": 0,
                    "before_thought_index": 1,
                    "pause_type": "short",
                },
                {
                    "after_thought_index": 1,
                    "before_thought_index": 2,
                    "pause_type": "thought",
                },
            ]
        }
    )
    assert [
        item["pause_type"] for item in validate_pause_response(raw, thought_count=3)
    ] == ["short", "thought"]
    invalid = raw.replace('"before_thought_index": 2', '"before_thought_index": 3')
    with pytest.raises(PausePlanValidationError):
        validate_pause_response(invalid, thought_count=3)


def test_pause_planner_retries_malformed_json_once(tmp_path: Path) -> None:
    _, plan_path, _ = _synthetic_fixture(tmp_path)
    valid = json.dumps(
        {
            "transitions": [
                {
                    "after_thought_index": 0,
                    "before_thought_index": 1,
                    "pause_type": "short",
                },
                {
                    "after_thought_index": 1,
                    "before_thought_index": 2,
                    "pause_type": "thought",
                },
            ]
        }
    )
    backend = FakePauseBackend(["not-json", valid])
    output_dir = tmp_path / "pause_output"
    pause_plan = create_pause_plan(
        plan_path=plan_path,
        output_dir=output_dir,
        backend=backend,
    )
    assert pause_plan["transition_count"] == 2
    assert len(backend.prompts) == 2
    assert "VALIDATION ERROR" in backend.prompts[1]
    assert (output_dir / "pause_plan_attempt_1.raw.txt").is_file()
    assert (output_dir / "pause_plan_attempt_2.raw.json").is_file()


def test_quiet_insertion_requires_real_low_energy() -> None:
    sample_rate = 1000
    quiet_audio = np.zeros(1000, dtype=np.float32)
    quiet_audio[100:300] = 0.2
    quiet_audio[500:700] = 0.2
    point = find_quiet_insertion_point(
        quiet_audio,
        sample_rate=sample_rate,
        previous_word_end_seconds=0.30,
        next_word_start_seconds=0.50,
        clip_start_sample=50,
        clip_end_sample=750,
    )
    assert point is not None
    assert 300 <= point.source_sample <= 500
    active = np.full(1000, 0.2, dtype=np.float32)
    assert (
        find_quiet_insertion_point(
            active,
            sample_rate=sample_rate,
            previous_word_end_seconds=0.30,
            next_word_start_seconds=0.50,
            clip_start_sample=50,
            clip_end_sample=750,
        )
        is None
    )


def test_eof_tail_waits_180_ms_and_keeps_300_ms_tail() -> None:
    sample_rate = 1000
    mono = np.zeros(2000, dtype=np.float32)
    # Speech has already decayed before the deliberately delayed EOF search
    # begins.  Unlike an ordinary edit boundary, the safe-tail search may
    # accept this existing silence because there is no following word at risk.
    mono[100:1400] = 0.2
    decision = refine_eof_tail(
        mono,
        sample_rate=sample_rate,
        raw_end_seconds=1.30,
        previous_end_sample=1320,
        fade_ms=5.0,
    )
    assert decision.search_start_sample == 1480
    assert decision.stable_silence_start_sample == 1480
    assert decision.new_end_sample == (decision.stable_silence_start_sample + 300)
    assert decision.fade_out_samples == 5

    no_silence = np.full(2000, 0.2, dtype=np.float32)
    fallback = refine_eof_tail(
        no_silence,
        sample_rate=sample_rate,
        raw_end_seconds=1.30,
        previous_end_sample=1320,
        fade_ms=5.0,
    )
    assert fallback.stable_silence_start_sample is None
    assert fallback.new_end_sample == len(no_silence)
    assert fallback.fade_out_samples == 0


def test_insert_source_audio_preserves_every_original_sample() -> None:
    audio = np.arange(10, dtype=np.float32)[:, None]
    rendered = _insert_audio_segments(
        audio,
        insertions=[
            (3, np.full((2, 1), -0.25, dtype=np.float32)),
            (7, np.full((1, 1), 0.125, dtype=np.float32)),
        ],
    )
    restored = np.concatenate([rendered[:3], rendered[5:9], rendered[10:]])
    np.testing.assert_array_equal(restored, audio)


def test_semantic_render_preserves_ranges_and_replaces_join_pause(
    tmp_path: Path,
) -> None:
    _, plan_path, render_manifest_path = _synthetic_fixture(tmp_path)
    output_dir = tmp_path / "semantic"
    output_dir.mkdir()
    pause_plan = {
        "schema_version": 1,
        "planner": "semantic_pause_planner_v1",
        "backend": "gemini",
        "model": "fake-gemini",
        "streaming_plan": str(plan_path),
        "streaming_plan_sha256": sha256_file(plan_path),
        "thought_count": 3,
        "transition_count": 2,
        "transitions": [
            {
                "after_thought_index": 0,
                "before_thought_index": 1,
                "pause_type": "short",
            },
            {
                "after_thought_index": 1,
                "before_thought_index": 2,
                "pause_type": "thought",
            },
        ],
        "attempts": [],
    }
    pause_plan_path = output_dir / "pause_plan.json"
    write_json(pause_plan_path, pause_plan)
    manifest = render_semantic_pauses(
        render_manifest_path=render_manifest_path,
        pause_plan_path=pause_plan_path,
        output_dir=output_dir,
    )
    assert manifest["thought_transition_count"] == 2
    assert manifest["transitions"][0]["insertion_method"] == ("quiet_waveform_point")
    join_transition = manifest["transitions"][1]
    assert join_transition["estimated_existing_pause_ms"] == 140.0
    assert join_transition["inserted_pause_ms"] == 510.0
    assert join_transition["estimated_total_pause_ms"] == 650.0
    assert join_transition["pause_fill_method"] == "verified_source_room_tone"
    assert join_transition["room_tone_source_ranges_seconds"]
    rendered, _ = sf.read(
        manifest["rough_cut_with_semantic_pauses_wav"],
        dtype="float32",
        always_2d=True,
    )
    pause_start = round(join_transition["output_pause_start_seconds"] * 1000)
    pause_end = round(join_transition["output_pause_end_seconds"] * 1000)
    assert np.any(rendered[pause_start:pause_end] != 0.0)
    assert manifest["final_boundary"]["boundary_method"] == "eof_safe_tail"
    assert manifest["clips"][0]["source_word_start"] == 0
    assert manifest["clips"][0]["source_word_end"] == 2
    output_info = sf.info(manifest["rough_cut_with_semantic_pauses_wav"])
    assert output_info.frames == manifest["semantic_pause_output_frame_count"]


def test_semantic_render_does_not_apply_eof_tail_before_omitted_word(
    tmp_path: Path,
) -> None:
    _, plan_path, render_manifest_path = _synthetic_fixture(tmp_path)
    plan = read_json(plan_path)
    plan["words"].append(
        {
            "id": 4,
            "text": "direction",
            "start": 1.70,
            "end": 1.90,
        }
    )
    plan["word_count"] = 5
    write_json(plan_path, plan)
    render_manifest = read_json(render_manifest_path)
    render_manifest["streaming_plan_sha256"] = sha256_file(plan_path)
    write_json(render_manifest_path, render_manifest)

    output_dir = tmp_path / "not_eof"
    output_dir.mkdir()
    pause_plan = {
        "schema_version": 1,
        "planner": "semantic_pause_planner_v1",
        "backend": "gemini",
        "model": "fake-gemini",
        "streaming_plan": str(plan_path),
        "streaming_plan_sha256": sha256_file(plan_path),
        "thought_count": 3,
        "transition_count": 2,
        "transitions": [
            {
                "after_thought_index": 0,
                "before_thought_index": 1,
                "pause_type": "short",
            },
            {
                "after_thought_index": 1,
                "before_thought_index": 2,
                "pause_type": "thought",
            },
        ],
        "attempts": [],
    }
    pause_plan_path = output_dir / "pause_plan.json"
    write_json(pause_plan_path, pause_plan)
    manifest = render_semantic_pauses(
        render_manifest_path=render_manifest_path,
        pause_plan_path=pause_plan_path,
        output_dir=output_dir,
    )
    assert manifest["final_boundary"]["boundary_method"] == "not_end_of_file"
    assert manifest["final_boundary"]["new_final_end_seconds"] == 1.5
