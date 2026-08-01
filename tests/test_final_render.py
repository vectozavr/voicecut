from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import voicecut.final_render as final_render_module
from voicecut.breath_detection import (
    RESPIRO_CHECKPOINT_SHA256,
    RESPIRO_FRAME_HOP_MS,
    RESPIRO_UPSTREAM_COMMIT,
)
from voicecut.common import read_json, sha256_file, write_json
from voicecut.final_render import FinalRenderError, render_final_cut
from voicecut.mfa_alignment import MFA_MODEL_ID, MFA_VERSION

SAMPLE_RATE = 1000


@pytest.fixture(autouse=True)
def _forbid_real_mfa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test in this module must supply deterministic MFA evidence."""

    def explode(**_: Any) -> dict[str, Any]:
        raise AssertionError("unit tests must never invoke the real MFA runtime")

    monkeypatch.setattr(final_render_module, "align_mfa_contexts", explode)


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


def _completeness_job(
    *,
    clip_index: int,
    words: list[dict[str, Any]],
    local_word_ids: range,
    weak_terminal_word_id: int | None = None,
) -> dict[str, Any]:
    selected_words = [words[word_id] for word_id in local_word_ids]
    crop_start = max(0.0, float(selected_words[0]["start"]) - 0.4)
    word_segments: list[dict[str, Any]] = []
    chars: list[dict[str, Any]] = []
    for index, word in enumerate(selected_words):
        text = str(word["text"])
        alphabetic = [character for character in text if character.isalpha()]
        start = float(word["start"]) - crop_start
        end = float(word["end"]) - crop_start
        scores = [0.95] * len(alphabetic)
        word_score = 0.95
        if int(word["id"]) == weak_terminal_word_id:
            scores[-3:] = [0.08, 0.05, 0.03]
            word_score = 0.35
        word_segments.append(
            {
                "word": text,
                "start": start,
                "end": end,
                "score": word_score,
            }
        )
        duration = (end - start) / len(alphabetic)
        for character_index, (character, score) in enumerate(
            zip(alphabetic, scores, strict=True)
        ):
            character_start = start + character_index * duration
            character_end = start + (character_index + 1) * duration
            chars.append(
                {
                    "char": character,
                    "start": character_start,
                    "end": character_end,
                    "score": score,
                }
            )
        if index < len(selected_words) - 1:
            chars.append({"char": " "})
    return {
        "clip_index": clip_index,
        "error": None,
        "aligned": {
            "word_segments": word_segments,
            "segments": [
                {
                    "text": " ".join(str(word["text"]) for word in selected_words),
                    "words": word_segments,
                    "chars": chars,
                }
            ],
        },
    }


def _completeness_payload(
    *,
    words: list[dict[str, Any]],
    contexts: list[tuple[range, int | None]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "backend": "whisperx_alignment",
        "language": "en",
        "device": "cpu",
        "purpose": "retained_word_completeness_veto",
        "coordinate_authority": False,
        "jobs": [
            _completeness_job(
                clip_index=clip_index,
                words=words,
                local_word_ids=local_word_ids,
                weak_terminal_word_id=weak_word_id,
            )
            for clip_index, (local_word_ids, weak_word_id) in enumerate(contexts)
        ],
    }


def _mfa_context(
    *,
    context_id: str,
    words: list[dict[str, Any]],
    local_word_ids: range,
    selected_word_ids: set[int],
    crop_start_sample: int,
    crop_end_sample: int,
    boundary_overrides: dict[int, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    overrides = boundary_overrides or {}
    mapped_words: list[dict[str, Any]] = []
    phones: list[dict[str, Any]] = []
    for word_id in local_word_ids:
        source_word = words[word_id]
        start, end = overrides.get(
            word_id,
            (float(source_word["start"]), float(source_word["end"])),
        )
        start_sample = round(start * SAMPLE_RATE)
        end_sample = round(end * SAMPLE_RATE)
        phone = {
            "phone": "S"
            if str(source_word["text"]).lower().rstrip(".") == "words"
            else "AA",
            "start_seconds": start,
            "end_seconds": end,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "is_silence": False,
        }
        phones.append(phone)
        mapped_words.append(
            {
                "source_word_ids": [word_id],
                "source_text": str(source_word["text"]),
                "mfa_token": str(source_word["text"]).lower().strip(".,!?;:"),
                "start_seconds": start,
                "end_seconds": end,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "phones": [dict(phone)],
            }
        )
    crop_start = crop_start_sample / SAMPLE_RATE
    crop_end = crop_end_sample / SAMPLE_RATE
    return {
        "context_id": context_id,
        "crop_source_start_seconds": crop_start,
        "crop_source_end_seconds": crop_end,
        "crop_source_start_sample": crop_start_sample,
        "crop_source_end_sample": crop_end_sample,
        "ordered_source_word_ids": list(local_word_ids),
        "original_source_words": [
            {
                "word_id": word_id,
                "text": str(words[word_id]["text"]),
                "start_seconds": float(words[word_id]["start"]),
                "end_seconds": float(words[word_id]["end"]),
                "selected": word_id in selected_word_ids,
            }
            for word_id in local_word_ids
        ],
        "boundary_ids": [context_id],
        "words": mapped_words,
        "phones": phones,
        "mfa_output_json": f"mock/{context_id}.json",
    }


def _mfa_payload(
    *,
    words: list[dict[str, Any]],
    contexts: list[
        tuple[
            str,
            range,
            tuple[int, int],
            dict[int, tuple[float, float]] | None,
        ]
    ],
    selected_word_ids: set[int],
    source_audio_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "backend": "mfa",
        "mfa_version": MFA_VERSION,
        "model_id": MFA_MODEL_ID,
        "fine_tune": True,
        "sample_rate": SAMPLE_RATE,
        "source_audio_sha256": source_audio_sha256,
        "contexts": [
            _mfa_context(
                context_id=context_id,
                words=words,
                local_word_ids=local_word_ids,
                selected_word_ids=selected_word_ids,
                crop_start_sample=crop_samples[0],
                crop_end_sample=crop_samples[1],
                boundary_overrides=overrides,
            )
            for context_id, local_word_ids, crop_samples, overrides in contexts
        ],
    }


def _context_geometry(
    words: list[dict[str, Any]],
    *,
    role_word_ids: tuple[int, ...],
    total_samples: int,
) -> tuple[range, tuple[int, int]]:
    context_start = max(0, min(role_word_ids) - 3)
    context_end = min(len(words), max(role_word_ids) + 4)
    crop_start = max(
        0,
        math.floor(float(words[context_start]["start"]) * SAMPLE_RATE) - 400,
    )
    crop_end = min(
        total_samples,
        math.ceil(float(words[context_end - 1]["end"]) * SAMPLE_RATE) + 400,
    )
    while context_start > 0:
        previous = words[context_start - 1]
        if math.ceil(float(previous["end"]) * SAMPLE_RATE) <= crop_start:
            break
        context_start -= 1
        crop_start = min(
            crop_start,
            math.floor(float(previous["start"]) * SAMPLE_RATE),
        )
    while context_end < len(words):
        following = words[context_end]
        if math.floor(float(following["start"]) * SAMPLE_RATE) >= crop_end:
            break
        crop_end = max(
            crop_end,
            math.ceil(float(following["end"]) * SAMPLE_RATE),
        )
        context_end += 1
    first_word_start = math.floor(float(words[context_start]["start"]) * SAMPLE_RATE)
    previous_word_end = (
        math.ceil(float(words[context_start - 1]["end"]) * SAMPLE_RATE)
        if context_start > 0
        else 0
    )
    crop_start = min(crop_start, max(previous_word_end, first_word_start - 400))
    last_word_end = math.ceil(float(words[context_end - 1]["end"]) * SAMPLE_RATE)
    next_word_start = (
        math.floor(float(words[context_end]["start"]) * SAMPLE_RATE)
        if context_end < len(words)
        else total_samples
    )
    crop_end = max(crop_end, min(next_word_start, last_word_end + 400))
    return range(context_start, context_end), (crop_start, crop_end)


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
            "committed_words": [*range(6), *range(9, 14)],
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


def _breath_cleanup_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, list[dict[str, Any]]]:
    source = np.empty((2200, 1), dtype=np.float32)
    time = np.arange(len(source), dtype=np.float32) / SAMPLE_RATE
    source[:, 0] = 0.001 * np.sin(2.0 * np.pi * 13.0 * time)
    _speech(source, 100, 500, frequency=173.0)
    breath_time = np.arange(130, dtype=np.float32) / SAMPLE_RATE
    source[650:780, 0] += 0.002 * np.sin(2.0 * np.pi * 47.0 * breath_time)
    _speech(source, 1000, 1400, frequency=211.0)
    audio_path = tmp_path / "breath_source.wav"
    sf.write(audio_path, source, SAMPLE_RATE, subtype="FLOAT")

    transcript_path = tmp_path / "breath_source_transcript.json"
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
    words = [
        {"id": 0, "text": "words", "start": 0.10, "end": 0.50},
        {"id": 1, "text": "continue.", "start": 1.00, "end": 1.40},
    ]
    committed = [
        _familiar_thought(
            words=words,
            ranges=[(0, 2)],
            canonical_text="Words continue.",
        )
    ]
    grounding_path = tmp_path / "breath_grounding_validation.json"
    plan_path = tmp_path / "breath_streaming_plan.json"
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
        "selected_source_ranges": [{"start_word_id": 0, "end_word_id": 2}],
        "reconstructed_narration": "Words continue.",
    }
    write_json(plan_path, plan)
    write_json(
        grounding_path,
        {
            "schema_version": 1,
            "validator": "strict_bidirectional_range_source_grounding_v2",
            "status": "valid",
            "finalized_thoughts": 1,
            "source_ranges": 1,
            "canonical_tokens": 2,
            "supported_tokens": 2,
            "unsupported_tokens": [],
            "unrepresented_source_tokens": [],
            "planner_retries": 0,
            "plan_accepted": True,
            "error": None,
            "thoughts": [copy.deepcopy(committed[0]["grounding_validation"])],
        },
    )
    pause_path = tmp_path / "breath_pause_plan.json"
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


def _breath_cleanup_alignment_evidence(
    *,
    words: list[dict[str, Any]],
    audio_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    eof_words, eof_crop = _context_geometry(
        words,
        role_word_ids=(1,),
        total_samples=2200,
    )
    gap_words, gap_crop = _context_geometry(
        words,
        role_word_ids=(0, 1),
        total_samples=2200,
    )
    completeness = _completeness_payload(
        words=words,
        contexts=[(eof_words, None)],
    )
    mfa = _mfa_payload(
        words=words,
        selected_word_ids={0, 1},
        source_audio_sha256=sha256_file(audio_path),
        contexts=[
            ("eof_tail", eof_words, eof_crop, None),
            (
                "breath_retained_gap_0000",
                gap_words,
                gap_crop,
                None,
            ),
        ],
    )
    return completeness, mfa


def _breath_detector_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "backend": "respiro-en",
        "upstream_commit": RESPIRO_UPSTREAM_COMMIT,
        "checkpoint_sha256": RESPIRO_CHECKPOINT_SHA256,
        "frame_hop_ms": RESPIRO_FRAME_HOP_MS,
        "threshold": 0.5,
        "minimum_duration_ms": 80,
        "status": "complete",
        "events": [
            {
                "start_sample": 450,
                "end_sample": 530,
                "start_seconds": 0.45,
                "end_seconds": 0.53,
                "peak_probability": 0.91,
            },
            {
                "start_sample": 650,
                "end_sample": 780,
                "start_seconds": 0.65,
                "end_seconds": 0.78,
                "peak_probability": 0.97,
            },
        ],
    }


def _grounded_evidence(
    *,
    words: list[dict[str, Any]],
    audio_path: Path,
    dense_leading_boundary: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    completeness = _completeness_payload(
        words=words,
        contexts=[
            (range(3), None),
            (range(3), None),
            (range(3), None),
        ],
    )
    acoustic_intervals = {
        0: (0.10, 0.62),
        1: (0.72, 0.95),
        2: (1.20, 1.52),
    }
    dense_override = {
        **acoustic_intervals,
        1: (0.72, 1.20),
        2: (1.20, 1.52),
    }
    mfa = _mfa_payload(
        words=words,
        selected_word_ids={0, 2},
        source_audio_sha256=sha256_file(audio_path),
        contexts=[
            (
                "source_gap_0000_left",
                *_context_geometry(
                    words,
                    role_word_ids=(0, 1),
                    total_samples=2300,
                ),
                acoustic_intervals,
            ),
            (
                "source_gap_0000_right",
                *_context_geometry(
                    words,
                    role_word_ids=(1, 2),
                    total_samples=2300,
                ),
                dense_override if dense_leading_boundary else acoustic_intervals,
            ),
            (
                "eof_tail",
                *_context_geometry(
                    words,
                    role_word_ids=(2,),
                    total_samples=2300,
                ),
                acoustic_intervals,
            ),
        ],
    )
    return completeness, mfa


def _initial_familiar_completeness(
    words: list[dict[str, Any]],
) -> dict[str, Any]:
    left, _ = _context_geometry(
        words,
        role_word_ids=(5, 6),
        total_samples=3000,
    )
    right, _ = _context_geometry(
        words,
        role_word_ids=(8, 9),
        total_samples=3000,
    )
    eof, _ = _context_geometry(
        words,
        role_word_ids=(13,),
        total_samples=3000,
    )
    return _completeness_payload(
        words=words,
        contexts=[
            (left, 5),
            (right, None),
            (eof, None),
        ],
    )


def _repaired_familiar_evidence(
    words: list[dict[str, Any]],
    *,
    audio_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    left, left_crop = _context_geometry(
        words,
        role_word_ids=(3, 4),
        total_samples=3000,
    )
    right, right_crop = _context_geometry(
        words,
        role_word_ids=(5, 6),
        total_samples=3000,
    )
    eof, eof_crop = _context_geometry(
        words,
        role_word_ids=(13,),
        total_samples=3000,
    )
    completeness = _completeness_payload(
        words=words,
        contexts=[
            (left, None),
            (right, None),
            (eof, None),
        ],
    )
    mfa = _mfa_payload(
        words=words,
        selected_word_ids={*range(4), *range(6, 14)},
        source_audio_sha256=sha256_file(audio_path),
        contexts=[
            (
                "source_gap_0000_left",
                left,
                left_crop,
                {3: (0.15, 0.35), 4: (0.35, 0.42)},
            ),
            ("source_gap_0000_right", right, right_crop, None),
            ("eof_tail", eof, eof_crop, None),
        ],
    )
    return completeness, mfa


def test_final_render_builds_one_boundary_plan_and_renders_source_once(
    tmp_path: Path,
) -> None:
    audio_path, plan_path, pause_plan_path, _ = _grounded_fixture(tmp_path)
    words = read_json(plan_path)["words"]
    completeness, mfa = _grounded_evidence(words=words, audio_path=audio_path)
    output_dir = tmp_path / "final"

    manifest = render_final_cut(
        audio_path=audio_path,
        plan_path=plan_path,
        output_dir=output_dir,
        pause_plan_path=pause_plan_path,
        alignment_python=tmp_path / "model-must-not-run",
        pause_backend=ExplodingPauseBackend(),
        alignment_payload=completeness,
        mfa_payload=mfa,
    )

    assert manifest["status"] == "complete"
    assert manifest["semantic_thoughts"] == 2
    assert manifest["selected_source_ranges"] == 2
    assert manifest["acoustic_repair_attempts"] == 0
    assert manifest["effective_streaming_plan"] == str(plan_path.resolve())
    assert manifest["rendered_clips"] == 2
    assert manifest["alignment_backend"] == "mfa"
    assert manifest["mfa_version"] == MFA_VERSION
    assert manifest["mfa_model"] == MFA_MODEL_ID
    assert manifest["mfa_fine_tune"] is True
    assert manifest["alignment_contexts"] == 3
    assert manifest["alignment_resolved_boundaries"] == 3
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
    assert boundary_plan["schema_version"] == 2
    assert boundary_plan["planner"] == "authoritative_single_pass_boundary_plan_v2"
    assert boundary_plan["status"] == "safe"
    assert boundary_plan["alignment_backend"] == "mfa"
    assert boundary_plan["mfa_version"] == MFA_VERSION
    assert boundary_plan["mfa_model"] == MFA_MODEL_ID
    assert boundary_plan["mfa_fine_tune"] is True
    assert len(boundary_plan["source_intervals"]) == 2
    assert len(boundary_plan["output_segments"]) >= 2
    assert all(
        boundary["safety_status"] == "safe" for boundary in boundary_plan["boundaries"]
    )

    assert manifest["debug_artifacts_written"] is False
    assert (output_dir / "completeness_contexts/context_0000.wav").is_file()
    assert (output_dir / "mfa_alignment/metadata/mfa_alignment.json").is_file()
    assert not (output_dir / "work").exists()
    rendered_stage_wavs = [
        path
        for path in output_dir.rglob("*.wav")
        if path.name != "final_cut.wav" and "completeness_contexts" not in path.parts
    ]
    assert rendered_stage_wavs == []

    saved = read_json(output_dir / "final_render_manifest.json")
    assert saved == manifest


def test_cut_pause_policy_skips_planner_and_inserts_no_room_tone(
    tmp_path: Path,
) -> None:
    audio_path, plan_path, _, _ = _grounded_fixture(tmp_path)
    words = read_json(plan_path)["words"]
    completeness, mfa = _grounded_evidence(words=words, audio_path=audio_path)
    output_dir = tmp_path / "video-cuts"

    manifest = render_final_cut(
        audio_path=audio_path,
        plan_path=plan_path,
        output_dir=output_dir,
        alignment_python=tmp_path / "model-must-not-run",
        pause_backend=ExplodingPauseBackend(),
        alignment_payload=completeness,
        mfa_payload=mfa,
        pause_policy="cuts",
    )

    pause_plan = read_json(output_dir / "pause_plan.json")
    boundary_plan = read_json(output_dir / "final_boundary_plan.json")
    assert manifest["pause_policy"] == "cuts"
    assert manifest["pause_planner_backend"] == "deterministic_video_cuts"
    assert manifest["pause_planner_model"] is None
    assert pause_plan["pause_policy"] == "cuts"
    assert pause_plan["attempts"][0]["model_call_skipped"]
    assert boundary_plan["pause_policy"] == "cuts"
    assert all(join["target_pause_samples"] == 0 for join in boundary_plan["joins"])
    assert all(join["inserted_pause_samples"] == 0 for join in boundary_plan["joins"])
    assert all(
        segment["kind"] == "source" for segment in boundary_plan["output_segments"]
    )


def test_debug_flag_does_not_reintroduce_staged_production_wavs(
    tmp_path: Path,
) -> None:
    audio_path, plan_path, pause_plan_path, _ = _grounded_fixture(tmp_path)
    words = read_json(plan_path)["words"]
    completeness, mfa = _grounded_evidence(words=words, audio_path=audio_path)
    output_dir = tmp_path / "final-debug"

    manifest = render_final_cut(
        audio_path=audio_path,
        plan_path=plan_path,
        output_dir=output_dir,
        pause_plan_path=pause_plan_path,
        alignment_python=tmp_path / "model-must-not-run",
        pause_backend=ExplodingPauseBackend(),
        alignment_payload=completeness,
        mfa_payload=mfa,
        write_debug_artifacts=True,
    )

    assert manifest["debug_artifacts_requested"] is True
    assert manifest["debug_artifacts_written"] is True
    assert Path(manifest["breath_debug_manifest"]).is_file()
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
    words = read_json(plan_path)["words"]
    completeness, _ = _grounded_evidence(words=words, audio_path=audio_path)
    output_dir = tmp_path / "failed"

    with pytest.raises(
        FinalRenderError,
        match="mfa_alignment_failed",
    ):
        render_final_cut(
            audio_path=audio_path,
            plan_path=plan_path,
            output_dir=output_dir,
            pause_plan_path=pause_plan_path,
            alignment_python=tmp_path / "model-must-not-run",
            alignment_payload=completeness,
            mfa_payload={
                "schema_version": 1,
                "backend": "mfa",
                "mfa_version": MFA_VERSION,
                "model_id": MFA_MODEL_ID,
                "fine_tune": True,
                "contexts": [],
            },
        )

    assert not (output_dir / "final_cut.wav").exists()
    assert (output_dir / "pause_plan.json").is_file()
    boundary_plan = read_json(output_dir / "final_boundary_plan.json")
    assert boundary_plan["status"] == "unsafe"
    assert boundary_plan["alignment_failures"] == 3
    assert boundary_plan["alignment_backend"] == "mfa"


def test_final_render_preserves_local_source_context_after_retry_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path, plan_path, pause_plan_path, _ = _grounded_fixture(tmp_path)
    words = read_json(plan_path)["words"]
    initial_completeness, _ = _grounded_evidence(
        words=words,
        audio_path=audio_path,
    )
    fallback_completeness = _completeness_payload(
        words=words,
        contexts=[(range(3), None), (range(3), None)],
    )
    acoustic_intervals = {
        0: (0.10, 0.62),
        1: (0.72, 0.95),
        2: (1.20, 1.52),
    }
    fallback_mfa = _mfa_payload(
        words=words,
        selected_word_ids={0, 1, 2},
        source_audio_sha256=sha256_file(audio_path),
        contexts=[
            (
                "eof_tail",
                *_context_geometry(
                    words,
                    role_word_ids=(2,),
                    total_samples=2300,
                ),
                acoustic_intervals,
            ),
            (
                "internal_thought_gap_0000",
                *_context_geometry(
                    words,
                    role_word_ids=(1, 2),
                    total_samples=2300,
                ),
                acoustic_intervals,
            ),
        ],
    )
    invalid_mfa = {
        "schema_version": 1,
        "backend": "mfa",
        "mfa_version": MFA_VERSION,
        "model_id": MFA_MODEL_ID,
        "fine_tune": True,
        "sample_rate": SAMPLE_RATE,
        "source_audio_sha256": sha256_file(audio_path),
        "contexts": [],
    }
    render_calls = 0
    original_render = final_render_module.render_boundary_plan

    def count_render(**kwargs: Any) -> dict[str, Any]:
        nonlocal render_calls
        render_calls += 1
        return original_render(**kwargs)

    monkeypatch.setattr(final_render_module, "render_boundary_plan", count_render)
    output_dir = tmp_path / "conservative_delivery"

    with pytest.warns(RuntimeWarning, match="conservative source preservation"):
        manifest = render_final_cut(
            audio_path=audio_path,
            plan_path=plan_path,
            output_dir=output_dir,
            pause_plan_path=pause_plan_path,
            alignment_python=tmp_path / "model-must-not-run",
            alignment_payloads=[initial_completeness, fallback_completeness],
            mfa_payloads=[invalid_mfa, fallback_mfa],
            max_acoustic_retries=0,
        )

    assert render_calls == 1
    assert manifest["status"] == "complete"
    assert manifest["delivery_status"] == ("complete_with_preserved_source_context")
    assert Path(manifest["final_cut_wav"]).is_file()
    assert manifest["unresolved_boundaries"] == 0
    fallback = manifest["delivery_fallback"]
    assert fallback["preserved_intervals"]
    assert fallback["preserved_intervals"][0]["start_word_id"] == 1
    assert fallback["preserved_intervals"][0]["end_word_id"] == 2
    effective_plan = read_json(Path(manifest["effective_streaming_plan"]))
    assert "omitted" in effective_plan["committed"][0]["canonical_text"].lower()
    boundary_plan = read_json(Path(manifest["final_boundary_plan"]))
    assert boundary_plan["status"] == "safe"
    assert boundary_plan["delivery_status"] == (
        "complete_with_preserved_source_context"
    )
    assert len(boundary_plan["source_intervals"]) == 1


def test_final_render_accepts_dense_mfa_phone_boundary_without_silence(
    tmp_path: Path,
) -> None:
    audio_path, plan_path, pause_plan_path, _ = _grounded_fixture(
        tmp_path,
        dense_leading_boundary=True,
    )
    words = read_json(plan_path)["words"]
    completeness, mfa = _grounded_evidence(
        words=words,
        audio_path=audio_path,
        dense_leading_boundary=True,
    )
    output_dir = tmp_path / "dense_leading"

    manifest = render_final_cut(
        audio_path=audio_path,
        plan_path=plan_path,
        output_dir=output_dir,
        pause_plan_path=pause_plan_path,
        alignment_python=tmp_path / "model-must-not-run",
        alignment_payload=completeness,
        mfa_payload=mfa,
        max_acoustic_retries=0,
    )

    assert Path(manifest["final_cut_wav"]).is_file()
    boundary_plan = read_json(output_dir / "final_boundary_plan.json")
    assert boundary_plan["status"] == "safe"
    assert boundary_plan["unsafe_dense_boundaries"] == 0
    assert boundary_plan["mfa_dense_phone_boundaries"] == 1
    leading_boundary = next(
        boundary
        for boundary in boundary_plan["boundaries"]
        if boundary["boundary_kind"] == "omitted_to_selected"
    )
    assert leading_boundary["boundary_method"] == "mfa_dense_phone_boundary"
    assert leading_boundary["selected_source_sample"] == 1200
    assert leading_boundary["fade_intervals"] == []


def test_final_boundary_plan_cannot_change_during_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path, plan_path, pause_plan_path, _ = _grounded_fixture(tmp_path)
    words = read_json(plan_path)["words"]
    completeness, mfa = _grounded_evidence(words=words, audio_path=audio_path)
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
            alignment_payload=completeness,
            mfa_payload=mfa,
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
    repaired_completeness, repaired_mfa = _repaired_familiar_evidence(
        words,
        audio_path=audio_path,
    )

    manifest = render_final_cut(
        audio_path=audio_path,
        plan_path=plan_path,
        output_dir=output_dir,
        pause_plan_path=pause_path,
        alignment_python=tmp_path / "model-must-not-run",
        repair_backend=repair_backend,
        alignment_payloads=[
            _initial_familiar_completeness(words),
            repaired_completeness,
        ],
        mfa_payloads=[None, repaired_mfa],
        max_acoustic_retries=1,
    )

    assert render_calls == 1
    assert manifest["acoustic_repair_attempts"] == 1
    assert len(repair_backend.prompts) == 1

    rejected_path = Path(manifest["acoustic_repairs"][0]["rejected_boundary_plan"])
    rejected_plan = read_json(rejected_path)
    assert rejected_plan["mfa_alignment"] is None
    assert rejected_plan["mfa_error"] == "mfa_not_run_due_to_weak_retained_word"
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
    assert boundary_plan["alignment_backend"] == "mfa"
    assert boundary_plan["mfa_version"] == MFA_VERSION
    assert boundary_plan["mfa_dense_phone_boundaries"] == 1
    later_familiar = next(
        word
        for context in boundary_plan["alignment_contexts"]
        for word in context["words"]
        if word["source_word_ids"] == [8]
    )
    retained_phones = [
        phone for phone in later_familiar["phones"] if not phone["is_silence"]
    ]
    familiar_start = int(retained_phones[0]["start_sample"])
    familiar_end = int(retained_phones[-1]["end_sample"])
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
        if "completeness_contexts" not in path.parts
    )
    assert rendered_wavs == [Path(manifest["final_cut_wav"])]


def test_breath_cleanup_replaces_only_mfa_confirmed_non_speech_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path, plan_path, pause_path, words = _breath_cleanup_fixture(tmp_path)
    completeness, mfa = _breath_cleanup_alignment_evidence(
        words=words,
        audio_path=audio_path,
    )
    output_dir = tmp_path / "breath_cleanup_final"

    def reject_detector_adapter(**_: Any) -> dict[str, Any]:
        raise AssertionError("injected breath evidence must bypass the adapter")

    monkeypatch.setattr(
        final_render_module,
        "analyze_breath_evidence",
        reject_detector_adapter,
    )
    original_render = final_render_module.render_boundary_plan
    render_hashes: list[tuple[str, str]] = []

    def count_immutable_render(**kwargs: Any) -> dict[str, Any]:
        boundary_path = Path(kwargs["boundary_plan_path"])
        before = sha256_file(boundary_path)
        result = original_render(**kwargs)
        render_hashes.append((before, sha256_file(boundary_path)))
        return result

    monkeypatch.setattr(
        final_render_module,
        "render_boundary_plan",
        count_immutable_render,
    )

    manifest = render_final_cut(
        audio_path=audio_path,
        plan_path=plan_path,
        output_dir=output_dir,
        pause_plan_path=pause_path,
        alignment_python=tmp_path / "model-must-not-run",
        alignment_payload=completeness,
        mfa_payload=mfa,
        breath_cleanup="replace",
        breath_payload=_breath_detector_payload(),
        max_acoustic_retries=0,
    )

    assert manifest["breath_cleanup_status"] == "complete"
    assert manifest["breaths_detected"] == 2
    assert manifest["breaths_replaced"] == 1
    assert manifest["breaths_skipped_phone_overlap"] == 1
    assert render_hashes == [
        (
            manifest["final_boundary_plan_sha256"],
            manifest["final_boundary_plan_sha256"],
        )
    ]

    boundary_plan = read_json(Path(manifest["final_boundary_plan"]))
    cleanup = boundary_plan["breath_cleanup"]
    assert cleanup["status"] == "complete"
    assert len(cleanup["events"]) == 2
    false_s_event = next(
        event for event in cleanup["events"] if event["start_sample"] == 450
    )
    assert false_s_event["status"] == "breath_cleanup_skipped_phone_overlap"
    assert false_s_event["replacements"] == []
    assert false_s_event["protected_phone_intersections"]
    assert {
        interval["phone"] for interval in false_s_event["protected_phone_intersections"]
    } == {"S"}
    assert false_s_event["editable_intersection"] == [
        {
            "start_sample": 510,
            "end_sample": 990,
            "source_interval_index": 0,
            "verification": "mfa_confirmed_editable_non_speech",
            "intersection_start_sample": 510,
            "intersection_end_sample": 530,
        }
    ]

    real_breath = next(
        event for event in cleanup["events"] if event["start_sample"] == 650
    )
    assert real_breath["status"] == "breath_replaced_with_verified_room_tone"
    assert len(real_breath["replacements"]) == 1
    replacement = cleanup["replacements"][0]
    assert replacement["target_start_sample"] == 620
    assert replacement["target_end_sample"] == 810
    assert replacement["target_duration_samples"] == 190
    assert replacement["replacement_duration_samples"] == 190
    assert (
        sum(
            source_range["source_end_sample"] - source_range["source_start_sample"]
            for source_range in replacement["replacement_room_tone_source_ranges"]
        )
        == 190
    )

    protected_mask = boundary_plan["protected_speech_mask"]
    assert any(
        item["phone"] == "S"
        and item["phone_start_sample"] == 100
        and item["phone_end_sample"] == 500
        for item in protected_mask
    )
    for transition in replacement["transition_ranges"]:
        transition_start = int(transition["target_start_sample"])
        transition_end = int(transition["target_end_sample"])
        assert all(
            max(transition_start, int(protected["start_sample"]))
            >= min(transition_end, int(protected["end_sample"]))
            for protected in protected_mask
        )

    exclusions = cleanup["room_tone_exclusions"]
    assert any(
        exclusion["reason"] == "breath_event"
        and exclusion["start_sample"] == 650
        and exclusion["end_sample"] == 780
        for exclusion in exclusions
    )
    for source_range in replacement["replacement_room_tone_source_ranges"]:
        assert all(
            max(
                int(source_range["source_start_sample"]),
                int(exclusion["start_sample"]),
            )
            >= min(
                int(source_range["source_end_sample"]),
                int(exclusion["end_sample"]),
            )
            for exclusion in exclusions
        )
    assert {
        rejection["reason"] for rejection in cleanup["room_tone_candidate_rejections"]
    } >= {"breath_event", "breath_guard"}

    assert manifest["frame_count"] == boundary_plan["expected_output_frame_count"]
    assert manifest["frame_count"] == sum(
        int(segment["output_end_sample"]) - int(segment["output_start_sample"])
        for segment in boundary_plan["output_segments"]
    )
    source_segment = next(
        segment
        for segment in boundary_plan["output_segments"]
        if segment["kind"] == "source"
        and int(segment["source_start_sample"]) <= 450
        and int(segment["source_end_sample"]) >= 530
    )
    false_tail_output_start = int(source_segment["output_start_sample"]) + (
        450 - int(source_segment["source_start_sample"])
    )
    source_audio, _ = sf.read(audio_path, dtype="float32", always_2d=True)
    rendered_audio, _ = sf.read(
        Path(manifest["final_cut_wav"]),
        dtype="float32",
        always_2d=True,
    )
    assert np.array_equal(
        rendered_audio[false_tail_output_start : false_tail_output_start + 80],
        source_audio[450:530],
    )

    production_wavs = sorted(
        path
        for path in output_dir.rglob("*.wav")
        if "completeness_contexts" not in path.parts
    )
    assert production_wavs == [Path(manifest["final_cut_wav"])]


def test_breath_detector_failure_preserves_valid_boundary_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path, plan_path, pause_path, words = _breath_cleanup_fixture(tmp_path)
    completeness, mfa = _breath_cleanup_alignment_evidence(
        words=words,
        audio_path=audio_path,
    )
    output_dir = tmp_path / "breath_detector_failure"
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

    with pytest.warns(RuntimeWarning, match="breath cleanup was skipped"):
        manifest = render_final_cut(
            audio_path=audio_path,
            plan_path=plan_path,
            output_dir=output_dir,
            pause_plan_path=pause_path,
            alignment_python=tmp_path / "model-must-not-run",
            alignment_payload=completeness,
            mfa_payload=mfa,
            breath_cleanup="replace",
            breath_payload={"backend": "incompatible-detector"},
            max_acoustic_retries=0,
        )

    assert render_calls == 1
    assert manifest["status"] == "complete"
    assert manifest["breath_cleanup_status"] == (
        "breath_cleanup_skipped_detector_failure"
    )
    assert manifest["breaths_detected"] == 0
    assert manifest["breaths_replaced"] == 0

    boundary_plan = read_json(Path(manifest["final_boundary_plan"]))
    assert boundary_plan["status"] == "safe"
    assert boundary_plan["breath_cleanup"]["status"] == (
        "breath_cleanup_skipped_detector_failure"
    )
    assert boundary_plan["breath_cleanup"]["replacements"] == []
    assert boundary_plan["breath_cleanup"]["error"] is not None
    source_segment = next(
        segment
        for segment in boundary_plan["output_segments"]
        if segment["kind"] == "source"
        and int(segment["source_start_sample"]) <= 650
        and int(segment["source_end_sample"]) >= 780
    )
    breath_output_start = int(source_segment["output_start_sample"]) + (
        650 - int(source_segment["source_start_sample"])
    )
    source_audio, _ = sf.read(audio_path, dtype="float32", always_2d=True)
    rendered_audio, _ = sf.read(
        Path(manifest["final_cut_wav"]),
        dtype="float32",
        always_2d=True,
    )
    assert np.array_equal(
        rendered_audio[breath_output_start : breath_output_start + 130],
        source_audio[650:780],
    )
    production_wavs = sorted(
        path
        for path in output_dir.rglob("*.wav")
        if "completeness_contexts" not in path.parts
    )
    assert production_wavs == [Path(manifest["final_cut_wav"])]
