from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import voicecut.final_render as final_render_module
from voicecut.common import sha256_file, write_json
from voicecut.final_render import (
    FinalRenderError,
    _assert_boundary_plan_invariants,
    _resolve_mfa_cut,
    _resolve_mfa_eof_boundary,
    _snap_zero_crossing_in_mfa_gap,
    render_boundary_plan,
)
from voicecut.mfa_alignment import MFA_MODEL_ID, MFA_VERSION
from voicecut.rough_render import PlanWord

SAMPLE_RATE = 1000
TOTAL_SAMPLES = 1600


def _phone(
    label: str,
    start: float,
    end: float,
    *,
    silence: bool = False,
    start_sample: int | None = None,
    end_sample: int | None = None,
) -> dict[str, Any]:
    return {
        "phone": label,
        "start_seconds": start,
        "end_seconds": end,
        # The adapter intentionally preserves conservative floor/ceil samples.
        "start_sample": (
            math.floor(start * SAMPLE_RATE) if start_sample is None else start_sample
        ),
        "end_sample": (
            math.ceil(end * SAMPLE_RATE) if end_sample is None else end_sample
        ),
        "is_silence": silence,
    }


def _word(
    word_id: int,
    text: str,
    phones: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source_word_ids": [word_id],
        "source_text": text,
        "mfa_token": text.casefold(),
        "start_seconds": phones[0]["start_seconds"],
        "end_seconds": phones[-1]["end_seconds"],
        "start_sample": min(int(phone["start_sample"]) for phone in phones),
        "end_sample": max(int(phone["end_sample"]) for phone in phones),
        "phones": phones,
    }


def _context(
    *,
    context_id: str,
    words: list[dict[str, Any]],
    phones: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "context_id": context_id,
        "crop_source_start_seconds": 0.0,
        "crop_source_end_seconds": TOTAL_SAMPLES / SAMPLE_RATE,
        "crop_source_start_sample": 0,
        "crop_source_end_sample": TOTAL_SAMPLES,
        "ordered_source_word_ids": [
            int(word_id) for word in words for word_id in word["source_word_ids"]
        ],
        "original_source_words": [],
        "boundary_ids": [context_id],
        "words": words,
        "phones": phones,
        "mfa_output_json": f"mock://{context_id}.json",
    }


def _words(*texts: str) -> list[PlanWord]:
    return [
        PlanWord(
            id=index,
            text=text,
            # Deliberately inaccurate anchors prove that MFA owns coordinates.
            start=0.01 + index * 0.02,
            end=0.02 + index * 0.02,
        )
        for index, text in enumerate(texts)
    ]


def _supported() -> dict[str, Any]:
    return {
        "status": "supported_complete_word",
        "character_records": [],
        "minimum_edge_score": 0.95,
        "local_context_median_score": 0.95,
        "edge_to_context_score_ratio": 1.0,
    }


def _trailing_result(
    *,
    context: dict[str, Any] | None,
    words: list[PlanWord] | None = None,
    retained_support: dict[str, Any] | None = None,
    mfa_error: str | None = None,
) -> dict[str, Any]:
    return _resolve_mfa_cut(
        boundary_id="gap_0_left",
        boundary_kind="selected_to_omitted",
        spec={
            "event_key": "gap_0_left",
            "role_word_ids": {
                "last_retained_left": 0,
                "first_omitted": 1,
            },
        },
        mfa_context=context,
        mfa_error=mfa_error,
        retained_support=retained_support or _supported(),
        completeness_error=None,
        words=words or _words("kept", "removed"),
        mono=np.zeros(TOTAL_SAMPLES, dtype=np.float32),
        sample_rate=SAMPLE_RATE,
        total_samples=TOTAL_SAMPLES,
        retained_role="last_retained_left",
        omitted_role="first_omitted",
        direction="trailing",
    )


def test_mfa_silence_phone_resolves_cut_and_silence_local_fade() -> None:
    kept_phone = _phone("T", 0.100, 0.200)
    silence_phone = _phone("sil", 0.200, 0.260, silence=True)
    removed_phone = _phone("R", 0.260, 0.360)
    result = _trailing_result(
        context=_context(
            context_id="gap_0_left",
            words=[
                _word(0, "kept", [kept_phone]),
                _word(1, "removed", [removed_phone]),
            ],
            phones=[kept_phone, silence_phone, removed_phone],
        )
    )

    assert result["safety_status"] == "safe"
    assert result["boundary_method"] == "mfa_verified_silence"
    assert 200 <= result["selected_source_sample"] <= 260
    assert result["verified_silence_interval"]["silence_phone_intervals"] == [
        silence_phone
    ]
    assert result["fade_intervals"]
    assert all(
        int(fade["source_start_sample"]) >= 200 for fade in result["fade_intervals"]
    )


def test_mfa_dense_phone_to_phone_cut_needs_no_silence_or_fade() -> None:
    kept_phone = _phone("N", 0.100, 0.200)
    removed_phone = _phone("W", 0.200, 0.300)
    result = _trailing_result(
        context=_context(
            context_id="gap_0_left",
            words=[
                _word(0, "begin", [kept_phone]),
                _word(1, "with", [removed_phone]),
            ],
            phones=[kept_phone, removed_phone],
        ),
        words=_words("begin", "with"),
    )

    assert result["safety_status"] == "safe"
    assert result["boundary_method"] == "mfa_dense_phone_boundary"
    assert result["selected_source_sample"] == 200
    assert result["verified_silence_interval"] is None
    assert result["fade_intervals"] == []


def test_one_sample_floor_ceil_overlap_is_accepted() -> None:
    # Identical continuous-time edge, but conservative floor/ceil coordinates
    # overlap by one discrete sample.
    kept_phone = _phone(
        "T",
        0.1000,
        0.2004,
        start_sample=100,
        end_sample=201,
    )
    removed_phone = _phone(
        "R",
        0.2004,
        0.3004,
        start_sample=200,
        end_sample=301,
    )
    result = _trailing_result(
        context=_context(
            context_id="gap_0_left",
            words=[
                _word(0, "kept", [kept_phone]),
                _word(1, "removed", [removed_phone]),
            ],
            phones=[kept_phone, removed_phone],
        )
    )

    assert result["safety_status"] == "safe"
    assert result["boundary_method"] == "mfa_dense_phone_boundary"
    assert result["selected_source_sample"] == 201


def test_true_time_overlap_is_rejected_even_if_sample_overlap_is_small() -> None:
    kept_phone = _phone(
        "T",
        0.1000,
        0.2004,
        start_sample=100,
        end_sample=201,
    )
    removed_phone = _phone(
        "R",
        0.2003,
        0.3004,
        start_sample=200,
        end_sample=301,
    )
    result = _trailing_result(
        context=_context(
            context_id="gap_0_left",
            words=[
                _word(0, "kept", [kept_phone]),
                _word(1, "removed", [removed_phone]),
            ],
            phones=[kept_phone, removed_phone],
        )
    )

    assert result["safety_status"] == "mfa_word_mapping_failed"
    assert result["selected_source_sample"] is None


@pytest.mark.parametrize(
    ("context", "mfa_error"),
    [
        (None, "MFA subprocess failed"),
        (
            _context(
                context_id="gap_0_left",
                words=[],
                phones=[],
            ),
            None,
        ),
    ],
)
def test_mfa_failure_never_falls_back_to_whisper_anchor(
    context: dict[str, Any] | None,
    mfa_error: str | None,
) -> None:
    result = _trailing_result(context=context, mfa_error=mfa_error)

    assert result["whisper_anchors"]["retained_end_seconds"] == pytest.approx(0.02)
    assert result["safety_status"] in {
        "mfa_alignment_failed",
        "mfa_word_mapping_failed",
    }
    assert result["selected_source_sample"] is None
    assert result["selected_source_seconds"] is None


def test_zero_crossing_snap_stays_inside_mfa_gap_and_within_two_ms() -> None:
    mono = np.ones(400, dtype=np.float32)
    # A crossing just before the MFA-confirmed gap must be ignored.
    mono[198] = -1.0
    mono[199] = 1.0
    # A crossing within the two-millisecond search radius may be selected.
    mono[211] = -1.0
    mono[212] = 1.0
    # A farther crossing must not expand the search.
    mono[214] = -1.0
    mono[215] = 1.0

    selected = _snap_zero_crossing_in_mfa_gap(
        mono,
        candidate=210,
        interval_start=200,
        interval_end=240,
        sample_rate=SAMPLE_RATE,
    )

    assert 200 <= selected <= 240
    assert abs(selected - 210) <= 2
    assert selected != 199
    assert selected != 215


def _minimal_boundary_plan(*, fade_start: int, fade_end: int) -> dict[str, Any]:
    return {
        "alignment_backend": "mfa",
        "mfa_version": MFA_VERSION,
        "mfa_model": MFA_MODEL_ID,
        "mfa_fine_tune": True,
        "boundaries": [
            {
                "boundary_id": "cut",
                "safety_status": "safe",
                "selected_source_sample": 200,
                "protected_speech_intervals": [
                    {
                        "role": "last_retained_left",
                        "word_id": 0,
                        "start_sample": 100,
                        "end_sample": 200,
                    }
                ],
                "fade_intervals": [
                    {
                        "source_start_sample": fade_start,
                        "source_end_sample": fade_end,
                    }
                ],
            }
        ],
        "joins": [],
        "output_segments": [
            {
                "segment_index": 0,
                "kind": "source",
                "source_start_sample": 100,
                "source_end_sample": 300,
                "output_start_sample": 0,
                "output_end_sample": 200,
            }
        ],
        "expected_output_frame_count": 200,
    }


def test_boundary_plan_rejects_fade_over_retained_mfa_phone() -> None:
    with pytest.raises(FinalRenderError, match="fade overlaps retained speech"):
        _assert_boundary_plan_invariants(
            _minimal_boundary_plan(fade_start=195, fade_end=205)
        )

    _assert_boundary_plan_invariants(
        _minimal_boundary_plan(fade_start=200, fade_end=205)
    )


def test_eof_protects_complete_final_phone_and_delays_fade() -> None:
    final_phone = _phone("S", 0.500, 0.800)
    context = _context(
        context_id="eof_tail",
        words=[_word(0, "thanks", [final_phone])],
        phones=[final_phone],
    )
    mono = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
    time = np.arange(300, dtype=np.float32) / SAMPLE_RATE
    mono[500:800] = 0.2 * np.sin(2.0 * np.pi * 70.0 * time)
    result = _resolve_mfa_eof_boundary(
        spec={
            "event_key": "eof_tail",
            "role_word_ids": {"final_retained": 0},
        },
        mfa_context=context,
        mfa_error=None,
        retained_support=_supported(),
        completeness_error=None,
        words=_words("thanks"),
        mono=mono,
        sample_rate=SAMPLE_RATE,
        total_samples=TOTAL_SAMPLES,
    )

    assert result["safety_status"] == "safe"
    assert result["final_word_text"] == "thanks"
    assert result["final_phone"] == "S"
    assert result["mfa_phone_end"] == pytest.approx(0.8)
    assert result["selected_source_sample"] >= 800
    assert result["retained_tail_end"] >= 0.8
    assert all(
        int(fade["source_start_sample"]) >= 800 for fade in result["fade_intervals"]
    )


def test_bad_first_familiar_remains_a_forbidden_word_occurrence() -> None:
    words = _words(*(f"word-{index}" for index in range(34)))
    words[32] = PlanWord(id=32, text="familiar", start=1.0, end=1.4)
    weak_support = {
        "status": "weak_terminal_word_support",
        "character_records": [
            {"character": "i", "score": 0.08},
            {"character": "a", "score": 0.05},
            {"character": "r", "score": 0.03},
        ],
        "minimum_edge_score": 0.03,
        "local_context_median_score": 0.91,
        "edge_to_context_score_ratio": 0.033,
    }
    result = _resolve_mfa_cut(
        boundary_id="familiar_cut",
        boundary_kind="selected_to_omitted",
        spec={
            "event_key": "familiar_cut",
            "role_word_ids": {
                "last_retained_left": 32,
                "first_omitted": 33,
            },
        },
        # Even apparently valid MFA geometry may not override completeness.
        mfa_context=_context(
            context_id="familiar_cut",
            words=[
                _word(32, "familiar", [_phone("R", 1.0, 1.4)]),
                _word(33, "words", [_phone("W", 1.4, 1.7)]),
            ],
            phones=[_phone("R", 1.0, 1.4), _phone("W", 1.4, 1.7)],
        ),
        mfa_error=None,
        retained_support=weak_support,
        completeness_error=None,
        words=words,
        mono=np.zeros(2000, dtype=np.float32),
        sample_rate=SAMPLE_RATE,
        total_samples=2000,
        retained_role="last_retained_left",
        omitted_role="first_omitted",
        direction="trailing",
    )

    assert result["safety_status"] == "weak_retained_word_alignment"
    assert result["failure_reason"] == "weak_terminal_word_support"
    assert result["forbidden_word_ids"] == [32]
    assert result["selected_source_sample"] is None


def test_authoritative_plan_renders_canonical_source_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "canonical.wav"
    output_path = tmp_path / "final.wav"
    source = np.linspace(-0.2, 0.2, 300, dtype=np.float32)[:, None]
    sf.write(source_path, source, SAMPLE_RATE, subtype="FLOAT")
    plan_path = tmp_path / "final_boundary_plan.json"
    plan = {
        "schema_version": 2,
        "planner": "authoritative_single_pass_boundary_plan_v2",
        "status": "safe",
        "alignment_backend": "mfa",
        "mfa_version": MFA_VERSION,
        "mfa_model": MFA_MODEL_ID,
        "mfa_fine_tune": True,
        "source_audio_sha256": sha256_file(source_path),
        "source_sample_rate": SAMPLE_RATE,
        "source_channel_count": 1,
        "source_frame_count": 300,
        "boundaries": [],
        "joins": [],
        "output_segments": [
            {
                "segment_index": 0,
                "kind": "source",
                "source_start_sample": 50,
                "source_end_sample": 250,
                "output_start_sample": 0,
                "output_end_sample": 200,
                "gain_envelopes": [],
            }
        ],
        "expected_output_frame_count": 200,
    }
    write_json(plan_path, plan)

    original_read = final_render_module.sf.read
    original_write = final_render_module.sf.write
    canonical_reads = 0
    output_writes = 0

    def counted_read(path: str | Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal canonical_reads
        if Path(path) == source_path:
            canonical_reads += 1
        return original_read(path, *args, **kwargs)

    def counted_write(path: str | Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal output_writes
        if Path(path) == output_path:
            output_writes += 1
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(final_render_module.sf, "read", counted_read)
    monkeypatch.setattr(final_render_module.sf, "write", counted_write)
    render_boundary_plan(
        audio_path=source_path,
        boundary_plan_path=plan_path,
        output_path=output_path,
    )

    rendered, _ = original_read(output_path, dtype="float32", always_2d=True)
    canonical, _ = original_read(source_path, dtype="float32", always_2d=True)
    assert canonical_reads == 1
    assert output_writes == 1
    assert np.array_equal(rendered, canonical[50:250])
    assert list(tmp_path.glob("*.wav")) == [source_path, output_path]
