from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from voicecut.common import read_json, sha256_file, write_json
from voicecut.leading_align import (
    decide_leading_boundary,
    find_leading_waveform_candidate,
    leading_alignment_positions,
    render_leading_aligned_preview,
)


SAMPLE_RATE = 1000


def _mock_aligned_job(
    words: list[str],
    times: list[tuple[float, float]],
    *,
    clip_index: int = 0,
) -> dict[str, Any]:
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


def _decision_job() -> dict[str, Any]:
    return {
        "clip_index": 0,
        "local_words": [
            {"id": 0, "text": "same"},
            {"id": 1, "text": "same"},
            {"id": 2, "text": "omitted"},
            {"id": 3, "text": "same"},
            {"id": 4, "text": "same"},
        ],
        "omitted_local_index": 2,
        "kept_local_index": 3,
        "crop_start_seconds": 0.0,
        "crop_duration_seconds": 1.5,
        "previous_source_start_sample": 610,
        "previous_fade_in_samples": 5,
    }


def _hard_manifest_fixture(tmp_path: Path) -> Path:
    source = np.full((1000, 1), 0.01, dtype=np.float32)
    source[100:300, 0] = 0.18
    source[350:550, 0] = -0.2
    source[600:800, 0] = 0.16
    source[320, 0] = 0.0
    audio_path = tmp_path / "source.wav"
    sf.write(audio_path, source, SAMPLE_RATE, subtype="FLOAT")

    plan_path = tmp_path / "streaming_plan.json"
    write_json(
        plan_path,
        {
            "schema_version": 1,
            "status": "complete",
            "words": [
                {
                    "id": 0,
                    "text": "discarded",
                    "start": 0.10,
                    "end": 0.28,
                },
                {
                    "id": 1,
                    "text": "example",
                    "start": 0.35,
                    "end": 0.55,
                },
                {
                    "id": 2,
                    "text": "wrong",
                    "start": 0.60,
                    "end": 0.80,
                },
            ],
            "committed": [
                {
                    "canonical_text": "Example.",
                    "source_ranges": [{"start_word_id": 1, "end_word_id": 2}],
                }
            ],
            "selected_source_ranges": [{"start_word_id": 1, "end_word_id": 2}],
        },
    )

    previous_start = 300
    previous_end = 580
    old_clip = source[previous_start:previous_end].copy()
    old_clip[:5] *= np.linspace(0.0, 1.0, 5, endpoint=True, dtype=np.float32)[:, None]
    old_clip[-5:] *= np.linspace(1.0, 0.0, 5, endpoint=True, dtype=np.float32)[:, None]
    previous_wav = tmp_path / "hard_boundary_aligned.wav"
    sf.write(previous_wav, old_clip, SAMPLE_RATE, subtype="FLOAT")

    manifest_path = tmp_path / "render_manifest_forced_aligned.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "renderer": "streaming_plan_hard_boundary_alignment_v1",
            "source_audio": str(audio_path),
            "source_audio_sha256": sha256_file(audio_path),
            "streaming_plan": str(plan_path),
            "streaming_plan_sha256": sha256_file(plan_path),
            "source_sample_rate": SAMPLE_RATE,
            "source_channel_count": 1,
            "source_frame_count": len(source),
            "configuration": {
                "edge_padding_ms": 30.0,
                "clip_fade_ms": 5.0,
                "inter_clip_silence_ms": 80.0,
            },
            "hard_boundary_aligned_wav": str(previous_wav),
            "hard_boundary_aligned_wav_sha256": sha256_file(previous_wav),
            "hard_boundary_aligned_duration_seconds": len(old_clip) / SAMPLE_RATE,
            "hard_boundary_aligned_expected_output_frame_count": len(old_clip),
            "clips": [
                {
                    "clip_index": 0,
                    "source_word_start": 1,
                    "source_word_end": 2,
                    "boundary_method": "forced_alignment",
                    "final_cut_seconds": previous_end / SAMPLE_RATE,
                    "forced_aligned_kept_end_seconds": 0.55,
                    "forced_aligned_omitted_start_seconds": 0.60,
                    "final_source_start_sample": previous_start,
                    "final_source_end_sample": previous_end,
                    "final_fade_in_samples": 5,
                    "final_fade_out_samples": 5,
                    "final_output_start_sample": 0,
                    "final_output_end_sample": len(old_clip),
                    "final_output_start_seconds": 0.0,
                    "final_output_end_seconds": len(old_clip) / SAMPLE_RATE,
                    "final_frame_count": len(old_clip),
                }
            ],
        },
    )
    return manifest_path


def test_duplicate_words_map_by_sequential_position() -> None:
    worker = _mock_aligned_job(
        ["same", "same", "omitted", "same", "same"],
        [
            (0.10, 0.20),
            (0.25, 0.35),
            (0.40, 0.62),
            (0.65, 0.80),
            (0.85, 1.00),
        ],
    )

    omitted_end, kept_start, granularity = leading_alignment_positions(
        job=_decision_job(),
        worker_job=worker,
    )

    assert granularity == "characters"
    assert omitted_end == 0.62
    assert kept_start == 0.65


def test_leading_start_retains_approximately_30ms_quiet() -> None:
    mono = np.full(1500, 0.2, dtype=np.float32)
    mono[620] = 0.0
    worker = _mock_aligned_job(
        ["same", "same", "omitted", "same", "same"],
        [
            (0.10, 0.20),
            (0.25, 0.35),
            (0.40, 0.60),
            (0.65, 0.80),
            (0.85, 1.00),
        ],
    )

    decision = decide_leading_boundary(
        job=_decision_job(),
        worker_job=worker,
        mono=mono,
        sample_rate=SAMPLE_RATE,
    )

    assert decision.status == "leading_forced_alignment"
    assert decision.omitted_end_seconds == 0.60
    assert decision.kept_start_seconds == 0.65
    assert decision.start_seconds == 0.62
    assert decision.retained_leading_quiet_ms == 30.0
    assert decision.dense_boundary is False
    assert decision.fade_in_samples == 5


def test_dense_boundary_never_uses_more_than_2ms_fade() -> None:
    mono = np.full(1500, 0.2, dtype=np.float32)
    mono[620] = 0.0
    worker = _mock_aligned_job(
        ["same", "same", "omitted", "same", "same"],
        [
            (0.10, 0.20),
            (0.25, 0.35),
            (0.40, 0.62),
            (0.63, 0.80),
            (0.85, 1.00),
        ],
    )

    decision = decide_leading_boundary(
        job=_decision_job(),
        worker_job=worker,
        mono=mono,
        sample_rate=SAMPLE_RATE,
    )

    assert decision.status == "leading_forced_alignment"
    assert 0.62 <= decision.start_seconds <= 0.63
    assert decision.start_seconds <= decision.kept_start_seconds
    assert decision.dense_boundary is True
    assert decision.fade_in_samples <= 2


def test_failed_alignment_preserves_previous_start_and_fade() -> None:
    decision = decide_leading_boundary(
        job=_decision_job(),
        worker_job={
            "clip_index": 0,
            "error": "first kept word did not align",
            "aligned": None,
        },
        mono=np.full(1500, 0.2, dtype=np.float32),
        sample_rate=SAMPLE_RATE,
    )

    assert decision.status == "leading_forced_alignment_failed"
    assert decision.start_seconds == 0.61
    assert decision.fade_in_samples == 5
    assert "did not align" in (decision.error or "")


def test_clear_waveform_gap_overrides_gross_zero_score_alignment() -> None:
    mono = np.full(1500, 0.005, dtype=np.float32)
    mono[100:250] = 0.2
    mono[700:850] = -0.2
    mono[665] = 0.0
    job = {
        **_decision_job(),
        "crop_start_sample": 0,
        "crop_end_sample": 1500,
        "first_kept_word_start_seconds": 0.32,
        "first_kept_word_end_seconds": 0.85,
        "previous_source_start_sample": 300,
        "previous_fade_in_samples": 5,
    }
    worker = _mock_aligned_job(
        ["same", "same", "omitted", "same", "same"],
        [
            (0.10, 0.20),
            (0.25, 0.35),
            (0.40, 0.60),
            (1.00, 1.20),
            (1.25, 1.40),
        ],
    )
    worker["aligned"]["word_segments"][3]["score"] = 0.0
    worker["aligned"]["segments"][0]["chars"][6]["score"] = 0.0

    waveform = find_leading_waveform_candidate(
        mono,
        job=job,
        sample_rate=SAMPLE_RATE,
    )
    decision = decide_leading_boundary(
        job=job,
        worker_job=worker,
        mono=mono,
        sample_rate=SAMPLE_RATE,
    )

    assert waveform is not None
    assert decision.status == "leading_waveform_silence"
    assert decision.start_seconds == 0.665
    assert decision.waveform_active_onset_seconds == 0.695
    assert decision.retained_leading_quiet_ms == 30.0
    # A waveform-safe silence trim is allowed to exceed the alignment shift
    # guard because it removes only a verified quiet run.
    assert decision.shift_ms == 365.0
    assert decision.omitted_end_seconds is None
    assert decision.kept_start_seconds is None


def test_zero_confidence_alignment_fails_when_no_waveform_gap_exists() -> None:
    worker = _mock_aligned_job(
        ["same", "same", "omitted", "same", "same"],
        [
            (0.10, 0.20),
            (0.25, 0.35),
            (0.40, 0.62),
            (0.63, 0.80),
            (0.85, 1.00),
        ],
    )
    worker["aligned"]["word_segments"][3]["score"] = 0.0
    worker["aligned"]["segments"][0]["chars"][6]["score"] = 0.0

    decision = decide_leading_boundary(
        job=_decision_job(),
        worker_job=worker,
        mono=np.full(1500, 0.2, dtype=np.float32),
        sample_rate=SAMPLE_RATE,
    )

    assert decision.status == "leading_forced_alignment_failed"
    assert decision.start_seconds == 0.61
    assert "positive confidence" in (decision.error or "")


def test_render_aligns_first_clip_start_and_preserves_endpoint(
    tmp_path: Path,
) -> None:
    manifest_path = _hard_manifest_fixture(tmp_path)
    worker = _mock_aligned_job(
        ["discarded", "example", "wrong"],
        [(0.10, 0.30), (0.35, 0.55), (0.60, 0.80)],
    )

    manifest = render_leading_aligned_preview(
        aligned_manifest_path=manifest_path,
        output_dir=tmp_path / "leading",
        alignment_python=tmp_path / "unused-python",
        alignment_payload={"jobs": [worker]},
    )

    clip = manifest["clips"][0]
    assert clip["leading_alignment_status"] == "leading_forced_alignment"
    assert clip["leading_previous_omitted_word"] == "discarded"
    assert clip["leading_first_kept_word"] == "example"
    assert clip["final_source_start_sample"] == 320
    assert clip["leading_retained_quiet_ms"] == 30.0
    assert clip["final_source_end_sample"] == 580
    assert clip["final_cut_seconds"] == 0.58
    assert clip["boundary_method"] == "forced_alignment"
    assert clip["final_fade_out_samples"] == 5

    assert manifest["renderer"] == ("streaming_plan_full_boundary_alignment_v1")
    assert manifest["leading_boundaries_found"] == 1
    assert manifest["leading_boundaries_sent_to_whisperx"] == 1
    assert manifest["leading_boundaries_successfully_aligned"] == 1
    assert manifest["leading_alignment_failures"] == 0
    assert manifest["all_leading_boundaries_resolved"] is True
    full_path = Path(manifest["full_boundary_aligned_wav"])
    assert manifest["full_boundary_aligned_wav_sha256"] == sha256_file(full_path)
    assert manifest["full_boundary_aligned_expected_output_frame_count"] == 260
    assert sf.info(full_path).frames == 260

    debug = tmp_path / "leading" / "leading_alignment_debug" / "clip_000_start"
    for filename in (
        "context.wav",
        "old_clip.wav",
        "new_clip.wav",
        "alignment.json",
        "alignment_plot.png",
    ):
        assert (debug / filename).is_file(), filename
    saved = read_json(
        tmp_path / "leading" / "render_manifest_full_boundary_aligned.json"
    )
    assert saved["clips"][0]["final_source_end_sample"] == 580
    alignment_jobs = read_json(tmp_path / "leading" / "leading_alignment_jobs.json")
    assert [job["clip_index"] for job in alignment_jobs["jobs"]] == [0]


def test_render_failure_is_explicit_and_preserves_previous_audio(
    tmp_path: Path,
) -> None:
    manifest_path = _hard_manifest_fixture(tmp_path)
    previous_manifest = read_json(manifest_path)
    old_path = Path(previous_manifest["hard_boundary_aligned_wav"])
    old_audio, _ = sf.read(old_path, dtype="float32", always_2d=True)

    manifest = render_leading_aligned_preview(
        aligned_manifest_path=manifest_path,
        output_dir=tmp_path / "leading_failed",
        alignment_python=tmp_path / "unused-python",
        alignment_payload={
            "jobs": [
                {
                    "clip_index": 0,
                    "error": "alignment unavailable",
                    "aligned": None,
                }
            ]
        },
    )

    clip = manifest["clips"][0]
    assert clip["leading_alignment_status"] == "leading_forced_alignment_failed"
    assert clip["final_source_start_sample"] == 300
    assert clip["final_source_end_sample"] == 580
    assert manifest["leading_alignment_failures"] == 1
    assert manifest["all_leading_boundaries_resolved"] is False
    assert manifest["unresolved_leading_boundaries"] == [
        {
            "clip_index": 0,
            "previous_omitted_word": "discarded",
            "first_kept_word": "example",
            "error": (
                "waveform: no stable quiet-to-sustained-speech transition "
                "found; forced alignment: ValueError: alignment unavailable"
            ),
        }
    ]
    new_audio, _ = sf.read(
        manifest["full_boundary_aligned_wav"],
        dtype="float32",
        always_2d=True,
    )
    assert np.array_equal(new_audio, old_audio)


def test_dense_fade_configuration_cannot_exceed_2ms(
    tmp_path: Path,
) -> None:
    manifest_path = _hard_manifest_fixture(tmp_path)
    with pytest.raises(ValueError, match="must not exceed 2 ms"):
        render_leading_aligned_preview(
            aligned_manifest_path=manifest_path,
            output_dir=tmp_path / "invalid",
            dense_boundary_fade_ms=2.1,
            alignment_payload={"jobs": []},
        )
