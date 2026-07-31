from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicecut.common import sha256_file, write_json
from voicecut.media import prepare_media_input, probe_media
from voicecut.video_render import (
    VisualTimelineSegment,
    VideoRenderError,
    _filter_graph,
    build_visual_timeline,
    load_visual_timeline,
    render_edited_video,
)


def _semantic_manifest() -> dict[str, object]:
    return {
        "source_sample_rate": 1000,
        "semantic_pause_output_frame_count": 2000,
        "clips": [
            {
                "clip_index": 0,
                "final_source_start_sample": 1000,
                "semantic_source_end_sample": 2000,
                "semantic_output_start_sample": 0,
                "semantic_output_end_sample": 1200,
            },
            {
                "clip_index": 1,
                "final_source_start_sample": 3000,
                "semantic_source_end_sample": 3500,
                "semantic_output_start_sample": 1500,
                "semantic_output_end_sample": 2000,
            },
        ],
        "transitions": [
            {
                "boundary_location": "inside_continuous_source_clip",
                "previous_clip_index": 0,
                "next_clip_index": 0,
                "source_insertion_seconds": 1.5,
                "inserted_pause_ms": 200.0,
                "status": "pause_inserted",
                "output_pause_start_seconds": 0.5,
                "output_pause_end_seconds": 0.7,
            }
        ],
        "clip_joins": [
            {
                "join_index": 0,
                "left_clip_index": 0,
                "right_clip_index": 1,
                "inserted_pause_ms": 300.0,
                "output_pause_start_seconds": 1.2,
                "output_pause_end_seconds": 1.5,
            }
        ],
    }


def test_visual_timeline_preserves_motion_and_freezes_only_inserted_time() -> None:
    timeline, sample_rate, expected_frames = build_visual_timeline(_semantic_manifest())

    assert sample_rate == 1000
    assert expected_frames == 2000
    assert [
        (
            item.clip_index,
            item.source_start_sample,
            item.source_end_sample,
            item.freeze_after_samples,
            item.freeze_reason,
            item.output_start_sample,
            item.output_end_sample,
        )
        for item in timeline
    ] == [
        (0, 1000, 1500, 200, "internal_semantic_pause", 0, 700),
        (0, 1500, 2000, 300, "inter_clip_semantic_pause", 700, 1500),
        (1, 3000, 3500, 0, None, 1500, 2000),
    ]
    assert (
        sum(
            item.source_end_sample
            - item.source_start_sample
            + item.freeze_after_samples
            for item in timeline
        )
        == expected_frames
    )


def test_visual_timeline_rejects_pause_that_disagrees_with_audio_timeline() -> None:
    manifest = _semantic_manifest()
    manifest["clip_joins"][0]["output_pause_start_seconds"] = 1.1  # type: ignore[index]

    with pytest.raises(VideoRenderError, match="does not begin"):
        build_visual_timeline(manifest)


def test_final_manifest_seals_semantic_timeline_geometry(tmp_path: Path) -> None:
    semantic_path = tmp_path / "semantic.json"
    final_path = tmp_path / "final.json"
    write_json(semantic_path, _semantic_manifest())
    write_json(
        final_path,
        {
            "status": "complete",
            "semantic_pause_manifest": str(semantic_path.resolve()),
            "semantic_pause_manifest_sha256": sha256_file(semantic_path),
        },
    )
    changed = _semantic_manifest()
    changed["semantic_pause_output_frame_count"] = 2100
    write_json(semantic_path, changed)

    with pytest.raises(VideoRenderError, match="changed after final rendering"):
        load_visual_timeline(final_path)


def test_video_timeline_rejects_unsealed_semantic_manifest(
    tmp_path: Path,
) -> None:
    semantic_path = tmp_path / "semantic.json"
    final_path = tmp_path / "final.json"
    write_json(semantic_path, _semantic_manifest())
    write_json(
        final_path,
        {
            "status": "complete",
            "semantic_pause_manifest": str(semantic_path.resolve()),
        },
    )

    with pytest.raises(VideoRenderError, match="does not seal"):
        load_visual_timeline(final_path)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is not installed",
)
def test_off_frame_many_cut_render_keeps_late_visual_segments(
    tmp_path: Path,
) -> None:
    """Every span stays anchored even when cuts fall between video frames."""

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    source = tmp_path / "source.mkv"
    output = tmp_path / "output.mkv"
    filter_script = tmp_path / "timeline.txt"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=10:duration=5",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
    )
    timeline = [
        VisualTimelineSegment(
            clip_index=index,
            source_start_sample=index * 300,
            source_end_sample=index * 300 + 150,
            freeze_after_samples=0,
            freeze_reason=None,
            output_start_sample=index * 150,
            output_source_end_sample=(index + 1) * 150,
            output_end_sample=(index + 1) * 150,
        )
        for index in range(10)
    ]
    filter_script.write_text(
        _filter_graph(
            timeline,
            sample_rate=1000,
            video_stream_index=0,
            stream_offset_seconds=0.0,
            expected_duration_seconds=1.5,
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter_complex_script",
            str(filter_script),
            "-map",
            "[vout]",
            "-c:v",
            "ffv1",
            "-fps_mode",
            "vfr",
            str(output),
        ],
        check=True,
    )

    def frame_hashes(path: Path) -> list[str]:
        result = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "framemd5",
                "-",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return [
            line.rsplit(",", 1)[-1].strip()
            for line in result.stdout.splitlines()
            if line and not line.startswith("#")
        ]

    source_hashes = frame_hashes(source)
    output_hashes = frame_hashes(output)
    used_source_frames = [source_hashes.index(value) for value in output_hashes]
    assert 24 in used_source_frames
    assert 27 in used_source_frames
    assert used_source_frames[-1] >= 27


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is not installed",
)
def test_video_render_muxes_final_audio_and_holds_frame_for_internal_pause(
    tmp_path: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    source_video = tmp_path / "source.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=20:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=3",
            "-shortest",
            "-vf",
            "setsar=2/1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source_video),
        ],
        check=True,
    )
    media_input = prepare_media_input(source_video, tmp_path / "media")
    canonical = Path(media_input["canonical_audio"])
    sample_rate = sf.info(canonical).samplerate

    final_audio = tmp_path / "final.wav"
    sf.write(
        final_audio,
        np.zeros((round(1.5 * sample_rate), 1), dtype=np.float32),
        sample_rate,
        subtype="FLOAT",
    )
    semantic_path = tmp_path / "semantic.json"
    source_start = round(0.5 * sample_rate)
    insertion = round(1.0 * sample_rate)
    source_end = round(1.5 * sample_rate)
    expected_frames = round(1.5 * sample_rate)
    write_json(
        semantic_path,
        {
            "source_sample_rate": sample_rate,
            "semantic_pause_output_frame_count": expected_frames,
            "clips": [
                {
                    "clip_index": 0,
                    "final_source_start_sample": source_start,
                    "semantic_source_end_sample": source_end,
                    "semantic_output_start_sample": 0,
                    "semantic_output_end_sample": expected_frames,
                }
            ],
            "transitions": [
                {
                    "boundary_location": "inside_continuous_source_clip",
                    "previous_clip_index": 0,
                    "next_clip_index": 0,
                    "source_insertion_seconds": insertion / sample_rate,
                    "inserted_pause_ms": 500.0,
                    "status": "pause_inserted",
                    "output_pause_start_seconds": 0.5,
                    "output_pause_end_seconds": 1.0,
                }
            ],
            "clip_joins": [],
        },
    )
    final_manifest_path = tmp_path / "final_manifest.json"
    write_json(
        final_manifest_path,
        {
            "status": "complete",
            "source_audio": str(canonical.resolve()),
            "source_audio_sha256": media_input["canonical_audio_sha256"],
            "semantic_pause_manifest": str(semantic_path.resolve()),
            "semantic_pause_manifest_sha256": sha256_file(semantic_path),
            "final_cut_wav": str(final_audio.resolve()),
            "final_cut_wav_sha256": sha256_file(final_audio),
        },
    )

    output = tmp_path / "edited.mp4"
    manifest_path = tmp_path / "work/publication/video_render_manifest.json"
    manifest = render_edited_video(
        source_video=source_video,
        media_input_manifest_path=tmp_path / "media/media_input.json",
        final_render_manifest_path=final_manifest_path,
        output_path=output,
        manifest_path=manifest_path,
    )

    result = probe_media(output)
    assert output.is_file()
    assert manifest_path.is_file()
    assert not (tmp_path / "edited.video_render_manifest.json").exists()
    assert result.video_stream is not None
    assert result.duration_seconds == pytest.approx(1.5, abs=0.12)
    assert manifest["semantic_frame_holds"] == 1
    assert manifest["video_render_manifest"] == str(manifest_path.resolve())
    assert manifest["semantic_frame_hold_duration_seconds"] == pytest.approx(0.5)
    aspect = subprocess.run(
        [
            shutil.which("ffprobe") or "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=sample_aspect_ratio",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert aspect == "2:1"
