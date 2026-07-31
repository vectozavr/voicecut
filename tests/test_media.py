from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicecut.common import sha256_file
from voicecut.media import (
    MediaError,
    output_media_kind,
    prepare_media_input,
    probe_media,
    publish_audio,
)


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("cut.WAV", "audio"),
        ("cut.mp3", "audio"),
        ("cut.oga", "audio"),
        ("cut.m4a", "audio"),
        ("cut.MP4", "video"),
        ("cut.mov", "video"),
        ("cut.mkv", "video"),
        ("cut.webm", "video"),
    ],
)
def test_output_media_kind_is_case_insensitive(name: str, kind: str) -> None:
    assert output_media_kind(Path(name)) == kind


def test_output_media_kind_rejects_unknown_or_missing_suffix() -> None:
    with pytest.raises(MediaError, match="unsupported output extension"):
        output_media_kind(Path("cut.wma"))
    with pytest.raises(MediaError, match="unsupported output extension"):
        output_media_kind(Path("cut"))


def test_probe_uses_stream_metadata_and_ignores_attached_cover_art(
    tmp_path: Path,
) -> None:
    source = tmp_path / "recording.unknown"
    source.write_bytes(b"probe fixture")
    payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "mjpeg",
                "width": 600,
                "height": 600,
                "duration": "2.0",
                "disposition": {"attached_pic": 1},
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "mp3",
                "sample_rate": "44100",
                "channels": 2,
                "duration": "12.5",
                "start_time": "0.025",
            },
        ],
        "format": {"format_name": "mp3", "duration": "12.5"},
    }

    def fake_runner(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    info = probe_media(
        source,
        ffprobe=sys.executable,
        runner=fake_runner,
    )

    assert info.kind == "audio"
    assert info.video_stream is None
    assert info.audio_stream.index == 1
    assert info.audio_stream.start_seconds == pytest.approx(0.025)
    assert info.duration_seconds == pytest.approx(12.5)


def test_probe_rejects_media_without_audio(tmp_path: Path) -> None:
    source = tmp_path / "silent.mp4"
    source.write_bytes(b"probe fixture")
    payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "duration": "1",
                "disposition": {"attached_pic": 0},
            }
        ],
        "format": {"duration": "1"},
    }

    def fake_runner(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    with pytest.raises(MediaError, match="no audio stream"):
        probe_media(source, ffprobe=sys.executable, runner=fake_runner)


@pytest.mark.skipif(
    shutil.which("ffprobe") is None,
    reason="FFprobe is not installed",
)
def test_audio_publication_never_follows_an_output_symlink(tmp_path: Path) -> None:
    source = tmp_path / "final.wav"
    original = tmp_path / "original.wav"
    output_link = tmp_path / "edited.wav"
    waveform = np.zeros((8000, 1), dtype=np.float32)
    sf.write(source, waveform, 8000, subtype="FLOAT")
    sf.write(original, waveform + 0.25, 8000, subtype="FLOAT")
    original_sha = sha256_file(original)
    output_link.symlink_to(original)

    with pytest.raises(MediaError, match="symbolic link"):
        publish_audio(
            source,
            output_link,
            manifest_path=tmp_path / "publication.json",
            overwrite=True,
        )

    assert output_link.is_symlink()
    assert sha256_file(original) == original_sha


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is not installed",
)
def test_prepare_mp3_and_publish_common_audio_formats(tmp_path: Path) -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = (0.1 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    source_wav = tmp_path / "source.wav"
    sf.write(source_wav, waveform, sample_rate, subtype="FLOAT")
    source_mp3 = tmp_path / "source.mp3"
    subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source_wav),
            "-c:a",
            "libmp3lame",
            "-q:a",
            "4",
            str(source_mp3),
        ],
        check=True,
    )

    prepared = prepare_media_input(source_mp3, tmp_path / "prepared")
    canonical = Path(prepared["canonical_audio"])
    assert prepared["source_kind"] == "audio"
    assert canonical.is_file()
    assert sf.info(canonical).subtype == "FLOAT"

    for extension in (".wav", ".flac", ".m4a", ".ogg", ".opus", ".mp3"):
        output = tmp_path / f"published{extension}"
        manifest_path = (
            tmp_path / "publication" / f"{extension[1:]}.audio_manifest.json"
        )
        manifest = publish_audio(
            canonical,
            output,
            manifest_path=manifest_path,
        )
        assert output.is_file()
        assert manifest_path.is_file()
        assert not (tmp_path / "published.audio_publish_manifest.json").exists()
        assert manifest["output_extension"] == extension
        assert manifest["audio_publish_manifest"] == str(manifest_path.resolve())
        assert probe_media(output).video_stream is None
