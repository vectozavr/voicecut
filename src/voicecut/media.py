"""FFmpeg-backed media input preparation and audio publication.

The narration pipeline deliberately works on one canonical, lossless WAV.
This module keeps container/codec concerns outside the DSP and semantic
stages:

* any FFmpeg-readable audio stream can be decoded to a float32 WAV;
* audio-only and video inputs are identified from stream metadata, not names;
* a finished WAV can be published in common consumer audio formats.

All subprocesses use argument arrays (never a shell), write to a temporary
file, and validate the resulting stream before it replaces its destination.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .common import sha256_file, write_json


AUDIO_INPUT_EXTENSIONS = frozenset(
    {
        ".aac",
        ".aif",
        ".aiff",
        ".flac",
        ".m4a",
        ".mp3",
        ".oga",
        ".ogg",
        ".opus",
        ".wav",
    }
)
VIDEO_INPUT_EXTENSIONS = frozenset({".mkv", ".mov", ".mp4", ".webm"})
AUDIO_OUTPUT_EXTENSIONS = frozenset(
    {
        ".aac",
        ".aif",
        ".aiff",
        ".flac",
        ".m4a",
        ".mp3",
        ".oga",
        ".ogg",
        ".opus",
        ".wav",
    }
)
VIDEO_OUTPUT_EXTENSIONS = frozenset({".mkv", ".mov", ".mp4", ".webm"})


class MediaError(RuntimeError):
    """A source or generated media artifact is invalid."""


class MediaToolError(MediaError):
    """FFmpeg or FFprobe could not complete a requested operation."""


@dataclass(frozen=True)
class MediaStream:
    """The stream fields needed by VoiceCut without exposing FFprobe internals."""

    index: int
    codec_type: str
    codec_name: str | None
    duration_seconds: float | None
    start_seconds: float
    sample_rate: int | None
    channels: int | None
    width: int | None
    height: int | None
    average_frame_rate: float | None
    attached_picture: bool


@dataclass(frozen=True)
class MediaInfo:
    """Validated media metadata used by the pipeline and video renderer."""

    path: str
    format_name: str | None
    duration_seconds: float
    audio_stream: MediaStream
    video_stream: MediaStream | None
    stream_count: int

    @property
    def kind(self) -> str:
        return "video" if self.video_stream is not None else "audio"


Runner = Callable[..., subprocess.CompletedProcess[str]]


def output_media_kind(path: Path) -> str:
    """Return ``audio`` or ``video`` for a supported publication suffix."""

    extension = path.suffix.lower()
    if extension in AUDIO_OUTPUT_EXTENSIONS:
        return "audio"
    if extension in VIDEO_OUTPUT_EXTENSIONS:
        return "video"
    supported = ", ".join(sorted(AUDIO_OUTPUT_EXTENSIONS | VIDEO_OUTPUT_EXTENSIONS))
    raise MediaError(
        f"unsupported output extension {extension or '<none>'!r}; use {supported}"
    )


def _finite_float(value: Any, *, default: float | None = None) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def _positive_int(value: Any) -> int | None:
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return None
    return converted if converted > 0 else None


def _frame_rate(value: Any) -> float | None:
    if not isinstance(value, str) or not value or value == "0/0":
        return None
    numerator, separator, denominator = value.partition("/")
    if not separator:
        rate = _finite_float(value)
    else:
        top = _finite_float(numerator)
        bottom = _finite_float(denominator)
        rate = None if top is None or bottom in {None, 0.0} else top / bottom
    return rate if rate is not None and rate > 0 else None


def _tool_path(tool: str | Path) -> str:
    value = os.fspath(tool)
    if Path(value).parent != Path("."):
        candidate = Path(value)
        if not candidate.is_file():
            raise MediaToolError(f"media executable does not exist: {candidate}")
        return str(candidate)
    resolved = shutil.which(value)
    if resolved is None:
        raise MediaToolError(
            f"{value} is required but was not found on PATH; install FFmpeg"
        )
    return resolved


def _run(
    command: Sequence[str],
    *,
    runner: Runner = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise MediaToolError(f"could not launch {command[0]}: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout or "no diagnostic output").strip()
        raise MediaToolError(
            f"{Path(command[0]).name} failed with exit code "
            f"{result.returncode}:\n{detail[-4000:]}"
        )
    return result


def _stream_from_probe(value: dict[str, Any]) -> MediaStream:
    disposition = value.get("disposition")
    attached_picture = bool(
        isinstance(disposition, dict) and disposition.get("attached_pic") == 1
    )
    return MediaStream(
        index=int(value["index"]),
        codec_type=str(value.get("codec_type") or ""),
        codec_name=(
            str(value["codec_name"]) if value.get("codec_name") is not None else None
        ),
        duration_seconds=_finite_float(value.get("duration")),
        start_seconds=float(_finite_float(value.get("start_time"), default=0.0)),
        sample_rate=_positive_int(value.get("sample_rate")),
        channels=_positive_int(value.get("channels")),
        width=_positive_int(value.get("width")),
        height=_positive_int(value.get("height")),
        average_frame_rate=_frame_rate(
            value.get("avg_frame_rate") or value.get("r_frame_rate")
        ),
        attached_picture=attached_picture,
    )


def probe_media(
    path: Path,
    *,
    ffprobe: str | Path = "ffprobe",
    runner: Runner = subprocess.run,
) -> MediaInfo:
    """Inspect a media file and select its first real audio/video streams."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    result = _run(
        [
            _tool_path(ffprobe),
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        runner=runner,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError(f"FFprobe returned invalid JSON for {path}") from exc
    raw_streams = payload.get("streams")
    if not isinstance(raw_streams, list):
        raise MediaError(f"FFprobe returned no stream list for {path}")
    streams = [
        _stream_from_probe(stream)
        for stream in raw_streams
        if isinstance(stream, dict) and type(stream.get("index")) is int
    ]
    audio_stream = next(
        (stream for stream in streams if stream.codec_type == "audio"),
        None,
    )
    if audio_stream is None:
        raise MediaError(f"input contains no audio stream: {path}")
    video_stream = next(
        (
            stream
            for stream in streams
            if stream.codec_type == "video" and not stream.attached_picture
        ),
        None,
    )
    raw_format = payload.get("format")
    format_value = raw_format if isinstance(raw_format, dict) else {}
    format_duration = _finite_float(format_value.get("duration"))
    durations = [
        duration
        for duration in (
            format_duration,
            audio_stream.duration_seconds,
            video_stream.duration_seconds if video_stream is not None else None,
        )
        if duration is not None and duration > 0
    ]
    if not durations:
        raise MediaError(f"input has no finite positive duration: {path}")
    return MediaInfo(
        path=str(path),
        format_name=(
            str(format_value["format_name"])
            if format_value.get("format_name") is not None
            else None
        ),
        duration_seconds=max(durations),
        audio_stream=audio_stream,
        video_stream=video_stream,
        stream_count=len(streams),
    )


def _temporary_output(destination: Path) -> Path:
    return destination.with_name(
        f".{destination.stem}.voicecut-{os.getpid()}.tmp{destination.suffix}"
    )


def _resolve_output_destination(destination: Path) -> Path:
    """Resolve a publication target without ever following a final symlink."""

    requested = destination.absolute()
    if requested.is_symlink():
        raise MediaError(f"output path must not be a symbolic link: {requested}")
    return requested.resolve()


def _prepare_destination(destination: Path, *, overwrite: bool) -> Path:
    destination = _resolve_output_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    temporary = _temporary_output(destination)
    if temporary.exists():
        temporary.unlink()
    return temporary


def _replace_validated(temporary: Path, destination: Path) -> None:
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise MediaError(f"media encoder produced no output: {temporary}")
    temporary.replace(destination)


def extract_audio_to_wav(
    input_path: Path,
    output_path: Path,
    *,
    ffmpeg: str | Path = "ffmpeg",
    ffprobe: str | Path = "ffprobe",
    overwrite: bool = False,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Decode the selected input audio stream to a canonical float32 WAV."""

    source = probe_media(input_path, ffprobe=ffprobe, runner=runner)
    output_path = _resolve_output_destination(output_path)
    if output_path == Path(source.path):
        raise MediaError("canonical audio output cannot replace the source media")
    if output_path.suffix.lower() != ".wav":
        raise MediaError("canonical analysis audio must use a .wav destination")
    temporary = _prepare_destination(output_path, overwrite=overwrite)
    try:
        _run(
            [
                _tool_path(ffmpeg),
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                source.path,
                "-map",
                f"0:{source.audio_stream.index}",
                "-vn",
                "-map_metadata",
                "-1",
                "-c:a",
                "pcm_f32le",
                str(temporary),
            ],
            runner=runner,
        )
        decoded = probe_media(temporary, ffprobe=ffprobe, runner=runner)
        if decoded.video_stream is not None:
            raise MediaError("canonical analysis WAV unexpectedly contains video")
        if decoded.audio_stream.codec_name not in {"pcm_f32le", "pcm_f32be"}:
            raise MediaError(
                "canonical analysis WAV is not float PCM: "
                f"{decoded.audio_stream.codec_name}"
            )
        _replace_validated(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "schema_version": 1,
        "status": "complete",
        "source_media": source.path,
        "source_media_sha256": sha256_file(Path(source.path)),
        "source_kind": source.kind,
        "source_media_info": asdict(source),
        "canonical_audio": str(output_path),
        "canonical_audio_sha256": sha256_file(output_path),
        "canonical_audio_info": asdict(
            probe_media(output_path, ffprobe=ffprobe, runner=runner)
        ),
    }


def prepare_media_input(
    input_path: Path,
    output_dir: Path,
    *,
    ffmpeg: str | Path = "ffmpeg",
    ffprobe: str | Path = "ffprobe",
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Create the canonical input WAV and an immutable provenance manifest."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "media_input.json"
    audio_path = output_dir / "source_audio.wav"
    if manifest_path.exists() or audio_path.exists():
        raise MediaError(
            f"media preparation output already exists in {output_dir}; "
            "use an empty stage directory"
        )
    manifest = extract_audio_to_wav(
        input_path,
        audio_path,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        runner=runner,
    )
    write_json(manifest_path, manifest)
    return manifest


def _audio_codec_arguments(extension: str) -> list[str]:
    profiles = {
        ".aac": ["-c:a", "aac", "-b:a", "192k", "-f", "adts"],
        ".aif": ["-c:a", "pcm_s24be"],
        ".aiff": ["-c:a", "pcm_s24be"],
        ".flac": ["-c:a", "flac", "-compression_level", "8"],
        ".m4a": ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"],
        ".mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
        ".oga": ["-c:a", "libvorbis", "-q:a", "5"],
        ".ogg": ["-c:a", "libvorbis", "-q:a", "5"],
        ".opus": ["-c:a", "libopus", "-b:a", "160k", "-vbr", "on"],
        ".wav": ["-c:a", "pcm_s24le"],
    }
    try:
        return list(profiles[extension])
    except KeyError as exc:
        supported = ", ".join(sorted(AUDIO_OUTPUT_EXTENSIONS))
        raise MediaError(
            f"unsupported audio output extension {extension!r}; use {supported}"
        ) from exc


def publish_audio(
    final_wav: Path,
    output_path: Path,
    *,
    manifest_path: Path | None = None,
    ffmpeg: str | Path = "ffmpeg",
    ffprobe: str | Path = "ffprobe",
    overwrite: bool = False,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Encode a finished pipeline WAV to a requested consumer audio format."""

    source = probe_media(final_wav, ffprobe=ffprobe, runner=runner)
    if source.video_stream is not None:
        raise MediaError("audio publication input must not contain video")
    output_path = _resolve_output_destination(output_path)
    if output_path == Path(source.path):
        raise MediaError("published audio output cannot replace its source WAV")
    extension = output_path.suffix.lower()
    codec_arguments = _audio_codec_arguments(extension)
    resolved_manifest_path = (
        manifest_path.resolve()
        if manifest_path is not None
        else output_path.with_name(f"{output_path.stem}.audio_publish_manifest.json")
    )
    if resolved_manifest_path == output_path:
        raise MediaError("audio manifest path cannot equal the output audio")
    if resolved_manifest_path.exists() and not overwrite:
        raise FileExistsError(resolved_manifest_path)
    temporary = _prepare_destination(output_path, overwrite=overwrite)
    try:
        _run(
            [
                _tool_path(ffmpeg),
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(Path(source.path)),
                "-map",
                "0:a:0",
                "-vn",
                "-map_metadata",
                "-1",
                *codec_arguments,
                str(temporary),
            ],
            runner=runner,
        )
        published = probe_media(temporary, ffprobe=ffprobe, runner=runner)
        if published.video_stream is not None:
            raise MediaError("published audio unexpectedly contains video")
        tolerance = max(0.12, source.duration_seconds * 0.002)
        if abs(published.duration_seconds - source.duration_seconds) > tolerance:
            raise MediaError(
                "published audio duration changed unexpectedly: "
                f"{source.duration_seconds:.6f}s -> "
                f"{published.duration_seconds:.6f}s"
            )
        _replace_validated(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = {
        "schema_version": 1,
        "publisher": "voicecut_ffmpeg_audio_publication_v1",
        "status": "complete",
        "source_wav": source.path,
        "source_wav_sha256": sha256_file(Path(source.path)),
        "output_audio": str(output_path),
        "output_audio_sha256": sha256_file(output_path),
        "output_extension": extension,
        "audio_publish_manifest": str(resolved_manifest_path),
        "output_media_info": asdict(
            probe_media(output_path, ffprobe=ffprobe, runner=runner)
        ),
    }
    write_json(resolved_manifest_path, manifest)
    return manifest
