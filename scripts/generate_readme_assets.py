#!/usr/bin/env python3
"""Generate deterministic README visuals from the public VoiceCut examples.

The script deliberately reads only committed public media and the sanitized
demo manifest.  It does not read pipeline work directories, raw LLM replies,
or developer-specific absolute paths.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "examples" / "demo_manifest.json"
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "docs" / "assets" / "readme"
WAVEFORM_SAMPLE_RATE = 2_000
WAVEFORM_COLUMNS = 960

COLORS = {
    "background": "#f8fafc",
    "panel": "#ffffff",
    "ink": "#172033",
    "muted": "#64748b",
    "grid": "#dbe4ef",
    "source": "#94a3b8",
    "selected": "#11a683",
    "selected_dark": "#087b65",
    "removed": "#e2e8f0",
    "accent": "#625bf6",
    "accent_light": "#e8e7ff",
    "warning": "#f59e0b",
}


class AssetGenerationError(RuntimeError):
    """The public example assets cannot be generated safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_fingerprint() -> str:
    """Match VoiceCut's source-tree fingerprint without importing the package."""

    digest = hashlib.sha256()
    package_directory = REPOSITORY_ROOT / "src" / "voicecut"
    for path in sorted(package_directory.glob("*.py"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run(command: list[str], *, binary: bool = False) -> bytes | str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=not binary,
        )
    except FileNotFoundError as error:
        raise AssetGenerationError(
            f"required executable is unavailable: {command[0]}"
        ) from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise AssetGenerationError(
            f"command failed: {' '.join(command)}\n{stderr.strip()}"
        ) from error
    return completed.stdout


def _resolve_public_path(relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AssetGenerationError(
            f"manifest path must be repository-relative: {relative_path!r}"
        )
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise AssetGenerationError(
            f"manifest path escapes the repository: {relative_path!r}"
        ) from error
    return resolved


def _walk_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for nested in value for item in _walk_strings(nested)]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _walk_strings(nested)]
    return []


def _verify_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise AssetGenerationError("unsupported demo manifest schema")

    for value in _walk_strings(manifest):
        if value.startswith(("/", "~")):
            raise AssetGenerationError("manifest contains a local absolute path")
        if "GEMINI_API_KEY=" in value:
            raise AssetGenerationError("manifest appears to contain an API key value")

    repository_state = manifest.get("repository_state")
    if not isinstance(repository_state, dict):
        raise AssetGenerationError("manifest is missing repository_state")
    if repository_state.get("kind") != "voicecut_implementation_sha256":
        raise AssetGenerationError("manifest must use a VoiceCut implementation hash")
    expected_fingerprint = repository_state.get("value")
    actual_fingerprint = _implementation_fingerprint()
    if expected_fingerprint != actual_fingerprint:
        raise AssetGenerationError(
            "demo manifest implementation fingerprint is stale: "
            f"expected {expected_fingerprint}, current {actual_fingerprint}"
        )

    examples = manifest.get("examples")
    if not isinstance(examples, list) or not examples:
        raise AssetGenerationError("manifest must contain public examples")
    identifiers: set[str] = set()
    for example in examples:
        if not isinstance(example, dict):
            raise AssetGenerationError("each example must be an object")
        identifier = example.get("id")
        if not isinstance(identifier, str) or identifier in identifiers:
            raise AssetGenerationError("example IDs must be unique strings")
        identifiers.add(identifier)
        for artifact_name in ("input", "output"):
            artifact = example.get(artifact_name)
            if not isinstance(artifact, dict):
                raise AssetGenerationError(
                    f"{identifier}.{artifact_name} must be an object"
                )
            for path_key, hash_key in (
                ("path", "sha256"),
                ("preview_path", "preview_sha256"),
            ):
                relative_path = artifact.get(path_key)
                expected_hash = artifact.get(hash_key)
                if relative_path is None and expected_hash is None:
                    continue
                if not isinstance(relative_path, str) or not isinstance(
                    expected_hash, str
                ):
                    raise AssetGenerationError(
                        f"{identifier}.{artifact_name} has incomplete {path_key} metadata"
                    )
                path = _resolve_public_path(relative_path)
                if not path.is_file():
                    raise AssetGenerationError(
                        f"public media is missing: {relative_path}"
                    )
                actual_hash = _sha256(path)
                if actual_hash != expected_hash:
                    raise AssetGenerationError(
                        f"public media hash mismatch for {relative_path}: {actual_hash}"
                    )
            primary_path = _resolve_public_path(str(artifact["path"]))
            if primary_path.stat().st_size != int(artifact["bytes"]):
                raise AssetGenerationError(
                    f"public media size mismatch for {artifact['path']}"
                )


def _decode_mono(path: Path) -> list[float]:
    payload = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(WAVEFORM_SAMPLE_RATE),
            "-f",
            "f32le",
            "-",
        ],
        binary=True,
    )
    assert isinstance(payload, bytes)
    samples = array.array("f")
    samples.frombytes(payload)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise AssetGenerationError(f"no audio samples decoded from {path}")
    return list(samples)


def _waveform_polygon(
    samples: list[float],
    *,
    x: float,
    center_y: float,
    width: float,
    half_height: float,
) -> str:
    columns = min(WAVEFORM_COLUMNS, max(1, len(samples)))
    bucket_size = len(samples) / columns
    peaks: list[float] = []
    for column in range(columns):
        start = math.floor(column * bucket_size)
        end = max(start + 1, math.ceil((column + 1) * bucket_size))
        peak = max(abs(sample) for sample in samples[start:end])
        peaks.append(peak)
    scale = max(peaks) or 1.0
    top = [
        (
            x + index * width / max(1, columns - 1),
            center_y - (peak / scale) * half_height,
        )
        for index, peak in enumerate(peaks)
    ]
    bottom = [(point_x, 2 * center_y - point_y) for point_x, point_y in reversed(top)]
    return " ".join(f"{point_x:.2f},{point_y:.2f}" for point_x, point_y in top + bottom)


def _svg_document(width: int, height: int, content: list[str]) -> str:
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}" role="img">'
            ),
            "  <style>",
            (
                "    text { font-family: Inter, ui-sans-serif, -apple-system, "
                "BlinkMacSystemFont, 'Segoe UI', sans-serif; }"
            ),
            "  </style>",
            *content,
            "</svg>",
            "",
        ]
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = value.encode("utf-8")
    if path.is_file() and path.read_bytes() == encoded:
        return
    path.write_bytes(encoded)


def _generate_audio_waveforms(manifest: dict[str, Any], output_directory: Path) -> Path:
    audio_examples = [
        example for example in manifest["examples"] if example["kind"] == "audio"
    ]
    width = 1400
    height = 180 + len(audio_examples) * 230
    content = [
        f'  <rect width="{width}" height="{height}" rx="24" fill="{COLORS["background"]}"/>',
        f'  <text x="48" y="58" font-size="30" font-weight="700" fill="{COLORS["ink"]}">Before and after</text>',
        f'  <text x="48" y="91" font-size="16" fill="{COLORS["muted"]}">Real public examples · waveform amplitude normalized per track</text>',
    ]
    for example_index, example in enumerate(audio_examples):
        panel_y = 125 + example_index * 230
        content.append(
            f'  <rect x="32" y="{panel_y}" width="1304" height="206" rx="18" fill="{COLORS["panel"]}" stroke="{COLORS["grid"]}"/>'
        )
        title = escape(str(example["title"]))
        removed = float(example["result"]["duration_reduction_percent"])
        content.append(
            f'  <text x="56" y="{panel_y + 34}" font-size="19" font-weight="700" fill="{COLORS["ink"]}">{title}</text>'
        )
        content.append(
            f'  <text x="1300" y="{panel_y + 34}" text-anchor="end" font-size="14" font-weight="600" fill="{COLORS["selected_dark"]}">{removed:.1f}% shorter</text>'
        )
        for track_index, artifact_name in enumerate(("input", "output")):
            artifact = example[artifact_name]
            path = _resolve_public_path(str(artifact["path"]))
            samples = _decode_mono(path)
            center_y = panel_y + 86 + track_index * 76
            label = "Original" if artifact_name == "input" else "VoiceCut"
            color = COLORS["source"] if artifact_name == "input" else COLORS["selected"]
            polygon = _waveform_polygon(
                samples,
                x=180,
                center_y=center_y,
                width=1050,
                half_height=29,
            )
            content.extend(
                [
                    f'  <text x="56" y="{center_y + 5:.2f}" font-size="15" font-weight="600" fill="{COLORS["ink"]}">{label}</text>',
                    f'  <line x1="180" y1="{center_y:.2f}" x2="1230" y2="{center_y:.2f}" stroke="{COLORS["grid"]}"/>',
                    f'  <polygon points="{polygon}" fill="{color}" fill-opacity="0.88"/>',
                    (
                        f'  <text x="1300" y="{center_y + 5:.2f}" text-anchor="end" '
                        f'font-size="14" fill="{COLORS["muted"]}">'
                        f"{float(artifact['duration_seconds']):.2f}s</text>"
                    ),
                ]
            )
    output_path = output_directory / "audio-waveforms.svg"
    _write_text(output_path, _svg_document(width, height, content))
    return output_path


def _generate_pipeline_svg(output_directory: Path) -> Path:
    width = 1500
    height = 590
    content = [
        "  <defs>",
        f'    <marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="{COLORS["muted"]}"/></marker>',
        "  </defs>",
        f'  <rect width="{width}" height="{height}" rx="24" fill="{COLORS["background"]}"/>',
        f'  <text x="48" y="57" font-size="30" font-weight="700" fill="{COLORS["ink"]}">One semantic plan, one final render</text>',
        f'  <text x="48" y="90" font-size="16" fill="{COLORS["muted"]}">Production path reflected by voicecut.full_pipeline and voicecut.final_render</text>',
    ]

    boxes = [
        (45, 135, 205, 105, "Audio / video", "Canonical source WAV"),
        (285, 135, 205, 105, "Whisper", "Words + time anchors"),
        (525, 135, 245, 105, "Streaming planner", "Gemini selects source IDs"),
        (805, 135, 215, 105, "Grounding", "Exact source occurrences"),
        (1055, 135, 220, 105, "Completeness veto", "WhisperX · no cuts"),
        (925, 350, 255, 115, "MFA 3.4.1", "Phone-safe boundaries"),
        (610, 350, 270, 115, "Pause + ambience", "Breath-safe source tone"),
        (300, 350, 265, 115, "Immutable plan", "All samples resolved first"),
        (45, 350, 210, 115, "Single render", "Canonical samples once"),
    ]
    for index, (x, y, box_width, box_height, title, subtitle) in enumerate(boxes):
        highlight = index in {2, 5, 7, 8}
        fill = COLORS["accent_light"] if highlight else COLORS["panel"]
        stroke = COLORS["accent"] if highlight else COLORS["grid"]
        content.extend(
            [
                f'  <rect x="{x}" y="{y}" width="{box_width}" height="{box_height}" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="{2 if highlight else 1}"/>',
                f'  <text x="{x + 18}" y="{y + 39}" font-size="18" font-weight="700" fill="{COLORS["ink"]}">{escape(title)}</text>',
                f'  <text x="{x + 18}" y="{y + 70}" font-size="14" fill="{COLORS["muted"]}">{escape(subtitle)}</text>',
            ]
        )
    arrow_style = f'stroke="{COLORS["muted"]}" stroke-width="2" fill="none" marker-end="url(#arrow)"'
    for start_x, end_x in ((250, 285), (490, 525), (770, 805), (1020, 1055)):
        content.append(f'  <path d="M {start_x} 188 L {end_x - 8} 188" {arrow_style}/>')
    content.extend(
        [
            f'  <path d="M 1165 240 L 1165 294 Q 1165 320 1135 320 L 1060 320 L 1060 342" {arrow_style}/>',
            f'  <path d="M 925 407 L 888 407" {arrow_style}/>',
            f'  <path d="M 610 407 L 573 407" {arrow_style}/>',
            f'  <path d="M 300 407 L 263 407" {arrow_style}/>',
            f'  <text x="48" y="520" font-size="14" fill="{COLORS["muted"]}">Whisper timestamps are crop anchors. MFA phone alignment owns final cut coordinates.</text>',
            f'  <text x="48" y="546" font-size="14" fill="{COLORS["muted"]}">One immutable plan is rendered once; video publication uses clear cuts without inserted pauses.</text>',
        ]
    )
    output_path = output_directory / "pipeline.svg"
    _write_text(output_path, _svg_document(width, height, content))
    return output_path


def _generate_selection_boundaries(
    manifest: dict[str, Any], output_directory: Path
) -> Path:
    width = 1400
    lane_height = 125
    height = 185 + lane_height * len(manifest["examples"])
    timeline_x = 215.0
    timeline_width = 1030.0
    content = [
        f'  <rect width="{width}" height="{height}" rx="24" fill="{COLORS["background"]}"/>',
        f'  <text x="48" y="58" font-size="30" font-weight="700" fill="{COLORS["ink"]}">What VoiceCut kept</text>',
        f'  <text x="48" y="91" font-size="16" fill="{COLORS["muted"]}">Green spans are selected canonical-source samples; gray spans were omitted</text>',
    ]
    for lane_index, example in enumerate(manifest["examples"]):
        y = 140 + lane_index * lane_height
        duration = float(example["input"]["duration_seconds"])
        title = escape(str(example["title"]))
        content.extend(
            [
                f'  <text x="48" y="{y + 27}" font-size="17" font-weight="700" fill="{COLORS["ink"]}">{title}</text>',
                f'  <text x="48" y="{y + 53}" font-size="13" fill="{COLORS["muted"]}">{duration:.2f}s source</text>',
                f'  <rect x="{timeline_x}" y="{y}" width="{timeline_width}" height="58" rx="9" fill="{COLORS["removed"]}"/>',
            ]
        )
        for interval_index, interval in enumerate(example["selected_source_intervals"]):
            start = float(interval["start_seconds"])
            end = min(duration, float(interval["end_seconds"]))
            x = timeline_x + timeline_width * start / duration
            interval_width = max(1.5, timeline_width * (end - start) / duration)
            word_start = int(interval["start_word_id"])
            word_end = int(interval["end_word_id"])
            content.extend(
                [
                    f'  <rect x="{x:.2f}" y="{y}" width="{interval_width:.2f}" height="58" rx="7" fill="{COLORS["selected"]}"/>',
                    f'  <line x1="{x:.2f}" y1="{y - 7}" x2="{x:.2f}" y2="{y + 65}" stroke="{COLORS["selected_dark"]}" stroke-width="1.5"/>',
                ]
            )
            if interval_width >= 88:
                content.append(
                    f'  <text x="{x + interval_width / 2:.2f}" y="{y + 35}" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">words {word_start}–{word_end - 1}</text>'
                )
            if interval_index == len(example["selected_source_intervals"]) - 1:
                end_x = x + interval_width
                content.append(
                    f'  <line x1="{end_x:.2f}" y1="{y - 7}" x2="{end_x:.2f}" y2="{y + 65}" stroke="{COLORS["selected_dark"]}" stroke-width="1.5"/>'
                )
        for tick in range(5):
            tick_x = timeline_x + timeline_width * tick / 4
            tick_time = duration * tick / 4
            content.extend(
                [
                    f'  <line x1="{tick_x:.2f}" y1="{y + 64}" x2="{tick_x:.2f}" y2="{y + 70}" stroke="{COLORS["muted"]}"/>',
                    f'  <text x="{tick_x:.2f}" y="{y + 88}" text-anchor="middle" font-size="12" fill="{COLORS["muted"]}">{tick_time:.1f}s</text>',
                ]
            )
        result = example["result"]
        content.append(
            f'  <text x="48" y="{y + 78}" font-size="11" font-weight="600" fill="{COLORS["selected_dark"]}">{int(result["selected_source_ranges"])} ranges · {int(result["unsafe_boundaries"])} unsafe</text>'
        )
    output_path = output_directory / "selection-boundaries.svg"
    _write_text(output_path, _svg_document(width, height, content))
    return output_path


def _extract_poster(source: Path, destination: Path, timestamp: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".jpg", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(source),
                "-map_metadata",
                "-1",
                "-frames:v",
                "1",
                "-vf",
                "scale=960:-2:flags=lanczos",
                "-q:v",
                "3",
                str(temporary_path),
            ]
        )
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise AssetGenerationError(f"ffmpeg did not create {destination.name}")
        os.replace(temporary_path, destination)
        destination.chmod(0o644)
    finally:
        temporary_path.unlink(missing_ok=True)


def _generate_video_posters(
    manifest: dict[str, Any], output_directory: Path
) -> list[Path]:
    video_examples = [
        example for example in manifest["examples"] if example["kind"] == "video"
    ]
    if len(video_examples) != 1:
        raise AssetGenerationError("exactly one public video example is required")
    example = video_examples[0]
    before_path = output_directory / "video-before.jpg"
    after_path = output_directory / "video-after.jpg"
    _extract_poster(_resolve_public_path(example["input"]["path"]), before_path, 1.0)
    _extract_poster(_resolve_public_path(example["output"]["path"]), after_path, 1.0)
    return [before_path, after_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate public README visuals from verified VoiceCut media."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    manifest_path = arguments.manifest.resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"demo manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise AssetGenerationError("demo manifest root must be an object")
        _verify_manifest(manifest)
        output_directory = arguments.output_dir.resolve()
        generated = [
            _generate_audio_waveforms(manifest, output_directory),
            _generate_pipeline_svg(output_directory),
            _generate_selection_boundaries(manifest, output_directory),
            *_generate_video_posters(manifest, output_directory),
        ]
    except (AssetGenerationError, json.JSONDecodeError, OSError) as error:
        raise SystemExit(f"README asset generation failed: {error}") from error

    print("README ASSETS GENERATED")
    for path in generated:
        try:
            display_path = path.relative_to(REPOSITORY_ROOT)
        except ValueError:
            display_path = path
        print(display_path)


if __name__ == "__main__":
    main()
