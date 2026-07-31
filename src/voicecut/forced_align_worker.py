#!/usr/bin/env python3
"""CPU-only WhisperX alignment worker for local hard-boundary crops.

This process intentionally exposes only WhisperX's alignment stage. It never
loads a Whisper transcription model and never transcribes audio.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Sequence


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_alignment_jobs(
    *,
    jobs_path: Path,
    output_path: Path,
    language: str,
    device: str,
) -> dict[str, Any]:
    if device != "cpu":
        raise ValueError("the hard-boundary worker is CPU-only")

    source = json.loads(jobs_path.read_text(encoding="utf-8"))
    jobs = source.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("alignment job file must contain a non-empty jobs list")

    import whisperx

    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
        )
    except Exception as error:
        result = {
            "schema_version": 1,
            "backend": "whisperx_alignment",
            "language": language,
            "device": device,
            "fatal_error": f"{type(error).__name__}: {error}",
            "fatal_traceback": traceback.format_exc(),
            "jobs": [
                {
                    "clip_index": job.get("clip_index"),
                    "error": f"alignment model load failed: {error}",
                }
                for job in jobs
            ],
        }
        _write_json(output_path, result)
        return result

    aligned_jobs: list[dict[str, Any]] = []
    for job in jobs:
        clip_index = job.get("clip_index")
        crop_path = Path(str(job.get("crop_wav", ""))).resolve()
        text = str(job.get("local_source_text", "")).strip()
        crop_duration = float(job.get("crop_duration_seconds", 0.0))
        try:
            if not crop_path.is_file():
                raise FileNotFoundError(crop_path)
            if not text:
                raise ValueError("local source transcript is empty")
            if crop_duration <= 0.0:
                raise ValueError("crop duration must be positive")
            crop_audio = whisperx.load_audio(str(crop_path))
            actual_duration = len(crop_audio) / 16000.0
            segment_end = min(crop_duration, actual_duration)
            if segment_end <= 0.0:
                raise ValueError("decoded crop is empty")
            segments = [
                {
                    "start": 0.0,
                    "end": segment_end,
                    "text": text,
                }
            ]
            aligned = whisperx.align(
                segments,
                model_a,
                metadata,
                crop_audio,
                device,
                return_char_alignments=True,
                print_progress=False,
            )
            aligned_jobs.append(
                {
                    "clip_index": clip_index,
                    "crop_wav": str(crop_path),
                    "crop_duration_seconds": crop_duration,
                    "decoded_duration_seconds": actual_duration,
                    "local_source_text": text,
                    "aligned": aligned,
                    "error": None,
                }
            )
        except Exception as error:
            aligned_jobs.append(
                {
                    "clip_index": clip_index,
                    "crop_wav": str(crop_path),
                    "crop_duration_seconds": crop_duration,
                    "local_source_text": text,
                    "aligned": None,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )

    result = {
        "schema_version": 1,
        "backend": "whisperx_alignment",
        "language": language,
        "device": device,
        "model_metadata": {
            "language": metadata.get("language"),
            "type": metadata.get("type"),
        },
        "jobs": aligned_jobs,
    }
    _write_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run only WhisperX forced alignment on prepared local crops; "
            "never run transcription."
        )
    )
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_alignment_jobs(
        jobs_path=args.jobs.resolve(),
        output_path=args.output.resolve(),
        language=args.language,
        device=args.device,
    )
    failed = sum(bool(job.get("error")) for job in result["jobs"])
    print(
        json.dumps(
            {
                "jobs": len(result["jobs"]),
                "failures": failed,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
