#!/usr/bin/env python3
"""Two-pass loudness mastering with measured verification and correction."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

from .common import probe_audio, write_json


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=True)


def parse_loudnorm_json(stderr: str) -> dict[str, float]:
    matches = re.findall(r"\{\s*\"input_i\".*?\}", stderr, flags=re.S)
    if not matches:
        raise RuntimeError("FFmpeg did not emit loudnorm JSON.")
    raw = json.loads(matches[-1])
    numeric: dict[str, float] = {}
    for key, value in raw.items():
        try:
            numeric[key] = float(value)
        except (TypeError, ValueError):
            # FFmpeg also reports a textual normalization_type field.
            continue
    required = {"input_i", "input_lra", "input_tp", "input_thresh", "target_offset"}
    missing = sorted(required.difference(numeric))
    if missing:
        raise RuntimeError(
            "FFmpeg loudnorm report is missing numeric fields: " + ", ".join(missing)
        )
    return numeric


def measure_ebur128(path: Path) -> dict[str, float]:
    process = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-filter_complex",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    summaries = list(
        re.finditer(r"Summary:\s*(.*?)(?:\n\[|\Z)", process.stderr, flags=re.S)
    )
    summary = summaries[-1].group(1) if summaries else process.stderr[-5000:]

    def value(pattern: str) -> float:
        match = re.search(pattern, summary)
        return float(match.group(1)) if match else float("nan")

    return {
        "integrated_lufs": value(r"I:\s+(-?\d+(?:\.\d+)?)\s+LUFS"),
        "lra_lu": value(r"LRA:\s+(\d+(?:\.\d+)?)\s+LU"),
        "true_peak_dbtp": value(r"Peak:\s+(-?\d+(?:\.\d+)?)\s+dBFS"),
    }


def render_master(
    source: Path,
    destination: Path,
    *,
    measured: dict[str, float],
    target_lufs: float,
    target_lra: float,
    target_peak: float,
    highpass_hz: float,
    correction_db: float,
) -> None:
    limiter = 10.0 ** (target_peak / 20.0)
    base = (
        f"highpass=f={highpass_hz},"
        f"loudnorm=I={target_lufs}:LRA={target_lra}:TP={target_peak}:"
        f"measured_I={measured['input_i']}:"
        f"measured_LRA={measured['input_lra']}:"
        f"measured_TP={measured['input_tp']}:"
        f"measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:linear=true:print_format=summary,"
        f"volume={correction_db:.4f}dB,"
        f"alimiter=limit={limiter:.8f}:attack=5:release=50:level=false,"
        "aresample=48000:resampler=soxr:dither_method=triangular_hp"
    )
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(source),
            "-af",
            base,
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s24le",
            str(destination),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-lufs", type=float, default=-16.0)
    parser.add_argument("--target-lra", type=float, default=7.0)
    parser.add_argument("--target-peak", type=float, default=-1.5)
    parser.add_argument("--highpass-hz", type=float, default=70.0)
    args = parser.parse_args()

    first = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(args.audio),
            "-af",
            f"highpass=f={args.highpass_hz},"
            f"loudnorm=I={args.target_lufs}:LRA={args.target_lra}:"
            f"TP={args.target_peak}:print_format=json",
            "-f",
            "null",
            "-",
        ]
    )
    measured = parse_loudnorm_json(first.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    correction = 0.0
    attempts: list[dict[str, Any]] = []
    for attempt in range(3):
        render_master(
            args.audio,
            args.output,
            measured=measured,
            target_lufs=args.target_lufs,
            target_lra=args.target_lra,
            target_peak=args.target_peak,
            highpass_hz=args.highpass_hz,
            correction_db=correction,
        )
        metrics = measure_ebur128(args.output)
        attempts.append(
            {
                "attempt": attempt + 1,
                "correction_db": correction,
                **metrics,
            }
        )
        integrated = metrics["integrated_lufs"]
        if math.isfinite(integrated) and abs(integrated - args.target_lufs) <= 0.15:
            break
        if not math.isfinite(integrated):
            raise RuntimeError("Could not verify mastered integrated loudness.")
        correction += args.target_lufs - integrated

    final_metrics = measure_ebur128(args.output)
    report = {
        "source": str(args.audio.resolve()),
        "output": str(args.output.resolve()),
        "target_lufs": args.target_lufs,
        "target_lra": args.target_lra,
        "target_true_peak_dbtp": args.target_peak,
        "highpass_hz": args.highpass_hz,
        "first_pass": measured,
        "attempts": attempts,
        "verified": final_metrics,
        "format": probe_audio(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
