#!/usr/bin/env python3
"""Split a narration waveform into non-overlapping transcription atoms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import binary_closing, binary_dilation, binary_opening
from scipy.signal import resample_poly
from silero_vad import get_speech_timestamps, load_silero_vad

from .common import sha256_file, write_json


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def load_mono(path: Path) -> tuple[np.ndarray, int, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    channels = audio.shape[1]
    mono = np.mean(audio, axis=1, dtype=np.float32)
    return mono, int(sample_rate), int(channels)


def frame_rms_db(audio: np.ndarray, frame_samples: int) -> np.ndarray:
    """Compute the only frame feature required for VAD atom construction."""

    frame_count = math.ceil(len(audio) / frame_samples)
    padded = np.pad(audio, (0, frame_count * frame_samples - len(audio)))
    framed = padded.reshape(frame_count, frame_samples)
    rms = np.sqrt(np.mean(np.square(framed, dtype=np.float64), axis=1))
    return (20.0 * np.log10(np.maximum(rms, 1e-10))).astype(np.float32)


def silero_mask(
    audio: np.ndarray,
    sample_rate: int,
    frame_samples: int,
    threshold: float,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    analysis_rate = 16000
    divisor = math.gcd(sample_rate, analysis_rate)
    audio_16k = resample_poly(
        audio,
        analysis_rate // divisor,
        sample_rate // divisor,
    ).astype(np.float32, copy=False)
    model = load_silero_vad(onnx=False)
    raw = get_speech_timestamps(
        torch.from_numpy(audio_16k),
        model,
        sampling_rate=analysis_rate,
        threshold=threshold,
        min_speech_duration_ms=70,
        min_silence_duration_ms=75,
        speech_pad_ms=25,
        return_seconds=False,
    )
    frame_count = math.ceil(len(audio) / frame_samples)
    mask = np.zeros(frame_count, dtype=bool)
    regions: list[dict[str, float | int]] = []
    for item in raw:
        start = max(0, int(round(item["start"] / analysis_rate * sample_rate)))
        end = min(len(audio), int(round(item["end"] / analysis_rate * sample_rate)))
        start_frame = max(0, start // frame_samples)
        end_frame = min(frame_count, math.ceil(end / frame_samples))
        mask[start_frame:end_frame] = True
        regions.append(
            {
                "start_sample": start,
                "end_sample": end,
                "start": start / sample_rate,
                "end": end / sample_rate,
            }
        )
    return mask, regions


def split_long_atom(
    start_frame: int,
    end_frame: int,
    rms_db: np.ndarray,
    max_frames: int,
    minimum_frames: int,
) -> list[tuple[int, int]]:
    if end_frame - start_frame <= max_frames:
        return [(start_frame, end_frame)]
    result: list[tuple[int, int]] = []
    cursor = start_frame
    while end_frame - cursor > max_frames:
        target = cursor + max_frames
        search_start = max(cursor + minimum_frames, target - max_frames // 3)
        search_end = min(end_frame - minimum_frames, target + max_frames // 3)
        if search_end <= search_start:
            split = target
        else:
            split = search_start + int(np.argmin(rms_db[search_start:search_end]))
        result.append((cursor, split))
        cursor = split
    if end_frame > cursor:
        result.append((cursor, end_frame))
    return result


def build_atoms(
    *,
    rms_db: np.ndarray,
    vad_mask: np.ndarray,
    noise_floor_db: float,
    frame_samples: int,
    sample_rate: int,
    total_samples: int,
    max_atom_seconds: float,
) -> list[dict[str, float | int | str]]:
    frame_seconds = frame_samples / sample_rate
    activity_threshold = float(np.clip(noise_floor_db + 17.0, -56.0, -34.0))
    energy = rms_db >= activity_threshold

    # Keep waveform activity near VAD speech so quiet consonants and word tails
    # survive without admitting arbitrary clicks elsewhere in the recording.
    near_vad = binary_dilation(
        vad_mask, structure=np.ones(max(1, round(0.28 / frame_seconds)))
    )
    speech = vad_mask | (energy & near_vad)
    speech = binary_closing(
        speech, structure=np.ones(max(1, round(0.075 / frame_seconds)))
    )
    speech = binary_opening(
        speech, structure=np.ones(max(1, round(0.025 / frame_seconds)))
    )

    minimum_frames = max(3, round(0.07 / frame_seconds))
    max_frames = max(minimum_frames + 1, round(max_atom_seconds / frame_seconds))
    raw_regions: list[tuple[int, int]] = []
    for start, end in true_runs(speech):
        if end - start < minimum_frames:
            continue
        raw_regions.extend(
            split_long_atom(start, end, rms_db, max_frames, minimum_frames)
        )

    # Resolve padding at the midpoint of each quiet gap, which guarantees that
    # atom transcription windows never overlap and duplicate words.
    atoms: list[dict[str, float | int | str]] = []
    for index, (start_frame, end_frame) in enumerate(raw_regions):
        previous_end = raw_regions[index - 1][1] if index else 0
        next_start = (
            raw_regions[index + 1][0] if index + 1 < len(raw_regions) else len(rms_db)
        )
        left_limit = (previous_end + start_frame) // 2
        right_limit = (end_frame + next_start) // 2
        padded_start_frame = max(left_limit, start_frame - round(0.035 / frame_seconds))
        padded_end_frame = min(right_limit, end_frame + round(0.055 / frame_seconds))
        start_sample = max(0, padded_start_frame * frame_samples)
        end_sample = min(total_samples, padded_end_frame * frame_samples)
        if end_sample - start_sample < round(0.065 * sample_rate):
            continue
        atoms.append(
            {
                "atom_index": len(atoms),
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start": start_sample / sample_rate,
                "end": end_sample / sample_rate,
                "duration": (end_sample - start_sample) / sample_rate,
                "origin": "vad_waveform",
            }
        )
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-ms", type=float, default=10.0)
    parser.add_argument("--vad-threshold", type=float, default=0.35)
    parser.add_argument("--max-atom-seconds", type=float, default=12.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio, sample_rate, channels = load_mono(args.audio)
    if not len(audio):
        parser.error("audio file contains no samples")
    frame_samples = max(1, round(sample_rate * args.frame_ms / 1000.0))
    rms_db = frame_rms_db(audio, frame_samples)
    finite_rms = rms_db[np.isfinite(rms_db)]
    noise_floor_db = float(np.percentile(finite_rms, 12.0))
    vad_mask, vad_regions = silero_mask(
        audio, sample_rate, frame_samples, args.vad_threshold
    )
    atoms = build_atoms(
        rms_db=rms_db,
        vad_mask=vad_mask,
        noise_floor_db=noise_floor_db,
        frame_samples=frame_samples,
        sample_rate=sample_rate,
        total_samples=len(audio),
        max_atom_seconds=args.max_atom_seconds,
    )
    result = {
        "schema_version": 1,
        "audio": str(args.audio.resolve()),
        "audio_sha256": sha256_file(args.audio),
        "sample_rate": sample_rate,
        "channels": channels,
        "samples": len(audio),
        "duration": len(audio) / sample_rate,
        "frame_samples": frame_samples,
        "frame_ms": args.frame_ms,
        "noise_floor_db": noise_floor_db,
        "activity_threshold_db": float(np.clip(noise_floor_db + 17.0, -56.0, -34.0)),
        "vad_threshold": args.vad_threshold,
        "vad_regions": vad_regions,
        "atoms": atoms,
    }
    write_json(args.output_dir / "analysis.json", result)
    print(
        json.dumps(
            {
                "duration": result["duration"],
                "sample_rate": sample_rate,
                "channels": channels,
                "noise_floor_db": noise_floor_db,
                "vad_regions": len(vad_regions),
                "atoms": len(atoms),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
