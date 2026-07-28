#!/usr/bin/env python3
"""Primary local ASR for whole recordings and waveform-derived speech atoms.

This script intentionally runs each atom without previous-text conditioning.
That prevents a whole-file decoder from silently normalizing away restarts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from mlx_whisper import transcribe
from mlx_whisper.audio import load_audio


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def clean_number(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def serialize_result(
    result: dict[str, Any],
    *,
    offset: float,
    clip_start: float,
    clip_end: float,
) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    all_words: list[dict[str, Any]] = []
    for raw_segment in result.get("segments", []):
        segment_start = max(
            clip_start,
            min(clip_end, offset + clean_number(raw_segment.get("start"))),
        )
        segment_end = max(
            segment_start,
            min(clip_end, offset + clean_number(raw_segment.get("end"))),
        )
        words: list[dict[str, Any]] = []
        for raw_word in raw_segment.get("words") or []:
            text = str(raw_word.get("word", "")).strip()
            if not text:
                continue
            start = max(
                clip_start,
                min(clip_end, offset + clean_number(raw_word.get("start"))),
            )
            end = max(
                start,
                min(clip_end, offset + clean_number(raw_word.get("end"))),
            )
            item = {
                "word": text,
                "start": start,
                "end": end,
                "probability": clean_number(raw_word.get("probability"), 0.5),
            }
            words.append(item)
            all_words.append(item)
        segments.append(
            {
                "start": segment_start,
                "end": segment_end,
                "text": str(raw_segment.get("text", "")).strip(),
                "avg_logprob": clean_number(raw_segment.get("avg_logprob"), -1.0),
                "no_speech_prob": clean_number(
                    raw_segment.get("no_speech_prob"), 0.0
                ),
                "words": words,
            }
        )

    # Whisper occasionally returns text without word timings for a tiny atom.
    # Keep it usable for candidate generation by assigning conservative,
    # uniformly spaced pseudo-word anchors. CTC/waveform refinement later
    # prevents these approximate anchors from becoming final cut boundaries.
    text = str(result.get("text", "")).strip()
    if text and not all_words:
        raw_words = re.findall(r"\S+", text)
        usable_start = clip_start
        usable_end = max(usable_start + 0.001, clip_end)
        step = (usable_end - usable_start) / max(1, len(raw_words))
        for index, word in enumerate(raw_words):
            all_words.append(
                {
                    "word": word,
                    "start": usable_start + index * step,
                    "end": usable_start + (index + 1) * step,
                    "probability": 0.25,
                    "approximate": True,
                }
            )
    return {
        "text": text,
        "segments": segments,
        "words": all_words,
        "language": str(result.get("language", "en")),
    }


def run_asr(
    waveform: np.ndarray,
    *,
    model: str,
    language: str,
    word_timestamps: bool,
    prompt: str | None,
) -> dict[str, Any]:
    return transcribe(
        waveform,
        path_or_hf_repo=model,
        language=language,
        temperature=0.0,
        condition_on_previous_text=False,
        word_timestamps=word_timestamps,
        initial_prompt=prompt,
        # None disables mlx-whisper's per-call tqdm.  The pipeline emits one
        # useful aggregate progress line every N atoms instead.
        verbose=None,
        hallucination_silence_threshold=0.5,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument(
        "--mode",
        choices=("source", "file"),
        default="source",
        help="'source' transcribes the whole recording and every analysis atom.",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-large-v3-turbo",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--prompt")
    parser.add_argument("--skip-whole", action="store_true")
    parser.add_argument("--max-atoms", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the atom checkpoint beside --output.",
    )
    args = parser.parse_args()

    analysis = read_json(args.analysis) if args.analysis else None
    # mlx-whisper versions differ here: some return a NumPy array and newer
    # releases return an mlx.core.array.  Converting explicitly keeps the
    # remainder of the pipeline version-independent.
    waveform = np.asarray(load_audio(str(args.audio), sr=16000), dtype=np.float32)
    duration = len(waveform) / 16000.0
    started = time.time()

    partial_path = args.output.with_name(args.output.name + ".partial")
    whole: dict[str, Any] | None = None
    atom_results: list[dict[str, Any]] = []
    if args.resume and partial_path.exists():
        checkpoint = read_json(partial_path)
        expected_audio = str(args.audio.resolve())
        if (
            checkpoint.get("audio") != expected_audio
            or checkpoint.get("model") != args.model
            or checkpoint.get("language") != args.language
        ):
            raise RuntimeError("ASR checkpoint does not match this audio/model run.")
        whole = checkpoint.get("whole")
        atom_results = list(checkpoint.get("atoms", []))

    def save_checkpoint() -> None:
        write_json(
            partial_path,
            {
                "schema_version": 1,
                "complete": False,
                "audio": str(args.audio.resolve()),
                "duration": duration,
                "model": args.model,
                "language": args.language,
                "whole": whole,
                "atoms": atom_results,
            },
        )

    if not args.skip_whole and whole is None:
        raw_whole = run_asr(
            waveform,
            model=args.model,
            language=args.language,
            word_timestamps=True,
            prompt=args.prompt,
        )
        whole = serialize_result(
            raw_whole,
            offset=0.0,
            clip_start=0.0,
            clip_end=duration,
        )
        save_checkpoint()

    if args.mode == "source":
        if analysis is None:
            parser.error("--analysis is required in source mode")
        atoms = list(analysis.get("atoms", []))
        if args.max_atoms is not None:
            atoms = atoms[: args.max_atoms]
        completed_atoms = {
            int(item["atom_index"]): item for item in atom_results
        }
        atom_results = []
        for position, atom in enumerate(atoms, 1):
            atom_index = int(atom["atom_index"])
            if atom_index in completed_atoms:
                atom_results.append(completed_atoms[atom_index])
                continue
            start = float(atom["start"])
            end = float(atom["end"])
            first = max(0, round(start * 16000))
            last = min(len(waveform), round(end * 16000))
            clip = waveform[first:last]
            if len(clip) < 800:
                serialized = {
                    "text": "",
                    "segments": [],
                    "words": [],
                    "language": args.language,
                }
            else:
                raw = run_asr(
                    clip,
                    model=args.model,
                    language=args.language,
                    word_timestamps=True,
                    prompt=args.prompt,
                )
                serialized = serialize_result(
                    raw,
                    offset=start,
                    clip_start=start,
                    clip_end=end,
                )
            segment_scores = [
                float(segment["avg_logprob"])
                for segment in serialized["segments"]
                if math.isfinite(float(segment["avg_logprob"]))
            ]
            atom_results.append(
                {
                    "atom_index": int(atom["atom_index"]),
                    "start_sample": int(atom["start_sample"]),
                    "end_sample": int(atom["end_sample"]),
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "text": serialized["text"],
                    "segments": serialized["segments"],
                    "words": serialized["words"],
                    "mean_logprob": (
                        sum(segment_scores) / len(segment_scores)
                        if segment_scores
                        else -10.0
                    ),
                }
            )
            if args.progress_every and (
                position % args.progress_every == 0 or position == len(atoms)
            ):
                save_checkpoint()
                elapsed = time.time() - started
                print(
                    f"transcribed {position}/{len(atoms)} atoms in {elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )

    result = {
        "schema_version": 1,
        "audio": str(args.audio.resolve()),
        "duration": duration,
        "model": args.model,
        "language": args.language,
        "condition_on_previous_text": False,
        "temperature": 0.0,
        "whole": whole,
        "atoms": atom_results,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output, result)
    if partial_path.exists():
        partial_path.unlink()
    print(
        json.dumps(
            {
                "duration": duration,
                "model": args.model,
                "whole_words": len(whole["words"]) if whole else 0,
                "atoms": len(atom_results),
                "atom_words": sum(len(atom["words"]) for atom in atom_results),
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
