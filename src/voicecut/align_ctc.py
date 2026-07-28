#!/usr/bin/env python3
"""Force-align selected narration hypotheses with WhisperX's CTC aligner."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import whisperx


def serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable(item) for item in value]
    if isinstance(value, tuple):
        return [serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    segments = [
        {
            "phrase_index": int(segment["phrase_index"]),
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": str(segment["text"]),
        }
        for segment in source.get("segments", [])
    ]
    if not segments:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"segments": [], "word_segments": []}, indent=2),
            encoding="utf-8",
        )
        return

    audio, _ = librosa.load(args.audio, sr=16000, mono=True)
    model, metadata = whisperx.load_align_model(
        language_code=args.language,
        device=args.device,
    )
    # WhisperX is allowed to split one requested segment at punctuation.  A
    # single bulk call therefore cannot be mapped back by list position: one
    # split would shift every later phrase.  Align each requested phrase as its
    # own group and aggregate all returned subsegments under the explicit ID.
    partial_path = args.output.with_name(args.output.name + ".partial")
    aligned_segments: list[dict[str, Any]] = []
    all_words: list[dict[str, Any]] = []
    if args.resume and partial_path.exists():
        checkpoint = json.loads(partial_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("audio") != str(args.audio.resolve())
            or checkpoint.get("input") != str(args.input.resolve())
            or checkpoint.get("language") != args.language
        ):
            raise RuntimeError("CTC checkpoint does not match this alignment run.")
        aligned_segments = list(checkpoint.get("segments", []))
        all_words = list(checkpoint.get("word_segments", []))
    completed_ids = {
        int(segment["phrase_index"]) for segment in aligned_segments
    }
    for position, requested in enumerate(segments, 1):
        if requested["phrase_index"] in completed_ids:
            continue
        hypothesis = {
            "start": requested["start"],
            "end": requested["end"],
            "text": requested["text"],
        }
        result = whisperx.align(
            [hypothesis],
            model,
            metadata,
            audio,
            args.device,
            return_char_alignments=True,
            print_progress=False,
        )
        subsegments = list(result.get("segments", []))
        words: list[dict[str, Any]] = []
        characters: list[dict[str, Any]] = []
        for subsegment in subsegments:
            words.extend(
                word
                for word in subsegment.get("words", [])
                if "start" in word and "end" in word
            )
            characters.extend(subsegment.get("chars", []))
        valid_words = [
            word
            for word in words
            if math.isfinite(float(word["start"]))
            and math.isfinite(float(word["end"]))
            and float(word["end"]) >= float(word["start"])
        ]
        aligned_segments.append(
            {
                "phrase_index": requested["phrase_index"],
                "input_start": requested["start"],
                "input_end": requested["end"],
                "input_text": requested["text"],
                "start": (
                    min(float(word["start"]) for word in valid_words)
                    if valid_words
                    else None
                ),
                "end": (
                    max(float(word["end"]) for word in valid_words)
                    if valid_words
                    else None
                ),
                "text": " ".join(
                    str(segment.get("text", "")).strip()
                    for segment in subsegments
                    if str(segment.get("text", "")).strip()
                ),
                "words": valid_words,
                "chars": characters,
                "subsegment_count": len(subsegments),
            }
        )
        for word in valid_words:
            all_words.append(
                {
                    "phrase_index": requested["phrase_index"],
                    **word,
                }
            )
        if position % 25 == 0 or position == len(segments):
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(
                json.dumps(
                    serializable(
                        {
                            "schema_version": 2,
                            "complete": False,
                            "audio": str(args.audio.resolve()),
                            "input": str(args.input.resolve()),
                            "language": args.language,
                            "segments": aligned_segments,
                            "word_segments": all_words,
                        }
                    ),
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(
                f"aligned {position}/{len(segments)} phrases",
                file=sys.stderr,
                flush=True,
            )
    aligned = {
        "schema_version": 2,
        "segments": aligned_segments,
        "word_segments": all_words,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(serializable(aligned), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if partial_path.exists():
        partial_path.unlink()
    print(
        json.dumps(
            {
                "segments": len(aligned.get("segments", [])),
                "words": len(aligned.get("word_segments", [])),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
