#!/usr/bin/env python3
"""Independent Faster-Whisper transcript for final-output QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from faster_whisper import WhisperModel


def segment_rejection_reasons(segment: dict[str, object]) -> list[str]:
    """Reject decoder text whose timing/confidence is physically implausible."""

    words = list(segment.get("words", []))
    if not words:
        return ["no_words"]
    duration = max(
        0.0,
        float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)),
    )
    word_durations = [
        max(0.0, float(word["end"]) - float(word["start"]))
        for word in words
    ]
    degenerate_fraction = sum(value <= 0.015 for value in word_durations) / len(words)
    mean_probability = sum(float(word["probability"]) for word in words) / len(words)
    words_per_second = len(words) / max(duration, 0.01)
    reasons: list[str] = []
    if len(words) >= 4 and (
        degenerate_fraction >= 0.45 or words_per_second > 10.0
    ):
        reasons.append("implausible_word_timing")
    if len(words) >= 3 and duration < 0.12:
        reasons.append("implausibly_short_segment")
    if (
        len(words) >= 3
        and float(segment.get("avg_logprob", 0.0)) < -0.70
        and float(segment.get("no_speech_prob", 0.0)) > 0.20
        and mean_probability < 0.70
    ):
        reasons.append("low_confidence_non_speech")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require an already cached model instead of downloading it.",
    )
    args = parser.parse_args()

    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
        local_files_only=args.local_files_only,
    )
    segments, info = model.transcribe(
        str(args.audio),
        language=args.language,
        beam_size=5,
        best_of=5,
        condition_on_previous_text=False,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 250,
            "speech_pad_ms": 100,
        },
        temperature=0.0,
    )
    serialized = []
    rejected = []
    words = []
    for segment in segments:
        segment_words = [
            {
                "start": float(word.start),
                "end": float(word.end),
                "word": str(word.word),
                "probability": float(word.probability),
            }
            for word in (segment.words or [])
        ]
        item = {
            "start": float(segment.start),
            "end": float(segment.end),
            "text": str(segment.text),
            "avg_logprob": float(segment.avg_logprob),
            "no_speech_prob": float(segment.no_speech_prob),
            "words": segment_words,
        }
        reasons = segment_rejection_reasons(item)
        if reasons:
            rejected.append({**item, "rejection_reasons": reasons})
            continue
        words.extend(segment_words)
        serialized.append(item)
    result = {
        "schema_version": 1,
        "model": args.model,
        "language": info.language,
        "duration": float(info.duration),
        "whole": {
            "text": "".join(segment["text"] for segment in serialized),
            "segments": serialized,
            "words": words,
            "language": info.language,
        },
        "rejected_segments": rejected,
        "atoms": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "model": args.model,
                "segments": len(serialized),
                "rejected_segments": len(rejected),
                "words": len(words),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
