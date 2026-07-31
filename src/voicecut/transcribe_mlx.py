#!/usr/bin/env python3
"""Primary local ASR for whole recordings and waveform-derived speech atoms.

This script intentionally runs each atom without previous-text conditioning.
That prevents a whole-file decoder from silently normalizing away restarts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ARTIFACT_ROLES = (
    "source_primary",
    "final_primary",
    "source_independent",
    "final_independent",
)
ENGINE = "mlx-whisper"
SAMPLE_RATE = 16000
SOURCE_ATOM_STRATEGY = "analysis_acoustic_atoms_v1"
WHOLE_FILE_STRATEGY = "whole_file_v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def mlx_decode_config(
    *,
    mode: str,
    skip_whole: bool,
    max_atoms: int | None,
) -> dict[str, Any]:
    """Return every setting that materially controls this transcription run."""

    return {
        "sample_rate": SAMPLE_RATE,
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "hallucination_silence_threshold": 0.5,
        "mode": mode,
        "skip_whole": skip_whole,
        "max_atoms": max_atoms,
        "source_chunking": (
            SOURCE_ATOM_STRATEGY if mode == "source" else WHOLE_FILE_STRATEGY
        ),
    }


def build_provenance(
    audio: Path,
    *,
    artifact_role: str | None,
    model: str,
    language: str,
    prompt: str | None,
    decode_config: dict[str, Any],
    analysis: Path | None = None,
) -> dict[str, Any]:
    return {
        "audio": str(audio.resolve()),
        "audio_sha256": sha256_file(audio),
        "artifact_role": artifact_role,
        "engine": ENGINE,
        "model": model,
        "language": language,
        "decode_config": decode_config,
        "prompt_sha256": sha256_optional_text(prompt),
        "source_decode_strategy": (
            SOURCE_ATOM_STRATEGY if analysis is not None else WHOLE_FILE_STRATEGY
        ),
        "analysis": str(analysis.resolve()) if analysis is not None else None,
        "analysis_sha256": (sha256_file(analysis) if analysis is not None else None),
    }


def checkpoint_identity(
    provenance: dict[str, Any],
    *,
    mode: str,
    analysis_sha256: str | None,
) -> dict[str, Any]:
    """Select immutable run inputs that a partial checkpoint must match."""

    return {
        "audio": provenance["audio"],
        "audio_sha256": provenance["audio_sha256"],
        "artifact_role": provenance["artifact_role"],
        "engine": provenance["engine"],
        "model": provenance["model"],
        "language": provenance["language"],
        "decode_config": provenance["decode_config"],
        "prompt_sha256": provenance["prompt_sha256"],
        "mode": mode,
        "analysis_sha256": analysis_sha256,
    }


def validate_checkpoint_identity(
    checkpoint: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    mismatches = [
        field
        for field, expected_value in expected.items()
        if field not in checkpoint or checkpoint[field] != expected_value
    ]
    if mismatches:
        joined = ", ".join(mismatches)
        raise RuntimeError(f"ASR checkpoint does not match this run ({joined}).")


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
                "no_speech_prob": clean_number(raw_segment.get("no_speech_prob"), 0.0),
                "words": words,
            }
        )

    # Whisper occasionally returns text without word timings for a tiny atom.
    # Keep it usable for source grounding by assigning conservative, uniformly
    # spaced pseudo-word anchors. The renderer protects the entire selected
    # word envelope and only moves boundaries outward from these anchors.
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
    # Keep the optional MLX runtime out of module import so metadata and
    # checkpoint helpers can be tested in a regular Python environment.
    from mlx_whisper import transcribe as mlx_transcribe

    return mlx_transcribe(
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
    parser.add_argument("--artifact-role", choices=ARTIFACT_ROLES)
    parser.add_argument("--skip-whole", action="store_true")
    parser.add_argument("--max-atoms", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the atom checkpoint beside --output.",
    )
    args = parser.parse_args()

    decode_config = mlx_decode_config(
        mode=args.mode,
        skip_whole=args.skip_whole,
        max_atoms=args.max_atoms,
    )
    provenance = build_provenance(
        args.audio,
        artifact_role=args.artifact_role,
        model=args.model,
        language=args.language,
        prompt=args.prompt,
        decode_config=decode_config,
        analysis=args.analysis,
    )
    analysis_sha256 = sha256_file(args.analysis) if args.analysis else None
    expected_checkpoint = checkpoint_identity(
        provenance,
        mode=args.mode,
        analysis_sha256=analysis_sha256,
    )
    analysis = read_json(args.analysis) if args.analysis else None
    from mlx_whisper.audio import load_audio

    # mlx-whisper versions differ here: some return a NumPy array and newer
    # releases return an mlx.core.array.  Converting explicitly keeps the
    # remainder of the pipeline version-independent.
    waveform = np.asarray(
        load_audio(str(args.audio), sr=SAMPLE_RATE),
        dtype=np.float32,
    )
    duration = len(waveform) / float(SAMPLE_RATE)
    started = time.time()

    partial_path = args.output.with_name(args.output.name + ".partial")
    whole: dict[str, Any] | None = None
    atom_results: list[dict[str, Any]] = []
    if args.resume and partial_path.exists():
        checkpoint = read_json(partial_path)
        validate_checkpoint_identity(checkpoint, expected_checkpoint)
        whole = checkpoint.get("whole")
        atom_results = list(checkpoint.get("atoms", []))

    def save_checkpoint() -> None:
        write_json(
            partial_path,
            {
                "schema_version": 1,
                "complete": False,
                **expected_checkpoint,
                "duration": duration,
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
        completed_atoms = {int(item["atom_index"]): item for item in atom_results}
        atom_results = []
        for position, atom in enumerate(atoms, 1):
            atom_index = int(atom["atom_index"])
            if atom_index in completed_atoms:
                atom_results.append(completed_atoms[atom_index])
                continue
            start = float(atom["start"])
            end = float(atom["end"])
            first = max(0, round(start * SAMPLE_RATE))
            last = min(len(waveform), round(end * SAMPLE_RATE))
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
                    "decode_strategy": SOURCE_ATOM_STRATEGY,
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
        "schema_version": 2,
        **provenance,
        "duration": duration,
        "condition_on_previous_text": False,
        "temperature": 0.0,
        "preferred_evidence": "atoms" if args.mode == "source" else "whole",
        "analysis_atom_count": (
            len(list(analysis.get("atoms", [])))
            if args.mode == "source" and isinstance(analysis, dict)
            else 0
        ),
        "decoded_atom_count": len(atom_results),
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
