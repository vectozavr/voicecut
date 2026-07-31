#!/usr/bin/env python3
"""Expose high-confidence speech that Whisper collapsed inside a word span.

Whisper is the primary transcript. Raw greedy CTC is used only when the
existing acoustic-insertion detector proves that a selected atom contains a
spoken retry hidden by Whisper's language-model decoding. The resulting word
ledger keeps every physical occurrence so the semantic planner can decide
which occurrence to retain.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .align_ctc import align_expected_to_greedy, normalized_words
from .common import read_json, sha256_file, write_json


MIN_RETRY_CONFIDENCE = 0.75
MIN_CONTEXTUAL_SUBSTITUTION_SCORE = 0.65
MAX_CONTEXTUAL_SUBSTITUTION_WORDS = 3


class CtcEnrichmentError(RuntimeError):
    """The hidden-retry evidence cannot safely enrich the transcript."""


def build_alignment_input(transcript: dict[str, Any]) -> dict[str, Any]:
    atoms = transcript.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        raise CtcEnrichmentError("source transcript contains no acoustic atoms")
    segments: list[dict[str, Any]] = []
    seen_atom_indices: set[int] = set()
    for position, atom in enumerate(atoms):
        if not isinstance(atom, dict):
            raise CtcEnrichmentError(f"atom {position} is not an object")
        atom_index = atom.get("atom_index")
        start = atom.get("start")
        end = atom.get("end")
        text = atom.get("text")
        if (
            type(atom_index) is not int
            or type(start) not in {int, float}
            or type(end) not in {int, float}
            or not isinstance(text, str)
            or not text.strip()
            or float(end) <= float(start)
        ):
            raise CtcEnrichmentError(f"atom {position} has invalid alignment input")
        if atom_index in seen_atom_indices:
            raise CtcEnrichmentError(
                f"source transcript repeats atom_index {atom_index}"
            )
        seen_atom_indices.add(atom_index)
        segments.append(
            {
                "phrase_index": atom_index,
                "start": float(start),
                "end": float(end),
                "text": text.strip(),
            }
        )
    return {"schema_version": 1, "segments": segments}


def _validated_hidden_retries(
    segment: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_insertions = segment.get("acoustic_insertions")
    if not isinstance(raw_insertions, list):
        return []
    valid: list[dict[str, Any]] = []
    previous_end = -1.0
    for insertion in raw_insertions:
        if not isinstance(insertion, dict):
            continue
        confidence = insertion.get("confidence")
        start = insertion.get("safe_edit_start")
        end = insertion.get("safe_edit_end")
        words = insertion.get("words")
        if (
            insertion.get("type") != "spoken_retry"
            or insertion.get("reason") != "greedy_ctc_restart_before_selected_take"
            or type(confidence) not in {int, float}
            or float(confidence) < MIN_RETRY_CONFIDENCE
            or type(start) not in {int, float}
            or type(end) not in {int, float}
            or float(end) <= float(start)
            or float(start) < previous_end
            or not isinstance(words, list)
            or not words
            or not isinstance(insertion.get("left_anchor"), dict)
            or not isinstance(insertion.get("right_anchor"), dict)
        ):
            continue
        valid.append(copy.deepcopy(insertion))
        previous_end = float(end)
    return valid


def _enriched_greedy_words(
    *,
    expected_text: str,
    greedy_words: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    matches = align_expected_to_greedy(expected_text, greedy_words)
    expected_by_greedy = {
        int(match["greedy_index"]): str(match["expected_word"]) for match in matches
    }
    lexical_source_by_greedy = {
        int(match["greedy_index"]): "whisper_expected_match" for match in matches
    }
    expected_words = normalized_words(expected_text)
    # Raw CTC is deliberately decoded without a language model so it exposes
    # repeated physical occurrences, but short acoustically weak words may be
    # misspelled (for example a function word between two secure anchors).
    # When two strong ordered anchors enclose the same small number of expected
    # and observed words, restore the primary Whisper labels while retaining
    # the raw CTC occurrence times.  Unequal cardinality is never relabelled:
    # that is precisely how a hidden retry remains visible to the planner.
    for left, right in zip(matches, matches[1:]):
        expected_start = int(left["expected_index"]) + 1
        expected_end = int(right["expected_index"])
        greedy_start = int(left["greedy_index"]) + 1
        greedy_end = int(right["greedy_index"])
        expected_gap = expected_words[expected_start:expected_end]
        greedy_gap = list(greedy_words[greedy_start:greedy_end])
        if (
            not expected_gap
            or len(expected_gap) != len(greedy_gap)
            or len(expected_gap) > MAX_CONTEXTUAL_SUBSTITUTION_WORDS
            or float(left["lexical_score"]) < 0.999
            or float(right["lexical_score"]) < 0.999
        ):
            continue
        scores: list[float] = []
        for raw in greedy_gap:
            score = raw.get("score")
            if type(score) not in {int, float}:
                scores = []
                break
            scores.append(float(score))
        if not scores or min(scores) < MIN_CONTEXTUAL_SUBSTITUTION_SCORE:
            continue
        for offset, expected_word in enumerate(expected_gap):
            greedy_index = greedy_start + offset
            expected_by_greedy[greedy_index] = expected_word
            lexical_source_by_greedy[greedy_index] = (
                "whisper_contextually_grounded_substitution"
            )

    enriched: list[dict[str, Any]] = []
    previous_end = -1.0
    for index, raw in enumerate(greedy_words):
        if not isinstance(raw, dict):
            raise CtcEnrichmentError("raw CTC word is not an object")
        observed = str(raw.get("word", "")).strip()
        start = raw.get("start")
        end = raw.get("end")
        score = raw.get("score")
        if (
            not observed
            or type(start) not in {int, float}
            or type(end) not in {int, float}
            or float(end) <= float(start)
            or float(start) < previous_end
        ):
            raise CtcEnrichmentError(
                f"raw CTC word {index} has invalid chronological geometry"
            )
        normalized = expected_by_greedy.get(index, observed)
        enriched.append(
            {
                "word": normalized,
                "start": float(start),
                "end": float(end),
                "probability": (float(score) if type(score) in {int, float} else None),
                "ctc_observed_word": observed,
                "ctc_expected_match": expected_by_greedy.get(index),
                "ctc_lexical_source": lexical_source_by_greedy.get(
                    index,
                    "raw_unmatched_occurrence",
                ),
            }
        )
        previous_end = float(end)
    return enriched


def enrich_transcript(
    *,
    transcript: dict[str, Any],
    alignment: dict[str, Any],
    alignment_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    atoms = transcript.get("atoms")
    aligned_segments = alignment.get("segments")
    if not isinstance(atoms, list) or not atoms:
        raise CtcEnrichmentError("source transcript contains no atoms")
    if not isinstance(aligned_segments, list):
        raise CtcEnrichmentError("CTC alignment contains no segment ledger")
    by_index: dict[int, dict[str, Any]] = {}
    for segment in aligned_segments:
        if (
            not isinstance(segment, dict)
            or type(segment.get("phrase_index")) is not int
        ):
            raise CtcEnrichmentError("CTC alignment segment has no phrase index")
        phrase_index = int(segment["phrase_index"])
        if phrase_index in by_index:
            raise CtcEnrichmentError("CTC alignment repeats a phrase index")
        by_index[phrase_index] = segment

    enriched = copy.deepcopy(transcript)
    output_atoms = enriched["atoms"]
    recovered: list[dict[str, Any]] = []
    for position, atom in enumerate(output_atoms):
        if not isinstance(atom, dict) or type(atom.get("atom_index")) is not int:
            raise CtcEnrichmentError(f"source atom {position} is malformed")
        atom_index = int(atom["atom_index"])
        segment = by_index.get(atom_index)
        if segment is None:
            raise CtcEnrichmentError(f"CTC alignment omitted source atom {atom_index}")
        retries = _validated_hidden_retries(segment)
        if not retries:
            atom["ctc_enrichment"] = {
                "status": "unchanged_no_hidden_retry",
                "hidden_retries": [],
            }
            continue
        greedy_words = segment.get("greedy_ctc_words")
        if not isinstance(greedy_words, list) or not greedy_words:
            raise CtcEnrichmentError(
                f"atom {atom_index} has retry evidence but no raw CTC words"
            )
        expanded_words = _enriched_greedy_words(
            expected_text=str(atom.get("text", "")),
            greedy_words=greedy_words,
        )
        atom["original_whisper_text"] = atom.get("text")
        atom["original_whisper_words"] = copy.deepcopy(atom.get("words", []))
        atom["text"] = " ".join(word["word"] for word in expanded_words)
        atom["words"] = expanded_words
        atom["segments"] = [
            {
                "start": expanded_words[0]["start"],
                "end": expanded_words[-1]["end"],
                "text": atom["text"],
                "words": copy.deepcopy(expanded_words),
                "decode_strategy": "raw_ctc_hidden_retry_expansion_v1",
            }
        ]
        atom["ctc_enrichment"] = {
            "status": "expanded_hidden_retry",
            "hidden_retries": retries,
            "original_word_count": len(atom["original_whisper_words"]),
            "expanded_word_count": len(expanded_words),
        }
        recovered.append(
            {
                "atom_index": atom_index,
                "original_whisper_text": atom["original_whisper_text"],
                "expanded_text": atom["text"],
                "hidden_retries": retries,
            }
        )

    enriched["engine"] = "mlx_whisper_with_raw_ctc_hidden_retry_recovery"
    enriched["source_decode_strategy"] = (
        "whisper_primary_plus_gated_raw_ctc_insertions_v1"
    )
    enriched["ctc_enrichment"] = {
        "schema_version": 1,
        "alignment": str(alignment_path.resolve()) if alignment_path else None,
        "alignment_sha256": (
            sha256_file(alignment_path)
            if alignment_path is not None and alignment_path.is_file()
            else None
        ),
        "minimum_retry_confidence": MIN_RETRY_CONFIDENCE,
        "atoms_examined": len(output_atoms),
        "atoms_expanded": len(recovered),
        "hidden_retries_recovered": sum(
            len(item["hidden_retries"]) for item in recovered
        ),
        "recovered": recovered,
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        **copy.deepcopy(enriched["ctc_enrichment"]),
    }
    return enriched, report


def run_enrichment(
    *,
    audio_path: Path,
    transcript_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    audio_path = audio_path.resolve()
    transcript_path = transcript_path.resolve()
    output_dir = output_dir.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    if not transcript_path.is_file():
        raise FileNotFoundError(transcript_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"CTC enrichment output must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript = read_json(transcript_path)
    if not isinstance(transcript, dict):
        raise CtcEnrichmentError("source transcript root must be an object")
    expected_audio_sha = transcript.get("audio_sha256")
    if not isinstance(expected_audio_sha, str):
        raise CtcEnrichmentError("source transcript has no audio SHA-256")
    if sha256_file(audio_path) != expected_audio_sha:
        raise CtcEnrichmentError("audio does not match the source transcript")

    alignment_input_path = output_dir / "ctc_alignment_input.json"
    alignment_path = output_dir / "ctc_alignment.json"
    alignment_log_path = output_dir / "ctc_alignment.log"
    write_json(alignment_input_path, build_alignment_input(transcript))
    command = [
        sys.executable,
        "-m",
        "voicecut.align_ctc",
        "--audio",
        str(audio_path),
        "--input",
        str(alignment_input_path),
        "--output",
        str(alignment_path),
        "--language",
        "en",
        "--device",
        "cpu",
    ]
    environment = os.environ.copy()
    process = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=environment,
    )
    alignment_log_path.write_text(
        "$ "
        + " ".join(command)
        + "\n\n[stdout]\n"
        + process.stdout
        + "\n[stderr]\n"
        + process.stderr,
        encoding="utf-8",
    )
    if process.returncode or not alignment_path.is_file():
        raise RuntimeError(f"raw CTC enrichment failed; see {alignment_log_path}")
    alignment = read_json(alignment_path)
    if not isinstance(alignment, dict):
        raise CtcEnrichmentError("CTC alignment root must be an object")
    enriched, report = enrich_transcript(
        transcript=transcript,
        alignment=alignment,
        alignment_path=alignment_path,
    )
    enriched_path = output_dir / "source_transcript_ctc_enriched.json"
    report_path = output_dir / "ctc_enrichment_report.json"
    write_json(enriched_path, enriched)
    write_json(report_path, report)
    return enriched_path, report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover high-confidence spoken retries that Whisper collapsed "
            "inside a word span, producing a physical occurrence ledger for "
            "the semantic planner."
        )
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    enriched_path, report_path = run_enrichment(
        audio_path=args.audio,
        transcript_path=args.transcript,
        output_dir=args.output_dir,
    )
    report = read_json(report_path)
    print("\nCTC HIDDEN-RETRY ENRICHMENT COMPLETE")
    print(f"atoms examined: {report['atoms_examined']}")
    print(f"atoms expanded: {report['atoms_expanded']}")
    print(f"hidden retries recovered: {report['hidden_retries_recovered']}")
    print(f"enriched transcript: {enriched_path}")
    print(f"report: {report_path}")
    print(
        json.dumps(
            {
                "status": report["status"],
                "transcript": str(enriched_path),
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
