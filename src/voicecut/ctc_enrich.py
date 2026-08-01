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
import math
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
MIN_HALLUCINATED_SUFFIX_WORDS = 2
MIN_HALLUCINATED_SUFFIX_ZERO_DURATION_WORDS = 2
MIN_CTC_TRAILING_EMPTY_SECONDS = 0.10
SOURCE_DECODE_STRATEGY = "whisper_primary_plus_gated_raw_ctc_insertions_v2"
PASSTHROUGH_STATUS = "degraded_whisper_primary_passthrough"
# ``serialize_result`` clamps word timestamps to their atom boundaries.  JSON
# float round-trips can differ at the sub-microsecond level, so containment and
# ordering validation allows only this numerical tolerance, never an acoustic
# overlap large enough to affect a cut.
GEOMETRY_ROUNDING_TOLERANCE_SECONDS = 1e-6


class CtcEnrichmentError(RuntimeError):
    """The hidden-retry evidence cannot safely enrich the transcript."""


def _validated_atom_source(
    atom: Any,
    *,
    position: int,
) -> tuple[int, float, float, str | None, str]:
    """Validate one primary atom and return its CTC text, if it has words.

    An empty Whisper hypothesis is expected occasionally on long recordings.
    It is not structural corruption.  If word records are available, they are
    a more stable source for the CTC prompt; otherwise the atom is explicitly
    skipped.  Timestamp corruption remains fatal because silently accepting it
    could associate acoustic evidence with the wrong source region.
    """

    if not isinstance(atom, dict):
        raise CtcEnrichmentError(f"atom {position} is not an object")
    atom_index = atom.get("atom_index")
    start = atom.get("start")
    end = atom.get("end")
    if type(atom_index) is not int or atom_index < 0:
        raise CtcEnrichmentError(f"atom {position} has no integer atom_index")
    if type(start) not in {int, float} or type(end) not in {int, float}:
        raise CtcEnrichmentError(f"atom {position} has invalid timestamp geometry")
    start_value = float(start)
    end_value = float(end)
    if (
        not math.isfinite(start_value)
        or not math.isfinite(end_value)
        or start_value < 0.0
        or end_value <= start_value
    ):
        raise CtcEnrichmentError(f"atom {position} has invalid timestamp geometry")
    if "start_sample" in atom or "end_sample" in atom:
        start_sample = atom.get("start_sample")
        end_sample = atom.get("end_sample")
        if (
            type(start_sample) is not int
            or type(end_sample) is not int
            or start_sample < 0
            or end_sample <= start_sample
        ):
            raise CtcEnrichmentError(f"atom {position} has invalid sample geometry")
    if "duration" in atom:
        duration = atom.get("duration")
        if (
            type(duration) not in {int, float}
            or not math.isfinite(float(duration))
            or float(duration) <= 0.0
        ):
            raise CtcEnrichmentError(f"atom {position} has invalid duration")

    raw_text = atom.get("text", "")
    if raw_text is None:
        raw_text = ""
    if not isinstance(raw_text, str):
        raise CtcEnrichmentError(f"atom {position} has non-text transcription")

    raw_words = atom.get("words", [])
    if raw_words is None:
        raw_words = []
    if not isinstance(raw_words, list):
        raise CtcEnrichmentError(f"atom {position} has an invalid word ledger")
    derived_words: list[str] = []
    previous_word_end: float | None = None
    for word_position, word in enumerate(raw_words):
        if not isinstance(word, dict):
            raise CtcEnrichmentError(
                f"atom {atom_index} word {word_position} is not an object"
            )
        word_start = word.get("start")
        word_end = word.get("end")
        if type(word_start) not in {int, float} or type(word_end) not in {int, float}:
            raise CtcEnrichmentError(
                f"atom {atom_index} word {word_position} has invalid geometry"
            )
        word_start_value = float(word_start)
        word_end_value = float(word_end)
        # Zero-duration Whisper words are retained here.  The conservative CTC
        # suffix validator already handles that known decoder artifact.
        if (
            not math.isfinite(word_start_value)
            or not math.isfinite(word_end_value)
            or word_end_value < word_start_value
        ):
            raise CtcEnrichmentError(
                f"atom {atom_index} word {word_position} has invalid geometry"
            )
        if (
            word_start_value < start_value - GEOMETRY_ROUNDING_TOLERANCE_SECONDS
            or word_end_value > end_value + GEOMETRY_ROUNDING_TOLERANCE_SECONDS
        ):
            raise CtcEnrichmentError(
                f"atom {atom_index} word {word_position} lies outside its atom"
            )
        if (
            previous_word_end is not None
            and word_start_value
            < previous_word_end - GEOMETRY_ROUNDING_TOLERANCE_SECONDS
        ):
            raise CtcEnrichmentError(
                f"atom {atom_index} word ledger is not chronological"
            )
        previous_word_end = word_end_value
        value = word.get("word", word.get("text", ""))
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise CtcEnrichmentError(
                f"atom {atom_index} word {word_position} has non-text content"
            )
        cleaned = value.strip()
        if cleaned and normalized_words(cleaned):
            derived_words.append(cleaned)

    cleaned_text = raw_text.strip()
    if cleaned_text and normalized_words(cleaned_text):
        return atom_index, start_value, end_value, cleaned_text, "atom_text"
    if derived_words:
        return (
            atom_index,
            start_value,
            end_value,
            " ".join(derived_words),
            "derived_from_word_ledger",
        )
    return atom_index, start_value, end_value, None, "no_lexical_content"


def build_alignment_input(transcript: dict[str, Any]) -> dict[str, Any]:
    atoms = transcript.get("atoms")
    if not isinstance(atoms, list):
        raise CtcEnrichmentError("source transcript has no acoustic atom ledger")
    segments: list[dict[str, Any]] = []
    skipped_segments: list[dict[str, Any]] = []
    seen_atom_indices: set[int] = set()
    previous_atom_index: int | None = None
    previous_atom_end: float | None = None
    previous_end_sample: int | None = None
    for position, atom in enumerate(atoms):
        atom_index, start, end, text, text_source = _validated_atom_source(
            atom,
            position=position,
        )
        if atom_index in seen_atom_indices:
            raise CtcEnrichmentError(
                f"source transcript repeats atom_index {atom_index}"
            )
        if previous_atom_index is not None and atom_index <= previous_atom_index:
            raise CtcEnrichmentError(
                "source transcript atom IDs are not chronological: "
                f"{atom_index} follows {previous_atom_index}"
            )
        if (
            previous_atom_end is not None
            and start < previous_atom_end - GEOMETRY_ROUNDING_TOLERANCE_SECONDS
        ):
            raise CtcEnrichmentError(
                "source transcript atoms overlap or are not chronological: "
                f"atom {atom_index} starts at {start} before {previous_atom_end}"
            )
        start_sample = atom.get("start_sample") if isinstance(atom, dict) else None
        end_sample = atom.get("end_sample") if isinstance(atom, dict) else None
        if (
            previous_end_sample is not None
            and type(start_sample) is int
            and start_sample < previous_end_sample
        ):
            raise CtcEnrichmentError(
                "source transcript atom sample ranges overlap or are not "
                f"chronological at atom {atom_index}"
            )
        seen_atom_indices.add(atom_index)
        previous_atom_index = atom_index
        previous_atom_end = end
        previous_end_sample = end_sample if type(end_sample) is int else None
        if text is None:
            skipped_segments.append(
                {
                    "phrase_index": atom_index,
                    "input_start": start,
                    "input_end": end,
                    "reason": "no_lexical_content",
                }
            )
            continue
        segments.append(
            {
                "phrase_index": atom_index,
                "start": start,
                "end": end,
                "text": text,
                **({"text_source": text_source} if text_source != "atom_text" else {}),
            }
        )
    result: dict[str, Any] = {"schema_version": 1, "segments": segments}
    if skipped_segments:
        result["skipped_segments"] = skipped_segments
    return result


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


def _hallucinated_whisper_suffix_start(
    *,
    atom: dict[str, Any],
    segment: dict[str, Any],
) -> int | None:
    """Return a safely unsupported trailing Whisper word index, if proven.

    This is deliberately narrower than generic transcript voting. It only
    catches the characteristic tiny-atom failure where Whisper emits several
    trailing words at a collapsed timestamp, while raw CTC ends on an ordered
    prefix and leaves real waveform time after its last supported word.
    """

    whisper_words = atom.get("words")
    greedy_words = segment.get("greedy_ctc_words")
    if (
        not isinstance(whisper_words, list)
        or not whisper_words
        or not isinstance(greedy_words, list)
        or len(greedy_words) < 2
        or _validated_hidden_retries(segment)
    ):
        return None
    expected_words = normalized_words(str(atom.get("text", "")))
    if len(expected_words) != len(whisper_words):
        return None
    matches = align_expected_to_greedy(str(atom.get("text", "")), greedy_words)
    if len(matches) < max(2, len(greedy_words) - 1):
        return None
    final_match = matches[-1]
    if int(final_match["greedy_index"]) != len(greedy_words) - 1:
        return None
    prefix_end = int(final_match["expected_index"]) + 1
    suffix = whisper_words[prefix_end:]
    if len(suffix) < MIN_HALLUCINATED_SUFFIX_WORDS:
        return None
    zero_duration = 0
    for word in suffix:
        if not isinstance(word, dict):
            return None
        start = word.get("start")
        end = word.get("end")
        if type(start) not in {int, float} or type(end) not in {int, float}:
            return None
        if float(end) - float(start) <= 1e-4:
            zero_duration += 1
    if zero_duration < max(
        MIN_HALLUCINATED_SUFFIX_ZERO_DURATION_WORDS,
        len(suffix) // 2,
    ):
        return None
    greedy_end = greedy_words[-1].get("end")
    atom_end = atom.get("end")
    if (
        type(greedy_end) not in {int, float}
        or type(atom_end) not in {int, float}
        or float(atom_end) - float(greedy_end) < MIN_CTC_TRAILING_EMPTY_SECONDS
    ):
        return None
    return prefix_end


def enrich_transcript(
    *,
    transcript: dict[str, Any],
    alignment: dict[str, Any],
    alignment_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    atoms = transcript.get("atoms")
    aligned_segments = alignment.get("segments")
    if not isinstance(atoms, list):
        raise CtcEnrichmentError("source transcript contains no atom ledger")
    if not isinstance(aligned_segments, list):
        raise CtcEnrichmentError("CTC alignment contains no segment ledger")

    alignment_input = build_alignment_input(transcript)
    expected_segment_ids = {
        int(segment["phrase_index"]) for segment in alignment_input["segments"]
    }
    skipped_input = alignment_input.get("skipped_segments", [])
    expected_skips = {
        int(segment["phrase_index"]): copy.deepcopy(segment)
        for segment in skipped_input
    }

    raw_alignment_skips = alignment.get("skipped_segments", [])
    if not isinstance(raw_alignment_skips, list):
        raise CtcEnrichmentError("CTC alignment has an invalid skipped ledger")
    explicitly_skipped: set[int] = set()
    for position, skipped in enumerate(raw_alignment_skips):
        if (
            not isinstance(skipped, dict)
            or type(skipped.get("phrase_index")) is not int
        ):
            raise CtcEnrichmentError(
                f"CTC skipped segment {position} has no phrase index"
            )
        phrase_index = int(skipped["phrase_index"])
        if phrase_index in explicitly_skipped:
            raise CtcEnrichmentError(
                f"CTC alignment repeats skipped phrase index {phrase_index}"
            )
        expected = expected_skips.get(phrase_index)
        if expected is None or skipped.get("reason") != "no_lexical_content":
            raise CtcEnrichmentError(
                f"CTC alignment unexpectedly skipped source atom {phrase_index}"
            )
        explicitly_skipped.add(phrase_index)

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
        if phrase_index not in expected_segment_ids:
            raise CtcEnrichmentError(
                f"CTC alignment contains unknown source atom {phrase_index}"
            )
        if phrase_index in explicitly_skipped:
            raise CtcEnrichmentError(
                f"CTC alignment both decoded and skipped source atom {phrase_index}"
            )
        by_index[phrase_index] = segment

    enriched = copy.deepcopy(transcript)
    output_atoms = enriched["atoms"]
    recovered: list[dict[str, Any]] = []
    pruned_suffixes: list[dict[str, Any]] = []
    skipped_atoms: list[dict[str, Any]] = []
    decode_failures: list[dict[str, Any]] = []
    for position, atom in enumerate(output_atoms):
        if not isinstance(atom, dict) or type(atom.get("atom_index")) is not int:
            raise CtcEnrichmentError(f"source atom {position} is malformed")
        atom_index = int(atom["atom_index"])
        if atom_index in expected_skips:
            atom["ctc_enrichment"] = {
                "status": "unchanged_no_lexical_content",
                "hidden_retries": [],
                "skip_reason": "no_lexical_content",
            }
            skipped_atoms.append(
                {
                    "atom_index": atom_index,
                    "reason": "no_lexical_content",
                    "explicit_worker_skip": atom_index in explicitly_skipped,
                }
            )
            continue
        segment = by_index.get(atom_index)
        if segment is None:
            raise CtcEnrichmentError(f"CTC alignment omitted source atom {atom_index}")
        if segment.get("status") == "decode_failed":
            raw_error = segment.get("decode_error")
            error = copy.deepcopy(raw_error) if isinstance(raw_error, dict) else {}
            atom["ctc_enrichment"] = {
                "status": "unchanged_ctc_decode_failure",
                "hidden_retries": [],
                "decode_error": error,
            }
            decode_failures.append(
                {
                    "atom_index": atom_index,
                    "decode_error": error,
                }
            )
            continue
        if segment.get("status") not in {None, "decoded"}:
            raise CtcEnrichmentError(
                f"atom {atom_index} has unsupported CTC segment status"
            )
        raw_atom_text = atom.get("text", "")
        atom_text = raw_atom_text if isinstance(raw_atom_text, str) else ""
        expected_text = (
            atom_text
            if normalized_words(atom_text)
            else str(segment.get("input_text", atom_text))
        )
        retries = _validated_hidden_retries(segment)
        suffix_start = _hallucinated_whisper_suffix_start(
            atom=atom,
            segment=segment,
        )
        if suffix_start is not None:
            original_text = str(atom.get("text", ""))
            original_words = copy.deepcopy(atom.get("words", []))
            retained_words = original_words[:suffix_start]
            atom["original_whisper_text"] = original_text
            atom["original_whisper_words"] = original_words
            atom["words"] = retained_words
            atom["text"] = " ".join(str(word["word"]) for word in retained_words)
            atom["segments"] = [
                {
                    "start": float(retained_words[0]["start"]),
                    "end": float(retained_words[-1]["end"]),
                    "text": atom["text"],
                    "words": copy.deepcopy(retained_words),
                    "decode_strategy": "ctc_gated_whisper_suffix_pruning_v1",
                }
            ]
            atom["ctc_enrichment"] = {
                "status": "pruned_whisper_hallucinated_suffix",
                "hidden_retries": [],
                "original_word_count": len(original_words),
                "retained_word_count": len(retained_words),
                "pruned_words": copy.deepcopy(original_words[suffix_start:]),
            }
            pruned_suffixes.append(
                {
                    "atom_index": atom_index,
                    "original_whisper_text": original_text,
                    "retained_text": atom["text"],
                    "pruned_words": [
                        str(word["word"]) for word in original_words[suffix_start:]
                    ],
                }
            )
            continue
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
            expected_text=expected_text,
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
    enriched["source_decode_strategy"] = SOURCE_DECODE_STRATEGY
    enriched["ctc_enrichment"] = {
        "schema_version": 1,
        "status": (
            "complete_with_degraded_evidence" if decode_failures else "complete"
        ),
        "alignment": str(alignment_path.resolve()) if alignment_path else None,
        "alignment_sha256": (
            sha256_file(alignment_path)
            if alignment_path is not None and alignment_path.is_file()
            else None
        ),
        "minimum_retry_confidence": MIN_RETRY_CONFIDENCE,
        "atoms_examined": len(output_atoms),
        "atoms_expanded": len(recovered),
        "atoms_skipped_no_lexical_content": len(skipped_atoms),
        "atoms_with_decode_failures": len(decode_failures),
        "hallucinated_suffixes_pruned": len(pruned_suffixes),
        "hidden_retries_recovered": sum(
            len(item["hidden_retries"]) for item in recovered
        ),
        "recovered": recovered,
        "pruned_suffixes": pruned_suffixes,
        "skipped_atoms": skipped_atoms,
        "decode_failures": decode_failures,
    }
    report = {
        "schema_version": 1,
        **copy.deepcopy(enriched["ctc_enrichment"]),
    }
    return enriched, report


def _validated_source_transcript(
    *,
    audio_path: Path,
    transcript_path: Path,
) -> dict[str, Any]:
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    if not transcript_path.is_file():
        raise FileNotFoundError(transcript_path)
    transcript = read_json(transcript_path)
    if not isinstance(transcript, dict):
        raise CtcEnrichmentError("source transcript root must be an object")
    expected_audio_sha = transcript.get("audio_sha256")
    if not isinstance(expected_audio_sha, str):
        raise CtcEnrichmentError("source transcript has no audio SHA-256")
    if sha256_file(audio_path) != expected_audio_sha:
        raise CtcEnrichmentError("audio does not match the source transcript")
    # This validates duplicate IDs and all atom/word geometry before either
    # normal enrichment or fail-soft passthrough is allowed to write output.
    build_alignment_input(transcript)
    return transcript


def write_passthrough_enrichment(
    *,
    audio_path: Path,
    transcript_path: Path,
    output_dir: Path,
    reason: str,
    status: str = PASSTHROUGH_STATUS,
) -> tuple[Path, Path]:
    """Write a provenance-marked Whisper-primary artifact after CTC failure.

    CTC retry recovery is optional evidence.  Losing it must not discard an
    otherwise valid hour-long Whisper transcript.  Conversely, this helper is
    deliberately not a way around corrupt input: source identity, duplicate
    atom IDs, and timestamp geometry are validated before anything is written.
    """

    audio_path = audio_path.resolve()
    transcript_path = transcript_path.resolve()
    output_dir = output_dir.resolve()
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("CTC passthrough requires a non-empty failure reason")
    transcript = _validated_source_transcript(
        audio_path=audio_path,
        transcript_path=transcript_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    enriched = copy.deepcopy(transcript)
    atoms = enriched.get("atoms", [])
    alignment_input = build_alignment_input(transcript)
    skipped_atoms = [
        {
            "atom_index": int(item["phrase_index"]),
            "reason": "no_lexical_content",
            "explicit_worker_skip": False,
        }
        for item in alignment_input.get("skipped_segments", [])
    ]
    skipped_ids = {int(item["atom_index"]) for item in skipped_atoms}
    for atom in atoms:
        if isinstance(atom, dict):
            atom["ctc_enrichment"] = {
                "status": (
                    "unchanged_no_lexical_content"
                    if atom.get("atom_index") in skipped_ids
                    else "unchanged_global_ctc_failure"
                ),
                "hidden_retries": [],
                "failure_reason": reason.strip(),
            }
    original_engine = enriched.get("engine")
    enriched["engine"] = "whisper_primary_ctc_degraded_passthrough"
    enriched["source_decode_strategy"] = SOURCE_DECODE_STRATEGY
    provenance = {
        "schema_version": 1,
        "status": status,
        "failure_reason": reason.strip(),
        "fallback": "whisper_primary_passthrough",
        "primary_engine": original_engine,
        "source_transcript": str(transcript_path),
        "source_transcript_sha256": sha256_file(transcript_path),
        "audio": str(audio_path),
        "audio_sha256": sha256_file(audio_path),
        "atoms_examined": len(atoms),
        "atoms_expanded": 0,
        "atoms_passthrough": len(atoms),
        "atoms_skipped_no_lexical_content": len(skipped_atoms),
        "atoms_with_decode_failures": 0,
        "global_ctc_failure": True,
        "hallucinated_suffixes_pruned": 0,
        "hidden_retries_recovered": 0,
        "recovered": [],
        "pruned_suffixes": [],
        "skipped_atoms": skipped_atoms,
        "decode_failures": [],
    }
    enriched["ctc_enrichment"] = copy.deepcopy(provenance)
    enriched_path = output_dir / "source_transcript_ctc_enriched.json"
    report_path = output_dir / "ctc_enrichment_report.json"
    write_json(enriched_path, enriched)
    write_json(report_path, provenance)
    return enriched_path, report_path


def run_enrichment(
    *,
    audio_path: Path,
    transcript_path: Path,
    output_dir: Path,
    resume: bool = False,
) -> tuple[Path, Path]:
    audio_path = audio_path.resolve()
    transcript_path = transcript_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise RuntimeError(f"CTC enrichment output must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript = _validated_source_transcript(
        audio_path=audio_path,
        transcript_path=transcript_path,
    )

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
    if resume:
        command.append("--resume")
    environment = os.environ.copy()
    try:
        process = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=environment,
        )
    except OSError as error:
        alignment_log_path.write_text(
            "$ "
            + " ".join(command)
            + "\n\n[worker launch failure]\n"
            + f"{type(error).__name__}: {error}\n",
            encoding="utf-8",
        )
        return write_passthrough_enrichment(
            audio_path=audio_path,
            transcript_path=transcript_path,
            output_dir=output_dir,
            reason=f"raw CTC worker could not start: {type(error).__name__}: {error}",
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
        return write_passthrough_enrichment(
            audio_path=audio_path,
            transcript_path=transcript_path,
            output_dir=output_dir,
            reason=(
                f"raw CTC worker failed with exit status {process.returncode}; "
                f"see {alignment_log_path}"
            ),
        )
    try:
        alignment = read_json(alignment_path)
    except (OSError, json.JSONDecodeError) as error:
        return write_passthrough_enrichment(
            audio_path=audio_path,
            transcript_path=transcript_path,
            output_dir=output_dir,
            reason=f"raw CTC worker produced unreadable evidence: {error}",
        )
    if not isinstance(alignment, dict):
        return write_passthrough_enrichment(
            audio_path=audio_path,
            transcript_path=transcript_path,
            output_dir=output_dir,
            reason="raw CTC worker produced a non-object evidence root",
        )
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
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    enriched_path, report_path = run_enrichment(
        audio_path=args.audio,
        transcript_path=args.transcript,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    report = read_json(report_path)
    print("\nCTC HIDDEN-RETRY ENRICHMENT COMPLETE")
    print(f"atoms examined: {report['atoms_examined']}")
    print(f"atoms expanded: {report['atoms_expanded']}")
    print(
        "atoms skipped without lexical content: "
        f"{report.get('atoms_skipped_no_lexical_content', 0)}"
    )
    print(f"atom decode failures: {report.get('atoms_with_decode_failures', 0)}")
    print(f"hallucinated suffixes pruned: {report['hallucinated_suffixes_pruned']}")
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
