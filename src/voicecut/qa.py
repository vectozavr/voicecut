#!/usr/bin/env python3
"""Fail-closed content, repetition, waveform, seam, and mastering QA."""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
from rapidfuzz.distance import Levenshtein

from .common import (
    FILLERS,
    STOPWORDS,
    build_phrases,
    load_aliases,
    ngrams,
    parse_script,
    read_json,
    script_text,
    tokenize,
    write_json,
)


def transcript_tokens(
    data: dict[str, Any],
    aliases: dict[str, list[str]],
    *,
    prefer_atoms: bool,
) -> tuple[list[str], list[tuple[float, float]]]:
    raw_words: list[dict[str, Any]] = []
    if prefer_atoms and data.get("atoms"):
        for atom in sorted(data["atoms"], key=lambda item: int(item["atom_index"])):
            raw_words.extend(atom.get("words", []))
    elif data.get("whole"):
        raw_words.extend(data["whole"].get("words", []))
    elif data.get("segments"):
        for segment in data["segments"]:
            raw_words.extend(segment.get("words", []))

    tokens: list[str] = []
    times: list[tuple[float, float]] = []
    for word in raw_words:
        normalized = tokenize(str(word.get("word", "")), aliases)
        start = float(word.get("start", 0.0))
        end = float(word.get("end", start))
        for token in normalized:
            tokens.append(token)
            times.append((start, end))
    return tokens, times


def alignment_metrics(reference: list[str], observed: list[str]) -> dict[str, float]:
    matcher = difflib.SequenceMatcher(None, reference, observed, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    recall = matched / max(1, len(reference))
    precision = matched / max(1, len(observed))
    f1 = 2.0 * recall * precision / max(1e-9, recall + precision)
    return {
        "matched_tokens": matched,
        "reference_tokens": len(reference),
        "observed_tokens": len(observed),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "sequence_ratio": matcher.ratio(),
    }


def exact_surplus_repeats(
    reference: list[str],
    observed: list[str],
    times: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for size in range(8, 1, -1):
        reference_counts = Counter(ngrams(reference, size))
        positions: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
        for index, gram in enumerate(ngrams(observed, size)):
            positions[gram].append(index)
        for gram, starts in positions.items():
            surplus = len(starts) - reference_counts.get(gram, 0)
            if surplus <= 0 or len(starts) < 2:
                continue
            close_pairs = [
                (left, right)
                for left, right in zip(starts, starts[1:])
                if right - left <= 42
            ]
            if not close_pairs:
                continue
            left, right = close_pairs[0]
            findings.append(
                {
                    "kind": "surplus_exact_repeat",
                    "severity": "fail" if size >= 3 else "review",
                    "ngram_size": size,
                    "phrase": " ".join(gram),
                    "first_token": left,
                    "second_token": right,
                    "first_time": times[left][0] if left < len(times) else None,
                    "second_time": times[right][0] if right < len(times) else None,
                    "script_count": reference_counts.get(gram, 0),
                    "observed_count": len(starts),
                }
            )
    # Keep only maximal nested repeats at roughly the same positions.
    reduced: list[dict[str, Any]] = []
    for finding in sorted(
        findings, key=lambda item: int(item["ngram_size"]), reverse=True
    ):
        if any(
            abs(int(finding["first_token"]) - int(other["first_token"])) <= 2
            and abs(int(finding["second_token"]) - int(other["second_token"])) <= 2
            for other in reduced
        ):
            continue
        reduced.append(finding)
    return reduced


def restart_insertions(
    reference: list[str],
    observed: list[str],
    times: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    matcher = difflib.SequenceMatcher(None, reference, observed, autojunk=False)
    findings: list[dict[str, Any]] = []
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        observed_part = observed[j1:j2]
        reference_part = reference[i1:i2]
        extra = len(observed_part) - len(reference_part)
        if opcode not in {"insert", "replace"} or extra < 2 or len(observed_part) < 2:
            continue
        left_context = observed[max(0, j1 - 10) : j1]
        right_context = observed[j2 : min(len(observed), j2 + 14)]
        outside_counts = Counter(left_context + right_context)
        inserted_counts = Counter(observed_part)
        repeated_inserted = sum(
            min(count, outside_counts.get(token, 0))
            for token, count in inserted_counts.items()
        )
        context_overlap = repeated_inserted / max(1, len(observed_part))
        content_tokens = [token for token in observed_part if token not in STOPWORDS]
        repeated_content = sum(
            1 for token in content_tokens if outside_counts.get(token, 0)
        )
        repeats_on_both_sides = (
            any(token in left_context for token in observed_part)
            and any(token in right_context for token in observed_part)
        )
        two_sided_restart = (
            context_overlap >= 0.75
            and repeated_content >= 1
            and repeats_on_both_sides
        )
        length = min(8, len(observed_part))
        fragment = observed_part[:length]
        nearby_start = max(0, j1 - 10)
        nearby_end = min(len(observed), j2 + 14)
        best = 0.0
        best_window: list[str] = []
        for start in range(nearby_start, nearby_end - len(fragment) + 1):
            # The comparison window must be genuinely separate.  A window
            # beginning just before the insertion but overlapping most of it
            # otherwise compares the phrase with itself and creates a false
            # restart finding.
            if start < j2 and start + len(fragment) > j1:
                continue
            window = observed[start : start + len(fragment)]
            similarity = Levenshtein.normalized_similarity(fragment, window)
            if similarity > best:
                best = similarity
                best_window = window
        if best < 0.52 and not two_sided_restart:
            continue
        effective_similarity = max(best, context_overlap if two_sided_restart else 0.0)
        findings.append(
            {
                "kind": "restart_like_insertion",
                "severity": (
                    "fail"
                    if (
                        (extra >= 3 and best >= 0.68)
                        or (extra >= 2 and two_sided_restart)
                    )
                    else "review"
                ),
                "observed_fragment": " ".join(observed_part),
                "nearby_match": (
                    " ".join(best_window)
                    if best_window
                    else " ".join(left_context[-4:] + right_context[:4])
                ),
                "similarity": effective_similarity,
                "context_overlap": context_overlap,
                "two_sided_restart": two_sided_restart,
                "extra_tokens": extra,
                "start_time": times[j1][0] if j1 < len(times) else None,
                "end_time": times[j2 - 1][1] if j2 and j2 - 1 < len(times) else None,
            }
        )
    return findings


def single_token_disfluencies(
    reference: list[str],
    observed: list[str],
    times: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    """Report fillers and one-token stutters that corpus-level scores dilute.

    The script is authoritative: an intentional repeated word or filler aligns
    normally and is not reported.  We inspect only tokens that the alignment
    identifies as surplus relative to that script.
    """

    matcher = difflib.SequenceMatcher(None, reference, observed, autojunk=False)
    findings: list[dict[str, Any]] = []
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode not in {"insert", "replace"} or j2 <= j1:
            continue
        reference_part = reference[i1:i2]
        observed_part = observed[j1:j2]
        surplus = Counter(observed_part) - Counter(reference_part)
        if not surplus:
            continue
        remaining = Counter(surplus)
        for position in range(j1, j2):
            token = observed[position]
            if remaining.get(token, 0) <= 0:
                continue
            left = observed[position - 1] if position > 0 else None
            right = observed[position + 1] if position + 1 < len(observed) else None
            filler = token in FILLERS
            adjacent_duplicate = token == left or token == right
            if not filler and not adjacent_duplicate:
                continue
            remaining[token] -= 1
            start = times[position][0] if position < len(times) else None
            end = times[position][1] if position < len(times) else None
            findings.append(
                {
                    "kind": (
                        "surplus_spoken_filler"
                        if filler
                        else "single_token_stutter"
                    ),
                    # A filler is unambiguous text evidence.  A one-token
                    # duplicate remains REVIEW because ASR can occasionally
                    # hallucinate it; strict mode still blocks publication.
                    "severity": "fail" if filler else "review",
                    "observed_fragment": token,
                    "left_token": left,
                    "right_token": right,
                    "start_time": start,
                    "end_time": end,
                }
            )
    return findings


def fuzzy_nearby_repeats(
    reference: list[str],
    observed: list[str],
    times: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    reference_grams = {
        size: Counter(ngrams(reference, size)) for size in range(4, 9)
    }
    for size in range(8, 3, -1):
        for left in range(0, len(observed) - size + 1):
            first = observed[left : left + size]
            for right in range(left + 2, min(len(observed) - size + 1, left + 30)):
                second = observed[right : right + size]
                similarity = Levenshtein.normalized_similarity(first, second)
                if similarity < 0.88:
                    continue
                if reference_grams[size].get(tuple(first), 0) >= 2:
                    continue
                findings.append(
                    {
                        "kind": "fuzzy_nearby_repeat",
                        "severity": "review",
                        "size": size,
                        "first": " ".join(first),
                        "second": " ".join(second),
                        "similarity": similarity,
                        "first_time": times[left][0] if left < len(times) else None,
                        "second_time": times[right][0] if right < len(times) else None,
                    }
                )
                break
    reduced: list[dict[str, Any]] = []
    for finding in findings:
        if any(
            finding["first_time"] == other["first_time"]
            and finding["second_time"] == other["second_time"]
            for other in reduced
        ):
            continue
        reduced.append(finding)
    return reduced[:30]


def signal_checks(audio_path: Path, edl_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    edl = read_json(edl_path)
    issues: list[dict[str, Any]] = []
    finite = bool(np.all(np.isfinite(mono)))
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    dc = float(np.mean(mono)) if len(mono) else 0.0
    if not finite:
        issues.append({"kind": "invalid_samples", "severity": "fail"})
    if peak >= 1.0:
        issues.append(
            {"kind": "sample_clipping", "severity": "fail", "peak": peak}
        )
    if abs(dc) > 0.001:
        issues.append(
            {"kind": "excessive_dc", "severity": "review", "dc_offset": dc}
        )

    exact_zero = mono == 0.0
    padded = np.pad(exact_zero.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    runs = edges.reshape(-1, 2) if len(edges) else np.empty((0, 2), dtype=int)
    longest_zero = int(max((end - start for start, end in runs), default=0))
    if longest_zero >= round(0.005 * sample_rate):
        issues.append(
            {
                "kind": "digital_silence_run",
                "severity": "fail",
                "duration_ms": longest_zero / sample_rate * 1000.0,
            }
        )

    frame_samples = max(1, round(0.010 * sample_rate))
    frame_count = math.ceil(len(mono) / frame_samples) if len(mono) else 0
    if frame_count:
        padded_audio = np.pad(
            mono, (0, frame_count * frame_samples - len(mono))
        ).reshape(frame_count, frame_samples)
        frame_rms = np.sqrt(
            np.mean(np.square(padded_audio, dtype=np.float64), axis=1)
        )
        near_zero = frame_rms <= 10.0 ** (-115.0 / 20.0)
        padded_mask = np.pad(near_zero.astype(np.int8), (1, 1))
        near_edges = np.flatnonzero(np.diff(padded_mask))
        near_runs = (
            near_edges.reshape(-1, 2)
            if len(near_edges)
            else np.empty((0, 2), dtype=int)
        )
        longest_near_zero_frames = int(
            max((end - start for start, end in near_runs), default=0)
        )
    else:
        longest_near_zero_frames = 0
    longest_near_zero_ms = longest_near_zero_frames * frame_samples / max(
        1, sample_rate
    ) * 1000.0
    if longest_near_zero_ms >= 20.0:
        issues.append(
            {
                "kind": "near_silent_dropout",
                "severity": "fail",
                "duration_ms": longest_near_zero_ms,
                "threshold_dbfs": -115.0,
            }
        )

    derivative = np.abs(np.diff(mono, prepend=mono[:1]))
    natural_q999 = float(np.quantile(derivative, 0.999)) if len(derivative) else 0.0
    click_threshold = max(0.10, 3.5 * natural_q999)
    click_samples = np.flatnonzero(derivative > click_threshold)
    if len(click_samples):
        issues.append(
            {
                "kind": "impulsive_click_candidates",
                "severity": "review",
                "count": int(len(click_samples)),
                "maximum_jump": float(np.max(derivative[click_samples])),
                "threshold": click_threshold,
                "time": float(click_samples[0]) / sample_rate,
            }
        )
    seam_rows: list[dict[str, Any]] = []
    breath_edits_checked = 0
    protected_gap_samples_checked = 0
    source_path = Path(str(edl.get("audio", "")))

    def source_samples(start: int, end: int) -> np.ndarray | None:
        if not source_path.is_file() or end <= start:
            return None
        with sf.SoundFile(source_path) as handle:
            if int(handle.samplerate) != int(sample_rate):
                return None
            handle.seek(start)
            data = handle.read(end - start, dtype="float32", always_2d=True)
        return np.mean(data, axis=1, dtype=np.float32)

    for piece in edl.get("pieces", []):
        for label in ("output_start_sample", "output_end_sample", "output_gap_end_sample"):
            sample = int(piece.get(label, 0))
            if not 1 <= sample < len(mono):
                continue
            jump = abs(float(mono[sample] - mono[sample - 1]))
            threshold = max(0.012, 1.25 * natural_q999)
            row = {
                "piece_index": int(piece["piece_index"]),
                "kind": label,
                "sample": sample,
                "time": sample / sample_rate,
                "jump": jump,
                "threshold": threshold,
            }
            seam_rows.append(row)
            if jump > threshold:
                issues.append(
                    {
                        "kind": "seam_discontinuity",
                        "severity": "fail",
                        **row,
                    }
                )
        gap_seconds = float(piece["gap_after_samples"]) / sample_rate
        if gap_seconds > 1.2:
            issues.append(
                {
                    "kind": "excessive_pause",
                    "severity": "review",
                    "piece_index": int(piece["piece_index"]),
                    "gap_seconds": gap_seconds,
                }
            )
        if piece.get("review_reasons"):
            issues.append(
                {
                    "kind": "piece_boundary_or_selection_review",
                    "severity": "review",
                    "piece_index": int(piece["piece_index"]),
                    "time": int(piece.get("output_start_sample", 0)) / sample_rate,
                    "reasons": piece["review_reasons"],
                }
            )
        for edit in piece.get("breath_attenuations", []):
            breath_edits_checked += 1
            source_start = int(edit.get("source_start_sample", -1))
            source_end = int(edit.get("source_end_sample", -1))
            safe_start = int(edit.get("source_safe_start_sample", -1))
            safe_end = int(edit.get("source_safe_end_sample", -1))
            left_word_end = int(edit.get("source_left_word_end_sample", -1))
            right_word_start = int(
                edit.get("source_right_word_start_sample", -1)
            )
            guard = int(edit.get("guard_samples", -1))
            structurally_safe = (
                0 <= left_word_end
                <= safe_start
                <= source_start
                < source_end
                <= safe_end
                <= right_word_start
                and safe_start - left_word_end >= guard
                and right_word_start - safe_end >= guard
            )
            if not structurally_safe:
                issues.append(
                    {
                        "kind": "unsafe_interword_gap_edit",
                        "severity": "fail",
                        "piece_index": int(piece["piece_index"]),
                        "source_start_sample": source_start,
                        "source_end_sample": source_end,
                        "safe_start_sample": safe_start,
                        "safe_end_sample": safe_end,
                    }
                )
                continue

            excess_reduction = float(
                edit.get("central_excess_reduction_db", -200.0)
            )
            processed_center = float(
                edit.get("central_processed_rms_db", 200.0)
            )
            room_center = float(
                edit.get("central_room_tone_rms_db", -200.0)
            )
            if excess_reduction < 12.0:
                issues.append(
                    {
                        "kind": "weak_interword_breath_reduction",
                        "severity": "fail",
                        "piece_index": int(piece["piece_index"]),
                        "value_db": excess_reduction,
                        "minimum_db": 12.0,
                    }
                )
            if processed_center > room_center + 3.5:
                issues.append(
                    {
                        "kind": "breath_core_above_room_floor",
                        "severity": "fail",
                        "piece_index": int(piece["piece_index"]),
                        "processed_rms_db": processed_center,
                        "room_rms_db": room_center,
                        "maximum_difference_db": 3.5,
                    }
                )

            piece_source_start = int(piece["source_start_sample"])
            piece_output_start = int(piece["output_start_sample"])
            for protected_start, protected_end, label in (
                (left_word_end, source_start, "left"),
                (source_end, right_word_start, "right"),
            ):
                expected = source_samples(protected_start, protected_end)
                output_start = (
                    piece_output_start
                    + protected_start
                    - piece_source_start
                )
                output_end = output_start + max(
                    0, protected_end - protected_start
                )
                if (
                    expected is None
                    or output_start < 0
                    or output_end > len(mono)
                ):
                    issues.append(
                        {
                            "kind": "protected_gap_verification_unavailable",
                            "severity": "review",
                            "piece_index": int(piece["piece_index"]),
                            "side": label,
                        }
                    )
                    continue
                protected_gap_samples_checked += len(expected)
                if not np.array_equal(mono[output_start:output_end], expected):
                    issues.append(
                        {
                            "kind": "protected_interword_audio_changed",
                            "severity": "fail",
                            "piece_index": int(piece["piece_index"]),
                            "side": label,
                            "output_time": output_start / sample_rate,
                        }
                    )
    metrics = {
        "sample_rate": sample_rate,
        "channels": int(audio.shape[1]),
        "duration": len(mono) / sample_rate,
        "finite": finite,
        "peak": peak,
        "dc_offset": dc,
        "longest_exact_zero_ms": longest_zero / sample_rate * 1000.0,
        "longest_near_silent_ms": longest_near_zero_ms,
        "natural_derivative_q999": natural_q999,
        "impulsive_click_candidate_count": int(len(click_samples)),
        "maximum_seam_jump": max(
            (float(row["jump"]) for row in seam_rows), default=0.0
        ),
        "seams_checked": len(seam_rows),
        "breath_edits_checked": breath_edits_checked,
        "protected_gap_samples_checked": protected_gap_samples_checked,
    }
    return metrics, issues


def loudness_metrics(path: Path) -> dict[str, float]:
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
    text = process.stderr
    summaries = list(re.finditer(r"Summary:\s*(.*?)(?:\n\[|\Z)", text, re.S))
    summary = summaries[-1].group(1) if summaries else text[-4000:]

    def extract(pattern: str) -> float:
        match = re.search(pattern, summary)
        return float(match.group(1)) if match else float("nan")

    return {
        "integrated_lufs": extract(r"I:\s+(-?\d+(?:\.\d+)?)\s+LUFS"),
        "lra_lu": extract(r"LRA:\s+(\d+(?:\.\d+)?)\s+LU"),
        "true_peak_dbtp": extract(r"Peak:\s+(-?\d+(?:\.\d+)?)\s+dBFS"),
    }


def export_review_clips(
    audio_path: Path,
    issues: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    # This is a pipeline-owned evidence directory.  A resumed PASS must not
    # retain clips from an earlier REVIEW/FAIL quality-gate run.
    for previous in output_dir.glob("*.wav"):
        previous.unlink()
    exported = 0
    for index, issue in enumerate(issues):
        time_value = issue.get("start_time", issue.get("second_time", issue.get("time")))
        if time_value is None:
            continue
        center = float(time_value)
        start = max(0, round((center - 2.0) * sample_rate))
        end = min(len(audio), round((center + 3.0) * sample_rate))
        if end <= start:
            continue
        filename = output_dir / f"{index:03d}_{issue['kind']}.wav"
        sf.write(filename, audio[start:end], sample_rate, subtype="PCM_16")
        exported += 1
        if exported >= 30:
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--independent-transcript", type=Path)
    parser.add_argument("--edit-plan", type=Path, required=True)
    parser.add_argument("--edl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aliases", type=Path)
    parser.add_argument("--mastered", type=Path)
    parser.add_argument(
        "--allow-missing-unit",
        type=int,
        action="append",
        default=[],
    )
    parser.add_argument("--target-lufs", type=float, default=-16.0)
    parser.add_argument("--target-true-peak", type=float, default=-1.5)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    aliases = load_aliases(args.aliases)
    units = parse_script(args.script, aliases)
    _ = build_phrases(units, aliases)
    allowed_missing = set(args.allow_missing_unit)
    plan = read_json(args.edit_plan)
    selected_unit_numbers = {
        int(item["unit_number"])
        for item in plan.get("selections", [])
        if item.get("status") in {"selected", "review"}
    }
    actually_waived = allowed_missing.difference(selected_unit_numbers)
    reference = tokenize(script_text(units, actually_waived), aliases)
    transcript = read_json(args.transcript)
    observed, times = transcript_tokens(
        transcript, aliases, prefer_atoms=bool(transcript.get("atoms"))
    )
    whole_observed, whole_times = transcript_tokens(
        transcript, aliases, prefer_atoms=False
    )
    metrics = alignment_metrics(reference, observed)
    whole_metrics = alignment_metrics(reference, whole_observed)
    independent_metrics: dict[str, float] | None = None

    issues: list[dict[str, Any]] = []
    if metrics["recall"] < 0.90:
        issues.append(
            {
                "kind": "low_script_recall",
                "severity": "fail" if metrics["recall"] < 0.84 else "review",
                "value": metrics["recall"],
                "threshold": 0.90,
            }
        )
    if metrics["precision"] < 0.89:
        issues.append(
            {
                "kind": "low_script_precision",
                "severity": "fail" if metrics["precision"] < 0.82 else "review",
                "value": metrics["precision"],
                "threshold": 0.89,
            }
        )
    issues.extend(exact_surplus_repeats(reference, observed, times))
    issues.extend(restart_insertions(reference, observed, times))
    issues.extend(single_token_disfluencies(reference, observed, times))
    issues.extend(fuzzy_nearby_repeats(reference, observed, times))
    if whole_observed and whole_observed != observed:
        for finding in exact_surplus_repeats(reference, whole_observed, whole_times):
            finding["detector"] = "whole_file_asr"
            issues.append(finding)
        for finding in restart_insertions(reference, whole_observed, whole_times):
            finding["detector"] = "whole_file_asr"
            issues.append(finding)
        for finding in single_token_disfluencies(
            reference, whole_observed, whole_times
        ):
            finding["detector"] = "whole_file_asr"
            issues.append(finding)

    if args.independent_transcript:
        independent_data = read_json(args.independent_transcript)
        independent_observed, independent_times = transcript_tokens(
            independent_data, aliases, prefer_atoms=False
        )
        independent_metrics = alignment_metrics(reference, independent_observed)
        if independent_metrics["recall"] < 0.88:
            issues.append(
                {
                    "kind": "independent_asr_low_recall",
                    "severity": (
                        "fail" if independent_metrics["recall"] < 0.82 else "review"
                    ),
                    "value": independent_metrics["recall"],
                    "threshold": 0.88,
                }
            )
        for finding in exact_surplus_repeats(
            reference, independent_observed, independent_times
        ):
            finding["detector"] = "independent_asr"
            issues.append(finding)
        for finding in restart_insertions(
            reference, independent_observed, independent_times
        ):
            finding["detector"] = "independent_asr"
            issues.append(finding)
        for finding in single_token_disfluencies(
            reference, independent_observed, independent_times
        ):
            finding["detector"] = "independent_asr"
            issues.append(finding)
    for item in plan.get("review_items", []):
        if item.get("kind") == "missing_script_phrase" and int(
            item.get("unit_number", -1)
        ) in allowed_missing:
            continue
        issues.append(dict(item))

    signal, signal_issues = signal_checks(args.audio, args.edl)
    issues.extend(signal_issues)
    mastered_metrics: dict[str, Any] | None = None
    if args.mastered:
        mastered_metrics = loudness_metrics(args.mastered)
        mastered_audio, mastered_rate = sf.read(
            args.mastered, dtype="float32", always_2d=True
        )
        mastered_mono = np.mean(mastered_audio, axis=1, dtype=np.float32)
        mastered_peak = (
            float(np.max(np.abs(mastered_mono))) if len(mastered_mono) else 0.0
        )
        mastered_finite = bool(np.all(np.isfinite(mastered_mono)))
        mastered_metrics["sample_rate"] = int(mastered_rate)
        mastered_metrics["sample_peak"] = mastered_peak
        mastered_metrics["finite"] = mastered_finite
        if not mastered_finite:
            issues.append({"kind": "master_invalid_samples", "severity": "fail"})
        if mastered_peak >= 1.0:
            issues.append(
                {
                    "kind": "master_sample_clipping",
                    "severity": "fail",
                    "peak": mastered_peak,
                }
            )
        integrated = mastered_metrics["integrated_lufs"]
        true_peak = mastered_metrics["true_peak_dbtp"]
        lra = mastered_metrics["lra_lu"]
        if not math.isfinite(integrated) or abs(integrated - args.target_lufs) > 0.5:
            issues.append(
                {
                    "kind": "master_loudness_out_of_range",
                    "severity": "fail",
                    "value": integrated,
                    "target": args.target_lufs,
                }
            )
        if not math.isfinite(true_peak) or true_peak > args.target_true_peak + 0.1:
            issues.append(
                {
                    "kind": "master_true_peak_out_of_range",
                    "severity": "fail",
                    "value": true_peak,
                    "maximum": args.target_true_peak,
                }
            )
        minimum_lra = 2.0 if float(signal["duration"]) >= 30.0 else 0.0
        if not math.isfinite(lra) or not minimum_lra <= lra <= 8.0:
            issues.append(
                {
                    "kind": "master_lra_review",
                    "severity": "review",
                    "value": lra,
                    "expected_range": [minimum_lra, 8.0],
                }
            )

    # De-duplicate the same repeat reported by atom and whole-file decoders.
    unique: list[dict[str, Any]] = []
    signatures: set[tuple[Any, ...]] = set()
    for issue in issues:
        signature = (
            issue.get("kind"),
            issue.get("phrase"),
            issue.get("observed_fragment"),
            round(float(issue.get("second_time", -1.0)), 1)
            if issue.get("second_time") is not None
            else issue.get("phrase_index"),
            round(float(issue.get("start_time", -1.0)), 1)
            if issue.get("start_time") is not None
            else None,
            issue.get("piece_index"),
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        unique.append(issue)
    issues = unique

    fail_count = sum(issue.get("severity") == "fail" for issue in issues)
    review_count = sum(issue.get("severity") == "review" for issue in issues)
    if fail_count:
        verdict = "FAIL"
    elif review_count:
        verdict = "REVIEW"
    else:
        verdict = "PASS"
    report = {
        "schema_version": 1,
        "verdict": verdict,
        "strict": args.strict,
        "content": {
            "atom_or_local_asr": metrics,
            "whole_file_asr": whole_metrics,
            "independent_asr": independent_metrics,
        },
        "actually_waived_missing_units": sorted(actually_waived),
        "signal": signal,
        "mastered": mastered_metrics,
        "issue_count": len(issues),
        "fail_count": fail_count,
        "review_count": review_count,
        "issues": issues,
    }
    write_json(args.output, report)
    export_review_clips(args.audio, issues, args.output.parent / "review_clips")
    print(json.dumps(report, indent=2))
    if args.strict and verdict != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
