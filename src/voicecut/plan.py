#!/usr/bin/env python3
"""Generate a globally ordered, retry-aware narration edit plan."""

from __future__ import annotations

import argparse
import bisect
import difflib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein

from .common import (
    STOPWORDS,
    Candidate,
    ObservedWord,
    ScriptPhrase,
    build_phrases,
    load_aliases,
    parse_script,
    read_json,
    repeated_surplus,
    sha256_file,
    split_long_clause,
    tokenize,
    write_json,
)


def load_observed_words(
    transcript_path: Path,
    aliases: dict[str, list[str]],
) -> list[ObservedWord]:
    data = read_json(transcript_path)
    words: list[ObservedWord] = []
    for atom in sorted(data.get("atoms", []), key=lambda item: int(item["atom_index"])):
        atom_index = int(atom["atom_index"])
        atom_start = float(atom["start"])
        atom_end = float(atom["end"])
        for raw in atom.get("words", []):
            text = str(raw.get("word", "")).strip()
            tokens = tokenize(text, aliases)
            if not tokens:
                continue
            start = max(atom_start, min(atom_end, float(raw.get("start", atom_start))))
            end = max(start, min(atom_end, float(raw.get("end", atom_end))))
            words.append(
                ObservedWord(
                    word_index=len(words),
                    atom_index=atom_index,
                    text=text,
                    tokens=tokens,
                    start=start,
                    end=end,
                    probability=float(raw.get("probability", 0.5)),
                )
            )
    words.sort(key=lambda word: (word.start, word.end, word.atom_index))
    for index, word in enumerate(words):
        word.word_index = index
    return words


def sequence_stats(
    reference: list[str], observed: list[str]
) -> tuple[float, float, float, float]:
    matcher = difflib.SequenceMatcher(None, reference, observed, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    recall = matched / max(1, len(reference))
    precision = matched / max(1, len(observed))
    f1 = 2.0 * recall * precision / max(1e-9, recall + precision)
    edit_similarity = Levenshtein.normalized_similarity(reference, observed)
    return f1, recall, precision, edit_similarity


def phrase_anchors(
    phrases: list[ScriptPhrase],
    words: list[ObservedWord],
) -> list[list[int]]:
    script_tokens: list[str] = []
    phrase_starts: list[int] = []
    for phrase in phrases:
        phrase_starts.append(len(script_tokens))
        script_tokens.extend(phrase.tokens)
    observed_tokens: list[str] = []
    token_word_map: list[int] = []
    for word in words:
        for token in word.tokens:
            observed_tokens.append(token)
            token_word_map.append(word.word_index)

    anchors: list[list[int]] = [[] for _ in phrases]
    if not script_tokens or not observed_tokens:
        return anchors

    def phrase_for_token(position: int) -> int:
        return max(0, bisect.bisect_right(phrase_starts, position) - 1)

    for opcode, source_start, source_end, target_start, target_end in Levenshtein.opcodes(
        script_tokens, observed_tokens
    ):
        if opcode == "equal":
            for offset in range(source_end - source_start):
                phrase_index = phrase_for_token(source_start + offset)
                anchors[phrase_index].append(token_word_map[target_start + offset])
        elif opcode == "replace" and source_end > source_start and target_end > target_start:
            source_length = source_end - source_start
            target_length = target_end - target_start
            for offset in range(target_length):
                relative = min(
                    source_length - 1,
                    int(offset * source_length / max(1, target_length)),
                )
                phrase_index = phrase_for_token(source_start + relative)
                anchors[phrase_index].append(token_word_map[target_start + offset])
    return anchors


def search_windows(
    anchors: list[list[int]],
    word_count: int,
    radius: int = 72,
) -> list[tuple[int, int]]:
    minimums = [min(values) if values else None for values in anchors]
    maximums = [max(values) if values else None for values in anchors]
    windows: list[tuple[int, int]] = []
    for phrase_index in range(len(anchors)):
        own_min = minimums[phrase_index]
        own_max = maximums[phrase_index]
        previous = max(
            (
                maximums[index]
                for index in range(phrase_index)
                if maximums[index] is not None
            ),
            default=-1,
        )
        following = min(
            (
                minimums[index]
                for index in range(phrase_index + 1, len(anchors))
                if minimums[index] is not None
            ),
            default=word_count,
        )
        if own_min is not None and own_max is not None:
            start = max(0, own_min - radius, previous - 18)
            end = min(word_count, own_max + radius, following + 18)
        else:
            start = max(0, previous - 12)
            end = min(word_count, following + 12)
            if end - start > 2 * radius + 60:
                end = min(word_count, start + 2 * radius + 60)
        if end <= start:
            start = max(0, min(word_count - 1, previous + 1))
            end = min(word_count, start + radius)
        windows.append((start, end))
    return windows


def span_tokens(words: list[ObservedWord], start: int, end: int) -> list[str]:
    tokens: list[str] = []
    for word in words[start:end]:
        tokens.extend(word.tokens)
    return tokens


def candidate_for_span(
    phrase: ScriptPhrase,
    words: list[ObservedWord],
    start: int,
    end: int,
    search_start: int,
    search_end: int,
) -> Candidate:
    observed = span_tokens(words, start, end)
    reference = phrase.tokens
    f1, recall, precision, edit_similarity = sequence_stats(reference, observed)
    prefix_size = min(4, len(reference))
    suffix_size = min(4, len(reference))
    prefix_recall = sequence_stats(
        reference[:prefix_size], observed[: min(len(observed), prefix_size + 3)]
    )[1]
    suffix_recall = sequence_stats(
        reference[-suffix_size:], observed[-min(len(observed), suffix_size + 3) :]
    )[1]
    probabilities = [
        max(0.0, min(1.0, word.probability)) for word in words[start:end]
    ]
    mean_probability = sum(probabilities) / max(1, len(probabilities))
    gaps = [
        max(0.0, words[index + 1].start - words[index].end)
        for index in range(start, end - 1)
    ]
    max_gap = max(gaps, default=0.0)
    excessive_pause = sum(max(0.0, gap - 0.70) for gap in gaps)
    repeat_penalty = repeated_surplus(reference, observed)
    durations = [max(0.0, word.end - word.start) for word in words[start:end]]
    abnormal_word_time = sum(max(0.0, duration - 1.15) for duration in durations)
    span_seconds = max(0.001, words[end - 1].end - words[start].start)
    seconds_per_token = span_seconds / max(1, len(observed))
    rate_penalty = max(0.0, seconds_per_token - 0.85) + max(
        0.0, 0.055 - seconds_per_token
    )
    # A tiny, global later-take tie-break is deterministic across retrieval
    # windows.  It never outweighs content quality.
    recency = 0.018 * start / max(1, len(words) - 1)
    score = (
        0.27 * f1
        + 0.22 * recall
        + 0.13 * precision
        + 0.16 * edit_similarity
        + 0.07 * prefix_recall
        + 0.07 * suffix_recall
        + 0.08 * min(1.0, mean_probability / 0.82)
        + recency
        - min(0.24, 0.035 * repeat_penalty)
        - min(0.12, 0.025 * excessive_pause)
        - min(0.12, 0.035 * abnormal_word_time)
        - min(0.10, 0.08 * rate_penalty)
    )
    return Candidate(
        phrase_index=phrase.phrase_index,
        word_start=start,
        word_end=end,
        score=score,
        f1=f1,
        recall=recall,
        precision=precision,
        edit_similarity=edit_similarity,
        prefix_recall=prefix_recall,
        suffix_recall=suffix_recall,
        mean_probability=mean_probability,
        repeated_surplus=repeat_penalty,
        max_gap=max_gap,
        transcript=" ".join(word.text for word in words[start:end]),
        source_start=words[start].start,
        source_end=words[end - 1].end,
    )


def top_candidates(
    phrase: ScriptPhrase,
    words: list[ObservedWord],
    search_start: int,
    search_end: int,
    limit: int = 14,
) -> list[Candidate]:
    if search_end <= search_start or not phrase.tokens:
        return []
    reference_length = len(phrase.tokens)
    content = {
        token for token in phrase.tokens if token not in {"a", "an", "the"} and len(token) > 2
    }
    positions = [
        index
        for index in range(search_start, search_end)
        if content.intersection(words[index].tokens)
    ]
    if not positions:
        positions = list(range(search_start, search_end))

    starts: set[int] = set()
    for position in positions:
        for shift in range(-5, 3):
            candidate_start = position + shift
            if search_start <= candidate_start < search_end:
                starts.add(candidate_start)

    minimum_words = max(1, math.floor(reference_length * 0.38))
    maximum_words = max(minimum_words + 2, math.ceil(reference_length * 1.75) + 7)
    candidates: list[Candidate] = []
    for start in sorted(starts):
        upper = min(search_end, start + maximum_words)
        for end in range(start + minimum_words, upper + 1):
            candidate = candidate_for_span(
                phrase, words, start, end, search_start, search_end
            )
            partial_ok = (
                candidate.recall >= 0.52
                and candidate.precision >= 0.54
                and candidate.f1 >= 0.53
            )
            complete_ok = candidate.recall >= 0.68 and candidate.f1 >= 0.63
            if partial_ok or complete_ok:
                candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            item.score,
            item.recall,
            item.precision,
            item.word_start,
        ),
        reverse=True,
    )
    # Remove near-identical shifted spans so ambiguity reflects genuinely
    # different takes, not one take with a punctuation word added or removed.
    result: list[Candidate] = []
    for candidate in candidates:
        duplicate = False
        for kept in result:
            intersection = max(
                0,
                min(candidate.word_end, kept.word_end)
                - max(candidate.word_start, kept.word_start),
            )
            union = max(candidate.word_end, kept.word_end) - min(
                candidate.word_start, kept.word_start
            )
            if union and intersection / union >= 0.72:
                duplicate = True
                break
        if not duplicate:
            result.append(candidate)
        if len(result) >= limit:
            break
    return result


def alternate_search_windows(
    phrase: ScriptPhrase,
    words: list[ObservedWord],
    inverted: dict[str, list[int]],
    primary: tuple[int, int],
    *,
    limit: int = 5,
) -> list[tuple[int, int]]:
    """Retrieve distant retakes using rare lexical anchors.

    A single global sequence alignment can anchor a phrase to an abandoned
    attempt.  These additional windows make clean takes discoverable even when
    they are hundreds of words away from that alignment.
    """

    content = {
        token
        for token in phrase.tokens
        if token not in STOPWORDS and len(token) > 2 and token in inverted
    }
    if not content:
        content = {token for token in phrase.tokens if token in inverted}
    rare = sorted(content, key=lambda token: (len(inverted[token]), token))[:4]
    half_width = max(24, min(90, 2 * len(phrase.tokens) + 16))
    seeds: list[tuple[float, int]] = []
    phrase_content = set(content)
    for token in rare:
        positions = inverted[token]
        if len(positions) > 500:
            continue
        rarity = 1.0 / math.sqrt(max(1, len(positions)))
        for position in positions:
            left = max(0, position - half_width // 2)
            right = min(len(words), position + half_width // 2)
            local: set[str] = set()
            for word in words[left:right]:
                local.update(word.tokens)
            coverage = len(phrase_content.intersection(local)) / max(
                1, len(phrase_content)
            )
            seeds.append((coverage + 0.12 * rarity, position))
    seeds.sort(reverse=True)

    windows: list[tuple[int, int]] = []
    primary_start, primary_end = primary
    for _, position in seeds:
        start = max(0, position - half_width)
        end = min(len(words), position + half_width)
        overlap = max(0, min(end, primary_end) - max(start, primary_start))
        if overlap / max(1, end - start) >= 0.65:
            continue
        center = (start + end) // 2
        if any(abs(center - (left + right) // 2) < half_width for left, right in windows):
            continue
        windows.append((start, end))
        if len(windows) >= limit:
            break
    return windows


def combine_candidate_windows(
    phrase: ScriptPhrase,
    words: list[ObservedWord],
    windows: list[tuple[int, int]],
    *,
    limit: int,
) -> list[Candidate]:
    combined: dict[tuple[int, int], Candidate] = {}
    for start, end in windows:
        for candidate in top_candidates(
            phrase,
            words,
            start,
            end,
            limit=max(8, limit),
        ):
            key = (candidate.word_start, candidate.word_end)
            if key not in combined or candidate.score > combined[key].score:
                combined[key] = candidate
    ranked = sorted(
        combined.values(),
        key=lambda item: (item.score, item.recall, item.precision, item.word_start),
        reverse=True,
    )
    return ranked[:limit]


def select_global_path(
    phrases: list[ScriptPhrase],
    candidate_sets: list[list[Candidate]],
    allowed_missing_units: set[int],
    beam_width: int = 160,
) -> list[Candidate | None]:
    # (score, last_end, selections)
    states: list[tuple[float, int, list[Candidate | None]]] = [(0.0, 0, [])]
    for phrase, candidates in zip(phrases, candidate_sets):
        next_states: list[tuple[float, int, list[Candidate | None]]] = []
        missing_allowed = phrase.unit_index + 1 in allowed_missing_units
        for state_score, last_end, selections in states:
            missing_penalty = 0.08 if missing_allowed else 0.92
            next_states.append(
                (
                    state_score - missing_penalty,
                    last_end,
                    selections + [None],
                )
            )
            for candidate in candidates:
                if candidate.word_start < last_end:
                    continue
                skipped_words = candidate.word_start - last_end
                transition = (
                    0.025
                    if skipped_words <= 2
                    else -min(0.09, skipped_words * 0.0008)
                )
                quality = 1.35 * candidate.score + 0.12 * candidate.recall
                next_states.append(
                    (
                        state_score + quality + transition,
                        candidate.word_end,
                        selections + [candidate],
                    )
                )
        # Keep diverse end positions; otherwise many shifted versions of one
        # early take can crowd the beam and suppress the clean later path.
        next_states.sort(key=lambda item: item[0], reverse=True)
        states = []
        per_bucket: defaultdict[int, int] = defaultdict(int)
        for state in next_states:
            bucket = state[1] // 8
            if per_bucket[bucket] >= 8:
                continue
            states.append(state)
            per_bucket[bucket] += 1
            if len(states) >= beam_width:
                break
    return max(states, key=lambda item: item[0])[2] if states else [None] * len(phrases)


def nontrivial_alternative_margin(
    selected: Candidate,
    candidates: list[Candidate],
) -> float:
    alternatives: list[Candidate] = []
    for candidate in candidates:
        if candidate is selected:
            continue
        intersection = max(
            0,
            min(selected.word_end, candidate.word_end)
            - max(selected.word_start, candidate.word_start),
        )
        union = max(selected.word_end, candidate.word_end) - min(
            selected.word_start, candidate.word_start
        )
        overlap = intersection / max(1, union)
        if overlap < 0.50 or abs(selected.source_start - candidate.source_start) > 0.8:
            alternatives.append(candidate)
    best_alternative = max((candidate.score for candidate in alternatives), default=-1.0)
    return selected.score - best_alternative


def has_text_equivalent_alternative(
    selected: Candidate,
    candidates: list[Candidate],
    aliases: dict[str, list[str]],
) -> bool:
    selected_tokens = tokenize(selected.transcript, aliases)
    found_competitive = False
    for candidate in candidates:
        if candidate is selected:
            continue
        if candidate.score < selected.score - 0.018:
            continue
        intersection = max(
            0,
            min(selected.word_end, candidate.word_end)
            - max(selected.word_start, candidate.word_start),
        )
        union = max(selected.word_end, candidate.word_end) - min(
            selected.word_start, candidate.word_start
        )
        overlap = intersection / max(1, union)
        if overlap >= 0.50 and abs(selected.source_start - candidate.source_start) <= 0.8:
            continue
        found_competitive = True
        candidate_tokens = tokenize(candidate.transcript, aliases)
        f1, recall, precision, edit_similarity = sequence_stats(
            selected_tokens, candidate_tokens
        )
        equivalent = (
            edit_similarity >= 0.93
            and f1 >= 0.92
            and recall >= 0.90
            and precision >= 0.90
        )
        if not equivalent:
            return False
    return found_competitive


def retry_insertion_split_point(
    reference: list[str],
    observed: list[str],
) -> int | None:
    """Find a script boundary around a likely spoken restart insertion."""

    opcodes = difflib.SequenceMatcher(
        None,
        reference,
        observed,
        autojunk=False,
    ).get_opcodes()
    options: list[tuple[int, int]] = []
    for index, (tag, source_start, _, target_start, target_end) in enumerate(
        opcodes
    ):
        if tag != "insert" or target_end - target_start < 2:
            continue
        if not 0 < index < len(opcodes) - 1:
            continue
        previous = opcodes[index - 1]
        following = opcodes[index + 1]
        if previous[0] != "equal" or following[0] != "equal":
            continue
        left_match = previous[2] - previous[1]
        right_match = following[2] - following[1]
        if left_match < 3 or right_match < 3:
            continue
        if source_start < 3 or len(reference) - source_start < 3:
            continue
        options.append((target_end - target_start, source_start))
    return max(options)[1] if options else None


def split_text_at_token_position(
    text: str,
    token_position: int,
    aliases: dict[str, list[str]],
) -> list[str]:
    """Split original text at an exact normalized-token word boundary."""

    words = text.split()
    token_count = 0
    for word_index, word in enumerate(words):
        token_count += len(tokenize(word, aliases))
        if token_count == token_position:
            left = " ".join(words[: word_index + 1]).strip()
            right = " ".join(words[word_index + 1 :]).strip()
            return [left, right] if left and right else [text.strip()]
        if token_count > token_position:
            break
    return [text.strip()]


def path_is_globally_monotonic(path: list[Candidate | None]) -> bool:
    last_end = 0
    for candidate in path:
        if candidate is None:
            continue
        if candidate.word_start < last_end or candidate.word_end < candidate.word_start:
            return False
        last_end = candidate.word_end
    return True


def rejected_retry_record(
    phrase: ScriptPhrase,
    chosen: Candidate,
    reason: str,
    **diagnostics: Any,
) -> dict[str, Any]:
    return {
        "status": "rejected",
        "reason": reason,
        "coarse_phrase_index": phrase.phrase_index,
        "unit_number": phrase.unit_index + 1,
        "script": phrase.text,
        "coarse_word_start": chosen.word_start,
        "coarse_word_end": chosen.word_end,
        "coarse_transcript": chosen.transcript,
        "coarse_repeated_surplus": chosen.repeated_surplus,
        "coarse_recall": chosen.recall,
        "coarse_precision": chosen.precision,
        **diagnostics,
    }


def retry_repair_review_reasons(
    phrase: ScriptPhrase,
    chosen: Candidate,
    repair_records: list[dict[str, Any]],
) -> list[str]:
    rejected = any(
        record.get("status") == "rejected"
        and int(record.get("unit_number", -1)) == phrase.unit_index + 1
        and int(record.get("coarse_word_start", -1)) == chosen.word_start
        and int(record.get("coarse_word_end", -1)) == chosen.word_end
        for record in repair_records
    )
    return ["unsafe_retry_repair_rejected"] if rejected else []


def selection_quality_review_reasons(
    phrase: ScriptPhrase,
    chosen: Candidate,
) -> list[str]:
    short_phrase = len(phrase.tokens) <= 10
    minimum_recall = 0.90 if short_phrase else 0.85
    minimum_f1 = 0.85 if short_phrase else 0.82
    reasons: list[str] = []
    if chosen.recall < minimum_recall:
        reasons.append("low_script_recall")
    if chosen.precision < 0.80:
        reasons.append("low_take_precision")
    if chosen.f1 < minimum_f1:
        reasons.append("low_alignment_f1")
    return reasons


def refine_retry_path(
    phrases: list[ScriptPhrase],
    candidate_sets: list[list[Candidate]],
    path: list[Candidate | None],
    words: list[ObservedWord],
    aliases: dict[str, list[str]],
    allowed_missing_units: set[int],
    *,
    candidate_limit: int,
) -> tuple[
    list[ScriptPhrase],
    list[list[Candidate]],
    list[Candidate | None],
    list[dict[str, Any]],
]:
    """Replace a retry-heavy contiguous take with clean ordered subspans.

    This second pass is deliberately local.  The coarse global solve first
    establishes the correct sentence and source neighborhood.  Only a chosen
    take containing surplus repetition, a long internal pause, or severe extra
    words is subdivided; the repair is accepted only when every subphrase has a
    monotonic candidate and the combined script match improves.
    """

    refined_phrases: list[ScriptPhrase] = []
    refined_sets: list[list[Candidate]] = []
    refined_path: list[Candidate | None] = []
    repairs: list[dict[str, Any]] = []

    for index, (phrase, candidates, chosen) in enumerate(
        zip(phrases, candidate_sets, path)
    ):
        chosen_tokens = (
            span_tokens(words, chosen.word_start, chosen.word_end)
            if chosen is not None
            else []
        )
        retry_split = (
            retry_insertion_split_point(phrase.tokens, chosen_tokens)
            if chosen is not None
            else None
        )
        required_missing = (
            chosen is None
            and phrase.unit_index + 1 not in allowed_missing_units
            and len(phrase.tokens) >= 6
        )
        repairable_quality_mismatch = (
            chosen is not None
            and bool(selection_quality_review_reasons(phrase, chosen))
            and chosen.recall >= 0.75
            and chosen.precision >= 0.75
        )
        retry_heavy = chosen is not None and len(phrase.tokens) >= 8 and (
            retry_split is not None
            or chosen.repeated_surplus >= 1.0
            or chosen.max_gap > 1.20
            or (chosen.recall >= 0.85 and chosen.precision < 0.72)
            or repairable_quality_mismatch
        )
        needs_repair = required_missing or retry_heavy
        if not needs_repair:
            refined_phrases.append(phrase)
            refined_sets.append(candidates)
            refined_path.append(chosen)
            continue

        if retry_split is not None:
            parts = split_text_at_token_position(
                phrase.text,
                retry_split,
                aliases,
            )
        else:
            target_size = max(5, math.ceil(len(phrase.tokens) / 2))
            parts = split_long_clause(phrase.text, aliases, target_size)
        if len(parts) < 2:
            if retry_heavy and chosen is not None:
                repairs.append(
                    rejected_retry_record(phrase, chosen, "cannot_split_script")
                )
            refined_phrases.append(phrase)
            refined_sets.append(candidates)
            refined_path.append(chosen)
            continue

        previous_end = max(
            (
                candidate.word_end
                for candidate in refined_path
                if candidate is not None
            ),
            default=0,
        )
        following_start = min(
            (
                candidate.word_start
                for candidate in path[index + 1 :]
                if candidate is not None
            ),
            default=len(words),
        )
        if chosen is not None:
            search_start = max(previous_end, chosen.word_start - 18)
            search_end = min(following_start, chosen.word_end + 18)
        else:
            search_start = previous_end
            search_end = following_start
            if search_end - search_start > 180:
                refined_phrases.append(phrase)
                refined_sets.append(candidates)
                refined_path.append(chosen)
                continue
        if search_end <= search_start:
            if retry_heavy and chosen is not None:
                repairs.append(
                    rejected_retry_record(phrase, chosen, "invalid_search_window")
                )
            refined_phrases.append(phrase)
            refined_sets.append(candidates)
            refined_path.append(chosen)
            continue

        subphrases: list[ScriptPhrase] = []
        for part_index, part in enumerate(parts):
            tokens = tokenize(part, aliases)
            if not tokens:
                continue
            subphrases.append(
                ScriptPhrase(
                    phrase_index=part_index,
                    unit_index=phrase.unit_index,
                    phrase_in_unit=phrase.phrase_in_unit + part_index,
                    line_number=phrase.line_number,
                    paragraph_index=phrase.paragraph_index,
                    cue_before=phrase.cue_before and part_index == 0,
                    text=part,
                    tokens=tokens,
                    pause_after=(
                        phrase.pause_after
                        if part_index == len(parts) - 1
                        else "phrase"
                    ),
                )
            )
        sub_sets = [
            top_candidates(
                subphrase,
                words,
                search_start,
                search_end,
                limit=candidate_limit,
            )
            for subphrase in subphrases
        ]
        sub_path = select_global_path(
            subphrases,
            sub_sets,
            allowed_missing_units,
            beam_width=96,
        )
        selected_subs = [candidate for candidate in sub_path if candidate is not None]
        if len(selected_subs) != len(subphrases):
            if retry_heavy and chosen is not None:
                repairs.append(
                    rejected_retry_record(
                        phrase,
                        chosen,
                        "incomplete_subphrase_path",
                    )
                )
            refined_phrases.append(phrase)
            refined_sets.append(candidates)
            refined_path.append(chosen)
            continue
        repair_is_monotonic = all(
            left.word_end <= right.word_start
            for left, right in zip(selected_subs, selected_subs[1:])
        )
        repair_is_inside_fences = (
            selected_subs[0].word_start >= previous_end
            and selected_subs[-1].word_end <= following_start
        )
        if not (repair_is_monotonic and repair_is_inside_fences):
            if retry_heavy and chosen is not None:
                repairs.append(
                    rejected_retry_record(
                        phrase,
                        chosen,
                        "non_monotonic_or_out_of_fence_repair",
                    )
                )
            refined_phrases.append(phrase)
            refined_sets.append(candidates)
            refined_path.append(chosen)
            continue

        combined_tokens: list[str] = []
        for candidate in selected_subs:
            combined_tokens.extend(
                span_tokens(words, candidate.word_start, candidate.word_end)
            )
        combined_f1, combined_recall, combined_precision, _ = sequence_stats(
            phrase.tokens,
            combined_tokens,
        )
        combined_repeat = repeated_surplus(phrase.tokens, combined_tokens)
        combined_max_gap = max(
            (candidate.max_gap for candidate in selected_subs),
            default=0.0,
        )
        strong_subphrase_coverage = all(
            candidate.recall >= (0.90 if len(subphrase.tokens) <= 10 else 0.85)
            and candidate.precision >= 0.80
            and candidate.f1 >= (0.85 if len(subphrase.tokens) <= 10 else 0.82)
            for subphrase, candidate in zip(subphrases, selected_subs)
        )
        if chosen is None:
            improvement = True
            complete_enough = (
                strong_subphrase_coverage
                and combined_recall >= 0.90
                and combined_precision >= 0.85
                and combined_f1 >= 0.87
            )
        else:
            improvement = (
                combined_repeat + 0.35 < chosen.repeated_surplus
                or combined_precision >= chosen.precision + 0.08
                or (
                    chosen.max_gap > 1.20
                    and combined_max_gap <= 1.0
                    and combined_precision >= chosen.precision - 0.03
                )
            )
            complete_enough = (
                strong_subphrase_coverage
                and combined_recall + 1e-9 >= chosen.recall
                and combined_precision + 1e-9 >= chosen.precision
                and combined_f1 + 0.01 >= chosen.f1
            )
        if not (improvement and complete_enough):
            if retry_heavy and chosen is not None:
                repairs.append(
                    rejected_retry_record(
                        phrase,
                        chosen,
                        "insufficient_repair_quality",
                        repaired_recall=combined_recall,
                        repaired_precision=combined_precision,
                        repaired_f1=combined_f1,
                        strong_subphrase_coverage=strong_subphrase_coverage,
                    )
                )
            refined_phrases.append(phrase)
            refined_sets.append(candidates)
            refined_path.append(chosen)
            continue

        repairs.append(
            {
                "status": "accepted",
                "coarse_phrase_index": phrase.phrase_index,
                "unit_number": phrase.unit_index + 1,
                "script": phrase.text,
                "coarse_word_start": chosen.word_start if chosen else None,
                "coarse_word_end": chosen.word_end if chosen else None,
                "coarse_transcript": chosen.transcript if chosen else None,
                "repaired_transcripts": [
                    candidate.transcript for candidate in selected_subs
                ],
                "coarse_repeated_surplus": (
                    chosen.repeated_surplus if chosen else None
                ),
                "repaired_repeated_surplus": combined_repeat,
                "coarse_max_gap": chosen.max_gap if chosen else None,
                "repaired_max_gap": combined_max_gap,
                "coarse_precision": chosen.precision if chosen else None,
                "repaired_precision": combined_precision,
                "repaired_recall": combined_recall,
            }
        )
        refined_phrases.extend(subphrases)
        refined_sets.extend(sub_sets)
        refined_path.extend(sub_path)

    per_unit: defaultdict[int, int] = defaultdict(int)
    for phrase_index, (phrase, candidates, chosen) in enumerate(
        zip(refined_phrases, refined_sets, refined_path)
    ):
        phrase.phrase_index = phrase_index
        phrase.phrase_in_unit = per_unit[phrase.unit_index]
        per_unit[phrase.unit_index] += 1
        for candidate in candidates:
            candidate.phrase_index = phrase_index
        if chosen is not None:
            chosen.phrase_index = phrase_index
    if not path_is_globally_monotonic(refined_path):
        raise RuntimeError(
            "Internal error: retry refinement produced overlapping selections."
        )
    return refined_phrases, refined_sets, refined_path, repairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aliases", type=Path)
    parser.add_argument(
        "--allow-missing-unit",
        type=int,
        action="append",
        default=[],
    )
    parser.add_argument("--max-phrase-tokens", type=int, default=18)
    parser.add_argument("--candidate-limit", type=int, default=14)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aliases = load_aliases(args.aliases)
    units = parse_script(args.script, aliases)
    phrases = build_phrases(units, aliases, max_tokens=args.max_phrase_tokens)
    words = load_observed_words(args.transcript, aliases)
    anchors = phrase_anchors(phrases, words)
    windows = search_windows(anchors, len(words))
    inverted: defaultdict[str, list[int]] = defaultdict(list)
    for word in words:
        for token in set(word.tokens):
            inverted[token].append(word.word_index)
    candidate_sets: list[list[Candidate]] = []
    retrieval_windows: list[list[list[int]]] = []
    for phrase in phrases:
        primary = windows[phrase.phrase_index]
        alternatives = alternate_search_windows(
            phrase,
            words,
            inverted,
            primary,
        )
        all_windows = [primary, *alternatives]
        candidate_sets.append(
            combine_candidate_windows(
                phrase,
                words,
                all_windows,
                limit=args.candidate_limit,
            )
        )
        retrieval_windows.append([[start, end] for start, end in all_windows])
    allowed_missing = set(args.allow_missing_unit)
    path = select_global_path(phrases, candidate_sets, allowed_missing)
    phrases, candidate_sets, path, repair_records = refine_retry_path(
        phrases,
        candidate_sets,
        path,
        words,
        aliases,
        allowed_missing,
        candidate_limit=args.candidate_limit,
    )

    selections: list[dict[str, Any]] = []
    missing_required: list[int] = []
    review_items: list[dict[str, Any]] = []
    for phrase, chosen, candidates in zip(phrases, path, candidate_sets):
        if chosen is None:
            allowed = phrase.unit_index + 1 in allowed_missing
            item = {
                "phrase_index": phrase.phrase_index,
                "unit_index": phrase.unit_index,
                "unit_number": phrase.unit_index + 1,
                "text": phrase.text,
                "status": "allowed_missing" if allowed else "missing",
                "candidates": [asdict(candidate) for candidate in candidates[:4]],
            }
            selections.append(item)
            if not allowed:
                missing_required.append(phrase.phrase_index)
                review_items.append(
                    {
                        "severity": "fail",
                        "kind": "missing_script_phrase",
                        "phrase_index": phrase.phrase_index,
                        "unit_number": phrase.unit_index + 1,
                        "script": phrase.text,
                    }
                )
            continue

        margin = nontrivial_alternative_margin(chosen, candidates)
        reasons = retry_repair_review_reasons(
            phrase,
            chosen,
            repair_records,
        )
        reasons.extend(selection_quality_review_reasons(phrase, chosen))
        if margin < 0.018 and not has_text_equivalent_alternative(
            chosen, candidates, aliases
        ):
            reasons.append("ambiguous_alternative_take")
        if chosen.repeated_surplus >= 2.0:
            reasons.append("surplus_repetition_inside_take")
        if chosen.max_gap > 1.2:
            reasons.append("long_pause_inside_take")
        status = "review" if reasons else "selected"
        item = {
            "phrase_index": phrase.phrase_index,
            "unit_index": phrase.unit_index,
            "unit_number": phrase.unit_index + 1,
            "phrase_in_unit": phrase.phrase_in_unit,
            "line_number": phrase.line_number,
            "paragraph_index": phrase.paragraph_index,
            "cue_before": phrase.cue_before,
            "script": phrase.text,
            "script_tokens": phrase.tokens,
            "pause_after": phrase.pause_after,
            "status": status,
            "review_reasons": reasons,
            "score_margin": margin,
            "candidate": asdict(chosen),
            "alternatives": [asdict(candidate) for candidate in candidates[:5]],
        }
        selections.append(item)
        for reason in reasons:
            review_items.append(
                {
                    "severity": "review",
                    "kind": reason,
                    "phrase_index": phrase.phrase_index,
                    "unit_number": phrase.unit_index + 1,
                    "script": phrase.text,
                    "source_start": chosen.source_start,
                    "source_end": chosen.source_end,
                }
            )

    selected_items = [item for item in selections if item["status"] in {"selected", "review"}]
    ctc_segments = []
    duration = float(read_json(args.analysis)["duration"])
    for item in selected_items:
        candidate = item["candidate"]
        ctc_segments.append(
            {
                "phrase_index": int(item["phrase_index"]),
                "start": max(0.0, float(candidate["source_start"]) - 0.24),
                "end": min(duration, float(candidate["source_end"]) + 0.28),
                "text": str(candidate["transcript"]),
            }
        )
    ctc_input = {
        "segments": ctc_segments,
        "language": "en",
    }
    write_json(args.output_dir / "ctc_input.json", ctc_input)
    plan = {
        "schema_version": 1,
        "audio": str(args.audio.resolve()),
        "audio_sha256": sha256_file(args.audio),
        "script": str(args.script.resolve()),
        "script_sha256": sha256_file(args.script),
        "aliases": aliases,
        "allowed_missing_units": sorted(allowed_missing),
        "unit_count": len(units),
        "phrase_count": len(phrases),
        "observed_word_count": len(words),
        "coarse_retrieval_windows": retrieval_windows,
        "retry_repairs": repair_records,
        "phrases": [asdict(phrase) for phrase in phrases],
        "selections": selections,
        "missing_required_phrase_indices": missing_required,
        "review_items": review_items,
        "ctc_segment_phrase_indices": [
            int(item["phrase_index"]) for item in selected_items
        ],
    }
    write_json(args.output_dir / "edit_plan.json", plan)
    print(
        json.dumps(
            {
                "units": len(units),
                "phrases": len(phrases),
                "observed_words": len(words),
                "selected": len(selected_items),
                "missing_required": len(missing_required),
                "review_items": len(review_items),
                "edit_plan": str((args.output_dir / "edit_plan.json").resolve()),
                "ctc_input": str((args.output_dir / "ctc_input.json").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
