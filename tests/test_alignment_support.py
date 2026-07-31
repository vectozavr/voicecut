from __future__ import annotations

from typing import Any

from voicecut.final_render import _alignment_spans, evaluate_retained_word_support


def _span(
    *,
    word_id: int,
    text: str,
    word_score: float,
    scores: list[float],
    starts: list[float] | None = None,
) -> dict[str, Any]:
    selected_starts = starts or [word_id + index * 0.02 for index in range(len(text))]
    characters = []
    score_index = 0
    for character_index, character in enumerate(text):
        if not character.isalpha():
            continue
        start = selected_starts[character_index]
        characters.append(
            {
                "character": character,
                "start": start,
                "end": start + 0.02,
                "score": scores[score_index],
                "is_alphabetic": True,
            }
        )
        score_index += 1
    return {
        "word_id": word_id,
        "text": text,
        "source_text": text,
        "word_score": word_score,
        "characters": characters,
    }


def test_support_gate_rejects_bad_familiar_without_rejecting_good_words() -> None:
    bad = _span(
        word_id=32,
        text="familiar",
        word_score=0.783,
        scores=[0.997, 0.725, 1.0, 0.793, 0.990, 0.849, 0.414, 0.495],
    )
    good = _span(
        word_id=35,
        text="familiar",
        word_score=0.830,
        scores=[0.992, 0.998, 0.695, 0.984, 0.690, 0.722, 1.0, 0.557],
    )
    valid_s = _span(
        word_id=60,
        text="operations",
        word_score=0.915,
        scores=[0.771, 0.994, 0.631, 0.993, 0.999, 0.991, 0.999, 0.999, 0.711, 0.996],
    )
    context = [bad, good, valid_s]

    rejected = evaluate_retained_word_support(bad, context, edge="terminal")
    accepted_retry = evaluate_retained_word_support(good, context, edge="terminal")
    accepted_s = evaluate_retained_word_support(valid_s, context, edge="terminal")

    assert rejected["status"] == "weak_terminal_word_support"
    assert rejected["minimum_edge_score"] == 0.414
    assert rejected["edge_to_context_score_ratio"] < 0.55
    assert accepted_retry["status"] == "supported_complete_word"
    assert accepted_retry["minimum_edge_score"] == 0.557
    assert accepted_s["status"] == "supported_complete_word"
    assert accepted_s["edge_character_scores"][-1] == 0.996


def test_support_gate_reports_weak_initial_word_support() -> None:
    weak = _span(
        word_id=4,
        text="example",
        word_score=0.70,
        scores=[0.30, 0.40, 0.45, 0.95, 0.95, 0.95, 0.95],
    )
    context = [
        weak,
        _span(
            word_id=5,
            text="context",
            word_score=0.95,
            scores=[0.95] * 7,
        ),
    ]

    support = evaluate_retained_word_support(weak, context, edge="initial")

    assert support["status"] == "weak_initial_word_support"
    assert support["complete_character_coverage"] is True
    assert support["monotonic_character_timestamps"] is True


def test_support_gate_requires_complete_ordered_character_evidence() -> None:
    incomplete = _span(
        word_id=1,
        text="words",
        word_score=0.99,
        scores=[0.99] * 5,
    )
    incomplete["characters"] = incomplete["characters"][:-1]
    invalid = _span(
        word_id=2,
        text="audio",
        word_score=0.99,
        scores=[0.99] * 5,
        starts=[2.0, 2.02, 2.01, 2.06, 2.08],
    )

    incomplete_result = evaluate_retained_word_support(
        incomplete,
        [incomplete],
        edge="terminal",
    )
    invalid_result = evaluate_retained_word_support(
        invalid,
        [invalid],
        edge="terminal",
    )

    assert incomplete_result["status"] == "incomplete_character_coverage"
    assert incomplete_result["character_coverage"] == 0.8
    assert invalid_result["status"] == "invalid_alignment_geometry"


def test_alignment_spans_preserve_zero_score_character_evidence() -> None:
    local_word = {"id": 0, "text": "familiar", "start": 0.1, "end": 0.9}
    characters = []
    for index, (character, score) in enumerate(
        zip("familiar", [1.0, 0.7, 0.8, 0.9, 0.8, 0.9, 0.8, 0.0], strict=True)
    ):
        characters.append(
            {
                "char": character,
                "start": 0.1 + index * 0.1,
                "end": 0.2 + index * 0.1,
                "score": score,
            }
        )
    job = {
        "crop_start_seconds": 0.0,
        "crop_end_seconds": 1.0,
        "local_words": [local_word],
    }
    worker_job = {
        "error": None,
        "aligned": {
            "word_segments": [
                {"word": "familiar", "start": 0.1, "end": 0.9, "score": 0.789}
            ],
            "segments": [
                {
                    "words": [
                        {
                            "word": "familiar",
                            "start": 0.1,
                            "end": 0.9,
                            "score": 0.789,
                        }
                    ],
                    "chars": characters,
                }
            ],
        },
    }

    span = _alignment_spans(
        job=job,
        worker_job=worker_job,
        sample_rate=1000,
        total_samples=1000,
    )[0]

    assert span["word_score"] == 0.789
    assert span["expected_alignable_character_count"] == 8
    assert span["aligned_character_count"] == 8
    assert span["character_coverage"] == 1.0
    assert span["characters"][-1]["character"] == "r"
    assert span["characters"][-1]["score"] == 0.0
    assert span["terminal_edge_score"] == 0.0
    assert span["aligned_end"] == 0.9
