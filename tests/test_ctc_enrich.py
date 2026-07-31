from __future__ import annotations

from voicecut.ctc_enrich import build_alignment_input, enrich_transcript


def _word(word: str, start: float, end: float, score: float = 0.95) -> dict:
    return {"word": word, "start": start, "end": end, "score": score}


def test_hidden_retry_expands_physical_word_occurrences() -> None:
    transcript = {
        "schema_version": 1,
        "audio": "/tmp/source.wav",
        "audio_sha256": "abc",
        "engine": "mlx_whisper",
        "atoms": [
            {
                "atom_index": 11,
                "start": 1.0,
                "end": 3.0,
                "text": "begin with the words",
                "words": [
                    _word("begin", 1.0, 1.3),
                    _word("with", 1.4, 1.6),
                    _word("the", 1.6, 2.2),
                    _word("words", 2.2, 2.5),
                ],
            },
            {
                "atom_index": 3,
                "start": 4.0,
                "end": 5.0,
                "text": "next sentence",
                "words": [
                    _word("next", 4.0, 4.3),
                    _word("sentence", 4.4, 4.9),
                ],
            },
        ],
    }
    greedy = [
        _word("begin", 1.00, 1.30),
        _word("with", 1.40, 1.60),
        _word("familiar", 1.65, 1.95),
        _word("with", 2.10, 2.30),
        _word("the", 2.35, 2.45),
        _word("words", 2.50, 2.80),
    ]
    alignment = {
        "segments": [
            {
                "phrase_index": 3,
                "greedy_ctc_words": [],
                "acoustic_insertions": [],
                "acoustic_expected_substitutions": [],
            },
            {
                "phrase_index": 11,
                "greedy_ctc_words": greedy,
                "acoustic_insertions": [
                    {
                        "type": "spoken_retry",
                        "reason": "greedy_ctc_restart_before_selected_take",
                        "confidence": 0.91,
                        "safe_edit_start": 1.30,
                        "safe_edit_end": 2.10,
                        "words": [greedy[1], greedy[2]],
                        "left_anchor": {"word": "begin"},
                        "right_anchor": {"word": "with"},
                    }
                ],
            },
        ]
    }

    enriched, report = enrich_transcript(
        transcript=transcript,
        alignment=alignment,
    )

    expanded = enriched["atoms"][0]
    assert expanded["text"] == ("begin with familiar with the words")
    assert [word["ctc_lexical_source"] for word in expanded["words"]] == [
        "whisper_expected_match",
        "raw_unmatched_occurrence",
        "raw_unmatched_occurrence",
        "whisper_expected_match",
        "whisper_expected_match",
        "whisper_expected_match",
    ]
    assert enriched["atoms"][1]["text"] == "next sentence"
    assert report["atoms_expanded"] == 1
    assert report["hidden_retries_recovered"] == 1


def test_short_primary_words_are_restored_between_exact_ctc_anchors() -> None:
    transcript = {
        "atoms": [
            {
                "atom_index": 0,
                "start": 1.0,
                "end": 4.0,
                "text": "and it might begin with the familiar words",
                "words": [],
            }
        ]
    }
    greedy = [
        _word("and", 1.00, 1.10),
        _word("at", 1.12, 1.20, 0.85),
        _word("myg", 1.22, 1.35, 0.70),
        _word("begin", 1.37, 1.60),
        _word("with", 1.62, 1.80),
        _word("familiar", 1.82, 2.20),
        _word("with", 2.40, 2.58),
        _word("the", 2.60, 2.70),
        _word("familiar", 2.72, 3.10),
        _word("words", 3.12, 3.40),
    ]
    retry = {
        "type": "spoken_retry",
        "reason": "greedy_ctc_restart_before_selected_take",
        "confidence": 0.84,
        "safe_edit_start": 1.60,
        "safe_edit_end": 2.40,
        "words": greedy[4:6],
        "left_anchor": {"word": "begin"},
        "right_anchor": {"word": "with"},
    }
    alignment = {
        "segments": [
            {
                "phrase_index": 0,
                "greedy_ctc_words": greedy,
                "acoustic_insertions": [retry],
            }
        ]
    }

    enriched, _ = enrich_transcript(
        transcript=transcript,
        alignment=alignment,
    )

    words = enriched["atoms"][0]["words"]
    assert [word["word"] for word in words] == [
        "and",
        "it",
        "might",
        "begin",
        "with",
        "familiar",
        "with",
        "the",
        "familiar",
        "words",
    ]
    assert words[1]["ctc_observed_word"] == "at"
    assert words[2]["ctc_observed_word"] == "myg"
    assert words[1]["ctc_lexical_source"] == (
        "whisper_contextually_grounded_substitution"
    )
    assert words[4]["ctc_lexical_source"] == "raw_unmatched_occurrence"
    assert words[5]["ctc_lexical_source"] == "raw_unmatched_occurrence"


def test_low_confidence_insertion_does_not_change_words() -> None:
    transcript = {
        "atoms": [
            {
                "atom_index": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "keep this",
                "words": [
                    _word("keep", 0.1, 0.3),
                    _word("this", 0.4, 0.7),
                ],
            }
        ]
    }
    alignment = {
        "segments": [
            {
                "phrase_index": 0,
                "greedy_ctc_words": [
                    _word("keep", 0.1, 0.3),
                    _word("extra", 0.31, 0.39),
                    _word("this", 0.4, 0.7),
                ],
                "acoustic_insertions": [
                    {
                        "type": "spoken_retry",
                        "reason": "greedy_ctc_restart_before_selected_take",
                        "confidence": 0.4,
                        "safe_edit_start": 0.3,
                        "safe_edit_end": 0.4,
                        "words": [_word("extra", 0.31, 0.39)],
                        "left_anchor": {"word": "keep"},
                        "right_anchor": {"word": "this"},
                    }
                ],
            }
        ]
    }

    enriched, report = enrich_transcript(
        transcript=transcript,
        alignment=alignment,
    )

    assert enriched["atoms"][0]["text"] == "keep this"
    assert enriched["atoms"][0]["ctc_enrichment"]["status"] == (
        "unchanged_no_hidden_retry"
    )
    assert report["atoms_expanded"] == 0


def test_alignment_input_uses_complete_atom_text_and_times() -> None:
    transcript = {
        "atoms": [
            {
                "atom_index": 7,
                "start": 2.5,
                "end": 4.0,
                "text": "A complete atom.",
                "words": [],
            }
        ]
    }

    assert build_alignment_input(transcript) == {
        "schema_version": 1,
        "segments": [
            {
                "phrase_index": 7,
                "start": 2.5,
                "end": 4.0,
                "text": "A complete atom.",
            }
        ],
    }
