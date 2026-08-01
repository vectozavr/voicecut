from __future__ import annotations

import copy
import subprocess
from unittest import mock

import pytest

from voicecut.common import read_json, sha256_file, write_json
from voicecut.ctc_enrich import (
    CtcEnrichmentError,
    build_alignment_input,
    enrich_transcript,
    run_enrichment,
    write_passthrough_enrichment,
)


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
                "atom_index": 3,
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
                "atom_index": 11,
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
                "phrase_index": 11,
                "greedy_ctc_words": [],
                "acoustic_insertions": [],
                "acoustic_expected_substitutions": [],
            },
            {
                "phrase_index": 3,
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


def test_collapsed_whisper_suffix_is_pruned_when_ctc_ends_on_supported_prefix() -> None:
    transcript = {
        "atoms": [
            {
                "atom_index": 678,
                "start": 1982.17,
                "end": 1983.61,
                "text": "Pruning and fine tuning are a little bit more.",
                "words": [
                    _word("Pruning", 1982.17, 1982.51),
                    _word("and", 1982.51, 1982.73),
                    _word("fine", 1982.73, 1982.99),
                    _word("tuning", 1982.99, 1983.25),
                    _word("are", 1983.25, 1983.59),
                    _word("a", 1983.59, 1983.59),
                    _word("little", 1983.59, 1983.59),
                    _word("bit", 1983.59, 1983.59),
                    _word("more.", 1983.59, 1983.59),
                ],
            }
        ]
    }
    alignment = {
        "segments": [
            {
                "phrase_index": 678,
                "greedy_ctc_words": [
                    _word("brooning", 1982.23, 1982.58),
                    _word("and", 1982.64, 1982.72),
                    _word("fint", 1982.78, 1983.02),
                    _word("uning", 1983.14, 1983.37),
                ],
                "acoustic_insertions": [],
            }
        ]
    }

    enriched, report = enrich_transcript(
        transcript=transcript,
        alignment=alignment,
    )

    atom = enriched["atoms"][0]
    assert atom["text"] == "Pruning and fine tuning"
    assert [word["word"] for word in atom["words"]] == [
        "Pruning",
        "and",
        "fine",
        "tuning",
    ]
    assert atom["ctc_enrichment"]["status"] == ("pruned_whisper_hallucinated_suffix")
    assert report["hallucinated_suffixes_pruned"] == 1
    assert report["pruned_suffixes"][0]["pruned_words"] == [
        "are",
        "a",
        "little",
        "bit",
        "more.",
    ]


def test_ctc_prefix_does_not_prune_ordinarily_timed_whisper_suffix() -> None:
    transcript = {
        "atoms": [
            {
                "atom_index": 1,
                "start": 0.0,
                "end": 2.0,
                "text": "keep this real suffix",
                "words": [
                    _word("keep", 0.1, 0.4),
                    _word("this", 0.5, 0.8),
                    _word("real", 0.9, 1.2),
                    _word("suffix", 1.3, 1.7),
                ],
            }
        ]
    }
    alignment = {
        "segments": [
            {
                "phrase_index": 1,
                "greedy_ctc_words": [
                    _word("keep", 0.1, 0.4),
                    _word("this", 0.5, 0.8),
                ],
                "acoustic_insertions": [],
            }
        ]
    }

    enriched, report = enrich_transcript(
        transcript=transcript,
        alignment=alignment,
    )

    assert enriched["atoms"][0]["text"] == "keep this real suffix"
    assert report["hallucinated_suffixes_pruned"] == 0


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


def test_empty_atom_is_explicitly_skipped_and_preserved_unchanged() -> None:
    atom = {
        "atom_index": 271,
        "start": 802.0,
        "end": 802.4,
        "text": "",
        "words": [],
        "segments": [],
    }
    transcript = {"atoms": [copy.deepcopy(atom)]}
    alignment_input = build_alignment_input(transcript)

    assert alignment_input["segments"] == []
    assert alignment_input["skipped_segments"] == [
        {
            "phrase_index": 271,
            "input_start": 802.0,
            "input_end": 802.4,
            "reason": "no_lexical_content",
        }
    ]

    enriched, report = enrich_transcript(
        transcript=transcript,
        alignment={
            "segments": [],
            "skipped_segments": alignment_input["skipped_segments"],
        },
    )

    output_atom = enriched["atoms"][0]
    for key, value in atom.items():
        assert output_atom[key] == value
    assert output_atom["ctc_enrichment"]["status"] == ("unchanged_no_lexical_content")
    assert report["atoms_skipped_no_lexical_content"] == 1
    assert report["skipped_atoms"] == [
        {
            "atom_index": 271,
            "reason": "no_lexical_content",
            "explicit_worker_skip": True,
        }
    ]


def test_empty_atom_text_derives_ctc_text_from_valid_word_ledger() -> None:
    words = [_word("Stable", 1.1, 1.4), _word("words.", 1.5, 1.9)]
    transcript = {
        "atoms": [
            {
                "atom_index": 9,
                "start": 1.0,
                "end": 2.0,
                "text": "",
                "words": copy.deepcopy(words),
            }
        ]
    }

    alignment_input = build_alignment_input(transcript)
    assert alignment_input["segments"] == [
        {
            "phrase_index": 9,
            "start": 1.0,
            "end": 2.0,
            "text": "Stable words.",
            "text_source": "derived_from_word_ledger",
        }
    ]

    enriched, report = enrich_transcript(
        transcript=transcript,
        alignment={
            "segments": [
                {
                    "phrase_index": 9,
                    "greedy_ctc_words": [],
                    "acoustic_insertions": [],
                    "acoustic_expected_substitutions": [],
                }
            ]
        },
    )
    assert enriched["atoms"][0]["text"] == ""
    assert enriched["atoms"][0]["words"] == words
    assert report["atoms_skipped_no_lexical_content"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda atoms: atoms.append(copy.deepcopy(atoms[0])), "repeats atom_index"),
        (lambda atoms: atoms[0].update(end=float("nan")), "timestamp geometry"),
        (lambda atoms: atoms[0].update(end=0.5), "timestamp geometry"),
        (
            lambda atoms: atoms[0]["words"][0].update(start=float("inf")),
            "invalid geometry",
        ),
        (lambda atoms: atoms[0]["words"][0].update(end=0.5), "invalid geometry"),
    ],
)
def test_structural_transcript_corruption_remains_fatal(mutation, message) -> None:
    atoms = [
        {
            "atom_index": 1,
            "start": 1.0,
            "end": 2.0,
            "text": "valid",
            "words": [_word("valid", 1.1, 1.5)],
        }
    ]
    mutation(atoms)
    with pytest.raises(CtcEnrichmentError, match=message):
        build_alignment_input({"atoms": atoms})


def test_negative_or_decreasing_atom_ids_are_rejected() -> None:
    negative = {
        "atom_index": -1,
        "start": 0.0,
        "end": 1.0,
        "text": "invalid id",
        "words": [_word("invalid", 0.1, 0.4)],
    }
    with pytest.raises(CtcEnrichmentError, match="atom_index"):
        build_alignment_input({"atoms": [negative]})

    atoms = [
        {
            "atom_index": 9,
            "start": 0.0,
            "end": 1.0,
            "text": "first",
            "words": [_word("first", 0.1, 0.5)],
        },
        {
            "atom_index": 4,
            "start": 2.0,
            "end": 3.0,
            "text": "second",
            "words": [_word("second", 2.1, 2.5)],
        },
    ]
    with pytest.raises(CtcEnrichmentError, match="IDs are not chronological"):
        build_alignment_input({"atoms": atoms})


def test_overlapping_atom_time_or_sample_ranges_are_rejected() -> None:
    first = {
        "atom_index": 0,
        "start_sample": 0,
        "end_sample": 16_000,
        "start": 0.0,
        "end": 1.0,
        "duration": 1.0,
        "text": "first",
        "words": [_word("first", 0.1, 0.5)],
    }
    time_overlap = {
        "atom_index": 1,
        "start_sample": 16_000,
        "end_sample": 32_000,
        "start": 0.9,
        "end": 2.0,
        "duration": 1.1,
        "text": "second",
        "words": [_word("second", 1.1, 1.5)],
    }
    with pytest.raises(CtcEnrichmentError, match="atoms overlap"):
        build_alignment_input({"atoms": [first, time_overlap]})

    sample_overlap = {
        **time_overlap,
        "start": 1.0,
        "start_sample": 15_999,
    }
    with pytest.raises(CtcEnrichmentError, match="sample ranges overlap"):
        build_alignment_input({"atoms": [first, sample_overlap]})


@pytest.mark.parametrize(
    "words",
    [
        [_word("early", 0.99, 1.2)],
        [_word("late", 1.2, 2.01)],
        [_word("first", 1.1, 1.5), _word("overlap", 1.49, 1.8)],
    ],
)
def test_words_outside_or_overlapping_inside_an_atom_are_rejected(words) -> None:
    transcript = {
        "atoms": [
            {
                "atom_index": 0,
                "start": 1.0,
                "end": 2.0,
                "text": "word geometry",
                "words": words,
            }
        ]
    }
    with pytest.raises(
        CtcEnrichmentError,
        match="outside its atom|word ledger is not chronological",
    ):
        build_alignment_input(transcript)


def test_rounding_tolerance_and_zero_duration_whisper_words_remain_valid() -> None:
    transcript = {
        "atoms": [
            {
                "atom_index": 0,
                "start": 1.0,
                "end": 2.0,
                "text": "valid collapsed words",
                "words": [
                    _word("valid", 1.0 - 5e-7, 1.5),
                    _word("collapsed", 1.5, 1.5),
                    _word("words", 1.5, 2.0 + 5e-7),
                ],
            },
            {
                "atom_index": 1,
                "start": 2.0 - 5e-7,
                "end": 3.0,
                "text": "next atom",
                "words": [_word("next", 2.1, 2.5)],
            },
        ]
    }

    alignment_input = build_alignment_input(transcript)

    assert [item["phrase_index"] for item in alignment_input["segments"]] == [0, 1]


def test_1300_atom_transcript_skips_sparse_empty_atoms_without_aborting() -> None:
    atoms = []
    for atom_index in range(1300):
        start = float(atom_index * 2)
        if atom_index % 101 == 0:
            atoms.append(
                {
                    "atom_index": atom_index,
                    "start": start,
                    "end": start + 0.4,
                    "text": "",
                    "words": [],
                }
            )
        else:
            atoms.append(
                {
                    "atom_index": atom_index,
                    "start": start,
                    "end": start + 1.0,
                    "text": f"atom {atom_index}",
                    "words": [
                        _word("atom", start + 0.1, start + 0.3),
                        _word(str(atom_index), start + 0.4, start + 0.7),
                    ],
                }
            )
    transcript = {"atoms": atoms}
    alignment_input = build_alignment_input(transcript)
    alignment = {
        "segments": [
            {
                "phrase_index": segment["phrase_index"],
                "greedy_ctc_words": [],
                "acoustic_insertions": [],
                "acoustic_expected_substitutions": [],
            }
            for segment in alignment_input["segments"]
        ],
        "skipped_segments": alignment_input["skipped_segments"],
    }

    enriched, report = enrich_transcript(
        transcript=transcript,
        alignment=alignment,
    )

    assert len(enriched["atoms"]) == 1300
    assert report["atoms_examined"] == 1300
    assert report["atoms_skipped_no_lexical_content"] == 13
    assert report["atoms_with_decode_failures"] == 0


def test_per_segment_decode_failure_is_preserved_as_primary_whisper() -> None:
    transcript = {
        "atoms": [
            {
                "atom_index": 4,
                "start": 2.0,
                "end": 3.0,
                "text": "keep this atom",
                "words": [_word("keep", 2.1, 2.4), _word("this", 2.5, 2.8)],
            }
        ]
    }
    alignment = {
        "segments": [
            {
                "phrase_index": 4,
                "input_start": 2.0,
                "input_end": 3.0,
                "input_text": "keep this atom",
                "greedy_ctc_words": [],
                "acoustic_insertions": [],
                "acoustic_expected_substitutions": [],
                "status": "decode_failed",
                "decode_error": {"type": "RuntimeError", "message": "bad crop"},
            }
        ]
    }

    enriched, report = enrich_transcript(
        transcript=transcript,
        alignment=alignment,
    )

    assert enriched["atoms"][0]["text"] == "keep this atom"
    assert enriched["atoms"][0]["ctc_enrichment"]["status"] == (
        "unchanged_ctc_decode_failure"
    )
    assert report["status"] == "complete_with_degraded_evidence"
    assert report["atoms_with_decode_failures"] == 1


def test_global_worker_failure_writes_validated_primary_passthrough(tmp_path) -> None:
    audio_path = tmp_path / "source.wav"
    transcript_path = tmp_path / "source_transcript.json"
    output_dir = tmp_path / "ctc"
    audio_path.write_bytes(b"canonical audio identity")
    transcript = {
        "audio_sha256": sha256_file(audio_path),
        "engine": "mlx-whisper",
        "atoms": [
            {
                "atom_index": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "keep this",
                "words": [_word("keep", 0.1, 0.4), _word("this", 0.5, 0.8)],
            }
        ],
    }
    write_json(transcript_path, transcript)

    with mock.patch(
        "voicecut.ctc_enrich.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["align_ctc"],
            returncode=9,
            stdout="",
            stderr="model load failed",
        ),
    ) as worker:
        enriched_path, report_path = run_enrichment(
            audio_path=audio_path,
            transcript_path=transcript_path,
            output_dir=output_dir,
            resume=True,
        )

    command = worker.call_args.args[0]
    assert command[-1] == "--resume"
    enriched = read_json(enriched_path)
    report = read_json(report_path)
    assert enriched["atoms"][0]["text"] == "keep this"
    assert enriched["source_decode_strategy"].startswith("whisper_primary_plus")
    assert report["status"] == "degraded_whisper_primary_passthrough"
    assert report["fallback"] == "whisper_primary_passthrough"
    assert "exit status 9" in report["failure_reason"]


def test_passthrough_rejects_audio_identity_mismatch(tmp_path) -> None:
    audio_path = tmp_path / "source.wav"
    transcript_path = tmp_path / "source_transcript.json"
    audio_path.write_bytes(b"different audio")
    write_json(
        transcript_path,
        {
            "audio_sha256": "not-the-audio-hash",
            "atoms": [],
        },
    )

    with pytest.raises(CtcEnrichmentError, match="audio does not match"):
        write_passthrough_enrichment(
            audio_path=audio_path,
            transcript_path=transcript_path,
            output_dir=tmp_path / "output",
            reason="worker unavailable",
        )
    assert not (tmp_path / "output").exists()
