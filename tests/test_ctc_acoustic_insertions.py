from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from voicecut import align_ctc


def greedy_word(
    word: str,
    start: float,
    end: float,
    score: float = 0.92,
) -> dict[str, float | str]:
    return {
        "word": word,
        "start": start,
        "end": end,
        "score": score,
    }


def familiar_retry_words() -> list[dict[str, float | str]]:
    return [
        greedy_word("begin", 10.00, 10.30),
        greedy_word("with", 10.40, 10.62),
        greedy_word("famile", 10.68, 11.02),
        greedy_word("with", 11.15, 11.36),
        greedy_word("the", 11.40, 11.55),
        greedy_word("famile", 11.60, 11.95),
        greedy_word("words", 12.02, 12.36),
    ]


class GreedyCtcDecodeTests(unittest.TestCase):
    def test_collapses_tokens_into_absolute_timed_words(self) -> None:
        dictionary = {
            "-": 0,
            "|": 1,
            "b": 2,
            "e": 3,
            "g": 4,
            "i": 5,
            "n": 6,
            "w": 7,
            "t": 8,
            "h": 9,
        }
        token_path = [0, 2, 2, 0, 3, 4, 5, 6, 1, 1, 7, 5, 8, 9, 0]
        emissions = np.full((len(token_path), len(dictionary)), 0.01)
        for frame, token_id in enumerate(token_path):
            emissions[frame, token_id] = 0.91

        words = align_ctc.decode_collapsed_ctc_words(
            emissions,
            dictionary,
            absolute_start=10.0,
            absolute_end=13.0,
        )

        self.assertEqual([word["word"] for word in words], ["begin", "with"])
        self.assertEqual((words[0]["start"], words[0]["end"]), (10.2, 11.6))
        self.assertEqual((words[1]["start"], words[1]["end"]), (12.0, 12.8))
        self.assertAlmostEqual(float(words[0]["score"]), 0.91)


class AcousticRetryDiscoveryTests(unittest.TestCase):
    def test_latest_complete_occurrence_leaves_earlier_retry_unmatched(self) -> None:
        alignment = align_ctc.align_expected_to_greedy(
            "begin with the familiar words",
            familiar_retry_words(),
        )

        self.assertEqual(
            [match["greedy_index"] for match in alignment],
            [0, 3, 4, 5, 6],
        )
        self.assertGreater(float(alignment[3]["lexical_score"]), 0.68)

    def test_fenced_earlier_with_famile_is_reported_as_retry(self) -> None:
        insertions = align_ctc.discover_acoustic_insertions(
            "begin with the familiar words",
            familiar_retry_words(),
        )

        self.assertEqual(len(insertions), 1)
        retry = insertions[0]
        self.assertEqual(retry["type"], "spoken_retry")
        self.assertEqual(retry["text"], "with famile")
        self.assertEqual((retry["start"], retry["end"]), (10.4, 11.02))
        self.assertEqual(
            (retry["safe_edit_start"], retry["safe_edit_end"]),
            (10.3, 11.15),
        )
        self.assertEqual(retry["retry_expected_indices"], [1, 3])

    def test_clean_take_and_single_spelling_variant_are_not_insertions(self) -> None:
        clean = [
            greedy_word("begin", 0.0, 0.2),
            greedy_word("with", 0.3, 0.5),
            greedy_word("the", 0.6, 0.7),
            greedy_word("famile", 0.8, 1.1),
            greedy_word("words", 1.2, 1.5),
        ]

        self.assertEqual(
            align_ctc.discover_acoustic_insertions(
                "begin with the familiar words",
                clean,
            ),
            [],
        )

    def test_paraphrase_is_not_misclassified_as_a_retry(self) -> None:
        paraphrase = [
            greedy_word("begin", 0.0, 0.2),
            greedy_word("using", 0.3, 0.6),
            greedy_word("familiar", 0.7, 1.1),
            greedy_word("words", 1.2, 1.5),
        ]

        self.assertEqual(
            align_ctc.discover_acoustic_insertions(
                "begin with the familiar words",
                paraphrase,
            ),
            [],
        )


class AcousticExpectedSubstitutionTests(unittest.TestCase):
    def test_real_shaped_numeral_is_grounded_as_spoken_once(self) -> None:
        greedy = [
            greedy_word("in", 140.071115, 140.111254, 0.997473),
            greedy_word("this", 140.211603, 140.352091, 0.998547),
            greedy_word("two", 140.412300, 140.532718, 0.995829),
            greedy_word("example", 140.653136, 141.094669, 0.999795),
            greedy_word("once", 141.415784, 141.576341, 0.941859),
            greedy_word("receives", 141.716829, 142.037944, 0.823424),
            greedy_word("the", 142.118223, 142.218571, 0.932469),
            greedy_word("highest", 142.318920, 142.599895, 0.958208),
            greedy_word("probability", 142.680174, 143.282265, 0.999756),
        ]

        substitutions = align_ctc.discover_acoustic_expected_substitutions(
            "In these two examples, 1 receives the highest probability.",
            greedy,
        )

        numeral = next(item for item in substitutions if item["expected_word"] == "1")
        self.assertEqual(numeral["expected_index"], 4)
        self.assertEqual(numeral["greedy_index"], 4)
        self.assertEqual(numeral["greedy_word"], "once")
        self.assertEqual(
            (numeral["start"], numeral["end"]),
            (141.415784, 141.576341),
        )
        self.assertEqual(numeral["acoustic_score"], 0.941859)
        self.assertEqual(
            numeral["reason"],
            "greedy_ctc_source_grounded_expected_substitution",
        )
        self.assertEqual(
            (
                numeral["left_anchor"]["expected_word"],
                numeral["left_anchor"]["greedy_word"],
            ),
            ("examples", "example"),
        )
        self.assertEqual(
            (
                numeral["right_anchor"]["expected_word"],
                numeral["right_anchor"]["greedy_word"],
            ),
            ("receives", "receives"),
        )
        self.assertGreaterEqual(
            numeral["left_anchor"]["lexical_score"],
            0.90,
        )
        self.assertGreaterEqual(
            numeral["right_anchor"]["acoustic_score"],
            0.75,
        )

    def test_retry_and_filler_insertions_are_not_substitutions(self) -> None:
        filler = [
            greedy_word("begin", 0.0, 0.2),
            greedy_word("um", 0.3, 0.4),
            greedy_word("with", 0.5, 0.7),
            greedy_word("words", 0.8, 1.1),
        ]

        self.assertEqual(
            align_ctc.discover_acoustic_expected_substitutions(
                "begin with the familiar words",
                familiar_retry_words(),
            ),
            [],
        )
        self.assertEqual(
            align_ctc.discover_acoustic_expected_substitutions(
                "begin with words",
                filler,
            ),
            [],
        )

    def test_unequal_or_low_confidence_spans_are_not_substitutions(self) -> None:
        unequal = [
            greedy_word("alpha", 0.0, 0.2),
            greedy_word("spoken", 0.3, 0.5),
            greedy_word("number", 0.6, 0.8),
            greedy_word("omega", 0.9, 1.1),
        ]
        weak = [
            greedy_word("alpha", 0.0, 0.2),
            greedy_word("once", 0.3, 0.5, 0.74),
            greedy_word("omega", 0.6, 0.8),
        ]

        self.assertEqual(
            align_ctc.discover_acoustic_expected_substitutions(
                "alpha 1 omega",
                unequal,
            ),
            [],
        )
        self.assertEqual(
            align_ctc.discover_acoustic_expected_substitutions(
                "alpha 1 omega",
                weak,
            ),
            [],
        )

    def test_repeated_anchor_and_reordered_words_remain_visible(self) -> None:
        repeated_anchor = [
            greedy_word("say", 0.0, 0.2),
            greedy_word("say", 0.3, 0.5),
            greedy_word("now", 0.6, 0.8),
        ]
        reordered = [
            greedy_word("start", 0.0, 0.2),
            greedy_word("blue", 0.3, 0.5),
            greedy_word("red", 0.6, 0.8),
            greedy_word("end", 0.9, 1.1),
        ]

        self.assertEqual(
            align_ctc.discover_acoustic_expected_substitutions(
                "say color now",
                repeated_anchor,
            ),
            [],
        )
        self.assertEqual(
            align_ctc.discover_acoustic_expected_substitutions(
                "start red blue end",
                reordered,
            ),
            [],
        )


class AcousticRetryOutputIntegrationTests(unittest.TestCase):
    def test_main_emits_raw_evidence_only_for_multiple_phrase_ids(self) -> None:
        requested = {
            "segments": [
                {
                    "phrase_index": 41,
                    "start": 9.8,
                    "end": 12.5,
                    "text": "begin with the familiar words",
                },
                {
                    "phrase_index": 7,
                    "start": 13.0,
                    "end": 14.5,
                    "text": "next sentence",
                },
            ]
        }
        next_words = [
            greedy_word("next", 13.1, 13.5),
            greedy_word("sentence", 13.6, 14.2),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "ctc_input.json"
            output_path = directory / "ctc_output.json"
            audio_path = directory / "source.wav"
            input_path.write_text(json.dumps(requested), encoding="utf-8")
            audio_path.write_bytes(b"test audio identity")
            with (
                mock.patch.object(
                    align_ctc.librosa,
                    "load",
                    return_value=(np.zeros(16_000 * 15, dtype=np.float32), 16_000),
                ),
                mock.patch.object(
                    align_ctc.whisperx,
                    "load_align_model",
                    return_value=(object(), {}),
                ) as load_model,
                mock.patch.object(align_ctc.whisperx, "align") as known_text_align,
                mock.patch.object(
                    align_ctc,
                    "decode_greedy_ctc_segment",
                    side_effect=[familiar_retry_words(), next_words],
                ) as raw_decode,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "align_ctc.py",
                        "--audio",
                        str(audio_path),
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ],
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                align_ctc.main()

            output = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(output["schema_version"], 3)
        self.assertEqual(output["mode"], "raw_greedy_ctc_retry_evidence")
        self.assertNotIn("word_segments", output)
        self.assertEqual(
            [segment["phrase_index"] for segment in output["segments"]],
            [41, 7],
        )
        segment = output["segments"][0]
        self.assertEqual(
            set(segment),
            {
                "phrase_index",
                "input_start",
                "input_end",
                "input_text",
                "greedy_ctc_words",
                "acoustic_insertions",
                "acoustic_expected_substitutions",
            },
        )
        self.assertEqual(
            [word["word"] for word in segment["greedy_ctc_words"]],
            ["begin", "with", "famile", "with", "the", "famile", "words"],
        )
        self.assertEqual(segment["acoustic_insertions"][0]["text"], "with famile")
        self.assertEqual(segment["acoustic_expected_substitutions"], [])
        self.assertEqual(
            [word["word"] for word in output["segments"][1]["greedy_ctc_words"]],
            ["next", "sentence"],
        )
        load_model.assert_called_once_with(language_code="en", device="cpu")
        self.assertEqual(raw_decode.call_count, 2)
        known_text_align.assert_not_called()

    def test_resume_uses_phrase_ids_and_current_checkpoint_schema(
        self,
    ) -> None:
        requested = {
            "segments": [
                {
                    "phrase_index": 73,
                    "start": 1.0,
                    "end": 2.0,
                    "text": "first phrase",
                },
                {
                    "phrase_index": 5,
                    "start": 3.0,
                    "end": 4.0,
                    "text": "second phrase",
                },
            ]
        }
        first_words = [
            greedy_word("first", 1.1, 1.4),
            greedy_word("phrase", 1.5, 1.9),
        ]
        second_words = [
            greedy_word("second", 3.1, 3.5),
            greedy_word("phrase", 3.6, 3.9),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "ctc_input.json"
            output_path = directory / "ctc_output.json"
            partial_path = directory / "ctc_output.json.partial"
            audio_path = directory / "source.wav"
            input_path.write_text(json.dumps(requested), encoding="utf-8")
            audio_path.write_bytes(b"stable audio")
            identity = align_ctc._checkpoint_identity(
                audio_path=audio_path,
                input_path=input_path,
                language="en",
                device="cpu",
            )
            partial_path.write_text(
                json.dumps(
                    {
                        "schema_version": align_ctc.OUTPUT_SCHEMA_VERSION,
                        "mode": align_ctc.OUTPUT_MODE,
                        "complete": False,
                        **identity,
                        "segments": [
                            {
                                "phrase_index": 73,
                                "input_start": 1.0,
                                "input_end": 2.0,
                                "input_text": "first phrase",
                                "greedy_ctc_words": first_words,
                                "acoustic_insertions": [],
                                "acoustic_expected_substitutions": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    align_ctc.librosa,
                    "load",
                    return_value=(np.zeros(16_000 * 5, dtype=np.float32), 16_000),
                ),
                mock.patch.object(
                    align_ctc.whisperx,
                    "load_align_model",
                    return_value=(object(), {}),
                ),
                mock.patch.object(align_ctc.whisperx, "align") as known_text_align,
                mock.patch.object(
                    align_ctc,
                    "decode_greedy_ctc_segment",
                    return_value=second_words,
                ) as raw_decode,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "align_ctc.py",
                        "--audio",
                        str(audio_path),
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--resume",
                    ],
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                align_ctc.main()

            output = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            [segment["phrase_index"] for segment in output["segments"]],
            [73, 5],
        )
        self.assertEqual(raw_decode.call_count, 1)
        self.assertEqual(raw_decode.call_args.kwargs["start"], 3.0)
        known_text_align.assert_not_called()

    def test_resume_rejects_changed_input_content(self) -> None:
        requested = {
            "segments": [
                {
                    "phrase_index": 2,
                    "start": 1.0,
                    "end": 2.0,
                    "text": "one phrase",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "ctc_input.json"
            output_path = directory / "ctc_output.json"
            partial_path = directory / "ctc_output.json.partial"
            audio_path = directory / "source.wav"
            input_path.write_text(json.dumps(requested), encoding="utf-8")
            audio_path.write_bytes(b"stable audio")
            identity = align_ctc._checkpoint_identity(
                audio_path=audio_path,
                input_path=input_path,
                language="en",
                device="cpu",
            )
            identity["input_sha256"] = "not-the-current-digest"
            partial_path.write_text(
                json.dumps(
                    {
                        "schema_version": align_ctc.OUTPUT_SCHEMA_VERSION,
                        "mode": align_ctc.OUTPUT_MODE,
                        "complete": False,
                        **identity,
                        "segments": [],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                sys,
                "argv",
                [
                    "align_ctc.py",
                    "--audio",
                    str(audio_path),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--resume",
                ],
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "checkpoint does not match",
                ):
                    align_ctc.main()


if __name__ == "__main__":
    unittest.main()
