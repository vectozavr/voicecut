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
from voicecut.render import ctc_boundaries, fuse_asr_ctc_interval


class ExplicitPhraseMappingTests(unittest.TestCase):
    def test_split_alignment_keeps_explicit_phrase_identity(self) -> None:
        requested = {
            "segments": [
                {
                    "phrase_index": 7,
                    "start": 1.0,
                    "end": 2.0,
                    "text": "Hello, world.",
                },
                {
                    "phrase_index": 42,
                    "start": 3.0,
                    "end": 4.0,
                    "text": "Next phrase.",
                },
            ]
        }
        results = [
            {
                "segments": [
                    {
                        "text": "Hello,",
                        "words": [
                            {
                                "word": "Hello",
                                "start": 1.02,
                                "end": 1.24,
                                "score": 0.91,
                            }
                        ],
                        "chars": [{"char": "H", "start": 1.02, "end": 1.04}],
                    },
                    {
                        "text": "world.",
                        "words": [
                            {
                                "word": "world",
                                "start": 1.31,
                                "end": 1.67,
                                "score": 0.87,
                            }
                        ],
                        "chars": [{"char": "w", "start": 1.31, "end": 1.33}],
                    },
                ]
            },
            {
                "segments": [
                    {
                        "text": "Next phrase.",
                        "words": [
                            {
                                "word": "Next",
                                "start": 3.04,
                                "end": 3.30,
                                "score": 0.93,
                            },
                            {
                                "word": "phrase",
                                "start": 3.35,
                                "end": 3.72,
                                "score": 0.89,
                            },
                        ],
                        "chars": [],
                    }
                ]
            },
        ]

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "ctc_input.json"
            output_path = directory / "ctc_output.json"
            input_path.write_text(json.dumps(requested), encoding="utf-8")

            with (
                mock.patch.object(
                    align_ctc.librosa,
                    "load",
                    return_value=(np.zeros(16_000, dtype=np.float32), 16_000),
                ),
                mock.patch.object(
                    align_ctc.whisperx,
                    "load_align_model",
                    return_value=(object(), {}),
                ),
                mock.patch.object(
                    align_ctc.whisperx,
                    "align",
                    side_effect=results,
                ) as align_mock,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "align_ctc.py",
                        "--audio",
                        str(directory / "unused.wav"),
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

            aligned = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(align_mock.call_count, 2)
        self.assertEqual(
            [segment["phrase_index"] for segment in aligned["segments"]],
            [7, 42],
        )
        self.assertEqual(aligned["segments"][0]["subsegment_count"], 2)
        self.assertEqual(aligned["segments"][0]["start"], 1.02)
        self.assertEqual(aligned["segments"][0]["end"], 1.67)
        self.assertEqual(
            [word["phrase_index"] for word in aligned["word_segments"]],
            [7, 7, 42, 42],
        )

        mapped = ctc_boundaries(
            aligned,
            {"ctc_segment_phrase_indices": [999, 1000]},
        )
        self.assertEqual(set(mapped), {7, 42})
        self.assertEqual(mapped[7]["end"], 1.67)
        self.assertEqual(mapped[42]["start"], 3.04)


class ConservativeEdgeFusionTests(unittest.TestCase):
    def test_under_aligned_low_confidence_ctc_never_shortens_asr_end(self) -> None:
        speech_start, speech_end = fuse_asr_ctc_interval(
            5.0,
            10.0,
            {
                "start": 5.18,
                "end": 9.35,
                "mean_score": 0.20,
                "first_score": 0.18,
                "last_score": 0.16,
            },
        )
        self.assertEqual(speech_start, 5.0)
        self.assertEqual(speech_end, 10.0)

    def test_low_confidence_ctc_cannot_extend_either_edge(self) -> None:
        self.assertEqual(
            fuse_asr_ctc_interval(
                5.0,
                10.0,
                {
                    "start": 4.85,
                    "end": 10.20,
                    "first_score": 0.30,
                    "last_score": 0.30,
                },
            ),
            (5.0, 10.0),
        )

    def test_nearby_confident_ctc_may_only_extend_asr_interval(self) -> None:
        speech_start, speech_end = fuse_asr_ctc_interval(
            5.0,
            10.0,
            {
                "start": 4.82,
                "end": 10.24,
                "first_score": 0.90,
                "last_score": 0.88,
            },
        )
        self.assertEqual((speech_start, speech_end), (4.82, 10.24))
        self.assertLessEqual(speech_start, 5.0)
        self.assertGreaterEqual(speech_end, 10.0)

    def test_distant_ctc_cannot_pull_a_retry_into_the_clip(self) -> None:
        self.assertEqual(
            fuse_asr_ctc_interval(
                5.0,
                10.0,
                {
                    "start": 3.0,
                    "end": 11.0,
                    "first_score": 0.99,
                    "last_score": 0.99,
                },
            ),
            (5.0, 10.0),
        )


if __name__ == "__main__":
    unittest.main()
