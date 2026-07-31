from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from voicecut.common import read_json, write_json
from voicecut.rough_render import (
    MergedRange,
    PlanWord,
    RoughRenderError,
    SelectedRange,
    merge_adjacent_ranges,
    range_to_sample_bounds,
    render_rough_cut,
    timestamp_to_sample,
)


def selected(
    start: int,
    end: int,
    *,
    thought: int,
    index: int = 0,
) -> SelectedRange:
    return SelectedRange(start, end, thought, index)


class RoughRangeTests(unittest.TestCase):
    def test_exactly_adjacent_ranges_merge_across_thoughts(self) -> None:
        ranges = [
            selected(99, 108, thought=0),
            selected(108, 113, thought=1),
            selected(113, 119, thought=2),
            selected(120, 125, thought=3),
        ]

        merged = merge_adjacent_ranges(ranges)

        self.assertEqual(
            [(item.start_word_id, item.end_word_id) for item in merged],
            [(99, 119), (120, 125)],
        )
        self.assertEqual(len(merged[0].original_ranges), 3)
        self.assertEqual(len(merged[1].original_ranges), 1)

    def test_gap_of_one_omitted_word_is_not_merged(self) -> None:
        merged = merge_adjacent_ranges(
            [
                selected(10, 14, thought=0),
                selected(15, 20, thought=1),
            ]
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].end_word_id, 14)
        self.assertEqual(merged[1].start_word_id, 15)

    def test_overlapping_ranges_are_rejected(self) -> None:
        with self.assertRaisesRegex(RoughRenderError, "overlapping"):
            merge_adjacent_ranges(
                [
                    selected(10, 15, thought=0),
                    selected(14, 20, thought=1),
                ]
            )

    def test_timestamp_conversion_respects_excluded_word_fences(self) -> None:
        words = [
            PlanWord(0, "previous", 0.1, 0.9),
            PlanWord(1, "selected", 1.0, 1.9),
            PlanWord(2, "words", 2.0, 3.0),
            PlanWord(3, "next", 3.1, 3.9),
        ]
        source_range = MergedRange(
            1,
            3,
            (selected(1, 3, thought=0),),
        )

        bounds = range_to_sample_bounds(
            words=words,
            source_range=source_range,
            sample_rate=100,
            total_samples=500,
            edge_padding_ms=300.0,
        )

        self.assertEqual(bounds, (90, 310, 100, 300))

    def test_end_word_id_is_exclusive_in_timestamp_conversion(self) -> None:
        words = [
            PlanWord(0, "zero", 0.2, 0.5),
            PlanWord(1, "one", 0.8, 1.1),
            PlanWord(2, "excluded", 1.4, 1.8),
        ]
        source_range = MergedRange(
            0,
            2,
            (selected(0, 2, thought=0),),
        )

        bounds = range_to_sample_bounds(
            words=words,
            source_range=source_range,
            sample_rate=1000,
            total_samples=2000,
            edge_padding_ms=0.0,
        )

        self.assertEqual(bounds, (200, 1100, 200, 1100))

    def test_shared_float_boundary_does_not_invent_one_sample_overlap(
        self,
    ) -> None:
        shared = 9.920000000000002
        self.assertEqual(
            timestamp_to_sample(
                shared,
                sample_rate=48000,
                total_samples=1_000_000,
                rounding="ceil",
            ),
            476160,
        )
        self.assertEqual(
            timestamp_to_sample(
                shared,
                sample_rate=48000,
                total_samples=1_000_000,
                rounding="floor",
            ),
            476160,
        )
        words = [
            PlanWord(0, "selected", 9.4, shared),
            PlanWord(1, "excluded", shared, 10.64),
        ]
        bounds = range_to_sample_bounds(
            words=words,
            source_range=MergedRange(
                0,
                1,
                (selected(0, 1, thought=0),),
            ),
            sample_rate=48000,
            total_samples=1_000_000,
            edge_padding_ms=30.0,
        )
        self.assertEqual(bounds[1], 476160)
        self.assertEqual(bounds[3], 476160)

    def test_real_neighbor_overlap_preserves_selected_word_without_padding(
        self,
    ) -> None:
        words = [
            PlanWord(0, "previous", 0.1, 1.05),
            PlanWord(1, "selected", 1.0, 2.0),
            PlanWord(2, "excluded", 1.95, 2.5),
        ]
        bounds = range_to_sample_bounds(
            words=words,
            source_range=MergedRange(
                1,
                2,
                (selected(1, 2, thought=0),),
            ),
            sample_rate=1000,
            total_samples=3000,
            edge_padding_ms=30.0,
        )
        self.assertEqual(bounds, (1000, 2000, 1000, 2000))


class RoughRenderIntegrationTests(unittest.TestCase):
    def test_output_duration_is_clips_plus_one_silence(self) -> None:
        sample_rate = 1000
        frames = 6000
        time = np.arange(frames, dtype=np.float32) / sample_rate
        source = np.stack(
            [
                np.sin(2.0 * np.pi * 5.0 * time),
                np.cos(2.0 * np.pi * 7.0 * time),
            ],
            axis=1,
        ).astype(np.float32)
        words = [
            {
                "id": index,
                "text": f"word{index}",
                "start": index + 0.1,
                "end": index + 0.9,
            }
            for index in range(6)
        ]
        plan = {
            "status": "complete",
            "words": words,
            "committed": [
                {
                    "canonical_text": "ignored",
                    "source_ranges": [{"start_word_id": 0, "end_word_id": 2}],
                },
                {
                    "canonical_text": "also ignored",
                    "source_ranges": [
                        {"start_word_id": 2, "end_word_id": 3},
                        {"start_word_id": 4, "end_word_id": 5},
                    ],
                },
            ],
            "selected_source_ranges": [
                {"start_word_id": 0, "end_word_id": 2},
                {"start_word_id": 2, "end_word_id": 3},
                {"start_word_id": 4, "end_word_id": 5},
            ],
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / "source.wav"
            plan_path = root / "streaming_plan.json"
            output_dir = root / "render"
            sf.write(audio_path, source, sample_rate, subtype="FLOAT")
            write_json(plan_path, plan)
            # A failed pre-render attempt may leave only this empty skeleton.
            (output_dir / "clips").mkdir(parents=True)

            manifest = render_rough_cut(
                audio_path=audio_path,
                plan_path=plan_path,
                output_dir=output_dir,
                edge_padding_ms=0.0,
                clip_fade_ms=5.0,
                inter_clip_silence_ms=80.0,
            )

            self.assertEqual(
                manifest["original_selected_range_count"],
                3,
            )
            self.assertEqual(manifest["merged_rendered_clip_count"], 2)
            self.assertEqual(
                manifest["clips"][0]["merged_original_ranges"],
                [
                    {"start_word_id": 0, "end_word_id": 2},
                    {"start_word_id": 2, "end_word_id": 3},
                ],
            )
            self.assertEqual(
                manifest["omitted_word_intervals"],
                [
                    {
                        "start_word_id": 3,
                        "end_word_id": 4,
                        "source_start_seconds": 3.1,
                        "source_end_seconds": 3.9,
                        "source_text": "word3",
                    },
                    {
                        "start_word_id": 5,
                        "end_word_id": 6,
                        "source_start_seconds": 5.1,
                        "source_end_seconds": 5.9,
                        "source_text": "word5",
                    },
                ],
            )
            # [0.1, 2.9) + 80 ms + [4.1, 4.9)
            self.assertEqual(manifest["expected_output_frame_count"], 3680)
            rough_info = sf.info(output_dir / "rough_cut.wav")
            self.assertEqual(rough_info.frames, 3680)
            self.assertEqual(rough_info.samplerate, sample_rate)
            self.assertEqual(rough_info.channels, 2)
            self.assertAlmostEqual(
                rough_info.duration,
                manifest["rough_cut_duration_seconds"],
                delta=1.0 / sample_rate,
            )
            saved = read_json(output_dir / "render_manifest.json")
            self.assertEqual(
                saved["rough_cut_wav_sha256"],
                manifest["rough_cut_wav_sha256"],
            )
            self.assertTrue((output_dir / "clips" / "clip_000.wav").is_file())
            self.assertTrue((output_dir / "clips" / "clip_001.wav").is_file())


if __name__ == "__main__":
    unittest.main()
