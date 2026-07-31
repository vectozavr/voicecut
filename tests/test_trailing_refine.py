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
    SelectedRange,
)
from voicecut.trailing_refine import (
    refine_trailing_boundary,
    render_trailing_refined_preview,
)


SAMPLE_RATE = 1000


def selected_range(start: int, end: int) -> MergedRange:
    selected = SelectedRange(
        start_word_id=start,
        end_word_id=end,
        thought_index=0,
        thought_range_index=0,
    )
    return MergedRange(start, end, (selected,))


def two_words(*, raw_end: float, next_start: float) -> list[PlanWord]:
    return [
        PlanWord(0, "selected", 0.1, raw_end),
        PlanWord(1, "omitted", next_start, next_start + 0.2),
    ]


class TrailingBoundarySyntheticTests(unittest.TestCase):
    def test_waveform_continuing_200ms_past_raw_timestamp_is_preserved(
        self,
    ) -> None:
        waveform = np.zeros(1500, dtype=np.float32)
        waveform[100:700] = 0.25

        result = refine_trailing_boundary(
            waveform,
            words=two_words(raw_end=0.5, next_start=1.2),
            source_range=selected_range(0, 1),
            sample_rate=SAMPLE_RATE,
        )

        self.assertEqual(result.boundary_method, "stable_silence")
        self.assertGreaterEqual(result.refined_end_sample, 715)
        self.assertLessEqual(result.refined_end_sample, 725)
        self.assertLessEqual(result.refined_end_sample, 1190)

    def test_normal_silent_gap_uses_first_stable_silence(self) -> None:
        waveform = np.zeros(1200, dtype=np.float32)
        waveform[100:500] = 0.25

        result = refine_trailing_boundary(
            waveform,
            words=two_words(raw_end=0.5, next_start=0.9),
            source_range=selected_range(0, 1),
            sample_rate=SAMPLE_RATE,
        )

        self.assertEqual(result.boundary_method, "stable_silence")
        self.assertGreaterEqual(result.refined_end_sample, 515)
        self.assertLessEqual(result.refined_end_sample, 525)
        self.assertLessEqual(result.refined_end_sample, 890)

    def test_quiet_run_without_active_lead_is_not_selected(self) -> None:
        waveform = np.zeros(1500, dtype=np.float32)
        waveform[100:400] = 0.25
        # Whisper's raw endpoint lies in this long quiet region, but speech
        # resumes later.  The initial quiet must not be called a word ending
        # because no active waveform leads into it inside the search window.
        waveform[650:850] = 0.25

        result = refine_trailing_boundary(
            waveform,
            words=two_words(raw_end=0.5, next_start=1.3),
            source_range=selected_range(0, 1),
            sample_rate=SAMPLE_RATE,
        )

        self.assertEqual(result.boundary_method, "stable_silence")
        self.assertIsNotNone(result.stable_silence_start_sample)
        self.assertGreaterEqual(result.stable_silence_start_sample, 845)
        self.assertGreaterEqual(result.refined_end_sample, 865)
        self.assertLessEqual(result.refined_end_sample, 875)

    def test_immediately_following_omitted_word_is_a_hard_boundary(
        self,
    ) -> None:
        waveform = np.full(1000, 0.25, dtype=np.float32)

        result = refine_trailing_boundary(
            waveform,
            words=two_words(raw_end=0.5, next_start=0.505),
            source_range=selected_range(0, 1),
            sample_rate=SAMPLE_RATE,
        )

        self.assertEqual(result.boundary_method, "hard_boundary")
        self.assertEqual(result.refined_end_sample, 500)
        self.assertLessEqual(result.refined_end_sample, 505)

    def test_final_word_at_audio_eof_is_marked_end_of_file(self) -> None:
        waveform = np.full(1000, 0.25, dtype=np.float32)
        words = [PlanWord(0, "final", 0.1, 1.0)]

        result = refine_trailing_boundary(
            waveform,
            words=words,
            source_range=selected_range(0, 1),
            sample_rate=SAMPLE_RATE,
        )

        self.assertEqual(result.boundary_method, "end_of_file")
        self.assertEqual(result.raw_end_sample, 1000)
        self.assertEqual(result.refined_end_sample, 1000)

    def test_final_word_without_stable_silence_preserves_audio_to_eof(
        self,
    ) -> None:
        waveform = np.full(1000, 0.25, dtype=np.float32)
        words = [PlanWord(0, "final", 0.1, 0.5)]

        result = refine_trailing_boundary(
            waveform,
            words=words,
            source_range=selected_range(0, 1),
            sample_rate=SAMPLE_RATE,
        )

        self.assertEqual(result.boundary_method, "end_of_file")
        self.assertEqual(result.raw_end_sample, 500)
        self.assertEqual(result.refined_end_sample, len(waveform))

    def test_no_stable_silence_keeps_raw_endpoint(self) -> None:
        waveform = np.full(1500, 0.25, dtype=np.float32)

        result = refine_trailing_boundary(
            waveform,
            words=two_words(raw_end=0.5, next_start=1.2),
            source_range=selected_range(0, 1),
            sample_rate=SAMPLE_RATE,
        )

        self.assertEqual(result.boundary_method, "hard_boundary")
        self.assertEqual(result.refined_end_sample, 500)
        self.assertLessEqual(result.refined_end_sample, 1190)

    def test_overlapping_next_word_timestamps_become_hard_boundary(self) -> None:
        waveform = np.zeros(1000, dtype=np.float32)

        result = refine_trailing_boundary(
            waveform,
            words=two_words(raw_end=0.5, next_start=0.49),
            source_range=selected_range(0, 1),
            sample_rate=SAMPLE_RATE,
        )

        self.assertEqual(result.boundary_method, "hard_boundary")
        self.assertEqual(result.raw_end_sample, 500)
        self.assertEqual(result.refined_end_sample, 500)
        self.assertEqual(result.next_omitted_word_start_sample, 490)


class TrailingRenderIntegrationTests(unittest.TestCase):
    def test_preview_keeps_start_and_fixed_pause_and_saves_diagnostics(
        self,
    ) -> None:
        frames = 3000
        source = np.zeros((frames, 2), dtype=np.float32)
        source[100:700, :] = 0.25
        source[1200:1700, :] = -0.2
        source[2100:2500, :] = 0.15
        plan = {
            "status": "complete",
            "words": [
                {"id": 0, "text": "first", "start": 0.1, "end": 0.5},
                {"id": 1, "text": "omitted", "start": 1.2, "end": 1.7},
                {"id": 2, "text": "last", "start": 2.1, "end": 2.5},
            ],
            "committed": [
                {
                    "canonical_text": "first",
                    "source_ranges": [{"start_word_id": 0, "end_word_id": 1}],
                },
                {
                    "canonical_text": "last",
                    "source_ranges": [{"start_word_id": 2, "end_word_id": 3}],
                },
            ],
            "selected_source_ranges": [
                {"start_word_id": 0, "end_word_id": 1},
                {"start_word_id": 2, "end_word_id": 3},
            ],
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / "source.wav"
            plan_path = root / "streaming_plan.json"
            output_dir = root / "preview"
            sf.write(audio_path, source, SAMPLE_RATE, subtype="FLOAT")
            write_json(plan_path, plan)

            manifest = render_trailing_refined_preview(
                audio_path=audio_path,
                plan_path=plan_path,
                output_dir=output_dir,
                edge_padding_ms=30.0,
                clip_fade_ms=5.0,
                inter_clip_silence_ms=80.0,
            )

            self.assertTrue((output_dir / "rough_cut.wav").is_file())
            self.assertTrue((output_dir / "rough_cut_refined.wav").is_file())
            self.assertTrue((output_dir / "render_manifest_refined.json").is_file())
            self.assertEqual(
                manifest["clips"][0]["refined_source_start_sample"],
                manifest["clips"][0]["source_start_sample"],
            )
            first_end = manifest["clips"][0]["refined_source_end_sample"]
            second_start = manifest["clips"][1]["refined_source_start_sample"]
            refined_frames = sf.info(output_dir / "rough_cut_refined.wav").frames
            self.assertEqual(
                refined_frames,
                first_end
                - manifest["clips"][0]["source_start_sample"]
                + 80
                + manifest["clips"][1]["refined_source_end_sample"]
                - second_start,
            )
            for index in range(2):
                self.assertTrue(
                    (
                        output_dir / "boundary_debug" / f"clip_{index:03d}_end.png"
                    ).is_file()
                )
            saved = read_json(output_dir / "render_manifest_refined.json")
            self.assertEqual(
                saved["rough_cut_refined_wav_sha256"],
                manifest["rough_cut_refined_wav_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
