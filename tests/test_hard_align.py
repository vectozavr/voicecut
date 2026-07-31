from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from voicecut.common import read_json, write_json
from voicecut.hard_align import (
    _prepare_alignment_jobs,
    alignment_positions,
    decide_forced_boundary,
    render_forced_aligned_preview,
)
from voicecut.rough_render import PlanWord
from voicecut.trailing_refine import render_trailing_refined_preview


SAMPLE_RATE = 1000


def mock_aligned_job(
    words: list[str],
    times: list[tuple[float, float]],
    *,
    clip_index: int = 0,
) -> dict:
    word_segments = [
        {
            "word": word,
            "start": start,
            "end": end,
            "score": 0.95,
        }
        for word, (start, end) in zip(words, times, strict=True)
    ]
    chars = []
    for index, (word, (start, end)) in enumerate(zip(words, times, strict=True)):
        chars.append(
            {
                "char": word[0],
                "start": start,
                "end": end,
                "score": 0.95,
            }
        )
        if index < len(words) - 1:
            chars.append({"char": " "})
    return {
        "clip_index": clip_index,
        "error": None,
        "aligned": {
            "word_segments": word_segments,
            "segments": [
                {
                    "text": " ".join(words),
                    "words": word_segments,
                    "chars": chars,
                }
            ],
        },
    }


def decision_job() -> dict:
    return {
        "clip_index": 0,
        "local_words": [
            {"id": 0, "text": "same"},
            {"id": 1, "text": "same"},
            {"id": 2, "text": "kept"},
            {"id": 3, "text": "same"},
            {"id": 4, "text": "same"},
        ],
        "kept_local_index": 2,
        "omitted_local_index": 3,
        "crop_start_seconds": 0.0,
        "crop_duration_seconds": 1.5,
        "raw_end_seconds": 0.5,
        "silence_refined_end_seconds": 0.5,
    }


class ForcedBoundaryDecisionTests(unittest.TestCase):
    def test_crop_transcript_expands_for_words_intersecting_300ms_handles(
        self,
    ) -> None:
        words = [
            PlanWord(0, "zero", 0.0, 0.3),
            PlanWord(1, "intersecting-left", 0.6, 0.9),
            PlanWord(2, "before-one", 1.0, 1.3),
            PlanWord(3, "before-two", 1.3, 1.6),
            PlanWord(4, "kept", 1.6, 1.9),
            PlanWord(5, "omitted", 1.9, 2.2),
            PlanWord(6, "after-one", 2.2, 2.5),
            PlanWord(7, "after-two", 2.7, 3.0),
            PlanWord(8, "intersecting-right", 3.2, 3.5),
            PlanWord(9, "nine", 4.0, 4.3),
        ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            debug_root = root / "debug"
            debug_root.mkdir()
            source = np.zeros((5000, 1), dtype=np.float32)
            previous_audio = source[:500]

            jobs = _prepare_alignment_jobs(
                hard_clips=[
                    {
                        "clip_index": 0,
                        "source_word_end": 5,
                        "refined_output_start_sample": 0,
                        "refined_output_end_sample": 500,
                        "refined_frame_count": 500,
                        "raw_end_seconds": 1.9,
                        "refined_end_seconds": 1.9,
                    }
                ],
                words=words,
                source_audio=source,
                previous_audio=previous_audio,
                sample_rate=SAMPLE_RATE,
                debug_root=debug_root,
                crop_context_ms=300.0,
                write_debug_artifacts=True,
            )

        job = jobs[0]
        self.assertEqual(job["requested_context_start_word_id"], 2)
        self.assertEqual(job["requested_context_end_word_id"], 8)
        self.assertEqual(job["context_start_word_id"], 1)
        self.assertEqual(job["context_end_word_id"], 9)
        self.assertEqual(
            [word["id"] for word in job["local_words"]],
            list(range(1, 9)),
        )
        self.assertEqual(job["crop_start_sample"], 600)
        self.assertEqual(job["crop_end_sample"], 3500)

    def test_duplicate_words_are_mapped_by_local_sequence_position(self) -> None:
        worker = mock_aligned_job(
            ["same", "same", "kept", "same", "same"],
            [
                (0.1, 0.2),
                (0.25, 0.35),
                (0.4, 0.62),
                (0.65, 0.8),
                (0.85, 1.0),
            ],
        )

        kept_end, omitted_start, granularity = alignment_positions(
            job=decision_job(),
            worker_job=worker,
        )

        self.assertEqual(granularity, "characters")
        self.assertEqual(kept_end, 0.62)
        self.assertEqual(omitted_start, 0.65)

    def test_cut_snaps_only_inside_aligned_interword_interval(self) -> None:
        waveform = np.full(1500, 0.2, dtype=np.float32)
        waveform[633:638] = 0.0
        worker = mock_aligned_job(
            ["same", "same", "kept", "same", "same"],
            [
                (0.1, 0.2),
                (0.25, 0.35),
                (0.4, 0.62),
                (0.65, 0.8),
                (0.85, 1.0),
            ],
        )

        decision = decide_forced_boundary(
            job=decision_job(),
            worker_job=worker,
            mono=waveform,
            sample_rate=SAMPLE_RATE,
        )

        self.assertEqual(decision.status, "forced_alignment")
        self.assertGreaterEqual(decision.cut_seconds, 0.62)
        self.assertLessEqual(decision.cut_seconds, 0.65)
        self.assertEqual(
            decision.snap_method,
            "low_amplitude_between_words",
        )

    def test_shift_over_350ms_fails_and_preserves_previous_boundary(
        self,
    ) -> None:
        worker = mock_aligned_job(
            ["same", "same", "kept", "same", "same"],
            [
                (0.1, 0.2),
                (0.25, 0.35),
                (0.4, 0.9),
                (0.95, 1.05),
                (1.1, 1.2),
            ],
        )

        decision = decide_forced_boundary(
            job=decision_job(),
            worker_job=worker,
            mono=np.full(1500, 0.2, dtype=np.float32),
            sample_rate=SAMPLE_RATE,
        )

        self.assertEqual(decision.status, "forced_alignment_failed")
        self.assertEqual(decision.cut_seconds, 0.5)
        self.assertIn("exceeds", decision.error or "")

    def test_snapping_is_limited_to_the_permitted_shift_interval(self) -> None:
        waveform = np.full(1500, 0.2, dtype=np.float32)
        waveform[990:1000] = 0.0
        worker = mock_aligned_job(
            ["same", "same", "kept", "same", "same"],
            [
                (0.1, 0.2),
                (0.25, 0.35),
                (0.4, 0.62),
                (1.05, 1.15),
                (1.2, 1.3),
            ],
        )

        decision = decide_forced_boundary(
            job=decision_job(),
            worker_job=worker,
            mono=waveform,
            sample_rate=SAMPLE_RATE,
        )

        self.assertEqual(decision.status, "forced_alignment")
        self.assertGreaterEqual(decision.cut_seconds, 0.62)
        self.assertLessEqual(decision.cut_seconds, 0.85)
        self.assertLessEqual(abs(decision.shift_ms), 350.0)

    def test_unaligned_boundary_word_fails_without_guessing(self) -> None:
        worker = mock_aligned_job(
            ["same", "same", "kept", "same", "same"],
            [
                (0.1, 0.2),
                (0.25, 0.35),
                (0.4, 0.62),
                (0.65, 0.8),
                (0.85, 1.0),
            ],
        )
        worker["aligned"]["segments"][0]["chars"] = []
        del worker["aligned"]["word_segments"][2]["score"]

        decision = decide_forced_boundary(
            job=decision_job(),
            worker_job=worker,
            mono=np.full(1500, 0.2, dtype=np.float32),
            sample_rate=SAMPLE_RATE,
        )

        self.assertEqual(decision.status, "forced_alignment_failed")
        self.assertEqual(decision.cut_seconds, 0.5)
        self.assertIn("not aligned", decision.error or "")


class ForcedBoundaryRenderTests(unittest.TestCase):
    def _source_and_plan(self, root: Path) -> tuple[Path, Path, Path]:
        source = np.zeros((2000, 2), dtype=np.float32)
        source[100:620, :] = 0.2
        source[650:950, :] = -0.18
        source[1200:1500, :] = 0.16
        audio_path = root / "source.wav"
        plan_path = root / "streaming_plan.json"
        trailing_dir = root / "trailing"
        sf.write(audio_path, source, SAMPLE_RATE, subtype="FLOAT")
        write_json(
            plan_path,
            {
                "status": "complete",
                "words": [
                    {
                        "id": 0,
                        "text": "selected",
                        "start": 0.1,
                        "end": 0.5,
                    },
                    {
                        "id": 1,
                        "text": "omitted",
                        "start": 0.5,
                        "end": 0.95,
                    },
                    {
                        "id": 2,
                        "text": "last",
                        "start": 1.2,
                        "end": 1.5,
                    },
                ],
                "committed": [
                    {
                        "canonical_text": "selected",
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
            },
        )
        render_trailing_refined_preview(
            audio_path=audio_path,
            plan_path=plan_path,
            output_dir=trailing_dir,
            edge_padding_ms=30.0,
            clip_fade_ms=5.0,
            inter_clip_silence_ms=80.0,
        )
        return (
            audio_path,
            plan_path,
            trailing_dir / "render_manifest_refined.json",
        )

    def test_success_changes_only_hard_clip_end_and_uses_2ms_fade(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, refined_manifest = self._source_and_plan(root)
            worker = mock_aligned_job(
                ["selected", "omitted", "last"],
                [(0.1, 0.62), (0.65, 0.95), (1.2, 1.5)],
            )

            manifest = render_forced_aligned_preview(
                refined_manifest_path=refined_manifest,
                output_dir=root / "forced",
                alignment_python=root / "unused-python",
                alignment_payload={"jobs": [worker]},
            )

            first, stable = manifest["clips"]
            self.assertEqual(first["boundary_method"], "forced_alignment")
            self.assertEqual(first["last_kept_word"], "selected")
            self.assertEqual(first["first_omitted_word"], "omitted")
            self.assertGreaterEqual(first["final_cut_seconds"], 0.62)
            self.assertLessEqual(first["final_cut_seconds"], 0.65)
            self.assertEqual(first["hard_boundary_fade_ms"], 2.0)
            self.assertEqual(stable["boundary_method"], "stable_silence")
            self.assertEqual(
                stable["final_cut_seconds"],
                stable["refined_end_seconds"],
            )
            self.assertEqual(manifest["hard_boundaries_found"], 1)
            self.assertEqual(manifest["successfully_aligned"], 1)
            self.assertEqual(manifest["alignment_failures"], 0)

            final_audio, _ = sf.read(
                manifest["hard_boundary_aligned_wav"],
                dtype="float32",
                always_2d=True,
            )
            stable_audio, _ = sf.read(
                stable["refined_clip_wav"],
                dtype="float32",
                always_2d=True,
            )
            start = stable["final_output_start_sample"]
            end = stable["final_output_end_sample"]
            self.assertTrue(np.array_equal(final_audio[start:end], stable_audio))
            gap_start = first["final_output_end_sample"]
            gap_end = stable["final_output_start_sample"]
            self.assertEqual(gap_end - gap_start, 80)
            self.assertTrue(np.all(final_audio[gap_start:gap_end] == 0.0))

            debug = root / "forced" / "forced_alignment_debug" / "clip_000"
            for filename in (
                "context.wav",
                "old_clip.wav",
                "new_clip.wav",
                "alignment.json",
                "alignment_plot.png",
            ):
                self.assertTrue((debug / filename).is_file(), filename)

    def test_failed_alignment_preserves_previous_hard_clip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, refined_manifest = self._source_and_plan(root)
            worker = {
                "clip_index": 0,
                "error": "kept word did not align",
                "aligned": None,
            }

            manifest = render_forced_aligned_preview(
                refined_manifest_path=refined_manifest,
                output_dir=root / "forced",
                alignment_python=root / "unused-python",
                alignment_payload={"jobs": [worker]},
            )

            first = manifest["clips"][0]
            self.assertEqual(
                first["boundary_method"],
                "forced_alignment_failed",
            )
            self.assertEqual(
                first["final_cut_seconds"],
                first["silence_refined_end_seconds"],
            )
            self.assertEqual(manifest["successfully_aligned"], 0)
            self.assertEqual(manifest["alignment_failures"], 1)
            old_audio, _ = sf.read(
                first["forced_alignment_old_clip_wav"],
                dtype="float32",
                always_2d=True,
            )
            new_audio, _ = sf.read(
                first["forced_alignment_new_clip_wav"],
                dtype="float32",
                always_2d=True,
            )
            self.assertTrue(np.array_equal(old_audio, new_audio))

    def test_zero_hard_boundaries_pass_through_without_loading_model(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path, plan_path, _ = self._source_and_plan(root)
            plan = read_json(plan_path)
            plan["committed"] = [
                {
                    "canonical_text": "selected omitted last",
                    "source_ranges": [{"start_word_id": 0, "end_word_id": 3}],
                }
            ]
            plan["selected_source_ranges"] = [{"start_word_id": 0, "end_word_id": 3}]
            write_json(plan_path, plan)
            trailing_dir = root / "trailing_no_hard"
            render_trailing_refined_preview(
                audio_path=audio_path,
                plan_path=plan_path,
                output_dir=trailing_dir,
            )

            manifest = render_forced_aligned_preview(
                refined_manifest_path=(trailing_dir / "render_manifest_refined.json"),
                output_dir=root / "forced_no_hard",
                alignment_python=root / "missing-python-is-not-needed",
            )

            self.assertEqual(manifest["hard_boundaries_found"], 0)
            self.assertEqual(manifest["successfully_aligned"], 0)
            self.assertEqual(manifest["alignment_failures"], 0)
            self.assertTrue(Path(manifest["hard_boundary_aligned_wav"]).is_file())


if __name__ == "__main__":
    unittest.main()
