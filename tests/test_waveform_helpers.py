from __future__ import annotations

import unittest

import numpy as np


from voicecut.render import (
    boundary_after,
    boundary_before,
    can_merge_adjacent_pieces,
    clamp_snapped_boundaries,
    corroborated_inward_ctc_start,
    equal_power_crossfade,
    room_tone_segment,
    snap_zero_crossing,
    true_runs,
)


class BoundaryTests(unittest.TestCase):
    def test_true_runs_finds_leading_middle_and_trailing_regions(self) -> None:
        mask = np.array([True, True, False, True, False, True, True, True])
        self.assertEqual(true_runs(mask), [(0, 2), (3, 4), (5, 8)])

    def test_quiet_valleys_are_preferred_for_both_boundaries(self) -> None:
        sample_rate = 1_000
        frame_samples = 10
        rms_frames = np.full(100, -20.0, dtype=np.float32)
        rms_frames[60:65] = -60.0
        rms_frames[72:78] = -60.0

        before, before_kind = boundary_before(
            target=700,
            lower=400,
            rms_frames=rms_frames,
            frame_samples=frame_samples,
            sample_rate=sample_rate,
            threshold_db=-40.0,
        )
        after, after_kind = boundary_after(
            target=700,
            upper=900,
            rms_frames=rms_frames,
            frame_samples=frame_samples,
            sample_rate=sample_rate,
            threshold_db=-40.0,
        )

        self.assertEqual((before, before_kind), (612, "quiet_valley"))
        self.assertEqual((after, after_kind), (780, "quiet_valley"))

    def test_active_audio_uses_small_bounded_fallback(self) -> None:
        rms_frames = np.full(100, -20.0, dtype=np.float32)
        before, before_kind = boundary_before(
            target=700,
            lower=400,
            rms_frames=rms_frames,
            frame_samples=10,
            sample_rate=1_000,
            threshold_db=-40.0,
        )
        after, after_kind = boundary_after(
            target=700,
            upper=900,
            rms_frames=rms_frames,
            frame_samples=10,
            sample_rate=1_000,
            threshold_db=-40.0,
        )
        self.assertEqual((before, before_kind), (652, "active_fallback"))
        self.assertEqual((after, after_kind), (790, "active_fallback"))

    def test_zero_crossing_snap_uses_nearest_crossing(self) -> None:
        audio = np.array([-0.5, -0.3, 0.2, 0.4, 0.3, -0.1, -0.2], dtype=np.float32)
        self.assertEqual(snap_zero_crossing(audio, proposed=4, radius=3), 5)

    def test_snapped_boundaries_cannot_cross_hard_fences_or_speech(self) -> None:
        self.assertEqual(
            clamp_snapped_boundaries(
                70,
                250,
                lower_fence=100,
                speech_start=120,
                speech_end=200,
                upper_fence=220,
            ),
            (100, 220),
        )
        self.assertEqual(
            clamp_snapped_boundaries(
                150,
                180,
                lower_fence=100,
                speech_start=120,
                speech_end=200,
                upper_fence=220,
            ),
            (120, 200),
        )

    def test_later_ctc_start_requires_a_corroborating_waveform_onset(self) -> None:
        sample_rate = 1_000
        source = np.full(2_000, 10.0 ** (-65.0 / 20.0), dtype=np.float32)
        source[1_200:1_300] = 10.0 ** (-24.0 / 20.0)

        anchored, used = corroborated_inward_ctc_start(
            source,
            sample_rate=sample_rate,
            asr_start=1.0,
            ctc_start=1.2,
            ctc_score=0.8,
            noise_floor_db=-65.0,
        )
        self.assertTrue(used)
        self.assertEqual(anchored, 1.2)

        quiet = np.full_like(source, 10.0 ** (-65.0 / 20.0))
        anchored, used = corroborated_inward_ctc_start(
            quiet,
            sample_rate=sample_rate,
            asr_start=1.0,
            ctc_start=1.2,
            ctc_score=0.8,
            noise_floor_db=-65.0,
        )
        self.assertFalse(used)
        self.assertEqual(anchored, 1.0)


class CrossfadeTests(unittest.TestCase):
    def test_equal_power_crossfade_has_expected_endpoints_and_shape(self) -> None:
        left = np.full(12, 0.25, dtype=np.float32)
        right = np.full(10, -0.5, dtype=np.float32)
        result = equal_power_crossfade(left, right, length=8)

        self.assertEqual(result.shape, (8,))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertAlmostEqual(float(result[0]), 0.25, places=6)
        self.assertAlmostEqual(float(result[-1]), -0.5, places=6)
        self.assertLessEqual(float(np.max(np.abs(result))), np.sqrt(0.25**2 + 0.5**2))

    def test_crossfade_length_is_clamped_to_shorter_input(self) -> None:
        left = np.ones(3, dtype=np.float32)
        right = np.zeros(7, dtype=np.float32)
        result = equal_power_crossfade(left, right, length=20)
        self.assertEqual(len(result), 3)
        np.testing.assert_allclose(
            result,
            np.cos(np.linspace(0.0, np.pi / 2.0, 3)),
            atol=1e-6,
        )

    def test_room_tone_tiling_adds_neither_zero_fill_nor_click_seam(self) -> None:
        sample_rate = 48_000
        source = np.linspace(0.02, 0.04, 4_800, dtype=np.float32)
        result = room_tone_segment(
            [source],
            length=14_000,
            index=0,
            sample_rate=sample_rate,
        )

        self.assertEqual(len(result), 14_000)
        self.assertFalse(np.any(result == 0.0))
        # A hard source-end/source-start join would jump by 0.02.  The 15 ms
        # crossfade keeps all single-sample changes orders of magnitude lower.
        self.assertLess(float(np.max(np.abs(np.diff(result)))), 0.001)


class PieceMergeTests(unittest.TestCase):
    @staticmethod
    def pieces(word_skip: int) -> tuple[dict[str, object], dict[str, object]]:
        previous: dict[str, object] = {
            "word_end": 10,
            "last_speech_sample": 10_000,
            "phrase_indices": [3],
        }
        current: dict[str, object] = {
            "word_start": 10 + word_skip,
            "first_speech_sample": 11_000,
            "phrase_indices": [4],
        }
        return previous, current

    def test_merge_requires_zero_skipped_words(self) -> None:
        previous, contiguous = self.pieces(word_skip=0)
        self.assertTrue(
            can_merge_adjacent_pieces(previous, contiguous, sample_rate=48_000)
        )

        previous, one_skipped = self.pieces(word_skip=1)
        self.assertFalse(
            can_merge_adjacent_pieces(previous, one_skipped, sample_rate=48_000)
        )

    def test_merge_also_requires_adjacent_phrases_and_small_source_gap(self) -> None:
        previous, current = self.pieces(word_skip=0)
        current["phrase_indices"] = [5]
        self.assertFalse(
            can_merge_adjacent_pieces(previous, current, sample_rate=48_000)
        )

        previous, current = self.pieces(word_skip=0)
        current["first_speech_sample"] = 25_000
        self.assertFalse(
            can_merge_adjacent_pieces(previous, current, sample_rate=48_000)
        )

    def test_reliable_ctc_gap_prevents_merging_across_an_asr_inhale(self) -> None:
        previous, current = self.pieces(word_skip=0)
        previous.update(
            {
                "ctc_end": 30.629,
                "ctc_last_score": 0.93,
            }
        )
        current.update(
            {
                "ctc_start": 31.031,
                "ctc_first_score": 0.65,
            }
        )
        # The coarse ASR samples imply only a 21 ms gap, but the reliable CTC
        # evidence identifies a 402 ms clause-boundary inhale.
        self.assertFalse(
            can_merge_adjacent_pieces(previous, current, sample_rate=48_000)
        )


if __name__ == "__main__":
    unittest.main()
