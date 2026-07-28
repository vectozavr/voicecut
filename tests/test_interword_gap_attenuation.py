from __future__ import annotations

import math
import unittest

import numpy as np


from voicecut import render


attenuate_interword_gap = getattr(render, "attenuate_interword_gap", None)


def scaled_noise(
    rng: np.random.Generator,
    length: int,
    *,
    rms_db: float,
) -> np.ndarray:
    noise = rng.standard_normal(length).astype(np.float32)
    current_rms = math.sqrt(float(np.mean(np.square(noise, dtype=np.float64))))
    target_rms = 10.0 ** (rms_db / 20.0)
    return (noise * (target_rms / current_rms)).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(audio, dtype=np.float64))))


@unittest.skipUnless(
    callable(attenuate_interword_gap),
    "render.attenuate_interword_gap has not been implemented yet",
)
class GuardedInterwordGapAttenuationTests(unittest.TestCase):
    sample_rate = 8_000
    guard_seconds = 0.070
    minimum_editable_seconds = 0.080
    attenuation_db = -36.0
    fade_seconds = 0.025

    def call(
        self,
        audio: np.ndarray,
        *,
        left_word_end_sample: int,
        right_word_start_sample: int,
        room_tone: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object] | None]:
        return attenuate_interword_gap(
            audio,
            sample_rate=self.sample_rate,
            left_word_end_sample=left_word_end_sample,
            right_word_start_sample=right_word_start_sample,
            room_tone=room_tone,
            guard_seconds=self.guard_seconds,
            minimum_editable_seconds=self.minimum_editable_seconds,
            attenuation_db=self.attenuation_db,
            fade_seconds=self.fade_seconds,
        )

    def breath_fixture(
        self,
    ) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
        rng = np.random.default_rng(20260725)
        length = round(1.35 * self.sample_rate)
        left_word_end = round(0.27 * self.sample_rate)
        right_word_start = round(1.08 * self.sample_rate)
        guard = round(self.guard_seconds * self.sample_rate)
        edit_start = left_word_end + guard
        edit_end = right_word_start - guard

        room_tone = scaled_noise(rng, length, rms_db=-65.0)
        audio = room_tone.copy()
        # A deterministic broadband breath is much louder than the captured
        # room, and extends into both word guard zones.  The implementation
        # may attenuate only the strictly guarded interior.
        breath = scaled_noise(
            np.random.default_rng(8675309),
            right_word_start - left_word_end,
            rms_db=-31.0,
        )
        audio[left_word_end:right_word_start] += breath
        original = audio.copy()
        return original, room_tone, left_word_end, right_word_start, edit_start, edit_end

    def test_word_guard_zones_and_all_outside_samples_are_bit_identical(
        self,
    ) -> None:
        (
            audio,
            room_tone,
            left_word_end,
            right_word_start,
            edit_start,
            edit_end,
        ) = self.breath_fixture()
        original = audio.copy()

        edited, report = self.call(
            audio,
            left_word_end_sample=left_word_end,
            right_word_start_sample=right_word_start,
            room_tone=room_tone,
        )

        self.assertIsInstance(report, dict)
        self.assertEqual(edited.shape, original.shape)
        self.assertEqual(edited.dtype, original.dtype)
        self.assertTrue(np.all(np.isfinite(edited)))
        # The helper must be functional rather than mutating its caller's
        # buffer, which is important when one clip has several candidate gaps.
        np.testing.assert_array_equal(audio, original)
        np.testing.assert_array_equal(edited[:edit_start], original[:edit_start])
        np.testing.assert_array_equal(edited[edit_end:], original[edit_end:])
        self.assertFalse(
            np.array_equal(
                edited[edit_start:edit_end],
                original[edit_start:edit_end],
            )
        )

    def test_synthetic_breath_is_strongly_reduced_but_not_replaced_by_zero(
        self,
    ) -> None:
        (
            audio,
            room_tone,
            left_word_end,
            right_word_start,
            edit_start,
            edit_end,
        ) = self.breath_fixture()
        fade = round(self.fade_seconds * self.sample_rate)
        measurement = slice(edit_start + 2 * fade, edit_end - 2 * fade)

        edited, report = self.call(
            audio,
            left_word_end_sample=left_word_end,
            right_word_start_sample=right_word_start,
            room_tone=room_tone,
        )

        self.assertIsInstance(report, dict)
        input_rms = rms(audio[measurement])
        output_rms = rms(edited[measurement])
        # The requested -36 dB gain plus room-tone mixing should comfortably
        # clear 22 dB of measured breath reduction.
        self.assertLess(output_rms / input_rms, 0.08)
        self.assertGreater(output_rms, 1e-7)
        self.assertFalse(np.any(edited[measurement] == 0.0))

    def test_short_phoneme_gap_is_never_gated(self) -> None:
        rng = np.random.default_rng(101)
        length = self.sample_rate
        audio = scaled_noise(rng, length, rms_db=-65.0)
        left_word_end = round(0.40 * self.sample_rate)
        right_word_start = round(0.58 * self.sample_rate)
        audio[left_word_end:right_word_start] += scaled_noise(
            np.random.default_rng(102),
            right_word_start - left_word_end,
            rms_db=-28.0,
        )
        original = audio.copy()
        room_tone = scaled_noise(
            np.random.default_rng(103),
            length,
            rms_db=-65.0,
        )

        edited, report = self.call(
            audio,
            left_word_end_sample=left_word_end,
            right_word_start_sample=right_word_start,
            room_tone=room_tone,
        )

        # 180 ms minus two 70 ms guards leaves only 40 ms, below the 80 ms
        # editable-core minimum.  Energy here may be a consonant or vowel tail.
        self.assertIsNone(report)
        np.testing.assert_array_equal(edited, original)
        np.testing.assert_array_equal(audio, original)

    def test_fades_reach_unity_at_guards_and_do_not_create_clicks(self) -> None:
        length = round(1.20 * self.sample_rate)
        left_word_end = round(0.20 * self.sample_rate)
        right_word_start = round(1.00 * self.sample_rate)
        guard = round(self.guard_seconds * self.sample_rate)
        edit_start = left_word_end + guard
        edit_end = right_word_start - guard
        fade = round(self.fade_seconds * self.sample_rate)

        audio = np.full(length, 0.020, dtype=np.float32)
        room_tone = np.full(length, 0.0005, dtype=np.float32)
        edited, report = self.call(
            audio,
            left_word_end_sample=left_word_end,
            right_word_start_sample=right_word_start,
            room_tone=room_tone,
        )

        self.assertIsInstance(report, dict)
        # Both edit edges begin at unity gain; otherwise the transition itself
        # adds a click even though samples outside the guards are untouched.
        self.assertAlmostEqual(float(edited[edit_start]), 0.020, places=6)
        self.assertAlmostEqual(float(edited[edit_end - 1]), 0.020, places=6)

        early_ratio = rms(edited[edit_start : edit_start + fade // 6]) / 0.020
        middle_ratio = rms(
            edited[edit_start + 2 * fade : edit_end - 2 * fade]
        ) / 0.020
        late_ratio = rms(edited[edit_end - fade // 6 : edit_end]) / 0.020
        self.assertGreater(early_ratio, 0.80)
        self.assertLess(middle_ratio, 0.08)
        self.assertGreater(late_ratio, 0.80)

        left_transition = edited[edit_start - 1 : edit_start + fade + 1]
        right_transition = edited[edit_end - fade - 1 : edit_end + 1]
        self.assertLess(float(np.max(np.abs(np.diff(left_transition)))), 0.001)
        self.assertLess(float(np.max(np.abs(np.diff(right_transition)))), 0.001)

    def test_ordinary_room_tone_is_preserved_bit_for_bit(self) -> None:
        length = round(1.10 * self.sample_rate)
        left_word_end = round(0.20 * self.sample_rate)
        right_word_start = round(0.90 * self.sample_rate)
        audio = scaled_noise(
            np.random.default_rng(2001),
            length,
            rms_db=-64.0,
        )
        original = audio.copy()
        # Use a different realization at nearly the same level.  If the
        # function needlessly "cleans" ordinary room tone, exact equality will
        # fail even though both noises have similar statistics.
        room_tone = scaled_noise(
            np.random.default_rng(2002),
            length,
            rms_db=-65.0,
        )

        edited, report = self.call(
            audio,
            left_word_end_sample=left_word_end,
            right_word_start_sample=right_word_start,
            room_tone=room_tone,
        )

        self.assertIsNone(report)
        np.testing.assert_array_equal(edited, original)
        np.testing.assert_array_equal(audio, original)


if __name__ == "__main__":
    unittest.main()
