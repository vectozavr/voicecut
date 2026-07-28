from __future__ import annotations

import math
import unittest

import numpy as np


from voicecut import render


classify_interword_gap_core = getattr(
    render,
    "classify_interword_gap_core",
    None,
)


def rms(audio: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(audio, dtype=np.float64))))


def scaled_noise(
    seed: int,
    length: int,
    *,
    rms_db: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(length).astype(np.float32)
    current = rms(noise)
    return (noise * (10.0 ** (rms_db / 20.0) / current)).astype(np.float32)


def smooth_broadband_event(
    seed: int,
    length: int,
    *,
    sample_rate: int,
    rms_db: float,
    ramp_seconds: float = 0.018,
) -> np.ndarray:
    # First-differenced white noise has the broad, high-frequency spectrum of
    # a synthetic inhalation without becoming a tonal speech surrogate.
    raw = scaled_noise(seed, length + 1, rms_db=rms_db)
    event = np.diff(raw).astype(np.float32)
    event *= 10.0 ** (rms_db / 20.0) / max(rms(event), 1e-12)
    ramp = min(round(ramp_seconds * sample_rate), max(1, length // 3))
    envelope = np.ones(length, dtype=np.float32)
    theta = np.linspace(0.0, math.pi / 2.0, ramp, endpoint=True)
    transition = np.square(np.sin(theta)).astype(np.float32)
    envelope[:ramp] = transition
    envelope[-ramp:] = transition[::-1]
    return event * envelope


@unittest.skipUnless(
    callable(classify_interword_gap_core),
    "render.classify_interword_gap_core has not been implemented yet",
)
class InterwordGapClassificationTests(unittest.TestCase):
    sample_rate = 16_000
    room_floor_db = -65.0
    guard_seconds = 0.070

    def full_gap_fixture(
        self,
        *,
        include_breath: bool,
        include_s_tail: bool,
    ) -> dict[str, object]:
        sample_rate = self.sample_rate
        length = round(1.40 * sample_rate)
        left_end = round(0.25 * sample_rate)
        right_start = round(1.15 * sample_rate)
        guard = round(self.guard_seconds * sample_rate)
        safe_start = left_end + guard
        safe_end = right_start - guard
        audio = scaled_noise(7001, length, rms_db=self.room_floor_db)
        room = scaled_noise(7002, length, rms_db=self.room_floor_db)

        if include_s_tail:
            tail_length = round(0.055 * sample_rate)
            time = np.arange(tail_length, dtype=np.float32) / sample_rate
            tail = (
                0.016
                * np.sin(2.0 * math.pi * 4_200.0 * time)
                * np.linspace(1.0, 0.15, tail_length, dtype=np.float32)
            )
            audio[left_end : left_end + tail_length] += tail
            s_tail_slice = slice(left_end, left_end + tail_length)
        else:
            s_tail_slice = slice(left_end, left_end)

        breath_start = safe_start + round(0.20 * sample_rate)
        breath_end = safe_start + round(0.45 * sample_rate)
        if include_breath:
            audio[breath_start:breath_end] += smooth_broadband_event(
                7003,
                breath_end - breath_start,
                sample_rate=sample_rate,
                rms_db=-39.0,
            )

        return {
            "audio": audio,
            "room": room,
            "left_end": left_end,
            "right_start": right_start,
            "safe_start": safe_start,
            "safe_end": safe_end,
            "s_tail_slice": s_tail_slice,
            "breath_start": breath_start,
            "breath_end": breath_end,
        }

    def classify_safe_core(
        self,
        fixture: dict[str, object],
    ) -> tuple[bool, dict[str, object]]:
        audio = fixture["audio"]
        self.assertIsInstance(audio, np.ndarray)
        safe_start = int(fixture["safe_start"])
        safe_end = int(fixture["safe_end"])
        accepted, report = classify_interword_gap_core(
            audio[safe_start:safe_end],
            self.sample_rate,
            self.room_floor_db,
        )
        self.assertIsInstance(report, dict)
        return bool(accepted), report

    def clean_if_accepted(
        self,
        fixture: dict[str, object],
    ) -> tuple[np.ndarray, dict[str, object], dict[str, object] | None]:
        audio = fixture["audio"]
        room = fixture["room"]
        self.assertIsInstance(audio, np.ndarray)
        self.assertIsInstance(room, np.ndarray)
        accepted, report = self.classify_safe_core(fixture)
        if not accepted:
            return audio.copy(), report, None
        cleaned, edit = render.attenuate_interword_gap(
            audio,
            sample_rate=self.sample_rate,
            left_word_end_sample=int(fixture["left_end"]),
            right_word_start_sample=int(fixture["right_start"]),
            room_tone=room,
            guard_seconds=self.guard_seconds,
        )
        return cleaned, report, edit

    def test_ctc_fenced_quiet_breath_is_localized_and_cleaned(self) -> None:
        fixture = self.full_gap_fixture(
            include_breath=True,
            include_s_tail=False,
        )
        audio = fixture["audio"]
        self.assertIsInstance(audio, np.ndarray)
        original = audio.copy()
        accepted, report = self.classify_safe_core(fixture)

        self.assertTrue(accepted)
        self.assertEqual(report.get("status"), "breath_candidate")
        self.assertIn("candidate_start_sample", report)
        self.assertIn("candidate_end_sample", report)

        expected_start = int(fixture["breath_start"]) - int(fixture["safe_start"])
        expected_end = int(fixture["breath_end"]) - int(fixture["safe_start"])
        tolerance = round(0.020 * self.sample_rate)
        self.assertLessEqual(
            abs(int(report["candidate_start_sample"]) - expected_start),
            tolerance,
        )
        self.assertLessEqual(
            abs(int(report["candidate_end_sample"]) - expected_end),
            tolerance,
        )

        cleaned, second_report, edit = self.clean_if_accepted(fixture)
        self.assertEqual(second_report.get("status"), "breath_candidate")
        self.assertIsInstance(edit, dict)
        self.assertEqual(len(cleaned), len(original))
        self.assertEqual(cleaned.dtype, original.dtype)
        self.assertTrue(np.all(np.isfinite(cleaned)))
        np.testing.assert_array_equal(audio, original)

        safe_start = int(fixture["safe_start"])
        safe_end = int(fixture["safe_end"])
        np.testing.assert_array_equal(cleaned[:safe_start], original[:safe_start])
        np.testing.assert_array_equal(cleaned[safe_end:], original[safe_end:])

        breath_slice = slice(
            int(fixture["breath_start"]) + round(0.030 * self.sample_rate),
            int(fixture["breath_end"]) - round(0.030 * self.sample_rate),
        )
        self.assertLess(
            rms(cleaned[breath_slice]) / rms(original[breath_slice]),
            0.10,
        )
        self.assertGreater(rms(cleaned[breath_slice]), 1e-7)

    def test_edge_attached_s_tail_stays_bit_identical_when_core_is_cleaned(
        self,
    ) -> None:
        fixture = self.full_gap_fixture(
            include_breath=True,
            include_s_tail=True,
        )
        audio = fixture["audio"]
        self.assertIsInstance(audio, np.ndarray)
        original = audio.copy()

        cleaned, report, edit = self.clean_if_accepted(fixture)

        self.assertEqual(report.get("status"), "breath_candidate")
        self.assertIsInstance(edit, dict)
        tail = fixture["s_tail_slice"]
        self.assertIsInstance(tail, slice)
        np.testing.assert_array_equal(cleaned[tail], original[tail])
        # The /s/ ends 15 ms before the 70 ms hard guard ends.
        self.assertLessEqual(tail.stop, int(fixture["safe_start"]))
        self.assertEqual(len(cleaned), len(original))

    def test_ordinary_room_tone_is_rejected_and_not_processed(self) -> None:
        fixture = self.full_gap_fixture(
            include_breath=False,
            include_s_tail=False,
        )
        audio = fixture["audio"]
        self.assertIsInstance(audio, np.ndarray)
        original = audio.copy()

        cleaned, report, edit = self.clean_if_accepted(fixture)

        self.assertEqual(report.get("status"), "room_tone")
        self.assertIsNone(edit)
        np.testing.assert_array_equal(cleaned, original)
        np.testing.assert_array_equal(audio, original)

    def test_fifteen_ms_high_crest_click_is_rejected(self) -> None:
        fixture = self.full_gap_fixture(
            include_breath=False,
            include_s_tail=False,
        )
        audio = fixture["audio"]
        self.assertIsInstance(audio, np.ndarray)
        safe_start = int(fixture["safe_start"])
        safe_end = int(fixture["safe_end"])
        click_length = round(0.015 * self.sample_rate)
        click_start = (safe_start + safe_end - click_length) // 2
        time = np.arange(click_length, dtype=np.float32) / self.sample_rate
        click = (
            0.16
            * np.exp(-time / 0.0012)
            * np.cos(2.0 * math.pi * 1_800.0 * time)
        ).astype(np.float32)
        # Retain a clear high-crest onset while making the complete artifact
        # exactly 15 ms long.
        click[0] = 0.22
        audio[click_start : click_start + click_length] += click
        original = audio.copy()

        cleaned, report, edit = self.clean_if_accepted(fixture)

        self.assertEqual(report.get("status"), "high_crest_transient")
        self.assertIsNone(edit)
        np.testing.assert_array_equal(cleaned, original)
        np.testing.assert_array_equal(audio, original)
        self.assertEqual(len(cleaned), len(original))


if __name__ == "__main__":
    unittest.main()
