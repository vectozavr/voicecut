#!/usr/bin/env python3
"""Render a take plan from the canonical source with energy-safe joins.

The take planner decides *which* source words remain.  This renderer decides
*where* the samples are cut, using three independent kinds of evidence:

* MFA word/phone alignment of every kept line identifies which stretch of the
  waveform belongs to which word (Whisper timings are only a fallback);
* the waveform energy decides where speech physically starts and stops, so a
  word's natural decay is kept and a cut can never land inside audible speech;
* Respiro-en breath probabilities stop a word's head or tail from swallowing an
  adjacent breath.

Every pause of the output -- at a cut or inside a contiguous take -- is then
normalized to a pause class chosen by the planner ("none", "comma",
"sentence", "paragraph").  Breaths, clicks and other non-speech sounds inside
pauses are replaced by clean ambience taken from the same recording.  Pieces
are joined by short equal-power crossfades that lie in silence, so retained
speech samples are copied unchanged except at dense word-to-word cuts, where a
4 ms fade is unavoidable.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import progress
from .breath_detection import DEFAULT_RESPIRO_CACHE_ROOT, breath_probabilities
from .common import read_json, sha256_file, write_json
from .language_profiles import get_language_profile
from .mfa_alignment import (
    DEFAULT_MFA_CACHE_ROOT,
    DEFAULT_MFA_PREFIX,
    MFA_VERSION,
    MFAAlignmentError,
    MFAContextSpec,
    MFASourceWord,
    align_mfa_contexts,
    source_word_alignment,
)

RENDERER_NAME = "cut_render_v1"
BOUNDARY_PLAN_NAME = "authoritative_single_pass_boundary_plan_v2"
HOP_SECONDS = 0.005
RESPIRO_HOP_SECONDS = 0.010
# (minimum, target, maximum) silence between two words, by pause class.
PAUSE_CLASSES: dict[str, tuple[float, float, float]] = {
    "none": (0.0, 0.10, 0.22),
    "comma": (0.12, 0.28, 0.45),
    "sentence": (0.35, 0.50, 0.75),
    "paragraph": (0.65, 0.85, 1.20),
}
LEAD_IN_SECONDS = 0.35
LEAD_OUT_SECONDS = 0.60
CROSSFADE_SECONDS = 0.010
DENSE_FADE_SECONDS = 0.004
SOFT_FADE_SECONDS = 0.030
DENSE_SEARCH_SECONDS = 0.080
LOUD_MARGIN_DB = 15.0
TAIL_DECAY_SECONDS = 0.35
HEAD_ONSET_SECONDS = 0.35
MIN_TAIL_SECONDS = 0.15
MIN_HEAD_SECONDS = 0.15
QUIET_RUN_SECONDS = 0.060
EDGE_MARGIN_SECONDS = 0.012
MIN_INTERNAL_PAUSE_SECONDS = float("inf")
CLICK_MAX_SECONDS = 0.04
BREATH_PROBABILITY = 0.5
MFA_CROP_PAD_SECONDS = 0.30
AMBIENCE_MIN_RUN_SECONDS = 0.25
AMBIENCE_WORD_GUARD_SECONDS = 0.12
AMBIENCE_CROSSFADE_SECONDS = 0.030
# Fallback when no run meets the above: shorter runs closer to the words.
AMBIENCE_SHORT_MIN_RUN_SECONDS = 0.08
AMBIENCE_SHORT_GUARD_SECONDS = 0.03


class CutRenderError(RuntimeError):
    """The take plan cannot be rendered safely."""


# ---------------------------------------------------------------------------
# Signal analysis


@dataclass
class SignalAnalysis:
    sample_rate: int
    db: np.ndarray  # RMS level per HOP_SECONDS frame
    breath: np.ndarray | None  # Respiro probability per RESPIRO_HOP_SECONDS
    breath_threshold: float
    noise_floor_db: float
    speech_db: float
    sound_threshold_db: float

    def frame(self, seconds: float) -> int:
        return int(max(0.0, seconds) / HOP_SECONDS)

    def is_sound(self, frame: int) -> bool:
        return 0 <= frame < len(self.db) and self.db[frame] >= self.sound_threshold_db

    def is_loud(self, frame: int) -> bool:
        """Speech-level sound: never part of a pause."""

        return (
            0 <= frame < len(self.db)
            and self.db[frame] >= self.speech_db - LOUD_MARGIN_DB
        )

    def breath_at(self, seconds: float) -> float:
        if self.breath is None:
            return 0.0
        index = round(seconds / RESPIRO_HOP_SECONDS)
        if 0 <= index < len(self.breath):
            return float(self.breath[index])
        return 0.0

    def is_breath(self, seconds: float) -> bool:
        return self.breath_at(seconds) >= self.breath_threshold

    def clean(self, start: float, end: float) -> bool:
        """Return true when ``[start, end)`` holds no sound and no breath."""

        if end <= start:
            return True
        first, last = self.frame(start), max(self.frame(start), self.frame(end) - 1)
        if np.any(self.db[first : last + 1] >= self.sound_threshold_db):
            return False
        if self.breath is not None:
            lo = int(start / RESPIRO_HOP_SECONDS)
            hi = math.ceil(end / RESPIRO_HOP_SECONDS)
            if np.any(self.breath[lo:hi] >= self.breath_threshold):
                return False
        return True


def frame_levels(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """RMS level (dB) of a two-hop window centred on every hop boundary."""

    hop = max(1, round(HOP_SECONDS * sample_rate))
    full = len(mono) // hop
    # blocks[j + 1] is the energy of samples [j * hop, (j + 1) * hop); computed
    # in chunks so an hour of audio never needs a float64 copy.
    blocks = np.zeros(full + 2, np.float64)
    step = hop * 100_000
    for start in range(0, full * hop, step):
        chunk = mono[start : min(start + step, full * hop)].reshape(-1, hop)
        first = start // hop + 1
        blocks[first : first + len(chunk)] = np.einsum("ij,ij->i", chunk, chunk)
    rest = mono[full * hop :]
    blocks[full + 1] = float(np.dot(rest, rest)) if len(rest) else 0.0
    energy = (blocks[:-1] + blocks[1:]) / (2 * hop)
    return (10.0 * np.log10(energy + 1e-14)).astype(np.float32)


def analyze_signal(
    mono: np.ndarray,
    sample_rate: int,
    *,
    word_spans: Sequence[tuple[float, float]],
    breath: np.ndarray | None,
    breath_threshold: float = BREATH_PROBABILITY,
) -> SignalAnalysis:
    db = frame_levels(mono, sample_rate)
    in_word = np.zeros(len(db), bool)
    near_word = np.zeros(len(db), bool)
    guard = int(0.15 / HOP_SECONDS)
    for start, end in word_spans:
        first, last = int(start / HOP_SECONDS), int(end / HOP_SECONDS) + 1
        in_word[first:last] = True
        near_word[max(0, first - guard) : last + guard] = True
    gaps = db[~near_word]
    if len(gaps) < 20:
        gaps = db
    floor = float(np.percentile(gaps, 20))
    speech = float(np.percentile(db[in_word], 50)) if np.any(in_word) else floor + 30
    dynamic = max(0.0, speech - floor)
    threshold = floor + max(6.0, min(15.0, 0.2 * dynamic))
    return SignalAnalysis(
        sample_rate=sample_rate,
        db=db,
        breath=breath,
        breath_threshold=breath_threshold,
        noise_floor_db=floor,
        speech_db=speech,
        sound_threshold_db=threshold,
    )


# ---------------------------------------------------------------------------
# Plan words


@dataclass
class PlanWord:
    id: int
    text: str
    start: float
    end: float
    kept: bool = False
    break_after: str = "none"
    core_start: float = 0.0
    core_end: float = 0.0
    alignment: str = "whisper"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text.casefold())


def _break_from_punctuation(text: str) -> str | None:
    stripped = text.rstrip("\"'”’»)]")
    if not stripped:
        return None
    if stripped.endswith(("...", "…")):
        return None
    if stripped[-1] in ".?!":
        return "sentence"
    if stripped[-1] in ",;:—–-":
        return "comma"
    return None


def assign_breaks(words: list[PlanWord], items: Sequence[dict[str, Any]]) -> None:
    """Mark kept words and the pause class after each kept word.

    The class after an item's last word is the planner's ``break_after``.  Inside
    an item, the planner's punctuated text is aligned to the source words and its
    punctuation becomes the pause class; unmatched words fall back to Whisper's.
    """

    for item in items:
        first, last = int(item["first_id"]), int(item["last_id"])
        span = words[first : last + 1]
        for word in span:
            word.kept = True
            word.break_after = "none"
        planner_pieces = str(item.get("text", "")).split()
        planner_tokens: list[tuple[str, str]] = []
        for piece in planner_pieces:
            for token in _tokens(piece):
                planner_tokens.append((token, piece))
        source_tokens: list[tuple[str, int]] = []
        for word in span:
            for token in _tokens(word.text):
                source_tokens.append((token, word.id))
        matcher = SequenceMatcher(
            a=[token for token, _ in source_tokens],
            b=[token for token, _ in planner_tokens],
            autojunk=False,
        )
        matched: dict[int, str] = {}
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                word_id = source_tokens[block.a + offset][1]
                matched[word_id] = planner_tokens[block.b + offset][1]
        for word in span[:-1]:
            text = matched.get(word.id, word.text)
            word.break_after = _break_from_punctuation(text) or "none"
        span[-1].break_after = str(item.get("break_after", "sentence"))


def mfa_core_timings(
    *,
    audio_path: Path,
    words: list[PlanWord],
    work_dir: Path,
    language: str,
    prefix: Path,
    cache_root: Path,
    micromamba: str,
    num_jobs: int,
    line_gap: float,
) -> dict[str, Any]:
    """Align every line that holds a kept word; set ``core_start/core_end``."""

    lines: list[list[PlanWord]] = []
    for word in words:
        if lines and word.start - max(w.end for w in lines[-1]) < line_gap:
            lines[-1].append(word)
        else:
            lines.append([word])
    # Kept lines need precise words; their neighbours bound how far a kept
    # word's head or tail may extend.
    needed = set()
    for index, line in enumerate(lines):
        if any(word.kept for word in line):
            needed.update({index - 1, index, index + 1})
    contexts: list[MFAContextSpec] = []
    for index, line in enumerate(lines):
        if index not in needed:
            continue
        previous_end = lines[index - 1][-1].end if index else 0.0
        next_start = lines[index + 1][0].start if index + 1 < len(lines) else None
        start = max(
            0.0,
            line[0].start
            - min(MFA_CROP_PAD_SECONDS, (line[0].start - previous_end) / 2),
        )
        end = line[-1].end + (
            MFA_CROP_PAD_SECONDS
            if next_start is None
            else min(MFA_CROP_PAD_SECONDS, (next_start - line[-1].end) / 2)
        )
        contexts.append(
            MFAContextSpec(
                context_id=f"line_{index:05d}",
                crop_source_start_seconds=start,
                crop_source_end_seconds=end,
                words=tuple(
                    MFASourceWord(
                        word_id=word.id,
                        text=word.text,
                        start_seconds=word.start,
                        end_seconds=word.end,
                        selected=word.kept,
                    )
                    for word in line
                ),
                boundary_ids=(),
            )
        )
    for word in words:
        word.core_start, word.core_end, word.alignment = word.start, word.end, "whisper"
    if not contexts:
        return {"status": "no_contexts", "aligned_words": 0, "context_errors": []}
    result = align_mfa_contexts(
        audio_path=audio_path,
        contexts=contexts,
        work_dir=work_dir,
        prefix=prefix,
        cache_root=cache_root,
        micromamba=micromamba,
        num_jobs=num_jobs,
        language=language,
        model_id=get_language_profile(language).mfa_model,
    )
    aligned = 0
    for context in result.get("contexts", []):
        for word_id in context.get("ordered_source_word_ids", []):
            try:
                evidence = source_word_alignment(context, int(word_id))
            except MFAAlignmentError:
                continue
            word = words[int(word_id)]
            start = float(evidence["first_non_silence_phone"]["start_seconds"])
            end = float(evidence["last_non_silence_phone"]["end_seconds"])
            if end <= start:
                continue
            word.core_start, word.core_end, word.alignment = start, end, "mfa"
            aligned += 1
    return {
        "status": "complete",
        "contexts": len(contexts),
        "aligned_words": aligned,
        "context_errors": [
            {"context_id": error.get("context_id"), "code": error.get("code")}
            for error in result.get("context_errors", [])
        ],
    }


# ---------------------------------------------------------------------------
# Boundary refinement


def refine_tail(
    signal: SignalAnalysis,
    *,
    core_end: float,
    limit: float,
    use_breath: bool,
) -> tuple[float, str]:
    """Where the audible end of a word stops, never beyond ``limit``.

    Speech-level sound after the aligned end always belongs to the word (MFA
    can end a word early), so loud speech is never cut.  The quiet decay that
    follows is kept until 20 ms of silence or a breath, for at most
    ``TAIL_DECAY_SECONDS``.  The status is ``quiet`` or ``breath`` when the
    word really ends, ``max`` when only a faint decay was cut short, and
    ``limit`` when sound runs into the next word: connected speech.
    """

    frame = signal.frame(core_end)
    last = signal.frame(limit)
    while frame < last and signal.is_loud(frame):
        frame += 1
    speech_end = max(core_end, frame * HOP_SECONDS)
    minimum = min(limit, max(core_end + MIN_TAIL_SECONDS, speech_end))
    cap = min(limit, speech_end + TAIL_DECAY_SECONDS)
    quiet_needed = max(1, int(QUIET_RUN_SECONDS / HOP_SECONDS))
    quiet = 0
    while frame < signal.frame(cap):
        time = frame * HOP_SECONDS
        if time >= minimum and use_breath and signal.is_breath(time):
            return max(minimum, time), "breath"
        if signal.is_sound(frame):
            quiet = 0
        else:
            quiet += 1
            if quiet >= quiet_needed:
                # End inside the quiet run, so the crossfade lies in silence.
                end = (frame - quiet_needed + 1) * HOP_SECONDS + EDGE_MARGIN_SECONDS
                return min(limit, max(minimum, end)), "quiet"
        frame += 1
    return cap, ("limit" if cap >= limit - 1e-9 else "max")


def refine_head(
    signal: SignalAnalysis,
    *,
    core_start: float,
    limit: float,
    use_breath: bool,
) -> tuple[float, str]:
    """Where the audible onset of a word starts, never before ``limit``.

    Mirror image of :func:`refine_tail`: loud sound before the aligned start is
    part of the word, then soft onsets are kept for at most
    ``HEAD_ONSET_SECONDS``.
    """

    frame = signal.frame(core_start) - 1
    first = signal.frame(limit)
    while frame > first and signal.is_loud(frame):
        frame -= 1
    speech_start = min(core_start, (frame + 1) * HOP_SECONDS)
    maximum = max(limit, min(core_start - MIN_HEAD_SECONDS, speech_start))
    cap = max(limit, speech_start - HEAD_ONSET_SECONDS)
    quiet_needed = max(1, int(QUIET_RUN_SECONDS / HOP_SECONDS))
    quiet = 0
    while frame > signal.frame(cap):
        time = frame * HOP_SECONDS
        if time <= maximum and use_breath and signal.is_breath(time):
            return min(maximum, time + HOP_SECONDS), "breath"
        if signal.is_sound(frame):
            quiet = 0
        else:
            quiet += 1
            if quiet >= quiet_needed:
                start = (frame + quiet_needed) * HOP_SECONDS - EDGE_MARGIN_SECONDS
                return max(limit, min(maximum, start)), "quiet"
        frame -= 1
    return cap, ("limit" if cap <= limit + 1e-9 else "max")


def loud_onset_after(signal: SignalAnalysis, start: float) -> float:
    """First clearly voiced frame of a word whose MFA span begins at ``start``.

    MFA assigns the closure silence of an initial stop consonant to the word,
    so the previous word may be cut inside that silence rather than at the MFA
    boundary.
    """

    loud = signal.speech_db - 12.0
    frame = signal.frame(start)
    last = min(len(signal.db), signal.frame(start + DENSE_SEARCH_SECONDS))
    # Skip the still-falling decay of the previous word; a rising level is
    # already the onset of this word.
    while (
        frame < last
        and signal.db[frame] >= loud
        and frame > 0
        and signal.db[frame] <= signal.db[frame - 1] + 0.5
    ):
        frame += 1
    while frame < last and signal.db[frame] < loud:
        frame += 1
    return frame * HOP_SECONDS


def quietest_point(signal: SignalAnalysis, start: float, end: float) -> float:
    """Centre of the quietest analysis frame inside ``[start, end]``."""

    first, last = signal.frame(start), max(signal.frame(start), signal.frame(end))
    window = signal.db[first : last + 1]
    if not len(window):
        return start
    return (first + int(np.argmin(window)) + 0.5) * HOP_SECONDS


def clean_extent_after(signal: SignalAnalysis, start: float, limit: float) -> float:
    """Length of clean silence following ``start`` (until sound or breath)."""

    time = start
    while time < limit and signal.clean(time, min(limit, time + HOP_SECONDS)):
        time += HOP_SECONDS
    return max(0.0, min(limit, time) - start)


def clean_extent_before(signal: SignalAnalysis, end: float, limit: float) -> float:
    time = end
    while time > limit and signal.clean(max(limit, time - HOP_SECONDS), time):
        time -= HOP_SECONDS
    return max(0.0, end - max(limit, time))


# ---------------------------------------------------------------------------
# Ambience


@dataclass
class AmbienceBank:
    runs: list[tuple[int, int]]  # source sample ranges of clean room tone
    sample_rate: int
    status: str
    cursor: int = 0

    def take(self, samples: int, near_sample: int) -> list[tuple[int, int]]:
        """Return source ranges that together provide ``samples`` of ambience."""

        if samples <= 0 or not self.runs:
            return []
        # Prefer nearby ambience (same mic position and room), and runs long
        # enough to cover the pause without looping.
        ordered = sorted(
            self.runs,
            key=lambda run: (
                run[1] - run[0] < samples,
                abs(run[0] - near_sample) // (30 * self.sample_rate),
                -(run[1] - run[0]),
            ),
        )
        pieces: list[tuple[int, int]] = []
        remaining = samples
        overlap = int(AMBIENCE_CROSSFADE_SECONDS * self.sample_rate)
        index = 0
        while remaining > 0:
            run = ordered[index % len(ordered)]
            run_length = run[1] - run[0]
            length = min(run_length, remaining + (overlap if pieces else 0))
            # Vary the offset so repeated use of a run does not repeat audio.
            slack = run_length - length
            offset = (self.cursor * 7919) % (slack + 1) if slack > 0 else 0
            self.cursor += 1
            pieces.append((run[0] + offset, run[0] + offset + length))
            remaining -= length - (overlap if len(pieces) > 1 else 0)
            index += 1
            if index > 10_000:
                raise CutRenderError("ambience assembly did not converge")
        return pieces


def build_ambience_bank(
    signal: SignalAnalysis,
    words: Sequence[PlanWord],
    total_samples: int,
) -> AmbienceBank:
    runs = quiet_runs(
        signal,
        words,
        total_samples,
        guard_seconds=AMBIENCE_WORD_GUARD_SECONDS,
        minimum_seconds=AMBIENCE_MIN_RUN_SECONDS,
    )
    if runs:
        return AmbienceBank(
            runs=runs, sample_rate=signal.sample_rate, status="complete"
        )
    # Tight narration, with a breath in every pause, can leave no long clean
    # stretch; shorter ones closer to the words still crossfade into room tone.
    runs = quiet_runs(
        signal,
        words,
        total_samples,
        guard_seconds=AMBIENCE_SHORT_GUARD_SECONDS,
        minimum_seconds=AMBIENCE_SHORT_MIN_RUN_SECONDS,
    )
    status = "short_runs" if runs else "unavailable"
    return AmbienceBank(runs=runs, sample_rate=signal.sample_rate, status=status)


def quiet_runs(
    signal: SignalAnalysis,
    words: Sequence[PlanWord],
    total_samples: int,
    *,
    guard_seconds: float,
    minimum_seconds: float,
) -> list[tuple[int, int]]:
    """Source ranges within 3 dB of the floor, ``guard_seconds`` away from words
    and sounds, without breath, and at least ``minimum_seconds`` long."""

    sample_rate = signal.sample_rate
    usable = signal.db < signal.noise_floor_db + 3.0
    for word in words:
        first = signal.frame(word.start - guard_seconds)
        last = signal.frame(word.end + guard_seconds) + 1
        usable[first:last] = False
    sound = signal.db >= signal.sound_threshold_db
    guard = int(guard_seconds / HOP_SECONDS)
    for index in np.flatnonzero(sound):
        usable[max(0, index - guard) : index + guard] = False
    if signal.breath is not None:
        ratio = round(RESPIRO_HOP_SECONDS / HOP_SECONDS)
        breathy = np.repeat(signal.breath >= 0.3, ratio)[: len(usable)]
        usable[: len(breathy)] &= ~breathy
    runs: list[tuple[int, int]] = []
    minimum = int(minimum_seconds / HOP_SECONDS)
    start = None
    for index, value in enumerate(np.append(usable, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= minimum:
                runs.append(
                    (
                        int(start * HOP_SECONDS * sample_rate),
                        min(total_samples, int(index * HOP_SECONDS * sample_rate)),
                    )
                )
            start = None
    return runs


# ---------------------------------------------------------------------------
# Edit timeline


@dataclass
class Piece:
    kind: str  # "source" or "room_tone"
    source_start: int
    source_end: int
    fade_in: int = 0
    fade_out: int = 0
    interval_index: int | None = None
    join_index: int | None = None
    trace: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Island:
    first_word: int
    last_word: int
    start: float  # source seconds, audible onset of the first word
    end: float  # source seconds, audible end of the last word


def _source_limit_after(
    words: Sequence[PlanWord], index: int, duration: float
) -> float:
    return words[index + 1].core_start if index + 1 < len(words) else duration


def _source_limit_before(words: Sequence[PlanWord], index: int) -> float:
    return words[index - 1].core_end if index > 0 else 0.0


def _core_mask(words: Sequence[PlanWord], frames: int) -> np.ndarray:
    mask = np.zeros(frames, bool)
    for word in words:
        first = int(word.core_start / HOP_SECONDS)
        mask[first : int(word.core_end / HOP_SECONDS) + 1] = True
    return mask


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open index runs where ``mask`` is true."""

    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def nonspeech_runs(
    signal: SignalAnalysis,
    *,
    start: float,
    end: float,
    core_mask: np.ndarray,
    use_breath: bool,
) -> list[tuple[float, float, bool]]:
    """Pauses inside ``[start, end]`` found from the waveform alone.

    Speech is every speech-level stretch longer than a click, plus its quiet
    decay (up to ``TAIL_DECAY_SECONDS``) and soft onset (up to
    ``HEAD_ONSET_SECONDS``); neither extends through a breath.  Everything else
    -- silence, breaths, faint noise, clicks -- is pause material.  Returns
    ``(start, end, clean)`` for every pause of at least
    ``MIN_INTERNAL_PAUSE_SECONDS``; it is clean when it holds neither sound nor
    breath.  Word alignment is not trusted here: MFA occasionally packs words
    into a silent hesitation.
    """

    first, last = signal.frame(start), signal.frame(end)
    if last - first < 2:
        return []
    db = signal.db[first:last]
    sound = db >= signal.sound_threshold_db
    loud = db >= signal.speech_db - LOUD_MARGIN_DB - 5.0
    breathy = np.zeros(len(db), bool)
    if use_breath and signal.breath is not None:
        index = np.clip(
            np.round(np.arange(first, last) * HOP_SECONDS / RESPIRO_HOP_SECONDS),
            0,
            len(signal.breath) - 1,
        ).astype(int)
        breathy = (signal.breath[index] >= signal.breath_threshold) & (
            ~core_mask[first:last] | ~loud
        )
    speech = loud & ~breathy
    # A click, or the edge of a breath the detector missed, is not speech.
    for run_start, run_stop in _runs(speech):
        if run_stop - run_start < CLICK_MAX_SECONDS / HOP_SECONDS:
            speech[run_start:run_stop] = False
    decay = int(TAIL_DECAY_SECONDS / HOP_SECONDS)
    onset = int(HEAD_ONSET_SECONDS / HOP_SECONDS)
    extended = speech.copy()
    budget = 0
    for frame in range(len(db)):  # decays after speech
        if speech[frame]:
            budget = decay
        elif budget and sound[frame] and not breathy[frame]:
            extended[frame] = True
            budget -= 1
        else:
            budget = 0
    budget = 0
    for frame in range(len(db) - 1, -1, -1):  # soft onsets before speech
        if speech[frame]:
            budget = onset
        elif budget and sound[frame] and not breathy[frame]:
            extended[frame] = True
            budget -= 1
        else:
            budget = 0
    runs: list[tuple[float, float, bool]] = []
    for run_start, run_stop in _runs(~extended):
        if run_stop - run_start < MIN_INTERNAL_PAUSE_SECONDS / HOP_SECONDS:
            continue
        clean = not use_breath or not (
            sound[run_start:run_stop].any() or breathy[run_start:run_stop].any()
        )
        runs.append(
            (
                (first + run_start) * HOP_SECONDS,
                (first + run_stop) * HOP_SECONDS,
                clean,
            )
        )
    return runs


def plan_timeline(
    *,
    words: list[PlanWord],
    signal: SignalAnalysis,
    duration: float,
    pause_policy: str,
    use_breath: bool,
) -> tuple[list[Island], list[dict[str, Any]]]:
    """Group kept words into islands and describe every join between them.

    Islands are first split where the plan removes source words (cuts).  Inside
    every contiguous take, pauses are then found from the waveform: a natural
    pause that is clean and not longer than its class maximum stays untouched;
    a longer one is shortened, and a noisy one is cleaned.
    """

    del pause_policy
    kept = [word for word in words if word.kept]
    if not kept:
        raise CutRenderError("the plan keeps no words")
    core_mask = _core_mask(words, len(signal.db))

    def head_of(word: PlanWord) -> tuple[float, str]:
        """Audible onset of ``word``; status ``limit`` means connected speech."""

        if word.id == 0:
            return refine_head(
                signal, core_start=word.core_start, limit=0.0, use_breath=use_breath
            )
        previous_end = _source_limit_before(words, word.id)
        limit = min(word.core_start, previous_end + 0.005)
        head, status = refine_head(
            signal, core_start=word.core_start, limit=limit, use_breath=use_breath
        )
        if status != "limit":
            return head, status
        # Connected speech: cut at the quietest point near the MFA boundary.
        return (
            quietest_point(
                signal,
                max(previous_end, word.core_start) - 0.015,
                word.core_start + 0.010,
            ),
            "limit",
        )

    def tail_of(word: PlanWord) -> tuple[float, str]:
        """Audible end of ``word``; status ``limit`` means connected speech."""

        if word.id + 1 >= len(words):
            return refine_tail(
                signal, core_end=word.core_end, limit=duration, use_breath=use_breath
            )
        next_start = _source_limit_after(words, word.id, duration)
        limit = max(word.core_end, loud_onset_after(signal, next_start) - 0.005)
        tail, status = refine_tail(
            signal, core_end=word.core_end, limit=limit, use_breath=use_breath
        )
        if status != "limit":
            return tail, status
        return (
            quietest_point(
                signal,
                word.core_end - 0.010,
                min(next_start, word.core_end) + 0.015,
            ),
            "limit",
        )

    takes: list[list[PlanWord]] = []
    for word in kept:
        if takes and word.id == takes[-1][-1].id + 1:
            takes[-1].append(word)
        else:
            takes.append([word])
    edges = [(head_of(take[0]), tail_of(take[-1])) for take in takes]

    islands: list[Island] = []
    joins: list[dict[str, Any]] = []
    for number, take in enumerate(takes):
        (head, _), (tail, tail_status) = edges[number]
        current = Island(take[0].id, take[-1].id, head, tail)
        for run_start, run_end, clean in nonspeech_runs(
            signal,
            start=head + 0.02,
            end=tail - 0.02,
            core_mask=core_mask,
            use_breath=use_breath,
        ):
            before = [w for w in take if w.core_end <= run_start + 0.15]
            left = before[-1] if before else take[0]
            left = left if left.id >= current.first_word else words[current.first_word]
            after = [
                w for w in take if w.core_start >= run_end - 0.15 and w.id > left.id
            ]
            right = after[0] if after else None
            if right is None:
                continue
            pause_class = (
                left.break_after if left.break_after in PAUSE_CLASSES else "none"
            )
            lo, target, hi = PAUSE_CLASSES[pause_class]
            gap = run_end - run_start
            if clean and gap <= hi:
                continue
            # Keep the crossfades inside the pause, clear of the words.
            run_start += EDGE_MARGIN_SECONDS
            run_end -= EDGE_MARGIN_SECONDS
            current.last_word = left.id
            current.end = run_start
            islands.append(current)
            joins.append(
                {
                    "kind": "pause",
                    "left_word_id": left.id,
                    "right_word_id": right.id,
                    "contiguous_in_source": True,
                    "pause_class": pause_class,
                    "original_gap_seconds": gap,
                    "desired_gap_seconds": target if gap > hi else max(gap, lo),
                    "left_tail": run_start,
                    "right_head": run_end,
                    "left_status": "quiet",
                    "right_status": "quiet",
                    "original_clean": clean,
                    "limit_after": run_end,
                    "limit_before": run_start,
                }
            )
            current = Island(right.id, take[-1].id, run_end, tail)
        islands.append(current)
        if number + 1 < len(takes):
            left, right = take[-1], takes[number + 1][0]
            (next_head, next_status), _ = edges[number + 1]
            pause_class = (
                left.break_after if left.break_after in PAUSE_CLASSES else "none"
            )
            joins.append(
                {
                    "kind": "cut",
                    "left_word_id": left.id,
                    "right_word_id": right.id,
                    "contiguous_in_source": False,
                    "pause_class": pause_class,
                    "original_gap_seconds": next_head - tail,
                    "desired_gap_seconds": PAUSE_CLASSES[pause_class][1],
                    "left_tail": tail,
                    "right_head": next_head,
                    "left_status": tail_status,
                    "right_status": next_status,
                    "original_clean": None,
                    "limit_after": _source_limit_after(words, left.id, duration),
                    "limit_before": _source_limit_before(words, right.id),
                }
            )
    for join in joins:
        join["tail_dense"] = join["left_status"] == "limit"
        join["head_dense"] = join["right_status"] == "limit"
    return islands, joins


def build_pieces(
    *,
    islands: Sequence[Island],
    joins: list[dict[str, Any]],
    words: Sequence[PlanWord],
    signal: SignalAnalysis,
    bank: AmbienceBank,
    sample_rate: int,
    total_samples: int,
    pause_policy: str,
    use_breath: bool,
) -> list[Piece]:
    """Turn islands and joins into overlap-added source/ambience pieces.

    The desired gap is the silence between the left word's audible end and the
    right word's audible onset.  It is filled first with clean original silence
    next to each word, then with ambience from the bank, or with silence when
    the recording has no clean quiet to offer.  A video inserts nothing (its
    pictures follow the source), so a pause clean silence cannot fill keeps what
    was recorded there, a breath included, rather than shrink.
    """

    del words
    fade = round(CROSSFADE_SECONDS * sample_rate)
    dense_fade = round(DENSE_FADE_SECONDS * sample_rate)
    soft_fade = round(SOFT_FADE_SECONDS * sample_rate)

    def seconds_to_sample(value: float) -> int:
        return max(0, min(total_samples, round(value * sample_rate)))

    starts = [seconds_to_sample(island.start) for island in islands]
    ends = [seconds_to_sample(island.end) for island in islands]
    fades_in = [fade] * len(islands)
    fades_out = [fade] * len(islands)
    room_tone: list[int] = [0] * len(joins)

    for index, join in enumerate(joins):
        tail = join["left_tail"]
        head = join["right_head"]
        need = join["desired_gap_seconds"]
        if join["tail_dense"]:
            fades_out[index] = dense_fade
            recorded_left = clean_left = 0.0
        else:
            if join["left_status"] == "max":
                # A faint decay was cut short: fade it out gently.
                fades_out[index] = soft_fade
            limit = join["limit_after"] - 0.01
            recorded_left = max(0.0, limit - tail)
            clean_left = (
                clean_extent_after(signal, tail, limit) if use_breath else recorded_left
            )
        if join["head_dense"]:
            fades_in[index + 1] = dense_fade
            recorded_right = clean_right = 0.0
        else:
            limit = join["limit_before"] + 0.01
            recorded_right = max(0.0, head - limit)
            clean_right = (
                clean_extent_before(signal, head, limit)
                if use_breath
                else recorded_right
            )
        if join["contiguous_in_source"]:
            # Both sides share one original gap; never let them overlap.
            shared = max(0.0, head - tail)
            recorded_left = min(recorded_left, shared / 2)
            recorded_right = min(recorded_right, shared / 2)
            clean_left = min(clean_left, shared / 2)
            clean_right = min(clean_right, shared / 2)
        keep_recorded = pause_policy == "cuts" and clean_left + clean_right < need
        available_left = recorded_left if keep_recorded else clean_left
        available_right = recorded_right if keep_recorded else clean_right
        use_left = min(available_left, need / 2)
        use_right = min(available_right, need - use_left)
        use_left = min(available_left, need - use_right)
        remainder = max(0.0, need - use_left - use_right)
        ends[index] = seconds_to_sample(tail + use_left)
        starts[index + 1] = seconds_to_sample(head - use_right)
        if pause_policy == "semantic":
            room_tone[index] = round(remainder * sample_rate)
        join["used_left_silence_seconds"] = use_left
        join["used_right_silence_seconds"] = use_right
        join["room_tone_seconds"] = room_tone[index] / sample_rate
        join["kept_recorded_gap"] = keep_recorded and use_breath
        join["silent_room_tone"] = room_tone[index] > 0 and not bank.runs

    pieces: list[Piece] = []
    lead = int(LEAD_IN_SECONDS * sample_rate) if pause_policy == "semantic" else 0
    if lead:
        pieces.append(
            Piece(
                "room_tone",
                0,
                lead + fade,
                fade_in=int(0.05 * sample_rate),
                fade_out=fade,
                trace=bank.take(lead + fade, starts[0]),
            )
        )
    for index in range(len(islands)):
        pieces.append(
            Piece(
                "source",
                starts[index],
                ends[index],
                fade_in=fades_in[index],
                fade_out=fades_out[index],
                interval_index=index,
            )
        )
        if index < len(joins) and room_tone[index] > 0:
            fade_in, fade_out = fades_out[index], fades_in[index + 1]
            length = room_tone[index] + fade_in + fade_out
            pieces.append(
                Piece(
                    "room_tone",
                    0,
                    length,
                    fade_in=fade_in,
                    fade_out=fade_out,
                    join_index=index,
                    trace=bank.take(length, ends[index]),
                )
            )
    tail_length = (
        int(LEAD_OUT_SECONDS * sample_rate) if pause_policy == "semantic" else 0
    )
    if tail_length:
        pieces.append(
            Piece(
                "room_tone",
                0,
                tail_length + fade,
                fade_in=fade,
                fade_out=int(0.1 * sample_rate),
                trace=bank.take(tail_length + fade, ends[-1]),
            )
        )
    return pieces


def _equal_power(length: int, rising: bool) -> np.ndarray:
    if length <= 0:
        return np.ones(0, np.float32)
    phase = (np.arange(length, dtype=np.float64) + 0.5) / length * (math.pi / 2)
    curve = np.sin(phase) if rising else np.cos(phase)
    return curve.astype(np.float32)


def _assemble_room_tone(
    audio: np.ndarray, trace: Sequence[tuple[int, int]], length: int, sample_rate: int
) -> np.ndarray:
    channels = audio.shape[1]
    if not trace:
        return np.zeros((length, channels), np.float32)
    overlap = int(AMBIENCE_CROSSFADE_SECONDS * sample_rate)
    out = np.zeros((length + overlap, channels), np.float32)
    cursor = 0
    for number, (start, end) in enumerate(trace):
        chunk = audio[start:end].astype(np.float32).copy()
        if number:
            chunk[:overlap] *= _equal_power(min(overlap, len(chunk)), True)[:, None]
            cursor -= overlap
        if number + 1 < len(trace):
            chunk[-overlap:] *= _equal_power(min(overlap, len(chunk)), False)[:, None]
        stop = min(len(out), cursor + len(chunk))
        out[cursor:stop] += chunk[: stop - cursor]
        cursor += len(chunk)
    return out[:length]


def place_pieces(pieces: Sequence[Piece]) -> list[dict[str, Any]]:
    """Output position and applied fades of every piece, without audio.

    Consecutive pieces overlap by the shorter of the left fade-out and the
    right fade-in; that overlap is their equal-power crossfade.
    """

    records: list[dict[str, Any]] = []
    cursor = 0
    for index, piece in enumerate(pieces):
        overlap_in = min(pieces[index - 1].fade_out, piece.fade_in) if index else 0
        cursor -= overlap_in
        fade_out = (
            min(piece.fade_out, pieces[index + 1].fade_in)
            if index + 1 < len(pieces)
            else piece.fade_out
        )
        length = piece.source_end - piece.source_start
        records.append(
            {
                "piece_index": index,
                "kind": piece.kind,
                "output_start_sample": cursor,
                "output_end_sample": cursor + length,
                "source_start_sample": (
                    piece.source_start if piece.kind == "source" else None
                ),
                "source_end_sample": piece.source_end
                if piece.kind == "source"
                else None,
                "fade_in_samples": overlap_in if index else piece.fade_in,
                "fade_out_samples": fade_out,
                "source_interval_index": piece.interval_index,
                "join_index": piece.join_index,
                "ambience_source_ranges": [list(item) for item in piece.trace],
            }
        )
        cursor += length
    return records


def render_pieces(
    audio: np.ndarray,
    pieces: Sequence[Piece],
    sample_rate: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Overlap-add pieces from the canonical source; see :func:`place_pieces`."""

    segments = place_pieces(pieces)
    total = segments[-1]["output_end_sample"] if segments else 0
    output = np.zeros((total, audio.shape[1]), np.float32)
    for piece, segment in zip(pieces, segments, strict=True):
        length = piece.source_end - piece.source_start
        if piece.kind == "source":
            chunk = (
                audio[piece.source_start : piece.source_end].astype(np.float32).copy()
            )
        else:
            chunk = _assemble_room_tone(audio, piece.trace, length, sample_rate)
        fade_in, fade_out = segment["fade_in_samples"], segment["fade_out_samples"]
        if fade_in:
            chunk[:fade_in] *= _equal_power(fade_in, True)[:, None]
        if fade_out:
            chunk[-fade_out:] *= _equal_power(fade_out, False)[:, None]
        start = segment["output_start_sample"]
        output[start : start + length] += chunk
    return output, segments


def _contiguous_trace(segments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Partition the overlap-added output into disjoint traceable segments.

    Each crossfade belongs to its source side (to the left piece when both
    sides are source), so a source segment always maps one output sample to
    one source sample -- the contract the video renderer relies on.
    """

    def left_owns(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return left["kind"] == "source" or right["kind"] != "source"

    trace: list[dict[str, Any]] = []
    for number, segment in enumerate(segments):
        start = segment["output_start_sample"]
        end = segment["output_end_sample"]
        if number and not left_owns(segments[number - 1], segment):
            pass  # this segment owns the incoming overlap
        elif number:
            start = segments[number - 1]["output_end_sample"]
        if number + 1 < len(segments) and not left_owns(segment, segments[number + 1]):
            end = segments[number + 1]["output_start_sample"]
        if end <= start:
            continue
        record: dict[str, Any] = {
            "segment_index": len(trace),
            "kind": segment["kind"],
            "output_start_sample": start,
            "output_end_sample": end,
        }
        if segment["kind"] == "source":
            offset = start - segment["output_start_sample"]
            record["source_interval_index"] = segment["source_interval_index"]
            record["source_start_sample"] = segment["source_start_sample"] + offset
            record["source_end_sample"] = record["source_start_sample"] + (end - start)
        else:
            record["join_id"] = (
                f"join_{segment['join_index']:05d}"
                if segment["join_index"] is not None
                else "lead"
            )
        trace.append(record)
    return trace


# ---------------------------------------------------------------------------
# Entry point


def render_take_plan(
    *,
    audio_path: Path,
    plan_path: Path,
    output_dir: Path,
    language: str = "en",
    pause_policy: str = "semantic",
    breath_cleanup: str = "replace",
    breath_threshold: float = BREATH_PROBABILITY,
    mfa_prefix: Path = DEFAULT_MFA_PREFIX,
    mfa_cache_root: Path = DEFAULT_MFA_CACHE_ROOT,
    mfa_micromamba: str = "micromamba",
    mfa_num_jobs: int = 4,
    respiro_cache_root: Path | None = None,
) -> dict[str, Any]:
    if pause_policy not in {"semantic", "cuts"}:
        raise ValueError("pause_policy must be semantic or cuts")
    audio_path = audio_path.resolve()
    plan_path = plan_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = read_json(plan_path)
    if not isinstance(plan, dict) or plan.get("status") != "complete":
        raise CutRenderError("take plan is not complete")
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    total_samples = len(audio)
    duration = total_samples / sample_rate
    mono = audio[:, 0] if audio.shape[1] == 1 else audio.mean(axis=1)

    words = [
        PlanWord(
            id=int(word["id"]),
            text=str(word["text"]),
            start=float(word["start"]),
            end=float(word["end"]),
        )
        for word in plan["words"]
    ]
    if [word.id for word in words] != list(range(len(words))):
        raise CutRenderError("plan words must have consecutive IDs")
    items = plan["items"]
    assign_breaks(words, items)

    print("aligning kept lines with MFA...", flush=True)
    progress.report(5, 100, "aligning")
    try:
        alignment = mfa_core_timings(
            audio_path=audio_path,
            words=words,
            work_dir=output_dir,
            language=language,
            prefix=mfa_prefix,
            cache_root=mfa_cache_root,
            micromamba=mfa_micromamba,
            num_jobs=mfa_num_jobs,
            line_gap=float(plan.get("line_gap_seconds", 0.25)),
        )
    except (MFAAlignmentError, OSError, RuntimeError) as error:
        print(
            f"warning: MFA alignment failed, using Whisper timing: {error}",
            file=sys.stderr,
        )
        for word in words:
            word.core_start, word.core_end, word.alignment = (
                word.start,
                word.end,
                "whisper",
            )
        alignment = {"status": "failed", "error": str(error), "aligned_words": 0}
    # Whisper's timing is coarse; keep fallback cores inside their neighbours.
    for index, word in enumerate(words):
        if word.alignment == "whisper":
            lower = words[index - 1].core_end if index else 0.0
            word.core_start = max(word.core_start, lower)
            word.core_end = max(word.core_end, word.core_start + 0.02)

    progress.report(70, 100, "aligned")
    use_breath = breath_cleanup == "replace"
    breath = None
    breath_status = "off"
    if use_breath:
        print("detecting breaths with Respiro-en...", flush=True)
        try:
            breath = breath_probabilities(
                mono,
                sample_rate,
                cache_root=respiro_cache_root or DEFAULT_RESPIRO_CACHE_ROOT,
            )
            breath_status = "complete"
        except Exception as error:  # noqa: BLE001 - detector failure keeps energy evidence
            print(f"warning: breath detector failed: {error}", file=sys.stderr)
            breath_status = "detector_failed_energy_only"
    progress.report(85, 100, "rendering")
    signal = analyze_signal(
        mono,
        sample_rate,
        word_spans=[(word.core_start, word.core_end) for word in words],
        breath=breath,
        breath_threshold=breath_threshold,
    )
    islands, joins = plan_timeline(
        words=words,
        signal=signal,
        duration=duration,
        pause_policy=pause_policy,
        use_breath=use_breath,
    )
    bank = build_ambience_bank(signal, words, total_samples)
    pieces = build_pieces(
        islands=islands,
        joins=joins,
        words=words,
        signal=signal,
        bank=bank,
        sample_rate=sample_rate,
        total_samples=total_samples,
        pause_policy=pause_policy,
        use_breath=use_breath,
    )
    output, segments = render_pieces(audio, pieces, sample_rate)
    if not output.size or not np.all(np.isfinite(output)):
        raise CutRenderError("rendered audio is empty or not finite")
    final_wav = output_dir / "final_cut.wav"
    sf.write(str(final_wav), output, sample_rate, subtype="FLOAT")

    source_pieces = [piece for piece in pieces if piece.kind == "source"]
    source_intervals = [
        {
            "source_interval_index": index,
            "source_start_sample": piece.source_start,
            "source_end_sample": piece.source_end,
            "start_word_id": islands[index].first_word,
            "end_word_id": islands[index].last_word + 1,
        }
        for index, piece in enumerate(source_pieces)
    ]
    join_records = [
        {
            "join_id": f"join_{index:05d}",
            "join_kind": "source_discontinuity",
            "left_source_interval_index": index,
            "right_source_interval_index": index + 1,
            "inserted_pause_samples": round(join["room_tone_seconds"] * sample_rate),
            **{key: value for key, value in join.items()},
        }
        for index, join in enumerate(joins)
    ]
    boundary_plan = {
        "schema_version": 1,
        "planner": BOUNDARY_PLAN_NAME,
        "renderer": RENDERER_NAME,
        "status": "safe",
        "delivery_status": "complete",
        "pause_policy": pause_policy,
        "language": language,
        "source_audio": str(audio_path),
        "source_audio_sha256": sha256_file(audio_path),
        "source_sample_rate": sample_rate,
        "source_channel_count": int(audio.shape[1]),
        "source_frame_count": total_samples,
        "expected_output_frame_count": len(output),
        "take_plan": str(plan_path),
        "take_plan_sha256": sha256_file(plan_path),
        "signal": {
            "noise_floor_db": signal.noise_floor_db,
            "speech_db": signal.speech_db,
            "sound_threshold_db": signal.sound_threshold_db,
        },
        "alignment": alignment,
        "breath_detection": breath_status,
        "ambience": {"status": bank.status, "runs": len(bank.runs)},
        "pause_classes": {key: list(value) for key, value in PAUSE_CLASSES.items()},
        "words": [asdict(word) for word in words],
        "source_intervals": source_intervals,
        "joins": join_records,
        "pieces": segments,
        "output_segments": _contiguous_trace(segments),
    }
    boundary_path = output_dir / "final_boundary_plan.json"
    write_json(boundary_path, boundary_plan)
    removed_breaths = sum(
        1
        for join in joins
        if join["contiguous_in_source"] and join.get("original_clean") is False
    )
    manifest = {
        "schema_version": 1,
        "renderer": RENDERER_NAME,
        "status": "complete",
        "delivery_status": "complete",
        "language": language,
        "pause_policy": pause_policy,
        "breath_cleanup_mode": breath_cleanup,
        "breath_threshold": breath_threshold,
        "breath_cleanup_status": breath_status,
        "alignment_backend": "mfa",
        "mfa_version": MFA_VERSION,
        "mfa_model": get_language_profile(language).mfa_model,
        "mfa_aligned_words": alignment.get("aligned_words", 0),
        "source_audio": str(audio_path),
        "source_audio_sha256": sha256_file(audio_path),
        "take_plan": str(plan_path),
        "take_plan_sha256": sha256_file(plan_path),
        "final_boundary_plan": str(boundary_path),
        "final_boundary_plan_sha256": sha256_file(boundary_path),
        "final_cut_wav": str(final_wav),
        "final_cut_wav_sha256": sha256_file(final_wav),
        "sample_rate": sample_rate,
        "channel_count": int(audio.shape[1]),
        "frame_count": len(output),
        "duration_seconds": len(output) / sample_rate,
        "source_duration_seconds": duration,
        "source_intervals": len(source_intervals),
        "joins": len(joins),
        "cuts": sum(1 for join in joins if not join["contiguous_in_source"]),
        "pauses_normalized": sum(1 for join in joins if join["contiguous_in_source"]),
        "pauses_with_noise_replaced": removed_breaths,
        "ambience_status": bank.status,
    }
    write_json(output_dir / "final_render_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m voicecut.cut_render",
        description="Render a take plan with energy-safe joins.",
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", choices=("en", "ru"), default="en")
    parser.add_argument(
        "--pause-policy", choices=("semantic", "cuts"), default="semantic"
    )
    parser.add_argument(
        "--breath-cleanup", choices=("off", "replace"), default="replace"
    )
    parser.add_argument("--mfa-prefix", type=Path, default=DEFAULT_MFA_PREFIX)
    parser.add_argument("--mfa-cache-root", type=Path, default=DEFAULT_MFA_CACHE_ROOT)
    parser.add_argument("--mfa-micromamba", default="micromamba")
    parser.add_argument("--mfa-num-jobs", type=int, default=4)
    parser.add_argument("--breath-threshold", type=float, default=BREATH_PROBABILITY)
    parser.add_argument("--respiro-cache-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = render_take_plan(
        audio_path=args.audio,
        plan_path=args.plan,
        output_dir=args.output_dir,
        language=args.language,
        pause_policy=args.pause_policy,
        breath_cleanup=args.breath_cleanup,
        mfa_prefix=args.mfa_prefix,
        mfa_cache_root=args.mfa_cache_root,
        mfa_micromamba=args.mfa_micromamba,
        mfa_num_jobs=args.mfa_num_jobs,
        breath_threshold=args.breath_threshold,
        respiro_cache_root=args.respiro_cache_root,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
