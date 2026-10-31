#!/usr/bin/env python3
"""Take selection: choose the final narration from a retake-heavy recording.

A human editor hears a whole passage before deciding which attempt of a
sentence survives: retakes often follow several other sentences, and a
speaker may re-read an entire paragraph minutes later.  This planner therefore
shows the model several minutes of transcript at once, together with the
audio of the same span when the backend can listen, plus a look-ahead it must
not decide yet.  The model may revoke its most recent decisions when a later
retake appears, and a final review re-decides every region where the
assembled narration still repeats itself.

The model returns only source word IDs, a readable text, and the pause class
after each kept item.  It never supplies sample coordinates; the renderer
derives those from the waveform.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from . import progress
from .common import read_json, sha256_file, write_json
from .planner_backends import (
    MediaPart,
    PermanentPlannerError,
    PlannerBackend,
    add_planner_backend_arguments,
    create_planner_backend,
    think_setting,
)

PLANNER_NAME = "take_planner_v1"
PLAN_FILENAME = "take_plan.json"
BREAK_TYPES = ("none", "comma", "sentence", "paragraph")
LINE_GAP_SECONDS = 0.25
DEFAULT_WINDOW_SECONDS = 240.0
DEFAULT_LOOKAHEAD_SECONDS = 120.0
DEFAULT_PLANNER_MAX_OUTPUT_TOKENS = 65536
PREVIOUS_CONTEXT_ITEMS = 12
MAX_ATTEMPTS = 3
MAX_TRANSPORT_RETRIES = 6
MAX_REVIEW_REGIONS = 24
REVIEW_MAX_SPAN_SECONDS = 420.0
REVIEW_CONTEXT_SECONDS = 20.0
AUDIO_CLIP_SAMPLE_RATE = 16_000
AUDIO_CLIP_PEAK = 0.7
# A rate limit says when to come back: "Please retry in 25.5s" (Gemini) or
# "'retryDelay': '25s'".  The Gemini API meters Gemma models at 16,000 input
# tokens a minute, about two windows.
RETRY_AFTER_PATTERN = re.compile(
    r"retry in (\d+(?:\.\d+)?)\s*s|'retryDelay': '(\d+(?:\.\d+)?)s'", re.IGNORECASE
)
# Errors that a retry cannot fix.  Everything else -- dropped connections,
# timeouts, overload, 5xx -- is retried with backoff.
PERMANENT_ERROR_PATTERN = re.compile(
    r"(\b40[0134]\b|api[ _-]?key|permission|unauthenticated|invalid[ _]argument|"
    r"not[ _]found|unsupported)",
    re.IGNORECASE,
)

SYSTEM_INSTRUCTION = (
    "You are a meticulous professional dialogue editor for voice-over "
    "narration. Return only the JSON requested by the schema."
)


class TakePlanError(RuntimeError):
    """The planner could not produce a valid take selection."""


class WindowResponseError(ValueError):
    """A model response violates the window contract."""


@dataclass(frozen=True)
class TranscriptWord:
    """One chronological transcript word occurrence; ``id`` is its position."""

    id: int
    text: str
    start: float
    end: float


def load_transcript_words(transcript: dict[str, Any]) -> list[TranscriptWord]:
    """Flatten the transcript atoms (in atom order) into numbered words."""

    raw_words: list[dict[str, Any]] = []
    atoms = transcript.get("atoms")
    if isinstance(atoms, list):
        ordered = sorted(
            (atom for atom in atoms if isinstance(atom, dict)),
            key=lambda atom: int(atom.get("atom_index", -1)),
        )
        for atom in ordered:
            raw_words.extend(
                word for word in atom.get("words") or [] if isinstance(word, dict)
            )
    if not raw_words:
        whole = transcript.get("whole")
        if isinstance(whole, dict):
            raw_words.extend(
                word for word in whole.get("words") or [] if isinstance(word, dict)
            )
    words: list[TranscriptWord] = []
    for raw in raw_words:
        text = str(raw.get("word", raw.get("text", ""))).strip()
        if not text:
            continue
        if "start" not in raw or "end" not in raw:
            raise ValueError("every transcript word requires start and end")
        start, end = float(raw["start"]), float(raw["end"])
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError(f"invalid timestamp geometry for transcript word {text!r}")
        words.append(TranscriptWord(id=len(words), text=text, start=start, end=end))
    if not words:
        raise ValueError("the transcript contains no timed words")
    return words


@dataclass(frozen=True)
class Line:
    """Continuous speech between two transcript pauses."""

    index: int
    first_id: int
    last_id: int
    start: float
    end: float
    pause_before: float | None


@dataclass
class KeptItem:
    """One kept sentence or sentence piece: an inclusive source word range."""

    first_id: int
    last_id: int
    text: str
    break_after: str
    origin: str


def build_lines(
    words: Sequence[TranscriptWord],
    *,
    gap_seconds: float = LINE_GAP_SECONDS,
) -> list[Line]:
    """Split the chronological transcript at pauses longer than ``gap``."""

    lines: list[Line] = []
    group: list[TranscriptWord] = []

    def flush() -> None:
        if not group:
            return
        previous_end = lines[-1].end if lines else None
        lines.append(
            Line(
                index=len(lines),
                first_id=group[0].id,
                last_id=group[-1].id,
                start=group[0].start,
                end=max(word.end for word in group),
                pause_before=(
                    None
                    if previous_end is None
                    else max(0.0, group[0].start - previous_end)
                ),
            )
        )
        group.clear()

    for word in words:
        if group and word.start - max(item.end for item in group) >= gap_seconds:
            flush()
        group.append(word)
    flush()
    return lines


def _clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    return f"{minutes}:{seconds - 60 * minutes:04.1f}"


def _render_lines(
    lines: Sequence[Line],
    words: Sequence[TranscriptWord],
    *,
    origin_seconds: float,
    editable: tuple[int, int] | None = None,
) -> str:
    """Render lines as ``m:ss.s (pause) | [id]word ...`` for the prompt."""

    rendered = []
    for line in lines:
        tokens = []
        for word in words[line.first_id : line.last_id + 1]:
            if editable is None or editable[0] <= word.id <= editable[1]:
                tokens.append(f"[{word.id}]{word.text}")
            else:
                tokens.append(f"({word.text})")
        pause = "" if line.pause_before is None else f" (+{line.pause_before:.1f}s)"
        rendered.append(
            f"{_clock(line.start - origin_seconds)}{pause} | " + " ".join(tokens)
        )
    return "\n".join(rendered)


def _render_items(items: Sequence[KeptItem], *, label: str = "K") -> str:
    return "\n".join(
        f"{label}{number} [{item.first_id}-{item.last_id}] "
        f"{json.dumps(item.text, ensure_ascii=False)} (break: {item.break_after})"
        for number, item in enumerate(items, start=1)
    )


EDITING_RULES = """\
EDITING RULES

Your role: Act as an experienced human audio editor specialising in natural,
intelligent speech editing. Your task is to turn raw voice recordings into
a clean, coherent, natural-sounding narration while preserving the
speaker's original voice, meaning, personality, tone, and intended
message.
The speaker may be reading from a prepared script or speaking
spontaneously without one. Expect imperfect delivery, false starts,
repeated attempts, self-corrections, unfinished sentences, repeated
phrases, and multiple versions of the same idea, sometimes expressed
using slightly different wording. Treat the recording as an editing
process in which the speaker is often thinking aloud and refining what
they want to say.
Your primary objective is to identify and retain the best complete take
of each sentence, phrase, or idea, removing discarded attempts,
unnecessary repetitions, abandoned wording, recording instructions,
and other material that does not belong in the final narration.
When the speaker repeats the same sentence or part of a sentence
several times, the preferred take is generally the last reasonably
complete and successful attempt before the speaker moves on to a
different idea. However, do not apply this mechanically: evaluate each
attempt in context and select the version with the clearest delivery,
most complete wording, best flow, and most appropriate meaning.
A single sentence may contain several failed attempts, partial
repetitions, corrections, and overlapping versions of individual
phrases before the speaker finally delivers it correctly. Reconstruct
the sentence by retaining the best intended version of each part, but
only where the audio provides sufficient evidence to do so.
Distinguish genuine repetition and discarded delivery from intentional
rhetorical repetition, emphasis, or meaningful restatement. Do not
unnecessarily rewrite, paraphrase, improve, or rearrange the speaker's
language. Preserve spontaneous phrasing and natural speech wherever
possible.
Make the result sound as though the speaker delivered it fluently in
one natural performance, rather than as though separate fragments
were mechanically assembled. Prioritise meaning, continuity,
intelligibility, and natural rhythm over aggressive removal of every
imperfection or pause.
You are responsible for making editorial decisions as a careful human
editor would, not simply applying automatic silence detection or
repetitive-word removal. The following instructions will provide more
specific editing rules and priorities that you must also follow:

1. The final narration contains every intended sentence exactly once and
   nothing twice.
2. When a sentence or passage is attempted several times, keep the best
   complete take: normally the LAST complete and fluent one, because speakers
   retake after a mistake. A retake may come right after the attempt or after
   several other sentences (re-reading a whole paragraph), so compare every
   sentence with everything kept before it.
3. A restart may begin mid-sentence ("A training pipeline can... can keep
   every GPU busy"): keep the good beginning of the earlier attempt and the
   continuation from the later one, so no word is duplicated and the grammar
   is intact.
4. Remove stutters, repeated or abandoned words, false starts, filler
   (um, uh), and remarks the speaker makes to themself rather than to the
   audience ("okay", "let me try again", "one more time", "so we can do
   this"), which usually sound different from the narration. A kept range must
   not contain repeated or abandoned words: split the item around them.
5. Never drop narration that has no better take, even if it is slightly
   imperfect. Keep sentence-initial connectors and ordinals ("So", "But",
   "Third", "And fourth") that belong to the kept sentence, even when they were
   spoken as a separate fragment just before the take you keep.
6. Prefer long contiguous ranges and cut at pauses (line breaks) when that
   does not hurt correctness.
7. Items are chronological: every item starts after the previous item ends
   (strictly increasing word IDs).
8. Be extremely cautious when cutting audio at the beginning of a sentence, 
   paragraph, phrase, or retake. Before making any cut, check whether the first
   consonant of the incoming word is subject to regressive voicing assimilation, 
   particularly regressive devoicing. In languages such as Russian, a voiced 
   consonant may be pronounced as its voiceless counterpart because of a following
   voiceless consonant. For example, вход may be pronounced approximately as [фход].
   The initial [ф] is not a separate word, filler, noise, or unwanted audio. 
   Removing or trimming this initial devoiced consonant can change the perceived
   word entirely: removing [ф] could leave something sounding like ход, which is
   a different word with a different meaning. Never automatically trim the initial
   consonant of a sentence, paragraph, phrase, or retake without first checking its
   linguistic function. Do not mistake a devoiced or assimilated consonant for
   background noise, a mouth click, an isolated sound, a hesitation, or an unwanted
   fragment. Preserve the complete phonetic onset of every word, including consonants
   whose voicing has changed due to regressive assimilation. Pay particular attention
   to voiced/voiceless consonant pairs such as Б/П, В/Ф, Г/К, Д/Т, Ж/Ш, and З/С. When
   uncertain, consult the Whisper-generated subtitles or transcription to identify the
   intended word and its context, using the transcription as supporting evidence 
   rather than an infallible source. Perform a linguistic and contextual sense check
   by listening to the audio before and after the proposed cut and verifying that the
   complete intended word remains intelligible. If the boundary is ambiguous, preserve
   the consonant and a little extra audio rather than risk cutting off a meaningful
   sound. Do not use silence detection, automatic breath removal, noise reduction, or
   word-boundary detection in a way that removes or damages these initial consonants.
   Linguistic integrity must always take priority over aggressive trimming and
   perfectly tight edit boundaries. When editing sentence beginnings, paragraph
   beginnings, or retakes, always verify that the first audible consonant belongs to
   the intended word, even if it sounds different from its written form due to
   regressive devoicing. If there is any doubt, do not cut: preserve the original
   audio and flag the segment for manual review. The goal is to produce clean edits
   without changing, distorting, or inadvertently altering the meaning of spoken 
   words."""

OUTPUT_CONTRACT = """\
OUTPUT
"keep" lists the final narration as ordered items. Each item is one sentence,
or one piece of a sentence when you join pieces of different takes:
- first_id, last_id: inclusive range of source word IDs kept for this item
  (contiguous in the source; every word in between is kept).
- text: exactly those words as clean text with correct punctuation (fix ASR
  spelling).
- break_after: the pause the listener should hear after this item:
  "none" (the sentence continues directly in the next item), "comma" (short
  phrase pause), "sentence" (end of a sentence), "paragraph" (end of a
  paragraph, topic, or slide)."""


def _audio_instruction(clip_seconds: float | None) -> str:
    if clip_seconds is None:
        return ""
    return (
        "\nAUDIO: the attached audio is the original recording of exactly the "
        "transcript shown below; times in the transcript are positions in the "
        f"attached audio (it is {_clock(clip_seconds)} long). The transcript "
        "is machine ASR and can mis-hear words or hide repetitions: listen to "
        "resolve mis-heard words, to hear stumbles, hesitations, "
        "mispronunciations and restarts, and to choose the take that sounds "
        "most fluent and natural.\n"
    )


def window_prompt(
    *,
    zone_text: str,
    lookahead_text: str,
    previous: Sequence[KeptItem],
    final: bool,
    clip_seconds: float | None,
    language: str,
) -> str:
    language_note = (
        "The narration is in Russian; keep every text value in Russian."
        if language == "ru"
        else "The narration is in English."
    )
    previous_block = (
        "ALREADY IN THE EDIT (the most recent items decided earlier; read-only "
        "except for drop_previous):\n" + _render_items(previous) + "\n"
        "If the transcript below contains a better retake of one of these "
        "items, list its K number in drop_previous and keep the new take "
        "instead. Otherwise return an empty drop_previous.\n"
        if previous
        else "This is the beginning of the recording. drop_previous must be empty.\n"
    )
    if final:
        tail = (
            "This part reaches the end of the recording: decide everything and "
            "return resume_from_id = null."
        )
        lookahead_block = ""
    else:
        tail = (
            "resume_from_id: the ID of the first word that should be decided in "
            "the next step instead of now -- the start of the last sentence of "
            "TRANSCRIPT TO EDIT if it is unfinished, or if the look-ahead might "
            "still retake it. Do not keep words at or after resume_from_id. Use "
            "null only when everything in TRANSCRIPT TO EDIT is settled."
        )
        lookahead_block = (
            "\nLOOK-AHEAD (do NOT keep these words; they are decided in the next "
            "step. Use them only to see whether something above is retaken or "
            "continued later):\n" + lookahead_text + "\n"
        )
    return f"""You are cutting a raw voice-over recording into the final
narration, the way a professional human editor would.

The speaker reads prepared narration and often restarts: false starts,
stumbles, repeated words, re-reads of sentences or whole paragraphs,
corrections, abandoned fragments, filler, and remarks to themself that are not
part of the narration. {language_note}

INPUT: a machine (ASR) transcript split into lines; a new line starts after
every pause. Each line shows its time and the pause before it. Every word has
an ID: [123]word.
{_audio_instruction(clip_seconds)}
{OUTPUT_CONTRACT}
drop_previous: K numbers of earlier items to remove (see below).
{tail}

{EDITING_RULES}

{previous_block}
TRANSCRIPT TO EDIT:
{zone_text}
{lookahead_block}"""


def take_response_schema() -> dict[str, Any]:
    item = {
        "type": "object",
        "properties": {
            "first_id": {"type": "integer"},
            "last_id": {"type": "integer"},
            "text": {"type": "string"},
            "break_after": {"type": "string", "enum": list(BREAK_TYPES)},
        },
        "required": ["first_id", "last_id", "text", "break_after"],
    }
    return {
        "type": "object",
        "properties": {
            "notes": {
                "type": "string",
                "description": "One or two sentences on the retakes you resolved.",
            },
            "keep": {"type": "array", "items": item},
            "drop_previous": {"type": "array", "items": {"type": "integer"}},
            "resume_from_id": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        },
        "required": ["notes", "keep", "drop_previous", "resume_from_id"],
    }


@dataclass
class WindowDecision:
    items: list[KeptItem]
    drop_previous: list[int]
    resume_from_id: int | None
    warnings: list[str]


def _line_start_at_or_before(lines: Sequence[Line], word_id: int) -> int:
    """Return the first word ID of the line that contains ``word_id``."""

    for line in reversed(lines):
        if line.first_id <= word_id:
            return line.first_id
    return lines[0].first_id


def _source_text(words: Sequence[TranscriptWord] | None, first: int, last: int) -> str:
    if words is None:
        return ""
    return " ".join(word.text for word in words[first : last + 1])


def parse_window_response(
    raw: str,
    *,
    zone_first_id: int,
    zone_last_id: int,
    previous_count: int,
    final: bool,
    zone_lines: Sequence[Line],
    origin: str,
    words: Sequence[TranscriptWord] | None = None,
    strict: bool = True,
) -> WindowDecision:
    """Validate one window response, repairing what has one reading.

    Models below frontier size often get the geometry of an otherwise good
    answer wrong.  Repairs that cannot lose narration this window owns are
    always made: items are put in source order, an item overlapping the one
    before it starts after it, words before the editable zone are cut off,
    and an item reaching past the zone is left to the next window (cut at
    the end of the recording).  Repairs that guess, or drop something the
    model meant to keep -- a reversed range (swapped), an item entirely
    before the zone, a malformed item (both dropped) -- reject the answer
    while ``strict``, so the model gets another try; otherwise they are made
    too.  A changed item's text becomes its source words'.
    """

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise WindowResponseError(f"response is not JSON: {error}") from None
    if not isinstance(value, dict) or not isinstance(value.get("keep"), list):
        raise WindowResponseError("response must be an object with a keep list")
    warnings: list[str] = []

    def lossy(message: str) -> None:
        if strict:
            raise WindowResponseError(message)
        warnings.append(f"{message}; dropped")

    candidates: list[KeptItem] = []
    for index, raw_item in enumerate(value["keep"]):
        if not isinstance(raw_item, dict):
            lossy(f"keep[{index}] is not an object")
            continue
        first = raw_item.get("first_id")
        last = raw_item.get("last_id")
        if type(first) is not int or type(last) is not int:
            lossy(f"keep[{index}] IDs must be integers")
            continue
        text = str(raw_item.get("text", "")).strip()
        if first > last:
            message = f"keep[{index}] has first_id {first} after last_id {last}"
            if strict:
                raise WindowResponseError(message)
            warnings.append(f"{message}; swapped")
            first, last = last, first
            text = _source_text(words, first, last) or text
        if last < zone_first_id:
            lossy(
                f"keep[{index}] range {first}-{last} precedes the editable IDs "
                f"{zone_first_id}-{zone_last_id}"
            )
            continue
        break_after = raw_item.get("break_after")
        if break_after not in BREAK_TYPES:
            warnings.append(f"keep[{index}] break_after {break_after!r} -> sentence")
            break_after = "sentence"
        candidates.append(
            KeptItem(
                first_id=first,
                last_id=last,
                text=text,
                break_after=str(break_after),
                origin=origin,
            )
        )
    if [item.first_id for item in candidates] != sorted(
        item.first_id for item in candidates
    ):
        warnings.append("items were not in source order; sorted")
        candidates.sort(key=lambda item: (item.first_id, item.last_id))

    items: list[KeptItem] = []
    beyond_zone: list[KeptItem] = []
    for item in candidates:
        first, last = item.first_id, item.last_id
        if final and first > zone_last_id:
            warnings.append(f"item {first}-{last} lies after the editable IDs; dropped")
            continue
        if items and first <= items[-1].last_id:
            first = items[-1].last_id + 1
        first = max(first, zone_first_id)
        if final:
            last = min(last, zone_last_id)
        if first > last:
            warnings.append(
                f"item {item.first_id}-{item.last_id} repeats words kept "
                "before it; dropped"
            )
            continue
        if (first, last) != (item.first_id, item.last_id):
            warnings.append(
                f"item {item.first_id}-{item.last_id} -> {first}-{last} "
                "(overlap or IDs outside the editable zone)"
            )
            item = KeptItem(
                first_id=first,
                last_id=last,
                text=_source_text(words, first, last) or item.text,
                break_after=item.break_after,
                origin=item.origin,
            )
        if item.first_id > zone_last_id:
            beyond_zone.append(item)
            continue
        items.append(item)
    warnings.extend(
        f"item {item.first_id}-{item.last_id} lies in the look-ahead; dropped"
        for item in beyond_zone
    )

    drops: list[int] = []
    for number in value.get("drop_previous") or []:
        if type(number) is int and 1 <= number <= previous_count:
            drops.append(number)
        else:
            warnings.append(f"ignored invalid drop_previous entry {number!r}")

    resume = value.get("resume_from_id")
    if final:
        if resume is not None:
            warnings.append("final window returned resume_from_id; ignored")
        resume = None
    else:
        if resume is not None and type(resume) is not int:
            warnings.append(f"resume_from_id {resume!r} is not an integer; ignored")
            resume = None
        if resume is not None and resume < zone_first_id:
            warnings.append(f"resume_from_id {resume} precedes the zone; ignored")
            resume = None
        if resume is None or resume > zone_last_id:
            resume = zone_last_id + 1
        # A sentence that runs past the zone is decided with the look-ahead.
        for item in items:
            if item.last_id > zone_last_id:
                resume = min(resume, item.first_id)
        if resume <= zone_last_id:
            resume = _line_start_at_or_before(zone_lines, resume)
        # An item cannot be kept now and re-decided later.
        settled = list(items)
        deferred: list[KeptItem] = []
        while settled and settled[-1].last_id >= resume:
            straddling = settled.pop()
            deferred.append(straddling)
            if straddling.first_id < resume:
                resume = _line_start_at_or_before(zone_lines, straddling.first_id)
        if resume <= zone_first_id:
            # No progress would loop forever.  A one-line zone is settled as
            # answered; a longer zone must decide at least its first lines.
            last_line_start = zone_lines[-1].first_id
            if last_line_start > zone_first_id and strict:
                raise WindowResponseError(
                    "resume_from_id defers the whole window; decide at least "
                    f"everything before ID {last_line_start}"
                )
            if last_line_start > zone_first_id:
                warnings.append("the answer deferred the whole window; kept it all")
            resume = zone_last_id + 1
            items = [
                KeptItem(
                    first_id=item.first_id,
                    last_id=min(item.last_id, zone_last_id),
                    text=(
                        item.text
                        if item.last_id <= zone_last_id
                        else _source_text(words, item.first_id, zone_last_id)
                        or item.text
                    ),
                    break_after=item.break_after,
                    origin=item.origin,
                )
                for item in items
            ]
        else:
            items = settled
            warnings.extend(
                f"item {item.first_id}-{item.last_id} deferred to the next window"
                for item in deferred
            )
    return WindowDecision(
        items=items,
        drop_previous=sorted(set(drops)),
        resume_from_id=resume,
        warnings=warnings,
    )


def parse_review_response(
    raw: str,
    *,
    zone_first_id: int,
    zone_last_id: int,
    zone_lines: Sequence[Line],
    origin: str,
    words: Sequence[TranscriptWord] | None = None,
    strict: bool = True,
) -> WindowDecision:
    """A review re-decides a region that held kept narration: never empty."""

    decision = parse_window_response(
        raw,
        zone_first_id=zone_first_id,
        zone_last_id=zone_last_id,
        previous_count=0,
        final=True,
        zone_lines=zone_lines,
        origin=origin,
        words=words,
        strict=strict,
    )
    if not decision.items:
        raise WindowResponseError(
            "the region contains narration; keep at least its best take"
        )
    return decision


class AudioClipper:
    """Cut normalized low-bitrate listening copies of source spans."""

    def __init__(self, audio_path: Path) -> None:
        import soundfile as sf

        self.path = audio_path
        info = sf.info(str(audio_path))
        self.sample_rate = int(info.samplerate)
        self.frames = int(info.frames)

    def clip(self, start_seconds: float, end_seconds: float) -> tuple[MediaPart, float]:
        from math import gcd

        import soundfile as sf
        from scipy.signal import resample_poly

        start = max(0, int(start_seconds * self.sample_rate))
        end = min(self.frames, math.ceil(end_seconds * self.sample_rate))
        data, _ = sf.read(
            str(self.path), start=start, stop=end, dtype="float32", always_2d=True
        )
        mono = data.mean(axis=1)
        divisor = gcd(self.sample_rate, AUDIO_CLIP_SAMPLE_RATE)
        mono = resample_poly(
            mono,
            AUDIO_CLIP_SAMPLE_RATE // divisor,
            self.sample_rate // divisor,
        ).astype(np.float32)
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        if peak > 0.0:
            mono *= AUDIO_CLIP_PEAK / peak
        buffer = io.BytesIO()
        sf.write(
            buffer,
            mono,
            AUDIO_CLIP_SAMPLE_RATE,
            format="MP3",
            subtype="MPEG_LAYER_III",
        )
        return (
            MediaPart(data=buffer.getvalue(), mime_type="audio/mp3"),
            (end - start) / self.sample_rate,
        )


def _generate(
    backend: PlannerBackend,
    prompt: str,
    *,
    schema: dict[str, Any],
    request_id: str,
    media: Sequence[MediaPart],
) -> str:
    """Call the backend, retrying transient transport failures."""

    delay = 5.0
    for attempt in range(1, MAX_TRANSPORT_RETRIES + 1):
        try:
            if media:
                return backend.generate(  # type: ignore[call-arg]
                    prompt,
                    response_schema=schema,
                    request_id=request_id,
                    media=media,
                )
            return backend.generate(
                prompt, response_schema=schema, request_id=request_id
            )
        except Exception as error:  # provider SDKs raise many types
            if (
                attempt == MAX_TRANSPORT_RETRIES
                or isinstance(error, PermanentPlannerError)
                or PERMANENT_ERROR_PATTERN.search(f"{type(error).__name__}: {error}")
            ):
                raise
            wait = retry_after(error)
            wait = delay if wait is None else min(wait + 1.0, 120.0)
            print(
                f"planner transport error ({error}); retrying in {wait:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
            delay = min(delay * 2.5, 120.0)
    raise AssertionError("unreachable")


def retry_after(error: BaseException) -> float | None:
    """The wait a rate-limit error asks for, in seconds, if it names one."""

    match = RETRY_AFTER_PATTERN.search(str(error))
    if match is None:
        return None
    return float(match.group(1) or match.group(2))


def _retry_prompt(prompt: str, *, raw: str, error: str) -> str:
    return (
        f"{prompt}\n\nYOUR PREVIOUS ANSWER WAS REJECTED: {error}\n"
        f"Previous answer:\n{raw}\n"
        "Return a corrected answer for exactly the same transcript."
    )


def _items_text(items: Sequence[KeptItem]) -> str:
    return " ".join(item.text for item in items if item.text).strip()


def merged_source_ranges(items: Sequence[KeptItem]) -> list[dict[str, int]]:
    """Merge contiguous items into half-open source word ranges."""

    ranges: list[dict[str, int]] = []
    for item in items:
        if ranges and ranges[-1]["end_word_id"] == item.first_id:
            ranges[-1]["end_word_id"] = item.last_id + 1
        else:
            ranges.append(
                {"start_word_id": item.first_id, "end_word_id": item.last_id + 1}
            )
    return ranges


def _normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text.casefold())


def _immediate_repeat(tokens: Sequence[str]) -> int | None:
    """Length of a phrase repeated back to back ("that might be that might be")."""

    for size in range(min(6, len(tokens) // 2), 1, -1):
        for start in range(len(tokens) - 2 * size + 1):
            if tokens[start : start + size] == tokens[start + size : start + 2 * size]:
                return size
    return None


def find_duplicate_suspects(
    items: Sequence[KeptItem],
    words: Sequence[TranscriptWord],
    *,
    max_distance_seconds: float = 180.0,
) -> list[tuple[int, int, str]]:
    """Return item index pairs whose content looks repeated in the edit.

    The check is language-agnostic.  It compares whole sentences, because
    narration legitimately repeats short phrases ("For LocalSGD, ...") while a
    left-over retake repeats most of a sentence, and only within a few minutes,
    the distance at which speakers retake.
    """

    token_lists = [
        _normalized_tokens(
            " ".join(word.text for word in words[item.first_id : item.last_id + 1])
        )
        for item in items
    ]
    sentences: list[tuple[int, int, list[str]]] = []
    first = 0
    for index, item in enumerate(items):
        if item.break_after in ("sentence", "paragraph") or index == len(items) - 1:
            tokens = [
                token for k in range(first, index + 1) for token in token_lists[k]
            ]
            sentences.append((first, index, tokens))
            first = index + 1

    suspects: list[tuple[int, int, str]] = []
    for first_item, last_item, tokens in sentences:
        size = _immediate_repeat(tokens)
        if size is not None:
            suspects.append(
                (first_item, last_item, f"{size} words repeated back to back")
            )
    for left in range(len(sentences)):
        left_first, left_last, left_tokens = sentences[left]
        for right in range(left + 1, len(sentences)):
            right_first, right_last, right_tokens = sentences[right]
            gap = (
                words[items[right_first].first_id].start
                - words[items[left_last].last_id].end
            )
            if gap > max_distance_seconds:
                break
            shorter = min(len(left_tokens), len(right_tokens))
            if shorter < 4:
                continue
            matcher = SequenceMatcher(a=left_tokens, b=right_tokens, autojunk=False)
            run = matcher.find_longest_match(
                0, len(left_tokens), 0, len(right_tokens)
            ).size
            ratio = matcher.ratio()
            if ratio >= 0.7 or run >= max(5, 0.6 * shorter):
                suspects.append(
                    (
                        left_first,
                        right_last,
                        f"similar sentences (shared run {run}, ratio {ratio:.2f})",
                    )
                )
    # Words repeated across a join: "When that delay enters the | When that ..."
    for left in range(len(items) - 1):
        left_tokens, right_tokens = token_lists[left], token_lists[left + 1]
        for size in range(min(len(left_tokens), len(right_tokens), 5), 1, -1):
            repeated = left_tokens[-size:]
            if repeated == right_tokens[:size] and (
                size >= 3 or len("".join(repeated)) >= 8
            ):
                suspects.append((left, left + 1, f"{size} words repeat at a join"))
                break
    return suspects


def _review_regions(
    suspects: Sequence[tuple[int, int, str]],
    items: Sequence[KeptItem],
    words: Sequence[TranscriptWord],
) -> list[dict[str, Any]]:
    """Merge suspect pairs into bounded item regions."""

    spans = sorted((left, right, reason) for left, right, reason in suspects)
    regions: list[dict[str, Any]] = []
    for left, right, reason in spans:
        if regions and left <= regions[-1]["last_item"] + 1:
            region = regions[-1]
            candidate_last = max(region["last_item"], right)
            span = (
                words[items[candidate_last].last_id].end
                - words[items[region["first_item"]].first_id].start
            )
            if span <= REVIEW_MAX_SPAN_SECONDS:
                region["last_item"] = candidate_last
                region["reasons"].append(reason)
                continue
            # Regions must stay disjoint so they can be replaced in place.
            left = region["last_item"] + 1
            if left > right:
                continue
        regions.append({"first_item": left, "last_item": right, "reasons": [reason]})
    return regions[:MAX_REVIEW_REGIONS]


def review_prompt(
    *,
    region_text: str,
    current: Sequence[KeptItem],
    before: Sequence[KeptItem],
    after: Sequence[KeptItem],
    editable: tuple[int, int],
    reasons: Sequence[str],
    clip_seconds: float | None,
    language: str,
) -> str:
    language_note = (
        "The narration is in Russian; keep every text value in Russian."
        if language == "ru"
        else "The narration is in English."
    )
    return f"""You are reviewing one region of a voice-over edit made by an
editor who cut a raw recording into the final narration. An automatic check
flagged that the edit may still repeat content here
({"; ".join(reasons)}). {language_note}

Decide again which words of this region belong to the final narration. If two
kept items say the same thing, keep only the better take (normally the later
complete one) -- but if they are genuinely different sentences that merely
share words, keep both. Fix any remaining stutter, false start or repeated
words. Do not drop narration that has no other take.

INPUT: the raw ASR transcript of the region, split into lines at pauses. Words
with an ID ([123]word) are editable; words in parentheses belong to the
neighbouring edit and must not be used.
{_audio_instruction(clip_seconds)}
{OUTPUT_CONTRACT}
Use only IDs {editable[0]}-{editable[1]}. drop_previous must be empty and
resume_from_id must be null.

{EDITING_RULES}

EDIT BEFORE THIS REGION (fixed):
{_render_items(before, label="B") or "(start of recording)"}

CURRENT EDIT OF THIS REGION (to be replaced by your answer):
{_render_items(current, label="C")}

EDIT AFTER THIS REGION (fixed):
{_render_items(after, label="A") or "(end of recording)"}

RAW TRANSCRIPT OF THE REGION:
{region_text}
"""


class TakePlanner:
    """Window planning, revocation and duplicate review over one transcript."""

    def __init__(
        self,
        *,
        words: Sequence[TranscriptWord],
        backend: PlannerBackend,
        output_dir: Path,
        audio: AudioClipper | None,
        language: str,
        window_seconds: float,
        lookahead_seconds: float,
    ) -> None:
        self.words = list(words)
        self.lines = build_lines(self.words)
        self.backend = backend
        self.output_dir = output_dir
        self.audio = audio
        self.language = language
        self.window_seconds = window_seconds
        self.lookahead_seconds = lookahead_seconds
        self.schema = take_response_schema()
        self.ledger: list[dict[str, Any]] = []
        self.calls_dir = output_dir / "calls"
        self.calls_dir.mkdir(parents=True, exist_ok=True)

    # -- request helpers -------------------------------------------------

    def _cached_call(
        self,
        name: str,
        prompt: str,
        media: Sequence[MediaPart],
    ) -> str:
        """Return a cached raw response for an identical request."""

        digest = hashlib.sha256()
        digest.update(self.backend.backend_name.encode())
        digest.update(self.backend.model.encode())
        digest.update(prompt.encode())
        for part in media:
            digest.update(hashlib.sha256(part.data).digest())
        key = digest.hexdigest()
        path = self.calls_dir / f"{name}.json"
        cached = read_json(path) if path.is_file() else None
        if isinstance(cached, dict) and cached.get("request_sha256") == key:
            return str(cached["raw"])
        started = time.monotonic()
        raw = _generate(
            self.backend,
            prompt,
            schema=self.schema,
            request_id=name,
            media=media,
        )
        write_json(
            path,
            {
                "request_sha256": key,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "prompt": prompt,
                "media_bytes": [len(part.data) for part in media],
                "raw": raw,
            },
        )
        return raw

    def _decide(
        self,
        name: str,
        prompt: str,
        media: Sequence[MediaPart],
        parse: Any,
    ) -> tuple[WindowDecision, list[dict[str, Any]]]:
        attempts: list[dict[str, Any]] = []
        request = prompt
        for attempt in range(1, MAX_ATTEMPTS + 1):
            raw = self._cached_call(f"{name}_a{attempt}", request, media)
            try:
                # The last try keeps what it can rather than end the plan.
                decision = parse(raw, strict=attempt < MAX_ATTEMPTS)
            except WindowResponseError as error:
                attempts.append({"attempt": attempt, "error": str(error)})
                print(f"{name}: rejected response ({error})", file=sys.stderr)
                request = _retry_prompt(prompt, raw=raw, error=str(error))
                continue
            attempts.append({"attempt": attempt, "accepted": True})
            return decision, attempts
        raise TakePlanError(
            f"{name}: no valid response after {MAX_ATTEMPTS} attempts: "
            f"{attempts[-1]['error']}"
        )

    def _media(self, start: float, end: float) -> tuple[list[MediaPart], float | None]:
        if self.audio is None:
            return [], None
        part, seconds = self.audio.clip(start, end)
        return [part], seconds

    # -- window planning -------------------------------------------------

    def _zone_end_line(self, start_line: int) -> int:
        lines = self.lines
        origin = lines[start_line].start
        target = start_line + 1
        while (
            target < len(lines) and lines[target].start - origin < self.window_seconds
        ):
            target += 1
        if target >= len(lines):
            return len(lines)
        # End the zone at the longest pause near the nominal size: a likely
        # thought boundary.
        lower = origin + 0.8 * self.window_seconds
        upper = origin + 1.2 * self.window_seconds
        candidates = [
            index
            for index in range(start_line + 1, len(lines))
            if lower <= lines[index].start <= upper
        ]
        if not candidates:
            return target
        return max(candidates, key=lambda index: lines[index].pause_before or 0.0)

    def _lookahead_end_line(self, zone_end: int) -> int:
        lines = self.lines
        if zone_end >= len(lines):
            return len(lines)
        origin = lines[zone_end].start
        end = zone_end
        while end < len(lines) and lines[end].start - origin < self.lookahead_seconds:
            end += 1
        return end

    def plan_windows(self) -> list[KeptItem]:
        items: list[KeptItem] = []
        line_by_first_id = {line.first_id: line.index for line in self.lines}
        start_line = 0
        window = 0
        while start_line < len(self.lines):
            window += 1
            zone_end = self._zone_end_line(start_line)
            look_end = self._lookahead_end_line(zone_end)
            final = zone_end >= len(self.lines)
            zone_lines = self.lines[start_line:zone_end]
            look_lines = self.lines[zone_end:look_end]
            clip_start = max(0.0, zone_lines[0].start - 0.3)
            clip_end = (look_lines or zone_lines)[-1].end + 0.3
            media, clip_seconds = self._media(clip_start, clip_end)
            origin = clip_start
            previous = items[-PREVIOUS_CONTEXT_ITEMS:]
            prompt = window_prompt(
                zone_text=_render_lines(zone_lines, self.words, origin_seconds=origin),
                lookahead_text=_render_lines(
                    look_lines, self.words, origin_seconds=origin
                ),
                previous=previous,
                final=final,
                clip_seconds=clip_seconds,
                language=self.language,
            )
            zone_first = zone_lines[0].first_id
            zone_last = zone_lines[-1].last_id
            name = f"window_{window:03d}"
            print(
                f"{name}: {_clock(zone_lines[0].start)}-{_clock(zone_lines[-1].end)} "
                f"({len(zone_lines)} lines, look-ahead {len(look_lines)})",
                flush=True,
            )
            decision, attempts = self._decide(
                name,
                prompt,
                media,
                partial(
                    parse_window_response,
                    zone_first_id=zone_first,
                    zone_last_id=zone_last,
                    previous_count=len(previous),
                    final=final,
                    zone_lines=zone_lines,
                    origin=name,
                    words=self.words,
                ),
            )
            dropped = [previous[number - 1] for number in decision.drop_previous]
            if dropped:
                drop_keys = {(item.first_id, item.last_id) for item in dropped}
                items = [
                    item
                    for item in items
                    if (item.first_id, item.last_id) not in drop_keys
                ]
            items.extend(decision.items)
            self.ledger.append(
                {
                    "name": name,
                    "zone_word_ids": [zone_first, zone_last],
                    "zone_seconds": [zone_lines[0].start, zone_lines[-1].end],
                    "lookahead_lines": len(look_lines),
                    "audio_seconds": clip_seconds,
                    "attempts": attempts,
                    "kept": [asdict(item) for item in decision.items],
                    "dropped_previous": [asdict(item) for item in dropped],
                    "resume_from_id": decision.resume_from_id,
                    "warnings": decision.warnings,
                }
            )
            for item in decision.items:
                print(f"  keep [{item.first_id}-{item.last_id}] {item.text}")
            for item in dropped:
                print(f"  revoke [{item.first_id}-{item.last_id}] {item.text}")
            for warning in decision.warnings:
                print(f"  note: {warning}")
            # Windows are nine tenths of the work, the duplicate review the rest.
            progress.report(
                round(900 * zone_lines[-1].end / max(self.words[-1].end, 1e-6)), 1000
            )
            if final or decision.resume_from_id is None:
                break
            next_line = line_by_first_id.get(decision.resume_from_id)
            if next_line is None:
                next_line = zone_end
            start_line = max(next_line, start_line + 1)
        return items

    # -- duplicate review ------------------------------------------------

    def review(self, items: list[KeptItem]) -> list[KeptItem]:
        suspects = find_duplicate_suspects(items, self.words)
        regions = _review_regions(suspects, items, self.words)
        review_ledger: list[dict[str, Any]] = []
        # Re-decide later regions first so earlier item indices stay valid.
        for done, (number, region) in enumerate(
            reversed(list(enumerate(regions, start=1)))
        ):
            progress.report(900 + round(100 * done / len(regions)), 1000)
            first_item, last_item = region["first_item"], region["last_item"]
            current = items[first_item : last_item + 1]
            editable = (current[0].first_id, current[-1].last_id)
            before = items[max(0, first_item - 6) : first_item]
            after = items[last_item + 1 : last_item + 4]
            region_lines = [
                line
                for line in self.lines
                if line.last_id >= editable[0] and line.first_id <= editable[1]
            ]
            context_start = self.words[editable[0]].start - REVIEW_CONTEXT_SECONDS
            context_end = self.words[editable[1]].end + REVIEW_CONTEXT_SECONDS
            context_lines = [
                line
                for line in self.lines
                if line.end >= context_start and line.start <= context_end
            ]
            shown = sorted(
                {line.index: line for line in [*context_lines, *region_lines]}.values(),
                key=lambda line: line.index,
            )
            clip_start = max(0.0, shown[0].start - 0.3)
            media, clip_seconds = self._media(clip_start, shown[-1].end + 0.3)
            prompt = review_prompt(
                region_text=_render_lines(
                    shown, self.words, origin_seconds=clip_start, editable=editable
                ),
                current=current,
                before=before,
                after=after,
                editable=editable,
                reasons=region["reasons"],
                clip_seconds=clip_seconds,
                language=self.language,
            )
            name = f"review_{number:03d}"
            print(
                f"{name}: items {first_item}-{last_item} "
                f"({'; '.join(region['reasons'])})",
                flush=True,
            )
            try:
                decision, attempts = self._decide(
                    name,
                    prompt,
                    media,
                    partial(
                        parse_review_response,
                        zone_first_id=editable[0],
                        zone_last_id=editable[1],
                        zone_lines=region_lines,
                        origin=name,
                        words=self.words,
                    ),
                )
            except TakePlanError as error:
                # The windows decided this region already; a review that
                # cannot answer leaves it as it is.
                print(f"  {error}; the region stays as it was", file=sys.stderr)
                decision = WindowDecision(
                    items=list(current),
                    drop_previous=[],
                    resume_from_id=None,
                    warnings=[f"review failed, region unchanged: {error}"],
                )
                attempts = [{"error": str(error)}]
            items[first_item : last_item + 1] = decision.items
            review_ledger.append(
                {
                    "name": name,
                    "reasons": region["reasons"],
                    "editable_word_ids": list(editable),
                    "before": [asdict(item) for item in current],
                    "after": [asdict(item) for item in decision.items],
                    "attempts": attempts,
                    "warnings": decision.warnings,
                }
            )
            if [asdict(item) for item in current] != [
                asdict(item) for item in decision.items
            ]:
                print("  before: " + _items_text(current))
                print("  after:  " + _items_text(decision.items))
        self.review_ledger = list(reversed(review_ledger))
        self.remaining_suspects = [
            {"items": [left, right], "reason": reason}
            for left, right, reason in find_duplicate_suspects(items, self.words)
        ]
        return items


def run_take_planner(
    *,
    transcript_path: Path,
    output_dir: Path,
    backend: PlannerBackend,
    audio_path: Path | None,
    language: str = "en",
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    lookahead_seconds: float = DEFAULT_LOOKAHEAD_SECONDS,
    review: bool = True,
) -> dict[str, Any]:
    """Plan the edit and write ``take_plan.json`` into ``output_dir``."""

    if window_seconds <= 0 or lookahead_seconds < 0:
        raise ValueError("window must be positive and look-ahead non-negative")
    transcript_path = transcript_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript = read_json(transcript_path)
    if not isinstance(transcript, dict):
        raise TypeError("transcript root must be an object")
    words = load_transcript_words(transcript)
    audio = None
    if audio_path is not None:
        if not getattr(backend, "supports_audio", False):
            raise ValueError(f"{backend.backend_name} cannot listen to audio")
        audio = AudioClipper(audio_path.resolve())
    planner = TakePlanner(
        words=words,
        backend=backend,
        output_dir=output_dir,
        audio=audio,
        language=language,
        window_seconds=window_seconds,
        lookahead_seconds=lookahead_seconds,
    )
    items = planner.plan_windows()
    planner.review_ledger = []
    planner.remaining_suspects = []
    if review and items:
        items = planner.review(items)
    if not items:
        raise TakePlanError("the planner kept no narration")
    plan = {
        "schema_version": 1,
        "planner": PLANNER_NAME,
        "status": "complete",
        "backend": backend.backend_name,
        "model": backend.model,
        "language": language,
        "transcript": str(transcript_path),
        "transcript_sha256": sha256_file(transcript_path),
        "audio": str(audio_path.resolve()) if audio_path is not None else None,
        "listened_to_audio": audio is not None,
        "window_seconds": window_seconds,
        "lookahead_seconds": lookahead_seconds,
        "line_gap_seconds": LINE_GAP_SECONDS,
        "word_count": len(words),
        "words": [asdict(word) for word in words],
        "items": [asdict(item) for item in items],
        "selected_source_ranges": merged_source_ranges(items),
        "reconstructed_narration": _items_text(items),
        "windows": planner.ledger,
        "review": planner.review_ledger,
        "remaining_duplicate_suspects": planner.remaining_suspects,
    }
    write_json(output_dir / PLAN_FILENAME, plan)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m voicecut.take_planner",
        description="Choose the final takes of a narration transcript.",
    )
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--audio",
        type=Path,
        help="Canonical source WAV; attached to requests when the model listens.",
    )
    parser.add_argument("--language", choices=("en", "ru"), default="en")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument(
        "--lookahead-seconds", type=float, default=DEFAULT_LOOKAHEAD_SECONDS
    )
    parser.add_argument("--no-review", action="store_true")
    parser.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_PLANNER_MAX_OUTPUT_TOKENS
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    add_planner_backend_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    backend = create_planner_backend(
        provider=args.planner_backend,
        model=args.planner_model,
        env_file=args.env_file.resolve(),
        max_output_tokens=args.max_output_tokens,
        system_instruction=SYSTEM_INSTRUCTION,
        base_url=args.planner_base_url,
        api_key_env=args.planner_api_key_env,
        think=think_setting(args.planner_think),
        gemini_temperature=None,
    )
    try:
        plan = run_take_planner(
            transcript_path=args.transcript,
            output_dir=args.output_dir,
            backend=backend,
            audio_path=args.audio,
            language=args.language,
            window_seconds=args.window_seconds,
            lookahead_seconds=args.lookahead_seconds,
            review=not args.no_review,
        )
    finally:
        backend.close()
    print(
        json.dumps(
            {
                "status": plan["status"],
                "backend": plan["backend"],
                "model": plan["model"],
                "listened_to_audio": plan["listened_to_audio"],
                "items": len(plan["items"]),
                "plan": str(args.output_dir.resolve() / PLAN_FILENAME),
                "reconstructed_narration": plan["reconstructed_narration"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
