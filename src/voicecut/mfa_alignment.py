"""Batched Montreal Forced Aligner adapter for production boundary evidence.

This module is intentionally independent from semantic planning and rendering.
It prepares local crops from the canonical source WAV, invokes the pinned MFA
CLI once for the complete batch, and converts MFA word/phone tiers to absolute
source coordinates.  It does not choose cuts and it never falls back to
Whisper timestamps.

MFA is accessed only through its documented command-line interface.  No MFA
Python package is imported here, which keeps VoiceCut's Python environment
isolated from MFA's Conda/Kaldi runtime.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import soundfile as sf

from .common import read_json, sha256_file, write_json

MFA_VERSION = "3.4.1"
MFA_MODEL_ID = "english_us_arpa"
MFA_SPEAKER = "narrator"
MFA_PHONE_ROUNDING_TOLERANCE_SECONDS = 0.002
MFA_TIME_BOUNDARY_TOLERANCE_SECONDS = 0.002
MFA_INTERVAL_ORDER_TOLERANCE_SECONDS = 0.000001
MFA_CONTEXT_RECOVERY_ATTEMPTS = 1

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MFA_PREFIX = REPOSITORY_ROOT / ".mfa-env"
DEFAULT_MFA_CACHE_ROOT = REPOSITORY_ROOT / ".voicecut-cache" / "runtime" / "mfa"

_VERSION_RE = re.compile(r"(?<![\d.])(\d+\.\d+\.\d+)(?![\d.])")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
_SILENCE_PHONES = frozenset({"", "<eps>", "sil", "silence", "sp"})
_SILENCE_WORD_LABELS = frozenset({"", "<eps>"})


class MFAAlignmentError(RuntimeError):
    """MFA could not produce safe and unambiguous alignment evidence."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class MFASourceWord:
    """One chronological source occurrence used to build an MFA context."""

    word_id: int
    text: str
    start_seconds: float
    end_seconds: float
    selected: bool


@dataclass(frozen=True)
class MFATokenMapping:
    """A reversible mapping from one MFA token to source word occurrences."""

    token: str
    source_word_ids: tuple[int, ...]
    source_text: str


@dataclass(frozen=True)
class MFAContextSpec:
    """A prepared local context request around one or more physical cuts."""

    context_id: str
    crop_source_start_seconds: float
    crop_source_end_seconds: float
    words: tuple[MFASourceWord, ...]
    boundary_ids: tuple[str, ...]
    token_mappings: tuple[MFATokenMapping, ...] | None = None


@dataclass(frozen=True)
class MFABatchPaths:
    """Filesystem layout for one batched MFA render attempt."""

    root: Path
    corpus: Path
    metadata: Path
    output: Path
    temporary: Path
    speaker: str = MFA_SPEAKER

    @property
    def speaker_corpus(self) -> Path:
        return self.corpus / self.speaker

    @property
    def speaker_output(self) -> Path:
        return self.output / self.speaker


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _finite_number(value: Any, *, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise MFAAlignmentError(
            "mfa_word_mapping_failed", f"{field} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(converted):
        raise MFAAlignmentError(
            "mfa_word_mapping_failed", f"{field} is not finite: {value!r}"
        )
    return converted


def seconds_to_sample(
    seconds: float,
    sample_rate: int,
    *,
    boundary: str = "nearest",
) -> int:
    """Convert a timestamp to a sample with explicit boundary rounding.

    ``floor`` is used for protected speech starts and ``ceil`` for protected
    speech ends, ensuring float conversion cannot move a cut into a phone.
    ``nearest`` remains useful for non-authoritative crop anchors.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("seconds must be finite and non-negative")
    position = seconds * sample_rate
    if boundary == "floor":
        return math.floor(position + 1e-9)
    if boundary == "ceil":
        return math.ceil(position - 1e-9)
    if boundary == "nearest":
        return round(position)
    raise ValueError("boundary must be 'floor', 'ceil', or 'nearest'")


def is_mfa_silence_phone(phone: str) -> bool:
    """Return whether MFA's phone label is a recognized non-speech symbol.

    The check is deliberately conservative.  In particular, ``spn`` is an
    unknown *spoken/noise* phone and is not considered verified silence.
    Position-dependent suffixes, when present, do not change the base phone.
    """

    normalized = str(phone).strip().casefold()
    if normalized in _SILENCE_PHONES:
        return True
    base = normalized.rsplit("_", 1)[0]
    return base in _SILENCE_PHONES


def _is_mfa_silence_word(word: str) -> bool:
    # MFA uses ``<eps>`` for non-lexical intervals in the word tier.  Phone
    # labels such as ``sil`` and ``silence`` are deliberately *not* reused
    # here: ``silence`` can be an ordinary spoken source token.
    return str(word).strip().casefold() in _SILENCE_WORD_LABELS


def _normalize_mfa_token_piece(value: str) -> list[str]:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    )
    normalized = (
        normalized.casefold()
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u02bc", "'")
        .replace("%", " percent ")
        .replace("&", " and ")
    )
    return _TOKEN_RE.findall(normalized)


def normalize_mfa_token(value: str) -> tuple[str, ...]:
    """Normalize source text to deterministic whitespace-delimited MFA tokens.

    Surrounding punctuation and Whisper punctuation artifacts are removed,
    curly apostrophes are normalized, and ordinary contractions remain one
    token.  Hyphenated forms and source entries containing whitespace can
    deterministically produce multiple MFA tokens.
    """

    return tuple(_normalize_mfa_token_piece(str(value)))


def normalize_source_words(
    words: Sequence[MFASourceWord | Mapping[str, Any]],
) -> tuple[tuple[MFATokenMapping, ...], tuple[int, ...]]:
    """Return ordered reversible mappings and explicit non-lexical word IDs.

    An alphabetic, numeric, or contraction-like source item is never silently
    dropped: inability to normalize it is a mapping failure.  Pure punctuation
    entries are recorded in the returned ``nonlexical`` tuple rather than
    disappearing without trace.
    """

    coerced = tuple(_coerce_source_word(word) for word in words)
    mappings: list[MFATokenMapping] = []
    nonlexical: list[int] = []
    for word in coerced:
        tokens = normalize_mfa_token(word.text)
        if not tokens:
            if any(character.isalnum() for character in word.text):
                raise MFAAlignmentError(
                    "mfa_word_mapping_failed",
                    f"source word {word.word_id} cannot be normalized: {word.text!r}",
                )
            nonlexical.append(word.word_id)
            continue
        mappings.extend(
            MFATokenMapping(
                token=token,
                source_word_ids=(word.word_id,),
                source_text=word.text,
            )
            for token in tokens
        )
    if not mappings:
        raise MFAAlignmentError(
            "mfa_word_mapping_failed", "alignment context has no lexical MFA tokens"
        )
    return tuple(mappings), tuple(nonlexical)


def _coerce_source_word(value: MFASourceWord | Mapping[str, Any]) -> MFASourceWord:
    if isinstance(value, MFASourceWord):
        result = value
    elif isinstance(value, Mapping):
        word_id = value.get("word_id", value.get("id"))
        result = MFASourceWord(
            word_id=int(word_id),
            text=str(value.get("text", "")),
            start_seconds=_finite_number(
                value.get("start_seconds", value.get("start")),
                field="source word start",
            ),
            end_seconds=_finite_number(
                value.get("end_seconds", value.get("end")),
                field="source word end",
            ),
            selected=bool(value.get("selected", False)),
        )
    else:
        raise TypeError(f"unsupported source word value: {type(value).__name__}")
    if result.word_id < 0:
        raise MFAAlignmentError(
            "mfa_word_mapping_failed", "source word IDs must be non-negative"
        )
    if not result.text.strip():
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"source word {result.word_id} has empty text",
        )
    # Whisper timestamps are approximate crop anchors, not alignment
    # geometry. Long-form decoding can legitimately assign several sequential
    # tokens the same zero-duration anchor. MFA maps those tokens by ordered
    # occurrence and supplies their authoritative word/phone intervals.
    # Preserve touching anchors verbatim; reject only negative or reversed
    # values rather than inventing duration by clamping timestamps.
    if result.start_seconds < 0.0 or result.end_seconds < result.start_seconds:
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"source word {result.word_id} has invalid anchor timestamps",
        )
    return result


def _coerce_token_mapping(
    value: MFATokenMapping | Mapping[str, Any],
) -> MFATokenMapping:
    if isinstance(value, MFATokenMapping):
        result = value
    elif isinstance(value, Mapping):
        source_ids = value.get("source_word_ids")
        if not isinstance(source_ids, Sequence) or isinstance(source_ids, str):
            raise MFAAlignmentError(
                "mfa_word_mapping_failed",
                "explicit MFA token mapping requires source_word_ids",
            )
        result = MFATokenMapping(
            token=str(value.get("token", "")),
            source_word_ids=tuple(int(word_id) for word_id in source_ids),
            source_text=str(value.get("source_text", "")),
        )
    else:
        raise TypeError(f"unsupported token mapping: {type(value).__name__}")
    normalized = normalize_mfa_token(result.token)
    if normalized != (result.token,):
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"explicit MFA token is not normalized: {result.token!r}",
        )
    if not result.source_word_ids or len(set(result.source_word_ids)) != len(
        result.source_word_ids
    ):
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"invalid source IDs for MFA token {result.token!r}",
        )
    return result


def _coerce_context(value: MFAContextSpec | Mapping[str, Any]) -> MFAContextSpec:
    if isinstance(value, MFAContextSpec):
        result = MFAContextSpec(
            context_id=value.context_id,
            crop_source_start_seconds=value.crop_source_start_seconds,
            crop_source_end_seconds=value.crop_source_end_seconds,
            words=tuple(_coerce_source_word(word) for word in value.words),
            boundary_ids=tuple(str(item) for item in value.boundary_ids),
            token_mappings=(
                tuple(_coerce_token_mapping(item) for item in value.token_mappings)
                if value.token_mappings is not None
                else None
            ),
        )
    elif isinstance(value, Mapping):
        raw_words = value.get("words")
        if not isinstance(raw_words, Sequence) or isinstance(raw_words, str):
            raise MFAAlignmentError(
                "mfa_word_mapping_failed", "context words must be an ordered list"
            )
        raw_mappings = value.get("token_mappings")
        mappings = (
            tuple(_coerce_token_mapping(item) for item in raw_mappings)
            if isinstance(raw_mappings, Sequence) and not isinstance(raw_mappings, str)
            else None
        )
        raw_boundary_ids = value.get("boundary_ids", ())
        if not isinstance(raw_boundary_ids, Sequence) or isinstance(
            raw_boundary_ids, str
        ):
            raise MFAAlignmentError(
                "mfa_word_mapping_failed",
                "context boundary_ids must be an ordered list",
            )
        result = MFAContextSpec(
            context_id=str(value.get("context_id", "")),
            crop_source_start_seconds=_finite_number(
                value.get("crop_source_start_seconds"), field="crop source start"
            ),
            crop_source_end_seconds=_finite_number(
                value.get("crop_source_end_seconds"), field="crop source end"
            ),
            words=tuple(_coerce_source_word(word) for word in raw_words),
            boundary_ids=tuple(str(item) for item in raw_boundary_ids),
            token_mappings=mappings,
        )
    else:
        raise TypeError(f"unsupported MFA context: {type(value).__name__}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", result.context_id):
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"unsafe or empty context ID: {result.context_id!r}",
        )
    if (
        result.crop_source_start_seconds < 0.0
        or result.crop_source_end_seconds <= result.crop_source_start_seconds
    ):
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"context {result.context_id} has invalid crop timestamps",
        )
    if not result.words:
        raise MFAAlignmentError(
            "mfa_word_mapping_failed", f"context {result.context_id} has no words"
        )
    word_ids = [word.word_id for word in result.words]
    if len(set(word_ids)) != len(word_ids) or word_ids != sorted(word_ids):
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"context {result.context_id} source words are not uniquely ordered",
        )
    return result


def _validated_context_mappings(
    context: MFAContextSpec,
) -> tuple[tuple[MFATokenMapping, ...], tuple[int, ...]]:
    if context.token_mappings is None:
        return normalize_source_words(context.words)
    mappings = tuple(_coerce_token_mapping(item) for item in context.token_mappings)
    known_ids = {word.word_id for word in context.words}
    mapped_ids: set[int] = set()
    for mapping in mappings:
        unknown = set(mapping.source_word_ids) - known_ids
        if unknown:
            raise MFAAlignmentError(
                "mfa_word_mapping_failed",
                f"context {context.context_id} token {mapping.token!r} maps unknown "
                f"source IDs {sorted(unknown)}",
            )
        mapped_ids.update(mapping.source_word_ids)
    nonlexical = tuple(
        word.word_id
        for word in context.words
        if word.word_id not in mapped_ids
        and not any(character.isalnum() for character in word.text)
    )
    missing = known_ids - mapped_ids - set(nonlexical)
    if missing:
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"context {context.context_id} has unmapped lexical source IDs "
            f"{sorted(missing)}",
        )
    if not mappings:
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"context {context.context_id} has no MFA tokens",
        )
    return mappings, nonlexical


def _reset_batch_directories(paths: MFABatchPaths) -> None:
    for directory in (paths.corpus, paths.metadata, paths.output, paths.temporary):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    paths.speaker_corpus.mkdir(parents=True, exist_ok=True)


def prepare_mfa_batch(
    *,
    audio_path: Path,
    contexts: Sequence[MFAContextSpec | Mapping[str, Any]],
    work_dir: Path,
) -> MFABatchPaths:
    """Create one MFA corpus containing every context for a render attempt."""

    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    coerced = tuple(_coerce_context(context) for context in contexts)
    if not coerced:
        raise MFAAlignmentError(
            "mfa_word_mapping_failed", "MFA batch requires at least one context"
        )
    context_ids = [context.context_id for context in coerced]
    if len(set(context_ids)) != len(context_ids):
        raise MFAAlignmentError(
            "mfa_word_mapping_failed", "MFA context IDs must be unique"
        )

    paths = MFABatchPaths(
        root=(work_dir.resolve() / "mfa_alignment"),
        corpus=(work_dir.resolve() / "mfa_alignment" / "corpus"),
        metadata=(work_dir.resolve() / "mfa_alignment" / "metadata"),
        output=(work_dir.resolve() / "mfa_alignment" / "output"),
        temporary=(work_dir.resolve() / "mfa_alignment" / "temp"),
    )
    _reset_batch_directories(paths)

    with sf.SoundFile(audio_path) as source:
        sample_rate = int(source.samplerate)
        channel_count = int(source.channels)
        total_samples = int(source.frames)
        if sample_rate <= 0 or channel_count <= 0 or total_samples <= 0:
            raise MFAAlignmentError(
                "mfa_word_mapping_failed", "canonical source WAV is empty or invalid"
            )
        audio_duration = total_samples / sample_rate

        batch_metadata: list[dict[str, Any]] = []
        for context in coerced:
            mappings, nonlexical = _validated_context_mappings(context)
            crop_start_sample = max(
                0,
                min(
                    total_samples,
                    seconds_to_sample(context.crop_source_start_seconds, sample_rate),
                ),
            )
            crop_end_sample = max(
                crop_start_sample,
                min(
                    total_samples,
                    seconds_to_sample(context.crop_source_end_seconds, sample_rate),
                ),
            )
            if crop_end_sample <= crop_start_sample:
                raise MFAAlignmentError(
                    "mfa_word_mapping_failed",
                    f"context {context.context_id} resolves to an empty crop",
                )
            actual_start = crop_start_sample / sample_rate
            actual_end = crop_end_sample / sample_rate
            if actual_start >= audio_duration:
                raise MFAAlignmentError(
                    "mfa_word_mapping_failed",
                    f"context {context.context_id} begins after source EOF",
                )

            source.seek(crop_start_sample)
            crop = source.read(
                crop_end_sample - crop_start_sample,
                dtype="float32",
                always_2d=True,
            )
            wav_path = paths.speaker_corpus / f"{context.context_id}.wav"
            lab_path = paths.speaker_corpus / f"{context.context_id}.lab"
            sf.write(wav_path, crop, sample_rate, format="WAV", subtype="PCM_16")
            lab_path.write_text(
                " ".join(mapping.token for mapping in mappings) + "\n",
                encoding="utf-8",
            )

            metadata = {
                "schema_version": 1,
                "context_id": context.context_id,
                "crop_source_start_seconds": actual_start,
                "crop_source_end_seconds": actual_end,
                "crop_source_start_sample": crop_start_sample,
                "crop_source_end_sample": crop_end_sample,
                "sample_rate": sample_rate,
                "channel_count": channel_count,
                "ordered_source_word_ids": [word.word_id for word in context.words],
                "original_source_words": [asdict(word) for word in context.words],
                "normalized_mfa_tokens": [mapping.token for mapping in mappings],
                "token_mappings": [asdict(mapping) for mapping in mappings],
                "nonlexical_source_word_ids": list(nonlexical),
                "boundary_ids": list(context.boundary_ids),
                "corpus_wav": str(wav_path),
                "corpus_lab": str(lab_path),
            }
            write_json(paths.metadata / f"{context.context_id}.json", metadata)
            batch_metadata.append(metadata)

    write_json(
        paths.metadata / "batch.json",
        {
            "schema_version": 1,
            "backend": "mfa",
            "mfa_version": MFA_VERSION,
            "model_id": MFA_MODEL_ID,
            "fine_tune": True,
            "source_audio": str(audio_path),
            "source_audio_sha256": sha256_file(audio_path),
            "sample_rate": sample_rate,
            "contexts": batch_metadata,
        },
    )
    return paths


def _mfa_environment(
    *,
    cache_root: Path,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    huggingface_root = cache_root / "huggingface"
    huggingface_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ if base_environment is None else base_environment)
    environment["MFA_ROOT_DIR"] = str(cache_root)
    environment["HF_HOME"] = str(huggingface_root)
    return environment


def _run_checked(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    runner: Runner,
) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            env=dict(environment),
        )
    except OSError as exc:
        raise MFAAlignmentError(
            "mfa_runtime_failed", f"could not launch {command[0]}: {exc}"
        ) from exc
    if result.returncode:
        diagnostic = (result.stderr or result.stdout or "no output").strip()
        raise MFAAlignmentError(
            "mfa_runtime_failed",
            f"command failed with exit code {result.returncode}: {diagnostic[-6000:]}",
        )
    return result


def verify_mfa_version(
    *,
    micromamba: str | Path = "micromamba",
    prefix: Path = DEFAULT_MFA_PREFIX,
    cache_root: Path = DEFAULT_MFA_CACHE_ROOT,
    runner: Runner = subprocess.run,
) -> str:
    """Verify that the isolated CLI reports exactly MFA 3.4.1."""

    prefix = prefix.resolve()
    if not prefix.is_dir():
        raise MFAAlignmentError(
            "mfa_runtime_failed", f"MFA environment does not exist: {prefix}"
        )
    environment = _mfa_environment(cache_root=cache_root)
    result = _run_checked(
        [os.fspath(micromamba), "run", "-p", str(prefix), "mfa", "version"],
        environment=environment,
        runner=runner,
    )
    output = "\n".join((result.stdout or "", result.stderr or ""))
    match = _VERSION_RE.search(output)
    reported = match.group(1) if match is not None else ""
    if reported != MFA_VERSION:
        raise MFAAlignmentError(
            "mfa_runtime_failed",
            f"expected MFA {MFA_VERSION}, got {reported or output.strip()!r}",
        )
    return reported


def build_mfa_align_command(
    *,
    paths: MFABatchPaths,
    prefix: Path = DEFAULT_MFA_PREFIX,
    micromamba: str | Path = "micromamba",
    num_jobs: int = 1,
) -> list[str]:
    """Build the exact documented MFA 3.4.1 batched ``align_hf`` command."""

    if num_jobs <= 0:
        raise ValueError("num_jobs must be positive")
    return [
        os.fspath(micromamba),
        "run",
        "-p",
        str(prefix.resolve()),
        "mfa",
        "align_hf",
        str(paths.corpus.resolve()),
        MFA_MODEL_ID,
        str(paths.output.resolve()),
        "--use_g2p",
        "--no_tokenization",
        "--fine_tune",
        "--output_format",
        "json",
        "--no_textgrid_cleanup",
        "--temporary_directory",
        str(paths.temporary.resolve()),
        "--num_jobs",
        str(num_jobs),
        "--clean",
        "--overwrite",
    ]


def invoke_mfa_batch(
    *,
    paths: MFABatchPaths,
    prefix: Path = DEFAULT_MFA_PREFIX,
    cache_root: Path = DEFAULT_MFA_CACHE_ROOT,
    micromamba: str | Path = "micromamba",
    num_jobs: int = 1,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Verify the pinned runtime and execute one batched MFA alignment."""

    version = verify_mfa_version(
        micromamba=micromamba,
        prefix=prefix,
        cache_root=cache_root,
        runner=runner,
    )
    command = build_mfa_align_command(
        paths=paths,
        prefix=prefix,
        micromamba=micromamba,
        num_jobs=num_jobs,
    )
    environment = _mfa_environment(cache_root=cache_root)
    result = _run_checked(command, environment=environment, runner=runner)
    record = {
        "schema_version": 1,
        "backend": "mfa",
        "mfa_version": version,
        "model_id": MFA_MODEL_ID,
        "fine_tune": True,
        "command": command,
        "environment": {
            "MFA_ROOT_DIR": environment["MFA_ROOT_DIR"],
            "HF_HOME": environment["HF_HOME"],
        },
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
    }
    write_json(paths.metadata / "mfa_invocation.json", record)
    return record


def _tier_entries(payload: Mapping[str, Any], tier_name: str) -> list[list[Any]]:
    tiers = payload.get("tiers")
    if not isinstance(tiers, Mapping):
        raise MFAAlignmentError(
            "mfa_word_mapping_failed", "MFA JSON has no tiers object"
        )
    tier = tiers.get(tier_name)
    if not isinstance(tier, Mapping) or tier.get("type") != "interval":
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"MFA JSON has no interval tier {tier_name!r}",
        )
    entries = tier.get("entries")
    if not isinstance(entries, list):
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"MFA tier {tier_name!r} has invalid entries",
        )
    result: list[list[Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, list) or len(entry) != 3:
            raise MFAAlignmentError(
                "mfa_word_mapping_failed",
                f"MFA {tier_name} entry {index} is not [start, end, label]",
            )
        result.append(entry)
    return result


def _parse_intervals(
    entries: Sequence[Sequence[Any]],
    *,
    tier_name: str,
    crop_start_seconds: float,
    crop_duration_seconds: float,
    sample_rate: int,
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    previous_end = 0.0
    for index, entry in enumerate(entries):
        relative_start = _finite_number(entry[0], field=f"{tier_name} start")
        relative_end = _finite_number(entry[1], field=f"{tier_name} end")
        label = str(entry[2]).strip()
        if (
            relative_start < -MFA_TIME_BOUNDARY_TOLERANCE_SECONDS
            or relative_end < relative_start
            or relative_end
            > crop_duration_seconds + MFA_TIME_BOUNDARY_TOLERANCE_SECONDS
        ):
            raise MFAAlignmentError(
                "mfa_word_mapping_failed",
                f"invalid {tier_name} interval {index}: "
                f"[{relative_start}, {relative_end}]",
            )
        if relative_start < previous_end - MFA_INTERVAL_ORDER_TOLERANCE_SECONDS:
            raise MFAAlignmentError(
                "mfa_word_mapping_failed",
                f"overlapping {tier_name} interval {index}",
            )
        relative_start = max(0.0, relative_start)
        relative_end = min(crop_duration_seconds, relative_end)
        absolute_start = crop_start_seconds + relative_start
        absolute_end = crop_start_seconds + relative_end
        parsed.append(
            {
                "label": label,
                "relative_start_seconds": relative_start,
                "relative_end_seconds": relative_end,
                "start_seconds": absolute_start,
                "end_seconds": absolute_end,
                "start_sample": seconds_to_sample(
                    absolute_start, sample_rate, boundary="floor"
                ),
                "end_sample": seconds_to_sample(
                    absolute_end, sample_rate, boundary="ceil"
                ),
            }
        )
        previous_end = max(previous_end, relative_end)
    return parsed


def _normalized_output_token(label: str) -> str:
    tokens = normalize_mfa_token(label)
    if len(tokens) != 1:
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"MFA returned a non-token word label: {label!r}",
        )
    return tokens[0]


def _word_interval_has_non_silence_phone(
    word_interval: Mapping[str, Any],
    phone_intervals: Sequence[Mapping[str, Any]],
) -> bool:
    """Reject lexical word-tier artifacts that cover only MFA silence.

    MFA can occasionally duplicate a lexical label across a long silent span.
    Such an interval is not word evidence even though its label looks valid;
    accepting it shifts every following ordered token mapping.  Keep a word
    interval only when at least one non-silence phone genuinely overlaps it.
    """

    word_start = float(word_interval["relative_start_seconds"])
    word_end = float(word_interval["relative_end_seconds"])
    return any(
        str(phone.get("label", "")).strip()
        and not is_mfa_silence_phone(str(phone["label"]))
        and float(phone["relative_end_seconds"]) > word_start
        and float(phone["relative_start_seconds"]) < word_end
        for phone in phone_intervals
    )


def _word_phones(
    word_interval: Mapping[str, Any],
    phone_intervals: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    word_start = float(word_interval["relative_start_seconds"])
    word_end = float(word_interval["relative_end_seconds"])
    selected: list[dict[str, Any]] = []
    for phone in phone_intervals:
        phone_start = float(phone["relative_start_seconds"])
        phone_end = float(phone["relative_end_seconds"])
        if not str(phone["label"]).strip():
            continue
        if phone_end <= word_start or phone_start >= word_end:
            continue
        if (
            phone_start < word_start - MFA_PHONE_ROUNDING_TOLERANCE_SECONDS
            or phone_end > word_end + MFA_PHONE_ROUNDING_TOLERANCE_SECONDS
        ):
            raise MFAAlignmentError(
                "mfa_word_mapping_failed",
                f"phone {phone['label']!r} is not contained in mapped word "
                f"{word_interval['label']!r}",
            )
        record = dict(phone)
        record["phone"] = record.pop("label")
        record["is_silence"] = is_mfa_silence_phone(str(record["phone"]))
        selected.append(record)
    if not any(not phone["is_silence"] for phone in selected):
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"mapped word {word_interval['label']!r} has no non-silence phone",
        )
    return selected


def _mapped_word_records(
    *,
    context_id: str,
    lexical_words: Sequence[Mapping[str, Any]],
    mappings: Sequence[MFATokenMapping],
    phone_intervals: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Map ordered MFA words, including MFA's split contraction clitics.

    MFA can export a transcript token such as ``model's`` as two word-tier
    intervals, ``model`` and ``'s``. The mapping remains deterministic because
    both streams are ordered; consume the shortest consecutive output sequence
    whose concatenated label normalizes to the requested input token.
    """

    mapped_words: list[dict[str, Any]] = []
    output_index = 0
    for mapping_index, mapping in enumerate(mappings):
        matched: list[Mapping[str, Any]] | None = None
        for end_index in range(
            output_index + 1,
            min(len(lexical_words), output_index + 3) + 1,
        ):
            candidate = "".join(
                str(interval["label"])
                for interval in lexical_words[output_index:end_index]
            )
            normalized = normalize_mfa_token(candidate)
            if normalized == (mapping.token,):
                matched = list(lexical_words[output_index:end_index])
                output_index = end_index
                break
        if matched is None:
            actual = [
                str(interval["label"])
                for interval in lexical_words[output_index : output_index + 3]
            ]
            raise MFAAlignmentError(
                "mfa_word_mapping_failed",
                f"context {context_id} token {mapping_index} expected "
                f"{mapping.token!r}, MFA returned {actual!r}",
            )
        combined = dict(matched[0])
        combined["label"] = mapping.token
        for prefix in ("relative_", ""):
            combined[f"{prefix}end_seconds"] = matched[-1][f"{prefix}end_seconds"]
        combined["end_sample"] = matched[-1]["end_sample"]
        record = dict(combined)
        record.pop("label")
        record.update(
            {
                "source_word_ids": list(mapping.source_word_ids),
                "source_text": mapping.source_text,
                "mfa_token": mapping.token,
                "mfa_word_tier_labels": [
                    str(interval["label"]) for interval in matched
                ],
                "phones": _word_phones(combined, phone_intervals),
            }
        )
        mapped_words.append(record)
    if output_index != len(lexical_words):
        remaining = [
            str(interval["label"]) for interval in lexical_words[output_index:]
        ]
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"context {context_id} returned unmapped MFA words {remaining!r}",
        )
    return mapped_words


def _find_output_json(paths: MFABatchPaths, context_id: str) -> Path:
    expected = paths.speaker_output / f"{context_id}.json"
    if expected.is_file():
        return expected
    candidates = sorted(paths.output.rglob(f"{context_id}.json"))
    if len(candidates) != 1:
        raise MFAAlignmentError(
            "mfa_utterance_unaligned",
            f"expected exactly one MFA JSON for {context_id}, found {len(candidates)}",
        )
    return candidates[0]


def _parse_mfa_context(
    *,
    paths: MFABatchPaths,
    raw_context: Mapping[str, Any],
    sample_rate: int,
) -> dict[str, Any]:
    context_id = str(raw_context["context_id"])
    crop_start = float(raw_context["crop_source_start_seconds"])
    crop_end = float(raw_context["crop_source_end_seconds"])
    crop_duration = crop_end - crop_start
    output_path = _find_output_json(paths, context_id)
    payload = read_json(output_path)
    if not isinstance(payload, Mapping):
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"MFA output for {context_id} is not an object",
        )
    output_end = _finite_number(payload.get("end"), field="MFA JSON end")
    if abs(output_end - crop_duration) > 0.05:
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"MFA output duration mismatch for {context_id}: "
            f"{output_end:.6f} vs {crop_duration:.6f}",
        )

    word_intervals = _parse_intervals(
        _tier_entries(payload, "words"),
        tier_name="word",
        crop_start_seconds=crop_start,
        crop_duration_seconds=crop_duration,
        sample_rate=sample_rate,
    )
    phone_intervals = _parse_intervals(
        _tier_entries(payload, "phones"),
        tier_name="phone",
        crop_start_seconds=crop_start,
        crop_duration_seconds=crop_duration,
        sample_rate=sample_rate,
    )
    lexical_word_candidates = [
        interval
        for interval in word_intervals
        if not _is_mfa_silence_word(str(interval["label"]))
    ]
    ignored_silence_only_words = [
        dict(interval)
        for interval in lexical_word_candidates
        if not _word_interval_has_non_silence_phone(interval, phone_intervals)
    ]
    lexical_words = [
        interval
        for interval in lexical_word_candidates
        if _word_interval_has_non_silence_phone(interval, phone_intervals)
    ]
    raw_mappings = raw_context.get("token_mappings")
    if not isinstance(raw_mappings, list):
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"context {context_id} has no token mapping",
        )
    mappings = tuple(_coerce_token_mapping(item) for item in raw_mappings)
    mapped_words = _mapped_word_records(
        context_id=context_id,
        lexical_words=lexical_words,
        mappings=mappings,
        phone_intervals=phone_intervals,
    )

    all_phones: list[dict[str, Any]] = []
    for phone in phone_intervals:
        record = dict(phone)
        record["phone"] = record.pop("label")
        record["is_silence"] = is_mfa_silence_phone(str(record["phone"]))
        if not record["is_silence"] and float(record["end_seconds"]) <= float(
            record["start_seconds"]
        ):
            raise MFAAlignmentError(
                "mfa_word_mapping_failed",
                f"non-silence phone {record['phone']!r} has zero duration",
            )
        all_phones.append(record)
    return {
        "context_id": context_id,
        "crop_source_start_seconds": crop_start,
        "crop_source_end_seconds": crop_end,
        "crop_source_start_sample": int(raw_context["crop_source_start_sample"]),
        "crop_source_end_sample": int(raw_context["crop_source_end_sample"]),
        "ordered_source_word_ids": list(raw_context["ordered_source_word_ids"]),
        "original_source_words": list(raw_context["original_source_words"]),
        "boundary_ids": list(raw_context["boundary_ids"]),
        "words": mapped_words,
        "phones": all_phones,
        "ignored_silence_only_word_intervals": ignored_silence_only_words,
        "mfa_output_json": str(output_path),
    }


def parse_mfa_batch(paths: MFABatchPaths) -> dict[str, Any]:
    """Parse every usable MFA context without discarding partial success.

    MFA reports per-utterance alignment failures by omitting their exported
    JSON files while still exiting successfully for the batch. A long
    recording can therefore contain hundreds of valid contexts and a handful
    of failures. Preserve the valid evidence and attach a fail-closed error to
    each missing or malformed context; the boundary planner decides which
    contexts are production-required and which are optional enhancements.
    """

    batch = read_json(paths.metadata / "batch.json")
    if not isinstance(batch, Mapping):
        raise MFAAlignmentError("mfa_word_mapping_failed", "invalid MFA batch metadata")
    sample_rate = int(batch.get("sample_rate", 0))
    if sample_rate <= 0:
        raise MFAAlignmentError(
            "mfa_word_mapping_failed", "invalid MFA batch sample rate"
        )
    raw_contexts = batch.get("contexts")
    if not isinstance(raw_contexts, list) or not raw_contexts:
        raise MFAAlignmentError(
            "mfa_word_mapping_failed", "MFA batch metadata has no contexts"
        )

    parsed_contexts: list[dict[str, Any]] = []
    context_errors: list[dict[str, Any]] = []
    for raw_context in raw_contexts:
        if not isinstance(raw_context, Mapping):
            raise MFAAlignmentError(
                "mfa_word_mapping_failed", "invalid MFA context metadata"
            )
        context_id = str(raw_context.get("context_id", ""))
        if not context_id:
            raise MFAAlignmentError(
                "mfa_word_mapping_failed", "MFA context metadata has no context ID"
            )
        try:
            parsed_contexts.append(
                _parse_mfa_context(
                    paths=paths,
                    raw_context=raw_context,
                    sample_rate=sample_rate,
                )
            )
        except (MFAAlignmentError, OSError, TypeError, ValueError) as error:
            context_errors.append(
                {
                    "context_id": context_id,
                    "code": (
                        error.code
                        if isinstance(error, MFAAlignmentError)
                        else "mfa_word_mapping_failed"
                    ),
                    "error": f"{type(error).__name__}: {error}",
                    "boundary_ids": list(raw_context.get("boundary_ids", [])),
                }
            )

    return {
        "schema_version": 1,
        "backend": "mfa",
        "mfa_version": MFA_VERSION,
        "model_id": MFA_MODEL_ID,
        "fine_tune": True,
        "sample_rate": sample_rate,
        "source_audio": batch.get("source_audio"),
        "source_audio_sha256": batch.get("source_audio_sha256"),
        "contexts": parsed_contexts,
        "context_errors": context_errors,
    }


def source_word_alignment(
    context: Mapping[str, Any], source_word_id: int
) -> dict[str, Any]:
    """Return unambiguous aggregate evidence for one source occurrence.

    One source occurrence may normalize to multiple sequential MFA tokens.  A
    token shared by multiple source occurrences is preserved in metadata but
    cannot provide an unambiguous word-level cut coordinate and therefore
    fails closed here.
    """

    words = context.get("words")
    if not isinstance(words, list):
        raise MFAAlignmentError(
            "mfa_word_mapping_failed", "parsed MFA context has no words"
        )
    matches: list[Mapping[str, Any]] = []
    for word in words:
        if not isinstance(word, Mapping):
            continue
        source_ids = word.get("source_word_ids")
        if not isinstance(source_ids, list):
            continue
        if source_word_id in source_ids:
            if len(source_ids) != 1:
                raise MFAAlignmentError(
                    "mfa_word_mapping_failed",
                    f"source word {source_word_id} shares MFA token "
                    f"{word.get('mfa_token')!r} with source IDs {source_ids}",
                )
            matches.append(word)
    if not matches:
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"source word {source_word_id} has no mapped MFA interval",
        )
    matches.sort(key=lambda item: float(item["start_seconds"]))
    for left, right in pairwise(matches):
        if (
            float(right["start_seconds"])
            < float(left["end_seconds"]) - MFA_PHONE_ROUNDING_TOLERANCE_SECONDS
        ):
            raise MFAAlignmentError(
                "mfa_word_mapping_failed",
                f"source word {source_word_id} has overlapping MFA token intervals",
            )
    phones = [phone for item in matches for phone in item.get("phones", [])]
    non_silence = [phone for phone in phones if not phone.get("is_silence")]
    if not non_silence:
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"source word {source_word_id} has no mapped non-silence phone",
        )
    source_texts = {str(item["source_text"]) for item in matches}
    if len(source_texts) != 1:
        raise MFAAlignmentError(
            "mfa_word_mapping_failed",
            f"source word {source_word_id} has inconsistent source text mappings",
        )
    return {
        "source_word_id": source_word_id,
        "source_text": next(iter(source_texts)),
        "mfa_tokens": [str(item["mfa_token"]) for item in matches],
        "start_seconds": float(matches[0]["start_seconds"]),
        "end_seconds": float(matches[-1]["end_seconds"]),
        "start_sample": int(matches[0]["start_sample"]),
        "end_sample": int(matches[-1]["end_sample"]),
        "phones": phones,
        "first_non_silence_phone": non_silence[0],
        "last_non_silence_phone": non_silence[-1],
    }


def _recovery_context_without_degenerate_optional_runs(
    context: MFAContextSpec,
) -> tuple[MFAContextSpec, list[int]]:
    """Drop only impossible optional anchor runs from an MFA recovery crop.

    Long-form Whisper can emit many omitted words at one identical
    zero-duration anchor.  Feeding such a transcript to MFA makes the local
    utterance acoustically impossible and can prevent it from exporting any
    alignment.  A single approximate/zero anchor is preserved; only runs of
    at least two consecutive *omitted* zero-duration occurrences at the same
    anchor are excluded on the recovery attempt.  Selected occurrences are
    never removed here.

    This transformation affects the reference transcript only.  It does not
    create coordinates or authorize a cut; all required retained words still
    need valid MFA word and phone evidence downstream.
    """

    words = list(context.words)
    excluded_ids: set[int] = set()
    run_start = 0
    while run_start < len(words):
        word = words[run_start]
        is_optional_zero = (
            not word.selected
            and abs(word.end_seconds - word.start_seconds)
            <= MFA_INTERVAL_ORDER_TOLERANCE_SECONDS
        )
        if not is_optional_zero:
            run_start += 1
            continue
        anchor = word.start_seconds
        run_end = run_start + 1
        while run_end < len(words):
            candidate = words[run_end]
            if (
                candidate.selected
                or abs(candidate.end_seconds - candidate.start_seconds)
                > MFA_INTERVAL_ORDER_TOLERANCE_SECONDS
                or abs(candidate.start_seconds - anchor)
                > MFA_INTERVAL_ORDER_TOLERANCE_SECONDS
            ):
                break
            run_end += 1
        if run_end - run_start >= 2:
            excluded_ids.update(item.word_id for item in words[run_start:run_end])
        run_start = run_end

    if not excluded_ids:
        return context, []
    retained_words = tuple(
        word for word in context.words if word.word_id not in excluded_ids
    )
    retained_mappings = (
        tuple(
            mapping
            for mapping in context.token_mappings
            if not set(mapping.source_word_ids) & excluded_ids
        )
        if context.token_mappings is not None
        else None
    )
    return (
        MFAContextSpec(
            context_id=context.context_id,
            crop_source_start_seconds=context.crop_source_start_seconds,
            crop_source_end_seconds=context.crop_source_end_seconds,
            words=retained_words,
            boundary_ids=context.boundary_ids,
            token_mappings=retained_mappings,
        ),
        sorted(excluded_ids),
    )


def align_mfa_contexts(
    *,
    audio_path: Path,
    contexts: Sequence[MFAContextSpec | Mapping[str, Any]],
    work_dir: Path,
    prefix: Path = DEFAULT_MFA_PREFIX,
    cache_root: Path = DEFAULT_MFA_CACHE_ROOT,
    micromamba: str | Path = "micromamba",
    num_jobs: int = 1,
    context_recovery_attempts: int = MFA_CONTEXT_RECOVERY_ATTEMPTS,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Align local contexts, retrying only utterances MFA did not export.

    MFA can exit successfully after aligning almost an entire long-form batch
    while omitting JSON for a handful of utterances.  Throwing away hundreds
    of valid contexts and rerunning the whole recording is both expensive and
    unnecessary.  Preserve the successful evidence and retry the missing
    utterances together in one small recovery batch.

    Recovery never changes the alignment backend and never substitutes
    Whisper coordinates.  If a context remains unaligned it stays an explicit
    fail-closed context error for the semantic/source-preservation policy.
    """

    if context_recovery_attempts < 0:
        raise ValueError("context_recovery_attempts must be non-negative")
    coerced_contexts = tuple(_coerce_context(context) for context in contexts)

    # One malformed ASR occurrence must not poison every otherwise independent
    # alignment context in a long recording.  Validate contexts before creating
    # the shared MFA corpus and retain failures as context-local, fail-closed
    # evidence.  The valid contexts still use exactly one batched MFA call.
    valid_contexts: list[MFAContextSpec] = []
    preflight_errors: dict[str, dict[str, Any]] = {}
    for context in coerced_contexts:
        try:
            _validated_context_mappings(context)
        except (MFAAlignmentError, TypeError, ValueError) as error:
            preflight_errors[context.context_id] = {
                "context_id": context.context_id,
                "code": (
                    error.code
                    if isinstance(error, MFAAlignmentError)
                    else "mfa_word_mapping_failed"
                ),
                "error": f"{type(error).__name__}: {error}",
                "boundary_ids": list(context.boundary_ids),
                "stage": "context_preflight",
            }
        else:
            valid_contexts.append(context)

    if not valid_contexts:
        paths = MFABatchPaths(
            root=(work_dir.resolve() / "mfa_alignment"),
            corpus=(work_dir.resolve() / "mfa_alignment" / "corpus"),
            metadata=(work_dir.resolve() / "mfa_alignment" / "metadata"),
            output=(work_dir.resolve() / "mfa_alignment" / "output"),
            temporary=(work_dir.resolve() / "mfa_alignment" / "temp"),
        )
        _reset_batch_directories(paths)
        with sf.SoundFile(audio_path.resolve()) as source:
            sample_rate = int(source.samplerate)
        result = {
            "schema_version": 1,
            "backend": "mfa",
            "mfa_version": MFA_VERSION,
            "model_id": MFA_MODEL_ID,
            "fine_tune": True,
            "sample_rate": sample_rate,
            "source_audio": str(audio_path.resolve()),
            "source_audio_sha256": sha256_file(audio_path.resolve()),
            "contexts": [],
            "context_errors": [
                preflight_errors[context.context_id] for context in coerced_contexts
            ],
            "preflight_context_errors": [
                preflight_errors[context.context_id] for context in coerced_contexts
            ],
            "invocation": {
                "status": "skipped_no_valid_contexts",
                "reason": "all MFA contexts failed independent preflight",
            },
            "recovery_batches": [],
        }
        write_json(paths.metadata / "mfa_alignment.json", result)
        return result

    paths = prepare_mfa_batch(
        audio_path=audio_path,
        contexts=valid_contexts,
        work_dir=work_dir,
    )
    try:
        invocation = invoke_mfa_batch(
            paths=paths,
            prefix=prefix,
            cache_root=cache_root,
            micromamba=micromamba,
            num_jobs=num_jobs,
            runner=runner,
        )
        invocation["status"] = "complete"
    except MFAAlignmentError as error:
        # MFA can return a non-zero process status even when it exported valid
        # JSON for some utterances.  Parse and retain independently validated
        # outputs instead of discarding them.  Missing contexts remain
        # fail-closed and are eligible for the bounded recovery batch below.
        invocation = {
            "schema_version": 1,
            "backend": "mfa",
            "mfa_version": MFA_VERSION,
            "model_id": MFA_MODEL_ID,
            "fine_tune": True,
            "status": "failed_with_possible_partial_outputs",
            "error_code": error.code,
            "error": f"{type(error).__name__}: {error}",
        }
        write_json(paths.metadata / "mfa_invocation.json", invocation)
    result = parse_mfa_batch(paths)
    result["invocation"] = invocation
    recovery_batches: list[dict[str, Any]] = []
    contexts_by_id = {
        str(context["context_id"]): context for context in result["contexts"]
    }
    errors_by_id = {
        str(error["context_id"]): error for error in result["context_errors"]
    }
    specs_by_id = {context.context_id: context for context in valid_contexts}

    for recovery_index in range(1, context_recovery_attempts + 1):
        retry_ids = [
            context.context_id
            for context in valid_contexts
            if context.context_id in errors_by_id
            and errors_by_id[context.context_id].get("code")
            == "mfa_utterance_unaligned"
        ]
        if not retry_ids:
            break
        retry_contexts: list[MFAContextSpec] = []
        excluded_anchor_words: dict[str, list[int]] = {}
        for context_id in retry_ids:
            recovery_context, excluded_ids = (
                _recovery_context_without_degenerate_optional_runs(
                    specs_by_id[context_id]
                )
            )
            retry_contexts.append(recovery_context)
            if excluded_ids:
                excluded_anchor_words[context_id] = excluded_ids
        retry_work_dir = paths.root / "recovery" / f"retry_{recovery_index:02d}"
        retry_paths = prepare_mfa_batch(
            audio_path=audio_path,
            contexts=retry_contexts,
            work_dir=retry_work_dir,
        )
        try:
            retry_invocation = invoke_mfa_batch(
                paths=retry_paths,
                prefix=prefix,
                cache_root=cache_root,
                micromamba=micromamba,
                num_jobs=min(num_jobs, len(retry_contexts)),
                runner=runner,
            )
            retry_invocation["status"] = "complete"
        except MFAAlignmentError as error:
            retry_invocation = {
                "schema_version": 1,
                "backend": "mfa",
                "mfa_version": MFA_VERSION,
                "model_id": MFA_MODEL_ID,
                "fine_tune": True,
                "status": "failed_with_possible_partial_outputs",
                "error_code": error.code,
                "error": f"{type(error).__name__}: {error}",
            }
            write_json(
                retry_paths.metadata / "mfa_invocation.json",
                retry_invocation,
            )
        retry_result = parse_mfa_batch(retry_paths)
        resolved_ids: list[str] = []
        for context in retry_result["contexts"]:
            context_id = str(context["context_id"])
            contexts_by_id[context_id] = context
            errors_by_id.pop(context_id, None)
            resolved_ids.append(context_id)
        retry_errors = {
            str(error["context_id"]): error for error in retry_result["context_errors"]
        }
        for context_id in retry_ids:
            if context_id in retry_errors:
                errors_by_id[context_id] = retry_errors[context_id]
        recovery_batches.append(
            {
                "recovery_index": recovery_index,
                "attempted_context_ids": retry_ids,
                "resolved_context_ids": resolved_ids,
                "remaining_context_errors": [
                    errors_by_id[context_id]
                    for context_id in retry_ids
                    if context_id in errors_by_id
                ],
                "excluded_degenerate_optional_word_ids": excluded_anchor_words,
                "invocation": retry_invocation,
                "artifact_root": str(retry_paths.root),
            }
        )

    errors_by_id.update(preflight_errors)
    ordered_ids = [context.context_id for context in coerced_contexts]
    result["contexts"] = [
        contexts_by_id[context_id]
        for context_id in ordered_ids
        if context_id in contexts_by_id
    ]
    result["context_errors"] = [
        errors_by_id[context_id]
        for context_id in ordered_ids
        if context_id in errors_by_id
    ]
    result["recovery_batches"] = recovery_batches
    result["preflight_context_errors"] = [
        preflight_errors[context_id]
        for context_id in ordered_ids
        if context_id in preflight_errors
    ]
    write_json(paths.metadata / "mfa_alignment.json", result)
    return result
