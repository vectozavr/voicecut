from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from voicecut.common import read_json
from voicecut.mfa_alignment import (
    MFA_MODEL_ID,
    MFA_VERSION,
    MFAAlignmentError,
    MFABatchPaths,
    MFASourceWord,
    MFATokenMapping,
    build_mfa_align_command,
    invoke_mfa_batch,
    normalize_mfa_token,
    normalize_source_words,
    parse_mfa_batch,
    prepare_mfa_batch,
    seconds_to_sample,
    source_word_alignment,
)

SAMPLE_RATE = 16_000


def _write_source(path: Path, *, seconds: float = 3.0) -> None:
    samples = np.zeros((round(seconds * SAMPLE_RATE), 1), dtype=np.float32)
    sf.write(path, samples, SAMPLE_RATE, subtype="PCM_16")


def _word(
    word_id: int,
    text: str,
    start: float,
    end: float,
    *,
    selected: bool = True,
) -> MFASourceWord:
    return MFASourceWord(
        word_id=word_id,
        text=text,
        start_seconds=start,
        end_seconds=end,
        selected=selected,
    )


def _prepare(
    tmp_path: Path,
    *,
    words: list[MFASourceWord],
    crop_start: float = 0.5,
    crop_end: float = 2.5,
    token_mappings: tuple[MFATokenMapping, ...] | None = None,
) -> MFABatchPaths:
    audio_path = tmp_path / "source.wav"
    _write_source(audio_path)
    return prepare_mfa_batch(
        audio_path=audio_path,
        contexts=[
            {
                "context_id": "context_000",
                "crop_source_start_seconds": crop_start,
                "crop_source_end_seconds": crop_end,
                "words": words,
                "boundary_ids": ["gap_000"],
                **(
                    {"token_mappings": token_mappings}
                    if token_mappings is not None
                    else {}
                ),
            }
        ],
        work_dir=tmp_path / "work",
    )


def test_zero_duration_whisper_anchor_is_valid_mfa_crop_metadata(
    tmp_path: Path,
) -> None:
    paths = _prepare(
        tmp_path,
        words=[
            _word(0, "turns", 0.80, 0.90),
            _word(1, "out", 0.90, 0.90),
            _word(2, "that", 0.90, 0.90),
        ],
    )

    assert (paths.speaker_corpus / "context_000.lab").read_text().strip() == (
        "turns out that"
    )


def test_reversed_whisper_anchor_remains_invalid_for_mfa(tmp_path: Path) -> None:
    with pytest.raises(MFAAlignmentError, match="invalid anchor timestamps"):
        _prepare(
            tmp_path,
            words=[_word(0, "word", 0.90, 0.80)],
        )


def _write_mfa_json(
    paths: MFABatchPaths,
    *,
    word_entries: list[list[Any]],
    phone_entries: list[list[Any]],
    duration: float = 2.0,
) -> Path:
    output_path = paths.speaker_output / "context_000.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "start": 0.0,
                "end": duration,
                "tiers": {
                    "words": {"type": "interval", "entries": word_entries},
                    "phones": {"type": "interval", "entries": phone_entries},
                },
            }
        ),
        encoding="utf-8",
    )
    return output_path


def test_parse_preserves_success_when_another_batch_context_is_missing(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "source.wav"
    _write_source(audio_path)
    paths = prepare_mfa_batch(
        audio_path=audio_path,
        contexts=[
            {
                "context_id": "context_000",
                "crop_source_start_seconds": 0.5,
                "crop_source_end_seconds": 2.5,
                "words": [_word(0, "valid", 0.8, 1.2)],
                "boundary_ids": ["gap_000"],
            },
            {
                "context_id": "context_001",
                "crop_source_start_seconds": 0.5,
                "crop_source_end_seconds": 2.5,
                "words": [_word(1, "missing", 1.3, 1.7)],
                "boundary_ids": ["gap_001"],
            },
        ],
        work_dir=tmp_path / "work",
    )
    _write_mfa_json(
        paths,
        word_entries=[
            [0.0, 0.20, "<eps>"],
            [0.20, 0.70, "valid"],
            [0.70, 2.0, "<eps>"],
        ],
        phone_entries=[
            [0.0, 0.20, "sil"],
            [0.20, 0.45, "V"],
            [0.45, 0.70, "D"],
            [0.70, 2.0, "sil"],
        ],
    )

    result = parse_mfa_batch(paths)

    assert [context["context_id"] for context in result["contexts"]] == ["context_000"]
    assert result["context_errors"] == [
        {
            "context_id": "context_001",
            "code": "mfa_word_mapping_failed",
            "error": (
                "MFAAlignmentError: mfa_word_mapping_failed: expected exactly one "
                "MFA JSON for context_001, found 0"
            ),
            "boundary_ids": ["gap_001"],
        }
    ]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("  Example?!  ", ("example",)),
        ("don’t", ("don't",)),
        ("Voice-Cut", ("voice", "cut")),
        ("OpenAI", ("openai",)),
        ("50%", ("50", "percent")),
    ],
)
def test_normalize_mfa_token_handles_punctuation_contractions_and_oov(
    source: str,
    expected: tuple[str, ...],
) -> None:
    assert normalize_mfa_token(source) == expected


def test_normalize_source_words_preserves_one_to_many_mapping() -> None:
    mappings, nonlexical = normalize_source_words(
        [
            _word(7, "Voice-Cut", 0.1, 0.4),
            _word(8, "won’t", 0.5, 0.8),
            _word(9, ".", 0.8, 0.81),
        ]
    )

    assert [mapping.token for mapping in mappings] == ["voice", "cut", "won't"]
    assert [mapping.source_word_ids for mapping in mappings] == [(7,), (7,), (8,)]
    assert all(mapping.source_text == "Voice-Cut" for mapping in mappings[:2])
    assert nonlexical == (9,)


def test_prepare_and_parse_mock_oov_output_with_reversible_tokens(
    tmp_path: Path,
) -> None:
    paths = _prepare(
        tmp_path,
        words=[
            _word(20, "VoiceCut,", 0.6, 0.9),
            _word(21, "won’t", 1.0, 1.2),
            _word(22, "re-render", 1.3, 1.7),
        ],
    )

    assert (paths.speaker_corpus / "context_000.lab").read_text(
        encoding="utf-8"
    ) == "voicecut won't re render\n"
    metadata = read_json(paths.metadata / "context_000.json")
    assert metadata["normalized_mfa_tokens"] == [
        "voicecut",
        "won't",
        "re",
        "render",
    ]
    assert metadata["token_mappings"][-2:] == [
        {
            "source_text": "re-render",
            "source_word_ids": [22],
            "token": "re",
        },
        {
            "source_text": "re-render",
            "source_word_ids": [22],
            "token": "render",
        },
    ]
    _write_mfa_json(
        paths,
        word_entries=[
            [0.0, 0.10, "<eps>"],
            [0.10, 0.50, "voicecut"],
            [0.50, 0.80, "won't"],
            [0.80, 1.05, "re"],
            [1.05, 1.50, "render"],
            [1.50, 2.0, "<eps>"],
        ],
        phone_entries=[
            [0.0, 0.10, "sil"],
            [0.10, 0.20, "V"],
            [0.20, 0.30, "OY1"],
            [0.30, 0.40, "S"],
            [0.40, 0.50, "T"],
            [0.50, 0.60, "W"],
            [0.60, 0.70, "OW1"],
            [0.70, 0.80, "N"],
            [0.80, 0.92, "R"],
            [0.92, 1.05, "IY1"],
            [1.05, 1.20, "R"],
            [1.20, 1.35, "ER0"],
            [1.35, 1.50, "D"],
            [1.50, 2.0, "sil"],
        ],
    )

    context = parse_mfa_batch(paths)["contexts"][0]
    assert [word["mfa_token"] for word in context["words"]] == [
        "voicecut",
        "won't",
        "re",
        "render",
    ]
    assert source_word_alignment(context, 20)["mfa_tokens"] == ["voicecut"]
    assert source_word_alignment(context, 22)["mfa_tokens"] == ["re", "render"]


def test_parse_maps_mfa_split_possessive_clitic_to_one_source_word(
    tmp_path: Path,
) -> None:
    paths = _prepare(
        tmp_path,
        words=[_word(23, "model's", 0.7, 1.2)],
    )
    _write_mfa_json(
        paths,
        word_entries=[
            [0.0, 0.20, "<eps>"],
            [0.20, 0.55, "model"],
            [0.55, 0.70, "'s"],
            [0.70, 2.0, "<eps>"],
        ],
        phone_entries=[
            [0.0, 0.20, "sil"],
            [0.20, 0.35, "M"],
            [0.35, 0.55, "AH0"],
            [0.55, 0.70, "Z"],
            [0.70, 2.0, "sil"],
        ],
    )

    result = parse_mfa_batch(paths)

    assert result["context_errors"] == []
    mapped = result["contexts"][0]["words"][0]
    assert mapped["source_word_ids"] == [23]
    assert mapped["mfa_token"] == "model's"
    assert mapped["mfa_word_tier_labels"] == ["model", "'s"]
    assert [phone["phone"] for phone in mapped["phones"]] == ["M", "AH0", "Z"]


def test_parse_mfa_json_maps_repeated_words_sequentially_and_preserves_phones(
    tmp_path: Path,
) -> None:
    paths = _prepare(
        tmp_path,
        words=[
            _word(10, "with", 0.7, 1.0),
            _word(11, "with", 1.1, 1.4),
        ],
    )
    _write_mfa_json(
        paths,
        word_entries=[
            [0.0, 0.20, "<eps>"],
            [0.20, 0.50, "with"],
            [0.50, 0.60, "<eps>"],
            [0.60, 0.95, "with"],
            [0.95, 2.0, "<eps>"],
        ],
        phone_entries=[
            [0.0, 0.20, "sil"],
            [0.20, 0.30, "W"],
            [0.30, 0.40, "IH1"],
            [0.40, 0.50, "TH"],
            [0.50, 0.60, "sil"],
            [0.60, 0.72, "W"],
            [0.72, 0.84, "IH1"],
            [0.84, 0.95, "TH"],
            [0.95, 2.0, "sil"],
        ],
    )

    result = parse_mfa_batch(paths)
    context = result["contexts"][0]

    assert result["backend"] == "mfa"
    assert result["mfa_version"] == "3.4.1"
    assert [item["source_word_ids"] for item in context["words"]] == [[10], [11]]
    assert [item["mfa_token"] for item in context["words"]] == ["with", "with"]
    assert [phone["phone"] for phone in context["words"][0]["phones"]] == [
        "W",
        "IH1",
        "TH",
    ]
    assert context["phones"][0]["is_silence"] is True


def test_parse_mfa_json_keeps_lexical_word_silence(tmp_path: Path) -> None:
    paths = _prepare(
        tmp_path,
        words=[
            _word(12, "ordinary", 0.6, 0.9),
            _word(13, "silence", 0.9, 1.3),
        ],
    )
    _write_mfa_json(
        paths,
        word_entries=[
            [0.0, 0.20, "<eps>"],
            [0.20, 0.55, "ordinary"],
            [0.55, 0.95, "silence"],
            [0.95, 2.0, "<eps>"],
        ],
        phone_entries=[
            [0.0, 0.20, "sil"],
            [0.20, 0.35, "AO1"],
            [0.35, 0.55, "IY0"],
            [0.55, 0.65, "S"],
            [0.65, 0.78, "AY1"],
            [0.78, 0.95, "S"],
            [0.95, 2.0, "sil"],
        ],
    )

    context = parse_mfa_batch(paths)["contexts"][0]

    assert [word["mfa_token"] for word in context["words"]] == [
        "ordinary",
        "silence",
    ]
    assert source_word_alignment(context, 13)["last_non_silence_phone"]["phone"] == "S"


def test_parse_mfa_json_converts_crop_relative_times_to_absolute_samples(
    tmp_path: Path,
) -> None:
    paths = _prepare(
        tmp_path,
        words=[_word(30, "example", 1.0, 1.4)],
        crop_start=0.5,
        crop_end=2.5,
    )
    _write_mfa_json(
        paths,
        word_entries=[
            [0.0, 0.25, "<eps>"],
            [0.25, 0.90, "example"],
            [0.90, 2.0, "<eps>"],
        ],
        phone_entries=[
            [0.0, 0.25, "sil"],
            [0.25, 0.40, "IH0"],
            [0.40, 0.55, "G"],
            [0.55, 0.70, "Z"],
            [0.70, 0.90, "AH0"],
            [0.90, 2.0, "sil"],
        ],
    )

    aligned = source_word_alignment(parse_mfa_batch(paths)["contexts"][0], 30)

    assert aligned["start_seconds"] == pytest.approx(0.75)
    assert aligned["end_seconds"] == pytest.approx(1.40)
    assert aligned["start_sample"] == 12_000
    assert aligned["end_sample"] == 22_400
    assert aligned["last_non_silence_phone"]["phone"] == "AH0"
    assert aligned["last_non_silence_phone"]["end_sample"] == 22_400


@pytest.mark.parametrize(
    ("boundary", "expected"),
    [("floor", 19), ("nearest", 20), ("ceil", 20)],
)
def test_seconds_to_sample_has_explicit_safe_rounding(
    boundary: str,
    expected: int,
) -> None:
    assert seconds_to_sample(0.0196, 1_000, boundary=boundary) == expected


def test_one_source_word_can_aggregate_several_mfa_tokens(tmp_path: Path) -> None:
    paths = _prepare(
        tmp_path,
        words=[_word(40, "re-render", 0.8, 1.4)],
    )
    _write_mfa_json(
        paths,
        word_entries=[
            [0.0, 0.20, "<eps>"],
            [0.20, 0.50, "re"],
            [0.50, 0.90, "render"],
            [0.90, 2.0, "<eps>"],
        ],
        phone_entries=[
            [0.0, 0.20, "sil"],
            [0.20, 0.35, "R"],
            [0.35, 0.50, "IY1"],
            [0.50, 0.65, "R"],
            [0.65, 0.78, "D"],
            [0.78, 0.90, "ER0"],
            [0.90, 2.0, "sil"],
        ],
    )

    aligned = source_word_alignment(parse_mfa_batch(paths)["contexts"][0], 40)

    assert aligned["mfa_tokens"] == ["re", "render"]
    assert aligned["start_seconds"] == pytest.approx(0.70)
    assert aligned["end_seconds"] == pytest.approx(1.40)
    assert [phone["phone"] for phone in aligned["phones"]] == [
        "R",
        "IY1",
        "R",
        "D",
        "ER0",
    ]


def test_parse_reports_final_mapped_word_without_non_silence_phone(
    tmp_path: Path,
) -> None:
    paths = _prepare(
        tmp_path,
        words=[
            _word(50, "valid", 0.7, 1.0),
            _word(51, "missing", 1.1, 1.5),
        ],
    )
    _write_mfa_json(
        paths,
        word_entries=[
            [0.0, 0.20, "<eps>"],
            [0.20, 0.50, "valid"],
            [0.50, 0.90, "missing"],
            [0.90, 2.0, "<eps>"],
        ],
        phone_entries=[
            [0.0, 0.20, "sil"],
            [0.20, 0.35, "V"],
            [0.35, 0.50, "D"],
            [0.50, 0.90, "sil"],
            [0.90, 2.0, "sil"],
        ],
    )

    result = parse_mfa_batch(paths)

    assert result["contexts"] == []
    assert result["context_errors"][0]["context_id"] == "context_000"
    assert (
        "mapped word 'missing' has no non-silence phone"
        in result["context_errors"][0]["error"]
    )


@pytest.mark.parametrize(
    "phone_entries",
    [
        [
            [0.0, 0.20, "sil"],
            [0.20, 0.45, "W"],
            [0.40, 0.70, "ER1"],
            [0.70, 2.0, "sil"],
        ],
        [
            [0.0, 0.20, "sil"],
            [0.10, 0.70, "W"],
            [0.70, 2.0, "sil"],
        ],
    ],
    ids=["overlapping-phone-tier", "phone-outside-word"],
)
def test_parse_reports_overlapping_or_invalid_phone_geometry(
    tmp_path: Path,
    phone_entries: list[list[Any]],
) -> None:
    paths = _prepare(tmp_path, words=[_word(60, "word", 0.8, 1.2)])
    _write_mfa_json(
        paths,
        word_entries=[
            [0.0, 0.20, "<eps>"],
            [0.20, 0.70, "word"],
            [0.70, 2.0, "<eps>"],
        ],
        phone_entries=phone_entries,
    )

    result = parse_mfa_batch(paths)

    assert result["contexts"] == []
    assert result["context_errors"] == [
        {
            "context_id": "context_000",
            "code": "mfa_word_mapping_failed",
            "error": result["context_errors"][0]["error"],
            "boundary_ids": ["gap_000"],
        }
    ]


def test_ambiguous_many_source_words_to_one_token_fails_closed(tmp_path: Path) -> None:
    mappings = (
        MFATokenMapping(
            token="cannot",
            source_word_ids=(70, 71),
            source_text="can not",
        ),
    )
    paths = _prepare(
        tmp_path,
        words=[
            _word(70, "can", 0.8, 1.0),
            _word(71, "not", 1.0, 1.2),
        ],
        token_mappings=mappings,
    )
    _write_mfa_json(
        paths,
        word_entries=[
            [0.0, 0.20, "<eps>"],
            [0.20, 0.70, "cannot"],
            [0.70, 2.0, "<eps>"],
        ],
        phone_entries=[
            [0.0, 0.20, "sil"],
            [0.20, 0.35, "K"],
            [0.35, 0.50, "AE1"],
            [0.50, 0.70, "T"],
            [0.70, 2.0, "sil"],
        ],
    )
    context = parse_mfa_batch(paths)["contexts"][0]

    with pytest.raises(
        MFAAlignmentError,
        match=r"shares MFA token 'cannot' with source IDs \[70, 71\]",
    ):
        source_word_alignment(context, 70)


def test_invoke_mfa_batch_uses_exact_flags_and_one_align_hf_call(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / ".mfa-env"
    prefix.mkdir()
    cache_root = tmp_path / "mfa-cache"
    paths = MFABatchPaths(
        root=tmp_path / "mfa_alignment",
        corpus=tmp_path / "mfa_alignment" / "corpus",
        metadata=tmp_path / "mfa_alignment" / "metadata",
        output=tmp_path / "mfa_alignment" / "output",
        temporary=tmp_path / "mfa_alignment" / "temp",
    )
    for directory in (paths.corpus, paths.metadata, paths.output, paths.temporary):
        directory.mkdir(parents=True, exist_ok=True)

    calls: list[tuple[list[str], dict[str, str]]] = []

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs["env"]))
        stdout = "3.4.1\n" if command[-1] == "version" else "aligned\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    invocation = invoke_mfa_batch(
        paths=paths,
        prefix=prefix,
        cache_root=cache_root,
        micromamba="mock-micromamba",
        num_jobs=3,
        runner=runner,
    )

    align_calls = [command for command, _ in calls if "align_hf" in command]
    assert len(align_calls) == 1
    assert align_calls[0] == build_mfa_align_command(
        paths=paths,
        prefix=prefix,
        micromamba="mock-micromamba",
        num_jobs=3,
    )
    assert align_calls[0][5:8] == [
        "align_hf",
        str(paths.corpus.resolve()),
        MFA_MODEL_ID,
    ]
    assert align_calls[0][9:] == [
        "--use_g2p",
        "--no_tokenization",
        "--fine_tune",
        "--output_format",
        "json",
        "--no_textgrid_cleanup",
        "--temporary_directory",
        str(paths.temporary.resolve()),
        "--num_jobs",
        "3",
        "--clean",
        "--overwrite",
    ]
    assert len(calls) == 2
    assert all(
        environment["MFA_ROOT_DIR"] == str(cache_root.resolve())
        for _, environment in calls
    )
    assert all(
        environment["HF_HOME"] == str((cache_root / "huggingface").resolve())
        for _, environment in calls
    )
    assert invocation["mfa_version"] == MFA_VERSION
    assert invocation["fine_tune"] is True
    assert (
        read_json(paths.metadata / "mfa_invocation.json")["command"] == align_calls[0]
    )


def test_build_mfa_command_rejects_nonpositive_jobs(tmp_path: Path) -> None:
    paths = MFABatchPaths(
        root=tmp_path,
        corpus=tmp_path / "corpus",
        metadata=tmp_path / "metadata",
        output=tmp_path / "output",
        temporary=tmp_path / "temp",
    )

    with pytest.raises(ValueError, match="num_jobs must be positive"):
        build_mfa_align_command(paths=paths, num_jobs=0)
