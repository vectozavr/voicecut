from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest
import soundfile as sf

from voicecut.mfa_alignment import (
    DEFAULT_MFA_CACHE_ROOT,
    DEFAULT_MFA_PREFIX,
    MFA_MODEL_ID,
    MFA_VERSION,
    MFAAlignmentError,
    MFAContextSpec,
    MFASourceWord,
    align_mfa_contexts,
    source_word_alignment,
)


RUN_MFA_INTEGRATION = os.environ.get("VOICECUT_RUN_MFA_INTEGRATION") == "1"


def _require_opted_in_runtime() -> tuple[Path, Path, str]:
    if platform.system() != "Darwin":
        pytest.fail(
            "VOICECUT_RUN_MFA_INTEGRATION=1 requires macOS because this test "
            "creates its permitted narration fixture with the built-in 'say' "
            "command"
        )

    say = shutil.which("say")
    if say is None:
        pytest.fail(
            "VOICECUT_RUN_MFA_INTEGRATION=1 was set, but macOS 'say' was not found"
        )

    micromamba = shutil.which("micromamba")
    if micromamba is None:
        pytest.fail(
            "VOICECUT_RUN_MFA_INTEGRATION=1 was set, but micromamba was not "
            "found; run scripts/install.sh first"
        )

    prefix = DEFAULT_MFA_PREFIX.resolve()
    if not prefix.is_dir() or not (prefix / "bin" / "mfa").is_file():
        pytest.fail(
            "VOICECUT_RUN_MFA_INTEGRATION=1 was set, but the pinned repository "
            f"MFA environment is missing at {prefix}; run scripts/install.sh first"
        )

    return prefix, DEFAULT_MFA_CACHE_ROOT.resolve(), micromamba


@pytest.mark.skipif(
    not RUN_MFA_INTEGRATION,
    reason="set VOICECUT_RUN_MFA_INTEGRATION=1 to run the real MFA integration",
)
def test_real_pinned_mfa_aligns_runtime_generated_narration(
    tmp_path: Path,
) -> None:
    prefix, cache_root, micromamba = _require_opted_in_runtime()

    source_aiff = tmp_path / "permitted_runtime_narration.aiff"
    canonical_wav = tmp_path / "canonical_source.wav"
    narration = "Voice cut aligns familiar words clearly."
    try:
        subprocess.run(
            ["/usr/bin/say", "-o", str(source_aiff), narration],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.fail(f"macOS 'say' could not create the narration fixture: {exc}")

    try:
        waveform, sample_rate = sf.read(
            source_aiff,
            dtype="float32",
            always_2d=True,
        )
        sf.write(canonical_wav, waveform, sample_rate, format="WAV", subtype="FLOAT")
    except (OSError, RuntimeError) as exc:
        pytest.fail(f"could not convert the generated narration to WAV: {exc}")

    duration = len(waveform) / sample_rate
    source_texts = ("Voice", "cut", "aligns", "familiar", "words", "clearly.")
    words = tuple(
        MFASourceWord(
            word_id=word_id,
            text=text,
            start_seconds=duration * word_id / len(source_texts),
            end_seconds=duration * (word_id + 1) / len(source_texts),
            selected=True,
        )
        for word_id, text in enumerate(source_texts)
    )
    context = MFAContextSpec(
        context_id="context_000",
        crop_source_start_seconds=0.0,
        crop_source_end_seconds=duration,
        words=words,
        boundary_ids=("integration_boundary",),
    )

    try:
        result = align_mfa_contexts(
            audio_path=canonical_wav,
            contexts=(context,),
            work_dir=tmp_path / "mfa_work",
            prefix=prefix,
            cache_root=cache_root,
            micromamba=micromamba,
            num_jobs=1,
        )
    except MFAAlignmentError as exc:
        pytest.fail(
            "the explicitly enabled real MFA integration failed; verify the "
            f"pinned runtime/model installation and network access: {exc}"
        )

    assert result["backend"] == "mfa"
    assert result["mfa_version"] == MFA_VERSION == "3.4.1"
    assert result["model_id"] == MFA_MODEL_ID == "english_us_arpa"
    assert result["fine_tune"] is True
    assert len(result["contexts"]) == 1

    invocation = result["invocation"]
    assert invocation["mfa_version"] == "3.4.1"
    assert invocation["model_id"] == "english_us_arpa"
    assert invocation["fine_tune"] is True
    assert invocation["environment"]["MFA_ROOT_DIR"] == str(cache_root)
    assert invocation["command"].count("align_hf") == 1
    assert "--fine_tune" in invocation["command"]

    aligned_context = result["contexts"][0]
    assert aligned_context["context_id"] == "context_000"
    for source_word in words:
        aligned_word = source_word_alignment(aligned_context, source_word.word_id)
        assert aligned_word["phones"]
        assert any(not phone["is_silence"] for phone in aligned_word["phones"])
        assert aligned_word["first_non_silence_phone"]["start_sample"] >= 0
        assert aligned_word["last_non_silence_phone"]["end_sample"] <= len(waveform)
