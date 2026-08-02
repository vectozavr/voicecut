from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from voicecut.language_profiles import (
    SUPPORTED_LANGUAGE_CODES,
    get_language_profile,
)


def test_language_profiles_preserve_english_defaults() -> None:
    profile = get_language_profile(" EN ")

    assert profile.code == "en"
    assert profile.whisper_language == "en"
    assert profile.whisperx_language == "en"
    assert profile.whisperx_model is None
    assert profile.mfa_model == "english_us_arpa"
    assert profile.ctc_enrichment_supported is True
    assert profile.default_breath_cleanup == "replace"


def test_russian_profile_pins_language_models_and_safe_feature_defaults() -> None:
    profile = get_language_profile("ru")

    assert profile.whisper_language == "ru"
    assert profile.whisperx_language == "ru"
    assert profile.whisperx_model == "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
    assert profile.mfa_model == (
        "MontrealCorpusTools/russian_mfa@88b81ae3eaf3bd8163bb3f7c43e1ae61478595af"
    )
    assert profile.ctc_enrichment_supported is False
    assert profile.default_breath_cleanup == "off"


def test_language_profiles_are_immutable_and_reject_unknown_codes() -> None:
    assert SUPPORTED_LANGUAGE_CODES == ("en", "ru")
    profile = get_language_profile("ru")

    with pytest.raises(FrozenInstanceError):
        profile.code = "en"  # type: ignore[misc]
    with pytest.raises(ValueError, match="unsupported language"):
        get_language_profile("de")
