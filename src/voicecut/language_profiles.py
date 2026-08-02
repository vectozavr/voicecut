"""Immutable language-specific configuration for VoiceCut.

Language profiles keep model selection and optional-feature policy in one
place.  They contain metadata only; importing this module never downloads or
loads a model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True, slots=True)
class LanguageProfile:
    """Configuration required by language-dependent pipeline components."""

    code: str
    display_name: str
    whisper_language: str
    whisperx_language: str
    whisperx_model: str | None
    mfa_model: str
    ctc_enrichment_supported: bool
    default_breath_cleanup: Literal["off", "replace"]


_LANGUAGE_PROFILES: Mapping[str, LanguageProfile] = MappingProxyType(
    {
        "en": LanguageProfile(
            code="en",
            display_name="English",
            whisper_language="en",
            whisperx_language="en",
            # Preserve WhisperX's existing documented English default instead
            # of duplicating an implementation-owned model identifier here.
            whisperx_model=None,
            mfa_model="english_us_arpa",
            ctc_enrichment_supported=True,
            default_breath_cleanup="replace",
        ),
        "ru": LanguageProfile(
            code="ru",
            display_name="Russian",
            whisper_language="ru",
            whisperx_language="ru",
            whisperx_model="jonatasgrosman/wav2vec2-large-xlsr-53-russian",
            mfa_model=(
                "MontrealCorpusTools/russian_mfa"
                "@88b81ae3eaf3bd8163bb3f7c43e1ae61478595af"
            ),
            # The existing hidden-retry CTC enrichment and Respiro-en policy
            # have not been validated for Russian.  Core transcription,
            # planning, completeness alignment, and MFA remain available.
            ctc_enrichment_supported=False,
            default_breath_cleanup="off",
        ),
    }
)

SUPPORTED_LANGUAGE_CODES: tuple[str, ...] = tuple(_LANGUAGE_PROFILES)


def get_language_profile(code: str) -> LanguageProfile:
    """Return the immutable profile for a supported language code."""

    normalized = str(code).strip().casefold()
    try:
        return _LANGUAGE_PROFILES[normalized]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_LANGUAGE_CODES)
        raise ValueError(
            f"unsupported language {code!r}; expected one of: {supported}"
        ) from exc
