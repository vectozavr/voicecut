# Contributing to VoiceCut

Thank you for helping improve VoiceCut. Contributions are welcome in code,
tests, documentation, reproducible bug reports, and publication-safe examples.
Please open or review an issue in the
[project issue tracker](https://github.com/vectozavr/voicecut/issues) before a
large change so parallel work does not diverge.

## Development setup

VoiceCut currently supports Apple Silicon macOS with Python 3.11 or 3.12. The
normal installer creates three repository-local runtimes without modifying
your shell initialization:

- `.venv` for the application, audio stack, WhisperX completeness veto, and
  cloud planner SDKs;
- `.venv-mlx` for MLX Whisper and optional local planners;
- `.mfa-env` for the pinned MFA 3.4.1 command-line runtime.

Install the project and then add the development dependencies:

```bash
./scripts/install.sh
source .venv/bin/activate
python -m pip install -e ".[audio,cloud,dev]"
.venv-mlx/bin/python -m pip install -e ".[mlx]"
```

The installer also verifies the pinned Respiro-en files and creates `.env`
from `.env.example` when needed. Unit tests do not require a planner API key.
Never commit `.env`, credentials, downloaded models, virtual environments, or
runtime caches.

## Run focused tests while developing

Run the smallest relevant test module first. Examples:

```bash
.venv/bin/python -m pytest -q tests/test_streaming_narration.py
.venv/bin/python -m pytest -q tests/test_mfa_alignment.py
.venv/bin/python -m pytest -q tests/test_final_render.py
.venv/bin/python -m pytest -q tests/test_video_render.py
```

You can target one regression directly:

```bash
.venv/bin/python -m pytest -q \
  tests/test_mfa_boundary_regressions.py::test_boundary_plan_rejects_fade_over_retained_mfa_phone
```

New unit tests must be deterministic, offline by default, and independent of
an existing developer cache. Mock cloud calls and expensive model subprocesses
or use small recorded JSON evidence where appropriate.

## Required checks

Before submitting a change, run the complete test and style suite from the
repository root:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
```

If formatting is the only failure, apply it deliberately and review the diff:

```bash
.venv/bin/ruff format src tests
```

Do not use a formatter or broad rewrite to alter unrelated files.

## Opt-in integration tests

Normal CI must not download MFA models or Respiro-en weights. Two real-runtime
checks are available to contributors who have completed the normal install:

```bash
VOICECUT_RUN_MFA_INTEGRATION=1 \
  .venv/bin/python -m pytest -q tests/test_mfa_integration.py

VOICECUT_RUN_BREATH_INTEGRATION=1 \
  .venv/bin/python -m pytest -q tests/test_breath_detection.py
```

The MFA test currently relies on macOS speech synthesis, FFmpeg, micromamba,
the repository-local `.mfa-env`, and model downloads or an existing model
cache. The breath test loads the pinned, hash-verified Respiro-en runtime.
Report which opt-in checks you ran, but do not commit their model caches,
temporary corpora, plots, or generated audio.

## Production architecture contract

Changes to the audio path must preserve these invariants:

1. The canonical source WAV remains untouched for semantic analysis, acoustic
   evidence, and rendering.
2. The semantic planner selects exact source word occurrences. It never
   selects sample coordinates.
3. WhisperX can veto a weak or incomplete retained occurrence, but its
   timestamps are not production cut coordinates.
4. MFA word and phone alignment is the sole authority for production source
   endpoints. Whisper timestamps and waveform measurements are approximate
   crop anchors or secondary evidence only.
5. Semantic pauses and optional Respiro-en breath cleanup may operate only in
   MFA-confirmed non-speech. Neither can move a resolved endpoint or modify a
   retained phone.
6. Every endpoint, pause, room-tone source, replacement, and permitted fade is
   frozen in one authoritative `final_boundary_plan.json` before rendering.
7. The renderer reads the canonical source and materializes that immutable
   plan exactly once. A production renderer must never consume a WAV written
   by another renderer.
8. Retained speech stays sample-identical to the canonical source. Every output
   sample must be traceable either to its canonical source coordinate or to
   verified source room tone used for an insertion or replacement.
9. An unresolved or invalid acoustic boundary must never fall back to a
   Whisper timestamp, RMS minimum, zero crossing, midpoint, or guessed fade.
   Use the existing conservative source-preservation behavior where it is
   structurally safe; otherwise stop before rendering.

Do not add a post-render cleanup stage to solve a boundary problem. Prefer a
pure evidence helper, a boundary-plan correction, or a stricter validation
inside the existing plan-building path. Optional debug previews must remain
outside the production dependency graph.

A change affecting cuts, fades, pauses, or replacements should include a
regression proving that retained MFA phones remain untouched, the boundary
plan cannot change after it is frozen, and only one production WAV render is
performed.

## Tests, fixtures, and media

Prefer synthetic waveforms and arrays generated inside `tmp_path` for unit
tests. Keep recorded JSON fixtures small and limited to the evidence needed by
the assertion. Tests should not depend on a personal recording, a moving model
revision, network access, or a prior VoiceCut run.

Audio and video can contain biometric, personal, copyrighted, or confidential
information. Do not add a recording to this public repository unless all of
the following are true:

- you have permission to publish and redistribute it;
- every identifiable speaker has consented to public distribution;
- its source, creator, recording method, and applicable license are documented;
- metadata that should not be public has been removed;
- the file is the smallest practical fixture or reviewed public example; and
- its inclusion has been explicitly reviewed as part of the contribution.

Generated or synthesized media still needs documented provenance and terms.
Never substitute a private user recording merely because it reproduces a bug.
Instead, create a minimal synthetic fixture, contribute non-audio evidence, or
describe private A/B paths in a local validation report.

The repository ignores media by default and narrowly allows known public
fixtures and examples. Do not broaden that allowlist to make arbitrary local
recordings trackable. In particular, do not commit:

- `.voicecut-cache/`, work directories, MFA corpora, model checkpoints, or
  virtual environments;
- edited outputs, debug plots, notebooks, or A/B excerpts generated during a
  private run;
- API responses containing private transcripts; or
- API keys, `.env`, tokens, machine-specific paths, or logs containing them.

When a cloud planner is used, transcript text and word IDs—not audio or
video—are sent to that provider. Still treat the transcript as potentially
sensitive and follow the provider's data-processing terms.

## Dependency and model changes

Do not add a model or processing backend only to mask a failing example. A new
runtime dependency must have a clear production role, primary upstream link,
license, version or immutable revision, cache strategy, failure policy, and
offline test coverage. Model weights may have terms different from their
software wrapper.

Update [`docs/credits.md`](docs/credits.md) whenever a major external component
is added or its role changes. If Respiro-en provenance changes, also update
[`docs/upstream/respiro-en.md`](docs/upstream/respiro-en.md), the installer
hashes, runtime verification, and tests in the same contribution.

## Submitting a change

Keep commits focused and avoid unrelated cleanup. In the pull request or issue,
include:

- the user-visible problem and the underlying cause;
- the production modules and invariants affected;
- focused and full commands run, with results;
- whether either opt-in integration test was run;
- relevant manifest evidence and concise before/after observations; and
- known limitations or cases that remain conservative.

For acoustic changes, listen to permitted local A/B outputs as well as checking
JSON invariants. Do not upload private A/B files to an issue or pull request.

By contributing, you agree that your contribution is distributed under
VoiceCut's [MIT License](LICENSE).
