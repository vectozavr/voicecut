# Troubleshooting

This guide covers the production `voicecut` command installed by
[`scripts/install.sh`](../scripts/install.sh). Start with the short checklist,
then use the section that matches the stage named in the error.

For supported options, see [Configuration](configuration.md). For the artifact
flow and safety policy, see [Architecture](architecture.md).

## Quick checklist

From the repository root:

```bash
source .venv/bin/activate
voicecut --help
ffmpeg -version
ffprobe -version
micromamba --version
```

Verify the pinned MFA environment:

```bash
MFA_ROOT_DIR="$PWD/.voicecut-cache/runtime/mfa" \
HF_HOME="$PWD/.voicecut-cache/runtime/mfa/huggingface" \
  micromamba run -p "$PWD/.mfa-env" mfa version
```

The final command must report MFA `3.4.1`.

For a diagnosable run, use a dedicated work directory:

```bash
voicecut input.wav \
  -o input_edited.wav \
  --work-dir .voicecut-cache/input-debug \
  --debug-artifacts
```

Do not manually edit files inside that directory.

## `voicecut: command not found`

Activate the primary environment from the repository root:

```bash
source .venv/bin/activate
voicecut --help
```

If `.venv/bin/voicecut` does not exist, rerun:

```bash
./scripts/install.sh
source .venv/bin/activate
```

Do not invoke a guessed `.venv-audio/bin/voicecut` path. The supported layout
uses `.venv` for the public CLI and `.venv-mlx` as an internal helper.

## Missing Python modules

Errors such as `No module named voicecut`, `silero_vad`, `mlx_whisper`, or
`whisperx` usually mean the wrong interpreter was used or installation was
partial.

The public command should run from `.venv`; VoiceCut launches the configured
MLX and alignment interpreters automatically. Repair the complete layout with:

```bash
./scripts/install.sh
source .venv/bin/activate
```

If you intentionally override `--asr-python`, `--alignment-python`, or
`--planner-python`, confirm that the selected interpreter contains the required
package. Remove the override before diagnosing the standard installation.

## Unsupported machine or Python version

The installer currently requires Apple Silicon macOS and Python 3.11 or 3.12.
On a supported Mac, point the installer at a compatible Homebrew Python:

```bash
brew install python@3.12
VOICECUT_PYTHON=/opt/homebrew/bin/python3.12 ./scripts/install.sh
```

If a repository-local environment was built with another Python version,
remove only the affected `.venv` or `.venv-mlx` directory and rerun the
installer. Do not delete an unrelated system or user environment.

## Planner preflight failures

VoiceCut validates planner credentials and SDK availability before media work.
This prevents a long transcription from finishing only to discover a missing
API key.

### Missing API key

Add the selected provider key to `.env` or export it:

```dotenv
GEMINI_API_KEY=...
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
```

Confirm that `--env-file` points to the intended file and that the variable
matches `--planner-api-key-env` when using a custom name.

### Cloud SDK unavailable

Reinstall the cloud extra in the primary environment:

```bash
source .venv/bin/activate
python -m pip install -e ".[audio,cloud]"
```

### Invalid custom endpoint

`--planner-base-url` is accepted only for OpenAI and DeepSeek. The URL must be
HTTP(S), include a host, and contain no credentials, query, or fragment.
OpenAI overrides must implement the Responses API.

Never embed a key in the URL. VoiceCut rejects credential-bearing URLs and
redacts validated endpoints from subprocess error output.

### Local planner does not load

Confirm `.venv-mlx/bin/python` exists and the model fits available unified
memory. If `--local-files-only` is set, the complete model must already exist in
the Hugging Face cache or at the supplied local path.

Local semantic editing is experimental. When diagnosing edit quality, first
compare with a strong cloud planner in a separate work directory.

## MFA runtime problems

### `.mfa-env` missing or wrong version

Check the isolated runtime:

```bash
MFA_ROOT_DIR="$PWD/.voicecut-cache/runtime/mfa" \
HF_HOME="$PWD/.voicecut-cache/runtime/mfa/huggingface" \
  micromamba run -p "$PWD/.mfa-env" mfa version
```

If the prefix is missing or does not report `3.4.1`, remove only the
repository-local `.mfa-env` and rerun `./scripts/install.sh`. Do not install MFA
into either Python virtual environment.

### First alignment is slow or downloads a model

The first English run downloads `english_us_arpa`; the first Russian run
downloads the revision-pinned Russian MFA bundle. This is expected. Later runs
reuse the model cache selected by `--mfa-cache-root`.

Changing the cache root causes a separate model download. Ensure the machine
has network access and enough disk space.

### `mfa_word_mapping_failed` or alignment failure

MFA maps local chronological transcript tokens to words and phones. VoiceCut
refuses to use approximate Whisper timestamps when a required word cannot be
mapped, a phone is missing, or geometry is invalid.

The current pipeline first tries bounded semantic reselection and then
conservative source preservation across the unsafe cut. Check:

- `05_final/final_boundary_plan.json`;
- `05_final/acoustic_retries/`;
- `05_final/conservative_delivery_fallback/`;
- `05_final/mfa_alignment/metadata/` and `output/`;
- the final `delivery_status` printed by the CLI.

A locally preserved result can be safe but under-edited. A structurally invalid
global alignment can still fail closed. Do not work around the failure by
clamping Whisper timestamps or editing boundary JSON.

### Why there is no Whisper fallback

Whisper timestamps are preliminary crop anchors and can overlap or cut off
quiet final phones. WhisperX is a completeness veto, not a coordinate backend.
MFA phone geometry is the sole coordinate authority. Falling back to an RMS
minimum, midpoint, or Whisper timestamp would reintroduce clipped-word errors.

## Russian-language runs

Russian must be selected explicitly:

```bash
voicecut input_ru.wav --language ru
```

Expected differences from English:

- hidden-retry CTC is reported as `skipped_unsupported_language`;
- the Russian WhisperX completeness model may download on first use;
- the revision-pinned Russian MFA model may download on first use;
- breath cleanup defaults to off because Respiro-en is not validated for
  Russian.

The CTC skip is informational, not an error. Do not force the English CTC model
onto a Russian transcript.

If Cyrillic narration was accidentally processed as English, use a new work
directory with `--language ru`. One explicit work directory cannot be reused
with a different language configuration.

## Respiro-en and breath cleanup

### Pinned runtime missing or hash mismatch

Run `./scripts/install.sh`. The installer downloads the exact pinned
`modules.py`, checkpoint, and license and checks their SHA-256 hashes. VoiceCut
does not download from a moving branch during execution.

Do not replace the files manually. Provenance is documented in
[the Respiro-en upstream record](upstream/respiro-en.md).

### Detector failure warning

Breath cleanup is optional. When detector loading or inference fails, VoiceCut
keeps the otherwise valid MFA edit, preserves original gap content, records
`breath_cleanup_skipped_detector_failure`, and continues. It does not substitute
an RMS, VAD, or generic sound-event detector.

To make the intended policy explicit while diagnosing:

```bash
voicecut input.wav --breath-cleanup off
```

With cleanup off, VoiceCut does not source inserted ambience from unscreened
audio. Requested pause extensions may be skipped; existing retained gaps remain
untouched.

### Breath remains in the result

A detected event is intentionally left unchanged when it overlaps an MFA
retained phone or when no verified clean ambience is available. Inspect the
event status in `final_boundary_plan.json`. This protects final fricatives and
speech at the cost of conservative cleanup.

With `--debug-artifacts`, event plots and excerpts are written below
`05_final/breath_debug/`. Listen to the full result; a probability plot alone is
not a quality judgment.

### `clean_ambience_unavailable`

This status means no source candidate passed the clean-ambience requirements.
VoiceCut does not use a breath-containing or transient candidate merely because
it has low RMS, and it does not use digital zero as normal room tone.

The affected pause may remain unchanged or receive no requested extension. The
status is safer than introducing a repeated breath, click, or room-tone loop.

## Output already exists

VoiceCut does not replace an existing destination by default:

```text
... already exists; pass --overwrite to replace it
```

Either choose a new destination or explicitly allow replacement:

```bash
voicecut input.wav -o output.wav --overwrite
```

Input and output must differ. A symbolic-link output is refused, and input or
output paths may not overlap the work-directory tree.

## Cache and work-directory errors

### Work directory belongs to another configuration

An explicit work directory is immutable with respect to input and production
configuration. Use a different directory when changing the source, language,
planner/model, alignment settings, or renderer policy:

```bash
voicecut input.wav \
  --work-dir .voicecut-cache/input-new-model \
  --planner-model NEW_MODEL
```

Do not modify `pipeline_config.json` to force reuse. It protects against stale
semantic plans and renders.

### Another process owns the cache

VoiceCut uses a non-blocking lock per work directory. Wait for the owner to
finish or select another directory. Do not run two processes against the same
cache.

If no process remains after an abnormal termination, a new process can acquire
the operating-system lock; the presence of the lock file itself is not the
lock.

### Resume an interrupted long recording

Rerun the exact same command with the same work directory. VoiceCut validates
and reuses completed stages. Whisper and the optional English CTC worker retain
validated per-region checkpoints, so the run can continue without starting at
the first region.

If configuration or implementation changed, use the new content-addressed
default or select a fresh work directory. A software fingerprint deliberately
prevents old output from being treated as current.

### Output appears unchanged

Check the printed `delivery status` and `pipeline_run.json`.

- `complete` means the accepted edit was rendered.
- `complete_with_preserved_source_context` means one or more unsafe cuts were
  removed by retaining nearby source audio. Small abandoned attempts may remain.
- `complete_with_full_source_passthrough` means the original semantic plan
  selected the full source and no safe internal cut plan could be completed;
  VoiceCut returned the canonical source rather than guessing.
- `complete_with_deterministic_pauses` means audio pause classification used a
  deterministic local fallback for one or more failed batches.

The CLI also reports preserved problem intervals and fallback counts. These are
not cache bugs; they are explicit safe-delivery outcomes.

## Long recordings and planner failures

VoiceCut handles planner failures per streaming window. After one corrective
retry, an invalid or unsupported response causes only that local source window
to be preserved. Later windows continue through normal editing, and unresolved
EOF text is preserved.

Relevant artifacts:

- `04_semantic_plan/iteration_*.json` for accepted decisions;
- `04_semantic_plan/iteration_*_attempt_*.raw.*` for raw responses;
- `04_semantic_plan/grounding_validation.json`;
- `04_semantic_plan/streaming_plan.json` and its `fallbacks` list;
- `05_final/final_render_manifest.json` for aggregate fallback counts.

This policy is intended to produce a usable, possibly under-edited long result
instead of discarding the entire edit because of one malformed local response.

Initial transcript identity errors, duplicate IDs, and irreconcilable source
grounding are structural failures and remain fatal.

## Awkward or clipped-sounding cuts

Do not diagnose a clipped word from the waveform alone. Determine which layer
made the decision:

1. `04_semantic_plan/streaming_plan.json` shows whether the complete source word
   occurrence was selected.
2. `04_semantic_plan/grounding_validation.json` shows whether canonical text is
   supported inside that range.
3. `05_final/completeness_worker_result.json` shows retained-word completeness
   evidence.
4. `05_final/mfa_alignment/metadata/mfa_alignment.json` shows MFA word/phone
   mapping.
5. `05_final/final_boundary_plan.json` shows the one final source sample
   decision and protected phone spans.

The renderer is prohibited from moving an MFA endpoint after the plan is
frozen. A bad semantic occurrence, a weak-word veto, an MFA mapping failure,
and a publication issue require different fixes; changing global padding can
damage otherwise correct words.

## Video publication problems

Video output requires a video input. Output formats are `.mkv`, `.mov`, `.mp4`,
and `.webm`.

The video renderer uses only selected source intervals with direct visual cuts.
It does not insert semantic pause time or frozen frames. Publication validates
the stream type, output duration, and audio/video synchronization.

If stage `06 MEDIA PUBLICATION` fails, inspect:

- `00_media/media_input.json` for selected source streams and offsets;
- `05_final/final_render_manifest.json` for the source interval timeline;
- `06_publication/video_render_manifest.json` when it was created;
- FFmpeg/FFprobe availability and codec support.

Converting a MOV input to MP4 is not required for processing. Choose `.mp4` as
the output when browser-compatible H.264/AAC delivery is desired.

## Unsupported input or output format

VoiceCut probes the input with FFmpeg, so some additional readable input
containers may work. Published output extensions are restricted.

Supported audio outputs:

```text
.aac .aif .aiff .flac .m4a .mp3 .oga .ogg .opus .wav
```

Supported video outputs:

```text
.mkv .mov .mp4 .webm
```

Choose the output container through `-o`. A video output cannot be created from
an audio-only input.

## Inspect a run

The most useful top-level files are:

| File | What it answers |
| --- | --- |
| `pipeline_config.json` | Which exact input, models, language, and settings own this cache? |
| `pipeline_run.json` | Which stages were created or reused, and what delivery warnings occurred? |
| `04_semantic_plan/streaming_plan.json` | Which word occurrences did the planner keep? |
| `04_semantic_plan/grounding_validation.json` | Did canonical text have source support? |
| `05_final/pause_plan.json` | How were audio thought transitions classified? |
| `05_final/final_boundary_plan.json` | Where are all final cuts, protected phones, pauses, and replacements? |
| `05_final/final_render_manifest.json` | What was rendered, and with which delivery status? |
| `06_publication/publication.json` | Which final media file was validated and published? |

Use `--debug-artifacts` before the run when probability plots and short audio
diagnostics are needed. Debug output is never consumed by production.

## Reporting an issue

Include:

- the VoiceCut commit;
- macOS, Python, FFmpeg, micromamba, and MFA versions;
- the exact command with secrets removed;
- `pipeline_config.json`, `pipeline_run.json`, and relevant manifests;
- the delivery status and stage named in the error;
- a short publication-safe source excerpt only when you have permission to
  share it.

Do not publish `.env`, provider keys, full private recordings, model weights,
or an entire work directory. Planner raw responses can contain transcript text,
so review them before sharing.
