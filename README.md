# VoiceCut

VoiceCut turns a retake-heavy spoken recording into coherent narration. It
transcribes the recording once, asks a selected language model which source
word occurrences belong in the intended take, validates that decision against
the source transcript, aligns retained boundaries to phones with Montreal
Forced Aligner (MFA), optionally replaces breaths only inside MFA-confirmed
non-speech, and renders the accepted edit once from the source.

It accepts audio or video. Video is edited from its speech track: selected
audio intervals select the corresponding picture intervals and are joined with
direct visual cuts. VoiceCut does not add artificial semantic pauses or frozen
frames to video, and it does not interpret the visual content.

VoiceCut never synthesizes speech, replaces words, denoises the recording, or
adds generated background noise. Retained voice samples come directly from the
input. Optional breath cleanup uses verified quiet material copied from the
same recording and preserves the duration of the edited pause.

## Supported system

- Apple Silicon Mac
- macOS
- Python 3.11 or 3.12
- FFmpeg and FFprobe
- micromamba (installed by the repository installer through Homebrew when
  absent)

The first transcription downloads the selected Whisper model. The first
alignment run downloads the `english_us_arpa` MFA model into VoiceCut's
persistent cache at `.voicecut-cache/runtime/mfa`. The installer downloads the
pinned Respiro-en implementation and checkpoint into
`.voicecut-cache/runtime/respiro-en`, verifies their SHA-256 hashes, and never
downloads from a moving branch during normal execution. A local language model
is downloaded only when a local planner backend is selected. Model files can
require several gigabytes of disk space.

## Install

The supported distribution is a source checkout installed by the repository
installer. A bare wheel or `pip install voicecut` does not create the separate
MLX and pinned MFA helper runtimes required by the pipeline.

Clone the repository and run the installer:

```bash
git clone https://github.com/vectozavr/voicecut.git
cd voicecut
./scripts/install.sh
source .venv/bin/activate
```

The installer:

1. verifies Apple Silicon and Python;
2. installs FFmpeg and micromamba with Homebrew when they are missing;
3. creates the repository-local `.mfa-env` from `environment-mfa.yml` and
   verifies that it contains exactly MFA 3.4.1;
4. creates the primary `.venv` and an internal `.venv-mlx` helper environment;
5. downloads and hash-verifies the pinned Respiro-en implementation, checkpoint,
   and MIT license into VoiceCut's runtime cache;
6. installs VoiceCut, audio dependencies, the WhisperX completeness-veto
   runtime, and cloud SDKs in `.venv`;
7. installs MLX Whisper and local-planner dependencies in `.venv-mlx`;
8. creates an empty `.env` from `.env.example` without overwriting an existing
   file and restricts it to the current user (`0600`).

You invoke only the `voicecut` command from `.venv`; VoiceCut launches both
helper runtimes automatically. `.venv-mlx` is separate because current
`mlx-lm` requires Transformers 5 and Hugging Face Hub 1.x, while WhisperX
requires Hugging Face Hub below 1. `.mfa-env` separately pins the Conda/Kaldi
stack required by MFA 3.4.1. The installer does not initialize or modify your
shell for micromamba.

To use a different Python installation:

```bash
VOICECUT_PYTHON=/opt/homebrew/bin/python3.12 ./scripts/install.sh
```

Manual installation is also supported:

```bash
brew install python@3.12 ffmpeg micromamba

micromamba create -y \
  -p "$PWD/.mfa-env" \
  -f "$PWD/environment-mfa.yml"
mkdir -p "$PWD/.voicecut-cache/runtime/mfa/huggingface"
MFA_ROOT_DIR="$PWD/.voicecut-cache/runtime/mfa" \
HF_HOME="$PWD/.voicecut-cache/runtime/mfa/huggingface" \
  micromamba run -p "$PWD/.mfa-env" mfa version

python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e ".[audio,cloud]"

python3.12 -m venv .venv-mlx
.venv-mlx/bin/python -m pip install --upgrade pip setuptools wheel
.venv-mlx/bin/python -m pip install -e ".[mlx]"

source .venv/bin/activate
[[ -e .env ]] || install -m 600 .env.example .env
chmod 600 .env
```

The version command in the manual installation must report exactly `3.4.1`.
VoiceCut invokes MFA only through `micromamba run -p .mfa-env`; it does not
import undocumented MFA Python internals into either virtual environment.

Cloud SDKs are optional when only local planning is needed:

```bash
.venv/bin/python -m pip install -e ".[audio]"
.venv-mlx/bin/python -m pip install -e ".[mlx]"
```

The `.mfa-env` runtime is required for production cuts regardless of which
semantic planner backend is selected.

The verified Respiro-en files are required only for optional breath cleanup.
If they are unavailable or inference fails, VoiceCut keeps the valid MFA edit,
preserves the original pause content, records the detector failure in the final
manifest, and prints a warning. It never substitutes an RMS or VAD breath
heuristic.

## Quick start

Edit a WAV with Gemini:

```bash
voicecut recording.wav \
  -o recording_edited.wav \
  --planner-backend gemini
```

Breath replacement is enabled by default. Disable it for a direct comparison:

```bash
voicecut recording.wav \
  -o recording_without_breath_cleanup.wav \
  --planner-backend gemini \
  --breath-cleanup off
```

The local-backend transport is also available for later evaluation:

```bash
voicecut recording.mp3 \
  -o recording_edited.mp3 \
  --planner-backend qwen
```

Edit a video based on its narration:

```bash
voicecut presentation.mp4 \
  -o presentation_edited.mp4 \
  --planner-backend gemini
```

If `-o` is omitted, VoiceCut writes `<input_stem>_edited` using the same
container type:

```bash
voicecut recording.m4a --planner-backend gemini
# recording_edited.m4a
```

An existing output is never replaced unless `--overwrite` is supplied.

## Planner backends

The planner receives transcript text and timestamped word IDs, not the source
audio or video. The renderer remains local for every backend.

| Backend | Default model | Credential |
| --- | --- | --- |
| `gemini` | `gemini-3.6-flash` | `GEMINI_API_KEY` |
| `openai` | `gpt-5-mini` | `OPENAI_API_KEY` |
| `deepseek` | `deepseek-v4-flash` | `DEEPSEEK_API_KEY` |
| `qwen` | `mlx-community/Qwen3.5-9B-4bit` | none |
| `gemma` | `mlx-community/gemma-3-12b-it-4bit` | none |
| `local` | `mlx-community/Qwen3.5-9B-4bit` | none |

### Cloud providers

Copy `.env.example` to `.env` and set only the key you use:

```dotenv
GEMINI_API_KEY=...
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
```

Then select the provider:

```bash
voicecut input.wav -o output.wav \
  --planner-backend openai \
  --planner-model gpt-5-mini
```

```bash
voicecut input.wav -o output.wav \
  --planner-backend deepseek \
  --planner-model deepseek-v4-flash
```

VoiceCut uses the Google Gen AI structured-output API for Gemini, the OpenAI
Responses API with a strict JSON schema for OpenAI, and the
OpenAI-compatible JSON-object API for DeepSeek. All responses pass the same
local ID, ordering, non-overlap, commitment, and source-grounding validators.
Malformed or ungrounded output is retried once with the validation error; a
second failure stops before rendering.

Use a different secret-variable name without putting the key on the command
line:

```bash
voicecut input.wav -o output.wav \
  --planner-backend openai \
  --planner-api-key-env COMPANY_OPENAI_KEY
```

`--planner-base-url` overrides the selected provider endpoint. For OpenAI, the
custom endpoint must implement the Responses API. DeepSeek defaults to
`https://api.deepseek.com`. Base URLs must not contain credentials, query
parameters, or fragments; keep secrets in the selected environment variable.

### Local models

`qwen` and `gemma` are convenience aliases for MLX model repositories:

```bash
voicecut input.wav -o output.wav --planner-backend qwen
voicecut input.wav -o output.wav --planner-backend gemma
```

Use `local` to run another MLX-LM-compatible Hugging Face repository or local
model directory:

```bash
voicecut input.wav -o output.wav \
  --planner-backend local \
  --planner-model mlx-community/Qwen3.5-27B-4bit
```

After a model is already in the Hugging Face cache, `--local-files-only`
prevents network downloads:

```bash
voicecut input.wav -o output.wav \
  --planner-backend local \
  --planner-model /absolute/path/to/model \
  --local-files-only
```

Local planner support is experimental: the loading and structured-response
transport are implemented, but its semantic editing quality has not been
production-validated. Use a strong cloud API model for reliable editing today.
Larger local models use more unified memory and take longer. Changing models
cannot bypass VoiceCut's deterministic source-grounding checks.

## Media formats

FFmpeg decodes the input to a lossless internal float WAV. The analysis and
renderer therefore use one canonical signal regardless of the source
container.

Common audio inputs include:

```text
AAC, AIFF, FLAC, M4A, MP3, OGG, Opus, WAV
```

Audio outputs:

```text
.aac .aif .aiff .flac .m4a .mp3 .oga .ogg .opus .wav
```

Video inputs and outputs:

```text
.mkv .mov .mp4 .webm
```

Other FFmpeg-readable inputs may work, but output extensions are intentionally
restricted to the lists above. Audio output is encoded once after editing.
Video output is re-encoded because its visual timeline must follow the audio
cuts.

For video:

- retained speech keeps the corresponding source pictures at normal speed;
- removed speech removes the corresponding pictures;
- selected source intervals are joined with direct cuts;
- no semantic pause time or frozen-frame hold is added;
- natural pauses already present inside a retained continuous take remain;
- the final edited audio replaces the original audio track;
- VoiceCut checks output audio/video duration and synchronization.

This is suitable for talking-head footage, narrated screen recordings, and
other material where speech determines the edit. It is not a visual continuity
editor.

## Work directory and caching

Use `--work-dir` to keep the expensive intermediate artifacts:

```bash
voicecut interview.mov \
  -o interview_edited.mov \
  --work-dir .voicecut-cache/interview \
  --planner-backend gemini
```

The work directory contains the canonical input WAV, analysis, word transcript,
acoustic retry evidence, streaming semantic plan, grounding report, local
WhisperX completeness-veto crops, the batched MFA corpus and word/phone
alignment JSON, a semantic audio-pause or deterministic video-cut transition
plan, one authoritative
`final_boundary_plan.json`, Respiro-en frame probabilities for relevant source
regions, breath-replacement provenance, and one final internal WAV. MFA context
WAVs are crops from the canonical source used only as alignment input; they are
not rendered narration. No rendered preview or breath-cleaned WAV feeds another
production step.
Completed stages are reused only when their input hashes, relevant model
settings, and VoiceCut implementation fingerprint match. A software update
therefore cannot silently reuse a plan or render produced by older code.

Legacy rough, trailing, hard-boundary, leading-boundary, and semantic-pause WAV
renderers remain available to developers as isolated preview helpers. They are
not in the one-command production call graph. `--debug-artifacts` may request
extra diagnostics when a renderer supports them, but it never changes that
single-pass production graph.

This has two important effects:

- rerendering the same accepted plan does not call the semantic planner again;
- changing the source file, planner backend, or model cannot silently reuse an
  incompatible plan.

Treat the work directory as an inspectable cache. Do not manually edit its JSON
or WAV files, and do not launch two processes against the same work directory;
VoiceCut rejects a concurrent owner. Use a different directory when comparing
models:

```bash
voicecut take.wav -o take_gemini.wav \
  --work-dir .voicecut-cache/take-gemini \
  --planner-backend gemini

voicecut take.wav -o take_qwen.wav \
  --work-dir .voicecut-cache/take-qwen \
  --planner-backend qwen
```

## How the pipeline works

### 1. Media preparation

FFprobe selects the real audio stream and distinguishes audio from video using
stream metadata. FFmpeg decodes that stream to a float PCM WAV without
normalization or denoising. The source and canonical-audio hashes are recorded.

### 2. Speech analysis

Silero VAD and waveform features divide long recordings into manageable speech
regions. These regions are processing units, not semantic sentences or edit
boundaries.

### 3. One primary transcript

MLX Whisper produces one chronological, word-level transcript. Every
occurrence receives an immutable integer ID plus approximate `start` and `end`
timestamps. Whisper text is evidence for semantic planning, while its
timestamps are only preliminary anchors.

### 4. Hidden-retry recovery

Language-model ASR can normalize away a spoken restart. A gated raw CTC pass
looks for high-confidence acoustic words hidden inside suspicious transcript
geometry. It expands only proven insertions, so a phrase such as
`with familiar, with the familiar words` remains visible to the planner rather
than collapsing to one apparent take.

### 5. Streaming semantic planning

The selected LLM sees the unresolved transcript suffix plus new look-ahead
words. It decides which exact source occurrences form complete intended
thoughts and leaves the newest thought pending until later speech proves that
the speaker moved on. This one-thought delay lets a later complete retake
replace an earlier attempt.

The model returns inclusive first/last word IDs and canonical text for each
selected source range. VoiceCut then verifies:

- every ID exists;
- ranges are ordered and non-overlapping;
- committed ranges never move backward or change;
- canonical content is supported inside its declared source ranges;
- every spoken source token inside a selected range is represented by its
  canonical phrase, so unwanted speech cannot be hidden inside a retained
  interval;
- finalized thoughts end before the pending suffix;
- a model cannot introduce text without corresponding source audio.

The LLM selects occurrences; it never selects sample coordinates.

### 6. One authoritative boundary plan and one render

VoiceCut first resolves every real source discontinuity without rendering any
output audio. WhisperX remains in production only as a retained-word
completeness veto: its character coverage and edge scores can reject an
incomplete occurrence as `weak_retained_word_alignment`, but no WhisperX
timestamp is allowed to become a final cut coordinate. Weak word occurrences
and rejected source edges remain forbidden during the bounded, source-grounded
semantic repair loop. If no acceptable selection is found within the retry
limit, the run stops before rendering.

After the completeness veto passes, VoiceCut creates all local alignment
contexts from the canonical source WAV. Each context contains the actual
chronological source words—including retained and omitted attempts—not the
LLM's polished narration. Whisper timestamps are used only as approximate
anchors for generous local crops. All contexts for a render attempt are sent
through one batched MFA 3.4.1 CLI invocation (the variables below are paths
inside the current VoiceCut run):

```bash
MFA_ROOT_DIR="$MFA_CACHE_ROOT" \
HF_HOME="$MFA_CACHE_ROOT/huggingface" \
micromamba run -p "$REPO_ROOT/.mfa-env" \
  mfa align_hf \
  "$CORPUS_DIR" \
  english_us_arpa \
  "$OUTPUT_DIR" \
  --use_g2p \
  --no_tokenization \
  --fine_tune \
  --output_format json \
  --no_textgrid_cleanup \
  --temporary_directory "$TEMP_DIR" \
  --num_jobs "$MFA_NUM_JOBS" \
  --clean \
  --overwrite
```

MFA word and phone intervals are mapped back to the ordered source word IDs.
The final non-silence phone of the retained word on the left and the first
non-silence phone of the retained word on the right define the protected speech
edges and authoritative source sample coordinates. Missing or ambiguous word
mapping, missing phones, invalid phone geometry, or an endpoint inside a
retained phone invalidates that cut. VoiceCut never falls back to a Whisper
timestamp. After bounded semantic repair is exhausted, a locally unresolved
cut is removed by conservatively preserving the original contiguous source
context across it. The WAV is still delivered, with
`delivery_status=complete_with_preserved_source_context` and the exact affected
word intervals in the manifest; that small region may retain an abandoned
attempt, but no guessed coordinate can clip speech.

When MFA confirms a phone-free or silence-phone interval, waveform energy and
zero crossings may choose a convenient splice only inside that interval. They
cannot decide that speech has ended or move a cut into a retained phone. A
dense word-to-word boundary does not require silence: if both retained words
are complete and their MFA phone geometry is valid, VoiceCut uses
`mfa_dense_phone_boundary` at the phone edge without fading either retained
phone. Fades are allowed only in MFA-confirmed non-speech, so retained speech,
including quiet final fricatives such as `/s/`, remains copied directly from
the canonical WAV.

For audio output, a separate semantic pause classification assigns
`continuation`, `short`, `thought`, or `section`. Existing natural quiet counts
toward the target total gap. Any deficit is filled with verified room tone only
at an MFA-resolved join, or inside an MFA-confirmed inter-word non-speech
interval within a contiguous take. If there is no confirmed internal gap, the
extra pause is skipped.

Video output automatically uses the `cuts` pause policy. It skips the semantic
pause model call, assigns zero inserted duration to every join, and joins only
the selected source-motion intervals. Natural timing inside each retained
continuous take is preserved, but no room-tone extension or frame hold is
created.

After pause planning, the pinned official Respiro-en model analyzes only source
regions that can appear in the output or supply room tone. It produces one
breath probability every 10 ms from a mono 16 kHz inference copy; the canonical
source is never resampled for rendering. Detected events may be replaced only
inside retained, MFA-confirmed non-speech. A 30 ms event guard and transitions
are clamped to that editable region. A detection overlapping a retained MFA
phone is left unchanged, including possible false positives over quiet final
`/s/`, `/z/`, `/f/`, `/th/`, `/sh/`, and `/tion/` phones. Breath-positive
source is also excluded from room-tone candidates. Replacement uses clean,
traceable room tone from the source and preserves the exact pause duration;
the selected pause policy and every MFA endpoint remain unchanged.

At EOF, the complete final MFA phone is protected before the existing safe
tail is retained and any fade can begin.

Only after all boundaries, pauses, breath decisions, room-tone source ranges,
and fade intervals have been frozen in `final_boundary_plan.json` does
`final_render.py` slice the canonical source WAV. It performs one render and
writes one internal `final_cut.wav`; no later stage may move or fade a boundary.
Every replaced sample is traceable to a verified source room-tone sample.

### 7. Publication

The validated internal WAV is encoded to the requested audio container. For
video, the same direct-cut source interval timeline is applied to the pictures
and the edited audio is muxed back in. Publication validates the output stream
type, duration, and synchronization before replacing the destination.

## CLI reference

```text
voicecut INPUT [options]
```

Important options:

| Option | Meaning |
| --- | --- |
| `-o`, `--output PATH` | Final audio/video path; defaults to `<stem>_edited.<ext>` |
| `--work-dir PATH` | Persistent intermediate artifacts and cache |
| `--overwrite` | Permit replacing an existing final output |
| `--planner-backend NAME` | `gemini`, `openai`, `deepseek`, `local`, `qwen`, or `gemma` |
| `--planner-model NAME` | Provider model name, Hugging Face repository, or local path |
| `--planner-base-url URL` | Override a cloud provider endpoint |
| `--planner-api-key-env NAME` | Environment variable containing the provider key |
| `--env-file PATH` | dotenv file; defaults to `.env` |
| `--local-files-only` | Do not download a local planner model |
| `--language en` | Source language; currently English |
| `--whisper-model NAME` | Override the MLX Whisper repository |
| `--alignment-backend mfa` | Production cut-coordinate backend; MFA is mandatory and the default |
| `--mfa-prefix PATH` | Repository-local micromamba prefix containing MFA 3.4.1; defaults to `.mfa-env` |
| `--mfa-cache-root PATH` | Persistent MFA model/cache directory passed as `MFA_ROOT_DIR`; defaults to `.voicecut-cache/runtime/mfa` |
| `--mfa-micromamba PATH` | micromamba executable used to run the pinned MFA prefix |
| `--mfa-num-jobs N` | Parallel jobs inside the one batched `mfa align_hf` invocation |
| `--breath-cleanup off\|replace` | Optional Respiro-en cleanup; defaults to `replace` |
| `--breath-threshold FLOAT` | Frame probability threshold; conservative default `0.5` |
| `--breath-min-duration-ms N` | Minimum consecutive positive duration; default `80` ms |
| `--respiro-cache-root PATH` | Verified pinned Respiro-en implementation/model cache |
| `--window-seconds N` | New transcript look-ahead added per planner iteration |
| `--max-output-tokens N` | Maximum structured planner response size |
| `--max-acoustic-retries N` | Bounded planner reselections after an acoustic failure or weak retained-word occurrence; defaults to 3 |
| `--debug-artifacts` | Request optional diagnostics without changing the single-pass render graph |
| `--asr-python PATH` | Advanced: Python executable for MLX ASR/local CTC stages |
| `--alignment-python PATH` | Advanced: Python executable containing WhisperX for the completeness veto only; it never supplies cut coordinates |
| `--planner-python PATH` | Advanced: Python executable for local MLX LLMs |

Run `voicecut --help` for the authoritative option list.

## Breath-cleanup behavior

VoiceCut pins the official [Respiro-en](https://github.com/ydqmkkx/Respiro-en)
implementation at commit
`70e01c60c2f582c41092730680f2894ab24d6467`. Exact file hashes, licensing, and
the paper citation are recorded in
[`docs/upstream/respiro-en.md`](docs/upstream/respiro-en.md). Runtime code
rechecks all three hashes before importing the model definition or loading the
checkpoint.

The production defaults are `--breath-threshold 0.5` and
`--breath-min-duration-ms 80`. They were selected conservatively after running
the combinations `0.064`, `0.2`, `0.5`, and `0.9` with minimum durations of
40, 80, and 120 ms on real VoiceCut recordings. They are detector settings,
not speech-safety thresholds: MFA phone protection is the non-negotiable safety
authority at every setting. Raising the threshold can split one physical event
into smaller detections, so a numerically stricter threshold does not
necessarily mean fewer replacements.

For inspection, run with `--debug-artifacts`. VoiceCut saves a probability plot
and short source/cleaned/room-tone excerpts for each detected event beneath the
run's `05_final/breath_debug/` directory. These are diagnostics created after
the frozen one-pass render; they are never inputs to production output. Human
listening remains required. Respiro-en was trained for frame-wise breath
detection, and its paper notes that exhalations are comparatively rare and does
not classify inhalation versus exhalation, so long sighs may be missed.

## MFA troubleshooting

Check the isolated runtime without activating or initializing micromamba:

```bash
MFA_ROOT_DIR="$PWD/.voicecut-cache/runtime/mfa" \
HF_HOME="$PWD/.voicecut-cache/runtime/mfa/huggingface" \
  micromamba run -p "$PWD/.mfa-env" mfa version
```

The command must report exactly `3.4.1`. If `.mfa-env` is missing or contains a
different version, remove that repository-local environment and rerun
`./scripts/install.sh`; do not install MFA into `.venv` or `.venv-mlx`.

The first real alignment needs network access to download `english_us_arpa`.
Later runs reuse the persistent directory selected by `--mfa-cache-root`.
Changing the cache root can therefore cause a new model download. The work
directory's `mfa_alignment/` artifacts and `final_boundary_plan.json` contain
the batch contexts, mapped word/phone intervals, final coordinates, and safety
statuses needed to diagnose a failed run.

VoiceCut does not automatically switch to WhisperX coordinates, Whisper
timestamps, an RMS minimum, or another aligner. A local MFA mapping or
phone-geometry failure first receives bounded semantic repair; if it remains
unresolved, VoiceCut preserves the original source context across that cut and
publishes a clearly marked degraded result. A global MFA runtime failure, an
ungrounded initial semantic plan, or a failure that cannot be removed without
inventing coordinates remains fatal. Use a fresh `--work-dir` after correcting
such a runtime, transcript, or input problem.

## Privacy

All waveform analysis, transcription, forced alignment, and rendering happen
locally. Local planner backends keep transcript text local as well.

When a cloud planner is selected, VoiceCut sends transcript text, word IDs, and
canonical thought text to that provider. It does not upload the audio or video.
The provider's own data-processing terms still apply to the submitted text.

API keys are read from the environment or `.env`; they are not written into
run manifests.

## Limitations

- English is the currently supported transcription/alignment language.
- The supported production platform is Apple Silicon macOS.
- Semantic editing is probabilistic. Inspect and listen to the result before
  publication, especially with unusual names, heavy accents, overlapping
  speakers, or severe recording damage.
- VoiceCut is intended for one primary narrator. Crosstalk and interviews with
  competing speakers are not yet diarized.
- Optional breath cleanup is conservative and may leave a detected event when
  MFA cannot separate it safely from retained speech or when clean source room
  tone is unavailable. Long sighs and exhalations may be missed.
- Video editing follows speech only and does not understand shots, gestures,
  slides, captions, or visual continuity.
- Audio/video output is re-encoded in common delivery codecs; VoiceCut is not a
  lossless container remuxer.
- A locally unresolved cut preserves its original source context and is listed
  in the final manifest instead of preventing publication. Global alignment or
  initial grounding failures still require a corrected run; VoiceCut never
  guesses an unsafe boundary.
- This is an editor, not a denoiser, mastering suite, speech generator, or
  voice-cloning system.

## Development

Run `./scripts/install.sh` first so the pinned MFA runtime exists, then install
development and cloud dependencies:

```bash
source .venv/bin/activate
python -m pip install -e ".[audio,cloud,dev]"
.venv-mlx/bin/python -m pip install -e ".[mlx]"
```

Run the test and style suites:

```bash
pytest -q
ruff check src tests
ruff format --check src tests
```

The tests include semantic validation and retry behavior, source grounding,
MFA phone-protected `/s/` and leading-word regressions, dense phone-to-phone
boundaries, overlapping Whisper anchors, confidence-aware weak-word rejection,
fail-closed MFA mapping and geometry errors, semantic audio pauses, clear-cut
video timelines without frame holds, EOF tails,
sample-trace invariants, media conversion, video timeline construction,
caching, one-command orchestration, frame-wise breath geometry, MFA-protected
breath replacement, exact-duration room-tone substitution, detector failure,
and the single-render invariant. The real MFA and Respiro-en integration tests
are opt-in with `VOICECUT_RUN_MFA_INTEGRATION=1` and
`VOICECUT_RUN_BREATH_INTEGRATION=1`; ordinary CI uses recorded or mocked
evidence and does not download model weights.

## License

VoiceCut is released under the MIT License. See [LICENSE](LICENSE).
