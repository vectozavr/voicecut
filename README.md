# VoiceCut

VoiceCut turns a retake-heavy spoken recording into coherent narration. It
transcribes the recording once, asks a selected language model which source
word occurrences belong in the intended take, validates that decision against
the source transcript, and renders waveform-safe cuts.

It accepts audio or video. Video is edited from its speech track: selected
audio intervals select the corresponding picture intervals, and an inserted
semantic pause holds the last selected frame. VoiceCut does not interpret the
visual content.

VoiceCut never synthesizes speech, replaces words, denoises the recording, or
adds generated background noise. The final voice samples come from the input.

## Supported system

- Apple Silicon Mac
- macOS
- Python 3.11 or 3.12
- FFmpeg and FFprobe

The first run downloads the selected Whisper and alignment models. A local
language model is downloaded only when a local planner backend is selected.
Model files can require several gigabytes of disk space.

## Install

The supported distribution is a source checkout installed by the repository
installer. A bare wheel or `pip install voicecut` does not create the separate
MLX helper environment required by the pipeline.

Clone the repository and run the installer:

```bash
git clone https://github.com/vectozavr/voicecut.git
cd voicecut
./scripts/install.sh
source .venv/bin/activate
```

The installer:

1. verifies Apple Silicon and Python;
2. installs FFmpeg with Homebrew when it is missing;
3. creates the primary `.venv` and an internal `.venv-mlx` helper environment;
4. installs VoiceCut, audio/alignment dependencies, and cloud SDKs in `.venv`;
5. installs MLX Whisper and local-planner dependencies in `.venv-mlx`;
6. creates an empty `.env` from `.env.example` without overwriting an existing
   file and restricts it to the current user (`0600`).

You invoke only the `voicecut` command from `.venv`. The helper environment is
launched automatically. It is separate because current `mlx-lm` requires
Transformers 5 and Hugging Face Hub 1.x, while WhisperX requires Hugging Face
Hub below 1; forcing both into one environment produces an unsatisfiable
installation.

To use a different Python installation:

```bash
VOICECUT_PYTHON=/opt/homebrew/bin/python3.12 ./scripts/install.sh
```

Manual installation is also supported:

```bash
brew install python@3.12 ffmpeg
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

Cloud SDKs are optional when only local planning is needed:

```bash
.venv/bin/python -m pip install -e ".[audio]"
.venv-mlx/bin/python -m pip install -e ".[mlx]"
```

## Quick start

Edit a WAV with Gemini:

```bash
voicecut recording.wav \
  -o recording_edited.wav \
  --planner-backend gemini
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
- extra semantic pause time holds the last retained frame;
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
acoustic retry evidence, streaming semantic plan, grounding report, alignment
context crops required by WhisperX, the semantic pause plan, one authoritative
`final_boundary_plan.json`, and one final internal WAV. No rendered preview WAV
feeds another production step. Completed stages are reused only when their input
hashes, relevant model settings, and VoiceCut implementation fingerprint match.
A software update therefore cannot silently reuse a plan or render produced by
older code.

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
output audio. For each omitted region, one local WhisperX context covers the
retained and omitted words on both sides. Character or word alignments define
protected speech spans with a small safety margin. Whisper timestamps remain
approximate anchors and are never clamped together or used as hard cut limits.

Waveform energy is secondary evidence: it may choose a splice only inside an
alignment-established interval that also contains verified quiet audio. If no
such interval exists, the boundary is recorded as `unsafe_dense_boundary` and
the run stops before rendering instead of guessing. Fades are confined to those
verified quiet intervals, so retained speech—including quiet final fricatives
such as `/s/`—remains sample-identical to the canonical WAV.

A separate semantic pause classification still assigns `continuation`, `short`,
`thought`, or `section`. Existing natural quiet counts toward the target total
gap. Any deficit is filled with verified room tone only at a resolved safe join,
or inside an aligned natural inter-word gap within a contiguous take. If there
is no safe internal gap, the extra pause is skipped. The final word uses a safe
EOF tail because no later unwanted word can be included.

Only after all boundaries, pauses, room-tone source ranges, and fade intervals
have been frozen in `final_boundary_plan.json` does `final_render.py` slice the
canonical source WAV. It performs one render and writes one internal
`final_cut.wav`; no later stage may move or fade a boundary.

### 7. Publication

The validated internal WAV is encoded to the requested audio container. For
video, the same source interval timeline is applied to the pictures and the
edited audio is muxed back in. Publication validates the output stream type,
duration, and synchronization before replacing the destination.

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
| `--window-seconds N` | New transcript look-ahead added per planner iteration |
| `--max-output-tokens N` | Maximum structured planner response size |
| `--debug-artifacts` | Request optional diagnostics without changing the single-pass render graph |
| `--asr-python PATH` | Advanced: Python executable for MLX ASR/local CTC stages |
| `--alignment-python PATH` | Advanced: Python executable for WhisperX alignment |
| `--planner-python PATH` | Advanced: Python executable for local MLX LLMs |

Run `voicecut --help` for the authoritative option list.

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
- Video editing follows speech only and does not understand shots, gestures,
  slides, captions, or visual continuity.
- Audio/video output is re-encoded in common delivery codecs; VoiceCut is not a
  lossless container remuxer.
- A fail-closed alignment or grounding error requires a new run or a better
  model; VoiceCut does not guess an unsafe boundary.
- This is an editor, not a denoiser, mastering suite, speech generator, or
  voice-cloning system.

## Development

Install development and cloud dependencies:

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
alignment-protected `/s/` and leading-word regressions, overlapping Whisper
timestamps, fail-closed dense boundaries, semantic pauses, EOF tails,
sample-trace invariants, media conversion, video timeline construction,
caching, and one-command orchestration.

## License

VoiceCut is released under the MIT License. See [LICENSE](LICENSE).
