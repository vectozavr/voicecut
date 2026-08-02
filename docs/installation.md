# Installation

VoiceCut currently supports Apple Silicon Macs running macOS. The production
pipeline uses three isolated runtimes, so the supported installation method is
a source checkout followed by the repository installer.

For the shortest path back to the product overview, see the
[main README](../README.md). Runtime options are documented in
[Configuration](configuration.md), and common setup failures are covered in
[Troubleshooting](troubleshooting.md).

## Requirements

- Apple Silicon (`arm64`) Mac
- macOS
- Python 3.11 or 3.12
- Homebrew
- Internet access for installation and first-time model downloads
- Enough free disk space for Whisper, MFA, WhisperX, and optional planner
  models; together these can require several gigabytes

The installer checks for FFmpeg, FFprobe, and micromamba. When they are
missing, it installs them with Homebrew. It does not initialize micromamba or
modify the user's shell configuration.

## Recommended installation

```bash
git clone https://github.com/vectozavr/voicecut.git
cd voicecut
./scripts/install.sh
source .venv/bin/activate
```

The installer is safe to run again. It preserves an existing `.env`, verifies
pinned runtime components, and reuses environments that already satisfy the
expected layout.

It performs the following work:

1. verifies that the machine is Apple Silicon and that Python is supported;
2. installs FFmpeg and micromamba through Homebrew when needed;
3. creates `.mfa-env` from [`environment-mfa.yml`](../environment-mfa.yml) and
   verifies Montreal Forced Aligner 3.4.1 exactly;
4. creates the primary `.venv` and the MLX helper environment `.venv-mlx`;
5. downloads the pinned Respiro-en source, checkpoint, and license into the
   runtime cache and verifies their SHA-256 hashes;
6. installs VoiceCut's audio, alignment, and cloud-provider dependencies in
   `.venv`;
7. installs MLX Whisper and local-planner dependencies in `.venv-mlx`;
8. creates `.env` from [`.env.example`](../.env.example) when absent and sets
   its permissions to `0600`.

After installation, use the `voicecut` executable from the activated primary
environment. VoiceCut starts the other runtimes itself.

```bash
voicecut --help
```

## Why there are three runtimes

The separation is intentional:

| Runtime | Purpose |
| --- | --- |
| `.venv` | CLI, audio analysis, WhisperX completeness checks, rendering, publication, and cloud SDKs |
| `.venv-mlx` | MLX Whisper transcription and optional local MLX language models |
| `.mfa-env` | Montreal Forced Aligner 3.4.1 and its Conda/Kaldi dependencies |

Current `mlx-lm` and WhisperX dependency requirements conflict around
Transformers and Hugging Face Hub versions. Keeping MLX in a helper environment
prevents one component from destabilizing the other. MFA is isolated because
its supported distribution uses the Conda/Kaldi ecosystem; VoiceCut invokes
the documented MFA CLI and does not import undocumented MFA Python internals.

Do not install MFA into `.venv` or `.venv-mlx`.

## Selecting a Python installation

By default, the installer chooses `python3.12` and then `python3.11`. To use a
specific compatible interpreter:

```bash
VOICECUT_PYTHON=/opt/homebrew/bin/python3.12 ./scripts/install.sh
```

If an existing `.venv` or `.venv-mlx` was created with an unsupported Python
version, remove only that repository-local environment and rerun the installer.

## Configure a planner

Gemini is the default planner. Add the key for the cloud provider you intend to
use to `.env`:

```dotenv
GEMINI_API_KEY=...
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
```

Only the selected provider's key is required. Environment variables override
values in `.env`. VoiceCut does not write API keys to its run manifests.

The first minimal run with the default Gemini planner is:

```bash
voicecut recording.wav
```

This writes `recording_edited.wav`. See
[Planner backends](configuration.md#planner-backends) for cloud and local
alternatives.

## First-run downloads

The installer provisions software runtimes, but model weights are generally
downloaded when a relevant feature first runs:

- MLX Whisper downloads the selected transcription model;
- English MFA downloads `english_us_arpa`;
- Russian MFA downloads the revision-pinned
  `MontrealCorpusTools/russian_mfa` bundle;
- Russian retained-word validation downloads
  `jonatasgrosman/wav2vec2-large-xlsr-53-russian`;
- a local semantic planner downloads its selected Hugging Face model unless a
  local path or `--local-files-only` is used.

MFA models are stored below `.voicecut-cache/runtime/mfa` by default. The
pinned Respiro-en files are stored below
`.voicecut-cache/runtime/respiro-en/<commit>`.

Do not assume that a first run has stalled merely because model download or MFA
setup takes time. Run with an explicit `--work-dir` to preserve completed work
if a long first run is interrupted.

## Advanced manual environment setup

The repository installer remains the supported and recommended method. The
following commands show the underlying environment layout for contributors who
need to reconstruct it manually:

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

The MFA version command must report exactly `3.4.1`.

This manual sequence does **not** provision the hash-verified Respiro-en files.
Use `./scripts/install.sh` for the complete production setup. Until those files
are installed, explicitly use `--breath-cleanup off`; breath cleanup is optional
and must never be replaced by an unverified download.

Cloud dependencies can be omitted for a local-only experimental installation:

```bash
.venv/bin/python -m pip install -e ".[audio]"
.venv-mlx/bin/python -m pip install -e ".[mlx]"
```

MFA remains required for production cut coordinates regardless of which
semantic planner is selected.

## Verify the installation

Check the public command:

```bash
source .venv/bin/activate
voicecut --help
```

Check the pinned alignment runtime without activating micromamba:

```bash
MFA_ROOT_DIR="$PWD/.voicecut-cache/runtime/mfa" \
HF_HOME="$PWD/.voicecut-cache/runtime/mfa/huggingface" \
  micromamba run -p "$PWD/.mfa-env" mfa version
```

For repository development, install the development extra and run the local
checks:

```bash
source .venv/bin/activate
python -m pip install -e ".[audio,cloud,dev]"
.venv-mlx/bin/python -m pip install -e ".[mlx]"

pytest -q
ruff check src tests
ruff format --check src tests
```

The real MFA and Respiro-en integration tests are opt-in because ordinary CI
must not download model weights:

```bash
VOICECUT_RUN_MFA_INTEGRATION=1 pytest -q
VOICECUT_RUN_BREATH_INTEGRATION=1 pytest -q
```

## Updating an existing checkout

After pulling a VoiceCut update, rerun the installer:

```bash
git pull
./scripts/install.sh
source .venv/bin/activate
```

VoiceCut includes an implementation fingerprint in its cache configuration.
When production code or relevant model settings change, incompatible stage
artifacts are not silently reused. A new content-addressed default work
directory may therefore appear after an update.
