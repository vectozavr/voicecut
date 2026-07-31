#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
MLX_VENV_DIR="${PROJECT_DIR}/.venv-mlx"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "VoiceCut currently supports Apple Silicon macOS (arm64)." >&2
  exit 1
fi

select_python() {
  if [[ -n "${VOICECUT_PYTHON:-}" ]]; then
    printf '%s\n' "${VOICECUT_PYTHON}"
    return
  fi
  local candidate
  for candidate in python3.12 python3.11; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return
    fi
  done
  return 1
}

PYTHON_BIN="$(select_python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Python 3.11 or 3.12 is required." >&2
  echo "Install it with: brew install python@3.12" >&2
  exit 1
fi

PYTHON_VERSION="$(
  "${PYTHON_BIN}" -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)"
if [[ "${PYTHON_VERSION}" != "3.11" && "${PYTHON_VERSION}" != "3.12" ]]; then
  echo "VOICECUT_PYTHON must point to Python 3.11 or 3.12." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1 \
  || ! command -v ffprobe >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing FFmpeg with Homebrew..."
    brew install ffmpeg
  else
    echo "FFmpeg and FFprobe are required." >&2
    echo "Install Homebrew, then run: brew install ffmpeg" >&2
    exit 1
  fi
fi

cd "${PROJECT_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating ${VENV_DIR} with Python ${PYTHON_VERSION}..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
if [[ ! -x "${MLX_VENV_DIR}/bin/python" ]]; then
  echo "Creating ${MLX_VENV_DIR} with Python ${PYTHON_VERSION}..."
  "${PYTHON_BIN}" -m venv "${MLX_VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
MLX_VENV_PYTHON="${MLX_VENV_DIR}/bin/python"
VENV_VERSION="$(
  "${VENV_PYTHON}" -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)"
MLX_VENV_VERSION="$(
  "${MLX_VENV_PYTHON}" -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)"
for version_and_path in \
  "${VENV_VERSION}:${VENV_DIR}" \
  "${MLX_VENV_VERSION}:${MLX_VENV_DIR}"; do
  version="${version_and_path%%:*}"
  path="${version_and_path#*:}"
  if [[ "${version}" != "3.11" && "${version}" != "3.12" ]]; then
    echo "${path} uses unsupported Python ${version}." >&2
    echo "Remove ${path} and rerun this installer." >&2
    exit 1
  fi
done

echo "Installing VoiceCut audio, alignment, and cloud-provider support..."
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
"${VENV_PYTHON}" -m pip install --editable ".[audio,cloud]"

# mlx-lm currently requires Transformers 5 / Hugging Face Hub 1.x, while
# WhisperX requires Hugging Face Hub below 1. Keep that upstream conflict out
# of the user-facing environment. VoiceCut automatically launches this helper.
echo "Installing MLX Whisper and local-planner support..."
"${MLX_VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
"${MLX_VENV_PYTHON}" -m pip install --editable ".[mlx]"

if [[ -L .env ]]; then
  echo "Refusing to use a symbolic-link .env file." >&2
  exit 1
elif [[ ! -e .env ]]; then
  install -m 600 .env.example .env
  echo "Created .env from .env.example with mode 0600."
elif [[ ! -f .env ]]; then
  echo ".env exists but is not a regular file." >&2
  exit 1
else
  chmod 600 .env
fi

"${VENV_DIR}/bin/voicecut" --help >/dev/null
"${MLX_VENV_PYTHON}" -c \
  'import mlx, mlx_lm, mlx_whisper, voicecut.transcribe_mlx'

echo
echo "VoiceCut is installed."
echo "Activate it with:"
echo "  source \"${VENV_DIR}/bin/activate\""
echo
echo "Then add an API key to ${PROJECT_DIR}/.env and run:"
echo "  voicecut input.wav -o input_edited.wav --planner-backend gemini"
echo
echo "Local MLX planners are available for experimentation but are not yet"
echo "recommended for production semantic editing."
