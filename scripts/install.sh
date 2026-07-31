#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
MLX_VENV_DIR="${PROJECT_DIR}/.venv-mlx"
MFA_ENV_DIR="${PROJECT_DIR}/.mfa-env"
MFA_CACHE_ROOT="${VOICECUT_MFA_CACHE_ROOT:-${PROJECT_DIR}/.voicecut-cache/runtime/mfa}"
MFA_HF_CACHE="${MFA_CACHE_ROOT}/huggingface"
RESPIRO_UPSTREAM_COMMIT="70e01c60c2f582c41092730680f2894ab24d6467"
RESPIRO_MODULES_SHA256="f789e0986e3090d7df5f9f0f596d9e3601c6da514c3ac01a65920a493b840e46"
RESPIRO_CHECKPOINT_SHA256="1f4a9b96f96645c480bf0e07b1e18cd68878ac0b4bb5dc920ad93f9b17df858a"
RESPIRO_LICENSE_SHA256="a34ad1af58dc7c02f867f620f7ddc952029b383c9b0dce349d54f6b875e079cd"
RESPIRO_CACHE_ROOT="${VOICECUT_RESPIRO_CACHE_ROOT:-${PROJECT_DIR}/.voicecut-cache/runtime/respiro-en/${RESPIRO_UPSTREAM_COMMIT}}"
RESPIRO_RAW_URL="https://raw.githubusercontent.com/ydqmkkx/Respiro-en/${RESPIRO_UPSTREAM_COMMIT}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "VoiceCut currently supports Apple Silicon macOS (arm64)." >&2
  exit 1
fi

if ! command -v micromamba >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing micromamba with Homebrew..."
    brew install micromamba
  else
    echo "micromamba is required for the pinned MFA runtime." >&2
    echo "Install Homebrew, then run: brew install micromamba" >&2
    exit 1
  fi
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

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

download_respiro_file() {
  local filename="$1"
  local expected_sha256="$2"
  local destination="${RESPIRO_CACHE_ROOT}/${filename}"
  local actual_sha256

  if [[ -e "${destination}" ]]; then
    if [[ ! -f "${destination}" || -L "${destination}" ]]; then
      echo "Refusing unsafe Respiro-en cache entry: ${destination}" >&2
      exit 1
    fi
    actual_sha256="$(sha256_file "${destination}")"
    if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
      echo "Respiro-en hash mismatch for existing ${destination}." >&2
      echo "Expected ${expected_sha256}, got ${actual_sha256}." >&2
      exit 1
    fi
    return
  fi

  local temporary
  temporary="$(mktemp "${RESPIRO_CACHE_ROOT}/.${filename}.XXXXXX")"
  if ! curl --fail --location --retry 3 \
    --output "${temporary}" \
    "${RESPIRO_RAW_URL}/${filename}"; then
    rm -f "${temporary}"
    echo "Failed to download pinned Respiro-en ${filename}." >&2
    exit 1
  fi
  actual_sha256="$(sha256_file "${temporary}")"
  if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    rm -f "${temporary}"
    echo "Respiro-en hash mismatch for downloaded ${filename}." >&2
    echo "Expected ${expected_sha256}, got ${actual_sha256}." >&2
    exit 1
  fi
  chmod 600 "${temporary}"
  mv "${temporary}" "${destination}"
}

mkdir -p "${RESPIRO_CACHE_ROOT}"
download_respiro_file "modules.py" "${RESPIRO_MODULES_SHA256}"
download_respiro_file "respiro-en.pt" "${RESPIRO_CHECKPOINT_SHA256}"
download_respiro_file "LICENSE" "${RESPIRO_LICENSE_SHA256}"
echo "Verified Respiro-en ${RESPIRO_UPSTREAM_COMMIT}."

if [[ ! -x "${MFA_ENV_DIR}/bin/mfa" ]]; then
  echo "Creating the pinned MFA 3.4.1 environment..."
  micromamba create -y \
    -p "${MFA_ENV_DIR}" \
    -f "${PROJECT_DIR}/environment-mfa.yml"
fi

mkdir -p "${MFA_CACHE_ROOT}" "${MFA_HF_CACHE}"
MFA_VERSION_OUTPUT="$(
  MFA_ROOT_DIR="${MFA_CACHE_ROOT}" \
    HF_HOME="${MFA_HF_CACHE}" \
    micromamba run -p "${MFA_ENV_DIR}" mfa version
)"
MFA_VERSION="$(
  printf '%s\n' "${MFA_VERSION_OUTPUT}" \
    | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' \
    | head -n 1
)"
if [[ "${MFA_VERSION}" != "3.4.1" ]]; then
  echo "Expected MFA 3.4.1, got: ${MFA_VERSION_OUTPUT}" >&2
  echo "Remove ${MFA_ENV_DIR} and rerun the installer." >&2
  exit 1
fi
echo "Verified Montreal Forced Aligner ${MFA_VERSION}."

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
