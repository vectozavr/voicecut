#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE="$REPOSITORY_ROOT/.env"
GENERATED_ROOT="$REPOSITORY_ROOT/examples/generated/work"

usage() {
  cat <<'EOF'
Run one or all public VoiceCut examples.

Usage:
  ./examples/run_examples.sh en
  ./examples/run_examples.sh ru
  ./examples/run_examples.sh video
  ./examples/run_examples.sh all

Derived outputs and caches are written only below examples/generated/work.
The committed reference media under examples/media is never overwritten.
EOF
}

MODE=${1:-all}
case "$MODE" in
  en|ru|video|all) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    echo >&2
    echo "Unknown example: $MODE (expected en, ru, video, or all)." >&2
    exit 2
    ;;
esac

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE" >&2
  echo "Create it with a non-empty GEMINI_API_KEY before running the demos." >&2
  exit 1
fi

if ! python3 - "$ENV_FILE" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
value = None
for line in path.read_text(encoding="utf-8").splitlines():
    match = re.match(r"^\s*(?:export\s+)?GEMINI_API_KEY\s*=\s*(.*?)\s*$", line)
    if match:
        value = match.group(1).strip()
if value is not None and len(value) >= 2 and value[0] == value[-1] in {'"', "'"}:
    value = value[1:-1].strip()
raise SystemExit(0 if value else 1)
PY
then
  echo "GEMINI_API_KEY is missing or empty in $ENV_FILE" >&2
  echo "The value is checked locally and is never printed or copied to demo artifacts." >&2
  exit 1
fi

if [[ -x "$REPOSITORY_ROOT/.venv/bin/voicecut" ]]; then
  VOICECUT="$REPOSITORY_ROOT/.venv/bin/voicecut"
elif command -v voicecut >/dev/null 2>&1; then
  VOICECUT=$(command -v voicecut)
else
  echo "VoiceCut is not installed. Run ./scripts/install.sh first." >&2
  exit 1
fi

mkdir -p "$GENERATED_ROOT"

run_example() {
  local example_id=$1
  local language=$2
  local input_relative=$3
  local output_name=$4
  local input_path="$REPOSITORY_ROOT/$input_relative"
  local example_root="$GENERATED_ROOT/$example_id"
  local output_path="$example_root/$output_name"
  local work_path="$example_root/cache"

  if [[ ! -f "$input_path" ]]; then
    echo "Missing committed example input: $input_relative" >&2
    exit 1
  fi
  case "$output_path" in
    "$GENERATED_ROOT"/*) ;;
    *)
      echo "Refusing to write outside examples/generated/work: $output_path" >&2
      exit 1
      ;;
  esac

  mkdir -p "$example_root"
  echo
  echo "=== $example_id ==="
  "$VOICECUT" "$input_path" \
    -o "$output_path" \
    --language "$language" \
    --planner-backend gemini \
    --work-dir "$work_path" \
    --env-file "$ENV_FILE" \
    --overwrite
  echo "Created: ${output_path#"$REPOSITORY_ROOT/"}"
}

case "$MODE" in
  en)
    run_example "example_en" "en" \
      "examples/media/audio/example_en.wav" "example_en_edited.wav"
    ;;
  ru)
    run_example "example_ru" "ru" \
      "examples/media/audio/example_ru.wav" "example_ru_edited.wav"
    ;;
  video)
    run_example "video" "en" \
      "examples/media/video/video.mp4" "video_edited.mp4"
    ;;
  all)
    run_example "example_en" "en" \
      "examples/media/audio/example_en.wav" "example_en_edited.wav"
    run_example "example_ru" "ru" \
      "examples/media/audio/example_ru.wav" "example_ru_edited.wav"
    run_example "video" "en" \
      "examples/media/video/video.mp4" "video_edited.mp4"
    ;;
esac
