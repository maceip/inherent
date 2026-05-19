#!/usr/bin/env bash
# Run person/event synthetic TTS on a remote Linux GPU box (Supertonic 3).
#
# Usage (on the remote machine):
#   git clone https://github.com/maceip/inherent.git && cd inherent
#   ./scripts/run_synthetic_remote.sh
#
# Optional: resume laptop partial output first:
#   rsync -avz LOCAL:data/synthetic_person_event_manifest.csv.partial LOCAL:data/synthetic_audio/ \
#     REMOTE:~/inherent/data/
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${INHERENT_SYNTHETIC_CONFIG:-configs/synthetic_person_event_remote.yaml}"
OUTPUT_MANIFEST="${INHERENT_SYNTHETIC_MANIFEST:-data/synthetic_person_event_manifest.csv}"
LOG_DIR="${INHERENT_LOG_DIR:-artifacts}"
LOG_FILE="${LOG_DIR}/synthetic_remote.log"

mkdir -p "$LOG_DIR" data

PYTHON=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done
if [[ -z "$PYTHON" ]]; then
  echo "No python3 found" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -q -U pip
pip install -q -e ".[tts]"

export PYTHONPATH=src
export INHERENT_SYNTHETIC_MIC_AUGMENT="${INHERENT_SYNTHETIC_MIC_AUGMENT:-1}"

echo "Starting synthesis: config=$CONFIG manifest=$OUTPUT_MANIFEST"
echo "Log: $LOG_FILE"

nohup .venv/bin/python -m inherent.scripts.prep_data \
  --config "$CONFIG" \
  --target synthesis \
  --output-manifest "$OUTPUT_MANIFEST" \
  >>"$LOG_FILE" 2>&1 &

echo "PID=$!"
echo "tail -f $LOG_FILE"
