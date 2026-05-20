#!/usr/bin/env bash
# Integrate inherent-sync Supertonic person/event TTS into the quality train set and fine-tune.
#
# Prereqs on the GPU box:
#   - ~/inherent with .venv (CUDA) and completed ec2-quality-balanced run
#   - ~/inherent-sync/data/synthetic_person_event_manifest.csv + synthetic_audio/
#
# Usage:
#   RUN_ID=ec2-person-event-supertonic ./scripts/ec2_integrate_person_event.sh
#   tail -f logs/ec2-person-event-supertonic.status
#
set -euo pipefail

ROOT="${ROOT:-/home/ubuntu/inherent}"
SYNC="${SYNC:-/home/ubuntu/inherent-sync}"
RUN_ID="${RUN_ID:-ec2-person-event-supertonic}"
CFG="${CFG:-configs/production_quality.yaml}"
INIT_CKPT="${INIT_CKPT:-artifacts/model-groups/ec2-quality-balanced/best.pt}"
MEL_WORKERS="${MEL_WORKERS:-4}"
PERSON_EVENT_MAX_STEPS="${PERSON_EVENT_MAX_STEPS:-12000}"

cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=src
export PATH="$ROOT/.venv/bin:$PATH"
export HF_HOME="$ROOT/data/.hf_home"
export HF_DATASETS_CACHE="$ROOT/data/.hf_datasets_cache"
export TF_CPP_MIN_LOG_LEVEL=1

mkdir -p logs data/quality_mels/person_event_sup "artifacts/model-groups/$RUN_ID"

stage() { echo "[$(date -u +%FT%TZ)] $RUN_ID $*" | tee -a "logs/${RUN_ID}.status"; }
mark_done() { touch "logs/${RUN_ID}.$1.done"; }
need() { [ ! -f "logs/${RUN_ID}.$1.done" ]; }

if [ ! -f "$SYNC/data/synthetic_person_event_manifest.csv" ]; then
  echo "missing $SYNC/data/synthetic_person_event_manifest.csv" >&2
  exit 1
fi

if need convert_raw; then
  stage "convert synthetic person/event manifest to raw 13-head CSV"
  SYNC="$SYNC" python - <<'PY'
import csv
import os
from pathlib import Path

from inherent import HEAD_ORDER

src = Path(os.environ["SYNC"]) / "data/synthetic_person_event_manifest.csv"
dst = Path("data/quality_raw_person_event_sup.csv")
count = 0
with src.open(newline="") as f, dst.open("w", newline="") as out:
    reader = csv.DictReader(f)
    writer = csv.DictWriter(out, fieldnames=["audio_path", *HEAD_ORDER])
    writer.writeheader()
    for row in reader:
        labels = {head: "0" for head in HEAD_ORDER}
        labels["isInteresting"] = "1"
        labels[row["head"].strip()] = "1"
        writer.writerow({"audio_path": row["audio_path"].strip(), **labels})
        count += 1
print(f"wrote {count} rows to {dst}")
PY
  mark_done convert_raw
fi

if need person_event_mels; then
  stage "materialize mels (${MEL_WORKERS} workers)"
  python -m inherent.scripts.prep_data \
    --config "$CFG" \
    --target mels \
    --input-manifest data/quality_raw_person_event_sup.csv \
    --output-manifest data/person_event_sup_train_mels.csv \
    --mel-dir data/quality_mels/person_event_sup \
    --frontend-model data/audio_frontend.tflite \
    --workers "$MEL_WORKERS"
  mark_done person_event_mels
fi

if need merge_train; then
  stage "append person/event mels to quality_train_manifest"
  python - <<'PY'
import csv
import shutil
from pathlib import Path

train = Path("data/quality_train_manifest.csv")
addon = Path("data/person_event_sup_train_mels.csv")
backup = train.with_suffix(".csv.bak-before-person-event-sup")
if not backup.is_file():
    shutil.copy2(train, backup)
    print(f"backup -> {backup}")
with train.open(newline="") as f:
    base_fields = csv.DictReader(f).fieldnames
with addon.open(newline="") as f:
    addon_fields = csv.DictReader(f).fieldnames
if base_fields != addon_fields:
    raise SystemExit(f"manifest field mismatch: {base_fields} vs {addon_fields}")
before = sum(1 for _ in csv.DictReader(train.open()))
added = 0
with train.open("a", newline="") as out_f, addon.open(newline="") as in_f:
    reader = csv.DictReader(in_f)
    writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
    for row in reader:
        writer.writerow(row)
        added += 1
after = sum(1 for _ in csv.DictReader(train.open()))
print(f"appended {added} rows ({before} -> {after})")
PY
  mark_done merge_train
fi

if need train; then
  stage "fine-tune from ${INIT_CKPT} max_steps=${PERSON_EVENT_MAX_STEPS}"
  python - <<PY
from pathlib import Path
from inherent.config import Config
from inherent.training.train import train

cfg = Config.load("$CFG")
cfg.training.max_steps = int("$PERSON_EVENT_MAX_STEPS")
cfg.training.eval_every_steps = min(2000, max(500, cfg.training.max_steps // 6))
cfg.training.save_every_steps = cfg.training.eval_every_steps
cfg.training.__post_init__()
init = Path("$INIT_CKPT")
train(
    cfg,
    Path("artifacts/model-groups/$RUN_ID"),
    init_checkpoint=init if init.is_file() else None,
)
PY
  mark_done train
fi

if need eval; then
  stage "eval on quality eval set"
  python -m inherent.scripts.eval \
    --checkpoint "artifacts/model-groups/$RUN_ID/best.pt" \
    --eval-set data/quality_eval_manifest.csv \
    --batch-size 32 \
    --device cuda \
    --config "$CFG" \
    --json-out "artifacts/model-groups/$RUN_ID/eval_metrics.json" \
    --gate-json-out "artifacts/model-groups/$RUN_ID/eval_gates.json" \
    | tee "artifacts/model-groups/$RUN_ID/eval_metrics.csv"
  python - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/model-groups/$RUN_ID/eval_metrics.json")
metrics = json.loads(path.read_text())
for head in ("hasPersonContext", "hasEventContext"):
    print(head, metrics[head])
PY
  mark_done eval
fi

stage "complete"
