#!/usr/bin/env bash
# Train + eval only (after mels merged). Run on EC2 with nohup.
set -euo pipefail
cd "${ROOT:-/home/ubuntu/inherent}"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=src
export TF_CPP_MIN_LOG_LEVEL=1
LOG="${LOG:-logs/train_eval_person_event.log}"
exec >>"$LOG" 2>&1

echo "[$(date -u +%FT%TZ)] starting train"
python - <<'PY'
from pathlib import Path
from inherent.config import Config
from inherent.training.train import train

cfg = Config.load("configs/production_quality.yaml")
max_steps = int("${PERSON_EVENT_MAX_STEPS:-8000}")
cfg.training.max_steps = max_steps
cfg.training.batch_size = int("${PERSON_EVENT_BATCH_SIZE:-4}")
cfg.training.eval_every_steps = min(2000, max(500, max_steps // 4))
cfg.training.save_every_steps = cfg.training.eval_every_steps
cfg.training.__post_init__()
out = Path("artifacts/model-groups/ec2-person-event-supertonic")
init = out / "last.pt"
fallback = Path("artifacts/model-groups/ec2-quality-balanced/best.pt")
train(cfg, out, init_checkpoint=init if init.is_file() else fallback)
PY
touch logs/ec2-person-event-supertonic.train.done
echo "[$(date -u +%FT%TZ)] train done"

echo "[$(date -u +%FT%TZ)] starting eval"
python -m inherent.scripts.eval \
  --checkpoint artifacts/model-groups/ec2-person-event-supertonic/best.pt \
  --eval-set data/quality_eval_manifest.csv \
  --batch-size 32 \
  --device cuda \
  --config configs/production_quality.yaml \
  --json-out artifacts/model-groups/ec2-person-event-supertonic/eval_metrics.json \
  --gate-json-out artifacts/model-groups/ec2-person-event-supertonic/eval_gates.json
touch logs/ec2-person-event-supertonic.eval.done
echo "[$(date -u +%FT%TZ)] eval done" >>logs/ec2-person-event-supertonic.status
echo "[$(date -u +%FT%TZ)] pipeline complete" >>logs/ec2-person-event-supertonic.status
