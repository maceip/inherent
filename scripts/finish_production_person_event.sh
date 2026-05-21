#!/usr/bin/env bash
# End-to-end finish: train (if needed) -> eval -> export production TFLite + metadata.
# Run on EC2 after person/event Supertonic data is merged into quality_train_manifest.csv.
set -euo pipefail

ROOT="${ROOT:-/home/ubuntu/inherent}"
RUN_ID="${RUN_ID:-ec2-person-event-supertonic}"
CFG="${CFG:-configs/production_quality.yaml}"
CKPT_DIR="artifacts/model-groups/${RUN_ID}"
PROD_DIR="artifacts/quality"

cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=src
export TF_CPP_MIN_LOG_LEVEL=1
export HF_HOME="$ROOT/data/.hf_home"
export HF_DATASETS_CACHE="$ROOT/data/.hf_datasets_cache"

mkdir -p logs "$CKPT_DIR" "$PROD_DIR"

stage() { echo "[$(date -u +%FT%TZ)] $RUN_ID $*" | tee -a "logs/${RUN_ID}.status"; }
mark_done() { touch "logs/${RUN_ID}.$1.done"; }
need() { [ ! -f "logs/${RUN_ID}.$1.done" ]; }

if need train; then
  stage "train max_steps=${PERSON_EVENT_MAX_STEPS:-8000}"
  PERSON_EVENT_MAX_STEPS="${PERSON_EVENT_MAX_STEPS:-8000}" \
  PERSON_EVENT_BATCH_SIZE="${PERSON_EVENT_BATCH_SIZE:-4}" \
  ./scripts/train_eval_person_event.sh
  # train_eval script marks train+eval; we only needed train here if split — it does both
  exit 0
fi

if need eval; then
  stage "eval checkpoint"
  python -m inherent.scripts.eval \
    --checkpoint "${CKPT_DIR}/best.pt" \
    --eval-set data/quality_eval_manifest.csv \
    --batch-size 32 \
    --device cuda \
    --config "$CFG" \
    --json-out "${CKPT_DIR}/eval_metrics.json" \
    --gate-json-out "${CKPT_DIR}/eval_gates.json"
  mark_done eval
fi

if need export_litert; then
  stage "export production TFLite"
  python -m inherent.scripts.export \
    --checkpoint "${CKPT_DIR}/best.pt" \
    --config "$CFG" \
    --output-dir "${CKPT_DIR}/export/litert" \
    --backend litert \
    --delegate cpu
  cp "${CKPT_DIR}/export/litert/inherent.tflite" "${PROD_DIR}/inherent.tflite"
  cp "${CKPT_DIR}/export/litert/inherent.metadata.json" "${PROD_DIR}/inherent.metadata.json"
  mark_done export_litert
fi

if need acceptance; then
  stage "acceptance smoke test on TFLite"
  python - <<'PY'
import numpy as np
import tensorflow as tf

m = "artifacts/quality/inherent.tflite"
i = tf.lite.Interpreter(model_path=m, num_threads=1)
i.allocate_tensors()
inp = i.get_input_details()[0]
out = i.get_output_details()[0]
assert inp["shape"].tolist() == [1, 3000, 128], inp
assert out["shape"].tolist() == [1, 13], out
x = np.zeros([1, 3000, 128], dtype=np.float32)
i.set_tensor(inp["index"], x)
i.invoke()
y = i.get_tensor(out["index"])
assert y.shape == (1, 13) and np.isfinite(y).all()
print("acceptance ok")
PY
  mark_done acceptance
fi

stage "production artifacts ready at ${PROD_DIR}/"
ls -lh "${PROD_DIR}/inherent.tflite" "${PROD_DIR}/inherent.metadata.json" data/audio_frontend.tflite
