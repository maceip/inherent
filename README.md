# Inherent

Inherent trains and exports a small audio-to-intent model for an ambient assistant.
It takes a mel-spectrogram window shaped `[1, T, 128]` and returns 13 sigmoid
scores: one "is this addressed to the assistant?" gate plus 12 intent/context
heads.

The point of the project is to make the assistant faster and cheaper at runtime.
Instead of running every bit of nearby speech through a heavier language stack,
Inherent can quickly decide whether audio is interesting and which skill family
it may belong to.

## What It Does

- Builds labeled audio manifests from recorded WAVs, public speech corpora, and
  approved synthetic speech.
- Trains a compact Conformer-style PyTorch model with 13 output heads.
- Evaluates per-head quality with AUC/EER-style metrics.
- Exports the model to ONNX, LiteRT/TFLite, MLX package scaffolds, and LiteRT-LM
  packaging when the external builder is available.
- Writes metadata and backend validation reports so app integration can check
  tensor names, shapes, head order, and delegate support.

## Current Technical Details

- Runtime input: `mel_spectrogram`, shape `[1, T, 128]`, float32.
- Runtime output: `intent_output`, shape `[1, 13]`, float32 sigmoid scores.
- Head order lives in `src/inherent/__init__.py` as `HEAD_ORDER`.
- Model architecture lives in `src/inherent/models/architecture.py`.
- Training config lives in `configs/base.yaml`, `configs/baseline.yaml`, and
  `configs/production.yaml`.
- Export backends live in `src/inherent/export/`.
- The recorded-data build pipeline lives in `src/inherent/scripts/build_recorded.py`.

Recent smoke exports produced:

- LiteRT/TFLite: `artifacts/model-groups/001-production-smoke/export/litert/inherent.tflite`
- ONNX: `artifacts/model-groups/001-production-smoke/export/onnx/inherent.onnx`
- MLX package: `artifacts/model-groups/001-production-smoke/export/mlx/mlx`

Generated artifacts and local datasets are intentionally ignored by git.

## Install

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[tts,dev]"
```

Verify:

```bash
PYTHONPATH=src python -c "import inherent; print(inherent.__version__, inherent.HEAD_ORDER)"
```

## Train

For the recorded-audio pipeline, provide a label CSV with this shape:

```csv
audio_path,transcript,utterance_id,speaker_id,device_id,environment,source,duration_seconds,split,isInteresting,hasAddToListIntent,hasPhotoQueryIntent,hasTermSearchIntent,hasCreateDocIntent,hasDeepResearchIntent,hasInsightIntent,hasBrowsingAgentIntent,hasCallingAgentIntent,hasCalendarEventIntent,hasPersonContext,hasEventContext,hasStartTimerIntent
```

Then run:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src .venv/bin/python -m inherent.scripts.build_recorded \
  --config configs/production.yaml \
  --labels data/recorded_labels.csv \
  --model-group-dir artifacts/model-groups/001 \
  --frontend-model data/audio_frontend.tflite \
  --export-backend litert \
  --export-backend onnx \
  --export-backend mlx \
  --device mps \
  --eval-device cpu
```

For a quick local pipeline check, use the dummy/smoke label set:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src .venv/bin/python -m inherent.scripts.build_recorded \
  --config configs/production.yaml \
  --labels data/model_group_001_labels.csv \
  --model-group-dir artifacts/model-groups/001-production-smoke \
  --frontend-model data/audio_frontend.tflite \
  --export-backend litert \
  --export-backend onnx \
  --export-backend mlx \
  --device mps \
  --eval-device cpu \
  --max-steps 1
```

## Export

Export an existing checkpoint:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src .venv/bin/python -m inherent.scripts.export \
  --checkpoint artifacts/model-groups/001/best.pt \
  --config configs/production.yaml \
  --output-dir artifacts/v0.1.0 \
  --backend all \
  --delegate all
```

Preflight export tooling without running a conversion:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src .venv/bin/python -m inherent.scripts.export \
  --config configs/production.yaml \
  --output-dir artifacts/v0.1.0 \
  --backend all \
  --delegate all \
  --preflight
```

LiteRT-LM export expects a real builder. Set `INHERENT_LITERTLM_BUILDER_BIN`
when you have one, or explicitly opt into the local Bazel fallback with
`INHERENT_LITERTLM_ALLOW_BAZEL=1`.

## Updating Heads

There are two different kinds of heads in this repo:

- Output intent heads: the 13 runtime labels returned by the model.
- Transformer attention heads: the internal multi-head self-attention setting.

To add, remove, or rename output intent heads:

1. Update `HEAD_ORDER` in `src/inherent/__init__.py`.
2. Update `NUM_HEADS` or related constants if the count changes.
3. Update label CSV templates, tests, and any app-side code that reads the output
   vector.
4. Update public/synthetic data mappings in `src/inherent/data/`.
5. Retrain from a new model group. Do not warm-start across incompatible output
   head layouts unless you write an explicit checkpoint migration.
6. Re-export every backend and verify metadata matches the new order.

To change the internal transformer attention heads, edit `num_attention_heads` in
the relevant config file. `hidden_size` must be divisible by `num_attention_heads`.

## Test

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src .venv/bin/python -m pytest
```

Focused export/model checks:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_architecture.py \
  tests/test_export_backends.py \
  tests/test_recorded_pipeline.py
```
