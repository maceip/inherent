# Inherent

<p align="center">
  <img src="docs/inherent-header.webp" alt="Inherent" width="480">
</p>

Inherent trains a small on-device model that decides two things in a single
forward pass over raw audio: is this speech meant for the assistant, and what
kind of request is it (add to a list, search for something, set a timer, look
up a photo, etc.).

The usual way to build this is two passes — first a speech-to-text model to
get a transcript, then a text classifier or LLM to figure out the intent.
That works, but it pays for transcription on every audio chunk even when the
speech isn't addressed to the assistant. Inherent does both jobs at once,
directly on the audio, so the gating and intent decision can happen before
any transcript exists. Use it as a fast front door: only run the heavier
speech-to-text and language-model stack when Inherent's scores say the audio
is interesting.

- **Input:** a mel-spectrogram window with shape `[1, T, 128]`, float32. Audio
  is expected at 16 kHz mono; other rates are resampled.
- **Output:** 13 scores with shape `[1, 13]`, float32, each between 0 and 1.
  The first is the "is this addressed to the assistant?" gate; the other 12
  are intent labels. Exact names and order are in `HEAD_ORDER` in
  `src/inherent/__init__.py`.

The model trains in PyTorch and exports to several on-device runtimes so the
same weights can run on phones, browsers, and Apple Silicon.

## Backends

After training, you can export to:

- **ONNX** — runs in ONNX Runtime on desktop, server, and the browser
  (WebGPU or WASM).
- **TFLite / LiteRT** — Android, iOS, and small Linux devices. Three
  delegate paths are exposed via `--delegate`: `cpu` (XNNPACK), `gpu`
  (mobile GPU — Adreno, Mali, Apple GPU), and `tpu`, which maps to the
  NPU class in LiteRT-LM (Qualcomm Hexagon HTP, Google Edge TPU, and
  similar dedicated neural accelerators).
- **LiteRT-LM** — the same LiteRT runtime with the delegate choice baked
  into the package; used when an external builder is available.
- **MLX** — native Apple Silicon on macOS, iPhone, and iPad.

### Rough latency per inference

For one ~1 second audio window (mel shape `[1, ~100, 128]`) on commodity
hardware. These are order-of-magnitude estimates, not measured benchmarks —
your numbers will vary with chip, batch size, and quantization.

| Path | Typical latency |
| --- | --- |
| Inherent → LiteRT NPU delegate (`tpu`, on Hexagon HTP / Edge TPU) | ~1–3 ms |
| Inherent → MLX on Apple Silicon (M1/M2/M3) | ~1–3 ms |
| Inherent → LiteRT GPU delegate (Adreno, Mali, Apple GPU) | ~2–5 ms |
| Inherent → ONNX Runtime on desktop CPU | ~3–8 ms |
| Inherent → LiteRT CPU delegate (XNNPACK on mobile) | ~4–10 ms |
| Inherent → ONNX Runtime Web (WebGPU) | ~5–12 ms |
| Same single-pass model, un-optimized PyTorch on CPU | ~40–120 ms |
| On-device speech-to-text → text intent classifier | ~80–400 ms |
| Cloud speech-to-text plus an intent classifier | ~250–800 ms |

The bottom two rows are the realistic baseline most projects would build:
run a small speech-to-text model (Whisper-tiny, Moonshine, a streaming
Conformer-CTC, etc.) to get a transcript, then parse the transcript with
rules or a small text classifier — locally or in the cloud. That two-pass
design works, but it pays for transcription on every audio chunk before it
can decide whether the speech was even meant for the assistant. Inherent
collapses both steps into one pass on the audio, which is where the
order-of-magnitude latency gap comes from.

NPUs are usually the fastest path for a model this small because the workload
is matrix-heavy and quantizes well to INT8. The catch is that not every
device ships one and the kernel coverage varies by vendor — fall back to GPU
or CPU when the delegate refuses an op. The un-optimized PyTorch row is
included for completeness: it shows what the same single-pass model costs
if you skip the export step entirely.

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

## Training data

You need two things:

1. **Audio files.** WAV, mono, ideally 16 kHz. Each file is one short
   utterance, typically 0.3 to 10 seconds.
2. **A labels CSV** with one row per audio file and these columns:

   ```
   audio_path, transcript, speaker_id, session_id, device, environment,
   source, duration_s, split,
   isInteresting,
   hasAddToListIntent, hasTermSearchQuery, hasPhotoQuery, hasCalendarEvent,
   hasCreateDocIntent, hasPersonContext, hasEventContext, hasDeepResearchIntent,
   hasInsightIntent, hasBrowsingAgentIntent, hasCallingAgentIntent,
   hasStartTimerIntent
   ```

   The 13 label columns are each `0` or `1`. `isInteresting` is `1` when the
   audio is meant for the assistant; the 12 intent columns should generally be
   `0` when `isInteresting` is `0`. `split` is one of `train`, `eval`, `test`.

A blank template is at `data/labels_template.csv`, and
`data/model_group_001_labels.csv` is a tiny working example.

For a usable model, aim for roughly half negatives (background and unrelated
speech) and at least a few hundred positive examples per intent.

## Train

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

To verify the pipeline on the bundled example, point `--labels` at
`data/model_group_001_labels.csv` and add `--max-steps 1`.

### Approximate wall time for a production run

Numbers from a real v0 run (146k mels, 100k steps, single L4 GPU box).

| Stage | Wall time |
|---|---|
| Mel materialization (146,032 mels through `audio_frontend.tflite`, CPU-bound) | ~115 min |
| Model training (100,000 steps, batch 32, L4 GPU, `num_workers=0`) | ~7h |

## Export an existing checkpoint

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src .venv/bin/python -m inherent.scripts.export \
  --checkpoint artifacts/model-groups/001/best.pt \
  --config configs/production.yaml \
  --output-dir artifacts/v0.1.0 \
  --backend all
```

Exported files land under `<output-dir>/{onnx,litert,mlx}/` with metadata
describing input/output tensor names, shapes, and label order.

## Use the model

The exported artifact takes a mel-spectrogram, not raw audio. The repo ships
`data/audio_frontend.tflite` (the same 128-bin mel frontend used during
training, 20 ms hop at 16 kHz mono) so you can go end-to-end from a WAV file
to scores. Below is a minimal Python example using the ONNX artifact; the
TFLite/LiteRT and MLX paths follow the same shape — only the runtime loader
changes.

```python
import numpy as np
import onnxruntime as ort
from inherent import HEAD_ORDER, THRESHOLD_KEYS, DEFAULT_THRESHOLDS_BY_KEY
from inherent.features.frontend import AudioFrontend

# 1. Raw 16 kHz mono WAV -> mel-spectrogram, shape [T, 128].
frontend = AudioFrontend("data/audio_frontend.tflite")
mel = frontend.wav_to_mel("some_clip.wav")
mel = mel[np.newaxis].astype(np.float32)   # shape [1, T, 128]

# 2. Mel -> 13 sigmoid scores.
session = ort.InferenceSession("artifacts/v0.1.0/onnx/inherent.onnx")
scores = session.run(None, {"mel_spectrogram": mel})[0][0]   # shape [13]

# 3. Apply per-label thresholds.
for name, key, score in zip(HEAD_ORDER, THRESHOLD_KEYS, scores):
    if score >= DEFAULT_THRESHOLDS_BY_KEY[key]:
        print(f"{name}: {score:.2f}")
```

`scores[0]` is the "is this addressed to the assistant?" gate; the remaining
12 are the intent labels in `HEAD_ORDER`. `AudioFrontend` requires TensorFlow
because the frontend is itself a TFLite model — you can substitute any
equivalent mel implementation (librosa, torchaudio, a mobile DSP block) as
long as the output is `[T, 128]` float32 with the same bin and hop layout.

On Android or iOS, load `inherent.tflite` with the standard LiteRT or TFLite
interpreter: the input tensor is `mel_spectrogram`, the output is
`intent_output`. MLX users load the package under `<output-dir>/mlx/`. The
metadata sidecar (`inherent.metadata.json`) carries the head order and
default thresholds so app code can pin to them at load time instead of
hard-coding.

The full integration contract (tensor names, shapes, head order, threshold
keys, versioning rules) is at `contracts/runtime_contract.md`.

## Changing the labels

To add, remove, or rename one of the 13 labels: edit `HEAD_ORDER` in
`src/inherent/__init__.py`, update your CSVs to match, and retrain from
scratch — checkpoints are not compatible across label changes.

## Test

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src .venv/bin/python -m pytest
```
