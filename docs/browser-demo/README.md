# Inherent browser demo

This directory is a static GitHub Pages demo for running a trained Inherent
export from the browser microphone.

## Browser backend support

Of the export backends currently registered by `inherent`, only ONNX is
directly browser-runnable in this repository:

| Export backend | Browser support | Notes |
| --- | --- | --- |
| `onnx` | Yes | Runs with ONNX Runtime Web. The page tries WebGPU first when available and falls back to WASM. |
| `tflite` / `litert` | No direct Pages path | These artifacts target native/mobile LiteRT/TFLite interpreters. A browser TFLite WASM runtime would be a separate integration. |
| `litertlm` | No | LiteRT-LM packages are for the LiteRT-LM runtime, not static web pages. |
| `mlx` | No | MLX packages target Apple Silicon runtimes. |

The demo computes a 16 kHz mono, 128-bin mel-spectrogram in JavaScript and
feeds it to the ONNX model as `mel_spectrogram`. The production mobile path
uses `audio_frontend.tflite`; treat browser scores as demo/runtime-validation
signals unless the JavaScript frontend has been parity-checked against that
TFLite frontend for your model.

## Page theme and live UI

The page uses the supplied dojo/arcade frame as its art direction: red walls,
yellow-green floor lanes, pink/purple trim, chunky outlined cards, and a
hand-drawn theme reference at `assets/theme-reference.svg`. Replace that SVG
with a production image asset if you want the exact uploaded frame served by
GitHub Pages.

While recording, the UI shows:

- a live oscilloscope canvas fed from the browser microphone PCM samples,
- the `mic -> mel -> gate -> heads` audio flow,
- 12 intent-head nodes that light up when their score crosses the model
  metadata threshold,
- the full 13-head score table, including the `isInteresting` gate.

## Prepare artifacts

Export an ONNX artifact and metadata sidecar:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src .venv/bin/python -m inherent.scripts.export \
  --checkpoint artifacts/model-groups/001/best.pt \
  --config configs/production.yaml \
  --output-dir docs/browser-demo/assets \
  --backend onnx
```

The default page paths are:

- `docs/browser-demo/assets/inherent.onnx`
- `docs/browser-demo/assets/inherent.onnx.metadata.json`

Model files are ignored by the repository `.gitignore`, so either host the
model elsewhere and paste its URL into the page, or intentionally add release
artifacts with `git add -f` when you want GitHub Pages to serve them.

## Publish on GitHub Pages

Configure Pages to serve the repository's `/docs` directory. The demo will be
available at:

```text
https://<owner>.github.io/<repo>/browser-demo/
```

Microphone access requires HTTPS or `localhost`, which GitHub Pages provides.
