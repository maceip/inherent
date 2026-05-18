# Inherent browser demo

The GitHub Pages app is served from the repository's `docs/` root. The
browser interface itself lives at `docs/index.html`; this directory keeps the
supporting JavaScript, CSS, theme asset, and compatibility redirect for the
old `/browser-demo/` URL. The page auto-loads the bundled ONNX model from
`docs/assets/` on startup, so visitors land directly on the record toolbar.

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
hand-drawn theme reference at `browser-demo/assets/theme-reference.svg`.
Replace that SVG with a production image asset if you want the exact uploaded
frame served by GitHub Pages.

While recording, the UI shows:

- a live oscilloscope canvas fed from the browser microphone PCM samples,
- the `mic -> mel -> gate -> heads` audio flow,
- 12 intent-head nodes that light up when their score crosses the model
  metadata threshold,
- the full 13-head score table, including the `isInteresting` gate.

## Bundled artifacts

The site ships with:

- `docs/assets/inherent.onnx`
- `docs/assets/inherent.onnx.metadata.json`

To replace the bundled demo artifact with a production trained export, run:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src .venv/bin/python -m inherent.scripts.export \
  --checkpoint artifacts/model-groups/001/best.pt \
  --config configs/production.yaml \
  --output-dir docs/assets \
  --backend onnx
```

The default page paths are:

- `docs/assets/inherent.onnx`
- `docs/assets/inherent.onnx.metadata.json`

Model files are ignored by the repository `.gitignore`, so production release
artifacts must be intentionally added with `git add -f` when you want GitHub
Pages to serve them.

## Publish on GitHub Pages

Configure Pages to serve the repository's `/docs` directory from this branch.
No GitHub Action is required because the site is plain static HTML/CSS/JS.

The Pages root has `docs/index.html`, which is the demo interface. The old
`/browser-demo/` path redirects back to the root for compatibility. These URLs
should both work after Pages finishes publishing:

```text
https://<owner>.github.io/<repo>/
https://<owner>.github.io/<repo>/browser-demo/
```

Microphone access requires HTTPS or `localhost`, which GitHub Pages provides.
