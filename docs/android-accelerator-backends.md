# Android Accelerator Backends

This repo should produce one portable runtime model first:

- `inherent.tflite`
- input `mel_spectrogram`, `float32`, `[1, 3000, 128]`
- output `intent_output`, `float32`, `[1, 13]`
- standard TFLite/LiteRT ops only, no recovered DarwiNN custom op

That artifact is the source for every Android backend. The recovered
`audio_gatekeeper_edgetpu.tflite` is useful as a teacher and reference, but it is
not the public deployment path for Inherent because it is already a compiled
Pixel/DarwiNN custom-op artifact.

## Runtime Tree

1. CPU baseline: always ship and always validate first.
   - Use LiteRT/TFLite Interpreter or LiteRT `CompiledModel` CPU.
   - This is the correctness fallback for every Android device.

2. Generic NPU: use LiteRT `CompiledModel`.
   - Request `kLiteRtHwAcceleratorNpu`.
   - Enable compiler cache in the app's writable files directory.
   - Validate JIT compilation on physical devices before attempting AOT.
   - Supported public setup paths are Qualcomm AI Engine Direct, MediaTek
     NeuroPilot, Intel OpenVINO, and experimental Google Tensor.

3. Pixel / Google Tensor TPU: use Tensor ML SDK access first.
   - Treat `tpu` as a Google Tensor target, not as the recovered DarwiNN service.
   - Use the SDK to check op support, estimate TPU latency, and profile CPU vs
     TPU on the Pixel.
   - If the SDK route accepts the model, wire it under the same Android
     `CompiledModel` runtime surface and keep CPU fallback.

4. GPU fallback: useful where NPU is absent or rejects the model.
   - Use LiteRT GPU where available.
   - Do not make GPU a release blocker; it is a middle tier between NPU and CPU.

## Export Delegates

`export.delegates` accepts:

- `cpu`
- `gpu`
- `npu`
- `tpu`
- `qualcomm`
- `mediatek`
- `intel`
- `google_tensor`
- `all`

The named NPU targets are packaging and validation intent. They all resolve to
the LiteRT NPU backend class at runtime, but keeping the names in metadata makes
CI and device reports explicit.

## Build Order

1. Keep the fixed-shape TFLite contract passing in Python export tests.
2. Keep the fixture-quality check green on the exported TFLite, not only the
   PyTorch checkpoint. The fixture config must use `training.padding:
   runtime_static`; otherwise checkpoint quality can look good while the Android
   artifact regresses because the export has no runtime padding mask.
   When checkpoint and TFLite quality disagree, run `inherent-parity` before
   retraining so padding drift and TFLite quantization drift are visible
   separately.
3. Add Android `CompiledModel` backend in the app with priority:
   `google_tensor`/`npu` -> `gpu` -> `cpu`.
4. Run JIT device smoke on Pixel and Xiaomi:
   - model loads
   - first inference succeeds
   - repeated inference uses cache
   - output shape and finite scores match CPU within a calibrated tolerance
5. Run AOT only on a Linux x86_64 AVX host or CI runner. Apple Silicon emulation
   is not a valid AOT compiler environment for the current vendor binaries.
6. Store compiled artifacts outside git unless the model storage policy changes.

## Current Official References

- LiteRT Android overview: https://ai.google.dev/edge/litert/android
- LiteRT NPU overview: https://ai.google.dev/edge/litert/next/npu
- Google Tensor ML SDK experimental access: https://ai.google.dev/edge/litert/next/tensor_ml_sdk
- Qualcomm NPU setup: https://ai.google.dev/edge/litert/next/qualcomm
- MediaTek NPU setup: https://ai.google.dev/edge/litert/next/mediatek
- Intel NPU setup: https://ai.google.dev/edge/litert/next/intel
