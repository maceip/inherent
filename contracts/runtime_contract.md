# Inherent Runtime Contract

This is the load-bearing interface between `inherent` (Python training project) and the genesis Android app's `InherentBackend`. Both sides depend on this contract; changes require coordination.

## Model Contract

The model contract is backend-independent. Every export backend must preserve:

- input tensor semantics: `mel_spectrogram`, `[1, T, 128]`, `float32`
- output tensor semantics: `intent_output`, `[1, 13]`, `float32`
- output head order and threshold keys below

For the Android TFLite/LiteRT artifact, `T` is fixed in the binary to 3000.
Shorter frontend outputs must be zero-padded before invocation.

## TFLite/LiteRT Artifact

A single TFLite file plus a metadata sidecar:

- `inherent.tflite` — joint audio→intent classifier. The release default is
  float16 TFLite for quality parity; int8 remains a performance target only
  when export-time TFLite parity gates pass on a held-out manifest.
- `inherent.metadata.json` — head names, calibrated thresholds, runtime tensor
  indexes/names, version, training hash

Additional export backends can produce:

- `inherent.onnx` plus `inherent.onnx.metadata.json` for ONNX Runtime.
- a LiteRT/TFLite artifact plus delegate reports under `delegates/*.json`.
- an MLX package with exported weights and an Apple Silicon runtime scaffold.
- a `.litertlm` package produced by the official LiteRT-LM builder from
  `~/LiteRT-LM` or the installed `litert-lm-builder` package binary. The
  classifier is packaged as a `tf_lite_aux` LiteRT-LM section with CPU/GPU/TPU
  delegate constraints recorded in metadata. Cold Bazel fallback is explicit
  opt-in via `INHERENT_LITERTLM_ALLOW_BAZEL=1`; normal builds should use the
  package binary or `INHERENT_LITERTLM_BUILDER_BIN` so export fails fast when
  the builder is unavailable.

## Input tensor

| Name | Shape | dtype | Semantics |
|---|---|---|---|
| `mel_spectrogram` | `[1, 3000, 128]` | `float32` | Output of cosmo's existing `audio_frontend.tflite`, zero-padded to 3000 frames when shorter. 128-bin mel at 20 ms hop. Matches the EdgeTPU teacher's input contract exactly. |

The Android side runs the existing `audio_frontend.tflite` first, then feeds its output here. **Do not change the input contract** — it has to stay drop-in-compatible with cosmo's frontend. Do not add a raw-audio input or a padding-mask input on this path.

## Output tensor

| Name | Shape | dtype | Semantics |
|---|---|---|---|
| `intent_output` | `[1, 13]` | `float32` | Sigmoid scores in fixed order (see below). |

## Output head order (fixed)

| Index | Name | Threshold key |
|---|---|---|
| 0 | `isInteresting` | `is_interesting` |
| 1 | `hasAddToListIntent` | `has_add_to_list_intent` |
| 2 | `hasTermSearchQuery` | `has_term_search_query` |
| 3 | `hasPhotoQuery` | `has_photo_query` |
| 4 | `hasCalendarEvent` | `has_calendar_event` |
| 5 | `hasCreateDocIntent` | `has_create_doc_intent` |
| 6 | `hasPersonContext` | `has_person_context` |
| 7 | `hasEventContext` | `has_event_context` |
| 8 | `hasDeepResearchIntent` | `has_deep_research_intent` |
| 9 | `hasInsightIntent` | `has_insight_intent` |
| 10 | `hasBrowsingAgentIntent` | `has_browsing_agent_intent` |
| 11 | `hasCallingAgentIntent` | `has_calling_agent_intent` |
| 12 | `hasStartTimerIntent` | `has_start_timer_intent` |

This order **matches cosmo's `AudioGatekeepingModel$PredictionThresholds` field order** (see `~/Downloads/cosmo_baksmali/.../impl/AudioGatekeepingModel$PredictionThresholds.smali`). Genesis can reuse cosmo's threshold-comparison logic verbatim.

## Metadata sidecar (`inherent.metadata.json`)

```json
{
  "version": "0.1.0",
  "training_hash": "<git sha + data manifest hash>",
  "input_tensor": "mel_spectrogram",
  "output_tensor": "intent_output",
  "runtime_tensor_contract": {
    "selection": "single_input_single_output_index",
    "input": {
      "logical_name": "mel_spectrogram",
      "actual_name": "serving_default_mel_spectrogram:0",
      "index": 0,
      "shape": [1, 3000, 128],
      "dtype": "float32"
    },
    "output": {
      "logical_name": "intent_output",
      "actual_name": "StatefulPartitionedCall:0",
      "index": 42,
      "shape": [1, 13],
      "dtype": "float32"
    }
  },
  "head_order": ["isInteresting", "hasAddToListIntent", ...],
  "default_thresholds": {
    "is_interesting": 0.57,
    "has_add_to_list_intent": 0.65,
    ...
  },
  "threshold_calibration": {
    "mel_manifest": "data/eval_manifest.csv",
    "rows": 1234,
    "score_source": "tflite_runtime_static",
    "min_recall": 0.95
  }
}
```

## Performance budget

- Inference latency on a mid-range Android CPU (e.g. MediaTek MT6991): under 50 ms per 1-second audio window.
- Model size on disk: under 50 MB.
- Memory at inference: under 200 MB working set.

## Versioning

`version` field follows semver. Major bump means head order changed (breaking). Minor bump means model retrained (compatible). Patch bump means metadata-only (e.g. threshold retune).

## Integration

Genesis loads the model via LiteRT/TFLite and binds tensors by
`runtime_tensor_contract`, not by hard-coded converter names. CPU is the
required baseline path; GPU/NPU/TPU delegate use is allowed only after the
export parity report passes for the exact artifact.

## What inherent does NOT provide

- Raw-PCM-to-mel frontend. Genesis already has cosmo's `audio_frontend.tflite` for that.
- The directedness *trigger* signal (separate from the score). That comes from SODA Magic Mic on devices where `inherent` isn't active.
- Pumpkin-style grammar matching. That's text-side and orthogonal to this model.
