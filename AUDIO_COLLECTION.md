# Audio Collection Pipeline

Use `data/labels_template.csv` for hand labeling. Keep one row per clip and
fill `speaker_id`, `session_id`, `device`, and `environment`; splits are grouped
by speaker/session so the same session never lands in both train and eval.

Recommended command flow:

```shell
inherent-labels validate --manifest data/recorded_labels.csv --json-out artifacts/label_report.json
inherent-labels normalize --manifest data/recorded_labels.csv --output-manifest data/recorded_normalized_manifest.csv --audio-dir data/recorded_normalized_audio
inherent-labels split --manifest data/recorded_normalized_manifest.csv --output-dir data/splits
inherent-prep-data --config configs/production.yaml --target mels --input-manifest data/splits/train_manifest.csv --output-manifest data/train_manifest.csv --mel-dir data/mels/train --frontend-model data/audio_frontend.tflite
inherent-prep-data --config configs/production.yaml --target mels --input-manifest data/splits/eval_manifest.csv --output-manifest data/eval_manifest.csv --mel-dir data/mels/eval --frontend-model data/audio_frontend.tflite
```

Production model-group flow:

```shell
PYTHONPATH=src python -m inherent.scripts.build_recorded \
  --config configs/production.yaml \
  --labels data/recorded_labels.csv \
  --model-group-dir artifacts/model-groups/001 \
  --frontend-model data/audio_frontend.tflite \
  --export-backend all \
  --device mps \
  --eval-device cpu
```

The first successful run creates the first model group. It performs strict label
validation, normalization, split creation or explicit split preservation, raw
manifest assembly, Cosmo frontend mel materialization, training, evaluation, and
export. It saves the generated train/eval/test split under the model group; you
do not need to bring a separate eval set.

Iteration flow:

```shell
PYTHONPATH=src python -m inherent.scripts.build_recorded \
  --config configs/production.yaml \
  --labels data/recorded_labels_expanded.csv \
  --model-group-dir artifacts/model-groups/002 \
  --previous-model-group artifacts/model-groups/001 \
  --frontend-model data/audio_frontend.tflite \
  --export-backend all \
  --device mps \
  --eval-device cpu
```

Same intent heads means pass `--previous-model-group`; the build reuses that
group's eval/test audio identities, puts new labeled clips into train, and
warm-starts from the previous checkpoint. Changed intent heads means start a new
model-group line and do not pass `--previous-model-group`.

Use `--export-backend all` to emit ONNX, LiteRT/TFLite delegate reports, MLX
package scaffolding, and the LiteRT-LM compatibility/package result in one run.
If `split` is set in the first label CSV it must be set on every row and must
not split a speaker/session group across train/eval/test.

Weights for collection:

- 40% ambient/non-directed negatives, all heads `0`.
- 40% real positive intent commands, balanced across the 12 intent heads.
- 20% hard negatives and near misses, all heads `0` unless the assistant should actually act.

Prioritize `hasCallingAgentIntent`, `hasPersonContext`, `hasEventContext`,
`hasPhotoQuery`, `hasDeepResearchIntent`, and `hasBrowsingAgentIntent` first.
