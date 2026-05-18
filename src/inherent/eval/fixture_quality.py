"""Evaluate fixture-gate quality for the multi-label audio head contract."""

from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import HEAD_ORDER
from ..config import ModelConfig
from ..models import JointAudioIntentModel
from ..training.dataset import MelManifestDataset, collate_mel_batches


def evaluate_fixture_quality(
    *,
    checkpoint_path: str | Path,
    mel_manifest: str | Path,
    source_filter: str | None = "gatekeeper_fixture:existing",
    interesting_threshold: float = 0.5,
    batch_size: int = 16,
    runtime_static: bool = True,
) -> dict[str, Any]:
    """Score fixture rows using the multi-label contract.

    For intent rows, success means:
    - `isInteresting` is above threshold, and
    - the expected intent wins among the 12 intent heads.

    For the pure `isInteresting` row, success means head 0 wins all-head top-1.
    This avoids the misleading all-head top-1 metric for intent rows, where a
    correct model should often assign both `isInteresting` and an intent high
    scores.
    """
    manifest = Path(mel_manifest).expanduser()
    rows = list(csv.DictReader(manifest.open()))
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
    model = JointAudioIntentModel(ModelConfig(**checkpoint["config"]["model"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    max_frames = int(model.backbone.max_frames)
    dataset = MelManifestDataset(manifest, mel_bins=model.backbone.mel_bins, max_frames=max_frames)
    collate_fn = (
        partial(collate_mel_batches, fixed_frames=max_frames)
        if runtime_static
        else collate_mel_batches
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    scores = []
    with torch.no_grad():
        for batch in loader:
            lengths = None if runtime_static else batch.lengths
            scores.extend(model.predict_proba(batch.mel, lengths=lengths).cpu().numpy())

    return score_fixture_rows(
        rows,
        scores,
        source_filter=source_filter,
        interesting_threshold=interesting_threshold,
    )


def evaluate_tflite_fixture_quality(
    *,
    tflite_path: str | Path,
    mel_manifest: str | Path,
    source_filter: str | None = "gatekeeper_fixture:existing",
    interesting_threshold: float = 0.5,
) -> dict[str, Any]:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("tensorflow is required to evaluate TFLite fixture quality") from exc

    manifest = Path(mel_manifest).expanduser()
    rows = list(csv.DictReader(manifest.open()))
    model_path = Path(tflite_path).expanduser()
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1:
        raise ValueError(f"TFLite model must have one input tensor, got {len(input_details)}")
    if len(output_details) != 1:
        raise ValueError(f"TFLite model must have one output tensor, got {len(output_details)}")
    input_detail = input_details[0]
    output_detail = output_details[0]
    input_shape = tuple(int(value) for value in input_detail["shape"])
    if len(input_shape) != 3 or input_shape[0] != 1:
        raise ValueError(f"TFLite fixture evaluator requires input shape [1,T,mel_bins], got {input_shape}")

    dataset = MelManifestDataset(manifest, mel_bins=input_shape[2], max_frames=input_shape[1])
    scores: list[np.ndarray] = []
    for sample in dataset:
        batch = collate_mel_batches([sample], fixed_frames=input_shape[1])
        interpreter.set_tensor(input_detail["index"], batch.mel.numpy().astype(np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        if output.shape != (1, len(HEAD_ORDER)):
            raise ValueError(f"TFLite output must have shape [1,{len(HEAD_ORDER)}], got {output.shape}")
        scores.append(output[0])

    return score_fixture_rows(
        rows,
        scores,
        source_filter=source_filter,
        interesting_threshold=interesting_threshold,
    )


def score_fixture_rows(
    rows: list[dict[str, str]],
    scores: list[np.ndarray] | list[Any],
    *,
    source_filter: str | None,
    interesting_threshold: float,
) -> dict[str, Any]:
    positive_total = 0
    positive_passed = 0
    rows_out: list[dict[str, Any]] = []
    for row, score in zip(rows, scores, strict=True):
        score = np.asarray(score, dtype=np.float32)
        if source_filter is not None and row.get("source") != source_filter:
            continue
        expected = _expected_positive_index(row)
        if expected is None:
            continue
        all_head_top = int(score.argmax())
        intent_top = 1 + int(score[1:].argmax())
        if expected == 0:
            passed = all_head_top == 0
        else:
            passed = intent_top == expected and float(score[0]) >= interesting_threshold
        positive_total += 1
        positive_passed += int(passed)
        rows_out.append(
            {
                "fixture": row.get("session_id", ""),
                "expected_index": expected,
                "all_head_top_index": all_head_top,
                "intent_top_index": intent_top,
                "is_interesting_score": float(score[0]),
                "expected_score": float(score[expected]),
                "passed": bool(passed),
            }
        )

    return {
        "positive_passed": positive_passed,
        "positive_total": positive_total,
        "rows": rows_out,
    }


def _expected_positive_index(row: dict[str, str]) -> int | None:
    labels = [float(row[head]) for head in HEAD_ORDER]
    if not any(labels):
        return None
    positive_intents = [index for index, value in enumerate(labels[1:], start=1) if value == 1.0]
    if not positive_intents:
        return 0
    if len(positive_intents) > 1:
        raise ValueError(f"fixture quality rows must have at most one positive intent: {positive_intents}")
    return positive_intents[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    artifact = parser.add_mutually_exclusive_group(required=True)
    artifact.add_argument("--checkpoint", type=Path)
    artifact.add_argument("--tflite-model", type=Path)
    parser.add_argument("--mel-manifest", required=True, type=Path)
    parser.add_argument("--source-filter", default="gatekeeper_fixture:existing")
    parser.add_argument("--all-sources", action="store_true")
    parser.add_argument("--interesting-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dynamic-padding", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.checkpoint is not None:
        report = evaluate_fixture_quality(
            checkpoint_path=args.checkpoint,
            mel_manifest=args.mel_manifest,
            source_filter=None if args.all_sources else args.source_filter,
            interesting_threshold=args.interesting_threshold,
            batch_size=args.batch_size,
            runtime_static=not args.dynamic_padding,
        )
    else:
        report = evaluate_tflite_fixture_quality(
            tflite_path=args.tflite_model,
            mel_manifest=args.mel_manifest,
            source_filter=None if args.all_sources else args.source_filter,
            interesting_threshold=args.interesting_threshold,
        )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
