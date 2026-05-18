"""Compare PyTorch dynamic/static scoring with exported TFLite scoring."""

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
from .evaluate import evaluate_per_head


def compare_checkpoint_tflite(
    *,
    checkpoint_path: str | Path | None,
    tflite_path: str | Path | None,
    mel_manifest: str | Path,
    batch_size: int = 32,
    limit: int | None = None,
) -> dict[str, Any]:
    manifest = Path(mel_manifest).expanduser()
    labels = _labels(manifest, limit=limit)
    result: dict[str, Any] = {
        "mel_manifest": str(manifest),
        "rows": int(labels.shape[0]),
        "head_order": list(HEAD_ORDER),
        "scores": {},
        "comparisons": {},
    }
    scores: dict[str, np.ndarray] = {}

    if checkpoint_path is not None:
        dynamic, static = _score_checkpoint(
            Path(checkpoint_path).expanduser(),
            manifest,
            batch_size=batch_size,
            limit=limit,
        )
        scores["checkpoint_dynamic"] = dynamic
        scores["checkpoint_runtime_static"] = static
    if tflite_path is not None:
        scores["tflite_runtime_static"] = _score_tflite(
            Path(tflite_path).expanduser(),
            manifest,
            limit=limit,
        )

    for name, values in scores.items():
        result["scores"][name] = _score_summary(values, labels)
    for left, right in (
        ("checkpoint_dynamic", "checkpoint_runtime_static"),
        ("checkpoint_runtime_static", "tflite_runtime_static"),
        ("checkpoint_dynamic", "tflite_runtime_static"),
    ):
        if left in scores and right in scores:
            result["comparisons"][f"{left}_vs_{right}"] = _diff_summary(scores[left], scores[right])
    return result


def _score_checkpoint(
    checkpoint_path: Path,
    manifest: Path,
    *,
    batch_size: int,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = ModelConfig(**checkpoint["config"]["model"])
    model = JointAudioIntentModel(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = _limited_dataset(manifest, mel_bins=model_cfg.mel_bins, max_frames=model_cfg.max_frames, limit=limit)
    dynamic_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_mel_batches,
    )
    static_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_mel_batches, fixed_frames=model_cfg.max_frames),
    )
    with torch.no_grad():
        dynamic = [
            model.predict_proba(batch.mel, lengths=batch.lengths).cpu().numpy()
            for batch in dynamic_loader
        ]
        static = [
            model.predict_proba(batch.mel, lengths=None).cpu().numpy()
            for batch in static_loader
        ]
    return np.concatenate(dynamic, axis=0), np.concatenate(static, axis=0)


def _score_tflite(tflite_path: Path, manifest: Path, *, limit: int | None) -> np.ndarray:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("tensorflow is required to score TFLite parity") from exc

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_shape = tuple(int(value) for value in input_detail["shape"])
    if len(input_shape) != 3 or input_shape[0] != 1:
        raise ValueError(f"TFLite input must have shape [1,T,mel_bins], got {input_shape}")
    dataset = _limited_dataset(manifest, mel_bins=input_shape[2], max_frames=input_shape[1], limit=limit)
    scores: list[np.ndarray] = []
    for sample in dataset:
        batch = collate_mel_batches([sample], fixed_frames=input_shape[1])
        interpreter.set_tensor(input_detail["index"], batch.mel.numpy().astype(np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        if output.shape != (1, len(HEAD_ORDER)):
            raise ValueError(f"TFLite output must have shape [1,{len(HEAD_ORDER)}], got {output.shape}")
        scores.append(np.asarray(output[0], dtype=np.float32))
    return np.stack(scores, axis=0)


def _limited_dataset(manifest: Path, *, mel_bins: int, max_frames: int, limit: int | None) -> MelManifestDataset:
    dataset = MelManifestDataset(manifest, mel_bins=mel_bins, max_frames=max_frames)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        dataset.rows = dataset.rows[:limit]
    return dataset


def _labels(manifest: Path, *, limit: int | None) -> np.ndarray:
    rows: list[list[float]] = []
    with manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {manifest}")
        missing = [head for head in HEAD_ORDER if head not in reader.fieldnames]
        if missing:
            raise ValueError(f"manifest {manifest} missing label columns: {missing}")
        for row in reader:
            rows.append([float(row[head]) for head in HEAD_ORDER])
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"manifest produced no rows: {manifest}")
    return np.asarray(rows, dtype=np.float32)


def _score_summary(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "shape": list(scores.shape),
        "finite": bool(np.isfinite(scores).all()),
        "score_mean_by_head": _head_map(np.mean(scores, axis=0)),
        "score_std_by_head": _head_map(np.std(scores, axis=0)),
    }
    try:
        metrics = evaluate_per_head(scores, labels)
    except ValueError as exc:
        summary["metrics_error"] = str(exc)
    else:
        summary["metrics"] = metrics
        summary["metric_summary"] = _metric_summary(metrics)
    return summary


def _diff_summary(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    diff = right - left
    abs_diff = np.abs(diff)
    return {
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "mean_signed_diff_by_head": _head_map(np.mean(diff, axis=0)),
        "mean_abs_diff_by_head": _head_map(np.mean(abs_diff, axis=0)),
        "max_abs_diff_by_head": _head_map(np.max(abs_diff, axis=0)),
        "top_changed_rows": _top_changed_rows(abs_diff, limit=10),
    }


def _metric_summary(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    intent_heads = HEAD_ORDER[1:]
    return {
        "is_interesting_auc": metrics["isInteresting"]["auc"],
        "is_interesting_eer": metrics["isInteresting"]["eer"],
        "intent_mean_auc": float(np.mean([metrics[head]["auc"] for head in intent_heads])),
        "intent_min_auc": float(np.min([metrics[head]["auc"] for head in intent_heads])),
        "intent_mean_eer": float(np.mean([metrics[head]["eer"] for head in intent_heads])),
        "intent_max_fpr_at_recall_95": float(
            np.max([metrics[head]["fpr_at_recall_95"] for head in intent_heads])
        ),
    }


def _head_map(values: np.ndarray) -> dict[str, float]:
    return {head: float(value) for head, value in zip(HEAD_ORDER, values, strict=True)}


def _top_changed_rows(abs_diff: np.ndarray, *, limit: int) -> list[dict[str, Any]]:
    row_delta = np.max(abs_diff, axis=1)
    indexes = np.argsort(row_delta)[::-1][:limit]
    return [
        {
            "row_index": int(index),
            "max_abs_diff": float(row_delta[index]),
            "head": HEAD_ORDER[int(np.argmax(abs_diff[index]))],
        }
        for index in indexes
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tflite-model", type=Path)
    parser.add_argument("--mel-manifest", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.checkpoint is None and args.tflite_model is None:
        parser.error("at least one of --checkpoint or --tflite-model is required")

    report = compare_checkpoint_tflite(
        checkpoint_path=args.checkpoint,
        tflite_path=args.tflite_model,
        mel_manifest=args.mel_manifest,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
