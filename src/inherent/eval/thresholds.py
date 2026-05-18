"""Calibrate runtime thresholds from labeled model scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .. import HEAD_ORDER, THRESHOLD_KEYS
from .evaluate import _runtime_static_for_checkpoint
from .parity import _labels, _score_checkpoint, _score_tflite


def calibrate_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    min_recall: float = 0.95,
    require_all_heads: bool = True,
) -> dict[str, Any]:
    """Pick one threshold per head, maximizing F1 subject to a recall floor."""

    if scores.shape != labels.shape:
        raise ValueError(f"scores and labels must have identical shape, got {scores.shape} vs {labels.shape}")
    if scores.ndim != 2 or scores.shape[1] != len(HEAD_ORDER):
        raise ValueError(f"scores must have shape [N, {len(HEAD_ORDER)}], got {scores.shape}")
    if not 0.0 < min_recall <= 1.0:
        raise ValueError("min_recall must be in (0, 1]")

    heads: dict[str, dict[str, float]] = {}
    thresholds_by_key: dict[str, float] = {}
    thresholds_by_head: dict[str, float] = {}
    skipped_heads: dict[str, str] = {}
    for index, (head, key) in enumerate(zip(HEAD_ORDER, THRESHOLD_KEYS, strict=True)):
        try:
            head_result = _calibrate_head(scores[:, index], labels[:, index], min_recall=min_recall)
        except ValueError as exc:
            if require_all_heads:
                raise ValueError(f"{head}: {exc}") from exc
            skipped_heads[head] = str(exc)
            continue
        heads[head] = head_result
        thresholds_by_key[key] = head_result["threshold"]
        thresholds_by_head[head] = head_result["threshold"]
    return {
        "min_recall": min_recall,
        "thresholds_by_key": thresholds_by_key,
        "thresholds_by_head": thresholds_by_head,
        "heads": heads,
        "skipped_heads": skipped_heads,
    }


def _calibrate_head(scores: np.ndarray, labels: np.ndarray, *, min_recall: float) -> dict[str, float]:
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")
    labels = labels.astype(np.float32)
    positives = int(labels.sum())
    negatives = int(labels.shape[0] - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("threshold calibration requires at least one positive and one negative label per head")

    candidates = np.unique(np.concatenate(([0.0, 1.0], scores.astype(np.float32))))
    best: dict[str, float] | None = None
    best_key: tuple[float, float, float] | None = None
    for threshold in candidates:
        predicted = scores >= threshold
        true_positive = int(np.logical_and(predicted, labels == 1.0).sum())
        false_positive = int(np.logical_and(predicted, labels == 0.0).sum())
        false_negative = positives - true_positive
        recall = true_positive / positives
        if recall < min_recall:
            continue
        precision = true_positive / max(true_positive + false_positive, 1)
        f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
        fpr = false_positive / negatives
        key = (f1, -fpr, float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fpr": float(fpr),
                "true_positive": float(true_positive),
                "false_positive": float(false_positive),
                "false_negative": float(false_negative),
            }
    if best is None:
        raise ValueError("no threshold satisfied the requested recall floor")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    artifact = parser.add_mutually_exclusive_group(required=True)
    artifact.add_argument("--checkpoint", type=Path)
    artifact.add_argument("--tflite-model", type=Path)
    parser.add_argument("--mel-manifest", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--allow-missing-heads", action="store_true")
    padding = parser.add_mutually_exclusive_group()
    padding.add_argument("--runtime-static", action="store_true")
    padding.add_argument("--dynamic-padding", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    manifest = args.mel_manifest.expanduser()
    labels = _labels(manifest, limit=args.limit)
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.expanduser()
        dynamic, static = _score_checkpoint(
            checkpoint_path,
            manifest,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        runtime_static = _runtime_static_for_checkpoint(
            checkpoint,
            override=True if args.runtime_static else False if args.dynamic_padding else None,
        )
        scores = static if runtime_static else dynamic
        score_source = "checkpoint_runtime_static" if runtime_static else "checkpoint_dynamic"
    else:
        scores = _score_tflite(args.tflite_model.expanduser(), manifest, limit=args.limit)
        score_source = "tflite_runtime_static"

    report = {
        "mel_manifest": str(manifest),
        "rows": int(labels.shape[0]),
        "score_source": score_source,
        **calibrate_thresholds(
            scores,
            labels,
            min_recall=args.min_recall,
            require_all_heads=not args.allow_missing_heads,
        ),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
