"""Per-head evaluation: AUC, EER, FPR-at-95-recall.

CLI: `inherent-eval --checkpoint artifacts/run/best.pt --eval-set data/eval_recorded`
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import HEAD_ORDER
from ..config import ModelConfig
from ..models import JointAudioIntentModel
from ..training.dataset import MelBatch, MelManifestDataset, collate_mel_batches


def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, scores)


def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2)


def fpr_at_recall(scores: np.ndarray, labels: np.ndarray, target_recall: float = 0.95) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = tpr >= target_recall
    if not np.any(valid):
        return 1.0
    return float(np.min(fpr[valid]))


def evaluate_per_head(scores: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, float]]:
    """scores: [N, 13], labels: [N, 13] (binary)."""
    if scores.shape != labels.shape:
        raise ValueError(f"scores and labels must have identical shape, got {scores.shape} vs {labels.shape}")
    if scores.ndim != 2 or scores.shape[1] != len(HEAD_ORDER):
        raise ValueError(f"scores must have shape [N, {len(HEAD_ORDER)}], got {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")
    out = {}
    for i, head in enumerate(HEAD_ORDER):
        unique_labels = np.unique(labels[:, i])
        if not np.array_equal(unique_labels, np.array([0.0, 1.0])):
            raise ValueError(f"{head} labels must contain both 0 and 1 for AUC/EER; got {unique_labels}")
        out[head] = {
            "auc": auc_roc(scores[:, i], labels[:, i]),
            "eer": equal_error_rate(scores[:, i], labels[:, i]),
            "fpr_at_recall_95": fpr_at_recall(scores[:, i], labels[:, i], 0.95),
        }
    return out


def evaluate_checkpoint(
    checkpoint_path: Path,
    eval_manifest: Path,
    *,
    batch_size: int = 32,
    device: str = "cpu",
    runtime_static: bool | None = None,
) -> dict[str, dict[str, float]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = _checkpoint_model_config(checkpoint)
    runtime_static = _runtime_static_for_checkpoint(checkpoint, override=runtime_static)
    runtime_device = _select_device(device)
    model = JointAudioIntentModel(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(runtime_device)
    model.eval()

    dataset = MelManifestDataset(
        eval_manifest,
        mel_bins=model_cfg.mel_bins,
        max_frames=model_cfg.max_frames,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_mel_batches, fixed_frames=model_cfg.max_frames)
        if runtime_static
        else collate_mel_batches,
        drop_last=False,
    )

    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            moved = _move_batch(batch, runtime_device)
            lengths = None if runtime_static else moved.lengths
            batch_scores = model.predict_proba(moved.mel, lengths=lengths)
            scores.append(batch_scores.cpu().numpy())
            labels.append(moved.targets.cpu().numpy())
    if not scores:
        raise ValueError(f"eval manifest produced no batches: {eval_manifest}")
    return evaluate_per_head(np.concatenate(scores, axis=0), np.concatenate(labels, axis=0))


def format_metrics(metrics: dict[str, dict[str, float]]) -> str:
    lines = ["head,auc,eer,fpr_at_recall_95"]
    for head in HEAD_ORDER:
        values = metrics[head]
        lines.append(
            f"{head},{values['auc']:.6f},{values['eer']:.6f},{values['fpr_at_recall_95']:.6f}"
        )
    return "\n".join(lines)


def evaluate_gates(metrics: dict[str, dict[str, float]], pass_threshold: dict) -> dict:
    """Return pass/fail gates for release evaluation metrics."""
    interesting_threshold = pass_threshold.get("is_interesting", {})
    intent_threshold = pass_threshold.get("intent_heads", {})
    intent_aucs = [metrics[head]["auc"] for head in HEAD_ORDER[1:]]
    intent_eers = [metrics[head]["eer"] for head in HEAD_ORDER[1:]]
    intent_fprs = [metrics[head]["fpr_at_recall_95"] for head in HEAD_ORDER[1:]]
    checks = {
        "is_interesting_auc": _check_min(
            metrics["isInteresting"]["auc"],
            interesting_threshold.get("auc"),
        ),
        "is_interesting_eer": _check_max(
            metrics["isInteresting"]["eer"],
            interesting_threshold.get("eer"),
        ),
        "intent_mean_auc": _check_min(float(np.mean(intent_aucs)), intent_threshold.get("mean_auc")),
        "intent_min_auc": _check_min(float(np.min(intent_aucs)), intent_threshold.get("min_auc")),
        "intent_mean_eer": _check_max(float(np.mean(intent_eers)), intent_threshold.get("mean_eer")),
        "intent_max_fpr_at_recall_95": _check_max(
            float(np.max(intent_fprs)),
            intent_threshold.get("max_fpr_at_recall_95"),
        ),
    }
    enabled_checks = {name: check for name, check in checks.items() if check["threshold"] is not None}
    return {
        "passed": all(check["passed"] for check in enabled_checks.values()),
        "checks": enabled_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval-set", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    padding = parser.add_mutually_exclusive_group()
    padding.add_argument("--runtime-static", action="store_true")
    padding.add_argument("--dynamic-padding", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--gate-json-out", type=Path)
    args = parser.parse_args()

    metrics = evaluate_checkpoint(
        args.checkpoint,
        args.eval_set,
        batch_size=args.batch_size,
        device=args.device,
        runtime_static=True if args.runtime_static else False if args.dynamic_padding else None,
    )
    print(format_metrics(metrics))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(metrics, indent=2))
    if args.gate_json_out is not None:
        if args.config is None:
            raise SystemExit("--gate-json-out requires --config")
        from ..config import Config

        cfg = Config.load(args.config)
        args.gate_json_out.write_text(json.dumps(evaluate_gates(metrics, cfg.eval.pass_threshold), indent=2))


def _check_min(value: float, threshold: float | None) -> dict:
    return {
        "value": value,
        "threshold": threshold,
        "passed": True if threshold is None else value >= threshold,
        "direction": "min",
    }


def _check_max(value: float, threshold: float | None) -> dict:
    return {
        "value": value,
        "threshold": threshold,
        "passed": True if threshold is None else value <= threshold,
        "direction": "max",
    }


def _checkpoint_model_config(checkpoint: dict) -> ModelConfig:
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint missing model_state_dict")
    if "config" not in checkpoint or "model" not in checkpoint["config"]:
        raise ValueError("checkpoint missing config.model")
    return ModelConfig(**checkpoint["config"]["model"])


def _runtime_static_for_checkpoint(checkpoint: dict, *, override: bool | None) -> bool:
    if override is not None:
        return override
    training = checkpoint.get("config", {}).get("training", {})
    padding = training.get("padding", "dynamic")
    if padding not in {"dynamic", "runtime_static"}:
        raise ValueError(f"checkpoint config.training.padding is invalid: {padding!r}")
    return padding == "runtime_static"


def _select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested CUDA evaluation but CUDA is not available")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("requested MPS evaluation but MPS is not available")
        return torch.device("mps")
    raise ValueError(f"unsupported eval device {name!r}")


def _move_batch(batch: MelBatch, device: torch.device) -> MelBatch:
    return MelBatch(
        mel=batch.mel.to(device),
        targets=batch.targets.to(device),
        lengths=batch.lengths.to(device),
    )


if __name__ == "__main__":
    main()
