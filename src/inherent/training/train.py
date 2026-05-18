"""Training loop for the joint audio→intent classifier.

CLI: `inherent-train --config configs/base.yaml`
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from functools import partial
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as _mp
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    _mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from .. import HEAD_ORDER
from ..config import Config
from ..eval.evaluate import evaluate_gates, evaluate_per_head
from ..models import JointAudioIntentModel
from .dataset import (
    MelBatch,
    MelManifestDataset,
    collate_mel_batches,
    compute_balanced_pos_weight,
    compute_label_balanced_sample_weights,
)


@dataclass(frozen=True)
class EvalResult:
    loss: float
    metrics: dict[str, dict[str, float]] | None = None
    gates: dict | None = None
    metrics_error: str | None = None


def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-head focal binary cross-entropy. logits: [B, 13], targets: [B, 13] (float)."""
    probs = torch.sigmoid(logits)
    eps = 1e-7
    pt = torch.where(targets > 0.5, probs, 1 - probs).clamp(min=eps, max=1 - eps)
    focal_weight = (1 - pt) ** gamma
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    return (focal_weight * bce).mean()


def train(
    cfg: Config,
    output_dir: Path,
    *,
    init_checkpoint: Path | None = None,
) -> None:
    """Train the joint audio conformer from precomputed mel manifests."""
    _set_seed(cfg.training.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(_config_dict(cfg), indent=2))

    device = _select_device(cfg.training.device)
    train_dataset = MelManifestDataset(
        cfg.training.train_manifest,
        mel_bins=cfg.model.mel_bins,
        max_frames=cfg.model.max_frames,
    )
    eval_dataset = (
        MelManifestDataset(
            cfg.training.eval_manifest,
            mel_bins=cfg.model.mel_bins,
            max_frames=cfg.model.max_frames,
        )
        if cfg.training.eval_manifest
        else None
    )
    sampler = _make_train_sampler(train_dataset, cfg)
    collate_fn = _collate_fn(cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=False,
        persistent_workers=(cfg.training.num_workers > 0),
        collate_fn=collate_fn,
        drop_last=True,
    )
    if len(train_loader) == 0:
        raise ValueError("train manifest does not contain enough samples for one full batch")

    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=False,
            persistent_workers=(cfg.training.num_workers > 0),
            collate_fn=collate_fn,
            drop_last=False,
        )

    model = JointAudioIntentModel(cfg.model).to(device)
    init_checkpoint = Path(init_checkpoint).expanduser() if init_checkpoint is not None else None
    if init_checkpoint is not None:
        _load_model_checkpoint(model, cfg, init_checkpoint)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_scale(step, cfg.training.warmup_steps, cfg.training.max_steps),
    )
    pos_weight = (
        compute_balanced_pos_weight(train_dataset).to(device)
        if cfg.training.class_weights == "balanced"
        else None
    )

    best_eval_loss = float("inf")
    best_release_key: tuple[float, ...] | None = None
    loader_iter = cycle(train_loader)
    for step in range(1, cfg.training.max_steps + 1):
        model.train()
        batch = _move_batch(next(loader_iter), device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.mel, lengths=_model_lengths(batch, cfg))
        loss = focal_bce_loss(
            logits,
            batch.targets,
            gamma=cfg.training.focal_gamma,
            pos_weight=pos_weight,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % cfg.training.save_every_steps == 0:
            print(
                json.dumps(
                    {
                        "event": "train_step",
                        "step": step,
                        "max_steps": cfg.training.max_steps,
                        "train_loss": float(loss.item()),
                    }
                ),
                flush=True,
            )
            _save_checkpoint(
                output_dir / "last.pt",
                cfg,
                model,
                optimizer,
                scheduler,
                step,
                loss.item(),
                init_checkpoint=init_checkpoint,
            )
        if eval_loader is not None and (step == 1 or step % cfg.training.eval_every_steps == 0):
            eval_result = _evaluate(model, eval_loader, cfg, device, pos_weight)
            eval_loss = eval_result.loss
            print(
                json.dumps(
                    {
                        "event": "eval_step",
                        "step": step,
                        "max_steps": cfg.training.max_steps,
                        "eval_loss": float(eval_loss),
                        "release_selection_key": _release_selection_key(
                            eval_result.metrics,
                            eval_result.gates,
                            eval_loss,
                        ),
                        "gate_passed": None if eval_result.gates is None else eval_result.gates["passed"],
                        "metric_summary": _metric_summary(eval_result.metrics),
                        "metrics_error": eval_result.metrics_error,
                        "best_eval_loss": None
                        if best_eval_loss == float("inf")
                        else float(best_eval_loss),
                    }
                ),
                flush=True,
            )
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                _save_checkpoint(
                    output_dir / "best_loss.pt",
                    cfg,
                    model,
                    optimizer,
                    scheduler,
                    step,
                    loss.item(),
                    eval_loss=eval_loss,
                    init_checkpoint=init_checkpoint,
                )
            release_key = _release_selection_key(eval_result.metrics, eval_result.gates, eval_loss)
            if best_release_key is None or release_key > best_release_key:
                best_release_key = release_key
                _save_checkpoint(
                    output_dir / "best.pt",
                    cfg,
                    model,
                    optimizer,
                    scheduler,
                    step,
                    loss.item(),
                    eval_loss=eval_loss,
                    init_checkpoint=init_checkpoint,
                    selection_key=release_key,
                    gate_result=eval_result.gates,
                    metrics=eval_result.metrics,
                    metrics_error=eval_result.metrics_error,
                )

    _save_checkpoint(
        output_dir / "last.pt",
        cfg,
        model,
        optimizer,
        scheduler,
        cfg.training.max_steps,
        loss.item(),
        init_checkpoint=init_checkpoint,
    )


def _load_model_checkpoint(model: JointAudioIntentModel, cfg: Config, checkpoint_path: Path) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"init checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if tuple(checkpoint.get("head_order", ())) != HEAD_ORDER:
        raise ValueError("init checkpoint head_order does not match inherent.HEAD_ORDER")
    if checkpoint.get("config", {}).get("model") != asdict(cfg.model):
        raise ValueError("init checkpoint model config does not match current config")
    model.load_state_dict(checkpoint["model_state_dict"])


def _make_train_sampler(dataset: MelManifestDataset, cfg: Config) -> WeightedRandomSampler | None:
    if cfg.training.sampler == "shuffle":
        return None
    if cfg.training.sampler != "label_balanced":
        raise ValueError(f"unsupported training.sampler {cfg.training.sampler!r}")
    generator = torch.Generator()
    generator.manual_seed(cfg.training.seed)
    return WeightedRandomSampler(
        weights=compute_label_balanced_sample_weights(dataset),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def _collate_fn(cfg: Config):
    if cfg.training.padding == "runtime_static":
        return partial(collate_mel_batches, fixed_frames=cfg.model.max_frames)
    return collate_mel_batches


def _model_lengths(batch: MelBatch, cfg: Config) -> torch.Tensor | None:
    if cfg.training.padding == "runtime_static":
        return None
    return batch.lengths


def _evaluate(
    model: JointAudioIntentModel,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    pos_weight: torch.Tensor | None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            moved = _move_batch(batch, device)
            logits = model(moved.mel, lengths=_model_lengths(moved, cfg))
            loss = focal_bce_loss(
                logits,
                moved.targets,
                gamma=cfg.training.focal_gamma,
                pos_weight=pos_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite eval loss: {loss.item()}")
            batch_size = moved.mel.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(moved.targets.cpu().numpy())
    if total_samples == 0:
        raise ValueError("eval manifest produced zero samples")
    eval_loss = total_loss / total_samples
    try:
        metrics = evaluate_per_head(np.concatenate(scores, axis=0), np.concatenate(labels, axis=0))
    except ValueError as exc:
        return EvalResult(loss=eval_loss, metrics_error=str(exc))
    gates = evaluate_gates(metrics, cfg.eval.pass_threshold)
    return EvalResult(loss=eval_loss, metrics=metrics, gates=gates)


def _release_selection_key(
    metrics: dict[str, dict[str, float]] | None,
    gates: dict | None,
    eval_loss: float,
) -> tuple[float, ...]:
    if metrics is None:
        return (-1.0, float("-inf"), float("-inf"), float("-inf"), float("-inf"), -eval_loss)
    intent_aucs = [metrics[head]["auc"] for head in HEAD_ORDER[1:]]
    intent_fprs = [metrics[head]["fpr_at_recall_95"] for head in HEAD_ORDER[1:]]
    return (
        1.0 if gates and gates["passed"] else 0.0,
        float(np.min(intent_aucs)),
        float(np.mean(intent_aucs)),
        -float(np.max(intent_fprs)),
        metrics["isInteresting"]["auc"],
        -eval_loss,
    )


def _metric_summary(metrics: dict[str, dict[str, float]] | None) -> dict[str, float] | None:
    if metrics is None:
        return None
    intent_aucs = [metrics[head]["auc"] for head in HEAD_ORDER[1:]]
    intent_fprs = [metrics[head]["fpr_at_recall_95"] for head in HEAD_ORDER[1:]]
    return {
        "is_interesting_auc": metrics["isInteresting"]["auc"],
        "intent_mean_auc": float(np.mean(intent_aucs)),
        "intent_min_auc": float(np.min(intent_aucs)),
        "intent_max_fpr_at_recall_95": float(np.max(intent_fprs)),
    }


def _move_batch(batch: MelBatch, device: torch.device) -> MelBatch:
    return MelBatch(
        mel=batch.mel.to(device, non_blocking=True),
        targets=batch.targets.to(device, non_blocking=True),
        lengths=batch.lengths.to(device, non_blocking=True),
    )


def _select_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training.device is 'cuda' but CUDA is not available")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("training.device is 'mps' but MPS is not available")
        return torch.device("mps")
    raise ValueError(f"unsupported training.device {name!r}")


def _lr_scale(step: int, warmup_steps: int, max_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return max(step, 1) / warmup_steps
    remaining = max(max_steps - step, 0)
    decay_steps = max(max_steps - warmup_steps, 1)
    return max(remaining / decay_steps, 0.0)


def _save_checkpoint(
    path: Path,
    cfg: Config,
    model: JointAudioIntentModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    train_loss: float,
    *,
    eval_loss: float | None = None,
    init_checkpoint: Path | None = None,
    selection_key: tuple[float, ...] | None = None,
    gate_result: dict | None = None,
    metrics: dict[str, dict[str, float]] | None = None,
    metrics_error: str | None = None,
) -> None:
    checkpoint = {
        "step": step,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "selection_key": selection_key,
        "gate_result": gate_result,
        "metrics": metrics,
        "metrics_error": metrics_error,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": _config_dict(cfg),
        "head_order": list(HEAD_ORDER),
        "init_checkpoint": None if init_checkpoint is None else str(init_checkpoint),
    }
    torch.save(checkpoint, path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _config_dict(cfg: Config) -> dict:
    return {
        "model": asdict(cfg.model),
        "data": asdict(cfg.data),
        "training": asdict(cfg.training),
        "export": asdict(cfg.export),
        "eval": asdict(cfg.eval),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-dir", default="artifacts/run")
    parser.add_argument("--train-manifest")
    parser.add_argument("--eval-manifest")
    parser.add_argument("--device", choices=["cuda", "mps"])
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.train_manifest is not None:
        cfg.training.train_manifest = args.train_manifest
    if args.eval_manifest is not None:
        cfg.training.eval_manifest = args.eval_manifest
    if args.device is not None:
        cfg.training.device = args.device
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    cfg.training.__post_init__()
    train(cfg, Path(args.output_dir))


if __name__ == "__main__":
    main()
