"""Build production artifacts from a hand-labeled recorded WAV library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inherent.config import Config
from inherent.pipeline import build_recorded_library


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/production.yaml")
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--model-group-dir", type=Path)
    parser.add_argument("--previous-model-group", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--frontend-model", required=True, type=Path)
    parser.add_argument("--extra-train-manifest", action="append", default=[], type=Path)
    parser.add_argument("--synthetic-train-manifest", action="append", default=[], type=Path)
    parser.add_argument(
        "--export-backend",
        action="append",
        help="Export backend to build: onnx, tflite, litert, mlx, litertlm, or all. May be repeated.",
    )
    parser.add_argument("--device", choices=["cuda", "mps"])
    parser.add_argument("--eval-device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--allow-validation-warnings", action="store_true")
    args = parser.parse_args()
    if args.model_group_dir is None and (args.work_dir is None or args.run_dir is None):
        parser.error("--work-dir and --run-dir are required unless --model-group-dir is set")

    result = build_recorded_library(
        cfg=Config.load(args.config),
        labels_manifest=args.labels,
        work_dir=args.work_dir,
        run_dir=args.run_dir,
        frontend_model=args.frontend_model,
        model_group_dir=args.model_group_dir,
        previous_model_group=args.previous_model_group,
        extra_train_manifests=args.extra_train_manifest,
        synthetic_train_manifests=args.synthetic_train_manifest,
        train=not args.prepare_only,
        evaluate=not args.prepare_only and not args.skip_eval,
        export=not args.prepare_only and not args.skip_export,
        export_backends=args.export_backend,
        training_device=args.device,
        eval_device=args.eval_device,
        max_steps=args.max_steps,
        fail_on_validation_warnings=not args.allow_validation_warnings,
    )
    print(json.dumps(_result_to_json(result), indent=2, sort_keys=True))


def _result_to_json(result) -> dict:
    return {
        "model_group_dir": None if result.model_group_dir is None else str(result.model_group_dir),
        "work_dir": str(result.work_dir),
        "run_dir": str(result.run_dir),
        "normalized_manifest": str(result.normalized_manifest),
        "split_manifests": {key: str(value) for key, value in result.split_manifests.items()},
        "split_identity_index": str(result.split_identity_index),
        "warm_start_checkpoint": None if result.warm_start_checkpoint is None else str(result.warm_start_checkpoint),
        "raw_manifests": {key: str(value) for key, value in result.raw_manifests.items()},
        "mel_manifests": {key: str(value) for key, value in result.mel_manifests.items()},
        "checkpoint": None if result.checkpoint is None else str(result.checkpoint),
        "metrics_json": None if result.metrics_json is None else str(result.metrics_json),
        "metrics_csv": None if result.metrics_csv is None else str(result.metrics_csv),
        "export_dir": None if result.export_dir is None else str(result.export_dir),
        "export_results": result.export_results,
        "model_group_json": None if result.model_group_json is None else str(result.model_group_json),
    }


if __name__ == "__main__":
    main()
