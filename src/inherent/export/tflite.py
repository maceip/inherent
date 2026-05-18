"""Compatibility wrapper for the TFLite export backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import Config
from .core import artifact_path as _artifact_path  # noqa: F401 - compatibility for existing tests/imports
from .core import write_artifact_metadata
from .litert import export_to_tflite


def write_metadata(
    checkpoint_path: Path,
    cfg: Config,
    tflite_path: Path,
    metadata_path: Path,
    default_thresholds: dict[str, float] | None = None,
) -> None:
    write_artifact_metadata(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        artifact_path=tflite_path,
        metadata_path=metadata_path,
        backend="tflite",
        artifact_format="tflite",
        default_thresholds=default_thresholds,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-dir", default="artifacts", type=Path)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_to_tflite(args.checkpoint, cfg, args.output_dir)


if __name__ == "__main__":
    main()
