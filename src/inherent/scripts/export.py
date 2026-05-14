"""Multi-backend model export CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inherent.config import Config
from inherent.export.preflight import preflight_backends, selected_backend_names
from inherent.export.registry import get_backend, list_backends

__all__ = ["main"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-dir", "--artifact-dir", default="artifacts", type=Path)
    parser.add_argument("--backend", choices=[*list_backends(), "all"])
    parser.add_argument("--delegate", action="append", help="LiteRT delegate: cpu, gpu, tpu, or all")
    parser.add_argument("--preflight", action="store_true", help="Check backend dependencies without exporting")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    backend_name = args.backend or cfg.export.backend
    delegates = tuple(_split_csv(args.delegate)) or tuple(cfg.export.delegates)
    names = selected_backend_names(cfg, backend_name)
    if args.preflight:
        results = preflight_backends(
            cfg=cfg,
            backend_names=names,
            delegates=delegates,
            output_dir=args.output_dir,
        )
        payload = results[0] if len(results) == 1 else results
        print(json.dumps(payload, indent=2))
        if any(result["status"] == "failed" for result in results):
            raise SystemExit(2)
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --preflight is set")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in names:
        backend = get_backend(name)
        results.append(
            backend.export(
                checkpoint_path=args.checkpoint,
                cfg=cfg,
                output_dir=args.output_dir / name if backend_name == "all" else args.output_dir,
                delegates=delegates,
            ).__dict__
        )
    print(json.dumps(results[0] if len(results) == 1 else results, indent=2))


def _split_csv(values: list[str] | None) -> list[str]:
    if not values:
        return []
    output: list[str] = []
    for value in values:
        output.extend(item.strip() for item in value.split(",") if item.strip())
    return output


if __name__ == "__main__":
    main()
