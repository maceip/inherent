"""Apple MLX package export backend."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import Config
from .core import ExportResult, load_export_model, write_artifact_metadata, write_json_report


class MLXBackend:
    name = "mlx"

    def export(
        self,
        *,
        checkpoint_path: Path,
        cfg: Config,
        output_dir: Path,
        delegates: tuple[str, ...] = (),
    ) -> ExportResult:
        del delegates
        checkpoint_path = Path(checkpoint_path).expanduser()
        package_dir = _package_dir(output_dir, cfg)
        package_dir.mkdir(parents=True, exist_ok=True)
        weights_path = package_dir / "weights.npz"
        manifest_path = package_dir / "manifest.json"
        runtime_path = package_dir / "inference.py"
        parity_report_path = package_dir / "reports" / "mlx_parity.json"

        model = load_export_model(checkpoint_path, cfg)
        _write_weights_npz(model, weights_path)
        runtime_path.write_text(_runtime_source())
        manifest = {
            "backend": self.name,
            "artifact_format": "mlx-package",
            "weights": weights_path.name,
            "runtime": runtime_path.name,
            "input_tensor": "mel_spectrogram",
            "output_tensor": "intent_output",
            "requires": ["mlx"],
            "model": cfg.model.__dict__,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        parity = _mlx_parity_status()
        write_json_report(parity_report_path, parity)
        reports = {"parity": str(parity_report_path)}
        write_artifact_metadata(
            checkpoint_path=checkpoint_path,
            cfg=cfg,
            artifact_path=weights_path,
            metadata_path=package_dir / "inherent.mlx.metadata.json",
            backend=self.name,
            artifact_format="mlx-package",
            reports=reports,
            extra={"package_dir": str(package_dir), "parity": parity},
        )
        return ExportResult(
            backend=self.name,
            artifacts={"mlx_package": str(package_dir), "weights": str(weights_path), "runtime": str(runtime_path)},
            metadata_path=str(package_dir / "inherent.mlx.metadata.json"),
            reports=reports,
            supported=parity["status"] != "failed",
        )


def _package_dir(output_dir: Path, cfg: Config) -> Path:
    configured = Path(cfg.export.mlx_dir).expanduser()
    if configured.is_absolute():
        return configured
    return Path(output_dir).expanduser() / configured.name


def _write_weights_npz(model, path: Path) -> None:
    import numpy as np

    arrays = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in model.state_dict().items()
        if tensor.dtype.is_floating_point
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _mlx_parity_status() -> dict:
    try:
        import mlx  # noqa: F401
    except ImportError:
        return {
            "backend": "mlx",
            "status": "skipped",
            "reason": "MLX is not installed; package contains weights and runtime scaffold for Apple Silicon validation",
        }
    return {
        "backend": "mlx",
        "status": "pending",
        "reason": "MLX is installed; generated package is ready for target-side parity validation",
    }


def _runtime_source() -> str:
    return '''"""Generated MLX runtime scaffold for inherent.

This package stores exported weights in `weights.npz`. The Python exporter
validates PyTorch parity before producing the package; target-side MLX parity
should be run on Apple Silicon with the `mlx` package installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_weights(package_dir: str | Path) -> dict[str, np.ndarray]:
    """Load exported inherent weights as NumPy arrays for MLX conversion."""
    weights = Path(package_dir) / "weights.npz"
    with np.load(weights) as data:
        return {name: data[name] for name in data.files}
'''
