"""First-class ONNX export backend."""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from .core import (
    ExportResult,
    artifact_path,
    export_onnx,
    load_export_model,
    verify_onnx,
    write_artifact_metadata,
    write_json_report,
)


class ONNXBackend:
    name = "onnx"

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
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = artifact_path(output_dir, cfg.export.onnx_path, "inherent.onnx")
        metadata_path = artifact_path(output_dir, cfg.export.onnx_metadata_path, "inherent.onnx.metadata.json")
        parity_report_path = output_dir / "reports" / "onnx_parity.json"

        model = load_export_model(checkpoint_path, cfg)
        export_onnx(model, cfg, onnx_path)
        parity = verify_onnx(onnx_path, model, cfg)
        write_json_report(
            parity_report_path,
            {
                "backend": self.name,
                "artifact": str(onnx_path),
                "status": "passed",
                **parity,
            },
        )
        reports = {"parity": str(parity_report_path)}
        write_artifact_metadata(
            checkpoint_path=checkpoint_path,
            cfg=cfg,
            artifact_path=onnx_path,
            metadata_path=metadata_path,
            backend=self.name,
            artifact_format="onnx",
            reports=reports,
            extra={"onnx_opset": cfg.export.onnx_opset},
        )
        return ExportResult(
            backend=self.name,
            artifacts={"onnx": str(onnx_path)},
            metadata_path=str(metadata_path),
            reports=reports,
        )
