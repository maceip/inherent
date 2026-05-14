"""LiteRT/TFLite export backends and delegate compatibility reports."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .. import HEAD_ORDER
from ..config import Config
from .core import (
    ExportResult,
    INPUT_TENSOR_NAME,
    OUTPUT_TENSOR_NAME,
    artifact_path,
    export_onnx,
    fit_frames,
    load_export_model,
    representative_dataset,
    verify_onnx,
    write_artifact_metadata,
    write_json_report,
)


class TFLiteBackend:
    name = "tflite"

    def export(
        self,
        *,
        checkpoint_path: Path,
        cfg: Config,
        output_dir: Path,
        delegates: tuple[str, ...] = (),
    ) -> ExportResult:
        del delegates
        return export_to_tflite(checkpoint_path, cfg, output_dir, backend_name=self.name)


class LiteRTBackend:
    name = "litert"

    def export(
        self,
        *,
        checkpoint_path: Path,
        cfg: Config,
        output_dir: Path,
        delegates: tuple[str, ...] = (),
    ) -> ExportResult:
        selected = _normalize_delegates(delegates or tuple(cfg.export.delegates))
        result = export_to_tflite(checkpoint_path, cfg, output_dir, backend_name=self.name)
        artifact = Path(result.artifacts["tflite"])
        report_paths = dict(result.reports or {})
        delegate_dir = Path(output_dir).expanduser() / "delegates"
        for delegate in selected:
            report = _delegate_report(delegate, artifact, cfg)
            path = write_json_report(delegate_dir / f"{delegate}.json", report)
            report_paths[f"delegate_{delegate}"] = str(path)
        metadata_path = Path(result.metadata_path) if result.metadata_path else artifact.with_suffix(".metadata.json")
        write_artifact_metadata(
            checkpoint_path=Path(checkpoint_path).expanduser(),
            cfg=cfg,
            artifact_path=artifact,
            metadata_path=metadata_path,
            backend=self.name,
            artifact_format="tflite",
            reports=report_paths,
            extra={"delegates": selected},
        )
        return ExportResult(
            backend=self.name,
            artifacts=result.artifacts,
            metadata_path=str(metadata_path),
            reports=report_paths,
        )


def export_to_tflite(
    checkpoint_path: Path,
    cfg: Config,
    output_dir: Path,
    *,
    backend_name: str = "tflite",
) -> ExportResult:
    """PyTorch -> ONNX -> TFLite int8 with representative dataset for activation calibration."""
    if cfg.export.quantization != "int8":
        raise ValueError(f"unsupported export.quantization {cfg.export.quantization!r}; only 'int8' is allowed")
    checkpoint_path = Path(checkpoint_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if shutil.which("onnx2tf") is None:
        raise RuntimeError("onnx2tf CLI is required for TFLite/LiteRT export")
    if cfg.export.strict_tensor_names and shutil.which("flatc") is None:
        raise RuntimeError(
            "flatc is required for strict TFLite tensor-name rewriting; "
            "install flatc or set export.strict_tensor_names=false only for smoke/debug exports"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "inherent.onnx"
    saved_model_dir = output_dir / "saved_model"
    tflite_path = artifact_path(output_dir, cfg.export.output_path, "inherent.tflite")
    metadata_path = artifact_path(output_dir, cfg.export.metadata_path, "inherent.metadata.json")

    model = load_export_model(checkpoint_path, cfg)
    export_onnx(model, cfg, onnx_path)
    parity = verify_onnx(onnx_path, model, cfg)
    parity_report = write_json_report(
        output_dir / "reports" / "onnx_parity.json",
        {"backend": backend_name, "artifact": str(onnx_path), "status": "passed", **parity},
    )
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    convert_onnx_to_saved_model(onnx_path, saved_model_dir, cfg)
    convert_saved_model_to_tflite(saved_model_dir, cfg, tflite_path)
    verify_tflite(tflite_path, cfg)
    verify_size(tflite_path, cfg.export.target_size_mb)
    reports = {"onnx_parity": str(parity_report)}
    write_artifact_metadata(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        artifact_path=tflite_path,
        metadata_path=metadata_path,
        backend=backend_name,
        artifact_format="tflite",
        reports=reports,
    )
    return ExportResult(
        backend=backend_name,
        artifacts={"tflite": str(tflite_path), "onnx_intermediate": str(onnx_path), "saved_model": str(saved_model_dir)},
        metadata_path=str(metadata_path),
        reports=reports,
    )


def convert_onnx_to_saved_model(onnx_path: Path, saved_model_dir: Path, cfg: Config) -> None:
    command = [
        "onnx2tf",
        "-i",
        str(onnx_path),
        "-o",
        str(saved_model_dir),
        "-osd",
        "-coion",
        "-dsm",
        "-kt",
        INPUT_TENSOR_NAME,
        "-b",
        "1",
    ]
    if cfg.export.onnx_static_frames is not None:
        command.extend([
            "-ois",
            f"{INPUT_TENSOR_NAME}:1,{cfg.export.onnx_static_frames},{cfg.model.mel_bins}",
        ])
    subprocess.run(command, check=True)


def convert_saved_model_to_tflite(saved_model_dir: Path, cfg: Config, tflite_path: Path) -> None:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("tensorflow is required for TFLite export") from exc

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(cfg)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_model = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)


def verify_tflite(tflite_path: Path, cfg: Config) -> None:
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("tensorflow and numpy are required to verify TFLite export") from exc

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1:
        raise ValueError(f"TFLite model must have one input tensor, got {len(input_details)}")
    if len(output_details) != 1:
        raise ValueError(f"TFLite model must have one output tensor, got {len(output_details)}")
    input_detail = input_details[0]
    output_detail = output_details[0]
    if cfg.export.strict_tensor_names and INPUT_TENSOR_NAME not in input_detail["name"]:
        raise ValueError(f"TFLite input tensor name must contain {INPUT_TENSOR_NAME!r}, got {input_detail['name']}")
    if cfg.export.strict_tensor_names and OUTPUT_TENSOR_NAME not in output_detail["name"]:
        raise ValueError(f"TFLite output tensor name must contain {OUTPUT_TENSOR_NAME!r}, got {output_detail['name']}")
    if input_detail["dtype"] != np.float32:
        raise TypeError(f"TFLite input dtype must be float32, got {input_detail['dtype']}")
    if output_detail["dtype"] != np.float32:
        raise TypeError(f"TFLite output dtype must be float32, got {output_detail['dtype']}")

    input_shape = tuple(int(dim) for dim in input_detail["shape"])
    if len(input_shape) != 3 or input_shape[0] != 1 or input_shape[2] != cfg.model.mel_bins:
        raise ValueError(f"TFLite input shape must be [1,T,{cfg.model.mel_bins}], got {input_shape}")
    output_shape = tuple(int(dim) for dim in output_detail["shape"])
    if output_shape[-1:] != (len(HEAD_ORDER),):
        raise ValueError(f"TFLite output last dimension must be {len(HEAD_ORDER)}, got {output_shape}")

    sample_frames = cfg.export.onnx_static_frames or cfg.export.onnx_sample_frames
    sample = np.zeros((1, sample_frames, cfg.model.mel_bins), dtype=np.float32)
    interpreter.resize_tensor_input(input_detail["index"], sample.shape, strict=False)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_detail["index"], sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])
    if output.shape != (1, len(HEAD_ORDER)):
        raise ValueError(f"TFLite invocation output shape must be [1,{len(HEAD_ORDER)}], got {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("TFLite invocation produced non-finite output")


def verify_size(tflite_path: Path, target_size_mb: int) -> None:
    size_mb = tflite_path.stat().st_size / (1024 * 1024)
    if size_mb > target_size_mb:
        raise ValueError(f"{tflite_path} is {size_mb:.2f} MB; target is <= {target_size_mb} MB")


def _normalize_delegates(delegates: tuple[str, ...]) -> tuple[str, ...]:
    if not delegates:
        return ("cpu",)
    if "all" in delegates:
        return ("cpu", "gpu", "tpu")
    return tuple(dict.fromkeys(delegates))


def _delegate_report(delegate: str, artifact: Path, cfg: Config) -> dict:
    base = {
        "delegate": delegate,
        "artifact": str(artifact),
        "input_tensor": INPUT_TENSOR_NAME,
        "output_tensor": OUTPUT_TENSOR_NAME,
        "quantization": cfg.export.quantization,
        "static_frames": cfg.export.onnx_static_frames,
    }
    if delegate == "cpu":
        return {**base, "status": "passed", "reason": "validated by local TFLite CPU interpreter"}
    if delegate == "gpu":
        return {
            **base,
            "status": "skipped",
            "reason": "LiteRT GPU delegate validation requires target GPU delegate libraries at runtime",
        }
    if delegate == "tpu":
        status = "skipped"
        reason = "Edge TPU validation requires compiler/delegate and a fixed-shape full-int8 compatible model"
        if cfg.export.onnx_static_frames is None:
            reason = "TPU delegate requires export.onnx_static_frames to be set"
        return {**base, "status": status, "reason": reason}
    raise ValueError(f"unsupported delegate {delegate!r}")
