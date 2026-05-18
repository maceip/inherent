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
    inspect_tflite_io_contract,
    load_export_model,
    representative_dataset,
    validate_tflite_io_contract,
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
    """PyTorch -> ONNX -> TFLite with the configured runtime quantization mode."""
    validate_tflite_export_config(cfg)
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
    reports = {"onnx_parity": str(parity_report)}
    tflite_parity_report = maybe_write_tflite_parity_report(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        tflite_path=tflite_path,
        output_dir=output_dir,
        backend_name=backend_name,
    )
    if tflite_parity_report is not None:
        reports["tflite_parity"] = str(tflite_parity_report)
    verify_size(tflite_path, cfg.export.target_size_mb)
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


def maybe_write_tflite_parity_report(
    *,
    checkpoint_path: Path,
    cfg: Config,
    tflite_path: Path,
    output_dir: Path,
    backend_name: str,
) -> Path | None:
    eval_manifest = Path(cfg.training.eval_manifest).expanduser() if cfg.training.eval_manifest else None
    if eval_manifest is None or not eval_manifest.is_file():
        if cfg.export.require_tflite_parity:
            raise FileNotFoundError(
                "TFLite parity is required but training.eval_manifest is missing or does not exist: "
                f"{cfg.training.eval_manifest!r}"
            )
        return None
    require_tflite_parity_thresholds(cfg)

    from ..eval.parity import compare_checkpoint_tflite

    report = compare_checkpoint_tflite(
        checkpoint_path=checkpoint_path,
        tflite_path=tflite_path,
        mel_manifest=eval_manifest,
        batch_size=cfg.training.batch_size,
        limit=cfg.export.tflite_parity_eval_samples,
    )
    report = {
        "backend": backend_name,
        "artifact": str(tflite_path),
        "status": "passed",
        **report,
    }
    report_path = output_dir / "reports" / "tflite_parity.json"
    try:
        enforce_tflite_parity_thresholds(report, cfg)
    except ValueError as exc:
        report["status"] = "failed"
        report["failure"] = str(exc)
        path = write_json_report(report_path, report)
        raise ValueError(f"{exc}; report={path}") from exc
    return write_json_report(report_path, report)


def enforce_tflite_parity_thresholds(report: dict, cfg: Config) -> None:
    require_tflite_parity_thresholds(cfg)
    comparison = report.get("comparisons", {}).get("checkpoint_runtime_static_vs_tflite_runtime_static")
    if comparison is None:
        raise ValueError("TFLite parity report missing checkpoint_runtime_static_vs_tflite_runtime_static comparison")
    max_abs_diff = float(comparison["max_abs_diff"])
    mean_abs_diff = float(comparison["mean_abs_diff"])
    max_threshold = cfg.export.tflite_parity_max_abs_diff
    mean_threshold = cfg.export.tflite_parity_mean_abs_diff
    failures = []
    if max_threshold is not None and max_abs_diff > max_threshold:
        failures.append(f"max_abs_diff={max_abs_diff:.6g} > {max_threshold:.6g}")
    if mean_threshold is not None and mean_abs_diff > mean_threshold:
        failures.append(f"mean_abs_diff={mean_abs_diff:.6g} > {mean_threshold:.6g}")
    if failures:
        raise ValueError("TFLite parity drift exceeds export threshold: " + "; ".join(failures))


def require_tflite_parity_thresholds(cfg: Config) -> None:
    if not cfg.export.require_tflite_parity:
        return
    missing = []
    if cfg.export.tflite_parity_max_abs_diff is None:
        missing.append("tflite_parity_max_abs_diff")
    if cfg.export.tflite_parity_mean_abs_diff is None:
        missing.append("tflite_parity_mean_abs_diff")
    if missing:
        raise ValueError(f"TFLite parity is required but export thresholds are unset: {missing}")


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
    if cfg.export.quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset(cfg)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    elif cfg.export.quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif cfg.export.quantization != "float32":
        raise ValueError(f"unsupported export.quantization {cfg.export.quantization!r}")
    tflite_model = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)


def verify_tflite(tflite_path: Path, cfg: Config) -> dict:
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("tensorflow and numpy are required to verify TFLite export") from exc

    tflite_io = inspect_tflite_io_contract(tflite_path)
    validate_tflite_io_contract(tflite_io, cfg)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1)
    interpreter.allocate_tensors()
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

    sample = np.zeros(tuple(tflite_io["input_shape"]), dtype=np.float32)
    interpreter.set_tensor(input_detail["index"], sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])
    if output.shape != (1, len(HEAD_ORDER)):
        raise ValueError(f"TFLite invocation output shape must be [1,{len(HEAD_ORDER)}], got {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("TFLite invocation produced non-finite output")
    return tflite_io


def validate_tflite_export_config(cfg: Config) -> None:
    if cfg.export.onnx_static_frames != cfg.model.max_frames:
        raise ValueError(
            "TFLite export requires export.onnx_static_frames to equal "
            f"model.max_frames ({cfg.model.max_frames}); got {cfg.export.onnx_static_frames}"
        )


def verify_size(tflite_path: Path, target_size_mb: int) -> None:
    size_mb = tflite_path.stat().st_size / (1024 * 1024)
    if size_mb > target_size_mb:
        raise ValueError(f"{tflite_path} is {size_mb:.2f} MB; target is <= {target_size_mb} MB")


def _normalize_delegates(delegates: tuple[str, ...]) -> tuple[str, ...]:
    if not delegates:
        return ("cpu",)
    if "all" in delegates:
        return ("cpu", "gpu", "npu", "tpu")
    return tuple(dict.fromkeys(delegate.lower() for delegate in delegates))


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
    if delegate in {"npu", "qualcomm", "mediatek", "intel"}:
        status = "skipped"
        reason = (
            "LiteRT NPU validation requires target vendor libraries and a device run through "
            "CompiledModel; this fixed-shape standard-op TFLite is the input artifact"
        )
        return {**base, "status": status, "reason": reason, "runtime_api": "LiteRT CompiledModel"}
    if delegate in {"tpu", "google_tensor"}:
        status = "skipped"
        reason = (
            "Google Tensor TPU validation requires Tensor ML SDK / Pixel device profiling; "
            "this fixed-shape standard-op TFLite is the input artifact"
        )
        if cfg.export.onnx_static_frames is None:
            reason = "TPU delegate requires export.onnx_static_frames to be set"
        return {**base, "status": status, "reason": reason, "runtime_api": "Tensor ML SDK / LiteRT CompiledModel"}
    raise ValueError(f"unsupported delegate {delegate!r}")
