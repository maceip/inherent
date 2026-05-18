"""LiteRT-LM packaging backend backed by the real LiteRT-LM builder."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .. import HEAD_ORDER, THRESHOLD_KEYS
from ..config import Config
from .core import (
    ExportResult,
    INPUT_TENSOR_NAME,
    OUTPUT_TENSOR_NAME,
    artifact_path,
    write_artifact_metadata,
    write_json_report,
)
from .litert import export_to_tflite

_DEFAULT_LITERTLM_ROOT = "~/LiteRT-LM"
_BUILDER_BIN_ENV = "INHERENT_LITERTLM_BUILDER_BIN"
_ALLOW_BAZEL_ENV = "INHERENT_LITERTLM_ALLOW_BAZEL"
_BUILDER_TIMEOUT_ENV = "INHERENT_LITERTLM_BUILDER_TIMEOUT_SECONDS"
_DEFAULT_BUILDER_TIMEOUT_SECONDS = 30.0
_DELEGATE_TO_LITERTLM_BACKEND = {
    "cpu": "cpu",
    "gpu": "gpu",
    "npu": "npu",
    # LiteRT-LM names accelerator-class execution "npu"; the named targets are
    # metadata-level requests that must still be validated on hardware.
    "tpu": "npu",
    "qualcomm": "npu",
    "mediatek": "npu",
    "intel": "npu",
    "google_tensor": "npu",
}


class LiteRTLMBackend:
    name = "litertlm"

    def export(
        self,
        *,
        checkpoint_path: Path,
        cfg: Config,
        output_dir: Path,
        delegates: tuple[str, ...] = (),
    ) -> ExportResult:
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        package_path = artifact_path(output_dir, cfg.export.litertlm_path, "inherent.litertlm")
        selected_delegates = _normalize_delegates(delegates or tuple(cfg.export.delegates))
        backend_constraint = _backend_constraint(selected_delegates)
        tflite_result = export_to_tflite(
            checkpoint_path=checkpoint_path,
            cfg=cfg,
            output_dir=output_dir,
            backend_name=self.name,
        )
        tflite_path = Path(tflite_result.artifacts["tflite"])
        toml_path = output_dir / "litertlm_package.toml"
        report_path = output_dir / "reports" / "litertlm_packaging.json"
        write_packaging_toml(
            toml_path=toml_path,
            tflite_path=tflite_path,
            cfg=cfg,
            delegates=selected_delegates,
            backend_constraint=backend_constraint,
        )
        builder = run_litertlm_builder(toml_path, package_path, report_path)
        metadata_path = package_path.with_suffix(".metadata.json")
        reports = {
            **(tflite_result.reports or {}),
            "packaging": str(report_path),
        }
        write_artifact_metadata(
            checkpoint_path=Path(checkpoint_path).expanduser(),
            cfg=cfg,
            artifact_path=package_path,
            metadata_path=metadata_path,
            backend=self.name,
            artifact_format="litertlm",
            reports=reports,
            extra={
                "source_tflite": str(tflite_path),
                "packaging_toml": str(toml_path),
                "delegates": selected_delegates,
                "litertlm_backend_constraint": backend_constraint,
                "litertlm_builder": builder,
            },
        )
        return ExportResult(
            backend=self.name,
            artifacts={
                "litertlm": str(package_path),
                "tflite_source": str(tflite_path),
                "packaging_toml": str(toml_path),
            },
            metadata_path=str(metadata_path),
            reports=reports,
        )


def write_packaging_toml(
    *,
    toml_path: Path,
    tflite_path: Path,
    cfg: Config,
    delegates: tuple[str, ...],
    backend_constraint: str,
) -> Path:
    """Writes the TOML consumed by the official LiteRT-LM builder."""
    metadata = {
        "model_role": "inherent_audio_intent_classifier",
        "input_tensor": INPUT_TENSOR_NAME,
        "output_tensor": OUTPUT_TENSOR_NAME,
        "head_order": json.dumps(list(HEAD_ORDER)),
        "threshold_keys_in_order": json.dumps(list(THRESHOLD_KEYS)),
        "quantization": cfg.export.quantization,
        "mel_bins": str(cfg.model.mel_bins),
        "max_frames": str(cfg.model.max_frames),
        "requested_delegates": ",".join(delegates),
        "litertlm_backend_constraint": backend_constraint,
    }
    lines = [
        "[system_metadata]",
        "entries = [",
        _inline_toml_entry("Author", "String", "inherent"),
        _inline_toml_entry("PackageRole", "String", "audio_intent_classifier"),
        "]",
        "",
        "[[section]]",
        'section_type = "TFLiteModel"',
        'model_type = "AUX"',
        f"data_path = {_toml_string(str(Path(tflite_path).resolve()))}",
        f"backend_constraint = {_toml_string(backend_constraint)}",
        "additional_metadata = [",
        *[_inline_toml_entry(key, "String", value) for key, value in metadata.items()],
        "]",
        "",
    ]
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text("\n".join(lines))
    return toml_path


def run_litertlm_builder(toml_path: Path, package_path: Path, report_path: Path) -> dict[str, Any]:
    """Invokes the installed builder CLI or the local Bazel target in ~/LiteRT-LM."""
    invocation = resolve_litertlm_builder_command(toml_path, package_path)
    package_path.parent.mkdir(parents=True, exist_ok=True)
    command_trace = shlex.join(invocation["command"])
    timeout_seconds = invocation["timeout_seconds"]
    try:
        completed = subprocess.run(
            invocation["command"],
            cwd=invocation["cwd"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.CalledProcessError as exc:
        report = {
            "backend": "litertlm",
            "status": "failed",
            "builder": invocation,
            "command_trace": command_trace,
            "returncode": exc.returncode,
            "stdout_tail": _tail(exc.stdout),
            "stderr_tail": _tail(exc.stderr),
        }
        write_json_report(report_path, report)
        raise RuntimeError(
            "LiteRT-LM packaging failed.\n"
            f"command trace: {command_trace}\n"
            f"error trace: {_tail(exc.stderr) or _tail(exc.stdout)}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        report = {
            "backend": "litertlm",
            "status": "failed",
            "builder": invocation,
            "command_trace": command_trace,
            "timeout_seconds": timeout_seconds,
            "stdout_tail": _tail(exc.stdout),
            "stderr_tail": _tail(exc.stderr),
            "reason": "builder timed out before producing a package",
        }
        write_json_report(report_path, report)
        raise RuntimeError(
            "LiteRT-LM packaging failed.\n"
            f"command trace: {command_trace}\n"
            f"error trace: builder timed out after {timeout_seconds:g}s"
        ) from exc

    if not package_path.is_file() or package_path.stat().st_size == 0:
        report = {
            "backend": "litertlm",
            "status": "failed",
            "builder": invocation,
            "command_trace": command_trace,
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
            "reason": f"builder completed but did not create a non-empty package at {package_path}",
        }
        write_json_report(report_path, report)
        raise RuntimeError(
            "LiteRT-LM packaging failed.\n"
            f"command trace: {command_trace}\n"
            f"error trace: {report['reason']}"
        )

    report = {
        "backend": "litertlm",
        "status": "passed",
        "builder": invocation,
        "command_trace": command_trace,
        "artifact": str(package_path),
        "artifact_size_bytes": package_path.stat().st_size,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }
    write_json_report(report_path, report)
    return invocation


def resolve_litertlm_builder_command(toml_path: Path, package_path: Path) -> dict[str, Any]:
    timeout_seconds = _builder_timeout_seconds()
    builder_bin = _builder_binary()
    if builder_bin:
        return {
            "kind": "python_package_binary",
            "command": [
                builder_bin,
                "toml",
                "--path",
                str(toml_path),
                "output",
                "--path",
                str(package_path),
            ],
            "cwd": None,
            "timeout_seconds": timeout_seconds,
        }

    repo_root = Path(os.environ.get("INHERENT_LITERTLM_ROOT", _DEFAULT_LITERTLM_ROOT)).expanduser()
    if not repo_root.is_dir():
        raise RuntimeError(
            "LiteRT-LM builder not found: install the litert-lm-builder package "
            f"or set INHERENT_LITERTLM_ROOT to the local LiteRT-LM checkout (default {repo_root})"
        )
    if os.environ.get(_ALLOW_BAZEL_ENV, "").strip() not in {"1", "true", "yes"}:
        raise RuntimeError(
            "LiteRT-LM builder binary not found. Install the litert-lm-builder package, "
            f"set {_BUILDER_BIN_ENV} to a builder executable, or explicitly set "
            f"{_ALLOW_BAZEL_ENV}=1 to let inherent invoke the Bazel target in {repo_root}. "
            "Bazel fallback is opt-in because cold Bazel startup can spend minutes in repository setup."
        )
    bazel_bin = shutil.which("bazelisk") or shutil.which("bazel")
    if bazel_bin is None:
        raise RuntimeError(
            "LiteRT-LM builder not found: install litert-lm-builder, or install bazel/bazelisk "
            f"so inherent can run the builder target in {repo_root}"
        )
    return {
        "kind": "local_litertlm_bazel_target",
        "command": [
            bazel_bin,
            "run",
            "//python/litert_lm_builder:litertlm_builder_cli",
            "--",
            "toml",
            "--path",
            str(Path(toml_path).resolve()),
            "output",
            "--path",
            str(Path(package_path).resolve()),
        ],
        "cwd": str(repo_root),
        "timeout_seconds": timeout_seconds,
    }


def _builder_binary() -> str | None:
    configured = os.environ.get(_BUILDER_BIN_ENV, "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file() and os.access(configured_path, os.X_OK):
            return str(configured_path)
        resolved = shutil.which(configured)
        if resolved is not None:
            return resolved
        raise RuntimeError(f"{_BUILDER_BIN_ENV} does not point to an executable: {configured}")
    return shutil.which("litert-lm-builder")


def _builder_timeout_seconds() -> float:
    raw_value = os.environ.get(_BUILDER_TIMEOUT_ENV, "").strip()
    if not raw_value:
        return _DEFAULT_BUILDER_TIMEOUT_SECONDS
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{_BUILDER_TIMEOUT_ENV} must be a number of seconds") from exc
    if value <= 0:
        raise ValueError(f"{_BUILDER_TIMEOUT_ENV} must be positive")
    return value


def _normalize_delegates(delegates: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(delegate.lower() for delegate in delegates)
    if not normalized:
        return ("cpu",)
    if "all" in normalized:
        return ("cpu", "gpu", "npu", "tpu")
    return tuple(dict.fromkeys(normalized))


def _backend_constraint(delegates: tuple[str, ...]) -> str:
    backends: list[str] = []
    for delegate in delegates:
        if delegate not in _DELEGATE_TO_LITERTLM_BACKEND:
            raise ValueError(f"unsupported LiteRT-LM delegate {delegate!r}")
        backends.append(_DELEGATE_TO_LITERTLM_BACKEND[delegate])
    return ",".join(dict.fromkeys(backends))


def _inline_toml_entry(key: str, value_type: str, value: str) -> str:
    return (
        "  { key = "
        f"{_toml_string(key)}, value_type = {_toml_string(value_type)}, value = {_toml_string(value)}"
        " },"
    )


def _toml_string(value: str) -> str:
    return json.dumps(str(value))


def _tail(text: str | None, *, max_chars: int = 4000) -> str:
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    return text[-max_chars:]
