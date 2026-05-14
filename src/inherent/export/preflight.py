"""Fast, read-only export backend preflight checks."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

from ..config import Config
from .litertlm import resolve_litertlm_builder_command
from .registry import list_backends


def selected_backend_names(cfg: Config, backend_name: str | None) -> tuple[str, ...]:
    name = backend_name or cfg.export.backend
    if name == "all":
        return tuple(backend for backend in list_backends() if backend != "tflite")
    return (name,)


def preflight_backends(
    *,
    cfg: Config,
    backend_names: tuple[str, ...],
    delegates: tuple[str, ...],
    output_dir: Path,
) -> list[dict[str, Any]]:
    return [
        preflight_backend(cfg=cfg, backend_name=name, delegates=delegates, output_dir=output_dir)
        for name in backend_names
    ]


def preflight_backend(
    *,
    cfg: Config,
    backend_name: str,
    delegates: tuple[str, ...],
    output_dir: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    if backend_name == "onnx":
        checks.extend(_module_checks("torch", "onnx", "onnxruntime", "numpy"))
    elif backend_name == "tflite":
        checks.extend(_tflite_checks(cfg))
    elif backend_name == "litert":
        checks.extend(_tflite_checks(cfg))
        checks.append(_delegate_check(delegates or tuple(cfg.export.delegates)))
    elif backend_name == "mlx":
        checks.extend(_module_checks("torch", "numpy"))
        checks.append(_optional_module_check("mlx", "target-side MLX parity validation will be skipped until mlx is installed"))
    elif backend_name == "litertlm":
        checks.extend(_tflite_checks(cfg))
        checks.append(_delegate_check(delegates or tuple(cfg.export.delegates)))
        checks.append(_litertlm_builder_check(output_dir))
    else:
        checks.append({"name": "backend", "status": "failed", "reason": f"unknown export backend {backend_name!r}"})

    return {
        "backend": backend_name,
        "status": _overall_status(checks),
        "checks": checks,
    }


def _tflite_checks(cfg: Config) -> list[dict[str, Any]]:
    checks = [
        *_module_checks("torch", "onnx", "onnxruntime", "numpy", "tensorflow"),
        _command_check("onnx2tf"),
    ]
    if cfg.export.strict_tensor_names:
        checks.append(_command_check("flatc"))
    return checks


def _module_checks(*names: str) -> list[dict[str, Any]]:
    return [_module_check(name) for name in names]


def _module_check(name: str) -> dict[str, Any]:
    if importlib.util.find_spec(name) is None:
        return {"name": f"python:{name}", "status": "failed", "reason": f"missing Python package {name}"}
    return {"name": f"python:{name}", "status": "passed"}


def _optional_module_check(name: str, missing_reason: str) -> dict[str, Any]:
    if importlib.util.find_spec(name) is None:
        return {"name": f"python:{name}", "status": "warning", "reason": missing_reason}
    return {"name": f"python:{name}", "status": "passed"}


def _command_check(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    if path is None:
        return {"name": f"command:{name}", "status": "failed", "reason": f"missing command {name}"}
    return {"name": f"command:{name}", "status": "passed", "path": path}


def _delegate_check(delegates: tuple[str, ...]) -> dict[str, Any]:
    allowed = {"cpu", "gpu", "tpu", "all"}
    normalized = tuple(delegate.lower() for delegate in delegates) or ("cpu",)
    unexpected = sorted(set(normalized) - allowed)
    if unexpected:
        return {"name": "delegates", "status": "failed", "reason": f"unsupported delegates: {unexpected}"}
    expanded = ("cpu", "gpu", "tpu") if "all" in normalized else tuple(dict.fromkeys(normalized))
    return {"name": "delegates", "status": "passed", "delegates": expanded}


def _litertlm_builder_check(output_dir: Path) -> dict[str, Any]:
    try:
        invocation = resolve_litertlm_builder_command(
            output_dir / "litertlm_package.toml",
            output_dir / "inherent.litertlm",
        )
    except Exception as exc:
        return {"name": "litertlm_builder", "status": "failed", "reason": str(exc)}
    return {
        "name": "litertlm_builder",
        "status": "passed",
        "kind": invocation["kind"],
        "command": invocation["command"],
        "cwd": invocation["cwd"],
        "timeout_seconds": invocation["timeout_seconds"],
    }


def _overall_status(checks: list[dict[str, Any]]) -> str:
    statuses = {check["status"] for check in checks}
    if "failed" in statuses:
        return "failed"
    if "warning" in statuses:
        return "warning"
    return "passed"
