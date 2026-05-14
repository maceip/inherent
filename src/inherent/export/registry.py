"""Backend registry for model export targets."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..config import Config
from .core import ExportResult


class ExportBackend(Protocol):
    name: str

    def export(
        self,
        *,
        checkpoint_path: Path,
        cfg: Config,
        output_dir: Path,
        delegates: tuple[str, ...] = (),
    ) -> ExportResult:
        ...


_BACKENDS: dict[str, ExportBackend] = {}


def register_backend(backend: ExportBackend) -> None:
    if backend.name in _BACKENDS:
        raise ValueError(f"export backend already registered: {backend.name}")
    _BACKENDS[backend.name] = backend


def get_backend(name: str) -> ExportBackend:
    _ensure_default_backends()
    try:
        return _BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"unknown export backend {name!r}; available={sorted(_BACKENDS)}") from exc


def list_backends() -> tuple[str, ...]:
    _ensure_default_backends()
    return tuple(sorted(_BACKENDS))


def _ensure_default_backends() -> None:
    if _BACKENDS:
        return
    from .litert import LiteRTBackend, TFLiteBackend
    from .litertlm import LiteRTLMBackend
    from .mlx import MLXBackend
    from .onnx import ONNXBackend

    for backend in (
        ONNXBackend(),
        TFLiteBackend(),
        LiteRTBackend(),
        MLXBackend(),
        LiteRTLMBackend(),
    ):
        register_backend(backend)
