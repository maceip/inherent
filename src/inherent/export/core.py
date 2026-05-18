"""Shared export utilities and artifact metadata."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .. import DEFAULT_THRESHOLDS_BY_KEY, HEAD_ORDER, THRESHOLD_KEYS, __version__
from ..config import Config

INPUT_TENSOR_NAME = "mel_spectrogram"
OUTPUT_TENSOR_NAME = "intent_output"


@dataclass(frozen=True)
class ExportResult:
    backend: str
    artifacts: dict[str, str]
    metadata_path: str | None = None
    reports: dict[str, str] | None = None
    supported: bool = True


def load_export_model(checkpoint_path: Path, cfg: Config):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for export") from exc

    from ..models import JointAudioIntentInferenceModel, JointAudioIntentModel

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    required_keys = {"model_state_dict", "head_order", "config"}
    missing = required_keys - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint missing required keys for export: {sorted(missing)}")
    if tuple(checkpoint["head_order"]) != HEAD_ORDER:
        raise ValueError("checkpoint head_order does not match inherent.HEAD_ORDER")
    checkpoint_model_cfg = checkpoint["config"].get("model")
    if checkpoint_model_cfg != asdict(cfg.model):
        raise ValueError("checkpoint model config does not match export config")

    model = JointAudioIntentModel(cfg.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return JointAudioIntentInferenceModel(model).eval()


def export_onnx(model, cfg: Config, onnx_path: Path) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for ONNX export") from exc

    dummy_frames = cfg.export.onnx_static_frames or cfg.export.onnx_sample_frames
    dummy = torch.zeros(1, dummy_frames, cfg.model.mel_bins, dtype=torch.float32)
    export_kwargs: dict[str, Any] = {
        "input_names": [INPUT_TENSOR_NAME],
        "output_names": [OUTPUT_TENSOR_NAME],
        "opset_version": cfg.export.onnx_opset,
        "do_constant_folding": True,
        "dynamo": False,
    }
    if cfg.export.onnx_static_frames is None:
        export_kwargs["dynamic_axes"] = {
            INPUT_TENSOR_NAME: {1: "frames"},
            OUTPUT_TENSOR_NAME: {},
        }
    torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs)


def verify_onnx(onnx_path: Path, model, cfg: Config) -> dict[str, Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise RuntimeError("onnx, onnxruntime, numpy, and torch are required to verify ONNX export") from exc

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    frames = cfg.export.onnx_static_frames or cfg.export.onnx_sample_frames
    sample = np.zeros((1, frames, cfg.model.mel_bins), dtype=np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(sample)).detach().cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run([OUTPUT_TENSOR_NAME], {INPUT_TENSOR_NAME: sample})[0]
    if onnx_out.shape != (1, len(HEAD_ORDER)):
        raise ValueError(f"ONNX output shape must be [1,{len(HEAD_ORDER)}], got {onnx_out.shape}")
    max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))
    if max_abs_diff > cfg.export.parity_atol:
        raise ValueError(f"ONNX export drift too high: max_abs_diff={max_abs_diff}")
    return {"max_abs_diff": max_abs_diff, "frames": frames, "provider": "CPUExecutionProvider"}


def write_artifact_metadata(
    *,
    checkpoint_path: Path,
    cfg: Config,
    artifact_path: Path,
    metadata_path: Path,
    backend: str,
    artifact_format: str,
    reports: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
    default_thresholds: dict[str, float] | None = None,
) -> None:
    if default_thresholds is None:
        default_thresholds = DEFAULT_THRESHOLDS_BY_KEY
    validate_thresholds(default_thresholds)
    metadata = {
        "version": __version__,
        "backend": backend,
        "artifact_format": artifact_format,
        "training_hash": compute_training_hash(checkpoint_path, cfg),
        "input_tensor": INPUT_TENSOR_NAME,
        "output_tensor": OUTPUT_TENSOR_NAME,
        "head_order": list(HEAD_ORDER),
        "threshold_keys_in_order": list(THRESHOLD_KEYS),
        "default_thresholds": default_thresholds,
        "artifact_size_bytes": artifact_path.stat().st_size if artifact_path.is_file() else None,
        "reports": reports or {},
        "config": {
            "model": asdict(cfg.model),
            "training": asdict(cfg.training),
            "export": asdict(cfg.export),
        },
    }
    if extra:
        metadata.update(extra)
    if artifact_format == "tflite" and artifact_path.is_file():
        metadata["tflite_size_bytes"] = artifact_path.stat().st_size
        tflite_io = inspect_tflite_io_contract(artifact_path)
        validate_tflite_io_contract(tflite_io, cfg)
        metadata["tflite_io"] = tflite_io
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))


def compute_training_hash(checkpoint_path: Path, cfg: Config) -> str:
    h = hashlib.sha256()
    h.update(checkpoint_path.read_bytes())
    h.update(json.dumps(asdict(cfg.model), sort_keys=True).encode())
    hash_manifest_if_present(h, cfg.training.train_manifest)
    if cfg.training.eval_manifest is not None:
        hash_manifest_if_present(h, cfg.training.eval_manifest)
    revision = git_sha()
    return f"{revision}:{h.hexdigest()[:16]}"


def hash_manifest_if_present(h: hashlib._Hash, manifest_path: str) -> None:
    path = Path(manifest_path).expanduser()
    if path.is_file():
        h.update(path.read_bytes())


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def validate_thresholds(default_thresholds: dict[str, float]) -> None:
    expected = set(THRESHOLD_KEYS)
    actual = set(default_thresholds)
    if actual != expected:
        raise ValueError(
            f"default_thresholds keys must match THRESHOLD_KEYS; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    for key, value in default_thresholds.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"default threshold {key} must be in [0, 1], got {value}")


def inspect_tflite_io_contract(tflite_path: Path) -> dict[str, Any]:
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("tensorflow and numpy are required to inspect TFLite metadata") from exc

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
    return {
        "input_name": str(input_detail["name"]),
        "input_shape": [int(dim) for dim in input_detail["shape"]],
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "output_name": str(output_detail["name"]),
        "output_shape": [int(dim) for dim in output_detail["shape"]],
        "output_dtype": np.dtype(output_detail["dtype"]).name,
    }


def validate_tflite_io_contract(tflite_io: dict[str, Any], cfg: Config) -> None:
    input_shape = _shape_from_contract(tflite_io, "input_shape")
    output_shape = _shape_from_contract(tflite_io, "output_shape")
    input_dtype = str(tflite_io.get("input_dtype", ""))
    output_dtype = str(tflite_io.get("output_dtype", ""))

    expected_input = (1, cfg.model.max_frames, cfg.model.mel_bins)
    expected_output = (1, len(HEAD_ORDER))
    if input_shape != expected_input:
        raise ValueError(
            "TFLite input shape must match metadata model.max_frames: "
            f"expected {list(expected_input)}, got {list(input_shape)}"
        )
    if input_dtype != "float32":
        raise TypeError(f"TFLite input dtype must be float32, got {input_dtype}")
    if output_shape != expected_output:
        raise ValueError(f"TFLite output shape must be {list(expected_output)}, got {list(output_shape)}")
    if output_dtype != "float32":
        raise TypeError(f"TFLite output dtype must be float32, got {output_dtype}")


def _shape_from_contract(tflite_io: dict[str, Any], key: str) -> tuple[int, ...]:
    if key not in tflite_io:
        raise ValueError(f"TFLite IO contract missing {key}")
    try:
        return tuple(int(dim) for dim in tflite_io[key])
    except TypeError as exc:
        raise ValueError(f"TFLite IO contract {key} must be a sequence of ints") from exc


def artifact_path(output_dir: Path, configured: str, default_name: str) -> Path:
    configured_path = Path(configured).expanduser()
    if configured_path.is_absolute():
        return configured_path
    name = configured_path.name or default_name
    return output_dir / name


def fit_frames(mel, frames: int):
    import numpy as np

    if mel.shape[0] > frames:
        return mel[:frames]
    if mel.shape[0] < frames:
        pad = np.zeros((frames - mel.shape[0], mel.shape[1]), dtype=mel.dtype)
        return np.concatenate([mel, pad], axis=0)
    return mel


def representative_dataset(cfg: Config):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for representative export data") from exc

    from ..training.dataset import MelManifestDataset

    dataset = MelManifestDataset(
        cfg.training.train_manifest,
        mel_bins=cfg.model.mel_bins,
        max_frames=cfg.model.max_frames,
    )
    indexes = representative_sample_indexes(dataset, cfg.export.representative_dataset_size)
    if not indexes:
        raise ValueError("representative_dataset_size must select at least one sample")
    for index in indexes:
        mel, _, _ = dataset[index]
        sample = mel.numpy().astype(np.float32)
        if cfg.export.onnx_static_frames is not None:
            sample = fit_frames(sample, cfg.export.onnx_static_frames)
        yield {INPUT_TENSOR_NAME: np.expand_dims(sample, axis=0)}


def representative_sample_indexes(dataset, max_count: int) -> list[int]:
    """Select deterministic, label-stratified rows for int8 calibration.

    The exported Android model is fixed-shape and sees one sample at a time.
    Feeding the first N manifest rows can easily miss rare intent heads or
    negative-only audio, which makes int8 calibration unstable. This selector
    first round-robins positive examples per head, then includes negative-only
    rows, then fills any remaining budget evenly over the manifest order.
    """

    import numpy as np

    if max_count < 1:
        raise ValueError("representative_dataset_size must be positive")
    count = min(max_count, len(dataset))
    if count < 1:
        return []
    labels = dataset.label_matrix().numpy()
    selected: list[int] = []
    selected_set: set[int] = set()

    def add(index: int) -> bool:
        if index in selected_set:
            return False
        selected.append(index)
        selected_set.add(index)
        return True

    negative_only = [int(index) for index in np.flatnonzero(labels.sum(axis=1) == 0)]
    negative_budget = min(len(negative_only), max(1, count // 5)) if negative_only else 0
    positive_budget = count - negative_budget
    positive_indexes = [
        _evenly_spaced_indexes([int(index) for index in np.flatnonzero(labels[:, head_index] >= 0.5)])
        for head_index in range(len(HEAD_ORDER))
    ]
    cursors = [0 for _ in positive_indexes]
    while len(selected) < positive_budget:
        progressed = False
        for head_index, indexes in enumerate(positive_indexes):
            while cursors[head_index] < len(indexes) and indexes[cursors[head_index]] in selected_set:
                cursors[head_index] += 1
            if cursors[head_index] >= len(indexes):
                continue
            progressed = add(indexes[cursors[head_index]]) or progressed
            cursors[head_index] += 1
            if len(selected) >= count:
                break
        if not progressed:
            break

    for index in _evenly_spaced_indexes(negative_only, max_count=count - len(selected)):
        if len(selected) >= count:
            break
        add(index)

    for index in _evenly_spaced_indexes(list(range(len(dataset))), max_count=count - len(selected)):
        if len(selected) >= count:
            break
        add(index)
    return selected


def _evenly_spaced_indexes(indexes: list[int], *, max_count: int | None = None) -> list[int]:
    if max_count is not None and max_count <= 0:
        return []
    if len(indexes) <= 2:
        return indexes[:max_count]
    target = len(indexes) if max_count is None else min(max_count, len(indexes))
    ordered_positions: list[int] = []
    pending = deque([(0, len(indexes) - 1)])
    while pending and len(ordered_positions) < target:
        start, end = pending.popleft()
        if start > end:
            continue
        middle = (start + end) // 2
        ordered_positions.append(middle)
        pending.append((start, middle - 1))
        pending.append((middle + 1, end))
    return [indexes[position] for position in ordered_positions]


def write_json_report(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path
