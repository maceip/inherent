import json
import sys
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inherent import HEAD_ORDER, THRESHOLD_KEYS
from inherent.config import Config, ExportConfig, TrainingConfig
from inherent.export import core, tflite
from inherent.export.core import representative_sample_indexes, validate_tflite_io_contract
from inherent.export.core import load_export_model
from inherent.export.litert import (
    convert_saved_model_to_tflite,
    enforce_tflite_parity_thresholds,
    maybe_write_tflite_parity_report,
    require_tflite_parity_thresholds,
    validate_tflite_export_config,
)


def _tflite_io(
    *,
    input_shape=(1, 3000, 128),
    input_dtype="float32",
    output_shape=(1, 13),
    output_dtype="float32",
):
    return {
        "input_index": 0,
        "input_name": "serving_default_mel_spectrogram:0",
        "input_shape": list(input_shape),
        "input_dtype": input_dtype,
        "output_index": 42,
        "output_name": "StatefulPartitionedCall:0",
        "output_shape": list(output_shape),
        "output_dtype": output_dtype,
    }


def test_artifact_path_uses_output_dir_name():
    path = tflite._artifact_path(
        output_dir=tflite.Path("artifacts/v0.1.0"),
        configured="artifacts/inherent.tflite",
        default_name="inherent.tflite",
    )

    assert path == tflite.Path("artifacts/v0.1.0/inherent.tflite")


def test_write_metadata_writes_contract_fields(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    model = tmp_path / "inherent.tflite"
    model.write_bytes(b"tflite")
    metadata = tmp_path / "inherent.metadata.json"
    monkeypatch.setattr(core, "inspect_tflite_io_contract", lambda _path: _tflite_io())

    tflite.write_metadata(checkpoint, Config(), model, metadata)

    data = json.loads(metadata.read_text())
    assert data["input_tensor"] == "mel_spectrogram"
    assert data["output_tensor"] == "intent_output"
    assert tuple(data["head_order"]) == HEAD_ORDER
    assert tuple(data["threshold_keys_in_order"]) == THRESHOLD_KEYS
    assert data["tflite_size_bytes"] == len(b"tflite")
    assert data["runtime_tensor_contract"]["selection"] == "single_input_single_output_index"
    assert data["runtime_tensor_contract"]["input"]["logical_name"] == "mel_spectrogram"
    assert data["runtime_tensor_contract"]["input"]["index"] == 0
    assert data["runtime_tensor_contract"]["output"]["logical_name"] == "intent_output"
    assert data["runtime_tensor_contract"]["output"]["actual_name"] == "StatefulPartitionedCall:0"
    assert data["runtime_tensor_contract"]["output"]["index"] == 42
    assert data["tflite_io"]["input_shape"] == [1, 3000, 128]
    assert data["tflite_io"]["output_shape"] == [1, 13]
    assert ":" in data["training_hash"]


def test_write_metadata_rejects_tflite_shape_that_disagrees_with_model_max_frames(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    model = tmp_path / "inherent.tflite"
    model.write_bytes(b"tflite")
    metadata = tmp_path / "inherent.metadata.json"
    monkeypatch.setattr(core, "inspect_tflite_io_contract", lambda _path: _tflite_io(input_shape=(1, 50, 128)))

    with pytest.raises(ValueError, match=r"expected \[1, 3000, 128\]"):
        tflite.write_metadata(checkpoint, Config(), model, metadata)


def test_tflite_io_contract_requires_drop_in_android_shape():
    validate_tflite_io_contract(_tflite_io(), Config())

    with pytest.raises(ValueError, match=r"expected \[1, 3000, 128\]"):
        validate_tflite_io_contract(_tflite_io(input_shape=(1, 50, 128)), Config())

    with pytest.raises(ValueError, match=r"TFLite output shape"):
        validate_tflite_io_contract(_tflite_io(output_shape=(1, 12)), Config())

    with pytest.raises(TypeError, match=r"input dtype must be float32"):
        validate_tflite_io_contract(_tflite_io(input_dtype="int8"), Config())


def test_tflite_export_requires_static_max_frames():
    validate_tflite_export_config(Config())

    with pytest.raises(ValueError, match=r"model.max_frames \(3000\); got 50"):
        validate_tflite_export_config(Config(export=ExportConfig(onnx_static_frames=50)))

    with pytest.raises(ValueError, match=r"model.max_frames \(3000\); got None"):
        validate_tflite_export_config(Config(export=ExportConfig(onnx_static_frames=None)))


def test_tflite_parity_thresholds_fail_on_excessive_drift():
    report = {
        "comparisons": {
            "checkpoint_runtime_static_vs_tflite_runtime_static": {
                "max_abs_diff": 0.02,
                "mean_abs_diff": 0.001,
            }
        }
    }
    cfg = Config(export=ExportConfig(tflite_parity_max_abs_diff=0.01, tflite_parity_mean_abs_diff=0.002))

    with pytest.raises(ValueError, match="max_abs_diff"):
        enforce_tflite_parity_thresholds(report, cfg)

    enforce_tflite_parity_thresholds(
        report,
        Config(export=ExportConfig(tflite_parity_max_abs_diff=0.03, tflite_parity_mean_abs_diff=0.002)),
    )


def test_required_tflite_parity_fails_when_eval_manifest_missing(tmp_path):
    cfg = Config(
        training=TrainingConfig(eval_manifest=str(tmp_path / "missing.csv")),
        export=ExportConfig(require_tflite_parity=True),
    )

    with pytest.raises(FileNotFoundError, match="TFLite parity is required"):
        maybe_write_tflite_parity_report(
            checkpoint_path=tmp_path / "best.pt",
            cfg=cfg,
            tflite_path=tmp_path / "inherent.tflite",
            output_dir=tmp_path,
            backend_name="tflite",
        )


def test_required_tflite_parity_needs_thresholds():
    cfg = Config(
        export=ExportConfig(
            require_tflite_parity=False,
            tflite_parity_max_abs_diff=None,
            tflite_parity_mean_abs_diff=None,
        )
    )
    cfg.export.require_tflite_parity = True

    with pytest.raises(ValueError, match="thresholds are unset"):
        require_tflite_parity_thresholds(cfg)


def test_export_rejects_checkpoint_with_different_padding_mode(tmp_path):
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": {},
            "head_order": list(HEAD_ORDER),
            "config": {
                "model": asdict(Config().model),
                "training": asdict(TrainingConfig(padding="dynamic")),
            },
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="training.padding"):
        load_export_model(checkpoint, Config(training=TrainingConfig(padding="runtime_static")))


class _LabelOnlyDataset:
    def __init__(self, labels):
        self._labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return self._labels.shape[0]

    def label_matrix(self):
        return self._labels


def test_representative_samples_are_label_stratified():
    labels = np.zeros((32, len(HEAD_ORDER)), dtype=np.float32)
    labels[:10, 0] = 1.0
    for head_index in range(len(HEAD_ORDER)):
        labels[10 + head_index, head_index] = 1.0

    indexes = representative_sample_indexes(_LabelOnlyDataset(labels), max_count=16)

    selected = labels[indexes]
    assert len(indexes) == 16
    assert selected.sum(axis=0).min() >= 1.0
    assert np.any(selected.sum(axis=1) == 0.0)
    assert indexes != list(range(16))


def test_representative_samples_bound_manifest_fill_work():
    labels = np.zeros((10_000, len(HEAD_ORDER)), dtype=np.float32)
    labels[::100, 0] = 1.0

    indexes = representative_sample_indexes(_LabelOnlyDataset(labels), max_count=8)

    assert len(indexes) == 8
    assert max(indexes) > 5_000
    assert indexes != list(range(8))


@pytest.mark.parametrize(
    ("quantization", "expected"),
    [
        ("int8", {"optimizations": ["DEFAULT"], "supported_ops": ["INT8"], "supported_types": []}),
        ("float16", {"optimizations": ["DEFAULT"], "supported_ops": [], "supported_types": ["float16"]}),
        ("float32", {"optimizations": [], "supported_ops": [], "supported_types": []}),
    ],
)
def test_saved_model_conversion_supports_quality_fallback_quantization_modes(
    tmp_path,
    monkeypatch,
    quantization,
    expected,
):
    converter = _FakeConverter()
    fake_tf = SimpleNamespace(
        float16="float16",
        lite=SimpleNamespace(
            Optimize=SimpleNamespace(DEFAULT="DEFAULT"),
            OpsSet=SimpleNamespace(TFLITE_BUILTINS_INT8="INT8"),
            TFLiteConverter=SimpleNamespace(from_saved_model=lambda _path: converter),
        ),
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    output = tmp_path / "inherent.tflite"
    convert_saved_model_to_tflite(
        tmp_path / "saved_model",
        Config(export=ExportConfig(quantization=quantization)),
        output,
    )

    assert output.read_bytes() == b"TFLITE"
    assert converter.optimizations == expected["optimizations"]
    assert converter.target_spec.supported_ops == expected["supported_ops"]
    assert converter.target_spec.supported_types == expected["supported_types"]


class _FakeConverter:
    def __init__(self):
        self.optimizations = []
        self.representative_dataset = None
        self.target_spec = SimpleNamespace(supported_ops=[], supported_types=[])

    def convert(self):
        return b"TFLITE"
