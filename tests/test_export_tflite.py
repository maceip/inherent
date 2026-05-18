import json

import pytest

from inherent import HEAD_ORDER, THRESHOLD_KEYS
from inherent.config import Config, ExportConfig
from inherent.export import core, tflite
from inherent.export.core import validate_tflite_io_contract
from inherent.export.litert import validate_tflite_export_config


def _tflite_io(
    *,
    input_shape=(1, 3000, 128),
    input_dtype="float32",
    output_shape=(1, 13),
    output_dtype="float32",
):
    return {
        "input_name": "serving_default_mel_spectrogram:0",
        "input_shape": list(input_shape),
        "input_dtype": input_dtype,
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
