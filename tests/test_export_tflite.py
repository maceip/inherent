import json

from inherent import HEAD_ORDER, THRESHOLD_KEYS
from inherent.config import Config
from inherent.export import tflite


def test_artifact_path_uses_output_dir_name():
    path = tflite._artifact_path(
        output_dir=tflite.Path("artifacts/v0.1.0"),
        configured="artifacts/inherent.tflite",
        default_name="inherent.tflite",
    )

    assert path == tflite.Path("artifacts/v0.1.0/inherent.tflite")


def test_write_metadata_writes_contract_fields(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    model = tmp_path / "inherent.tflite"
    model.write_bytes(b"tflite")
    metadata = tmp_path / "inherent.metadata.json"

    tflite.write_metadata(checkpoint, Config(), model, metadata)

    data = json.loads(metadata.read_text())
    assert data["input_tensor"] == "mel_spectrogram"
    assert data["output_tensor"] == "intent_output"
    assert tuple(data["head_order"]) == HEAD_ORDER
    assert tuple(data["threshold_keys_in_order"]) == THRESHOLD_KEYS
    assert data["tflite_size_bytes"] == len(b"tflite")
    assert ":" in data["training_hash"]
