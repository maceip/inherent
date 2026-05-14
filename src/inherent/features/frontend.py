"""Run the runtime audio_frontend.tflite from Python.

This is the training-time counterpart of genesis's on-device frontend path. It
does not approximate the mel transform with librosa; it executes the same TFLite
frontend so training/eval features match runtime features.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .. import HEAD_ORDER
from ..config import RUNTIME_MAX_FRAMES, RUNTIME_MEL_BINS
from ..data.schema import OPTIONAL_LABEL_COLUMNS

SAMPLE_RATE = 16_000
HOP_SAMPLES = 320
MAX_SECONDS = 60
MAX_SAMPLES = SAMPLE_RATE * MAX_SECONDS
INPUT_TENSOR_NAME = "audio_raw"
OUTPUT_TENSOR_NAME = "StatefulPartitionedCall"


class AudioFrontend:
    """Strict wrapper around the runtime `audio_frontend.tflite`."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"audio frontend model not found: {self.model_path}")

        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("tensorflow is required to run audio_frontend.tflite") from exc

        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_detail = _select_input_tensor(self.interpreter.get_input_details())
        self.output_detail = _select_output_tensor(self.interpreter.get_output_details())

    def wav_to_mel(self, wav_path: str | Path) -> np.ndarray:
        audio = load_wav_16k_mono(wav_path)
        return self.audio_to_mel(audio)

    def audio_to_mel(self, audio: np.ndarray) -> np.ndarray:
        audio = _validate_audio(audio)
        frame_count = frame_count_for_samples(audio.shape[0])
        padded = np.zeros((1, MAX_SAMPLES), dtype=np.float32)
        padded[0, : audio.shape[0]] = audio
        self.interpreter.set_tensor(self.input_detail["index"], padded)
        self.interpreter.invoke()
        mel = self.interpreter.get_tensor(self.output_detail["index"])
        mel = _normalize_frontend_output(mel)
        return mel[:frame_count].copy()

    def write_mel(self, wav_path: str | Path, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        mel = self.wav_to_mel(wav_path)
        if output.suffix == ".npy":
            np.save(output, mel)
        elif output.suffix == ".npz":
            np.savez_compressed(output, mel_spectrogram=mel)
        else:
            raise ValueError(f"mel output path must end in .npy or .npz: {output}")
        return output


def load_wav_16k_mono(path: str | Path) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required to load raw audio") from exc

    wav_path = Path(path).expanduser()
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{wav_path} sample rate must be {SAMPLE_RATE}, got {sample_rate}")
    if audio.ndim != 1:
        raise ValueError(f"{wav_path} must be mono, got shape {audio.shape}")
    return _validate_audio(audio)


def frame_count_for_samples(num_samples: int) -> int:
    if num_samples < 1 or num_samples > MAX_SAMPLES:
        raise ValueError(f"sample count must be in [1, {MAX_SAMPLES}], got {num_samples}")
    return min(RUNTIME_MAX_FRAMES, (num_samples + HOP_SAMPLES - 1) // HOP_SAMPLES)


def materialize_mel_manifest(
    *,
    input_manifest: str | Path,
    output_manifest: str | Path,
    mel_dir: str | Path,
    frontend_model: str | Path,
    extension: str = ".npy",
) -> int:
    """Convert a raw-audio 13-label manifest into a mel manifest.

    Input CSV must contain `audio_path` and every head in HEAD_ORDER. Output CSV
    contains `mel_path` and the same labels, suitable for MelManifestDataset.
    """
    if extension not in {".npy", ".npz"}:
        raise ValueError("extension must be '.npy' or '.npz'")
    input_path = Path(input_manifest).expanduser()
    output_path = Path(output_manifest).expanduser()
    mel_root = Path(mel_dir).expanduser()
    frontend = AudioFrontend(frontend_model)

    rows = _read_audio_label_manifest(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mel_root.mkdir(parents=True, exist_ok=True)
    metadata_columns = _metadata_columns(input_path)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mel_path", *metadata_columns, *HEAD_ORDER])
        writer.writeheader()
        for index, row in enumerate(rows):
            mel_path = mel_root / f"{index:08d}{extension}"
            frontend.write_mel(row["audio_path"], mel_path)
            writer.writerow(
                {
                    "mel_path": str(mel_path.resolve()),
                    **{column: row.get(column, "") for column in metadata_columns},
                    **{head: row[head] for head in HEAD_ORDER},
                }
            )
    return len(rows)


def _read_audio_label_manifest(path: Path) -> list[dict[str, str]]:
    allowed = {"audio_path", *OPTIONAL_LABEL_COLUMNS, *HEAD_ORDER}
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {path}")
        missing = [column for column in ("audio_path", *HEAD_ORDER) if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"manifest {path} missing required columns: {missing}")
        unexpected = [column for column in reader.fieldnames if column not in allowed]
        if unexpected:
            raise ValueError(f"manifest {path} has unexpected columns: {unexpected}")
        for row_number, row in enumerate(reader, start=2):
            audio_path = _resolve_path(row["audio_path"], path.parent)
            labels = {
                head: _parse_label(row[head], path, row_number, head)
                for head in HEAD_ORDER
            }
            metadata = {column: row.get(column, "") for column in OPTIONAL_LABEL_COLUMNS}
            rows.append({"audio_path": str(audio_path), **metadata, **labels})
    if not rows:
        raise ValueError(f"manifest contains no rows: {path}")
    return rows


def _metadata_columns(path: Path) -> list[str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {path}")
        return [column for column in reader.fieldnames if column in OPTIONAL_LABEL_COLUMNS]


def _select_input_tensor(details: list[dict]) -> dict:
    candidates = [detail for detail in details if INPUT_TENSOR_NAME in detail["name"]]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one frontend input containing {INPUT_TENSOR_NAME!r}")
    detail = candidates[0]
    _require_dtype(detail, np.float32, "frontend input")
    _require_shape(detail, (1, MAX_SAMPLES), "frontend input")
    return detail


def _select_output_tensor(details: list[dict]) -> dict:
    candidates = [
        detail for detail in details
        if OUTPUT_TENSOR_NAME in detail["name"] or "mel" in detail["name"].lower()
    ]
    if len(candidates) != 1:
        raise ValueError("expected exactly one frontend mel output tensor")
    detail = candidates[0]
    _require_dtype(detail, np.float32, "frontend output")
    shape = tuple(int(dim) for dim in detail["shape"])
    if shape not in {(1, RUNTIME_MAX_FRAMES, RUNTIME_MEL_BINS), (RUNTIME_MAX_FRAMES, RUNTIME_MEL_BINS)}:
        raise ValueError(
            f"frontend output must be [1,{RUNTIME_MAX_FRAMES},{RUNTIME_MEL_BINS}] "
            f"or [{RUNTIME_MAX_FRAMES},{RUNTIME_MEL_BINS}], got {shape}"
        )
    return detail


def _normalize_frontend_output(mel: np.ndarray) -> np.ndarray:
    if mel.ndim == 3 and mel.shape[0] == 1:
        mel = mel[0]
    if mel.shape != (RUNTIME_MAX_FRAMES, RUNTIME_MEL_BINS):
        raise ValueError(f"frontend returned unexpected mel shape {mel.shape}")
    if not np.isfinite(mel).all():
        raise ValueError("frontend returned non-finite mel values")
    return np.asarray(mel, dtype=np.float32)


def _validate_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1-D mono, got shape {audio.shape}")
    if audio.shape[0] < 1 or audio.shape[0] > MAX_SAMPLES:
        raise ValueError(f"audio sample count must be in [1, {MAX_SAMPLES}], got {audio.shape[0]}")
    if not np.isfinite(audio).all():
        raise ValueError("audio contains non-finite samples")
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        raise ValueError(f"audio must be normalized to [-1, 1], peak={peak}")
    return audio


def _require_dtype(detail: dict, dtype: np.dtype, label: str) -> None:
    if detail["dtype"] != dtype:
        raise TypeError(f"{label} dtype must be {dtype}, got {detail['dtype']}")


def _require_shape(detail: dict, expected: tuple[int, ...], label: str) -> None:
    shape = tuple(int(dim) for dim in detail["shape"])
    if shape != expected:
        raise ValueError(f"{label} shape must be {expected}, got {shape}")


def _resolve_path(value: str, base_dir: Path) -> Path:
    if value.strip() == "":
        raise ValueError("audio_path must be non-empty")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _parse_label(value: str, path: Path, row_number: int, column: str) -> str:
    normalized = value.strip()
    if normalized in {"0", "1"}:
        return normalized
    raise ValueError(f"invalid label {value!r} for {column} at {path}:{row_number}; expected 0 or 1")
