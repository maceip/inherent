"""Strict mel-spectrogram manifest dataset for training."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .. import HEAD_ORDER, NUM_HEADS
from ..data.schema import OPTIONAL_LABEL_COLUMNS


@dataclass(frozen=True)
class MelManifestRow:
    mel_path: Path
    labels: tuple[float, ...]


@dataclass(frozen=True)
class MelBatch:
    mel: torch.Tensor
    targets: torch.Tensor
    lengths: torch.Tensor


class MelManifestDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Dataset backed by precomputed frontend mel tensors.

    CSV contract:
      - required: `mel_path`
      - required: every head in `inherent.HEAD_ORDER`, as 0/1 labels

    Supported tensor file formats:
      - `.npy` arrays with shape `[T, 128]`
      - `.npz` files with key `mel_spectrogram`, shape `[T, 128]`
    """

    def __init__(self, manifest_path: str | Path, *, mel_bins: int, max_frames: int):
        self.manifest_path = Path(manifest_path).expanduser()
        self.mel_bins = mel_bins
        self.max_frames = max_frames
        self.rows = _read_manifest(self.manifest_path)
        if not self.rows:
            raise ValueError(f"manifest contains no rows: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows[index]
        mel = _load_mel(row.mel_path, self.mel_bins, self.max_frames)
        targets = torch.tensor(row.labels, dtype=torch.float32)
        return mel, targets, int(mel.shape[0])

    def label_matrix(self) -> torch.Tensor:
        return torch.tensor([row.labels for row in self.rows], dtype=torch.float32)


def collate_mel_batches(samples: Iterable[tuple[torch.Tensor, torch.Tensor, int]]) -> MelBatch:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("cannot collate an empty batch")
    lengths = torch.tensor([sample[2] for sample in sample_list], dtype=torch.long)
    max_len = int(lengths.max().item())
    mel_bins = sample_list[0][0].shape[1]
    mel = torch.zeros(len(sample_list), max_len, mel_bins, dtype=torch.float32)
    targets = torch.zeros(len(sample_list), NUM_HEADS, dtype=torch.float32)
    for i, (sample_mel, sample_targets, sample_len) in enumerate(sample_list):
        if sample_mel.shape[1] != mel_bins:
            raise ValueError(f"batch mel_bins mismatch: {sample_mel.shape[1]} != {mel_bins}")
        mel[i, :sample_len] = sample_mel
        targets[i] = sample_targets
    return MelBatch(mel=mel, targets=targets, lengths=lengths)


def compute_balanced_pos_weight(dataset: MelManifestDataset) -> torch.Tensor:
    labels = dataset.label_matrix()
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    missing_positive = [HEAD_ORDER[i] for i, value in enumerate(positives.tolist()) if value == 0]
    missing_negative = [HEAD_ORDER[i] for i, value in enumerate(negatives.tolist()) if value == 0]
    if missing_positive or missing_negative:
        raise ValueError(
            "balanced class weights require at least one positive and one negative per head; "
            f"missing_positive={missing_positive}, missing_negative={missing_negative}"
        )
    return negatives / positives


def _read_manifest(path: Path) -> list[MelManifestRow]:
    rows: list[MelManifestRow] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {path}")
        allowed_columns = {"mel_path", *OPTIONAL_LABEL_COLUMNS, *HEAD_ORDER}
        missing = [column for column in ("mel_path", *HEAD_ORDER) if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"manifest {path} missing required columns: {missing}")
        unexpected = [column for column in reader.fieldnames if column not in allowed_columns]
        if unexpected:
            raise ValueError(f"manifest {path} has unexpected columns: {unexpected}")
        for row_number, row in enumerate(reader, start=2):
            mel_path = _resolve_path(row["mel_path"], path.parent)
            labels = tuple(_parse_label(row[head], path, row_number, head) for head in HEAD_ORDER)
            rows.append(MelManifestRow(mel_path=mel_path, labels=labels))
    return rows


def _load_mel(path: Path, mel_bins: int, max_frames: int) -> torch.Tensor:
    if path.suffix == ".npy":
        array = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path) as data:
            if "mel_spectrogram" not in data:
                raise ValueError(f"{path} missing 'mel_spectrogram' array")
            array = data["mel_spectrogram"]
    else:
        raise ValueError(f"unsupported mel tensor file type: {path.suffix}")
    if array.ndim != 2:
        raise ValueError(f"{path} must have shape [T, {mel_bins}], got {array.shape}")
    if array.shape[0] < 1 or array.shape[0] > max_frames:
        raise ValueError(f"{path} has invalid frame count {array.shape[0]}; expected 1..{max_frames}")
    if array.shape[1] != mel_bins:
        raise ValueError(f"{path} has {array.shape[1]} mel bins; expected {mel_bins}")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{path} must contain floating point mel values, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path} contains non-finite mel values")
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def _resolve_path(value: str, base_dir: Path) -> Path:
    if value.strip() == "":
        raise ValueError("mel_path must be non-empty")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _parse_label(value: str, path: Path, row_number: int, column: str) -> float:
    normalized = value.strip()
    if normalized == "1":
        return 1.0
    if normalized == "0":
        return 0.0
    raise ValueError(f"invalid label {value!r} for {column} at {path}:{row_number}; expected 0 or 1")
