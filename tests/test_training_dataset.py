import csv

import numpy as np
import pytest
import torch

from inherent import HEAD_ORDER, NUM_HEADS
from inherent.training.dataset import (
    MelManifestDataset,
    collate_mel_batches,
    compute_balanced_pos_weight,
    compute_label_balanced_sample_weights,
)


def write_manifest(path, rows, extra_fieldnames=()):
    fieldnames = ["mel_path", *HEAD_ORDER, *extra_fieldnames]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def label_row(mel_path, positive_index):
    row = {"mel_path": mel_path}
    for i, head in enumerate(HEAD_ORDER):
        row[head] = "1" if i == positive_index else "0"
    return row


def test_mel_manifest_dataset_loads_and_collates(tmp_path):
    np.save(tmp_path / "a.npy", np.ones((5, 128), dtype=np.float32))
    np.save(tmp_path / "b.npy", np.ones((7, 128), dtype=np.float32) * 2)
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            label_row("a.npy", 0),
            label_row("b.npy", 12),
        ],
    )

    dataset = MelManifestDataset(manifest, mel_bins=128, max_frames=3000)
    first = dataset[0]
    second = dataset[1]
    batch = collate_mel_batches([first, second])

    assert batch.mel.shape == (2, 7, 128)
    assert batch.targets.shape == (2, NUM_HEADS)
    assert batch.lengths.tolist() == [5, 7]
    assert batch.targets[0, 0].item() == 1.0
    assert batch.targets[1, 12].item() == 1.0


def test_collate_can_pad_to_runtime_fixed_frames(tmp_path):
    np.save(tmp_path / "a.npy", np.ones((5, 128), dtype=np.float32))
    np.save(tmp_path / "b.npy", np.ones((7, 128), dtype=np.float32) * 2)
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            label_row("a.npy", 0),
            label_row("b.npy", 12),
        ],
    )
    dataset = MelManifestDataset(manifest, mel_bins=128, max_frames=3000)

    batch = collate_mel_batches([dataset[0], dataset[1]], fixed_frames=10)

    assert batch.mel.shape == (2, 10, 128)
    assert batch.lengths.tolist() == [5, 7]
    assert torch.all(batch.mel[0, 5:] == 0)
    assert torch.all(batch.mel[1, 7:] == 0)


def test_collate_rejects_samples_longer_than_fixed_frames(tmp_path):
    np.save(tmp_path / "a.npy", np.ones((11, 128), dtype=np.float32))
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [label_row("a.npy", 0)])
    dataset = MelManifestDataset(manifest, mel_bins=128, max_frames=3000)

    with pytest.raises(ValueError, match="fixed_frames=10"):
        collate_mel_batches([dataset[0]], fixed_frames=10)


def test_manifest_rejects_unexpected_columns(tmp_path):
    np.save(tmp_path / "a.npy", np.ones((5, 128), dtype=np.float32))
    manifest = tmp_path / "manifest.csv"
    row = label_row("a.npy", 0)
    row["hasTypoIntent"] = "1"
    write_manifest(manifest, [row], extra_fieldnames=("hasTypoIntent",))

    with pytest.raises(ValueError, match="unexpected columns"):
        MelManifestDataset(manifest, mel_bins=128, max_frames=3000)


def test_dataset_rejects_bad_mel_shape(tmp_path):
    np.save(tmp_path / "bad.npy", np.ones((5, 64), dtype=np.float32))
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [label_row("bad.npy", 0)])

    dataset = MelManifestDataset(manifest, mel_bins=128, max_frames=3000)
    with pytest.raises(ValueError, match="mel bins"):
        dataset[0]


def test_balanced_weights_require_each_head_to_have_both_classes(tmp_path):
    np.save(tmp_path / "a.npy", np.ones((5, 128), dtype=np.float32))
    np.save(tmp_path / "b.npy", np.ones((5, 128), dtype=np.float32))
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            label_row("a.npy", 0),
            label_row("b.npy", 0),
        ],
    )
    dataset = MelManifestDataset(manifest, mel_bins=128, max_frames=3000)

    with pytest.raises(ValueError, match="missing_positive"):
        compute_balanced_pos_weight(dataset)


def test_label_balanced_sample_weights_raise_rare_positive_rows(tmp_path):
    for index in range(6):
        np.save(tmp_path / f"{index}.npy", np.ones((5, 128), dtype=np.float32))
    manifest = tmp_path / "manifest.csv"
    rows = []
    for index in range(6):
        row = {"mel_path": f"{index}.npy"}
        for head_index, head in enumerate(HEAD_ORDER):
            row[head] = "0"
            if head_index == 0:
                row[head] = "1"
        rows.append(row)
    rows[0][HEAD_ORDER[1]] = "1"
    rows[1][HEAD_ORDER[1]] = "1"
    rows[2][HEAD_ORDER[2]] = "1"
    for head in HEAD_ORDER[3:]:
        rows[3][head] = "1"
    write_manifest(manifest, rows)
    dataset = MelManifestDataset(manifest, mel_bins=128, max_frames=3000)

    weights = compute_label_balanced_sample_weights(dataset)

    assert weights.shape == (6,)
    assert torch.isfinite(weights).all()
    assert weights[2] > weights[0]
