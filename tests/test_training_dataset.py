import csv

import numpy as np
import pytest

from inherent import HEAD_ORDER, NUM_HEADS
from inherent.training.dataset import (
    MelManifestDataset,
    collate_mel_batches,
    compute_balanced_pos_weight,
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
