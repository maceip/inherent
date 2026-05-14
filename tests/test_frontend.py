import csv

import numpy as np
import pytest

from inherent import HEAD_ORDER
from inherent.features import frontend


def write_audio_manifest(path, rows, extra_fieldnames=()):
    fieldnames = ["audio_path", *HEAD_ORDER, *extra_fieldnames]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def label_row(audio_path, positive_index=0):
    row = {"audio_path": audio_path}
    for i, head in enumerate(HEAD_ORDER):
        row[head] = "1" if i == positive_index else "0"
    return row


class FakeFrontend:
    def __init__(self, model_path):
        self.model_path = model_path

    def write_mel(self, wav_path, output_path):
        np.save(output_path, np.ones((3, 128), dtype=np.float32))


def test_frame_count_matches_cosmo_hop():
    assert frontend.frame_count_for_samples(1) == 1
    assert frontend.frame_count_for_samples(320) == 1
    assert frontend.frame_count_for_samples(321) == 2
    assert frontend.frame_count_for_samples(960_000) == 3000


def test_materialize_mel_manifest_writes_training_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(frontend, "AudioFrontend", FakeFrontend)
    input_manifest = tmp_path / "audio.csv"
    output_manifest = tmp_path / "train.csv"
    mel_dir = tmp_path / "mels"
    write_audio_manifest(input_manifest, [label_row("clip.wav", 2)])

    count = frontend.materialize_mel_manifest(
        input_manifest=input_manifest,
        output_manifest=output_manifest,
        mel_dir=mel_dir,
        frontend_model=tmp_path / "audio_frontend.tflite",
    )

    assert count == 1
    rows = list(csv.DictReader(output_manifest.open()))
    assert len(rows) == 1
    assert rows[0]["mel_path"].endswith("00000000.npy")
    assert rows[0]["hasTermSearchQuery"] == "1"
    assert (mel_dir / "00000000.npy").is_file()


def test_materialize_mel_manifest_rejects_unexpected_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(frontend, "AudioFrontend", FakeFrontend)
    input_manifest = tmp_path / "audio.csv"
    row = label_row("clip.wav", 0)
    row["hasTypoIntent"] = "1"
    write_audio_manifest(input_manifest, [row], extra_fieldnames=("hasTypoIntent",))

    with pytest.raises(ValueError, match="unexpected columns"):
        frontend.materialize_mel_manifest(
            input_manifest=input_manifest,
            output_manifest=tmp_path / "train.csv",
            mel_dir=tmp_path / "mels",
            frontend_model=tmp_path / "audio_frontend.tflite",
        )
