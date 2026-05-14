import csv
from types import SimpleNamespace

import numpy as np
import pytest

from inherent import HEAD_ORDER
from inherent.config import DirectednessDataConfig
from inherent.data import directedness


def test_build_index_combines_public_positives_and_recorded_negatives(tmp_path, monkeypatch):
    positive_audio = tmp_path / "positive.wav"
    negative_audio = tmp_path / "negative.wav"
    positive_audio.write_bytes(b"positive")
    negative_audio.write_bytes(b"negative")

    def fake_slurp(root):
        return [
            directedness.DirectednessSample(
                audio_path=positive_audio,
                label=1,
                source="directed:slurp",
                duration_s=1.0,
            )
        ]

    monkeypatch.setattr(directedness, "load_slurp_positives", fake_slurp)

    manifest = tmp_path / "negatives.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "duration_s"])
        writer.writeheader()
        writer.writerow({"audio_path": "negative.wav", "duration_s": "1.0"})

    samples = directedness.build_index(
        DirectednessDataConfig(positives=["slurp"], negative_manifests=[str(manifest)]),
        tmp_path,
    )

    assert [sample.label for sample in samples] == [0, 1]
    assert {sample.source for sample in samples} == {"recorded:negatives.csv", "directed:slurp"}


def test_full_head_order_manifest_labels_directedness(tmp_path):
    positive_audio = tmp_path / "positive.wav"
    negative_audio = tmp_path / "negative.wav"
    positive_audio.write_bytes(b"positive")
    negative_audio.write_bytes(b"negative")
    manifest = tmp_path / "full_labels.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", *HEAD_ORDER])
        writer.writeheader()
        for audio_path, label in [(positive_audio, "1"), (negative_audio, "0")]:
            row = {"audio_path": audio_path.name}
            row.update({head: "0" for head in HEAD_ORDER})
            row["isInteresting"] = label
            writer.writerow(row)

    samples = directedness.build_index(
        DirectednessDataConfig(labeled_manifests=[str(manifest)]),
        tmp_path,
    )

    assert [sample.label for sample in samples] == [1, 0]


def test_ami_negative_loader_indexes_strict_16k_mono_wav_tree(tmp_path, monkeypatch):
    (tmp_path / "session").mkdir()
    wav = tmp_path / "session" / "meeting.wav"
    wav.write_bytes(b"wav")
    monkeypatch.setattr(
        directedness,
        "_read_audio_info",
        lambda path: SimpleNamespace(samplerate=16000, channels=1, frames=32000),
    )

    samples = directedness.load_ami_negatives(tmp_path)

    assert len(samples) == 1
    assert samples[0].label == 0
    assert samples[0].source == "ami"
    assert samples[0].duration_s == 2.0


def test_ami_negative_loader_rejects_non_16k_audio(tmp_path, monkeypatch):
    wav = tmp_path / "bad.wav"
    wav.write_bytes(b"wav")
    monkeypatch.setattr(
        directedness,
        "_read_audio_info",
        lambda path: SimpleNamespace(samplerate=8000, channels=1, frames=32000),
    )

    with pytest.raises(ValueError, match="sample rate"):
        directedness.load_ami_negatives(tmp_path)


def test_librispeech_negative_loader_materializes_capped_public_audio(tmp_path, monkeypatch):
    rows = [
        directedness._HfAudioRow(
            row={
                "audio": {
                    "array": np.zeros(1600, dtype=np.float32),
                    "sampling_rate": 16000,
                },
                "sentence": f"not a command {index}",
            },
            dataset_id="openslr/librispeech_asr",
            config_name="clean",
            split="train",
            row_index=index,
            features={},
        )
        for index in range(3)
    ]

    def fake_iter_hf_audio_rows(*, root, spec, max_samples):
        assert spec.source == "librispeech"
        assert max_samples == 2
        yield from rows[:max_samples]

    def fake_write_wav(path, array, sampling_rate):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"wav")

    monkeypatch.setattr(directedness, "_iter_hf_audio_rows", fake_iter_hf_audio_rows)
    monkeypatch.setattr(directedness, "_write_wav_16k", fake_write_wav)

    samples = directedness.load_librispeech_negatives(tmp_path / "librispeech", max_samples=2)

    assert len(samples) == 2
    assert all(sample.label == 0 for sample in samples)
    assert {sample.source for sample in samples} == {"librispeech:clean:train"}
    assert all(sample.audio_path.is_file() for sample in samples)


def test_public_negative_sources_are_registered():
    assert directedness.PUBLIC_HF_NEGATIVE_SOURCES["librispeech"].license == "CC BY 4.0"
    assert directedness.PUBLIC_HF_NEGATIVE_SOURCES["speech_commands"].license == "CC BY 4.0"
