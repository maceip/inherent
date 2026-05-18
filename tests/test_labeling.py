import csv

import numpy as np

from inherent import HEAD_ORDER
from inherent.data import directedness, intents
from inherent.data.labeling import (
    LABEL_TEMPLATE_COLUMNS,
    normalize_audio_manifest,
    split_label_coverage_report,
    split_label_manifest,
    validate_split_label_coverage,
    validate_label_manifest,
    write_label_template,
)
from inherent.training.dataset import MelManifestDataset


def write_wav(path, sample_rate=16000, seconds=0.25):
    import soundfile as sf

    audio = np.zeros(int(sample_rate * seconds), dtype=np.float32)
    audio[::100] = 0.2
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def label_row(audio_path, *, speaker_id="speaker-a", session_id="session-a", positive_head=None):
    row = {
        "audio_path": str(audio_path),
        "transcript": "",
        "speaker_id": speaker_id,
        "session_id": session_id,
        "device": "phone",
        "environment": "kitchen",
        "source": "recorded",
        "duration_s": "0.25",
        "split": "",
    }
    for head in HEAD_ORDER:
        row[head] = "0"
    if positive_head is not None:
        row["isInteresting"] = "1"
        row[positive_head] = "1"
    return row


def write_label_manifest(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_template_contains_metadata_and_heads(tmp_path):
    output = write_label_template(tmp_path / "labels.csv")

    header = output.read_text().splitlines()[0].split(",")
    assert header == list(LABEL_TEMPLATE_COLUMNS)


def test_validate_normalize_and_split_label_manifest(tmp_path):
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    wav_c = tmp_path / "c.wav"
    write_wav(wav_a)
    write_wav(wav_b)
    write_wav(wav_c)
    manifest = tmp_path / "labels.csv"
    write_label_manifest(
        manifest,
        [
            label_row(wav_a, speaker_id="a", session_id="1", positive_head="hasAddToListIntent"),
            label_row(wav_b, speaker_id="b", session_id="1", positive_head="hasCallingAgentIntent"),
            label_row(wav_c, speaker_id="c", session_id="1"),
        ],
    )

    report = validate_label_manifest(manifest)
    assert report["ok"] is True
    assert report["stats"]["heads"]["hasAddToListIntent"]["positive"] == 1

    normalized = tmp_path / "normalized.csv"
    count = normalize_audio_manifest(manifest, normalized, tmp_path / "normalized_audio")
    assert count == 3
    normalized_rows = list(csv.DictReader(normalized.open()))
    assert normalized_rows[0]["speaker_id"] == "a"
    assert normalized_rows[0]["audio_path"].endswith(".wav")

    counts = split_label_manifest(normalized, tmp_path / "splits", seed=0)
    assert sum(counts.values()) == 3
    split_rows = []
    for split_name in ("train", "eval", "test"):
        split_rows.extend(csv.DictReader((tmp_path / "splits" / f"{split_name}_manifest.csv").open()))
    groups = {}
    for row in split_rows:
        groups.setdefault((row["speaker_id"], row["session_id"]), set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())


def test_split_label_coverage_requires_each_head_in_each_split(tmp_path):
    split_manifests = {}
    for split in ("train", "eval", "test"):
        manifest = tmp_path / f"{split}.csv"
        rows = []
        for head in HEAD_ORDER[1:]:
            rows.append(
                label_row(
                    tmp_path / f"{split}_{head}.wav",
                    speaker_id=f"{split}-{head}",
                    session_id="1",
                    positive_head=head,
                )
            )
        rows.append(label_row(tmp_path / f"{split}_negative.wav", speaker_id=f"{split}-negative", session_id="1"))
        write_label_manifest(manifest, rows)
        split_manifests[split] = manifest

    report = split_label_coverage_report(split_manifests)

    assert report["ok"] is True
    validate_split_label_coverage(split_manifests)

    rows = list(csv.DictReader(split_manifests["eval"].open()))
    rows = [row for row in rows if row["hasStartTimerIntent"] != "1"]
    write_label_manifest(split_manifests["eval"], rows)

    report = split_label_coverage_report(split_manifests)
    assert report["ok"] is False
    assert any(issue["head"] == "hasStartTimerIntent" for issue in report["issues"])


def test_split_label_manifest_stratifies_label_coverage_when_feasible(tmp_path):
    rows = []
    for replica in range(3):
        for head in HEAD_ORDER[1:]:
            rows.append(
                label_row(
                    tmp_path / f"{replica}_{head}.wav",
                    speaker_id=f"{replica}-{head}",
                    session_id="1",
                    positive_head=head,
                )
            )
        rows.append(
            label_row(
                tmp_path / f"{replica}_negative.wav",
                speaker_id=f"{replica}-negative",
                session_id="1",
            )
        )
    manifest = tmp_path / "labels.csv"
    write_label_manifest(manifest, rows)

    split_label_manifest(manifest, tmp_path / "splits", seed=0)

    validate_split_label_coverage(
        {
            split: tmp_path / "splits" / f"{split}_manifest.csv"
            for split in ("train", "eval", "test")
        }
    )


def test_recorded_loaders_accept_collection_metadata(tmp_path):
    wav = tmp_path / "a.wav"
    write_wav(wav)
    manifest = tmp_path / "labels.csv"
    write_label_manifest(manifest, [label_row(wav.name, positive_head="hasAddToListIntent")])

    directed = directedness.load_recorded_manifest(manifest, default_label=None)
    intent = intents.load_recorded_manifest(manifest)

    assert len(directed) == 1
    assert len(intent) == 1
    assert intent[0].head_labels["hasAddToListIntent"] is True


def test_mel_manifest_dataset_accepts_collection_metadata(tmp_path):
    np.save(tmp_path / "a.npy", np.ones((5, 128), dtype=np.float32))
    manifest = tmp_path / "mel.csv"
    row = label_row("a.npy", positive_head="hasAddToListIntent")
    row["mel_path"] = row.pop("audio_path")
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mel_path", *LABEL_TEMPLATE_COLUMNS[1:]])
        writer.writeheader()
        writer.writerow(row)

    dataset = MelManifestDataset(manifest, mel_bins=128, max_frames=3000)

    assert len(dataset) == 1
