import numpy as np

from inherent import HEAD_ORDER
from inherent.data import intents


def test_slurp_loader_maps_six_public_heads(tmp_path, monkeypatch):
    labels = [
        "lists_createoradd",
        "qa_factoid",
        "calendar_set",
        "alarm_set",
        "recommendation",
        "calendar_query",
        "play_music",
    ]

    def fake_iter_hf_rows(dataset_id, root, configs):
        for index, label in enumerate(labels):
            yield intents._HfRow(
                row={
                    "audio": {
                        "array": np.zeros(1600, dtype=np.float32),
                        "sampling_rate": 16000,
                    },
                    "intent": label,
                    "sentence": f"utterance {index}",
                },
                dataset_id=dataset_id,
                config_name="default",
                split="train",
                row_index=index,
                features={},
            )

    def fake_write_wav(path, array, sampling_rate):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"wav")

    monkeypatch.setattr(intents, "_iter_hf_rows", fake_iter_hf_rows)
    monkeypatch.setattr(intents, "_write_wav_16k", fake_write_wav)

    samples = intents.load_slurp(tmp_path / "slurp")

    assert len(samples) == 6
    assert samples[0].head_labels["hasAddToListIntent"] is True
    assert samples[1].head_labels["hasTermSearchQuery"] is True
    assert samples[2].head_labels["hasCalendarEvent"] is True
    assert samples[3].head_labels["hasStartTimerIntent"] is True
    assert samples[4].head_labels["hasPersonContext"] is True
    assert samples[5].head_labels["hasEventContext"] is True
    assert all(sample.audio_path.is_file() for sample in samples)


def test_stop_loader_reads_manifest_triples(tmp_path):
    root = tmp_path / "stop"
    audio_root = root / "audio"
    manifest = root / "train.tsv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "\n".join(
            [
                "audio",
                "timer.wav\t1600",
                "event.wav\t1600",
                "music.wav\t1600",
            ]
        )
        + "\n"
    )
    manifest.with_suffix(".ltr").write_text("start a timer\ncreate an event\nplay music\n")
    manifest.with_suffix(".parse").write_text(
        "[IN:CREATE_TIMER ]\n[IN:CREATE_EVENT ]\n[IN:PLAY_MUSIC ]\n"
    )
    audio_root.mkdir()
    (audio_root / "timer.wav").write_bytes(b"wav")
    (audio_root / "event.wav").write_bytes(b"wav")
    (audio_root / "music.wav").write_bytes(b"wav")

    samples = intents.load_stop(root)

    assert len(samples) == 2
    assert samples[0].audio_path == audio_root / "timer.wav"
    assert samples[0].head_labels["hasStartTimerIntent"] is True
    assert samples[1].audio_path == audio_root / "event.wav"
    assert samples[1].head_labels["hasCalendarEvent"] is True


def test_recorded_intent_loader_skips_rows_without_positive_intent(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"wav")
    manifest = tmp_path / "recorded.csv"
    with manifest.open("w", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=["audio_path", *HEAD_ORDER])
        writer.writeheader()
        negative = {"audio_path": audio.name, **{head: "0" for head in HEAD_ORDER}}
        positive = {"audio_path": audio.name, **{head: "0" for head in HEAD_ORDER}}
        positive["hasPhotoQuery"] = "1"
        writer.writerow(negative)
        writer.writerow(positive)

    samples = intents.load_recorded_manifest(manifest)

    assert len(samples) == 1
    assert samples[0].head_labels["hasPhotoQuery"] is True
