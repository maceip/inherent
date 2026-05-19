import csv

from inherent.scripts.finalize_synthetic_slice import trim_synthetic_manifest


def test_trim_synthetic_manifest_caps_per_head(tmp_path):
    manifest = tmp_path / "partial.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["audio_path", "transcript", "head", "voice_id", "tts_engine"],
        )
        writer.writeheader()
        for index in range(3):
            for head in ("hasPersonContext", "hasEventContext"):
                audio = tmp_path / f"{head}_{index}.wav"
                audio.write_bytes(b"wav")
                writer.writerow(
                    {
                        "audio_path": str(audio),
                        "transcript": f"{head} line {index}",
                        "head": head,
                        "voice_id": "openvoice",
                        "tts_engine": "openf5-tts",
                    }
                )

    output = tmp_path / "slice.csv"
    count = trim_synthetic_manifest(
        manifest,
        output,
        max_per_head={"hasPersonContext": 2, "hasEventContext": 1},
    )

    assert count == 3
    rows = list(csv.DictReader(output.open()))
    assert sum(1 for row in rows if row["head"] == "hasPersonContext") == 2
    assert sum(1 for row in rows if row["head"] == "hasEventContext") == 1
