import csv
import json
import wave

from inherent import HEAD_ORDER
from inherent.data.fixture_manifest import write_gatekeeper_fixture_manifest


def test_gatekeeper_fixture_manifest_maps_audio_head_indices(tmp_path):
    audio_dir = tmp_path / "audio"
    _write_wav(audio_dir / "timer.wav")
    _write_wav(audio_dir / "memory.wav")
    _write_wav(audio_dir / "negative.wav")
    index = tmp_path / "gatekeeper-utterances.json"
    index.write_text(
        json.dumps(
            {
                "audio_fixtures": [
                    {
                        "file": "audio/timer.wav",
                        "phrase": "Set a timer for ten minutes.",
                        "expected_audio_gatekeeper_top_index": 12,
                    },
                    {
                        "file": "audio/memory.wav",
                        "phrase": "Key insight from the meeting was churn dropped.",
                        "expected_audio_gatekeeper_top_index": 0,
                    },
                    {
                        "file": "audio/negative.wav",
                        "phrase": "Play music.",
                        "expected_audio_gatekeeper_top_index": None,
                    },
                ]
            }
        )
    )
    output = tmp_path / "fixtures.csv"

    result = write_gatekeeper_fixture_manifest(index_path=index, output_manifest=output)

    assert result.rows_written == 3
    rows = list(csv.DictReader(output.open()))
    timer, memory, negative = rows
    assert timer["isInteresting"] == "1"
    assert timer["hasStartTimerIntent"] == "1"
    assert all(timer[head] == "0" for head in HEAD_ORDER[1:-1])
    assert memory["isInteresting"] == "1"
    assert all(memory[head] == "0" for head in HEAD_ORDER[1:])
    assert all(negative[head] == "0" for head in HEAD_ORDER)
    assert timer["source"] == "gatekeeper_fixture:existing"
    assert timer["duration_s"] == "0.010"


def _write_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 160)
