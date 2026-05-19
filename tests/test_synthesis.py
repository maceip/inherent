import csv
from types import SimpleNamespace

import pytest

from inherent.data import intents, synthesis
from inherent.data.tts_engines import OPENF5_TTS_ENGINE
from inherent.scripts import prep_data


def test_expand_prompts_returns_unique_prompts():
    prompts = synthesis.expand_prompts("hasPhotoQuery", 8)

    assert len(prompts) == 8
    assert len(set(prompts)) == 8
    assert all(prompt == prompt.lower() for prompt in prompts)


def test_default_synthetic_heads_can_expand_requested_count():
    for head in synthesis.SYNTHETIC_HEADS:
        prompts = synthesis.expand_prompts(head, 8000)

        assert len(prompts) == 8000
        assert len(set(prompts)) == 8000


def test_expand_prompts_rejects_non_synthetic_head():
    with pytest.raises(ValueError, match="not a TTS-only"):
        synthesis.expand_prompts("hasStartTimerIntent", 1)


def test_prep_data_balances_synthetic_tasks_by_head_and_voice():
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            intents={
                "synthetic": {
                    "photo_query": {"count": 2, "voices": ["voice_a", "voice_b"]},
                    "create_doc": {"count": 2, "voices": ["voice_a"]},
                }
            }
        )
    )

    tasks = list(prep_data._iter_balanced_synthetic_tasks(cfg))

    assert [(head, voice) for head, _, voice in tasks] == [
        ("hasPhotoQuery", "voice_a"),
        ("hasPhotoQuery", "voice_b"),
        ("hasCreateDocIntent", "voice_a"),
        ("hasPhotoQuery", "voice_a"),
        ("hasPhotoQuery", "voice_b"),
        ("hasCreateDocIntent", "voice_a"),
    ]


def test_synthesize_uses_openf5_cli_and_writes_manifest(tmp_path, monkeypatch):
    voice_root = tmp_path / "voices"
    voice_dir = voice_root / "voice_a"
    voice_dir.mkdir(parents=True)
    (voice_dir / "ref.wav").write_bytes(b"reference")
    (voice_dir / "ref.txt").write_text("this is the reference voice")
    monkeypatch.setenv("INHERENT_TTS_VOICE_DIR", str(voice_root))
    monkeypatch.setenv("INHERENT_OPENF5_MODEL", "mrfakename/OpenF5-TTS-Base")
    monkeypatch.setattr(synthesis.shutil, "which", lambda command: "/usr/bin/f5-tts_infer-cli")
    model_files = synthesis.OpenF5ModelFiles(
        model_cfg=tmp_path / "config.yaml",
        ckpt_file=tmp_path / "model.pt",
        vocab_file=tmp_path / "vocab.txt",
    )
    for path in model_files:
        path.write_text("model")
    monkeypatch.setattr(synthesis, "_openf5_model_files", lambda: model_files)
    monkeypatch.setattr(
        synthesis,
        "_normalize_wav",
        lambda input_path, output_path: output_path.write_bytes(input_path.read_bytes()),
    )
    monkeypatch.setattr(synthesis, "augment_wav_file", lambda path, **kwargs: path)

    class FakeRuntime:
        def __init__(self, runtime_model_files):
            assert runtime_model_files == model_files

        def synthesize_to_wav(self, *, prompt, ref_audio, ref_text, output_path):
            assert prompt == "show me photos of the receipt"
            assert ref_audio == voice_dir / "ref.wav"
            assert ref_text == "this is the reference voice"
            output_path.write_bytes(b"tts")

    monkeypatch.setattr(synthesis, "_OpenF5Runtime", FakeRuntime)

    samples = synthesis.synthesize(
        ["show me photos of the receipt"],
        "hasPhotoQuery",
        tmp_path / "audio",
        voices=("voice_a",),
        tts_engine=OPENF5_TTS_ENGINE,
    )
    manifest = tmp_path / "synthetic.csv"
    count = synthesis.write_synthetic_manifest(samples, manifest)
    loaded = intents.load_synthetic_manifest(manifest)

    assert count == 1
    assert samples[0].audio_path.is_file()
    assert samples[0].head == "hasPhotoQuery"
    assert loaded[0].head_labels["hasPhotoQuery"] is True
    rows = list(csv.DictReader(manifest.open()))
    assert rows[0]["tts_engine"] == "openf5-tts"


def test_openf5_model_guard_rejects_disallowed_models(monkeypatch):
    monkeypatch.setenv("INHERENT_OPENF5_MODEL", "F5TTS_v1_Base")

    with pytest.raises(ValueError, match="disallowed"):
        synthesis._openf5_model_reference()
