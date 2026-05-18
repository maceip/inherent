import csv

import pytest

from inherent import HEAD_ORDER
from inherent import INTENT_HEAD_ORDER
from inherent.config import Config
from inherent.config import ExportConfig
from inherent.data import combine_indexes, write_raw_audio_manifest
from inherent.data.directedness import DirectednessSample
from inherent.data.intents import _heads_for_public_intent
from inherent.data.intents import IntentSample
from inherent.scripts.prep_data import _synthetic_head_from_key


def test_base_config_names_only_implemented_default_sources():
    cfg = Config.load("configs/base.yaml")

    assert cfg.data.directedness.positives == ["slurp", "speech_massive"]
    assert cfg.data.directedness.negatives == ["ami"]
    assert cfg.data.directedness.max_public_samples_per_source == 20000
    assert cfg.data.directedness.labeled_manifests == []
    assert cfg.data.intents["public_sources"] == ["slurp", "speech_massive"]
    assert cfg.data.intents["recorded"] == []
    assert cfg.data.intents["synthetic_manifests"] == ["synthetic_manifest.csv"]
    assert "person_context" not in cfg.data.intents["public"]
    assert "event_context" not in cfg.data.intents["public"]
    assert "calling_agent" not in cfg.data.intents["public"]
    assert "person_context" in cfg.data.intents["synthetic"]
    assert "event_context" in cfg.data.intents["synthetic"]


def test_pipeline_configs_load():
    for path in (
        "configs/base.yaml",
        "configs/baseline.yaml",
        "configs/production.yaml",
        "configs/production_quality.yaml",
        "configs/fixture_quality.yaml",
        "configs/smoke.yaml",
        "configs/high_performance_local.yaml",
    ):
        assert Config.load(path).model.num_heads == len(HEAD_ORDER)


def test_export_config_rejects_unknown_quantization():
    default_export = ExportConfig()
    assert default_export.quantization == "float16"
    assert default_export.require_tflite_parity is True
    assert default_export.tflite_parity_max_abs_diff is not None
    assert default_export.tflite_parity_mean_abs_diff is not None

    for quantization in ("int8", "float16", "float32"):
        assert ExportConfig(quantization=quantization).quantization == quantization

    with pytest.raises(ValueError, match="export.quantization"):
        ExportConfig(quantization="int4")


def test_export_configs_pin_android_tflite_shape():
    for path in (
        "configs/base.yaml",
        "configs/baseline.yaml",
        "configs/production.yaml",
        "configs/production_quality.yaml",
        "configs/production_local.yaml",
        "configs/production_local_shard_a.yaml",
        "configs/production_local_shard_b.yaml",
        "configs/fixture_quality.yaml",
        "configs/smoke.yaml",
        "configs/high_performance_local.yaml",
    ):
        cfg = Config.load(path)

        assert cfg.export.onnx_static_frames == cfg.model.max_frames == 3000


def test_release_configs_use_float16_with_required_tflite_parity():
    for path in (
        "configs/base.yaml",
        "configs/baseline.yaml",
        "configs/production.yaml",
        "configs/production_quality.yaml",
        "configs/production_local.yaml",
        "configs/production_local_shard_a.yaml",
        "configs/production_local_shard_b.yaml",
        "configs/high_performance_local.yaml",
    ):
        cfg = Config.load(path)

        assert cfg.export.quantization == "float16"
        assert cfg.export.require_tflite_parity is True
        assert cfg.export.tflite_parity_max_abs_diff is not None
        assert cfg.export.tflite_parity_mean_abs_diff is not None


def test_smoke_configs_can_opt_out_of_release_parity_for_fast_debug_exports():
    for path in ("configs/fixture_quality.yaml", "configs/smoke.yaml"):
        cfg = Config.load(path)

        assert cfg.export.quantization == "int8"
        assert cfg.export.require_tflite_parity is False


def test_default_config_sources_cover_all_intent_heads():
    cfg = Config.load("configs/base.yaml")
    public_labels = [
        "lists_createoradd",
        "qa_factoid",
        "calendar_set",
        "alarm_set",
    ]
    covered = {
        head
        for label in public_labels
        for head, enabled in _heads_for_public_intent(label).items()
        if enabled
    }
    covered.update(
        _synthetic_head_from_key(head_key)
        for head_key in cfg.data.intents["synthetic"]
    )

    assert covered == set(INTENT_HEAD_ORDER)


def test_raw_manifest_can_cover_all_heads_from_default_source_mix(tmp_path):
    positive_audio = tmp_path / "positive.wav"
    negative_audio = tmp_path / "negative.wav"
    positive_audio.write_bytes(b"positive")
    negative_audio.write_bytes(b"negative")
    directedness_samples = [
        DirectednessSample(audio_path=positive_audio, label=1, source="directed", duration_s=1.0),
        DirectednessSample(audio_path=negative_audio, label=0, source="ambient", duration_s=1.0),
    ]
    intent_samples = [
        IntentSample(
            audio_path=positive_audio,
            transcript=head,
            head_labels={intent_head: intent_head == head for intent_head in INTENT_HEAD_ORDER},
            source="configured",
            duration_s=1.0,
        )
        for head in INTENT_HEAD_ORDER
    ]
    output = tmp_path / "raw.csv"

    count = write_raw_audio_manifest(combine_indexes(directedness_samples, intent_samples), output)

    assert count == 2
    rows = list(csv.DictReader(output.open()))
    positives = {
        head
        for head in HEAD_ORDER
        if any(row[head] == "1" for row in rows)
    }
    negatives = {
        head
        for head in HEAD_ORDER
        if any(row[head] == "0" for row in rows)
    }
    assert positives == set(HEAD_ORDER)
    assert negatives == set(HEAD_ORDER)


def test_raw_manifest_rejects_conflicting_duplicate_directedness_labels(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"wav")
    directedness_samples = [
        DirectednessSample(audio_path=audio, label=1, source="positive", duration_s=1.0),
        DirectednessSample(audio_path=audio, label=0, source="negative", duration_s=1.0),
    ]

    with pytest.raises(ValueError, match="conflicting isInteresting"):
        combine_indexes(directedness_samples, [])
