import csv
import json
from pathlib import Path

import numpy as np
import pytest

from inherent import HEAD_ORDER, INTENT_HEAD_ORDER
from inherent.config import Config
from inherent.data.schema import LABEL_TEMPLATE_COLUMNS, METADATA_COLUMNS
from inherent.pipeline import recorded


def test_recorded_library_build_orchestrates_production_components(tmp_path, monkeypatch):
    labels = _write_dummy_recorded_library(tmp_path / "library")

    def fake_materialize_mel_manifest(*, input_manifest, output_manifest, mel_dir, frontend_model):
        rows = list(csv.DictReader(Path(input_manifest).open()))
        Path(mel_dir).mkdir(parents=True, exist_ok=True)
        Path(output_manifest).parent.mkdir(parents=True, exist_ok=True)
        with Path(output_manifest).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mel_path", *METADATA_COLUMNS, *HEAD_ORDER])
            writer.writeheader()
            for index, row in enumerate(rows):
                mel_path = Path(mel_dir) / f"{index:08d}.npy"
                np.save(mel_path, np.ones((3, 128), dtype=np.float32))
                writer.writerow(
                    {
                        "mel_path": str(mel_path.resolve()),
                        **{column: "" for column in METADATA_COLUMNS},
                        **{head: row[head] for head in HEAD_ORDER},
                    }
                )
        return len(rows)

    def fake_train(cfg, output_dir, *, init_checkpoint=None):
        assert init_checkpoint is None
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "best.pt").write_bytes(b"checkpoint")

    def fake_evaluate(checkpoint_path, eval_manifest, *, batch_size, device):
        return {
            head: {"auc": 1.0, "eer": 0.0, "fpr_at_recall_95": 0.0}
            for head in HEAD_ORDER
        }

    def fake_export_backends(*, checkpoint, cfg, export_dir, backend_names):
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        artifact = Path(export_dir) / "inherent.tflite"
        artifact.write_bytes(b"tflite")
        return [{"backend": "tflite", "artifacts": {"tflite": str(artifact)}, "supported": True}]

    monkeypatch.setattr(recorded, "materialize_mel_manifest", fake_materialize_mel_manifest)
    monkeypatch.setattr(recorded, "train_model", fake_train)
    monkeypatch.setattr(recorded, "evaluate_checkpoint", fake_evaluate)
    monkeypatch.setattr(recorded, "_export_backends", fake_export_backends)

    result = recorded.build_recorded_library(
        cfg=Config.load("configs/smoke.yaml"),
        labels_manifest=labels,
        work_dir=tmp_path / "work",
        run_dir=tmp_path / "run",
        frontend_model=tmp_path / "audio_frontend.tflite",
        training_device="mps",
        eval_device="cpu",
        max_steps=4,
    )

    assert result.normalized_manifest.is_file()
    assert result.raw_manifests["train"].is_file()
    assert result.mel_manifests["eval"].is_file()
    assert result.checkpoint == tmp_path / "run" / "best.pt"
    assert json.loads(result.metrics_json.read_text())["isInteresting"]["auc"] == 1.0
    assert (result.export_dir / "inherent.tflite").is_file()
    assert result.export_results[0]["backend"] == "tflite"
    resolved = json.loads((tmp_path / "run" / "resolved_config.json").read_text())
    assert resolved["training"]["train_manifest"] == str(result.mel_manifests["train"])


def test_recorded_library_rejects_split_leakage(tmp_path):
    labels = _write_dummy_recorded_library(tmp_path / "library")
    rows = list(csv.DictReader(labels.open()))
    rows[1]["speaker_id"] = rows[0]["speaker_id"]
    rows[1]["session_id"] = rows[0]["session_id"]
    rows[1]["split"] = "eval" if rows[0]["split"] == "train" else "train"
    with labels.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="speaker/session group crosses splits"):
        recorded.build_recorded_library(
            cfg=Config.load("configs/smoke.yaml"),
            labels_manifest=labels,
            work_dir=tmp_path / "work",
            run_dir=tmp_path / "run",
            frontend_model=tmp_path / "audio_frontend.tflite",
            train=False,
            evaluate=False,
            export=False,
        )


def test_model_group_reuses_previous_eval_and_test_identities(tmp_path, monkeypatch):
    labels = _write_dummy_recorded_library(tmp_path / "library")
    monkeypatch.setattr(recorded, "materialize_mel_manifest", _fake_materialize_mel_manifest)

    first_group = tmp_path / "model-groups" / "001"
    first = recorded.build_recorded_library(
        cfg=Config.load("configs/smoke.yaml"),
        labels_manifest=labels,
        model_group_dir=first_group,
        frontend_model=tmp_path / "audio_frontend.tflite",
        train=False,
        evaluate=False,
        export=False,
    )
    first_index = json.loads(first.split_identity_index.read_text())
    first_eval_ids = {row["identity"] for row in first_index["splits"]["eval"]}
    first_test_ids = {row["identity"] for row in first_index["splits"]["test"]}

    rows = list(csv.DictReader(labels.open()))
    for row in rows:
        row["split"] = ""
    new_audio = _write_wav(labels.parent / "audio" / "new_training_hasStartTimerIntent.wav")
    rows.append(
        _label_row(
            new_audio,
            split="",
            positive_head="hasStartTimerIntent",
            speaker_id="speaker-new",
            session_id="session-new",
        )
    )
    expanded = tmp_path / "library" / "labels_expanded.csv"
    with expanded.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    second_group = tmp_path / "model-groups" / "002"
    second = recorded.build_recorded_library(
        cfg=Config.load("configs/smoke.yaml"),
        labels_manifest=expanded,
        model_group_dir=second_group,
        previous_model_group=first_group,
        frontend_model=tmp_path / "audio_frontend.tflite",
        train=False,
        evaluate=False,
        export=False,
    )
    second_index = json.loads(second.split_identity_index.read_text())
    assert {row["identity"] for row in second_index["splits"]["eval"]} == first_eval_ids
    assert {row["identity"] for row in second_index["splits"]["test"]} == first_test_ids
    assert second_index["counts"]["train"] == first_index["counts"]["train"] + 1
    assert json.loads(second.model_group_json.read_text())["previous_model_group"] == str(first_group)


def test_previous_model_group_head_mismatch_requires_new_group(tmp_path):
    labels = _write_dummy_recorded_library(tmp_path / "library")
    previous_group = tmp_path / "model-groups" / "001"
    previous_group.mkdir(parents=True)
    (previous_group / "model_group.json").write_text(json.dumps({"head_order": ["oldHead"]}))

    with pytest.raises(ValueError, match="start a new model group without --previous-model-group"):
        recorded.build_recorded_library(
            cfg=Config.load("configs/smoke.yaml"),
            labels_manifest=labels,
            model_group_dir=tmp_path / "model-groups" / "002",
            previous_model_group=previous_group,
            frontend_model=tmp_path / "audio_frontend.tflite",
            train=False,
            evaluate=False,
            export=False,
        )


def test_iteration_warm_starts_from_previous_group_checkpoint(tmp_path, monkeypatch):
    labels = _write_dummy_recorded_library(tmp_path / "library")
    monkeypatch.setattr(recorded, "materialize_mel_manifest", _fake_materialize_mel_manifest)

    first_group = tmp_path / "model-groups" / "001"
    recorded.build_recorded_library(
        cfg=Config.load("configs/smoke.yaml"),
        labels_manifest=labels,
        model_group_dir=first_group,
        frontend_model=tmp_path / "audio_frontend.tflite",
        train=False,
        evaluate=False,
        export=False,
    )
    previous_checkpoint = first_group / "best.pt"
    previous_checkpoint.write_bytes(b"checkpoint")
    group_json = json.loads((first_group / "model_group.json").read_text())
    group_json["checkpoint"] = str(previous_checkpoint)
    (first_group / "model_group.json").write_text(json.dumps(group_json))

    seen = {}

    def fake_train(cfg, output_dir, *, init_checkpoint=None):
        seen["init_checkpoint"] = init_checkpoint
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "best.pt").write_bytes(b"new checkpoint")

    monkeypatch.setattr(recorded, "train_model", fake_train)
    result = recorded.build_recorded_library(
        cfg=Config.load("configs/smoke.yaml"),
        labels_manifest=labels,
        model_group_dir=tmp_path / "model-groups" / "002",
        previous_model_group=first_group,
        frontend_model=tmp_path / "audio_frontend.tflite",
        train=True,
        evaluate=False,
        export=False,
    )

    assert seen == {"init_checkpoint": previous_checkpoint}
    assert result.warm_start_checkpoint == previous_checkpoint


def _fake_materialize_mel_manifest(*, input_manifest, output_manifest, mel_dir, frontend_model):
    rows = list(csv.DictReader(Path(input_manifest).open()))
    Path(mel_dir).mkdir(parents=True, exist_ok=True)
    Path(output_manifest).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_manifest).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mel_path", *METADATA_COLUMNS, *HEAD_ORDER])
        writer.writeheader()
        for index, row in enumerate(rows):
            mel_path = Path(mel_dir) / f"{index:08d}.npy"
            np.save(mel_path, np.ones((3, 128), dtype=np.float32))
            writer.writerow(
                {
                    "mel_path": str(mel_path.resolve()),
                    **{column: row.get(column, "") for column in METADATA_COLUMNS},
                    **{head: row[head] for head in HEAD_ORDER},
                }
            )
    return len(rows)


def _write_dummy_recorded_library(root: Path) -> Path:
    root.mkdir(parents=True)
    audio_dir = root / "audio"
    audio_dir.mkdir()
    manifest = root / "labels.csv"
    rows = []
    for split in ("train", "eval", "test"):
        rows.append(_label_row(_write_wav(audio_dir / f"{split}_negative.wav"), split=split))
        for head in INTENT_HEAD_ORDER:
            rows.append(_label_row(_write_wav(audio_dir / f"{split}_{head}.wav"), split=split, positive_head=head))
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _label_row(
    path: Path,
    *,
    split: str,
    positive_head: str | None = None,
    speaker_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, str]:
    row = {
        "audio_path": str(path.relative_to(path.parents[1])),
        "transcript": positive_head or "ambient office conversation",
        "speaker_id": speaker_id or f"speaker-{split}-{positive_head or 'negative'}",
        "session_id": session_id or f"session-{split}-{positive_head or 'negative'}",
        "device": "fixture",
        "environment": "quiet",
        "source": "dummy_recorded_library",
        "duration_s": "0.25",
        "split": split,
    }
    for head in HEAD_ORDER:
        row[head] = "0"
    if positive_head is not None:
        row["isInteresting"] = "1"
        row[positive_head] = "1"
    return row


def _write_wav(path: Path) -> Path:
    import soundfile as sf

    sample_rate = 16_000
    t = np.linspace(0.0, 0.25, int(sample_rate * 0.25), endpoint=False)
    frequency = 220 + (sum(path.name.encode()) % 600)
    audio = (0.08 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    return path
