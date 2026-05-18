import json
import sys

import numpy as np
import pytest
import torch

from inherent import HEAD_ORDER
from inherent.config import Config, ExportConfig
from inherent.export.core import ExportResult, write_artifact_metadata
from inherent.export.litert import _delegate_report
from inherent.export import litertlm as litertlm_module
from inherent.export.preflight import preflight_backend, selected_backend_names
from inherent.export.litertlm import LiteRTLMBackend
from inherent.export.mlx import MLXBackend
from inherent.export.registry import get_backend, list_backends
from inherent.models import JointAudioIntentModel


def test_backend_registry_lists_expected_backends():
    assert {"onnx", "tflite", "litert", "mlx", "litertlm"}.issubset(set(list_backends()))
    assert get_backend("onnx").name == "onnx"


def test_export_config_accepts_backend_profiles():
    cfg = ExportConfig(backend="litert", delegates=["cpu", "gpu", "npu", "tpu"], onnx_opset=18)

    assert cfg.backend == "litert"
    assert cfg.delegates == ["cpu", "gpu", "npu", "tpu"]


def test_artifact_metadata_is_backend_aware(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    artifact = tmp_path / "inherent.onnx"
    artifact.write_bytes(b"onnx")
    metadata = tmp_path / "inherent.onnx.metadata.json"

    write_artifact_metadata(
        checkpoint_path=checkpoint,
        cfg=Config(),
        artifact_path=artifact,
        metadata_path=metadata,
        backend="onnx",
        artifact_format="onnx",
        reports={"parity": "report.json"},
    )

    data = json.loads(metadata.read_text())
    assert data["backend"] == "onnx"
    assert data["artifact_format"] == "onnx"
    assert data["artifact_size_bytes"] == len(b"onnx")
    assert tuple(data["head_order"]) == HEAD_ORDER
    assert data["reports"]["parity"] == "report.json"


def test_delegate_reports_explain_cpu_and_tpu_status(tmp_path):
    artifact = tmp_path / "model.tflite"
    artifact.write_bytes(b"tflite")
    cfg = Config()

    cpu = _delegate_report("cpu", artifact, cfg)
    npu = _delegate_report("npu", artifact, cfg)
    tpu = _delegate_report("tpu", artifact, cfg)

    assert cpu["status"] == "passed"
    assert npu["status"] == "skipped"
    assert npu["runtime_api"] == "LiteRT CompiledModel"
    assert tpu["status"] == "skipped"
    assert "fixed-shape" in tpu["reason"].lower()
    assert tpu["runtime_api"] == "Tensor ML SDK / LiteRT CompiledModel"


def test_litertlm_backend_invokes_builder_and_writes_package(monkeypatch, tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")

    def fake_tflite_export(checkpoint_path, cfg, output_dir, backend_name):
        tflite = output_dir / "inherent.tflite"
        tflite.write_bytes(b"tflite")
        return ExportResult(
            backend=backend_name,
            artifacts={"tflite": str(tflite)},
            reports={"onnx_parity": str(output_dir / "reports" / "onnx_parity.json")},
        )

    def fake_run_builder(toml_path, package_path, report_path):
        package_path.write_bytes(b"LITERTLM")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"backend": "litertlm", "status": "passed"}))
        return {"kind": "test_builder", "command": ["litert-lm-builder"], "cwd": None}

    monkeypatch.setattr(litertlm_module, "export_to_tflite", fake_tflite_export)
    monkeypatch.setattr(litertlm_module, "run_litertlm_builder", fake_run_builder)

    result = LiteRTLMBackend().export(
        checkpoint_path=checkpoint,
        cfg=Config(),
        output_dir=tmp_path / "export",
        delegates=("cpu", "gpu", "tpu"),
    )

    assert isinstance(result, ExportResult)
    assert result.supported is True
    assert (tmp_path / "export" / "inherent.litertlm").read_bytes() == b"LITERTLM"
    toml = (tmp_path / "export" / "litertlm_package.toml").read_text()
    assert 'model_type = "AUX"' in toml
    assert 'backend_constraint = "cpu,gpu,npu"' in toml
    metadata = json.loads((tmp_path / "export" / "inherent.metadata.json").read_text())
    assert metadata["artifact_format"] == "litertlm"
    assert metadata["source_tflite"].endswith("inherent.tflite")
    assert metadata["litertlm_builder"]["kind"] == "test_builder"


def test_litertlm_backend_propagates_packaging_failure(monkeypatch, tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")

    def fake_tflite_export(checkpoint_path, cfg, output_dir, backend_name):
        tflite = output_dir / "inherent.tflite"
        tflite.write_bytes(b"tflite")
        return ExportResult(backend=backend_name, artifacts={"tflite": str(tflite)})

    def fake_run_builder(toml_path, package_path, report_path):
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"backend": "litertlm", "status": "failed"}))
        raise RuntimeError("LiteRT-LM packaging failed.\ncommand trace: test\nerror trace: boom")

    monkeypatch.setattr(litertlm_module, "export_to_tflite", fake_tflite_export)
    monkeypatch.setattr(litertlm_module, "run_litertlm_builder", fake_run_builder)

    with pytest.raises(RuntimeError, match="LiteRT-LM packaging failed"):
        LiteRTLMBackend().export(
            checkpoint_path=checkpoint,
            cfg=Config(),
            output_dir=tmp_path / "export",
        )

    report = json.loads((tmp_path / "export" / "reports" / "litertlm_packaging.json").read_text())
    assert report["status"] == "failed"


def test_litertlm_resolver_fails_fast_before_bazel_without_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("INHERENT_LITERTLM_BUILDER_BIN", raising=False)
    monkeypatch.delenv("INHERENT_LITERTLM_ALLOW_BAZEL", raising=False)
    monkeypatch.setenv("INHERENT_LITERTLM_ROOT", str(tmp_path))
    monkeypatch.setattr(litertlm_module.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="Bazel fallback is opt-in"):
        litertlm_module.resolve_litertlm_builder_command(
            tmp_path / "litertlm_package.toml",
            tmp_path / "inherent.litertlm",
        )


def test_litertlm_resolver_uses_configured_builder_binary(monkeypatch, tmp_path):
    builder = tmp_path / "litert-lm-builder"
    builder.write_text("#!/bin/sh\nexit 0\n")
    builder.chmod(0o755)
    monkeypatch.setenv("INHERENT_LITERTLM_BUILDER_BIN", str(builder))
    monkeypatch.setenv("INHERENT_LITERTLM_BUILDER_TIMEOUT_SECONDS", "7.5")

    invocation = litertlm_module.resolve_litertlm_builder_command(
        tmp_path / "litertlm_package.toml",
        tmp_path / "inherent.litertlm",
    )

    assert invocation["kind"] == "python_package_binary"
    assert invocation["command"][0] == str(builder)
    assert invocation["timeout_seconds"] == 7.5


def test_litertlm_builder_timeout_writes_failure_report(monkeypatch, tmp_path):
    def fake_resolve(toml_path, package_path):
        return {
            "kind": "slow_test_builder",
            "command": [sys.executable, "-c", "import time; time.sleep(2)"],
            "cwd": None,
            "timeout_seconds": 0.01,
        }

    monkeypatch.setattr(litertlm_module, "resolve_litertlm_builder_command", fake_resolve)
    report_path = tmp_path / "reports" / "litertlm_packaging.json"

    with pytest.raises(RuntimeError, match="timed out"):
        litertlm_module.run_litertlm_builder(
            tmp_path / "litertlm_package.toml",
            tmp_path / "inherent.litertlm",
            report_path,
        )

    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["timeout_seconds"] == 0.01


def test_export_preflight_reports_litertlm_builder_blocker(monkeypatch, tmp_path):
    monkeypatch.delenv("INHERENT_LITERTLM_BUILDER_BIN", raising=False)
    monkeypatch.delenv("INHERENT_LITERTLM_ALLOW_BAZEL", raising=False)
    monkeypatch.setenv("INHERENT_LITERTLM_ROOT", str(tmp_path))
    monkeypatch.setattr(litertlm_module.shutil, "which", lambda _name: None)

    result = preflight_backend(
        cfg=Config(),
        backend_name="litertlm",
        delegates=("cpu",),
        output_dir=tmp_path / "export",
    )

    assert result["status"] == "failed"
    builder = next(check for check in result["checks"] if check["name"] == "litertlm_builder")
    assert "Bazel fallback is opt-in" in builder["reason"]


def test_selected_backend_names_expands_all_without_legacy_tflite():
    assert "tflite" not in selected_backend_names(Config(export=ExportConfig(backend="all")), None)
    assert selected_backend_names(Config(), "onnx") == ("onnx",)


def test_mlx_backend_exports_weight_package(tmp_path):
    cfg = Config()
    model = JointAudioIntentModel(cfg.model)
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "head_order": list(HEAD_ORDER),
            "config": {"model": cfg.model.__dict__},
        },
        checkpoint,
    )

    result = MLXBackend().export(
        checkpoint_path=checkpoint,
        cfg=cfg,
        output_dir=tmp_path / "export",
    )

    package = tmp_path / "export" / "mlx"
    assert result.artifacts["mlx_package"] == str(package)
    assert (package / "weights.npz").is_file()
    assert (package / "inference.py").is_file()
    with np.load(package / "weights.npz") as data:
        assert data.files
