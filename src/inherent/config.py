"""Config dataclasses loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import NUM_HEADS


RUNTIME_MEL_BINS = 128
RUNTIME_MAX_FRAMES = 3000
SUPPORTED_BACKBONE = "small_conformer"
SUPPORTED_QUANTIZATIONS = {"int8", "float16", "float32"}


@dataclass
class ModelConfig:
    backbone: str = SUPPORTED_BACKBONE
    num_heads: int = NUM_HEADS
    hidden_size: int = 256
    num_layers: int = 6
    num_attention_heads: int = 4
    conv_kernel_size: int = 31
    mel_bins: int = RUNTIME_MEL_BINS
    max_frames: int = RUNTIME_MAX_FRAMES

    def __post_init__(self) -> None:
        if self.backbone != SUPPORTED_BACKBONE:
            raise ValueError(f"unsupported backbone {self.backbone!r}; only {SUPPORTED_BACKBONE!r} is allowed")
        if self.num_heads != NUM_HEADS:
            raise ValueError(f"num_heads must be {NUM_HEADS}, got {self.num_heads}")
        if self.mel_bins != RUNTIME_MEL_BINS:
            raise ValueError(f"mel_bins must be {RUNTIME_MEL_BINS}, got {self.mel_bins}")
        if self.max_frames != RUNTIME_MAX_FRAMES:
            raise ValueError(f"max_frames must be {RUNTIME_MAX_FRAMES}, got {self.max_frames}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by "
                f"num_attention_heads={self.num_attention_heads}"
            )
        if self.conv_kernel_size % 2 == 0:
            raise ValueError(f"conv_kernel_size must be odd, got {self.conv_kernel_size}")


@dataclass
class DirectednessDataConfig:
    positives: list[str] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)
    positive_manifests: list[str] = field(default_factory=list)
    negative_manifests: list[str] = field(default_factory=list)
    labeled_manifests: list[str] = field(default_factory=list)
    max_public_samples_per_source: int | None = 20_000
    noise_mix: list[str] = field(default_factory=list)
    snr_db_range: tuple[int, int] = (0, 25)

    def __post_init__(self) -> None:
        if self.max_public_samples_per_source is not None and self.max_public_samples_per_source < 1:
            raise ValueError("max_public_samples_per_source must be positive when set")


@dataclass
class DataConfig:
    sample_rate: int = 16000
    hop_ms: int = 20
    mel_bins: int = 128
    max_seconds: int = 60
    directedness: DirectednessDataConfig = field(default_factory=DirectednessDataConfig)
    intents: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_rate != 16000:
            raise ValueError(f"sample_rate must be 16000, got {self.sample_rate}")
        if self.hop_ms != 20:
            raise ValueError(f"hop_ms must be 20, got {self.hop_ms}")
        if self.mel_bins != RUNTIME_MEL_BINS:
            raise ValueError(f"data.mel_bins must be {RUNTIME_MEL_BINS}, got {self.mel_bins}")
        if self.max_seconds != 60:
            raise ValueError(f"max_seconds must be 60, got {self.max_seconds}")


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    max_steps: int = 100_000
    eval_every_steps: int = 2000
    save_every_steps: int = 2000
    loss: str = "focal_bce"
    focal_gamma: float = 2.0
    class_weights: str = "balanced"
    sampler: str = "shuffle"
    padding: str = "dynamic"
    train_manifest: str = "data/train_manifest.csv"
    eval_manifest: str | None = None
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.eval_every_steps < 1:
            raise ValueError(f"eval_every_steps must be positive, got {self.eval_every_steps}")
        if self.save_every_steps < 1:
            raise ValueError(f"save_every_steps must be positive, got {self.save_every_steps}")
        if self.loss != "focal_bce":
            raise ValueError(f"unsupported training.loss {self.loss!r}; only 'focal_bce' is allowed")
        if self.focal_gamma < 0:
            raise ValueError(f"focal_gamma must be non-negative, got {self.focal_gamma}")
        if self.class_weights not in {"balanced", "none"}:
            raise ValueError("class_weights must be 'balanced' or 'none'")
        if self.sampler not in {"shuffle", "label_balanced"}:
            raise ValueError("sampler must be 'shuffle' or 'label_balanced'")
        if self.padding not in {"dynamic", "runtime_static"}:
            raise ValueError("padding must be 'dynamic' or 'runtime_static'")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")
        if self.device not in {"cuda", "mps"}:
            raise ValueError("training.device must be 'cuda' or 'mps'; CPU training is unsupported")


@dataclass
class ExportConfig:
    backend: str = "tflite"
    delegates: list[str] = field(default_factory=lambda: ["cpu"])
    quantization: str = "int8"
    representative_dataset_size: int = 500
    target_size_mb: int = 50
    output_path: str = "artifacts/inherent.tflite"
    metadata_path: str = "artifacts/inherent.metadata.json"
    onnx_path: str = "artifacts/inherent.onnx"
    onnx_metadata_path: str = "artifacts/inherent.onnx.metadata.json"
    mlx_dir: str = "artifacts/mlx"
    litertlm_path: str = "artifacts/inherent.litertlm"
    onnx_opset: int = 17
    onnx_sample_frames: int = 50
    onnx_static_frames: int | None = RUNTIME_MAX_FRAMES
    strict_tensor_names: bool = True
    parity_atol: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.backend not in {"tflite", "litert", "onnx", "mlx", "litertlm", "all"}:
            raise ValueError("export.backend must be one of: tflite, litert, onnx, mlx, litertlm, all")
        allowed_delegates = {
            "cpu",
            "gpu",
            "npu",
            "tpu",
            "qualcomm",
            "mediatek",
            "intel",
            "google_tensor",
            "all",
        }
        unexpected_delegates = sorted(set(self.delegates) - allowed_delegates)
        if unexpected_delegates:
            raise ValueError(f"unsupported export delegates: {unexpected_delegates}")
        if self.quantization not in SUPPORTED_QUANTIZATIONS:
            raise ValueError(f"export.quantization must be one of {sorted(SUPPORTED_QUANTIZATIONS)}")
        if self.representative_dataset_size < 1:
            raise ValueError("representative_dataset_size must be positive")
        if self.target_size_mb < 1:
            raise ValueError("target_size_mb must be positive")
        if self.onnx_opset < 17:
            raise ValueError("onnx_opset must be >= 17")
        if self.onnx_sample_frames < 1:
            raise ValueError("onnx_sample_frames must be positive")
        if self.onnx_static_frames is not None and self.onnx_static_frames < 1:
            raise ValueError("onnx_static_frames must be positive when set")
        if self.parity_atol <= 0:
            raise ValueError("parity_atol must be positive")


@dataclass
class EvalConfig:
    held_out_audio_dir: str = "data/eval_recorded"
    metrics: list[str] = field(default_factory=list)
    pass_threshold: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(
                **{k: v for k, v in raw.get("data", {}).items() if k != "directedness"},
                directedness=DirectednessDataConfig(**raw.get("data", {}).get("directedness", {})),
            ),
            training=TrainingConfig(**raw.get("training", {})),
            export=ExportConfig(**raw.get("export", {})),
            eval=EvalConfig(**raw.get("eval", {})),
        )
