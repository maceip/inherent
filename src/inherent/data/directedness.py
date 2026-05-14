"""Build the directedness corpus.

The `isInteresting` head asks: "is this audio addressed at the assistant?"
No public DDSD-labeled corpus exists. We construct labels by source:

- Positive (directed): SLURP, MASSIVE-audio, Speech-MASSIVE, STOP, FalAI — every
  utterance in these is by definition a directed command to a voice assistant.
- Negative (undirected): openwakeword_features (30,000h precomputed negatives),
  AMI/ICSI meetings, LibriSpeech, Speech Commands, CallHome, MagicData-RAMC, and
  vetted ambient/non-speech trees such as FSD50K or Freesound exports.
- Augmentation: mix with MUSAN + DEMAND noise at randomized SNR; apply room IRs
  for far-field simulation.

This module assembles the corpus index but does not load audio into memory.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import HEAD_ORDER, INTERESTING_HEAD
from .schema import OPTIONAL_LABEL_COLUMNS
from ..config import DirectednessDataConfig
from ..features.frontend import MAX_SAMPLES, SAMPLE_RATE


@dataclass(frozen=True)
class DirectednessSample:
    audio_path: Path
    label: int  # 1 = directed, 0 = undirected
    source: str
    duration_s: float
    snr_db: float | None = None
    noise_path: Path | None = None


@dataclass(frozen=True)
class _HfAudioSource:
    dataset_id: str
    configs: tuple[str, ...]
    splits: tuple[str, ...]
    source: str
    license: str
    notes: str


@dataclass(frozen=True)
class _HfAudioRow:
    row: Mapping[str, Any]
    dataset_id: str
    config_name: str
    split: str
    row_index: int
    features: Mapping[str, Any]


PUBLIC_HF_NEGATIVE_SOURCES = {
    "librispeech": _HfAudioSource(
        dataset_id="openslr/librispeech_asr",
        configs=("clean", "other"),
        splits=("train.100", "train.360", "validation", "test"),
        source="librispeech",
        license="CC BY 4.0",
        notes="read audiobook speech; useful as broad non-command speech negatives",
    ),
    "speech_commands": _HfAudioSource(
        dataset_id="google/speech_commands",
        configs=("v0.02",),
        splits=("train", "validation", "test"),
        source="speech_commands",
        license="CC BY 4.0",
        notes="keyword-command clips; useful as short hard negatives and near misses",
    ),
}


def build_index(cfg: DirectednessDataConfig, data_root: Path) -> list[DirectednessSample]:
    """Walk configured corpora, return a manifest of (path, label, source) entries.

    Implementation notes:
    - Each corpus loader is a function below (not implemented yet); add as needed.
    - Total count target: ~200k positives + 200k negatives, balanced.
    - Audio files stay on disk; we lazy-load in the dataloader.
    """
    samples: list[DirectednessSample] = []
    for manifest in cfg.positive_manifests:
        samples.extend(load_recorded_manifest(_resolve_path(manifest, data_root), default_label=1))
    for manifest in cfg.negative_manifests:
        samples.extend(load_recorded_manifest(_resolve_path(manifest, data_root), default_label=0))
    for manifest in cfg.labeled_manifests:
        samples.extend(load_recorded_manifest(_resolve_path(manifest, data_root), default_label=None))
    source_loaders = {
        "slurp": load_slurp_positives,
        "massive_audio": load_massive_audio_positives,
        "speech_massive": load_speech_massive_positives,
        "stop": load_stop_positives,
        "openwakeword_features": load_openwakeword_negatives,
        "ami": load_ami_negatives,
        "icsi": load_icsi_negatives,
        "librispeech": lambda root: load_librispeech_negatives(
            root,
            max_samples=cfg.max_public_samples_per_source,
        ),
        "speech_commands": lambda root: load_speech_commands_negatives(
            root,
            max_samples=cfg.max_public_samples_per_source,
        ),
        "callhome": load_callhome_negatives,
        "magicdata_ramc": load_magicdata_ramc_negatives,
        "fsd50k": load_fsd50k_negatives,
        "esc10": load_esc10_negatives,
        "esc50": load_esc50_negatives,
        "urbansound8k": load_urbansound8k_negatives,
        "freesound_ambient": load_freesound_ambient_negatives,
    }
    for source in cfg.positives + cfg.negatives:
        if source not in source_loaders:
            raise ValueError(f"unknown directedness source {source!r}")
        samples.extend(source_loaders[source](data_root / source))
    _require_both_classes(samples)
    return samples


def load_recorded_manifest(path: Path, default_label: int | None) -> list[DirectednessSample]:
    """Load directedness examples from a strict CSV manifest.

    Required column:
      audio_path

    Optional columns:
      label, isInteresting, duration_s, source, and ignored intent heads.

    If default_label is None, either label or isInteresting is required. This
    lets one 13-head recorded CSV serve both directedness and intent ingestion.
    """
    rows: list[DirectednessSample] = []
    allowed_columns = {"audio_path", "label", "duration_s", "source", *OPTIONAL_LABEL_COLUMNS, *HEAD_ORDER}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"directedness manifest has no header: {path}")
        missing = ["audio_path"] if "audio_path" not in reader.fieldnames else []
        if default_label is None and "label" not in reader.fieldnames and INTERESTING_HEAD not in reader.fieldnames:
            missing.append(f"label or {INTERESTING_HEAD}")
        if missing:
            raise ValueError(f"directedness manifest {path} missing required columns: {missing}")
        unexpected = [column for column in reader.fieldnames if column not in allowed_columns]
        if unexpected:
            raise ValueError(f"directedness manifest {path} has unexpected columns: {unexpected}")
        for row_number, row in enumerate(reader, start=2):
            label = (
                default_label
                if default_label is not None
                else _parse_label(_label_value(row, reader.fieldnames), path, row_number)
            )
            audio_path = _resolve_path(row["audio_path"], path.parent)
            if not audio_path.is_file():
                raise FileNotFoundError(f"directedness audio path does not exist at {path}:{row_number}: {audio_path}")
            rows.append(
                DirectednessSample(
                    audio_path=audio_path,
                    label=label,
                    source=_empty_to_default(row.get("source"), f"recorded:{path.name}"),
                    duration_s=_parse_float(row.get("duration_s"), default=0.0),
                )
            )
    if not rows:
        raise ValueError(f"directedness manifest contains no rows: {path}")
    return rows


def load_slurp_positives(root: Path) -> list[DirectednessSample]:
    from . import intents

    return _from_intent_samples(intents.load_slurp(root), source_prefix="directed:slurp")


def load_massive_audio_positives(root: Path) -> list[DirectednessSample]:
    from . import intents

    return _from_intent_samples(intents.load_speech_massive(root), source_prefix="directed:speech_massive")


def load_speech_massive_positives(root: Path) -> list[DirectednessSample]:
    from . import intents

    return _from_intent_samples(intents.load_speech_massive(root), source_prefix="directed:speech_massive")


def load_stop_positives(root: Path) -> list[DirectednessSample]:
    from . import intents

    return _from_intent_samples(intents.load_stop(root), source_prefix="directed:stop")


def load_openwakeword_negatives(root: Path) -> list[DirectednessSample]:
    """davidscripka/openwakeword_features — ~30,000h, precomputed features.

    For our purposes we want raw audio not features. Use the underlying source
    corpora (MUSAN, Freesound, etc.) referenced by the openwakeword config.
    """
    raise NotImplementedError


def load_ami_negatives(root: Path) -> list[DirectednessSample]:
    """AMI meeting corpus, preprocessed to 16 kHz mono WAV files."""
    return _load_negative_audio_tree(root, source="ami")


def load_icsi_negatives(root: Path) -> list[DirectednessSample]:
    """ICSI meeting corpus, preprocessed to 16 kHz mono WAV files."""
    return _load_negative_audio_tree(root, source="icsi")


def load_librispeech_negatives(root: Path, *, max_samples: int | None) -> list[DirectednessSample]:
    """LibriSpeech ASR, decoded via Hugging Face Datasets."""
    return _load_hf_audio_negatives(
        root,
        spec=PUBLIC_HF_NEGATIVE_SOURCES["librispeech"],
        max_samples=max_samples,
    )


def load_speech_commands_negatives(root: Path, *, max_samples: int | None) -> list[DirectednessSample]:
    """Google Speech Commands, decoded via Hugging Face Datasets."""
    return _load_hf_audio_negatives(
        root,
        spec=PUBLIC_HF_NEGATIVE_SOURCES["speech_commands"],
        max_samples=max_samples,
    )


def load_callhome_negatives(root: Path) -> list[DirectednessSample]:
    """CallHome conversations, preprocessed to 16 kHz mono WAV files."""
    return _load_negative_audio_tree(root, source="callhome")


def load_magicdata_ramc_negatives(root: Path) -> list[DirectednessSample]:
    """MagicData-RAMC conversations, preprocessed to 16 kHz mono WAV files."""
    return _load_negative_audio_tree(root, source="magicdata_ramc")


def load_fsd50k_negatives(root: Path) -> list[DirectednessSample]:
    """FSD50K ambient/non-speech export, pre-filtered to product-acceptable licenses."""
    return _load_negative_audio_tree(root, source="fsd50k")


def load_esc10_negatives(root: Path) -> list[DirectednessSample]:
    """ESC-10 ambient audio tree. ESC-10 is permissive; keep attribution metadata externally."""
    return _load_negative_audio_tree(root, source="esc10")


def load_esc50_negatives(root: Path) -> list[DirectednessSample]:
    """ESC-50 ambient audio tree. ESC-50 is NC; use only when that is acceptable."""
    return _load_negative_audio_tree(root, source="esc50")


def load_urbansound8k_negatives(root: Path) -> list[DirectednessSample]:
    """UrbanSound8K ambient audio tree, preprocessed to 16 kHz mono WAV files."""
    return _load_negative_audio_tree(root, source="urbansound8k")


def load_freesound_ambient_negatives(root: Path) -> list[DirectednessSample]:
    """Custom Freesound ambient export, pre-filtered to CC0/CC-BY clips."""
    return _load_negative_audio_tree(root, source="freesound_ambient")


def _require_both_classes(samples: list[DirectednessSample]) -> None:
    labels = {sample.label for sample in samples}
    if labels != {0, 1}:
        raise ValueError(f"directedness samples must contain both classes, got {sorted(labels)}")


def _from_intent_samples(samples, *, source_prefix: str) -> list[DirectednessSample]:
    rows = [
        DirectednessSample(
            audio_path=sample.audio_path,
            label=1,
            source=f"{source_prefix}:{sample.source}",
            duration_s=sample.duration_s,
        )
        for sample in samples
    ]
    if not rows:
        raise ValueError(f"{source_prefix} produced no directedness positives")
    return rows


def _load_hf_audio_negatives(
    root: Path,
    *,
    spec: _HfAudioSource,
    max_samples: int | None,
) -> list[DirectednessSample]:
    samples: list[DirectednessSample] = []
    for hf_row in _iter_hf_audio_rows(root=root, spec=spec, max_samples=max_samples):
        audio = _extract_audio(hf_row.row, hf_row.dataset_id, hf_row.split, hf_row.row_index)
        duration_s = _duration_seconds(audio["array"], audio["sampling_rate"])
        audio_path = _materialize_audio(root, spec.source, hf_row, audio)
        samples.append(
            DirectednessSample(
                audio_path=audio_path,
                label=0,
                source=f"{spec.source}:{hf_row.config_name}:{hf_row.split}",
                duration_s=duration_s,
            )
        )
    if not samples:
        raise ValueError(f"{spec.dataset_id} produced no negative speech samples")
    return samples


def _iter_hf_audio_rows(
    *,
    root: Path,
    spec: _HfAudioSource,
    max_samples: int | None,
) -> Iterator[_HfAudioRow]:
    try:
        from datasets import Audio, get_dataset_config_names, get_dataset_split_names, load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for public-corpus directedness loaders") from exc

    root.mkdir(parents=True, exist_ok=True)
    cache_dir = root / ".hf_cache"
    available_configs = set(get_dataset_config_names(spec.dataset_id))
    selected_configs = tuple(config for config in spec.configs if config in available_configs)
    if not selected_configs:
        raise ValueError(
            f"{spec.dataset_id} configs {spec.configs} not found; available={sorted(available_configs)}"
        )

    yielded = 0
    for config_name in selected_configs:
        available_splits = tuple(get_dataset_split_names(spec.dataset_id, config_name))
        selected_splits = tuple(split for split in spec.splits if split in available_splits) or available_splits
        for split in selected_splits:
            dataset = load_dataset(
                spec.dataset_id,
                config_name,
                split=split,
                cache_dir=str(cache_dir),
                trust_remote_code=True,
            )
            audio_column = _find_audio_column(dataset.features)
            dataset = dataset.cast_column(audio_column, Audio(sampling_rate=SAMPLE_RATE, num_channels=1))
            for row_index, row in enumerate(dataset):
                yield _HfAudioRow(
                    row=row,
                    dataset_id=spec.dataset_id,
                    config_name=config_name,
                    split=split,
                    row_index=row_index,
                    features=dataset.features,
                )
                yielded += 1
                if max_samples is not None and yielded >= max_samples:
                    return


def _load_negative_audio_tree(root: Path, *, source: str) -> list[DirectednessSample]:
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"{source} negative corpus root not found: {root}")
    audio_paths = sorted(path for path in root.rglob("*.wav") if path.is_file())
    if not audio_paths:
        raise ValueError(f"{source} negative corpus contains no .wav files: {root}")

    samples: list[DirectednessSample] = []
    for audio_path in audio_paths:
        info = _read_audio_info(audio_path)
        if info.samplerate != SAMPLE_RATE:
            raise ValueError(f"{audio_path} sample rate must be {SAMPLE_RATE}, got {info.samplerate}")
        if info.channels != 1:
            raise ValueError(f"{audio_path} must be mono, got {info.channels} channels")
        if info.frames < 1 or info.frames > MAX_SAMPLES:
            raise ValueError(f"{audio_path} frame count must be 1..{MAX_SAMPLES}, got {info.frames}")
        samples.append(
            DirectednessSample(
                audio_path=audio_path,
                label=0,
                source=source,
                duration_s=info.frames / SAMPLE_RATE,
            )
        )
    return samples


def _extract_audio(
    row: Mapping[str, Any],
    dataset_id: str,
    split: str,
    row_index: int,
) -> Mapping[str, Any]:
    audio_columns = [column for column, value in row.items() if isinstance(value, Mapping) and "array" in value]
    if len(audio_columns) != 1:
        raise ValueError(
            f"{dataset_id}:{split}:{row_index} expected exactly one decoded audio column, got {audio_columns}"
        )
    return row[audio_columns[0]]


def _find_audio_column(features: Mapping[str, Any]) -> str:
    candidates = [name for name in features if name == "audio"]
    if len(candidates) == 1:
        return candidates[0]
    audio_like = [name for name, feature in features.items() if feature.__class__.__name__ == "Audio"]
    if len(audio_like) == 1:
        return audio_like[0]
    raise ValueError(f"expected exactly one audio feature, got {audio_like}")


def _materialize_audio(root: Path, source_name: str, hf_row: _HfAudioRow, audio: Mapping[str, Any]) -> Path:
    output_dir = root / "materialized_audio" / source_name / hf_row.config_name / hf_row.split
    digest = hashlib.sha1(
        f"{hf_row.dataset_id}:{hf_row.config_name}:{hf_row.split}:{hf_row.row_index}".encode()
    ).hexdigest()[:16]
    output_path = output_dir / f"{hf_row.row_index:08d}_{digest}.wav"
    if output_path.is_file():
        return output_path
    _write_wav_16k(output_path, audio["array"], audio["sampling_rate"])
    return output_path


def _write_wav_16k(path: Path, array: Any, sampling_rate: int) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required to materialize public-corpus audio") from exc

    if sampling_rate != SAMPLE_RATE:
        raise ValueError(f"decoded public-corpus audio must be {SAMPLE_RATE} Hz, got {sampling_rate}")
    samples = np.asarray(array, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError(f"decoded public-corpus audio must be mono, got shape {samples.shape}")
    if samples.shape[0] < 1 or samples.shape[0] > MAX_SAMPLES:
        raise ValueError(f"decoded public-corpus audio length must be 1..{MAX_SAMPLES}, got {samples.shape[0]}")
    if not np.isfinite(samples).all():
        raise ValueError("decoded public-corpus audio contains non-finite samples")
    peak = float(np.max(np.abs(samples)))
    if peak > 1.0:
        raise ValueError(f"decoded public-corpus audio must be normalized to [-1, 1], peak={peak}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, SAMPLE_RATE, subtype="PCM_16")


def _duration_seconds(array: Any, sampling_rate: int) -> float:
    if sampling_rate != SAMPLE_RATE:
        raise ValueError(f"decoded audio sampling_rate must be {SAMPLE_RATE}, got {sampling_rate}")
    samples = np.asarray(array)
    return float(samples.shape[0] / SAMPLE_RATE)


def _read_audio_info(path: Path):
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required to index directedness audio trees") from exc
    return sf.info(path)


def _label_value(row: dict[str, str], fieldnames: list[str] | None) -> str:
    if fieldnames is not None and "label" in fieldnames:
        return row["label"]
    return row[INTERESTING_HEAD]


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    if str(value).strip() == "":
        raise ValueError("audio_path must be non-empty")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _parse_label(value: str, path: Path, row_number: int) -> int:
    normalized = value.strip()
    if normalized == "1":
        return 1
    if normalized == "0":
        return 0
    raise ValueError(f"invalid directedness label {value!r} at {path}:{row_number}; expected 0 or 1")


def _parse_float(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"duration_s must be non-negative, got {parsed}")
    return parsed


def _empty_to_default(value: str | None, default: str) -> str:
    if value is None or value.strip() == "":
        return default
    return value
