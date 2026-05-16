"""Build the per-head intent corpus from public sources.

Coverage map (verified against HF + ModelScope as of 2026-05):

| Head | Public source | Notes |
|---|---|---|
| hasAddToListIntent | SLURP `lists_*`, MASSIVE `lists_createoradd` | clean coverage |
| hasTermSearchQuery | SLURP `qa_*`, MINDS-14, MASSIVE `general_quirky` | clean coverage |
| hasCalendarEvent | SLURP `calendar_*`, STOP `reminder` | clean coverage |
| hasPersonContext | SLURP `recommendation` | proxy coverage |
| hasEventContext | SLURP `calendar_query` | proxy coverage |
| hasStartTimerIntent | SLURP/MASSIVE `alarm_set`, STOP `timer` | clean coverage |
| hasPhotoQuery | NONE | synthesize via TTS (data/synthesis.py) |
| hasCreateDocIntent | NONE | synthesize via TTS |
| hasDeepResearchIntent | NONE | synthesize via TTS |
| hasInsightIntent | NONE | synthesize via TTS |
| hasBrowsingAgentIntent | NONE | synthesize via TTS |
| hasCallingAgentIntent | NONE | synthesize via TTS |

This module assembles the public-source manifest. It maps clean public coverage
where available and two documented SLURP proxy heads. Synthetic data is built by
synthesis.py and merged separately.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import HEAD_ORDER, INTENT_HEAD_ORDER
from .schema import OPTIONAL_LABEL_COLUMNS
from ..features.frontend import MAX_SAMPLES, SAMPLE_RATE


@dataclass(frozen=True)
class IntentSample:
    audio_path: Path
    transcript: str | None
    head_labels: dict[str, bool]  # multi-label: {"hasAddToListIntent": True, ...}
    source: str
    duration_s: float


@dataclass(frozen=True)
class _HfRow:
    row: Mapping[str, Any]
    dataset_id: str
    config_name: str
    split: str
    row_index: int
    features: Mapping[str, Any]


PUBLIC_SOURCE_LOADERS = {
    "slurp": "load_slurp",
    "speech_massive": "load_speech_massive",
    "massive": "load_speech_massive",
    "stop": "load_stop",
}

PUBLIC_CORPUS_HEADS = (
    "hasAddToListIntent",
    "hasTermSearchQuery",
    "hasCalendarEvent",
    "hasPersonContext",
    "hasEventContext",
    "hasStartTimerIntent",
)

SPEECH_MASSIVE_CONFIGS = (
    "ar-SA",
    "de-DE",
    "es-ES",
    "fr-FR",
    "hu-HU",
    "ko-KR",
    "nl-NL",
    "pl-PL",
    "pt-PT",
    "ru-RU",
    "tr-TR",
    "vi-VN",
)

STOP_PARSE_HEAD_TOKENS = {
    "hasCalendarEvent": (
        "IN:CREATE_EVENT",
        "IN:UPDATE_EVENT",
        "IN:DELETE_EVENT",
        "IN:GET_EVENT",
        "IN:CREATE_REMINDER",
        "IN:UPDATE_REMINDER",
        "IN:DELETE_REMINDER",
        "IN:GET_REMINDER",
    ),
    "hasStartTimerIntent": (
        "IN:CREATE_TIMER",
        "IN:UPDATE_TIMER",
        "IN:DELETE_TIMER",
        "IN:GET_TIMER",
        "IN:CREATE_ALARM",
        "IN:UPDATE_ALARM",
        "IN:DELETE_ALARM",
        "IN:GET_ALARM",
    ),
}


def build_index(intents_cfg: dict, data_root: Path) -> list[IntentSample]:
    """Assemble the manifest from configured public sources."""
    samples: list[IntentSample] = []
    for manifest in intents_cfg.get("recorded", []):
        samples.extend(load_recorded_manifest(_resolve_path(manifest, data_root)))
    for manifest in intents_cfg.get("synthetic_manifests", []):
        samples.extend(load_synthetic_manifest(_resolve_path(manifest, data_root)))

    sources = _configured_public_sources(intents_cfg, required=not samples)
    for source in sources:
        if source == "slurp":
            samples.extend(load_slurp(data_root / "slurp"))
        elif source in {"speech_massive", "massive"}:
            samples.extend(load_speech_massive(data_root / "speech_massive"))
        elif source == "stop":
            samples.extend(load_stop(data_root / "stop"))
        else:
            raise ValueError(f"unknown public intent source {source!r}")
    if sources:
        _require_public_intent_coverage(samples)
    elif not samples:
        raise ValueError("no intent sources configured")
    return samples


def load_recorded_manifest(path: Path) -> list[IntentSample]:
    """Load user-recorded clips from a CSV manifest.

    Required column:
      audio_path

    Optional columns:
      transcript, duration_s, and any head in HEAD_ORDER. The isInteresting
      column is accepted for shared eval manifests but ignored here because
      this loader returns per-intent labels.
    """
    samples: list[IntentSample] = []
    base_dir = path.parent
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"recorded manifest has no header: {path}")
        allowed_columns = {"audio_path", "transcript", "duration_s", *OPTIONAL_LABEL_COLUMNS, *HEAD_ORDER}
        missing = ["audio_path"] if "audio_path" not in reader.fieldnames else []
        if missing:
            raise ValueError(f"recorded manifest missing required columns {missing}: {path}")
        unexpected = [column for column in reader.fieldnames if column not in allowed_columns]
        if unexpected:
            raise ValueError(f"recorded manifest has unexpected columns {unexpected}: {path}")
        for row_number, row in enumerate(reader, start=2):
            audio_path = _resolve_path(row["audio_path"], base_dir)
            labels = {
                head: _parse_bool(row.get(head, "0"), path, row_number, head)
                for head in INTENT_HEAD_ORDER
            }
            if not any(labels.values()):
                continue
            samples.append(
                IntentSample(
                    audio_path=audio_path,
                    transcript=_empty_to_none(row.get("transcript")),
                    head_labels=labels,
                    source=f"recorded:{path.name}",
                    duration_s=_parse_float(row.get("duration_s"), default=0.0),
                )
            )
    return samples


def load_slurp(root: Path) -> list[IntentSample]:
    """SLURP — 177h / 72k, 18 domains, 60 intents.

    Map SLURP intents to our 13-head labels:
      lists_* → hasAddToListIntent
      qa_* → hasTermSearchQuery
      calendar_create → hasCalendarEvent
      alarm_set → hasStartTimerIntent
      recommendation → hasPersonContext
      calendar_query → hasEventContext
    """
    return _load_hf_intent_dataset(
        dataset_id="qmeeus/slurp",
        root=root,
        source_name="slurp",
        configs=("default",),
    )


def load_speech_massive(root: Path) -> list[IntentSample]:
    """FBK-MT/Speech-MASSIVE — 48.8k recordings, 12 langs, same 60 intents as MASSIVE."""
    return _load_hf_intent_dataset(
        dataset_id="FBK-MT/Speech-MASSIVE",
        root=root,
        source_name="speech_massive",
        configs=SPEECH_MASSIVE_CONFIGS,
    )


def load_stop(root: Path) -> list[IntentSample]:
    """Meta STOP — ~200h, includes TTS-augmented split."""
    return _load_stop_local(root)


def load_falai(root: Path) -> list[IntentSample]:
    """GTM-UVigo/FalAI — 250h Galician, calendar/lists/e-health/e-gov."""
    raise NotImplementedError


def load_minds14(root: Path) -> list[IntentSample]:
    """PolyAI/minds14 — 654 audio, 14 banking intents."""
    raise NotImplementedError


def load_slue_hvb(root: Path) -> list[IntentSample]:
    """asapp/slue-phase-2 HVB split — 23h call-context audio."""
    raise NotImplementedError


def load_axondata_call_center(root: Path) -> list[IntentSample]:
    """AxonData/multilingual-call-center-speech-dataset — calling agent context."""
    raise NotImplementedError


def load_synthetic_manifest(path: Path) -> list[IntentSample]:
    """Load TTS output from data.synthesis.write_synthetic_manifest."""
    samples: list[IntentSample] = []
    allowed_columns = {"audio_path", "transcript", "head", "voice_id", "tts_engine"}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"synthetic manifest has no header: {path}")
        missing = [column for column in ("audio_path", "transcript", "head") if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"synthetic manifest {path} missing required columns: {missing}")
        unexpected = [column for column in reader.fieldnames if column not in allowed_columns]
        if unexpected:
            raise ValueError(f"synthetic manifest {path} has unexpected columns: {unexpected}")
        for row_number, row in enumerate(reader, start=2):
            head = row["head"].strip()
            if head not in INTENT_HEAD_ORDER:
                raise ValueError(f"invalid synthetic head {head!r} at {path}:{row_number}")
            labels = _empty_intent_labels()
            labels[head] = True
            samples.append(
                IntentSample(
                    audio_path=_resolve_path(row["audio_path"], path.parent),
                    transcript=row["transcript"],
                    head_labels=labels,
                    source=f"synthetic:{_empty_to_none(row.get('tts_engine')) or 'tts'}",
                    duration_s=0.0,
                )
            )
    if not samples:
        raise ValueError(f"synthetic manifest contains no rows: {path}")
    return samples


def _configured_public_sources(intents_cfg: dict, *, required: bool) -> tuple[str, ...]:
    explicit = intents_cfg.get("public_sources")
    if explicit is not None:
        sources = tuple(str(source) for source in explicit)
    else:
        source_names: set[str] = set()
        for entries in intents_cfg.get("public", {}).values():
            for entry in entries:
                prefix = str(entry).split(".", 1)[0]
                if prefix in PUBLIC_SOURCE_LOADERS:
                    source_names.add("speech_massive" if prefix == "massive" else prefix)
        sources = tuple(sorted(source_names))
    if not sources:
        if required:
            raise ValueError("no public intent sources configured")
        return ()
    unknown = [source for source in sources if source not in PUBLIC_SOURCE_LOADERS]
    if unknown:
        raise ValueError(f"unknown public intent sources: {unknown}")
    return sources


def _load_hf_intent_dataset(
    *,
    dataset_id: str,
    root: Path,
    source_name: str,
    configs: Sequence[str],
) -> list[IntentSample]:
    samples: list[IntentSample] = []
    for hf_row in _iter_hf_rows(dataset_id=dataset_id, root=root, configs=configs):
        label = _extract_label(hf_row.row, hf_row.features)
        head_labels = _heads_for_public_intent(label)
        if not any(head_labels.values()):
            continue
        audio = _extract_audio(hf_row.row, hf_row.dataset_id, hf_row.split, hf_row.row_index)
        duration_s = _duration_seconds(audio["array"], audio["sampling_rate"])
        audio_path = _materialize_audio(root, source_name, hf_row, audio)
        samples.append(
            IntentSample(
                audio_path=audio_path,
                transcript=_extract_transcript(hf_row.row),
                head_labels=head_labels,
                source=f"{source_name}:{hf_row.config_name}:{hf_row.split}",
                duration_s=duration_s,
            )
        )
    if not samples:
        raise ValueError(f"{dataset_id} produced no mapped intent samples")
    return samples


def _iter_hf_rows(*, dataset_id: str, root: Path, configs: Sequence[str]) -> Iterator[_HfRow]:
    try:
        from datasets import Audio, get_dataset_config_names, get_dataset_split_names, load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for public-corpus intent loaders") from exc

    root.mkdir(parents=True, exist_ok=True)
    cache_dir = root / ".hf_cache"
    available_configs = set(get_dataset_config_names(dataset_id))
    selected_configs = tuple(config for config in configs if config in available_configs)
    if not selected_configs:
        raise ValueError(
            f"{dataset_id} configs {tuple(configs)} not found; available={sorted(available_configs)}"
        )

    for config_name in selected_configs:
        splits = get_dataset_split_names(dataset_id, config_name)
        for split in splits:
            dataset = load_dataset(
                dataset_id,
                config_name,
                split=split,
                cache_dir=str(cache_dir),
                trust_remote_code=False,
            )
            audio_column = _find_audio_column(dataset.features)
            dataset = dataset.cast_column(audio_column, Audio(sampling_rate=SAMPLE_RATE, mono=True))
            for row_index, row in enumerate(dataset):
                yield _HfRow(
                    row=row,
                    dataset_id=dataset_id,
                    config_name=config_name,
                    split=split,
                    row_index=row_index,
                    features=dataset.features,
                )


def _load_stop_local(root: Path) -> list[IntentSample]:
    if not root.is_dir():
        raise FileNotFoundError(
            f"STOP corpus root not found: {root}. Download STOP from Meta/fairseq first; "
            "the loader expects .tsv/.ltr/.parse manifest triples."
        )
    triples = _find_stop_manifest_triples(root)
    if not triples:
        raise ValueError(f"no STOP .tsv/.ltr/.parse manifest triples found under {root}")
    samples: list[IntentSample] = []
    for tsv_path, ltr_path, parse_path in triples:
        samples.extend(_load_stop_manifest_triple(tsv_path, ltr_path, parse_path))
    if not samples:
        raise ValueError(f"STOP corpus {root} produced no mapped intent samples")
    return samples


def _find_stop_manifest_triples(root: Path) -> list[tuple[Path, Path, Path]]:
    triples: list[tuple[Path, Path, Path]] = []
    for tsv_path in sorted(root.rglob("*.tsv")):
        ltr_path = tsv_path.with_suffix(".ltr")
        parse_path = tsv_path.with_suffix(".parse")
        if ltr_path.is_file() and parse_path.is_file():
            triples.append((tsv_path, ltr_path, parse_path))
    return triples


def _load_stop_manifest_triple(tsv_path: Path, ltr_path: Path, parse_path: Path) -> list[IntentSample]:
    audio_paths = _read_stop_tsv(tsv_path)
    transcripts = ltr_path.read_text().splitlines()
    parses = parse_path.read_text().splitlines()
    if not (len(audio_paths) == len(transcripts) == len(parses)):
        raise ValueError(
            f"STOP manifest length mismatch for {tsv_path}: "
            f"audio={len(audio_paths)}, text={len(transcripts)}, parse={len(parses)}"
        )

    samples: list[IntentSample] = []
    for audio_path, transcript, parse in zip(audio_paths, transcripts, parses, strict=True):
        head_labels = _heads_for_stop_parse(parse)
        if not any(head_labels.values()):
            continue
        samples.append(
            IntentSample(
                audio_path=audio_path,
                transcript=transcript.strip(),
                head_labels=head_labels,
                source=f"stop:{tsv_path.stem}",
                duration_s=0.0,
            )
        )
    return samples


def _read_stop_tsv(path: Path) -> list[Path]:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"STOP tsv must contain a root line and at least one sample: {path}")
    audio_root = _resolve_path(lines[0].strip(), path.parent)
    audio_paths: list[Path] = []
    for line_number, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) < 1 or parts[0].strip() == "":
            raise ValueError(f"invalid STOP tsv row at {path}:{line_number}")
        audio_path = _resolve_path(parts[0].strip(), audio_root)
        if not audio_path.is_file():
            raise FileNotFoundError(f"STOP audio path does not exist at {path}:{line_number}: {audio_path}")
        audio_paths.append(audio_path)
    return audio_paths


def _heads_for_public_intent(label: str) -> dict[str, bool]:
    normalized = _normalize_label(label)
    labels = _empty_intent_labels()
    if normalized.startswith("lists_") or normalized in {"add_to_list", "create_list", "lists_createoradd"}:
        labels["hasAddToListIntent"] = True
    if normalized.startswith("qa_") or normalized in {"general_quirky", "query", "search_query"}:
        labels["hasTermSearchQuery"] = True
    if normalized in {
        "calendar_set",
        "calendar_create",
        "calendar_update",
        "calendar_remove",
        "reminder_set",
        "reminder_create",
    }:
        labels["hasCalendarEvent"] = True
    if normalized.startswith("recommendation"):
        labels["hasPersonContext"] = True
    if normalized.startswith("calendar_query"):
        labels["hasEventContext"] = True
    if normalized in {"alarm_set", "timer_set", "timer_create", "alarm_create"}:
        labels["hasStartTimerIntent"] = True
    return labels


def _heads_for_stop_parse(parse: str) -> dict[str, bool]:
    labels = _empty_intent_labels()
    upper_parse = parse.upper()
    for head, tokens in STOP_PARSE_HEAD_TOKENS.items():
        if any(token in upper_parse for token in tokens):
            labels[head] = True
    return labels


def _empty_intent_labels() -> dict[str, bool]:
    return {head: False for head in INTENT_HEAD_ORDER}


def _extract_label(row: Mapping[str, Any], features: Mapping[str, Any]) -> str:
    for column in ("intent_str", "intent", "label", "scenario"):
        if column not in row:
            continue
        value = row[column]
        feature = features.get(column)
        if isinstance(value, (int, np.integer)) and hasattr(feature, "int2str"):
            return str(feature.int2str(int(value)))
        return str(value)
    raise ValueError(f"public corpus row has no supported intent column: {sorted(row)}")


def _extract_transcript(row: Mapping[str, Any]) -> str | None:
    for column in ("sentence", "transcript", "utt", "text", "transcription"):
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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


def _materialize_audio(root: Path, source_name: str, hf_row: _HfRow, audio: Mapping[str, Any]) -> Path:
    output_dir = root / "materialized_audio" / hf_row.config_name / hf_row.split
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
    if peak > 1.5:
        raise ValueError(f"decoded public-corpus audio must be normalized to [-1, 1], peak={peak}")
    if peak > 1.0:
        samples = np.clip(samples, -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, SAMPLE_RATE, subtype="PCM_16")


def _duration_seconds(array: Any, sampling_rate: int) -> float:
    if sampling_rate != SAMPLE_RATE:
        raise ValueError(f"decoded audio sampling_rate must be {SAMPLE_RATE}, got {sampling_rate}")
    samples = np.asarray(array)
    return float(samples.shape[0] / SAMPLE_RATE)


def _normalize_label(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "":
        raise ValueError("intent label must be non-empty")
    return normalized


def _require_public_intent_coverage(samples: Sequence[IntentSample]) -> None:
    if not samples:
        raise ValueError("intent index contains no samples")
    positives = {
        head: sum(1 for sample in samples if sample.head_labels.get(head, False))
        for head in PUBLIC_CORPUS_HEADS
    }
    missing = [head for head, count in positives.items() if count == 0]
    if missing:
        raise ValueError(f"public/recorded intent index missing required head coverage: {missing}")


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    if str(value).strip() == "":
        raise ValueError("path value must be non-empty")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _empty_to_none(value: str | None) -> str | None:
    if value is None or value.strip() == "":
        return None
    return value


def _parse_float(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    return float(value)


def _parse_bool(value: str | None, path: Path, row_number: int, column: str) -> bool:
    if value is None or value.strip() == "":
        return False
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    if column not in HEAD_ORDER:
        raise ValueError(f"unknown head column {column!r} in {path}")
    raise ValueError(f"invalid boolean {value!r} for {column} at {path}:{row_number}")
