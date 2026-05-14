"""Labeling, validation, normalization, and split helpers for recorded audio."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .. import HEAD_ORDER, INTENT_HEAD_ORDER, INTERESTING_HEAD
from ..features.frontend import MAX_SECONDS, SAMPLE_RATE
from .schema import ALLOWED_RAW_LABEL_COLUMNS, LABEL_TEMPLATE_COLUMNS


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    row: int
    field: str
    message: str


def write_label_template(path: str | Path) -> Path:
    """Write an empty CSV template for human labeling."""
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_TEMPLATE_COLUMNS)
        writer.writeheader()
    return output


def validate_label_manifest(path: str | Path) -> dict:
    """Validate a recorded-audio label manifest and return a JSON-ready report."""
    manifest_path = Path(path).expanduser()
    rows = _read_rows(manifest_path)
    issues: list[ValidationIssue] = []
    stats = _empty_stats()
    hashes: dict[str, list[int]] = defaultdict(list)

    for row_number, row in rows:
        labels = _parse_labels(row, row_number, issues)
        audio_path = _resolve_audio(row.get("audio_path", ""), manifest_path.parent)
        _update_label_stats(stats, labels)
        _update_metadata_stats(stats, row)
        if not audio_path.is_file():
            issues.append(ValidationIssue("error", row_number, "audio_path", f"missing file: {audio_path}"))
            continue
        audio_info = inspect_audio_file(audio_path)
        _update_audio_stats(stats, audio_info)
        if audio_info["duration_s"] <= 0:
            issues.append(ValidationIssue("error", row_number, "audio_path", "audio has zero duration"))
        if audio_info["duration_s"] > MAX_SECONDS:
            issues.append(
                ValidationIssue("error", row_number, "duration_s", f"audio exceeds {MAX_SECONDS}s runtime limit")
            )
        if audio_info["rms"] < 1e-4:
            issues.append(ValidationIssue("warning", row_number, "audio_path", "near-silent audio"))
        if audio_info["clipped_fraction"] > 0.001:
            issues.append(ValidationIssue("warning", row_number, "audio_path", "possible clipped audio"))
        digest = _sha256(audio_path)
        hashes[digest].append(row_number)
        _check_label_consistency(row_number, labels, issues)

    duplicates = {digest: row_numbers for digest, row_numbers in hashes.items() if len(row_numbers) > 1}
    for row_numbers in duplicates.values():
        issues.append(
            ValidationIssue("warning", row_numbers[0], "audio_path", f"duplicate audio rows: {row_numbers}")
        )

    report = {
        "manifest": str(manifest_path),
        "rows": len(rows),
        "ok": not any(issue.severity == "error" for issue in issues),
        "issues": [issue.__dict__ for issue in issues],
        "stats": stats,
        "duplicates": duplicates,
    }
    return report


def normalize_audio_manifest(
    input_manifest: str | Path,
    output_manifest: str | Path,
    output_audio_dir: str | Path,
) -> int:
    """Convert manifest audio to 16 kHz mono WAV and preserve labels/metadata."""
    input_path = Path(input_manifest).expanduser()
    output_path = Path(output_manifest).expanduser()
    audio_dir = Path(output_audio_dir).expanduser()
    rows = _read_rows(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_output_fieldnames(input_path))
        writer.writeheader()
        for index, (row_number, row) in enumerate(rows):
            source = _resolve_audio(row["audio_path"], input_path.parent)
            audio = load_audio_any(source)
            output_audio = audio_dir / f"{index:08d}_{source.stem}.wav"
            write_wav_16k_mono(output_audio, audio)
            row = dict(row)
            row["audio_path"] = str(output_audio.resolve())
            if "duration_s" in writer.fieldnames:
                row["duration_s"] = f"{len(audio) / SAMPLE_RATE:.3f}"
            writer.writerow({field: row.get(field, "") for field in writer.fieldnames})
            if row_number < 2:
                raise AssertionError("row numbers are expected to be CSV line numbers")
    return len(rows)


def split_label_manifest(
    input_manifest: str | Path,
    output_dir: str | Path,
    *,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 1337,
) -> dict[str, int]:
    """Create train/eval/test manifests without splitting speaker/session groups."""
    if round(train_ratio + eval_ratio + test_ratio, 6) != 1.0:
        raise ValueError("train/eval/test ratios must sum to 1.0")
    input_path = Path(input_manifest).expanduser()
    split_dir = Path(output_dir).expanduser()
    rows = _read_rows(input_path)
    fieldnames = _output_fieldnames(input_path, include_split=True)
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for row_number, row in rows:
        grouped[_group_key(row, row_number)].append((row_number, row))

    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    assignments = _assign_groups(keys, train_ratio, eval_ratio)
    split_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, csv.DictWriter] = {}
    files = {}
    counts = {"train": 0, "eval": 0, "test": 0}
    try:
        for split in counts:
            files[split] = (split_dir / f"{split}_manifest.csv").open("w", newline="")
            writers[split] = csv.DictWriter(files[split], fieldnames=fieldnames)
            writers[split].writeheader()
        for group_key, split in assignments.items():
            for _, row in grouped[group_key]:
                output_row = dict(row)
                output_row["split"] = split
                writers[split].writerow({field: output_row.get(field, "") for field in fieldnames})
                counts[split] += 1
    finally:
        for f in files.values():
            f.close()
    return counts


def inspect_audio_file(path: str | Path) -> dict:
    audio = load_audio_any(path)
    return {
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "duration_s": len(audio) / SAMPLE_RATE,
        "rms": float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0,
        "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
        "clipped_fraction": float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 0.0,
    }


def load_audio_any(path: str | Path) -> np.ndarray:
    """Load audio, convert to mono 16 kHz float32 in [-1, 1]."""
    audio_path = Path(path).expanduser()
    try:
        import soundfile as sf
        from scipy.signal import resample_poly

        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            audio = resample_poly(audio, SAMPLE_RATE, sample_rate).astype(np.float32)
    except Exception:
        import librosa

        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"audio must load as mono 1-D samples: {audio_path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"audio contains non-finite samples: {audio_path}")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio


def write_wav_16k_mono(path: str | Path, audio: np.ndarray) -> Path:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required to write normalized WAV files") from exc

    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    sf.write(output, audio, SAMPLE_RATE, subtype="PCM_16")
    return output


def report_to_text(report: dict) -> str:
    lines = [
        f"manifest: {report['manifest']}",
        f"rows: {report['rows']}",
        f"ok: {report['ok']}",
        "head,positive,negative",
    ]
    counts = report["stats"]["heads"]
    for head in HEAD_ORDER:
        lines.append(f"{head},{counts[head]['positive']},{counts[head]['negative']}")
    if report["issues"]:
        lines.append("issues:")
        for issue in report["issues"]:
            lines.append(f"{issue['severity']} row={issue['row']} field={issue['field']}: {issue['message']}")
    return "\n".join(lines)


def write_report(report: dict, path: str | Path) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return output


def _read_rows(path: Path) -> list[tuple[int, dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {path}")
        missing = [column for column in ("audio_path", *HEAD_ORDER) if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"manifest {path} missing required columns: {missing}")
        unexpected = [column for column in reader.fieldnames if column not in ALLOWED_RAW_LABEL_COLUMNS]
        if unexpected:
            raise ValueError(f"manifest {path} has unexpected columns: {unexpected}")
        rows = [(row_number, row) for row_number, row in enumerate(reader, start=2)]
    if not rows:
        raise ValueError(f"manifest contains no rows: {path}")
    return rows


def _parse_labels(row: dict[str, str], row_number: int, issues: list[ValidationIssue]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for head in HEAD_ORDER:
        value = row.get(head, "").strip()
        if value not in {"0", "1"}:
            issues.append(ValidationIssue("error", row_number, head, f"expected 0 or 1, got {value!r}"))
            labels[head] = 0
        else:
            labels[head] = int(value)
    return labels


def _check_label_consistency(row_number: int, labels: dict[str, int], issues: list[ValidationIssue]) -> None:
    intent_positive = any(labels[head] == 1 for head in INTENT_HEAD_ORDER)
    if intent_positive and labels[INTERESTING_HEAD] != 1:
        issues.append(ValidationIssue("error", row_number, INTERESTING_HEAD, "intent positives require isInteresting=1"))
    if not intent_positive and labels[INTERESTING_HEAD] == 1:
        issues.append(ValidationIssue("warning", row_number, INTERESTING_HEAD, "directed positive has no intent head"))


def _empty_stats() -> dict:
    return {
        "heads": {head: {"positive": 0, "negative": 0} for head in HEAD_ORDER},
        "metadata": {column: Counter() for column in ("speaker_id", "session_id", "device", "environment", "source", "split")},
        "audio": {
            "duration_s_total": 0.0,
            "duration_s_min": None,
            "duration_s_max": None,
            "sample_rates": Counter(),
        },
    }


def _update_label_stats(stats: dict, labels: dict[str, int]) -> None:
    for head, value in labels.items():
        stats["heads"][head]["positive" if value else "negative"] += 1


def _update_metadata_stats(stats: dict, row: dict[str, str]) -> None:
    for column, counter in stats["metadata"].items():
        value = row.get(column, "").strip() or "<missing>"
        counter[value] += 1


def _update_audio_stats(stats: dict, audio_info: dict) -> None:
    audio_stats = stats["audio"]
    duration = float(audio_info["duration_s"])
    audio_stats["duration_s_total"] += duration
    audio_stats["duration_s_min"] = duration if audio_stats["duration_s_min"] is None else min(audio_stats["duration_s_min"], duration)
    audio_stats["duration_s_max"] = duration if audio_stats["duration_s_max"] is None else max(audio_stats["duration_s_max"], duration)
    audio_stats["sample_rates"][str(audio_info["sample_rate"])] += 1


def _resolve_audio(value: str, base_dir: Path) -> Path:
    if value.strip() == "":
        return base_dir / "<missing>"
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _output_fieldnames(path: Path, *, include_split: bool = False) -> list[str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {path}")
        fieldnames = list(reader.fieldnames)
    if include_split and "split" not in fieldnames:
        insert_at = 1
        for metadata in ("transcript", "speaker_id", "session_id", "device", "environment", "source", "duration_s"):
            if metadata in fieldnames:
                insert_at = max(insert_at, fieldnames.index(metadata) + 1)
        fieldnames.insert(insert_at, "split")
    return fieldnames


def _group_key(row: dict[str, str], row_number: int) -> str:
    speaker = row.get("speaker_id", "").strip()
    session = row.get("session_id", "").strip()
    if speaker or session:
        return f"speaker={speaker or '<missing>'}|session={session or '<missing>'}"
    return f"row={row_number}"


def _assign_groups(keys: list[str], train_ratio: float, eval_ratio: float) -> dict[str, str]:
    total = len(keys)
    train_cut = int(round(total * train_ratio))
    eval_cut = train_cut + int(round(total * eval_ratio))
    assignments = {}
    for index, key in enumerate(keys):
        if index < train_cut:
            split = "train"
        elif index < eval_cut:
            split = "eval"
        else:
            split = "test"
        assignments[key] = split
    if total >= 3:
        for split in ("train", "eval", "test"):
            if split not in assignments.values():
                donor_key = keys[-1]
                assignments[donor_key] = split
    return assignments


def copy_manifest_audio(input_manifest: str | Path, output_dir: str | Path) -> int:
    """Copy referenced audio files next to a manifest for handoff/debug bundles."""
    input_path = Path(input_manifest).expanduser()
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for _, row in _read_rows(input_path):
        source = _resolve_audio(row["audio_path"], input_path.parent)
        shutil.copy2(source, output / source.name)
        count += 1
    return count
