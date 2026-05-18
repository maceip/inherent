"""Production build path for a hand-labeled recorded WAV library.

This module is the single orchestration layer for the real recorded-audio path:

1. validate labels
2. normalize WAVs to the runtime audio contract
3. split train/eval/test without speaker/session leakage
4. assemble strict raw 13-head manifests
5. materialize Cosmo frontend mel tensors
6. train
7. evaluate
8. export configured runtime artifacts + metadata/reports

Dummy WAVs can exercise this module, but the module itself does not special-case
dummy data.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import HEAD_ORDER
from ..config import Config
from ..data.intents import load_synthetic_manifest
from ..data.labeling import normalize_audio_manifest, split_label_manifest, validate_label_manifest
from ..data.labeling import validate_split_label_coverage
from ..data.manifest import RawAudioLabelSample, from_intent, write_raw_audio_manifest
from ..data.schema import ALLOWED_RAW_LABEL_COLUMNS
from ..eval.evaluate import evaluate_checkpoint, format_metrics
from ..export.registry import get_backend, list_backends
from ..features import materialize_mel_manifest
from ..training.train import train as train_model


@dataclass(frozen=True)
class RecordedBuildResult:
    model_group_dir: Path | None
    work_dir: Path
    run_dir: Path
    normalized_manifest: Path
    split_manifests: dict[str, Path]
    split_identity_index: Path
    warm_start_checkpoint: Path | None
    raw_manifests: dict[str, Path]
    mel_manifests: dict[str, Path]
    checkpoint: Path | None
    metrics_json: Path | None
    metrics_csv: Path | None
    export_dir: Path | None
    export_results: list[dict] | None
    model_group_json: Path | None


def build_recorded_library(
    *,
    cfg: Config,
    labels_manifest: str | Path,
    work_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
    frontend_model: str | Path,
    model_group_dir: str | Path | None = None,
    previous_model_group: str | Path | None = None,
    extra_train_manifests: Sequence[str | Path] = (),
    synthetic_train_manifests: Sequence[str | Path] = (),
    train: bool = True,
    evaluate: bool = True,
    export: bool = True,
    export_backends: Sequence[str] | None = None,
    training_device: str | None = None,
    eval_device: str = "cpu",
    max_steps: int | None = None,
    fail_on_validation_warnings: bool = True,
) -> RecordedBuildResult:
    """Build train/eval/export artifacts from a labeled recorded WAV manifest.

    The input manifest must contain `audio_path` and every head in HEAD_ORDER.
    If the manifest has non-empty `split` values, they are honored strictly.
    Otherwise speaker/session grouped splits are created. When
    `previous_model_group` is set, that group's saved eval/test audio identities
    are reused, every new clip goes to train, and training warm-starts from the
    previous checkpoint.
    """
    runtime_cfg = copy.deepcopy(cfg)
    labels_path = Path(labels_manifest).expanduser()
    group = Path(model_group_dir).expanduser() if model_group_dir is not None else None
    work, run = _resolve_build_dirs(work_dir=work_dir, run_dir=run_dir, model_group_dir=group)
    previous_group = Path(previous_model_group).expanduser() if previous_model_group is not None else None
    _validate_previous_model_group(previous_group)
    warm_start_checkpoint = _resolve_previous_checkpoint(previous_group) if train else None
    frontend = Path(frontend_model).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    run.mkdir(parents=True, exist_ok=True)

    _validate_label_report(
        labels_path,
        fail_on_validation_warnings=fail_on_validation_warnings,
        report_path=run / "label_report.json",
    )
    normalized_manifest = work / "recorded_normalized_manifest.csv"
    normalized_audio_dir = work / "recorded_audio_16k"
    normalize_audio_manifest(labels_path, normalized_manifest, normalized_audio_dir)

    split_dir = run / "splits" if group is not None else work / "splits"
    split_manifests = _split_or_preserve_explicit(
        normalized_manifest,
        split_dir,
        previous_model_group=previous_group,
    )
    split_label_coverage = validate_split_label_coverage(split_manifests)
    (run / "split_label_coverage.json").write_text(json.dumps(split_label_coverage, indent=2))
    split_identity_index = _write_split_identity_index(split_manifests, run / "split_identity_index.json")

    raw_dir = work / "raw"
    raw_manifests = _write_split_raw_manifests(
        split_manifests=split_manifests,
        raw_dir=raw_dir,
        extra_train_manifests=extra_train_manifests,
        synthetic_train_manifests=synthetic_train_manifests,
    )

    mel_dir = work / "mels"
    mel_manifests = _materialize_split_mels(raw_manifests, mel_dir=mel_dir, frontend_model=frontend)

    runtime_cfg.training.train_manifest = str(mel_manifests["train"])
    runtime_cfg.training.eval_manifest = str(mel_manifests["eval"])
    if training_device is not None:
        runtime_cfg.training.device = training_device
    if max_steps is not None:
        runtime_cfg.training.max_steps = max_steps
    runtime_cfg.training.__post_init__()
    _write_resolved_config(runtime_cfg, run / "resolved_config.json")

    checkpoint: Path | None = None
    metrics_json: Path | None = None
    metrics_csv: Path | None = None
    export_dir: Path | None = None
    export_results: list[dict] | None = None
    if train:
        train_model(
            runtime_cfg,
            run,
            init_checkpoint=warm_start_checkpoint,
        )
        checkpoint = _select_checkpoint(run)
    if evaluate:
        if checkpoint is None:
            checkpoint = _select_checkpoint(run)
        metrics = evaluate_checkpoint(
            checkpoint,
            mel_manifests["eval"],
            batch_size=runtime_cfg.training.batch_size,
            device=eval_device,
        )
        metrics_json = run / "eval_metrics.json"
        metrics_csv = run / "eval_metrics.csv"
        metrics_json.write_text(json.dumps(metrics, indent=2))
        metrics_csv.write_text(format_metrics(metrics) + "\n")
    if export:
        if checkpoint is None:
            checkpoint = _select_checkpoint(run)
        export_dir = run / "export"
        export_results = _export_backends(
            checkpoint=checkpoint,
            cfg=runtime_cfg,
            export_dir=export_dir,
            backend_names=export_backends,
        )

    model_group_json = None
    if group is not None:
        model_group_json = _write_model_group_json(
            group_dir=group,
            labels_manifest=labels_path,
            previous_model_group=previous_group,
            result_paths={
                "work_dir": work,
                "run_dir": run,
                "normalized_manifest": normalized_manifest,
                "split_identity_index": split_identity_index,
                "warm_start_checkpoint": warm_start_checkpoint,
                "split_manifests": split_manifests,
                "raw_manifests": raw_manifests,
                "mel_manifests": mel_manifests,
                "checkpoint": checkpoint,
                "metrics_json": metrics_json,
                "metrics_csv": metrics_csv,
                "export_dir": export_dir,
                "export_results": export_results,
            },
        )

    return RecordedBuildResult(
        model_group_dir=group,
        work_dir=work,
        run_dir=run,
        normalized_manifest=normalized_manifest,
        split_manifests=split_manifests,
        split_identity_index=split_identity_index,
        warm_start_checkpoint=warm_start_checkpoint,
        raw_manifests=raw_manifests,
        mel_manifests=mel_manifests,
        checkpoint=checkpoint,
        metrics_json=metrics_json,
        metrics_csv=metrics_csv,
        export_dir=export_dir,
        export_results=export_results,
        model_group_json=model_group_json,
    )


def _resolve_build_dirs(
    *,
    work_dir: str | Path | None,
    run_dir: str | Path | None,
    model_group_dir: Path | None,
) -> tuple[Path, Path]:
    if model_group_dir is not None:
        work = Path(work_dir).expanduser() if work_dir is not None else model_group_dir / "work"
        run = Path(run_dir).expanduser() if run_dir is not None else model_group_dir
        return work, run
    if work_dir is None or run_dir is None:
        raise ValueError("work_dir and run_dir are required unless model_group_dir is set")
    return Path(work_dir).expanduser(), Path(run_dir).expanduser()


def _validate_previous_model_group(previous_model_group: Path | None) -> None:
    if previous_model_group is None:
        return
    previous_head_order = _read_previous_head_order(previous_model_group)
    if tuple(previous_head_order) != HEAD_ORDER:
        raise ValueError(
            "previous model group head_order does not match current HEAD_ORDER; "
            "start a new model group without --previous-model-group"
        )


def _read_previous_head_order(previous_model_group: Path) -> tuple[str, ...]:
    manifest = previous_model_group / "model_group.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        return tuple(data.get("head_order", ()))
    index = previous_model_group / "split_identity_index.json"
    if index.is_file():
        data = json.loads(index.read_text())
        return tuple(data.get("head_order", ()))
    raise FileNotFoundError(f"previous model group missing model_group.json or split_identity_index.json: {previous_model_group}")


def _resolve_previous_checkpoint(previous_model_group: Path | None) -> Path | None:
    if previous_model_group is None:
        return None
    manifest = previous_model_group / "model_group.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        checkpoint = data.get("checkpoint")
        if checkpoint:
            path = Path(checkpoint).expanduser()
            if path.is_file():
                return path
    for name in ("best.pt", "last.pt"):
        path = previous_model_group / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"previous model group has no checkpoint to warm-start from: {previous_model_group}; "
        "start a new model group without --previous-model-group to train from scratch"
    )


def _export_backends(
    *,
    checkpoint: Path,
    cfg: Config,
    export_dir: Path,
    backend_names: Sequence[str] | None,
) -> list[dict]:
    selected = _selected_export_backends(cfg, backend_names)
    export_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for name in selected:
        backend = get_backend(name)
        output_dir = export_dir / name if len(selected) > 1 else export_dir
        result = backend.export(
            checkpoint_path=checkpoint,
            cfg=cfg,
            output_dir=output_dir,
            delegates=tuple(cfg.export.delegates),
        )
        results.append(
            {
                "backend": result.backend,
                "artifacts": result.artifacts,
                "metadata_path": result.metadata_path,
                "reports": result.reports,
                "supported": result.supported,
            }
        )
    (export_dir / "export_results.json").write_text(json.dumps(results, indent=2))
    return results


def _selected_export_backends(cfg: Config, backend_names: Sequence[str] | None) -> tuple[str, ...]:
    raw_names = tuple(backend_names or (cfg.export.backend,))
    expanded: list[str] = []
    for name in raw_names:
        if name == "all":
            expanded.extend(backend for backend in list_backends() if backend != "tflite")
        else:
            expanded.append(name)
    deduped = tuple(dict.fromkeys(expanded))
    if not deduped:
        raise ValueError("at least one export backend is required")
    return deduped


def _validate_label_report(
    labels_path: Path,
    *,
    fail_on_validation_warnings: bool,
    report_path: Path,
) -> None:
    report = validate_label_manifest(labels_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    blocking = [issue for issue in report["issues"] if issue["severity"] == "error"]
    if fail_on_validation_warnings:
        blocking.extend(issue for issue in report["issues"] if issue["severity"] == "warning")
    if blocking:
        first = blocking[0]
        raise ValueError(
            f"label manifest failed strict validation: "
            f"{first['severity']} row={first['row']} field={first['field']}: {first['message']}"
        )


def _split_or_preserve_explicit(
    normalized_manifest: Path,
    split_dir: Path,
    *,
    previous_model_group: Path | None,
) -> dict[str, Path]:
    if previous_model_group is not None:
        _write_splits_from_previous_group(normalized_manifest, split_dir, previous_model_group)
    elif _has_complete_explicit_split(normalized_manifest):
        _write_explicit_splits(normalized_manifest, split_dir)
    else:
        split_label_manifest(normalized_manifest, split_dir)
    return {
        "train": split_dir / "train_manifest.csv",
        "eval": split_dir / "eval_manifest.csv",
        "test": split_dir / "test_manifest.csv",
    }


def _has_complete_explicit_split(path: Path) -> bool:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "split" not in reader.fieldnames:
            return False
        rows = list(reader)
    if not rows:
        raise ValueError(f"manifest contains no rows: {path}")
    values = [row.get("split", "").strip() for row in rows]
    if all(value == "" for value in values):
        return False
    if any(value == "" for value in values):
        raise ValueError("recorded manifest split column must be either empty for every row or set for every row")
    invalid = sorted({value for value in values if value not in {"train", "eval", "test"}})
    if invalid:
        raise ValueError(f"invalid split values: {invalid}")
    return True


def _write_explicit_splits(input_manifest: Path, split_dir: Path) -> None:
    with input_manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {input_manifest}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    _require_no_group_leakage(rows)
    split_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    writers = {}
    counts = {"train": 0, "eval": 0, "test": 0}
    try:
        for split in counts:
            files[split] = (split_dir / f"{split}_manifest.csv").open("w", newline="")
            writers[split] = csv.DictWriter(files[split], fieldnames=fieldnames)
            writers[split].writeheader()
        for row in rows:
            split = row["split"].strip()
            writers[split].writerow(row)
            counts[split] += 1
    finally:
        for file in files.values():
            file.close()
    missing = [split for split, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"explicit recorded splits must include rows for every split, missing={missing}")


def _write_splits_from_previous_group(input_manifest: Path, split_dir: Path, previous_model_group: Path) -> None:
    heldout_by_id = _load_previous_heldout_ids(previous_model_group)
    if not heldout_by_id:
        raise ValueError(f"previous model group has no eval/test identities: {previous_model_group}")
    with input_manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {input_manifest}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"manifest contains no rows: {input_manifest}")

    fieldnames = _fieldnames_with_split(input_manifest)
    split_dir.mkdir(parents=True, exist_ok=True)
    row_identities: list[tuple[dict[str, str], str, str]] = []
    forced_by_group: dict[str, str] = {}
    matched_heldout: set[str] = set()
    for index, row in enumerate(rows, start=2):
        identity = _row_audio_identity(row, input_manifest.parent)
        group = _row_group_key(row, index)
        previous_split = heldout_by_id.get(identity)
        row_identities.append((row, identity, group))
        if previous_split is None:
            continue
        matched_heldout.add(identity)
        existing = forced_by_group.get(group)
        if existing is not None and existing != previous_split:
            raise ValueError(f"speaker/session group maps to both {existing!r} and {previous_split!r}: {group}")
        forced_by_group[group] = previous_split

    missing = sorted(set(heldout_by_id) - matched_heldout)
    if missing:
        raise ValueError(
            f"current label library is missing {len(missing)} eval/test clips from previous model group; "
            f"first missing identity={missing[0]}"
        )

    files = {}
    writers = {}
    counts = {"train": 0, "eval": 0, "test": 0}
    try:
        for split in counts:
            files[split] = (split_dir / f"{split}_manifest.csv").open("w", newline="")
            writers[split] = csv.DictWriter(files[split], fieldnames=fieldnames)
            writers[split].writeheader()
        for row, _, group in row_identities:
            split = forced_by_group.get(group, "train")
            output_row = dict(row)
            output_row["split"] = split
            writers[split].writerow({field: output_row.get(field, "") for field in fieldnames})
            counts[split] += 1
    finally:
        for file in files.values():
            file.close()

    missing_splits = [split for split, count in counts.items() if count == 0]
    if missing_splits:
        raise ValueError(f"previous-model-group split reuse produced empty splits: {missing_splits}")


def _load_previous_heldout_ids(previous_model_group: Path) -> dict[str, str]:
    index_path = previous_model_group / "split_identity_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"previous model group missing split identity index: {index_path}")
    data = json.loads(index_path.read_text())
    if tuple(data.get("head_order", ())) != HEAD_ORDER:
        raise ValueError("previous split identity index head_order does not match current HEAD_ORDER")
    heldout: dict[str, str] = {}
    for split in ("eval", "test"):
        for row in data.get("splits", {}).get(split, []):
            identity = row["identity"]
            existing = heldout.get(identity)
            if existing is not None and existing != split:
                raise ValueError(f"audio identity appears in both {existing!r} and {split!r}: {identity}")
            heldout[identity] = split
    return heldout


def _require_no_group_leakage(rows: list[dict[str, str]]) -> None:
    groups: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows, start=2):
        groups[_row_group_key(row, index)].add(row["split"].strip())
    leaking = {key: sorted(splits) for key, splits in groups.items() if len(splits) > 1}
    if leaking:
        key, splits = next(iter(leaking.items()))
        raise ValueError(f"speaker/session group crosses splits: {key} -> {splits}")


def _fieldnames_with_split(path: Path) -> list[str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {path}")
        fieldnames = list(reader.fieldnames)
    if "split" not in fieldnames:
        insert_at = 1
        for metadata in ("transcript", "speaker_id", "session_id", "device", "environment", "source", "duration_s"):
            if metadata in fieldnames:
                insert_at = max(insert_at, fieldnames.index(metadata) + 1)
        fieldnames.insert(insert_at, "split")
    return fieldnames


def _row_group_key(row: dict[str, str], row_number: int) -> str:
    speaker = row.get("speaker_id", "").strip()
    session = row.get("session_id", "").strip()
    if speaker or session:
        return f"speaker={speaker or '<missing>'}|session={session or '<missing>'}"
    return f"row={row_number}"


def _write_split_raw_manifests(
    *,
    split_manifests: dict[str, Path],
    raw_dir: Path,
    extra_train_manifests: Sequence[str | Path],
    synthetic_train_manifests: Sequence[str | Path],
) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": raw_dir / "train_manifest.csv",
        "eval": raw_dir / "eval_manifest.csv",
        "test": raw_dir / "test_manifest.csv",
    }
    for split, output in outputs.items():
        samples = _raw_samples_from_audio_manifest(split_manifests[split])
        if split == "train":
            for manifest in extra_train_manifests:
                samples.extend(_raw_samples_from_audio_manifest(Path(manifest).expanduser()))
            for manifest in synthetic_train_manifests:
                samples.extend(from_intent(sample) for sample in load_synthetic_manifest(Path(manifest).expanduser()))
        write_raw_audio_manifest(samples, output, validate=False)
    return outputs


def _write_split_identity_index(split_manifests: dict[str, Path], output_path: Path) -> Path:
    payload = {
        "version": 1,
        "head_order": list(HEAD_ORDER),
        "splits": {},
        "counts": {},
    }
    for split in ("train", "eval", "test"):
        manifest = split_manifests[split]
        rows: list[dict] = []
        with manifest.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"manifest has no header: {manifest}")
            for row_number, row in enumerate(reader, start=2):
                rows.append(
                    {
                        "identity": _row_audio_identity(row, manifest.parent),
                        "audio_path": row["audio_path"],
                        "transcript": row.get("transcript", ""),
                        "speaker_id": row.get("speaker_id", ""),
                        "session_id": row.get("session_id", ""),
                        "source": row.get("source", ""),
                        "labels": {head: row[head] for head in HEAD_ORDER},
                        "row_number": row_number,
                    }
                )
        payload["splits"][split] = rows
        payload["counts"][split] = len(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def _write_model_group_json(
    *,
    group_dir: Path,
    labels_manifest: Path,
    previous_model_group: Path | None,
    result_paths: dict,
) -> Path:
    def path_or_none(value) -> str | None:
        return None if value is None else str(value)

    payload = {
        "version": 1,
        "mode": "iterate" if previous_model_group is not None else "new",
        "head_order": list(HEAD_ORDER),
        "labels_manifest": str(labels_manifest),
        "previous_model_group": path_or_none(previous_model_group),
        "work_dir": str(result_paths["work_dir"]),
        "run_dir": str(result_paths["run_dir"]),
        "normalized_manifest": str(result_paths["normalized_manifest"]),
        "split_identity_index": str(result_paths["split_identity_index"]),
        "warm_start_checkpoint": path_or_none(result_paths["warm_start_checkpoint"]),
        "split_manifests": {key: str(value) for key, value in result_paths["split_manifests"].items()},
        "raw_manifests": {key: str(value) for key, value in result_paths["raw_manifests"].items()},
        "mel_manifests": {key: str(value) for key, value in result_paths["mel_manifests"].items()},
        "checkpoint": path_or_none(result_paths["checkpoint"]),
        "metrics_json": path_or_none(result_paths["metrics_json"]),
        "metrics_csv": path_or_none(result_paths["metrics_csv"]),
        "export_dir": path_or_none(result_paths["export_dir"]),
        "export_results": result_paths["export_results"],
    }
    output = group_dir / "model_group.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return output


def _raw_samples_from_audio_manifest(path: Path) -> list[RawAudioLabelSample]:
    rows: list[RawAudioLabelSample] = []
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
        for row_number, row in enumerate(reader, start=2):
            audio_path = _resolve_path(row["audio_path"], path.parent)
            if not audio_path.is_file():
                raise FileNotFoundError(f"audio path does not exist at {path}:{row_number}: {audio_path}")
            labels = tuple(_parse_label(row[head], path, row_number, head) for head in HEAD_ORDER)
            rows.append(
                RawAudioLabelSample(
                    audio_path=audio_path,
                    labels=labels,
                    source=row.get("source", "").strip() or path.name,
                )
            )
    if not rows:
        raise ValueError(f"manifest contains no rows: {path}")
    return rows


def _materialize_split_mels(
    raw_manifests: dict[str, Path],
    *,
    mel_dir: Path,
    frontend_model: Path,
) -> dict[str, Path]:
    outputs = {
        "train": mel_dir / "train_manifest.csv",
        "eval": mel_dir / "eval_manifest.csv",
        "test": mel_dir / "test_manifest.csv",
    }
    for split, raw_manifest in raw_manifests.items():
        materialize_mel_manifest(
            input_manifest=raw_manifest,
            output_manifest=outputs[split],
            mel_dir=mel_dir / split,
            frontend_model=frontend_model,
        )
    return outputs


def _select_checkpoint(run_dir: Path) -> Path:
    best = run_dir / "best.pt"
    if best.is_file():
        return best
    last = run_dir / "last.pt"
    if last.is_file():
        return last
    raise FileNotFoundError(f"no checkpoint found in {run_dir}; expected best.pt or last.pt")


def _write_resolved_config(cfg: Config, path: Path) -> None:
    payload = {
        "model": asdict(cfg.model),
        "data": asdict(cfg.data),
        "training": asdict(cfg.training),
        "export": asdict(cfg.export),
        "eval": asdict(cfg.eval),
    }
    path.write_text(json.dumps(payload, indent=2))


def _resolve_path(value: str, base_dir: Path) -> Path:
    if value.strip() == "":
        raise ValueError("audio_path must be non-empty")
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _row_audio_identity(row: dict[str, str], base_dir: Path) -> str:
    audio_path = _resolve_path(row["audio_path"], base_dir)
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio path does not exist for identity index: {audio_path}")
    h = hashlib.sha256()
    with audio_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_label(value: str, path: Path, row_number: int, column: str) -> str:
    normalized = value.strip()
    if normalized in {"0", "1"}:
        return normalized
    raise ValueError(f"invalid label {value!r} for {column} at {path}:{row_number}; expected 0 or 1")
