"""Data preparation CLI.

This dispatches to the corpus assembly modules. The per-corpus public dataset
loaders are intentionally still explicit implementation points, but recorded
intent manifests can already be indexed through `data.intents`.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path

from inherent.config import Config
from inherent.data import combine_indexes, directedness, intents, synthesis, write_raw_audio_manifest
from inherent.features import materialize_mel_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--target",
        choices=["preflight", "directedness", "intents", "synthesis", "raw-manifest", "mels"],
        required=True,
    )
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--mel-dir", type=Path)
    parser.add_argument("--frontend-model", type=Path)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.target == "preflight":
        _preflight(cfg, args.data_root, args.frontend_model)
    elif args.target == "mels":
        required = {
            "--input-manifest": args.input_manifest,
            "--output-manifest": args.output_manifest,
            "--mel-dir": args.mel_dir,
            "--frontend-model": args.frontend_model,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SystemExit(f"--target mels requires: {', '.join(missing)}")
        count = materialize_mel_manifest(
            input_manifest=args.input_manifest,
            output_manifest=args.output_manifest,
            mel_dir=args.mel_dir,
            frontend_model=args.frontend_model,
        )
        print(f"materialized {count} mel samples")
    elif args.target == "raw-manifest":
        if args.output_manifest is None:
            raise SystemExit("--target raw-manifest requires: --output-manifest")
        directedness_samples = directedness.build_index(cfg.data.directedness, args.data_root)
        intent_samples = intents.build_index(cfg.data.intents, args.data_root)
        count = write_raw_audio_manifest(
            combine_indexes(directedness_samples, intent_samples, validate=False),
            args.output_manifest,
            validate=False,
        )
        print(f"wrote {count} raw labeled audio samples")
    elif args.target == "synthesis":
        if args.output_manifest is None:
            raise SystemExit("--target synthesis requires: --output-manifest")
        count = _build_synthetic_manifest(cfg, args.data_root, args.output_manifest)
        if count == 0:
            raise SystemExit("no synthetic samples were generated")
        print(f"synthesized {count} TTS samples")
    elif args.target == "directedness":
        samples = directedness.build_index(cfg.data.directedness, args.data_root)
        print(f"indexed {len(samples)} {args.target} samples")
    else:
        samples = intents.build_index(cfg.data.intents, args.data_root)
        print(f"indexed {len(samples)} {args.target} samples")


def _synthetic_head_from_key(key: str) -> str:
    mapping = {
        "photo_query": "hasPhotoQuery",
        "create_doc": "hasCreateDocIntent",
        "deep_research": "hasDeepResearchIntent",
        "insight": "hasInsightIntent",
        "browsing_agent": "hasBrowsingAgentIntent",
        "calling_agent": "hasCallingAgentIntent",
    }
    if key not in mapping:
        raise ValueError(f"unknown synthetic intent key {key!r}")
    return mapping[key]


def _write_synthetic_row(writer: csv.DictWriter, sample: synthesis.SyntheticSample) -> None:
    writer.writerow(
        {
            "audio_path": str(sample.audio_path.resolve()),
            "transcript": sample.transcript,
            "head": sample.head,
            "voice_id": sample.voice_id,
            "tts_engine": sample.tts_engine,
        }
    )


def _build_synthetic_manifest(cfg: Config, data_root: Path, output_manifest: Path) -> int:
    partial_manifest = output_manifest.with_suffix(output_manifest.suffix + ".partial")
    partial_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_dir = data_root / "synthetic_audio"
    resume_paths = (output_manifest,) if output_manifest.expanduser().is_file() else (partial_manifest,)
    existing_rows = _read_existing_synthetic_rows(resume_paths)
    existing_keys = {_synthetic_row_key(row) for row in existing_rows}
    total_tasks = _count_synthetic_tasks(cfg)
    remaining_tasks = sum(
        1
        for head, prompt, voice_id in _iter_balanced_synthetic_tasks(cfg)
        if (head, voice_id, prompt) not in existing_keys
    )
    if remaining_tasks <= 0:
        _write_synthetic_rows(output_manifest, existing_rows)
        return len(existing_rows)

    runtime = synthesis._OpenF5Runtime(synthesis._openf5_model_files())
    count = len(existing_rows)
    print(
        f"resuming synthetic manifest with {len(existing_rows)} existing rows; "
        f"{remaining_tasks} remaining",
        flush=True,
    )
    with partial_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["audio_path", "transcript", "head", "voice_id", "tts_engine"],
        )
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        f.flush()

        for head, prompt, voice_id in _iter_balanced_synthetic_tasks(cfg):
            key = (head, voice_id, prompt)
            if key in existing_keys:
                continue
            sample = next(
                synthesis.iter_synthesize(
                    [prompt],
                    head,
                    output_dir,
                    voices=(voice_id,),
                    runtime=runtime,
                )
            )
            _write_synthetic_row(writer, sample)
            existing_keys.add(key)
            count += 1
            if count == 1 or count % 100 == 0:
                f.flush()
                print(f"synthetic manifest rows: {count}/{total_tasks}", flush=True)
        f.flush()
    partial_manifest.replace(output_manifest)
    return count


def _iter_balanced_synthetic_tasks(cfg: Config) -> Iterable[tuple[str, str, str]]:
    plans: list[tuple[str, list[str], tuple[str, ...]]] = []
    for head_key, head_cfg in cfg.data.intents.get("synthetic", {}).items():
        head = _synthetic_head_from_key(head_key)
        prompt_count = int(head_cfg["count"])
        voices = tuple(str(voice) for voice in head_cfg["voices"])
        prompts = synthesis.expand_prompts(head, prompt_count)
        plans.append((head, prompts, voices))
    if not plans:
        return
    max_prompts = max(len(prompts) for _, prompts, _ in plans)
    for prompt_index in range(max_prompts):
        for head, prompts, voices in plans:
            if prompt_index >= len(prompts):
                continue
            prompt = prompts[prompt_index]
            for voice_id in voices:
                yield head, prompt, voice_id


def _count_synthetic_tasks(cfg: Config) -> int:
    total = 0
    for head_cfg in cfg.data.intents.get("synthetic", {}).values():
        total += int(head_cfg["count"]) * len(tuple(head_cfg["voices"]))
    return total


def _read_existing_synthetic_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        path = path.expanduser()
        if not path.is_file():
            continue
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"synthetic manifest has no header: {path}")
            expected = ["audio_path", "transcript", "head", "voice_id", "tts_engine"]
            if reader.fieldnames != expected:
                raise ValueError(f"synthetic manifest {path} header must be {expected}, got {reader.fieldnames}")
            for row_number, row in enumerate(reader, start=2):
                _validate_existing_synthetic_row(path, row_number, row)
                key = _synthetic_row_key(row)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({column: row[column] for column in expected})
    return rows


def _validate_existing_synthetic_row(path: Path, row_number: int, row: dict[str, str]) -> None:
    audio_path = Path(row["audio_path"]).expanduser()
    if not audio_path.is_file():
        raise FileNotFoundError(f"synthetic row audio does not exist at {path}:{row_number}: {audio_path}")
    if row["head"] not in synthesis.SYNTHETIC_HEADS:
        raise ValueError(f"invalid synthetic head {row['head']!r} at {path}:{row_number}")
    if row["voice_id"].strip() == "":
        raise ValueError(f"empty synthetic voice_id at {path}:{row_number}")
    if row["transcript"].strip() == "":
        raise ValueError(f"empty synthetic transcript at {path}:{row_number}")
    if row["tts_engine"] != synthesis.OPENF5_TTS_ENGINE:
        raise ValueError(f"unexpected synthetic tts_engine {row['tts_engine']!r} at {path}:{row_number}")


def _synthetic_row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["head"], row["voice_id"], row["transcript"])


def _write_synthetic_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["audio_path", "transcript", "head", "voice_id", "tts_engine"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _preflight(cfg: Config, data_root: Path, frontend_model: Path | None) -> None:
    blockers: list[str] = []
    data_root = data_root.expanduser()

    for manifest in cfg.data.directedness.positive_manifests:
        _require_file(blockers, data_root / manifest, "directedness positive manifest")
    for manifest in cfg.data.directedness.negative_manifests:
        _require_file(blockers, data_root / manifest, "directedness negative manifest")
    for manifest in cfg.data.directedness.labeled_manifests:
        _require_file(blockers, data_root / manifest, "directedness labeled manifest")
    for manifest in cfg.data.intents.get("recorded", []):
        _require_file(blockers, data_root / manifest, "recorded intent manifest")
    for manifest in cfg.data.intents.get("synthetic_manifests", []):
        _require_file(blockers, data_root / manifest, "synthetic intent manifest")

    public_sources = set(cfg.data.intents.get("public_sources", []))
    directed_sources = set(cfg.data.directedness.positives + cfg.data.directedness.negatives)
    if "stop" in public_sources | directed_sources:
        _require_dir(blockers, data_root / "stop", "STOP corpus root")
    for local_negative in ("ami", "icsi", "callhome", "magicdata_ramc", "fsd50k", "esc10", "esc50", "urbansound8k", "freesound_ambient"):
        if local_negative in directed_sources:
            _require_wavs(blockers, data_root / local_negative, f"{local_negative} negative corpus")

    if cfg.data.intents.get("synthetic"):
        if synthesis._openf5_command() is None:
            blockers.append("missing f5-tts_infer-cli for OpenF5 --target synthesis")
        try:
            synthesis._openf5_model_reference()
        except (RuntimeError, ValueError) as exc:
            blockers.append(str(exc))
        for head_cfg in cfg.data.intents["synthetic"].values():
            for voice in head_cfg["voices"]:
                _require_file(blockers, data_root / "tts_voices" / voice / "ref.wav", f"TTS {voice} ref.wav")
                _require_file(blockers, data_root / "tts_voices" / voice / "ref.txt", f"TTS {voice} ref.txt")

    if frontend_model is not None:
        _require_file(blockers, frontend_model.expanduser(), "audio frontend model")

    if blockers:
        raise SystemExit("preflight blockers:\n- " + "\n- ".join(blockers))
    print("preflight ok")


def _require_file(blockers: list[str], path: Path, label: str) -> None:
    if not path.expanduser().is_file():
        blockers.append(f"{label} not found: {path}")


def _require_dir(blockers: list[str], path: Path, label: str) -> None:
    if not path.expanduser().is_dir():
        blockers.append(f"{label} not found: {path}")


def _require_wavs(blockers: list[str], path: Path, label: str) -> None:
    root = path.expanduser()
    if not root.is_dir():
        blockers.append(f"{label} not found: {path}")
    elif not any(root.rglob("*.wav")):
        blockers.append(f"{label} contains no .wav files: {path}")


if __name__ == "__main__":
    main()
