"""Trim a partial synthetic manifest to a first-usable slice per head."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from inherent.data.intents import load_synthetic_manifest
from inherent.data.synthesis import SyntheticSample, write_synthetic_manifest
from inherent.scripts.prep_data import _synthetic_head_from_key


def trim_synthetic_manifest(
    input_manifest: Path,
    output_manifest: Path,
    *,
    max_per_head: int | dict[str, int],
) -> int:
    """Keep the first N rows per head in file order."""
    per_head_limits = max_per_head if isinstance(max_per_head, dict) else None
    rows_by_head: dict[str, list[dict[str, str]]] = defaultdict(list)
    with input_manifest.expanduser().open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {input_manifest}")
        for row in reader:
            head = row["head"].strip()
            if per_head_limits is not None and head not in per_head_limits:
                continue
            cap = per_head_limits[head] if per_head_limits is not None else int(max_per_head)
            if len(rows_by_head[head]) < cap:
                rows_by_head[head].append(row)

    if per_head_limits is not None:
        missing = [head for head in per_head_limits if not rows_by_head[head]]
        if missing:
            raise ValueError(f"no rows for heads: {missing}")

    kept: list[dict[str, str]] = []
    for head in sorted(rows_by_head):
        kept.extend(rows_by_head[head])

    samples = [
        SyntheticSample(
            audio_path=Path(row["audio_path"]),
            transcript=row["transcript"],
            head=row["head"],
            voice_id=row["voice_id"],
            tts_engine=row.get("tts_engine", "openf5-tts"),
        )
        for row in kept
    ]
    write_synthetic_manifest(samples, output_manifest)
    return len(samples)


def limits_from_config(config_path: Path) -> dict[str, int]:
    from inherent.config import Config

    cfg = Config.load(config_path)
    limits: dict[str, int] = {}
    for head_key, head_cfg in cfg.data.intents.get("synthetic", {}).items():
        if not isinstance(head_cfg, dict) or "count" not in head_cfg:
            continue
        head = _synthetic_head_from_key(head_key)
        limits[head] = int(head_cfg["count"]) * len(tuple(head_cfg["voices"]))
    return limits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-head", type=int)
    parser.add_argument("--config", type=Path, help="Use count×voices per head from synthetic config")
    args = parser.parse_args()

    if args.config is not None:
        cap = limits_from_config(args.config)
        if not cap:
            raise SystemExit(f"no synthetic heads in {args.config}")
    elif args.max_per_head is not None:
        cap = args.max_per_head
    else:
        raise SystemExit("provide --max-per-head or --config")

    count = trim_synthetic_manifest(args.input, args.output, max_per_head=cap)
    load_synthetic_manifest(args.output)
    print(f"wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
