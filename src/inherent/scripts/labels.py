"""CLI for recorded-audio labeling, validation, normalization, and splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inherent.data.labeling import (
    normalize_audio_manifest,
    report_to_text,
    split_label_manifest,
    validate_label_manifest,
    write_label_template,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser("template")
    template.add_argument("--output", type=Path, default=Path("data/labels_template.csv"))

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--json-out", type=Path)

    report = subparsers.add_parser("report")
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--json-out", type=Path)

    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("--manifest", type=Path, required=True)
    normalize.add_argument("--output-manifest", type=Path, required=True)
    normalize.add_argument("--audio-dir", type=Path, required=True)

    split = subparsers.add_parser("split")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--train-ratio", type=float, default=0.8)
    split.add_argument("--eval-ratio", type=float, default=0.1)
    split.add_argument("--test-ratio", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=1337)

    args = parser.parse_args()
    if args.command == "template":
        output = write_label_template(args.output)
        print(f"wrote {output}")
    elif args.command == "validate":
        validation = validate_label_manifest(args.manifest)
        print(report_to_text(validation))
        if args.json_out is not None:
            write_report(validation, args.json_out)
        if not validation["ok"]:
            raise SystemExit(1)
    elif args.command == "report":
        validation = validate_label_manifest(args.manifest)
        print(report_to_text(validation))
        if args.json_out is not None:
            write_report(validation, args.json_out)
    elif args.command == "normalize":
        count = normalize_audio_manifest(args.manifest, args.output_manifest, args.audio_dir)
        print(f"normalized {count} audio rows")
    elif args.command == "split":
        counts = split_label_manifest(
            args.manifest,
            args.output_dir,
            train_ratio=args.train_ratio,
            eval_ratio=args.eval_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
