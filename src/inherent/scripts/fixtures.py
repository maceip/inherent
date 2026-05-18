"""Create an Inherent label manifest from genesis gatekeeper audio fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inherent.data.fixture_manifest import write_gatekeeper_fixture_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--no-existing", action="store_true")
    parser.add_argument("--macos-voice", action="append", default=[])
    parser.add_argument("--generated-audio-dir", type=Path)
    parser.add_argument("--split", choices=("train", "eval", "test", ""), default="train")
    parser.add_argument("--overwrite-generated", action="store_true")
    args = parser.parse_args()

    result = write_gatekeeper_fixture_manifest(
        index_path=args.index,
        output_manifest=args.output_manifest,
        include_existing=not args.no_existing,
        macos_voices=tuple(args.macos_voice),
        generated_audio_dir=args.generated_audio_dir,
        split=args.split,
        overwrite_generated=args.overwrite_generated,
    )
    print(
        json.dumps(
            {
                "output_manifest": str(args.output_manifest),
                "rows_written": result.rows_written,
                "generated_audio": [str(path) for path in result.generated_audio],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
