"""Build labeled training manifests from gatekeeper audio fixture indexes."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Sequence

from .. import HEAD_ORDER
from .schema import LABEL_TEMPLATE_COLUMNS


@dataclass(frozen=True)
class FixtureManifestResult:
    rows_written: int
    generated_audio: tuple[Path, ...]


def write_gatekeeper_fixture_manifest(
    *,
    index_path: str | Path,
    output_manifest: str | Path,
    include_existing: bool = True,
    macos_voices: Sequence[str] = (),
    generated_audio_dir: str | Path | None = None,
    split: str = "train",
    overwrite_generated: bool = False,
) -> FixtureManifestResult:
    """Write a 13-head label manifest from a genesis gatekeeper fixture index.

    The fixture JSON already carries the expected audio gatekeeper head index.
    This function turns those labels into the same CSV contract used by the
    recorded-audio training pipeline. Positive intent rows also set
    ``isInteresting=1``; rows with ``expected_audio_gatekeeper_top_index=null``
    are negative controls.
    """
    index = Path(index_path).expanduser()
    output = Path(output_manifest).expanduser()
    fixture_data = json.loads(index.read_text())
    fixtures = fixture_data.get("audio_fixtures")
    if not isinstance(fixtures, list):
        raise ValueError(f"{index} missing audio_fixtures list")
    if split not in {"train", "eval", "test", ""}:
        raise ValueError("split must be one of: train, eval, test, or empty string")

    rows: list[dict[str, str]] = []
    generated: list[Path] = []
    for fixture in fixtures:
        fixture_file = fixture.get("file")
        if not isinstance(fixture_file, str) or not fixture_file:
            raise ValueError(f"fixture missing non-empty file: {fixture!r}")
        phrase = str(fixture.get("phrase", "")).strip()
        labels = _labels_for_fixture(fixture, index)
        stem = Path(fixture_file).stem

        if include_existing:
            audio_path = (index.parent / fixture_file).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"fixture audio not found: {audio_path}")
            rows.append(_manifest_row(audio_path, phrase, stem, "existing", split, labels))

        for voice in macos_voices:
            if not _is_speakable_phrase(phrase):
                continue
            audio_dir = Path(generated_audio_dir).expanduser() if generated_audio_dir else output.parent / "generated_audio"
            audio_path = (audio_dir / _safe_name(voice) / f"{stem}.wav").resolve()
            synthesize_macos_voice(phrase, voice, audio_path, overwrite=overwrite_generated)
            generated.append(audio_path)
            rows.append(_manifest_row(audio_path, phrase, stem, f"macos_say:{voice}", split, labels))

    if not rows:
        raise ValueError("fixture index produced no manifest rows")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return FixtureManifestResult(rows_written=len(rows), generated_audio=tuple(generated))


def synthesize_macos_voice(
    phrase: str,
    voice: str,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Render one phrase through macOS ``say`` and normalize to 16 kHz mono WAV."""
    output = Path(output_path).expanduser()
    if output.exists() and not overwrite:
        return output
    if shutil.which("say") is None or shutil.which("afconvert") is None:
        raise RuntimeError("macOS say and afconvert are required for --macos-voice synthesis")

    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".aiff", delete=True) as tmp:
        subprocess.run(["say", "-v", voice, "-o", tmp.name, "--", phrase], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", tmp.name, str(output)],
            check=True,
        )
    return output


def _labels_for_fixture(fixture: dict[str, Any], index_path: Path) -> dict[str, str]:
    raw_index = fixture.get("expected_audio_gatekeeper_top_index")
    labels = {head: "0" for head in HEAD_ORDER}
    if raw_index is None:
        return labels
    if not isinstance(raw_index, int):
        raise ValueError(
            f"expected_audio_gatekeeper_top_index must be int or null at {index_path}: {raw_index!r}"
        )
    if raw_index < 0 or raw_index >= len(HEAD_ORDER):
        raise ValueError(f"audio gatekeeper head index out of range at {index_path}: {raw_index}")
    labels[HEAD_ORDER[raw_index]] = "1"
    if raw_index > 0:
        labels[HEAD_ORDER[0]] = "1"
    return labels


def _manifest_row(
    audio_path: Path,
    phrase: str,
    stem: str,
    source_suffix: str,
    split: str,
    labels: dict[str, str],
) -> dict[str, str]:
    return {
        "audio_path": str(audio_path),
        "transcript": phrase,
        "speaker_id": f"fixture-{source_suffix}",
        "session_id": f"fixture-{stem}-{source_suffix}",
        "device": "synthetic",
        "environment": "clean",
        "source": f"gatekeeper_fixture:{source_suffix}",
        "duration_s": _duration_seconds(audio_path),
        "split": split,
        **labels,
    }


def _duration_seconds(path: Path) -> str:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
    if rate <= 0:
        raise ValueError(f"{path} has invalid WAV sample rate {rate}")
    return f"{frames / rate:.3f}"


def _is_speakable_phrase(phrase: str) -> bool:
    return bool(phrase) and not phrase.startswith("(")


def _safe_name(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    return "_".join(part for part in safe.split("_") if part) or "voice"
