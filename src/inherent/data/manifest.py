"""Build strict raw-audio 13-head manifests from corpus indexes."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .. import HEAD_ORDER, INTENT_HEAD_ORDER
from .directedness import DirectednessSample
from .intents import IntentSample


@dataclass(frozen=True)
class RawAudioLabelSample:
    audio_path: Path
    labels: tuple[str, ...]
    source: str


def from_directedness(sample: DirectednessSample) -> RawAudioLabelSample:
    labels = ["0"] * len(HEAD_ORDER)
    labels[0] = str(sample.label)
    return RawAudioLabelSample(audio_path=sample.audio_path, labels=tuple(labels), source=sample.source)


def from_intent(sample: IntentSample) -> RawAudioLabelSample:
    labels = ["0"] * len(HEAD_ORDER)
    labels[0] = "1"
    for index, head in enumerate(HEAD_ORDER):
        if head in INTENT_HEAD_ORDER and sample.head_labels.get(head, False):
            labels[index] = "1"
    if "1" not in labels[1:]:
        raise ValueError(f"intent sample has no positive intent labels: {sample.audio_path}")
    return RawAudioLabelSample(audio_path=sample.audio_path, labels=tuple(labels), source=sample.source)


def write_raw_audio_manifest(samples: Iterable[RawAudioLabelSample], output_path: str | Path) -> int:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("cannot write an empty raw-audio manifest")
    _validate_label_coverage(sample_list)
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", *HEAD_ORDER])
        writer.writeheader()
        for sample in sample_list:
            writer.writerow(
                {
                    "audio_path": str(sample.audio_path.resolve()),
                    **{head: sample.labels[i] for i, head in enumerate(HEAD_ORDER)},
                }
            )
    return len(sample_list)


def combine_indexes(
    directedness_samples: Iterable[DirectednessSample],
    intent_samples: Iterable[IntentSample],
) -> list[RawAudioLabelSample]:
    by_audio_path: dict[Path, RawAudioLabelSample] = {}
    for sample in directedness_samples:
        _merge_sample(by_audio_path, from_directedness(sample))
    for sample in intent_samples:
        _merge_sample(by_audio_path, from_intent(sample))
    combined = list(by_audio_path.values())
    _validate_label_coverage(combined)
    return combined


def _merge_sample(by_audio_path: dict[Path, RawAudioLabelSample], sample: RawAudioLabelSample) -> None:
    key = sample.audio_path.expanduser().resolve()
    existing = by_audio_path.get(key)
    if existing is None:
        by_audio_path[key] = RawAudioLabelSample(
            audio_path=key,
            labels=sample.labels,
            source=sample.source,
        )
        return

    if existing.labels[0] != sample.labels[0]:
        raise ValueError(
            f"conflicting isInteresting labels for duplicate audio path {key}: "
            f"{existing.labels[0]} from {existing.source}, {sample.labels[0]} from {sample.source}"
        )

    merged = tuple("1" if left == "1" or right == "1" else "0" for left, right in zip(existing.labels, sample.labels, strict=True))
    by_audio_path[key] = RawAudioLabelSample(
        audio_path=key,
        labels=merged,
        source=f"{existing.source};{sample.source}",
    )


def _validate_label_coverage(samples: list[RawAudioLabelSample]) -> None:
    positives = [0] * len(HEAD_ORDER)
    negatives = [0] * len(HEAD_ORDER)
    for sample in samples:
        if len(sample.labels) != len(HEAD_ORDER):
            raise ValueError(f"{sample.audio_path} has {len(sample.labels)} labels; expected {len(HEAD_ORDER)}")
        for index, value in enumerate(sample.labels):
            if value == "1":
                positives[index] += 1
            elif value == "0":
                negatives[index] += 1
            else:
                raise ValueError(f"{sample.audio_path} has invalid label {value!r}; expected 0 or 1")
    missing_positive = [HEAD_ORDER[i] for i, count in enumerate(positives) if count == 0]
    missing_negative = [HEAD_ORDER[i] for i, count in enumerate(negatives) if count == 0]
    if missing_positive or missing_negative:
        raise ValueError(
            "raw-audio manifest requires at least one positive and one negative per head; "
            f"missing_positive={missing_positive}, missing_negative={missing_negative}"
        )
