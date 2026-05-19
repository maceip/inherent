"""Scaffold a small held-out quality label manifest (CSV only; audio separate).

Usage:
  PYTHONPATH=src python -m inherent.scripts.scaffold_quality_eval \\
    --output data/quality_eval_sketch.csv \\
    --positives-per-head 20

Then synthesize or record audio at the listed paths (see data/neural_bootstrap_recording_tasks.csv)
and run build_recorded / inherent-eval against the mel manifest.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from inherent import HEAD_ORDER, INTENT_HEAD_ORDER, INTERESTING_HEAD
from inherent.data.schema import LABEL_TEMPLATE_COLUMNS

# Extra single-intent lines when bootstrap CSV has fewer than --positives-per-head rows.
_EXTRA_POSITIVE_TRANSCRIPTS: dict[str, list[str]] = {
    "hasAddToListIntent": [
        "Add bananas to the grocery list.",
        "Put dish soap on my shopping list.",
        "I need to remember to buy printer paper.",
    ],
    "hasTermSearchQuery": [
        "What does amortization mean?",
        "Look up the definition of epistemic.",
    ],
    "hasPhotoQuery": [
        "Show me pictures from last vacation.",
        "Find the screenshot of the wifi password.",
    ],
    "hasCalendarEvent": [
        "Block Friday afternoon for focus time.",
        "Move my dentist appointment to next week.",
    ],
    "hasCreateDocIntent": [
        "Turn these bullets into a doc.",
        "Start a memo about the pricing change.",
    ],
    "hasPersonContext": [
        "What did Jordan mention about the budget?",
        "Pull up context on my manager.",
    ],
    "hasEventContext": [
        "Summarize what we decided in standup.",
        "What was the outcome of the vendor review?",
    ],
    "hasDeepResearchIntent": [
        "Compare the leading on-device ASR stacks.",
        "Research competitors in wearable assistants.",
    ],
    "hasInsightIntent": [
        "What should I watch out for here?",
        "Help me think through this tradeoff.",
    ],
    "hasBrowsingAgentIntent": [
        "Check the order status on the retailer site.",
        "Fill out the return form online.",
    ],
    "hasCallingAgentIntent": [
        "Call the pharmacy about my refill.",
        "Ring the airline about the delay.",
    ],
    "hasStartTimerIntent": [
        "Set a fifteen minute timer.",
        "Wake me up in twenty minutes.",
    ],
}

_AMBIENT_NEGATIVE_TRANSCRIPTS = [
    "Office HVAC and distant chatter.",
    "Television dialogue in the next room.",
    "Cafe background noise only.",
    "Two coworkers talking, not to the assistant.",
]

_INTERESTING_ONLY_TRANSCRIPTS = [
    "The key takeaway is we should ship the smaller model first.",
    "Remember that the beta feedback was mostly about latency.",
    "I want to save this thought for the retro.",
]

_HARD_NEGATIVE_TRANSCRIPTS = [
    "It is a beautiful day outside.",
    "We should call this function after rendering.",
    "The calendar word appears in the article title.",
    "Search is mentioned but nobody asked for anything.",
    "I made a list of issues during the conversation.",
]


def _blank_labels() -> dict[str, str]:
    return {head: "0" for head in HEAD_ORDER}


def _row(
    audio_path: str,
    *,
    split: str,
    transcript: str,
    speaker_id: str,
    session_id: str,
    source: str,
    labels: dict[str, str],
) -> dict[str, str]:
    row = {
        "audio_path": audio_path,
        "transcript": transcript,
        "speaker_id": speaker_id,
        "session_id": session_id,
        "device": "quality_eval_scaffold",
        "environment": "scaffold",
        "source": source,
        "duration_s": "",
        "split": split,
    }
    row.update(labels)
    return row


def _load_bootstrap_tasks(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _transcripts_for_head(head: str, bootstrap_rows: list[dict[str, str]], count: int) -> list[str]:
    """Collect bootstrap lines first, then extras, then deterministic fillers."""
    transcripts = [row["transcript"] for row in bootstrap_rows if row.get("transcript")]
    transcripts.extend(_EXTRA_POSITIVE_TRANSCRIPTS.get(head, []))
    filler_index = 1
    while len(transcripts) < count:
        transcripts.append(f"Quality eval placeholder utterance {filler_index} for {head}.")
        filler_index += 1
    return transcripts[:count]


def scaffold_manifest(*, positives_per_head: int, output: Path, bootstrap_csv: Path) -> int:
    rows: list[dict[str, str]] = []
    bootstrap = _load_bootstrap_tasks(bootstrap_csv)
    bootstrap_by_head: dict[str, list[dict[str, str]]] = {head: [] for head in HEAD_ORDER}
    for task in bootstrap:
        head = task.get("head") or ""
        if head in bootstrap_by_head:
            bootstrap_by_head[head].append(task)
        elif task.get("task_type") == "hard_negative":
            bootstrap_by_head.setdefault("_hard_negative", []).append(task)

    # Ambient negatives (eval split): isInteresting=0, all intents 0.
    for index, transcript in enumerate(_AMBIENT_NEGATIVE_TRANSCRIPTS, start=1):
        labels = _blank_labels()
        rows.append(
            _row(
                f"data/eval_recorded/ambient/ambient_{index:03d}.wav",
                split="eval",
                transcript=transcript,
                speaker_id=f"ambient-{index:03d}",
                session_id=f"ambient-{index:03d}",
                source="quality_scaffold:ambient",
                labels=labels,
            )
        )

    # Hard near-miss negatives from bootstrap + scaffold.
    hard_tasks = bootstrap_by_head.pop("_hard_negative", [])
    hard_transcripts = [t["transcript"] for t in hard_tasks] + _HARD_NEGATIVE_TRANSCRIPTS
    for index, transcript in enumerate(hard_transcripts, start=1):
        rows.append(
            _row(
                f"data/eval_recorded/hard_negative/hard_{index:03d}.wav",
                split="eval",
                transcript=transcript,
                speaker_id=f"hard-{index:03d}",
                session_id=f"hard-{index:03d}",
                source="quality_scaffold:hard_negative",
                labels=_blank_labels(),
            )
        )

    # isInteresting-only positives (no intent head).
    interesting_tasks = bootstrap_by_head.pop(INTERESTING_HEAD, [])
    interesting_transcripts = [t["transcript"] for t in interesting_tasks] + _INTERESTING_ONLY_TRANSCRIPTS
    for index, transcript in enumerate(interesting_transcripts[: max(10, positives_per_head // 2)], start=1):
        labels = _blank_labels()
        labels[INTERESTING_HEAD] = "1"
        rows.append(
            _row(
                f"data/eval_recorded/isInteresting_only/interesting_{index:03d}.wav",
                split="eval",
                transcript=transcript,
                speaker_id=f"interesting-{index:03d}",
                session_id=f"interesting-{index:03d}",
                source="quality_scaffold:interesting_only",
                labels=labels,
            )
        )

    # Single-intent positives: positives_per_head each, prefer bootstrap transcripts.
    for head in INTENT_HEAD_ORDER:
        labels_base = _blank_labels()
        labels_base[INTERESTING_HEAD] = "1"
        transcripts = _transcripts_for_head(head, bootstrap_by_head.get(head, []), positives_per_head)
        for index, transcript in enumerate(transcripts, start=1):
            labels = dict(labels_base)
            labels[head] = "1"
            rows.append(
                _row(
                    f"data/eval_recorded/{head}/{head}_{index:03d}.wav",
                    split="eval",
                    transcript=transcript,
                    speaker_id=f"{head}-{index:03d}",
                    session_id=f"{head}-session",
                    source="quality_scaffold:positive",
                    labels=labels,
                )
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/quality_eval_sketch.csv"))
    parser.add_argument(
        "--bootstrap-csv",
        type=Path,
        default=Path("data/neural_bootstrap_recording_tasks.csv"),
    )
    parser.add_argument("--positives-per-head", type=int, default=20)
    args = parser.parse_args()
    count = scaffold_manifest(
        positives_per_head=args.positives_per_head,
        output=args.output,
        bootstrap_csv=args.bootstrap_csv,
    )
    print(f"wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
