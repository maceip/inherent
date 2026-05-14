"""Shared CSV column definitions for recorded audio datasets."""

from __future__ import annotations

from .. import HEAD_ORDER

METADATA_COLUMNS: tuple[str, ...] = (
    "transcript",
    "speaker_id",
    "session_id",
    "device",
    "environment",
    "source",
    "duration_s",
    "split",
)
LABEL_TEMPLATE_COLUMNS: tuple[str, ...] = ("audio_path", *METADATA_COLUMNS, *HEAD_ORDER)
OPTIONAL_LABEL_COLUMNS = set(METADATA_COLUMNS)
ALLOWED_RAW_LABEL_COLUMNS = {"audio_path", *OPTIONAL_LABEL_COLUMNS, *HEAD_ORDER}
