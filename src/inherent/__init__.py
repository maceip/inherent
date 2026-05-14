"""Public package constants for the inherent audio gatekeeper."""

from __future__ import annotations

__version__ = "0.1.0"

HEAD_ORDER: tuple[str, ...] = (
    "isInteresting",
    "hasAddToListIntent",
    "hasTermSearchQuery",
    "hasPhotoQuery",
    "hasCalendarEvent",
    "hasCreateDocIntent",
    "hasPersonContext",
    "hasEventContext",
    "hasDeepResearchIntent",
    "hasInsightIntent",
    "hasBrowsingAgentIntent",
    "hasCallingAgentIntent",
    "hasStartTimerIntent",
)

THRESHOLD_KEYS: tuple[str, ...] = (
    "is_interesting",
    "has_add_to_list_intent",
    "has_term_search_query",
    "has_photo_query",
    "has_calendar_event",
    "has_create_doc_intent",
    "has_person_context",
    "has_event_context",
    "has_deep_research_intent",
    "has_insight_intent",
    "has_browsing_agent_intent",
    "has_calling_agent_intent",
    "has_start_timer_intent",
)

# Seed defaults for the runtime threshold table. Training/eval should tune these
# before release.
DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.57,
    0.65,
    0.90,
    0.90,
    0.90,
    0.90,
    0.56,
    0.74,
    0.90,
    0.90,
    0.90,
    0.90,
    0.90,
)

NUM_HEADS = len(HEAD_ORDER)
INTENT_HEAD_ORDER = HEAD_ORDER[1:]
NUM_INTENT_HEADS = len(INTENT_HEAD_ORDER)
INTERESTING_HEAD = HEAD_ORDER[0]
DEFAULT_THRESHOLDS_BY_KEY = dict(zip(THRESHOLD_KEYS, DEFAULT_THRESHOLDS, strict=True))

__all__ = [
    "DEFAULT_THRESHOLDS",
    "DEFAULT_THRESHOLDS_BY_KEY",
    "HEAD_ORDER",
    "INTERESTING_HEAD",
    "INTENT_HEAD_ORDER",
    "NUM_HEADS",
    "NUM_INTENT_HEADS",
    "THRESHOLD_KEYS",
    "__version__",
]
