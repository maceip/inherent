from inherent import (
    DEFAULT_THRESHOLDS,
    DEFAULT_THRESHOLDS_BY_KEY,
    HEAD_ORDER,
    NUM_HEADS,
    THRESHOLD_KEYS,
    __version__,
)


def test_head_order_matches_runtime_contract():
    assert __version__ == "0.1.0"
    assert HEAD_ORDER == (
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
    assert NUM_HEADS == 13


def test_default_thresholds_match_runtime_order():
    assert DEFAULT_THRESHOLDS == (
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
    assert tuple(DEFAULT_THRESHOLDS_BY_KEY) == THRESHOLD_KEYS
