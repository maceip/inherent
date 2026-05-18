import pytest

from inherent import HEAD_ORDER
from inherent.eval.fixture_quality import _expected_positive_index, score_fixture_rows


def test_fixture_quality_expected_index_uses_intent_head_when_present():
    row = _row(isInteresting="1", hasStartTimerIntent="1")

    assert _expected_positive_index(row) == HEAD_ORDER.index("hasStartTimerIntent")


def test_fixture_quality_expected_index_allows_pure_interesting_row():
    row = _row(isInteresting="1")

    assert _expected_positive_index(row) == 0


def test_fixture_quality_expected_index_skips_negative_row():
    row = _row()

    assert _expected_positive_index(row) is None


def test_fixture_quality_rejects_multi_intent_rows():
    row = _row(isInteresting="1", hasStartTimerIntent="1", hasAddToListIntent="1")

    with pytest.raises(ValueError, match="at most one positive intent"):
        _expected_positive_index(row)


def test_fixture_quality_scores_intent_rows_against_intent_heads():
    timer_index = HEAD_ORDER.index("hasStartTimerIntent")
    rows = [
        {"session_id": "timer", "source": "gatekeeper_fixture:existing", **_row(isInteresting="1", hasStartTimerIntent="1")},
        {"session_id": "interesting", "source": "gatekeeper_fixture:existing", **_row(isInteresting="1")},
        {"session_id": "negative", "source": "gatekeeper_fixture:existing", **_row()},
    ]
    scores = [
        [0.98 if index == 0 else 0.95 if index == timer_index else 0.01 for index in range(len(HEAD_ORDER))],
        [0.80 if index == 0 else 0.02 for index in range(len(HEAD_ORDER))],
        [0.01 for _ in HEAD_ORDER],
    ]

    report = score_fixture_rows(rows, scores, source_filter="gatekeeper_fixture:existing", interesting_threshold=0.5)

    assert report["positive_passed"] == 2
    assert report["positive_total"] == 2


def _row(**labels):
    return {head: labels.get(head, "0") for head in HEAD_ORDER}
