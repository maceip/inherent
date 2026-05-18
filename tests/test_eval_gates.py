import pytest
import numpy as np

from inherent import HEAD_ORDER
from inherent import THRESHOLD_KEYS
from inherent.eval.evaluate import evaluate_gates
from inherent.eval.evaluate import _runtime_static_for_checkpoint
from inherent.eval.thresholds import calibrate_thresholds


def metrics(auc=0.9, eer=0.05, fpr=0.05):
    return {
        head: {
            "auc": auc,
            "eer": eer,
            "fpr_at_recall_95": fpr,
        }
        for head in HEAD_ORDER
    }


def test_evaluate_gates_passes_thresholds():
    result = evaluate_gates(
        metrics(),
        {
            "is_interesting": {"auc": 0.85, "eer": 0.15},
            "intent_heads": {"mean_auc": 0.8, "min_auc": 0.7, "max_fpr_at_recall_95": 0.25},
        },
    )

    assert result["passed"] is True


def test_evaluate_gates_fails_thresholds():
    values = metrics()
    values["hasCallingAgentIntent"]["auc"] = 0.4

    result = evaluate_gates(
        values,
        {
            "is_interesting": {"auc": 0.85, "eer": 0.15},
            "intent_heads": {"mean_auc": 0.8, "min_auc": 0.7},
        },
    )

    assert result["passed"] is False
    assert result["checks"]["intent_min_auc"]["passed"] is False


def test_checkpoint_eval_uses_checkpoint_padding_by_default():
    assert _runtime_static_for_checkpoint({"config": {"training": {"padding": "runtime_static"}}}, override=None)
    assert not _runtime_static_for_checkpoint({"config": {"training": {"padding": "dynamic"}}}, override=None)
    assert not _runtime_static_for_checkpoint({"config": {"training": {"padding": "runtime_static"}}}, override=False)
    assert _runtime_static_for_checkpoint({"config": {"training": {"padding": "dynamic"}}}, override=True)

    with pytest.raises(ValueError, match="padding"):
        _runtime_static_for_checkpoint({"config": {"training": {"padding": "sometimes"}}}, override=None)


def test_calibrate_thresholds_maximizes_f1_above_recall_floor():
    scores = np.tile(np.array([[0.1], [0.2], [0.6], [0.8], [0.9]], dtype=np.float32), (1, len(HEAD_ORDER)))
    labels = np.tile(np.array([[0.0], [0.0], [1.0], [1.0], [1.0]], dtype=np.float32), (1, len(HEAD_ORDER)))

    report = calibrate_thresholds(scores, labels, min_recall=2 / 3)

    assert tuple(report["thresholds_by_key"]) == THRESHOLD_KEYS
    assert report["thresholds_by_key"]["is_interesting"] == pytest.approx(0.6)
    assert report["heads"]["isInteresting"]["precision"] == pytest.approx(1.0)
    assert report["heads"]["isInteresting"]["recall"] == pytest.approx(1.0)


def test_calibrate_thresholds_can_skip_heads_missing_label_coverage():
    scores = np.full((4, len(HEAD_ORDER)), 0.5, dtype=np.float32)
    labels = np.zeros((4, len(HEAD_ORDER)), dtype=np.float32)
    labels[0, 0] = 1.0

    with pytest.raises(ValueError, match="hasAddToListIntent"):
        calibrate_thresholds(scores, labels)

    report = calibrate_thresholds(scores, labels, require_all_heads=False)

    assert "isInteresting" in report["heads"]
    assert "hasAddToListIntent" in report["skipped_heads"]
