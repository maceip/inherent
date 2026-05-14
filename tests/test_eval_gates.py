from inherent import HEAD_ORDER
from inherent.eval.evaluate import evaluate_gates


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
