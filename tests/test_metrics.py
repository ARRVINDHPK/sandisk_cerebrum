import numpy as np
import pandas as pd
import pytest
from src.dieyield.evaluate.metrics import filter_eligible, confusion_and_metrics, evaluate_predictions


def test_eligible_die_filtering():
    df = pd.DataFrame({
        "old_label": [0, 0, 1, 1, 0],
        "label":     [0, 1, 1, 1, 0],
        "wafer_id":  ["W1"] * 5
    })
    filtered = filter_eligible(df)
    assert len(filtered) == 3
    assert (filtered["old_label"] == 0).all()
    assert set(filtered.index) == {0, 1, 4}


def test_confusion_and_metrics_exact():
    # Construct toy dataset of 20 eligible dies
    # Actual: 5 fails (1), 15 passes (0)
    # Predicted:
    # 4 actual fails predicted fail (TP_fail = 4)
    # 1 actual fail predicted pass (FN_fail = 1)
    # 2 actual passes predicted fail (FP_fail = 2)
    # 13 actual passes predicted pass (TN_fail = 13)
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    metrics = confusion_and_metrics(y_true, y_pred)

    assert metrics["overall_accuracy"] == pytest.approx(17 / 20)
    assert metrics["fail_recall"] == pytest.approx(4 / 5)       # 0.8
    assert metrics["pass_recall"] == pytest.approx(13 / 15)     # 0.86666666...
    assert metrics["fail_precision"] == pytest.approx(4 / 6)    # 0.66666666...
    assert metrics["pass_precision"] == pytest.approx(13 / 14)  # 0.9285714...

    assert metrics["actual_fail_pred_fail"] == 4
    assert metrics["actual_fail_pred_pass"] == 1
    assert metrics["actual_pass_pred_fail"] == 2
    assert metrics["actual_pass_pred_pass"] == 13


def test_old_label_exclusion_in_evaluate_predictions():
    # 10 rows: 5 eligible (old_label=0), 5 pre-test fails (old_label=1)
    df = pd.DataFrame({
        "old_label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "label":     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    })
    # y_pred gives WRONG predictions on the old_label=1 dies (e.g., predicting 0 for old_label=1)
    y_pred = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    res = evaluate_predictions(df, y_pred)
    # Eligible rows: [0, 0, 0, 1, 1]
    # y_pred on eligible: [0, 0, 0, 1, 0]
    # Correct eligible: indices 0, 1, 2, 3 -> 4 out of 5 correct
    assert res["eligible_dies"] == 5
    assert res["overall_accuracy"] == pytest.approx(4 / 5)
    assert res["fail_recall"] == pytest.approx(1 / 2)
