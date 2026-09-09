import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, precision_recall_curve, auc


def filter_eligible(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to include eligible dies only (where old_label == 0).
    Dies that were already dead pre-test (old_label == 1) are excluded from evaluation metrics.
    """
    if "old_label" not in df.columns:
        raise KeyError("DataFrame must contain 'old_label' column to filter eligible dies.")
    return df[df["old_label"] == 0].copy()


def confusion_and_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute official hackathon metrics and confusion matrix on eligible dies.

    Classes:
    0 = Pass
    1 = Fail

    Metrics:
    - Overall Accuracy
    - Pass Accuracy (Recall) / Fail Accuracy (Recall)
    - Pass Precision / Fail Precision
    - Pass F1 / Fail F1
    - Confusion Matrix matching README layout
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))

    # precision, recall, f1 for [class 0 (pass), class 1 (fail)]
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    pass_prec, fail_prec = float(p[0]), float(p[1])
    pass_rec, fail_rec = float(r[0]), float(r[1])
    pass_f1, fail_f1 = float(f1[0]), float(f1[1])

    # Confusion matrix: rows = actual (Fail=1, Pass=0), cols = pred (Fail=1, Pass=0)
    # sklearn cm layout for labels [1, 0]:
    # [[TP_fail (Actual Fail, Pred Fail), FN_fail (Actual Fail, Pred Pass)],
    #  [FP_fail (Actual Pass, Pred Fail), TN_fail (Actual Pass, Pred Pass)]]
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    actual_fail_pred_fail = int(cm[0, 0])
    actual_fail_pred_pass = int(cm[0, 1])
    actual_pass_pred_fail = int(cm[1, 0])
    actual_pass_pred_pass = int(cm[1, 1])

    # Formatted confusion matrix table DataFrame as specified in README
    cm_df = pd.DataFrame(
        [
            {
                "Actual": "Actual Fail",
                "Pred Fail": actual_fail_pred_fail,
                "Pred Pass": actual_fail_pred_pass,
                "Metric": "Fail Accuracy (Recall)",
                "Value": fail_rec,
            },
            {
                "Actual": "Actual Pass",
                "Pred Fail": actual_pass_pred_fail,
                "Pred Pass": actual_pass_pred_pass,
                "Metric": "Pass Accuracy (Recall)",
                "Value": pass_rec,
            },
        ]
    )

    return {
        "overall_accuracy": acc,
        "pass_recall": pass_rec,
        "fail_recall": fail_rec,
        "pass_precision": pass_prec,
        "fail_precision": fail_prec,
        "pass_f1": pass_f1,
        "fail_f1": fail_f1,
        "actual_fail_pred_fail": actual_fail_pred_fail,
        "actual_fail_pred_pass": actual_fail_pred_pass,
        "actual_pass_pred_fail": actual_pass_pred_fail,
        "actual_pass_pred_pass": actual_pass_pred_pass,
        "confusion_matrix_df": cm_df,
    }


def compute_auc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Area Under Precision-Recall Curve for minority class (fail=1)."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    return float(auc(recall, precision))


def evaluate_predictions(
    df: pd.DataFrame, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> dict:
    """
    Main evaluation function for model predictions against ground truth dataset.
    Automatically filters for eligible dies (old_label == 0).
    """
    eligible_df = filter_eligible(df)
    eligible_indices = eligible_df.index.values

    y_true_eligible = eligible_df["label"].values
    y_pred_eligible = y_pred[eligible_indices]

    metrics = confusion_and_metrics(y_true_eligible, y_pred_eligible)
    metrics["total_dies"] = len(df)
    metrics["eligible_dies"] = len(eligible_df)
    metrics["eligible_fail_count"] = int(y_true_eligible.sum())

    if y_prob is not None:
        y_prob_eligible = y_prob[eligible_indices]
        metrics["auc_pr"] = compute_auc_pr(y_true_eligible, y_prob_eligible)

    return metrics
