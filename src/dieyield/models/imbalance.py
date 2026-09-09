import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """
    Compute scale_pos_weight ratio (negative_count / positive_count) for LightGBM/XGBoost.
    """
    y = np.asarray(y, dtype=int)
    pos_count = (y == 1).sum()
    neg_count = (y == 0).sum()
    if pos_count == 0:
        return 1.0
    return float(neg_count / pos_count)


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "fail_f1",
    min_thresh: float = 0.05,
    max_thresh: float = 0.95,
    step: float = 0.01
) -> tuple[float, float]:
    """
    Grid search optimal classification threshold on validation probabilities to maximize specified metric (default: fail_f1).
    Returns (best_threshold, best_score).
    """
    thresholds = np.arange(min_thresh, max_thresh + 1e-5, step)
    best_thresh = 0.5
    best_score = -1.0

    y_true = np.asarray(y_true, dtype=int)

    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, preds, labels=[0, 1], zero_division=0)
        
        if metric == "fail_f1":
            score = float(f1[1])
        elif metric == "fail_recall":
            score = float(r[1])
        elif metric == "fail_precision":
            score = float(p[1])
        else:
            raise ValueError(f"Unknown threshold search metric: {metric}")

        if score > best_score:
            best_score = score
            best_thresh = float(th)

    return best_thresh, best_score
