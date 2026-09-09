import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc

from src.dieyield.evaluate.metrics import filter_eligible, evaluate_predictions


def generate_comparison_table(metrics_a: dict, metrics_b: dict) -> pd.DataFrame:
    """
    Generate side-by-side comparison table between Model A and Model B with deltas.
    """
    metric_keys = [
        ("overall_accuracy", "Overall Accuracy"),
        ("fail_recall", "Fail Recall (Sensitivity)"),
        ("fail_precision", "Fail Precision"),
        ("fail_f1", "Fail F1-Score"),
        ("pass_recall", "Pass Recall"),
        ("pass_precision", "Pass Precision"),
        ("pass_f1", "Pass F1-Score"),
        ("auc_pr", "Minority Class AUC-PR"),
    ]

    rows = []
    for key, name in metric_keys:
        val_a = metrics_a.get(key, np.nan)
        val_b = metrics_b.get(key, np.nan)
        abs_delta = val_b - val_a if not (np.isnan(val_a) or np.isnan(val_b)) else np.nan
        rel_delta = (abs_delta / val_a * 100.0) if (val_a and val_a > 0) else np.nan

        rows.append({
            "Metric": name,
            "Model A (Spatial+Die)": f"{val_a:.4f}" if not np.isnan(val_a) else "N/A",
            "Model B (+Block Signal)": f"{val_b:.4f}" if not np.isnan(val_b) else "N/A",
            "Absolute Delta": f"{abs_delta:+.4f}" if not np.isnan(abs_delta) else "N/A",
            "Relative Delta (%)": f"{rel_delta:+.2f}%" if not np.isnan(rel_delta) else "N/A",
        })

    return pd.DataFrame(rows)


def segmented_delta_analysis(df: pd.DataFrame, preds_a: np.ndarray, preds_b: np.ndarray) -> dict:
    """
    Segment test-set failing dies into:
    1. 'Spatial-adjacent' fails (dies near pre-test failures: old_label_density > 0)
    2. 'Isolated marginal' fails (isolated dies with old_label_density == 0)

    Calculates Model A vs Model B fail recall per segment to show exact block signal value-add.
    """
    elig = filter_eligible(df)
    indices = elig.index.values

    y_true = elig["label"].values
    y_a = preds_a[indices]
    y_b = preds_b[indices]

    density_col = [c for c in elig.columns if "old_label_density" in c]
    if density_col:
        dens = elig[density_col[0]].values
    else:
        dens = np.zeros(len(elig))

    fails_mask = (y_true == 1)

    adj_mask = fails_mask & (dens > 0)
    iso_mask = fails_mask & (dens == 0)

    adj_recall_a = float((y_a[adj_mask] == 1).mean()) if adj_mask.sum() > 0 else 0.0
    adj_recall_b = float((y_b[adj_mask] == 1).mean()) if adj_mask.sum() > 0 else 0.0

    iso_recall_a = float((y_a[iso_mask] == 1).mean()) if iso_mask.sum() > 0 else 0.0
    iso_recall_b = float((y_b[iso_mask] == 1).mean()) if iso_mask.sum() > 0 else 0.0

    return {
        "spatial_adjacent_fails_count": int(adj_mask.sum()),
        "model_a_adjacent_fail_recall": adj_recall_a,
        "model_b_adjacent_fail_recall": adj_recall_b,
        "isolated_marginal_fails_count": int(iso_mask.sum()),
        "model_a_isolated_fail_recall": iso_recall_a,
        "model_b_isolated_fail_recall": iso_recall_b,
        "isolated_recall_lift": iso_recall_b - iso_recall_a,
    }


def plot_precision_recall_comparison(
    df: pd.DataFrame,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    output_path: str = "reports/figures/pr_curves.png"
):
    """Plot overlaid PR curves for Model A vs Model B on eligible dies."""
    elig = filter_eligible(df)
    indices = elig.index.values

    y_true = elig["label"].values
    pa = probs_a[indices]
    pb = probs_b[indices]

    prec_a, rec_a, _ = precision_recall_curve(y_true, pa, pos_label=1)
    prec_b, rec_b, _ = precision_recall_curve(y_true, pb, pos_label=1)

    auc_a = float(auc(rec_a, prec_a))
    auc_b = float(auc(rec_b, prec_b))

    plt.figure(figsize=(8, 6))
    plt.plot(rec_a, prec_a, label=f"Model A (Spatial+Die) - AUC: {auc_a:.4f}", color="#1f77b4", lw=2)
    plt.plot(rec_b, prec_b, label=f"Model B (+Block Signal) - AUC: {auc_b:.4f}", color="#ff7f0e", lw=2)
    plt.xlabel("Recall (Fail Class)", fontsize=12)
    plt.ylabel("Precision (Fail Class)", fontsize=12)
    plt.title("Precision-Recall Curve Comparison (Eligible Dies)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
