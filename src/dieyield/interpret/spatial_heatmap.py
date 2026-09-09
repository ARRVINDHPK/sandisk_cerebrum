from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render_wafer_spatial_heatmap(
    df: pd.DataFrame,
    wafer_id: str,
    probs: np.ndarray,
    shap_matrix: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    output_path: str | None = None
) -> plt.Figure:
    """
    Render 3-panel spatial wafer grid visualization:
    1. Ground Truth Status (old_label / label)
    2. Model Predicted Failure Probability
    3. Spatial Context SHAP Contribution Sum
    """
    wdf = df[df["wafer_id"] == wafer_id].copy()
    if wdf.empty:
        raise ValueError(f"Wafer ID '{wafer_id}' not found in DataFrame.")

    indices = wdf.index.values
    w_probs = probs[indices]

    max_r = wdf["die_row"].max() + 1
    max_c = wdf["die_col"].max() + 1

    grid_gt = np.full((max_r, max_c), np.nan)
    grid_prob = np.full((max_r, max_c), np.nan)
    grid_shap = np.full((max_r, max_c), np.nan)

    rows = wdf["die_row"].values
    cols = wdf["die_col"].values

    # Ground truth: 0=pass, 1=new fail, 2=old fail
    gt_vals = wdf["label"].values.copy()
    if "old_label" in wdf.columns:
        old_fails = (wdf["old_label"] == 1).values
        gt_vals[old_fails] = 2

    grid_gt[rows, cols] = gt_vals
    grid_prob[rows, cols] = w_probs

    if shap_matrix is not None and feature_names is not None:
        spatial_feat_indices = [
            i for i, col in enumerate(feature_names)
            if any(k in col for k in ["zone", "radial", "x_coord", "y_coord", "density"])
        ]
        if spatial_feat_indices:
            w_shap = shap_matrix[indices][:, spatial_feat_indices].sum(axis=1)
            grid_shap[rows, cols] = w_shap

    fig, axes = plt.subplots(1, 3 if shap_matrix is not None else 2, figsize=(15 if shap_matrix is not None else 10, 5))

    # 1. Ground Truth
    im0 = axes[0].imshow(grid_gt, cmap="YlOrRd", vmin=0, vmax=2)
    axes[0].set_title(f"Ground Truth (Wafer {wafer_id})", fontweight="bold")
    plt.colorbar(im0, ax=axes[0], ticks=[0, 1, 2], label="0:Pass, 1:New Fail, 2:Old Fail")

    # 2. Predicted Probabilities
    im1 = axes[1].imshow(grid_prob, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Predicted Fail Probability", fontweight="bold")
    plt.colorbar(im1, ax=axes[1], label="Probability")

    # 3. Spatial SHAP
    if shap_matrix is not None:
        vmax = np.nanmax(np.abs(grid_shap)) if not np.isnan(grid_shap).all() else 1.0
        im2 = axes[2].imshow(grid_shap, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[2].set_title("Spatial Context SHAP Contribution", fontweight="bold")
        plt.colorbar(im2, ax=axes[2], label="SHAP Value")

    for ax in axes:
        ax.set_xlabel("Die Column")
        ax.set_ylabel("Die Row")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)

    return fig
