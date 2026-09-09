import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml

from src.dieyield.data.loader import load_split_cached
from src.dieyield.evaluate.compare import (
    generate_comparison_table,
    segmented_delta_analysis,
    plot_precision_recall_comparison,
)
from src.dieyield.interpret.shap_explain import compute_shap_values, plot_global_shap_summary
from src.dieyield.interpret.spatial_heatmap import render_wafer_spatial_heatmap
from src.dieyield.interpret.block_attribution import render_block_anomaly_grid
from src.dieyield.models.model_a import ModelA
from src.dieyield.models.model_b import ModelB


def main():
    print("=== Running Interpretability & Report Generator ===")
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_test, blocks_test = load_split_cached("input/test.csv")

    with open("configs/model_a.yaml") as f:
        cfg_a = yaml.safe_load(f)
    with open("configs/model_b.yaml") as f:
        cfg_b = yaml.safe_load(f)

    model_a = ModelA(cfg_a)
    model_a.load("reports/runs/model_a")

    model_b = ModelB(cfg_b)
    model_b.load("reports/runs/model_b")

    # 1. SHAP Summary Plots
    print("Generating SHAP summary plots...")
    explainer_a, shap_a, X_a = compute_shap_values(model_a, df_test)
    plot_global_shap_summary(shap_a, X_a, str(reports_dir / "shap_summary_a.png"))

    explainer_b, shap_b, X_b = compute_shap_values(model_b, df_test, blocks_test)
    plot_global_shap_summary(shap_b, X_b, str(reports_dir / "shap_summary_b.png"))

    # 2. Wafer Grid Heatmap
    print("Generating Wafer Spatial Heatmap...")
    sample_wafer = df_test["wafer_id"].iloc[0]
    probs_b = model_b.predict_proba(df_test, blocks_test)
    render_wafer_spatial_heatmap(
        df_test, sample_wafer, probs_b, shap_b, model_b.feature_names, str(reports_dir / "wafer_heatmap.png")
    )

    # 3. Block Anomaly Example Plot
    print("Generating Block Anomaly Example Plot...")
    if blocks_test is not None:
        fail_indices = df_test[df_test["label"] == 1].index
        sample_idx = fail_indices[0] if len(fail_indices) > 0 else 0
        render_block_anomaly_grid(blocks_test[sample_idx], z_threshold=2.5, output_path=str(reports_dir / "block_anomaly_example.png"))

    # 4. Model Comparison Bar Chart & Table
    print("Generating Comparison Figure...")
    metrics_a = model_a.evaluate(df_test)
    metrics_b = model_b.evaluate(df_test, blocks_test)
    plot_precision_recall_comparison(df_test, model_a.predict_proba(df_test), probs_b, str(reports_dir / "comparison.png"))

    print("Done! All interpretability figures saved to reports/.")


if __name__ == "__main__":
    main()
