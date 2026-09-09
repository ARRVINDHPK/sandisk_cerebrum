import argparse
import json
from pathlib import Path
import pandas as pd
import yaml

from src.dieyield.data.loader import load_split_cached
from src.dieyield.data.splits import wafer_level_train_val_split
from src.dieyield.models.model_a import ModelA
from src.dieyield.models.model_b import ModelB
from src.dieyield.evaluate.compare import (
    generate_comparison_table,
    segmented_delta_analysis,
    plot_precision_recall_comparison,
)
from src.dieyield.interpret.shap_explain import compute_shap_values, plot_global_shap_summary
from src.dieyield.interpret.spatial_heatmap import render_wafer_spatial_heatmap
from src.dieyield.interpret.block_attribution import render_block_anomaly_grid


def train_model_a():
    print("=== Training Model A (Die + Spatial) ===")
    with open("configs/model_a.yaml") as f:
        config_a = yaml.safe_load(f)

    df_train, _ = load_split_cached("input/train.csv")
    train_df, val_df, _, _ = wafer_level_train_val_split(df_train, val_frac=0.2, seed=config_a.get("seed", 42))

    model_a = ModelA(config_a)
    model_a.train(train_df, val_df)

    runs_dir = Path("reports/runs/model_a")
    model_a.save(str(runs_dir))

    df_test, _ = load_split_cached("input/test.csv")
    metrics = model_a.evaluate(df_test)
    
    # Format confusion matrix for json serialization
    metrics_log = {k: float(v) if isinstance(v, (float, int)) else str(v) for k, v in metrics.items() if k != "confusion_matrix_df"}
    with open(runs_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    print(f"Model A Training Complete. Fail F1: {metrics['fail_f1']:.4f}, Fail Recall: {metrics['fail_recall']:.4f}")


def train_model_b():
    print("=== Training Model B (+ 2000-dim Block Signals) ===")
    with open("configs/model_b.yaml") as f:
        config_b = yaml.safe_load(f)

    df_train, blocks_train = load_split_cached("input/train.csv")
    train_df, val_df, train_blocks, val_blocks = wafer_level_train_val_split(
        df_train, blocks_train, val_frac=0.2, seed=config_b.get("seed", 42)
    )

    model_b = ModelB(config_b)
    model_b.train(train_df, train_blocks, val_df, val_blocks)

    runs_dir = Path("reports/runs/model_b")
    model_b.save(str(runs_dir))

    df_test, blocks_test = load_split_cached("input/test.csv")
    metrics = model_b.evaluate(df_test, blocks_test)

    metrics_log = {k: float(v) if isinstance(v, (float, int)) else str(v) for k, v in metrics.items() if k != "confusion_matrix_df"}
    with open(runs_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    print(f"Model B Training Complete. Fail F1: {metrics['fail_f1']:.4f}, Fail Recall: {metrics['fail_recall']:.4f}")


def run_evaluation():
    print("=== Running Evaluation & Comparison ===")
    df_test, blocks_test = load_split_cached("input/test.csv")

    with open("configs/model_a.yaml") as f:
        cfg_a = yaml.safe_load(f)
    with open("configs/model_b.yaml") as f:
        cfg_b = yaml.safe_load(f)

    model_a = ModelA(cfg_a)
    model_a.load("reports/runs/model_a")

    model_b = ModelB(cfg_b)
    model_b.load("reports/runs/model_b")

    metrics_a = model_a.evaluate(df_test)
    metrics_b = model_b.evaluate(df_test, blocks_test)

    table_df = generate_comparison_table(metrics_a, metrics_b)
    print("\n--- Model A vs Model B Comparison Summary ---")
    print(table_df.to_string(index=False))

    probs_a = model_a.predict_proba(df_test)
    probs_b = model_b.predict_proba(df_test, blocks_test)
    plot_precision_recall_comparison(df_test, probs_a, probs_b, "reports/figures/pr_curves.png")

    preds_a = model_a.predict(df_test)
    preds_b = model_b.predict(df_test, blocks_test)
    seg_res = segmented_delta_analysis(df_test, preds_a, preds_b)
    
    print("\n--- Segmented Performance Delta (Isolated vs Adjacent Fails) ---")
    print(json.dumps(seg_res, indent=2))


def generate_submission(model_choice: str = "b", input_path: str = "input/validation.csv", output_path: str = "input/submission.csv"):
    print(f"=== Generating Submission CSV using Model {model_choice.upper()} ===")
    df, blocks = load_split_cached(input_path)

    if model_choice == "a":
        with open("configs/model_a.yaml") as f:
            cfg = yaml.safe_load(f)
        model = ModelA(cfg)
        model.load("reports/runs/model_a")
        preds = model.predict(df)
    else:
        with open("configs/model_b.yaml") as f:
            cfg = yaml.safe_load(f)
        model = ModelB(cfg)
        model.load("reports/runs/model_b")
        preds = model.predict(df, blocks)

    sub_df = pd.DataFrame({
        "wafer_id": df["wafer_id"],
        "die_row": df["die_row"],
        "die_col": df["die_col"],
        "predicted_label": preds
    })

    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} (Total dies: {len(sub_df)}, Predicted Fails: {(preds==1).sum()})")


def run_interpretability():
    print("=== Generating Interpretability Figures & Attribution Reports ===")
    df_test, blocks_test = load_split_cached("input/test.csv")

    with open("configs/model_a.yaml") as f:
        cfg_a = yaml.safe_load(f)
    model_a = ModelA(cfg_a)
    model_a.load("reports/runs/model_a")

    explainer, shap_mat, X_prep = compute_shap_values(model_a, df_test)
    plot_global_shap_summary(shap_mat, X_prep, "reports/figures/shap_summary.png")

    sample_wafer = df_test["wafer_id"].iloc[0]
    probs_a = model_a.predict_proba(df_test)
    render_wafer_spatial_heatmap(df_test, sample_wafer, probs_a, shap_mat, model_a.feature_names, "reports/figures/wafer_spatial_heatmap.png")

    if blocks_test is not None:
        sample_fail_idx = df_test[df_test["label"] == 1].index[0]
        render_block_anomaly_grid(blocks_test[sample_fail_idx], z_threshold=2.5, output_path="reports/figures/block_anomaly_heatmap.png")

    print("Interpretability artifacts generated in reports/figures/")


def main():
    parser = argparse.ArgumentParser(description="Die Yield Prediction CLI Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    train_p = subparsers.add_parser("train", help="Train model pipeline")
    train_p.add_argument("--model", type=str, choices=["a", "b", "all"], default="all", help="Model choice")

    eval_p = subparsers.add_parser("evaluate", help="Evaluate models and generate PR curves")
    
    sub_p = subparsers.add_parser("predict", help="Generate submission CSV")
    sub_p.add_argument("--model", type=str, choices=["a", "b"], default="b")
    sub_p.add_argument("--input", type=str, default="input/validation.csv")
    sub_p.add_argument("--output", type=str, default="input/submission.csv")

    interp_p = subparsers.add_parser("interpret", help="Generate interpretability figures")

    args = parser.parse_args()

    if args.command == "train":
        if args.model in ["a", "all"]:
            train_model_a()
        if args.model in ["b", "all"]:
            train_model_b()
    elif args.command == "evaluate":
        run_evaluation()
    elif args.command == "predict":
        generate_submission(args.model, args.input, args.output)
    elif args.command == "interpret":
        run_interpretability()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
