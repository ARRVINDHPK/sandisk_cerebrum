from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def compute_shap_values(model_wrapper, df: pd.DataFrame, blocks: np.ndarray | None = None) -> tuple[shap.Explainer, np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values for model wrapper (supports both TreeExplainer and Explainer).
    Returns (explainer, shap_matrix, feature_df).
    """
    if hasattr(model_wrapper, "encoder"):
        df_prep, feature_names = model_wrapper.prepare_features(df, blocks, is_train=False)
    else:
        df_prep, feature_names = model_wrapper.prepare_features(df, is_train=False)

    X = df_prep[feature_names]


    try:
        explainer = shap.TreeExplainer(model_wrapper.model)
        shap_values = explainer.shap_values(X)
    except Exception:
        # Fallback to Permutation/Generic Explainer if TreeExplainer doesn't support model
        explainer = shap.Explainer(model_wrapper.model.predict_proba, X.iloc[:50])
        shap_values = explainer(X.iloc[:200]).values

    if isinstance(shap_values, list):
        shap_matrix = shap_values[1]
    elif hasattr(shap_values, "values"):
        shap_matrix = shap_values.values
        if len(shap_matrix.shape) == 3:
            shap_matrix = shap_matrix[:, :, 1]
    else:
        shap_matrix = shap_values
        if len(shap_matrix.shape) == 3:
            shap_matrix = shap_matrix[:, :, 1]

    return explainer, shap_matrix, X.iloc[:len(shap_matrix)]


def plot_global_shap_summary(
    shap_matrix: np.ndarray,
    X: pd.DataFrame,
    output_path: str = "reports/figures/shap_summary.png",
    top_k: int = 20
):
    """Save global SHAP summary plot."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_matrix, X, max_display=top_k, show=False)
    plt.title("Global Feature Importance (SHAP Values)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def explain_single_die(
    shap_matrix: np.ndarray,
    X: pd.DataFrame,
    die_index: int,
    top_k: int = 10
) -> pd.DataFrame:
    """Extract top-K positive and negative SHAP feature contributions for a single die."""
    if die_index >= len(shap_matrix):
        die_index = 0

    row_shap = shap_matrix[die_index]
    row_vals = X.iloc[die_index]

    df_exp = pd.DataFrame({
        "feature": X.columns,
        "feature_value": row_vals.values,
        "shap_value": row_shap,
        "abs_shap": np.abs(row_shap),
    })

    df_exp = df_exp.sort_values(by="abs_shap", ascending=False).head(top_k)
    return df_exp[["feature", "feature_value", "shap_value"]]
