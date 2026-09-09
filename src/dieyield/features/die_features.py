import numpy as np
import pandas as pd


def prune_collinear_features(
    df: pd.DataFrame, feature_cols: list[str], threshold: float = 0.98
) -> list[str]:
    """
    Drop features with pairwise correlation > threshold to reduce redundant 500-feature space.
    """
    if len(feature_cols) <= 1:
        return feature_cols

    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    selected = [c for c in feature_cols if c not in to_drop]
    return selected


def preprocess_die_features(
    df: pd.DataFrame,
    correlation_threshold: float = 0.98,
    drop_zero_var: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """
    Clean die-level parametric features: fill NaNs with column median,
    remove zero-variance features, and prune collinear features.
    """
    df = df.copy()
    
    # Identify parametric feature columns (feature_1, feature_2, ...)
    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feat_cols:
        return df, []

    # Median imputation
    for col in feat_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Zero-variance pruning
    if drop_zero_var:
        stds = df[feat_cols].std()
        feat_cols = stds[stds > 1e-8].index.tolist()

    # Collinearity pruning
    if correlation_threshold < 1.0:
        feat_cols = prune_collinear_features(df, feat_cols, threshold=correlation_threshold)

    return df, feat_cols
