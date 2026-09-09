import numpy as np
import pandas as pd


def wafer_level_train_val_split(
    df: pd.DataFrame,
    blocks: np.ndarray | None = None,
    val_frac: float = 0.2,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray | None, np.ndarray | None]:
    """
    Perform a strict wafer-level train/validation split.
    Guarantees all dies of a single wafer stay together in either train or val.

    Returns (train_df, val_df, train_blocks, val_blocks).
    """
    if "wafer_id" not in df.columns:
        raise ValueError("DataFrame must contain 'wafer_id' column for wafer-level split.")

    unique_wafers = df["wafer_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled_wafers = rng.permutation(unique_wafers)

    n_val_wafers = max(1, int(len(unique_wafers) * val_frac))
    val_wafers = set(shuffled_wafers[:n_val_wafers])
    train_wafers = set(shuffled_wafers[n_val_wafers:])

    val_mask = df["wafer_id"].isin(val_wafers).values
    train_mask = df["wafer_id"].isin(train_wafers).values

    train_df = df.iloc[train_mask].reset_index(drop=True)
    val_df = df.iloc[val_mask].reset_index(drop=True)

    train_blocks = blocks[train_mask] if blocks is not None else None
    val_blocks = blocks[val_mask] if blocks is not None else None

    return train_df, val_df, train_blocks, val_blocks
