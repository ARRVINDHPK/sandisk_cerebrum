import os
from pathlib import Path
import numpy as np
import pandas as pd


def parse_block_readings_to_array(series: pd.Series) -> np.ndarray:
    """
    Parse a series of space-separated block_readings strings into a 2D float32 numpy array (N x K).
    """
    if series.empty:
        return np.empty((0, 2000), dtype=np.float32)

    first_val = series.iloc[0]
    if isinstance(first_val, np.ndarray):
        return np.vstack(series.values).astype(np.float32)
    elif isinstance(first_val, (list, tuple)):
        return np.array(series.tolist(), dtype=np.float32)

    # Space-separated string parsing
    parsed = [np.fromstring(s, sep=" ", dtype=np.float32) for s in series]
    return np.array(parsed, dtype=np.float32)


def load_split(path_str: str, parse_blocks: bool = True) -> tuple[pd.DataFrame, np.ndarray | None]:
    """
    Load a dataset split (train.csv, test.csv, or validation.csv).
    Returns (df_tabular, blocks_array).
    Cache block arrays to disk (.npy) for ultra-fast repeated loads.
    """
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    # Check for parquet or csv
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    blocks_array = None
    if parse_blocks and "block_readings" in df.columns:
        cache_path = path.parent / f"{path.stem}_blocks.npy"
        if cache_path.exists():
            blocks_array = np.load(cache_path)
            if len(blocks_array) != len(df):
                # Invalidation check
                blocks_array = parse_block_readings_to_array(df["block_readings"])
                np.save(cache_path, blocks_array)
        else:
            blocks_array = parse_block_readings_to_array(df["block_readings"])
            np.save(cache_path, blocks_array)

        # Drop large string column from tabular dataframe to conserve memory
        df = df.drop(columns=["block_readings"])

    return df, blocks_array


def load_split_cached(path_str: str) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Convenience wrapper for cached dataset loading."""
    return load_split(path_str, parse_blocks=True)
