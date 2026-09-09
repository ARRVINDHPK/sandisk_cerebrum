import numpy as np
import pytest
from src.dieyield.features.block_stats import compute_block_stats_vectorized


def test_block_stats_shapes_and_values():
    # 5 dies, 2000 blocks each
    blocks = np.random.normal(100.0, 15.0, size=(5, 2000)).astype(np.float32)
    # Inject outlier spikes in die 0
    blocks[0, 10:20] += 100.0

    df_stats = compute_block_stats_vectorized(blocks)

    assert len(df_stats) == 5
    assert "block_mean" in df_stats.columns
    assert "block_outliers_z3" in df_stats.columns
    assert "block_max_z" in df_stats.columns
    assert "block_cluster_score" in df_stats.columns

    # Die 0 should have z3 outliers > 0
    assert df_stats.loc[0, "block_outliers_z3"] > 0
    assert df_stats.loc[0, "block_max_z"] > 5.0
