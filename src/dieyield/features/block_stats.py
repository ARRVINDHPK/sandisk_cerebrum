import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.ndimage import uniform_filter


def compute_block_stats_vectorized(blocks: np.ndarray) -> pd.DataFrame:
    """
    Extract summary statistics from 2000-dim block readings (N x K float32 array).
    Targeting sparse defect signals (|z|>2, |z|>3 outlier counts, cluster scores).
    """
    if blocks is None or len(blocks) == 0:
        return pd.DataFrame()

    N, K = blocks.shape
    
    means = np.mean(blocks, axis=1)
    stds = np.std(blocks, axis=1)
    stds_safe = np.where(stds < 1e-8, 1e-8, stds)
    
    mins = np.min(blocks, axis=1)
    maxs = np.max(blocks, axis=1)
    
    p10 = np.percentile(blocks, 10, axis=1)
    p25 = np.percentile(blocks, 25, axis=1)
    p50 = np.percentile(blocks, 50, axis=1)
    p75 = np.percentile(blocks, 75, axis=1)
    p90 = np.percentile(blocks, 90, axis=1)
    p99 = np.percentile(blocks, 99, axis=1)
    
    skews = skew(blocks, axis=1)
    kurts = kurtosis(blocks, axis=1)

    # Z-score outlier counts
    z_scores = np.abs((blocks - means[:, None]) / stds_safe[:, None])
    z_outliers_2 = np.sum(z_scores > 2.0, axis=1)
    z_outliers_3 = np.sum(z_scores > 3.0, axis=1)
    max_z = np.max(z_scores, axis=1)
    max_z_pos = np.argmax(z_scores, axis=1) / float(K)

    # Spatial cluster score within die (reshape to ~45x45 grid)
    grid_side = int(np.ceil(np.sqrt(K)))
    pad_len = grid_side * grid_side - K
    
    cluster_scores = np.zeros(N, dtype=np.float32)
    for i in range(N):
        arr = blocks[i]
        if pad_len > 0:
            arr = np.pad(arr, (0, pad_len), mode="constant", constant_values=means[i])
        grid = arr.reshape(grid_side, grid_side)
        grid_z = np.abs((grid - means[i]) / stds_safe[i])
        smoothed = uniform_filter(grid_z, size=3, mode="constant", cval=0.0)
        cluster_scores[i] = np.max(smoothed)

    df_stats = pd.DataFrame({
        "block_mean": means,
        "block_std": stds,
        "block_min": mins,
        "block_max": maxs,
        "block_p10": p10,
        "block_p25": p25,
        "block_p50": p50,
        "block_p75": p75,
        "block_p90": p90,
        "block_p99": p99,
        "block_skew": skews,
        "block_kurt": kurts,
        "block_outliers_z2": z_outliers_2,
        "block_outliers_z3": z_outliers_3,
        "block_max_z": max_z,
        "block_max_z_pos": max_z_pos,
        "block_cluster_score": cluster_scores,
    })
    
    return df_stats
