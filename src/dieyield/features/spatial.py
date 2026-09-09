import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter


def add_zone_id(df: pd.DataFrame, zone_rows: int = 4, zone_cols: int = 4) -> pd.DataFrame:
    """
    Bucket dies on each wafer into zone_rows x zone_cols grid.
    Adds 'zone_row', 'zone_col', and 'zone_id' columns.
    """
    df = df.copy()
    
    def process_wafer(wdf):
        min_r, max_r = wdf["die_row"].min(), wdf["die_row"].max()
        min_c, max_c = wdf["die_col"].min(), wdf["die_col"].max()
        
        r_span = max(max_r - min_r, 1)
        c_span = max(max_c - min_c, 1)
        
        z_r = np.clip(((wdf["die_row"] - min_r) / r_span * zone_rows).astype(int), 0, zone_rows - 1)
        z_c = np.clip(((wdf["die_col"] - min_c) / c_span * zone_cols).astype(int), 0, zone_cols - 1)
        
        z_id = z_r * zone_cols + z_c
        return z_r, z_c, z_id

    z_r_all, z_c_all, z_id_all = [], [], []
    for _, wdf in df.groupby("wafer_id", sort=False):
        zr, zc, zid = process_wafer(wdf)
        z_r_all.append(zr)
        z_c_all.append(zc)
        z_id_all.append(zid)
        
    df["zone_row"] = pd.concat(z_r_all).values
    df["zone_col"] = pd.concat(z_c_all).values
    df["zone_id"] = pd.concat(z_id_all).values
    return df


def add_radial_linear_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalized radial distance from wafer center and x/y linear coordinates.
    Adds 'norm_radial_dist', 'norm_x_coord', 'norm_y_coord'.
    """
    df = df.copy()
    
    def process_wafer(wdf):
        cy = wdf["die_row"].mean()
        cx = wdf["die_col"].mean()
        
        y_diff = wdf["die_row"] - cy
        x_diff = wdf["die_col"] - cx
        
        dist = np.sqrt(y_diff**2 + x_diff**2)
        max_dist = dist.max() if dist.max() > 0 else 1.0
        
        norm_r = dist / max_dist
        max_span_y = np.abs(y_diff).max() if np.abs(y_diff).max() > 0 else 1.0
        max_span_x = np.abs(x_diff).max() if np.abs(x_diff).max() > 0 else 1.0
        
        norm_y = y_diff / max_span_y
        norm_x = x_diff / max_span_x
        return norm_r, norm_x, norm_y

    r_list, x_list, y_list = [], [], []
    for _, wdf in df.groupby("wafer_id", sort=False):
        r, x, y = process_wafer(wdf)
        r_list.append(r)
        x_list.append(x)
        y_list.append(y)
        
    df["norm_radial_dist"] = pd.concat(r_list).values
    df["norm_x_coord"] = pd.concat(x_list).values
    df["norm_y_coord"] = pd.concat(y_list).values
    return df


def add_old_label_neighborhood_density(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """
    Compute pre-test fail density in an m x m neighborhood window.
    STRICT NO-LEAKAGE REQUIREMENT: Uses old_label ONLY! Never touches post-test target status.
    """
    if "old_label" not in df.columns:
        raise KeyError("DataFrame must contain 'old_label' for spatial neighborhood density.")

    df = df.copy()
    densities = []

    for _, wdf in df.groupby("wafer_id", sort=False):
        max_r = wdf["die_row"].max() + 1
        max_c = wdf["die_col"].max() + 1
        
        grid_fail = np.zeros((max_r, max_c), dtype=float)
        grid_exist = np.zeros((max_r, max_c), dtype=float)
        
        rows = wdf["die_row"].values
        cols = wdf["die_col"].values
        old_lbls = wdf["old_label"].values
        
        grid_fail[rows, cols] = (old_lbls == 1).astype(float)
        grid_exist[rows, cols] = 1.0
        
        fail_sum = uniform_filter(grid_fail, size=window_size, mode="constant", cval=0.0)
        exist_sum = uniform_filter(grid_exist, size=window_size, mode="constant", cval=0.0)
        
        with np.errstate(divide="ignore", invalid="ignore"):
            dens_grid = np.where(exist_sum > 0, fail_sum / exist_sum, 0.0)
            
        die_dens = dens_grid[rows, cols]
        densities.append(pd.Series(die_dens, index=wdf.index))

    df[f"old_label_density_w{window_size}"] = pd.concat(densities).sort_index().values
    return df


def add_spatial_features(df: pd.DataFrame, zone_rows: int = 4, zone_cols: int = 4, window_size: int = 5) -> pd.DataFrame:
    """
    Convenience wrapper to extract all zero-leakage spatial features.
    """
    df = add_zone_id(df, zone_rows, zone_cols)
    df = add_radial_linear_position(df)
    df = add_old_label_neighborhood_density(df, window_size)
    return df
