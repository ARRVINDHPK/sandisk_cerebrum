import inspect
import pandas as pd
import pytest
import src.dieyield.features.spatial as spatial_module
from src.dieyield.features.spatial import add_spatial_features


def test_spatial_code_static_check():
    """Verify that spatial.py source code never references the target column 'label' directly."""
    source_code = inspect.getsource(spatial_module)
    # Check that 'label' (except old_label) is not referenced in spatial.py
    # Remove 'old_label' occurrences first
    cleaned_code = source_code.replace("old_label", "PRE_TEST_STATUS")
    assert '"label"' not in cleaned_code, "spatial.py contains reference to 'label'"
    assert "'label'" not in cleaned_code, "spatial.py contains reference to 'label'"


def test_spatial_features_runtime_no_label():
    """Assert that running spatial feature extraction when 'label' column is dropped raises no error."""
    df = pd.DataFrame({
        "wafer_id": ["W1", "W1", "W1", "W1"],
        "die_row":  [0, 0, 1, 1],
        "die_col":  [0, 1, 0, 1],
        "old_label": [0, 1, 0, 0],
        "label":    [0, 1, 1, 0]
    })
    
    df_no_label = df.drop(columns=["label"])
    
    df_feat = add_spatial_features(df_no_label, zone_rows=2, zone_cols=2, window_size=3)
    
    assert "old_label_density_w3" in df_feat.columns
    assert "norm_radial_dist" in df_feat.columns
    assert "zone_id" in df_feat.columns
