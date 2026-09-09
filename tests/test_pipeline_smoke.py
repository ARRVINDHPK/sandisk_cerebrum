import numpy as np
import pandas as pd
import pytest
import yaml

from src.dieyield.models.model_a import ModelA
from src.dieyield.models.model_b import ModelB
from src.dieyield.evaluate.metrics import evaluate_predictions


def test_model_a_smoke_run():
    # 20 synthetic dies, 2 wafers
    df = pd.DataFrame({
        "wafer_id": ["W1"] * 10 + ["W2"] * 10,
        "die_row": list(range(10)) * 2,
        "die_col": list(range(10)) * 2,
        "feature_1": np.random.randn(20),
        "feature_2": np.random.randn(20),
        "old_label": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] * 2,
        "label":     [0, 0, 0, 1, 1, 0, 0, 0, 1, 0] * 2,
    })

    with open("configs/model_a.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["model_params"]["n_estimators"] = 10

    model = ModelA(cfg)
    model.train(df.iloc[:10], df.iloc[10:])

    preds = model.predict(df)
    assert len(preds) == 20
    assert (preds[df["old_label"] == 1] == 1).all()

    res = evaluate_predictions(df, preds)
    assert "overall_accuracy" in res
    assert "fail_f1" in res


def test_model_b_smoke_run():
    df = pd.DataFrame({
        "wafer_id": ["W1"] * 10 + ["W2"] * 10,
        "die_row": list(range(10)) * 2,
        "die_col": list(range(10)) * 2,
        "feature_1": np.random.randn(20),
        "feature_2": np.random.randn(20),
        "old_label": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] * 2,
        "label":     [0, 0, 0, 1, 1, 0, 0, 0, 1, 0] * 2,
    })
    blocks = np.random.normal(100.0, 15.0, size=(20, 2000)).astype(np.float32)

    with open("configs/model_b.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["model_params"]["n_estimators"] = 10
    cfg["embedding_params"]["epochs"] = 1

    model = ModelB(cfg)
    model.train(df.iloc[:10], blocks[:10], df.iloc[10:], blocks[10:])

    preds = model.predict(df, blocks)
    assert len(preds) == 20
    res = evaluate_predictions(df, preds)
    assert "overall_accuracy" in res
