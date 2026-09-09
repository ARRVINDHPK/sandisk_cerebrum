import json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGB = True
except (ImportError, OSError):
    HAS_LGB = False

from sklearn.ensemble import HistGradientBoostingClassifier
from src.dieyield.features.die_features import preprocess_die_features
from src.dieyield.features.spatial import add_spatial_features
from src.dieyield.models.imbalance import compute_scale_pos_weight, find_best_threshold
from src.dieyield.evaluate.metrics import evaluate_predictions, filter_eligible


class ModelA:
    """
    Model A: Die-Level Features + Spatial Context (No Block Data)
    Supports LightGBM with robust fallback to HistGradientBoostingClassifier.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.best_threshold = 0.5
        self.feature_names = []
        self.use_lgb = HAS_LGB and (config.get("model_type") == "lightgbm")

    def prepare_features(self, df: pd.DataFrame, is_train: bool = True) -> tuple[pd.DataFrame, list[str]]:
        """Extract spatial and die-level features."""
        df_feat = add_spatial_features(df)

        if is_train:
            df_feat, clean_die_cols = preprocess_die_features(
                df_feat,
                correlation_threshold=self.config["feature_params"].get("correlation_threshold", 0.98),
                drop_zero_var=self.config["feature_params"].get("drop_zero_variance", True)
            )
            spatial_cols = ["zone_row", "zone_col", "zone_id", "norm_radial_dist", "norm_x_coord", "norm_y_coord", "old_label_density_w5"]
            self.feature_names = [c for c in spatial_cols if c in df_feat.columns] + clean_die_cols

        return df_feat, self.feature_names

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train Model A on eligible training dies (old_label == 0)."""
        train_elig = filter_eligible(train_df)
        val_elig = filter_eligible(val_df)

        train_prep, feature_names = self.prepare_features(train_elig, is_train=True)
        val_prep, _ = self.prepare_features(val_elig, is_train=False)

        X_train = train_prep[feature_names]
        y_train = train_elig["label"].values

        X_val = val_prep[feature_names]
        y_val = val_elig["label"].values

        spw = compute_scale_pos_weight(y_train)

        if self.use_lgb:
            params = self.config["model_params"].copy()
            params.pop("objective", None)
            if self.config.get("imbalance_strategy", {}).get("use_scale_pos_weight", True):
                params["scale_pos_weight"] = spw

            self.model = lgb.LGBMClassifier(**params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
        else:
            # Fallback to HistGradientBoostingClassifier
            class_weight = {0: 1.0, 1: float(spw)} if self.config.get("imbalance_strategy", {}).get("use_scale_pos_weight", True) else "balanced"
            self.model = HistGradientBoostingClassifier(
                max_iter=self.config["model_params"].get("n_estimators", 300),
                learning_rate=self.config["model_params"].get("learning_rate", 0.05),
                max_leaf_nodes=self.config["model_params"].get("num_leaves", 31),
                max_depth=self.config["model_params"].get("max_depth", 6),
                class_weight=class_weight,
                random_state=self.config["model_params"].get("random_state", 42)
            )
            self.model.fit(X_train, y_train)

        val_probs = self.model.predict_proba(X_val)[:, 1]
        self.best_threshold, best_f1 = find_best_threshold(y_val, val_probs, metric="fail_f1")
        print(f"[Model A] Trained successfully. Optimal Validation Threshold: {self.best_threshold:.4f} (Fail F1: {best_f1:.4f})")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict fail probabilities for input dies."""
        df_prep, _ = self.prepare_features(df, is_train=False)
        X = df_prep[self.feature_names]
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict binary labels overriding pre-test fails."""
        probs = self.predict_proba(df)
        preds = (probs >= self.best_threshold).astype(int)

        if "old_label" in df.columns:
            old_fails = (df["old_label"] == 1).values
            preds[old_fails] = 1

        return preds

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Evaluate Model A on a dataset split."""
        probs = self.predict_proba(df)
        preds = self.predict(df)
        metrics = evaluate_predictions(df, preds, probs)
        metrics["model_name"] = "Model A"
        metrics["threshold"] = self.best_threshold
        return metrics

    def save(self, output_dir: str):
        """Save model metadata and checkpoint."""
        import pickle
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        with open(out_path / "model_a.pkl", "wb") as f:
            pickle.dump(self.model, f)

        meta = {
            "best_threshold": self.best_threshold,
            "feature_names": self.feature_names,
            "config": self.config
        }
        with open(out_path / "meta_a.json", "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, output_dir: str):
        """Load saved checkpoint."""
        import pickle
        in_path = Path(output_dir)
        with open(in_path / "meta_a.json", "r") as f:
            meta = json.load(f)
        self.best_threshold = meta["best_threshold"]
        self.feature_names = meta["feature_names"]
        self.config = meta["config"]

        with open(in_path / "model_a.pkl", "rb") as f:
            self.model = pickle.load(f)
