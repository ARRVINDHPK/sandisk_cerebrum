import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch

try:
    import lightgbm as lgb
    HAS_LGB = True
except (ImportError, OSError):
    HAS_LGB = False

from sklearn.ensemble import HistGradientBoostingClassifier
from src.dieyield.features.die_features import preprocess_die_features
from src.dieyield.features.spatial import add_spatial_features
from src.dieyield.features.block_stats import compute_block_stats_vectorized
from src.dieyield.features.block_embedding import (
    Block1DCNNEncoder,
    train_block_encoder,
    extract_block_embeddings,
)
from src.dieyield.models.imbalance import compute_scale_pos_weight, find_best_threshold
from src.dieyield.evaluate.metrics import evaluate_predictions, filter_eligible


class ModelB:
    """
    Model B: Model A (Die + Spatial) + Block Statistics + PyTorch CNN Block Embeddings
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.encoder = None
        self.best_threshold = 0.5
        self.feature_names = []
        self.use_lgb = HAS_LGB and (config.get("model_type") == "lightgbm")

    def prepare_features(
        self, df: pd.DataFrame, blocks: np.ndarray | None = None, is_train: bool = True
    ) -> tuple[pd.DataFrame, list[str]]:
        """Extract spatial, die-level, block stats, and block embeddings."""
        df_feat = add_spatial_features(df)

        if blocks is not None:
            df_stats = compute_block_stats_vectorized(blocks)
            df_feat = pd.concat([df_feat.reset_index(drop=True), df_stats.reset_index(drop=True)], axis=1)

            if self.encoder is not None:
                df_embed = extract_block_embeddings(self.encoder, blocks)
                df_feat = pd.concat([df_feat.reset_index(drop=True), df_embed.reset_index(drop=True)], axis=1)

        if is_train:
            df_feat, clean_die_cols = preprocess_die_features(
                df_feat,
                correlation_threshold=self.config["feature_params"].get("correlation_threshold", 0.98),
                drop_zero_var=self.config["feature_params"].get("drop_zero_variance", True)
            )
            spatial_cols = ["zone_row", "zone_col", "zone_id", "norm_radial_dist", "norm_x_coord", "norm_y_coord", "old_label_density_w5"]
            block_stat_cols = [c for c in df_feat.columns if c.startswith("block_")]

            self.feature_names = (
                [c for c in spatial_cols if c in df_feat.columns]
                + clean_die_cols
                + block_stat_cols
            )

        return df_feat, self.feature_names

    def train(
        self,
        train_df: pd.DataFrame,
        train_blocks: np.ndarray,
        val_df: pd.DataFrame,
        val_blocks: np.ndarray
    ):
        """Train PyTorch block encoder + GBDT Model B on eligible dies."""
        train_elig_mask = (train_df["old_label"] == 0).values
        val_elig_mask = (val_df["old_label"] == 0).values

        train_elig_df = train_df.iloc[train_elig_mask].reset_index(drop=True)
        train_elig_blocks = train_blocks[train_elig_mask]

        val_elig_df = val_df.iloc[val_elig_mask].reset_index(drop=True)
        val_elig_blocks = val_blocks[val_elig_mask]

        # 1. Train PyTorch CNN Block Encoder
        emb_params = self.config.get("embedding_params", {})
        print("[Model B] Training PyTorch 1D-CNN Block Encoder...")
        self.encoder = train_block_encoder(
            train_elig_blocks,
            train_elig_df["label"].values,
            embedding_dim=emb_params.get("embedding_dim", 16),
            epochs=emb_params.get("epochs", 5),
            batch_size=emb_params.get("batch_size", 256),
            lr=emb_params.get("learning_rate", 0.001),
            device=emb_params.get("device", "cpu")
        )

        # 2. Prepare features
        train_prep, feature_names = self.prepare_features(train_elig_df, train_elig_blocks, is_train=True)
        val_prep, _ = self.prepare_features(val_elig_df, val_elig_blocks, is_train=False)

        X_train = train_prep[feature_names]
        y_train = train_elig_df["label"].values

        X_val = val_prep[feature_names]
        y_val = val_elig_df["label"].values

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
            class_weight = {0: 1.0, 1: float(spw)} if self.config.get("imbalance_strategy", {}).get("use_scale_pos_weight", True) else "balanced"
            self.model = HistGradientBoostingClassifier(
                max_iter=self.config["model_params"].get("n_estimators", 350),
                learning_rate=self.config["model_params"].get("learning_rate", 0.05),
                max_leaf_nodes=self.config["model_params"].get("num_leaves", 31),
                max_depth=self.config["model_params"].get("max_depth", 6),
                class_weight=class_weight,
                random_state=self.config["model_params"].get("random_state", 42)
            )
            self.model.fit(X_train, y_train)

        val_probs = self.model.predict_proba(X_val)[:, 1]
        self.best_threshold, best_f1 = find_best_threshold(y_val, val_probs, metric="fail_f1")
        print(f"[Model B] Trained successfully. Optimal Validation Threshold: {self.best_threshold:.4f} (Fail F1: {best_f1:.4f})")

    def predict_proba(self, df: pd.DataFrame, blocks: np.ndarray | None = None) -> np.ndarray:
        """Predict fail probabilities."""
        df_prep, _ = self.prepare_features(df, blocks, is_train=False)
        X = df_prep[self.feature_names]
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, blocks: np.ndarray | None = None) -> np.ndarray:
        """Predict binary labels overriding pre-test fails."""
        probs = self.predict_proba(df, blocks)
        preds = (probs >= self.best_threshold).astype(int)

        if "old_label" in df.columns:
            old_fails = (df["old_label"] == 1).values
            preds[old_fails] = 1

        return preds

    def evaluate(self, df: pd.DataFrame, blocks: np.ndarray | None = None) -> dict:
        """Evaluate Model B."""
        probs = self.predict_proba(df, blocks)
        preds = self.predict(df, blocks)
        metrics = evaluate_predictions(df, preds, probs)
        metrics["model_name"] = "Model B"
        metrics["threshold"] = self.best_threshold
        return metrics

    def save(self, output_dir: str):
        """Save model and encoder checkpoint."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        with open(out_path / "model_b.pkl", "wb") as f:
            pickle.dump(self.model, f)

        if self.encoder is not None:
            torch.save(self.encoder.state_dict(), str(out_path / "block_encoder.pt"))

        meta = {
            "best_threshold": self.best_threshold,
            "feature_names": self.feature_names,
            "config": self.config
        }
        with open(out_path / "meta_b.json", "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, output_dir: str):
        """Load saved checkpoint."""
        in_path = Path(output_dir)
        with open(in_path / "meta_b.json", "r") as f:
            meta = json.load(f)
        self.best_threshold = meta["best_threshold"]
        self.feature_names = meta["feature_names"]
        self.config = meta["config"]

        with open(in_path / "model_b.pkl", "rb") as f:
            self.model = pickle.load(f)

        encoder_path = in_path / "block_encoder.pt"
        if encoder_path.exists():
            emb_params = self.config.get("embedding_params", {})
            self.encoder = Block1DCNNEncoder(embedding_dim=emb_params.get("embedding_dim", 16))
            self.encoder.load_state_dict(torch.load(str(encoder_path), map_location="cpu"))
            self.encoder.eval()
