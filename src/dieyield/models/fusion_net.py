import torch
import torch.nn as nn
from src.dieyield.features.block_embedding import Block1DCNNEncoder


class TabularBlockFusionNet(nn.Module):
    """
    Joint PyTorch Fusion Network concatenating tabular die+spatial features
    with 1D-CNN block signal embeddings.
    """

    def __init__(self, num_tabular_features: int, block_input_dim: int = 2000, embedding_dim: int = 16):
        super().__init__()
        self.block_encoder = Block1DCNNEncoder(input_dim=block_input_dim, embedding_dim=embedding_dim)

        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        fusion_dim = 32 + embedding_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, tabular_x: torch.Tensor, block_x: torch.Tensor) -> torch.Tensor:
        tab_feat = self.tabular_mlp(tabular_x)
        block_embed = self.block_encoder(block_x)
        fused = torch.cat([tab_feat, block_embed], dim=1)
        logits = self.fusion_head(fused)
        return logits
