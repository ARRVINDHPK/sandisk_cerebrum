import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd


class Block1DCNNEncoder(nn.Module):
    """
    1D-CNN Encoder for 2000-dim block readings.
    Maps (Batch, 1, 2000) -> (Batch, embedding_dim).
    """

    def __init__(self, input_dim: int = 2000, embedding_dim: int = 16):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)

        self.fc_embed = nn.Linear(64 * 8, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, 2000)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc_embed(x)
        return embedding


class BlockClassifier(nn.Module):
    """
    Auxiliary classification network wrapping Block1DCNNEncoder to train embeddings.
    """

    def __init__(self, encoder: Block1DCNNEncoder, embedding_dim: int = 16):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.encoder(x)
        logits = self.classifier(embed)
        return logits


def train_block_encoder(
    blocks_train: np.ndarray,
    labels_train: np.ndarray,
    embedding_dim: int = 16,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 0.001,
    device: str = "cpu"
) -> Block1DCNNEncoder:
    """
    Train auxiliary 1D-CNN block encoder using focal/bce loss with pos_weight.
    """
    encoder = Block1DCNNEncoder(input_dim=blocks_train.shape[1], embedding_dim=embedding_dim).to(device)
    model = BlockClassifier(encoder, embedding_dim=embedding_dim).to(device)

    X_t = torch.tensor(blocks_train, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(labels_train, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    pos_count = (labels_train == 1).sum()
    neg_count = (labels_train == 0).sum()
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    encoder.eval()
    return encoder


def extract_block_embeddings(
    encoder: Block1DCNNEncoder,
    blocks: np.ndarray,
    batch_size: int = 512,
    device: str = "cpu"
) -> pd.DataFrame:
    """
    Extract dense embedding dimensions (block_embed_0 ... block_embed_{dim-1}) from blocks array.
    """
    if blocks is None or len(blocks) == 0:
        return pd.DataFrame()

    encoder.to(device)
    encoder.eval()

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(blocks), batch_size):
            batch = torch.tensor(blocks[i : i + batch_size], dtype=torch.float32).unsqueeze(1).to(device)
            embed = encoder(batch).cpu().numpy()
            embeddings.append(embed)

    embed_matrix = np.vstack(embeddings)
    dim = embed_matrix.shape[1]
    col_names = [f"block_embed_{d}" for d in range(dim)]
    return pd.DataFrame(embed_matrix, columns=col_names)
