from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients
from src.dieyield.features.block_embedding import Block1DCNNEncoder, BlockClassifier


def render_block_anomaly_grid(
    block_signal: np.ndarray,
    z_threshold: float = 2.5,
    output_path: str | None = None
) -> plt.Figure:
    """
    Reshape 2000 block readings into a 2D grid (~45x45) and highlight anomalous outlier regions.
    """
    K = len(block_signal)
    grid_side = int(np.ceil(np.sqrt(K)))
    pad_len = grid_side * grid_side - K

    mean_val = float(np.mean(block_signal))
    std_val = float(np.std(block_signal))
    std_safe = std_val if std_val > 1e-8 else 1e-8

    padded_signal = np.pad(block_signal, (0, pad_len), mode="constant", constant_values=mean_val) if pad_len > 0 else block_signal
    grid_vals = padded_signal.reshape(grid_side, grid_side)

    z_grid = np.abs((grid_vals - mean_val) / std_safe)
    anomalous_mask = z_grid > z_threshold

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Signal grid
    im0 = axes[0].imshow(grid_vals, cmap="plasma")
    axes[0].set_title(f"Sub-Die Block Readings (Mean: {mean_val:.1f}, Std: {std_val:.1f})", fontweight="bold")
    plt.colorbar(im0, ax=axes[0], label="Reading Value")

    # Anomaly Heatmap
    im1 = axes[1].imshow(anomalous_mask, cmap="Reds")
    axes[1].set_title(f"Outlier Anomaly Map (|Z| > {z_threshold})", fontweight="bold")
    plt.colorbar(im1, ax=axes[1], label="1 = Anomaly Spike")

    for ax in axes:
        ax.set_xlabel("Sub-Block X")
        ax.set_ylabel("Sub-Block Y")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)

    return fig


def compute_captum_integrated_gradients(
    encoder: Block1DCNNEncoder,
    block_signal: np.ndarray,
    device: str = "cpu"
) -> np.ndarray:
    """
    Compute Integrated Gradients feature attributions over raw 2000-length sequence using Captum.
    """
    classifier = BlockClassifier(encoder).to(device)
    classifier.eval()

    ig = IntegratedGradients(classifier)
    input_tensor = torch.tensor(block_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    baseline = torch.zeros_like(input_tensor).to(device)

    attributions, _ = ig.attribute(input_tensor, baseline, target=0, return_convergence_delta=True)
    attr_np = attributions.squeeze().cpu().detach().numpy()
    return attr_np
