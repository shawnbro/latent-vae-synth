"""
visualize_latent.py — Latent space explorer for the trained VAE

Since latent_dim=2 we can plot mu directly — no t-SNE needed.
Each point is one waveform, colored by its AKWF category (top-level folder).

Usage:
    python notebooks/visualize_latent.py --checkpoint model/checkpoints/best.pt
    python notebooks/visualize_latent.py --checkpoint model/checkpoints/best.pt --save plots/latent.png
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader

# Make sure model/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from dataset import AKWFDataset
from vae import VAE


# ---------------------------------------------------------------------------
# Encode entire dataset → latent coordinates
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_dataset(model, dataset, batch_size=512, device="cpu"):
    """
    Run every waveform through the encoder and collect (mu_x, mu_y) coords.
    Returns:
        mus:   np.ndarray (N, 2)
        names: list[str]  — relative filenames (used to extract category)
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    all_mus = []
    for batch in loader:
        batch = batch.to(device)
        mu, _ = model.encoder(batch)
        all_mus.append(mu.cpu().numpy())

    return np.concatenate(all_mus, axis=0), dataset.filenames


def extract_category(filename: str) -> str:
    """
    Pull the top-level folder name as the category label.
    e.g. 'AKWF--Akai-MPC/AKWF_0001/AKWF_0001.WAV' → 'AKWF--Akai-MPC'
    """
    parts = filename.replace("\\", "/").split("/")
    return parts[0] if parts else "unknown"


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_latent(mus, filenames, title="Latent Space", save_path=None):
    categories = [extract_category(f) for f in filenames]
    unique_cats = sorted(set(categories))
    n_cats = len(unique_cats)

    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    colors = [cat_to_idx[c] for c in categories]

    # Use a colormap with enough distinct colors
    cmap = cm.get_cmap("tab20", min(n_cats, 20))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ------------------------------------------------------------------
    # Left: all points, colored by category
    # ------------------------------------------------------------------
    ax = axes[0]
    sc = ax.scatter(
        mus[:, 0], mus[:, 1],
        c=colors,
        cmap=cmap,
        s=2,
        alpha=0.5,
        linewidths=0,
    )
    ax.set_title(f"All waveforms ({len(mus):,} points, {n_cats} categories)")
    ax.set_xlabel("z₀  (X pad)")
    ax.set_ylabel("z₁  (Y pad)")
    ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="white", linewidth=0.5, alpha=0.3)
    ax.set_facecolor("#111111")
    fig.colorbar(sc, ax=ax, label="Category index")

    # ------------------------------------------------------------------
    # Right: density heatmap — shows where the model actually lives
    # ------------------------------------------------------------------
    ax2 = axes[1]
    h = ax2.hist2d(
        mus[:, 0], mus[:, 1],
        bins=120,
        cmap="inferno",
    )
    fig.colorbar(h[3], ax=ax2, label="Count")
    ax2.set_title("Density heatmap")
    ax2.set_xlabel("z₀  (X pad)")
    ax2.set_ylabel("z₁  (Y pad)")
    ax2.set_facecolor("#111111")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Per-category plot: highlight one category at a time
# ---------------------------------------------------------------------------

def plot_category_highlights(mus, filenames, top_n=6, save_path=None):
    """
    Shows the top_n largest categories highlighted against the full cloud.
    Useful for understanding how the latent space organizes different timbres.
    """
    categories = [extract_category(f) for f in filenames]
    unique_cats = sorted(set(categories))

    # Sort categories by count, take top_n
    cat_counts = {c: categories.count(c) for c in unique_cats}
    top_cats = sorted(cat_counts, key=lambda c: -cat_counts[c])[:top_n]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Category highlights in latent space", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    mus_arr = np.array(mus)
    cats_arr = np.array(categories)

    for i, cat in enumerate(top_cats):
        ax = axes[i]
        mask = cats_arr == cat

        # Background: all points in grey
        ax.scatter(mus_arr[:, 0], mus_arr[:, 1], s=1, alpha=0.15, color="#444444", linewidths=0)
        # Foreground: this category in color
        ax.scatter(mus_arr[mask, 0], mus_arr[mask, 1], s=4, alpha=0.8, color="#ff6b35", linewidths=0)

        ax.set_title(f"{cat}\n({mask.sum()} waveforms)", fontsize=9)
        ax.set_xlabel("z₀")
        ax.set_ylabel("z₁")
        ax.set_facecolor("#111111")

    plt.tight_layout()

    if save_path:
        base, ext = os.path.splitext(save_path)
        cat_path = f"{base}_categories{ext}"
        plt.savefig(cat_path, dpi=150, bbox_inches="tight")
        print(f"Category plot saved to: {cat_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Latent space grid: decode a grid of points and show waveforms
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_decode_grid(model, mus, grid_size=6, device="cpu", save_path=None):
    """
    Sample a grid of latent points spanning the data range,
    decode each to a waveform, and display them.
    Gives a direct sense of what the XY pad will sound like.
    """
    model.eval()

    z0_min, z0_max = mus[:, 0].min(), mus[:, 0].max()
    z1_min, z1_max = mus[:, 1].min(), mus[:, 1].max()

    # Add a little padding
    pad = 0.1
    z0_range = np.linspace(z0_min - pad, z0_max + pad, grid_size)
    z1_range = np.linspace(z1_max + pad, z1_min - pad, grid_size)  # flip Y for natural orientation

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    fig.suptitle("Decoded waveforms across latent space (XY pad preview)", fontsize=12, fontweight="bold")

    for row, z1 in enumerate(z1_range):
        for col, z0 in enumerate(z0_range):
            z = torch.tensor([[z0, z1]], dtype=torch.float32).to(device)
            waveform = model.decode(z).squeeze().cpu().numpy()  # (2048,)

            ax = axes[row][col]
            ax.plot(waveform, linewidth=0.6, color="#4fc3f7")
            ax.set_ylim(-1.1, 1.1)
            ax.set_facecolor("#0d1117")
            ax.set_xticks([])
            ax.set_yticks([])

            if row == grid_size - 1:
                ax.set_xlabel(f"z₀={z0:.1f}", fontsize=6)
            if col == 0:
                ax.set_ylabel(f"z₁={z1:.1f}", fontsize=6)

    plt.tight_layout()

    if save_path:
        base, ext = os.path.splitext(save_path)
        grid_path = f"{base}_grid{ext}"
        plt.savefig(grid_path, dpi=120, bbox_inches="tight")
        print(f"Grid plot saved to: {grid_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize VAE latent space")
    parser.add_argument("--checkpoint", required=True,                  help="Path to trained checkpoint (best.pt)")
    parser.add_argument("--data",       default="data/akwf",            help="AKWF data directory")
    parser.add_argument("--cache",      default="data/akwf_cache.pt",   help="Preprocessed cache")
    parser.add_argument("--save",       default=None,                   help="Save plots to this path (e.g. plots/latent.png)")
    parser.add_argument("--grid",       action="store_true",            help="Also show decoded waveform grid")
    parser.add_argument("--batch-size", type=int, default=512,          help="Encoding batch size")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    latent_dim = ckpt["config"].get("latent_dim", 2)

    model = VAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model"])
    epoch = ckpt["epoch"]
    val_loss = ckpt["val_loss"]
    print(f"Loaded epoch {epoch}, val_loss={val_loss:.5f}")

    # Load dataset
    print("Loading dataset...")
    dataset = AKWFDataset(data_dir=args.data, cache_path=args.cache, verbose=False)

    # Encode
    print(f"Encoding {len(dataset):,} waveforms...")
    mus, filenames = encode_dataset(model, dataset, batch_size=args.batch_size, device=device)
    print(f"Latent range — z₀: [{mus[:,0].min():.2f}, {mus[:,0].max():.2f}]  "
          f"z₁: [{mus[:,1].min():.2f}, {mus[:,1].max():.2f}]")

    title = f"Latent Space — Epoch {epoch}  |  Val Loss {val_loss:.5f}"

    # Main scatter + density plot
    plot_latent(mus, filenames, title=title, save_path=args.save)

    # Category highlights
    plot_category_highlights(mus, filenames, top_n=6, save_path=args.save)

    # Optional: decoded waveform grid
    if args.grid:
        plot_decode_grid(model, mus, grid_size=6, device=device, save_path=args.save)


if __name__ == "__main__":
    main()
