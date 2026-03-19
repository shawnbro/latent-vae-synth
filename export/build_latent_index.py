"""
build_latent_index.py — Encode all AKWF waveforms and save latent coordinates.

Run once to create export/latent_index.npz, which the synth uses for
nearest-neighbour waveform name lookup at the current XY pad position.

Usage:
    python export/build_latent_index.py
    python export/build_latent_index.py --checkpoint model/checkpoints/best.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from dataset import AKWFDataset
from vae import VAE


@torch.no_grad()
def encode_dataset(model, dataset, batch_size=512, device="cpu"):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, drop_last=False, num_workers=0)
    all_mus = []
    total = len(dataset)
    done  = 0
    for batch in loader:
        mu, _ = model.encoder(batch.to(device))
        all_mus.append(mu.cpu().numpy())
        done += len(batch)
        print(f"  {done:>6,} / {total:,}", end="\r", flush=True)
    print()
    return np.concatenate(all_mus, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Build latent-space index for waveform name lookup")
    parser.add_argument("--checkpoint", default="model/checkpoints/best.pt")
    parser.add_argument("--data",       default="data/akwf")
    parser.add_argument("--cache",      default="data/akwf_cache.pt")
    parser.add_argument("--output",     default="export/latent_index.npz")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model  = VAE(latent_dim=ckpt["config"].get("latent_dim", 2)).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"  epoch {ckpt['epoch']},  val_loss = {ckpt['val_loss']:.5f}")

    print("Loading dataset...")
    dataset = AKWFDataset(data_dir=args.data, cache_path=args.cache, verbose=False)
    print(f"  {len(dataset):,} waveforms")

    print("Encoding...")
    mus       = encode_dataset(model, dataset, batch_size=args.batch_size, device=device)
    filenames = np.array(dataset.filenames, dtype=object)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, mus=mus.astype(np.float32), filenames=filenames)

    print(f"Saved {len(mus):,} entries → {args.output}")
    print(f"z₀ range: [{mus[:,0].min():.3f}, {mus[:,0].max():.3f}]  "
          f"z₁ range: [{mus[:,1].min():.3f}, {mus[:,1].max():.3f}]")


if __name__ == "__main__":
    main()
