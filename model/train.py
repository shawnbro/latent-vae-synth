"""
train.py — Training loop for the Latent VAE

Usage:
    python model/train.py --data data/akwf --epochs 100
    python model/train.py --data data/akwf --epochs 100 --beta 2.0 --resume model/checkpoints/epoch_050.pt

Checkpoints are saved to model/checkpoints/.
"""

import os
import csv
import argparse
import torch
from torch.utils.data import DataLoader, random_split

from dataset import AKWFDataset
from vae import VAE, vae_loss


# ---------------------------------------------------------------------------
# β annealing
# ---------------------------------------------------------------------------

def get_beta(epoch: int, warmup_epochs: int, target_beta: float) -> float:
    """
    Linearly ramp β from 0 → target_beta over warmup_epochs.

    Starting at β=0 lets the model first learn to reconstruct before the KL
    term pushes the latent space toward a standard normal. This prevents
    posterior collapse, where the decoder ignores z and KL → 0.
    """
    if warmup_epochs == 0:
        return target_beta
    return min(target_beta, target_beta * epoch / warmup_epochs)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, beta, device, train=True):
    model.train(train)
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum    = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)

            recon, mu, logvar = model(batch)
            total, recon_l, kl_l = vae_loss(recon, batch, mu, logvar, beta)

            if train:
                optimizer.zero_grad()
                total.backward()
                # Gradient clipping — prevents occasional large gradient spikes
                # that can destabilize training on diverse waveform shapes
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss_sum += total.item()
            recon_loss_sum += recon_l.item()
            kl_loss_sum    += kl_l.item()

    n = len(loader)
    return total_loss_sum / n, recon_loss_sum / n, kl_loss_sum / n


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, epoch, config, val_loss):
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "val_loss":   val_loss,
        "config":     config,
    }, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.5f})")
    return ckpt["epoch"], ckpt["val_loss"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Latent VAE on AKWF waveforms")
    parser.add_argument("--data",        default="data/akwf",                help="Path to AKWF .wav directory")
    parser.add_argument("--cache",       default="data/akwf_cache.pt",       help="Preprocessed tensor cache path")
    parser.add_argument("--checkpoint-dir", default="model/checkpoints",     help="Where to save checkpoints")
    parser.add_argument("--epochs",      type=int,   default=100,            help="Total training epochs")
    parser.add_argument("--batch-size",  type=int,   default=256,            help="Batch size")
    parser.add_argument("--lr",          type=float, default=1e-3,           help="Learning rate")
    parser.add_argument("--latent-dim",  type=int,   default=2,              help="Latent space dimensionality")
    parser.add_argument("--beta",        type=float, default=1.0,            help="Target KL weight (β)")
    parser.add_argument("--warmup",      type=int,   default=20,             help="Epochs to ramp β from 0 → target")
    parser.add_argument("--val-split",   type=float, default=0.1,            help="Fraction of data for validation")
    parser.add_argument("--save-every",  type=int,   default=10,             help="Save periodic checkpoint every N epochs")
    parser.add_argument("--resume",      default=None,                       help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Config dict stored in checkpoints for reproducibility
    config = vars(args)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Dataset + DataLoader
    # ------------------------------------------------------------------
    print("Loading dataset...")
    dataset = AKWFDataset(data_dir=args.data, cache_path=args.cache, verbose=True)

    val_size   = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=(device.type != "mps"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=(device.type != "mps"),
    )

    print(f"Train: {train_size} | Val: {val_size}")

    # ------------------------------------------------------------------
    # Model + optimizer
    # ------------------------------------------------------------------
    model     = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # LR scheduler: reduce on plateau if val loss stalls
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, device
        )
        start_epoch += 1  # continue from next epoch

    # ------------------------------------------------------------------
    # CSV log
    # ------------------------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, "train_log.csv")
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if start_epoch == 1:
        log_writer.writerow(["epoch", "beta", "train_total", "train_recon", "train_kl",
                              "val_total", "val_recon", "val_kl", "lr"])

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\nStarting training for {args.epochs} epochs (β warmup over {args.warmup} epochs)\n")
    print(f"{'Epoch':>6}  {'β':>5}  {'Train':>8}  {'Recon':>8}  {'KL':>7}  {'Val':>8}  {'Best':>8}")
    print("-" * 65)

    for epoch in range(start_epoch, args.epochs + 1):
        beta = get_beta(epoch, args.warmup, args.beta)

        # Train
        train_total, train_recon, train_kl = run_epoch(
            model, train_loader, optimizer, beta, device, train=True
        )

        # Validate
        val_total, val_recon, val_kl = run_epoch(
            model, val_loader, optimizer, beta, device, train=False
        )

        scheduler.step(val_total)
        current_lr = optimizer.param_groups[0]["lr"]

        # Track best
        is_best = val_total < best_val_loss
        if is_best:
            best_val_loss = val_total
            save_checkpoint(
                os.path.join(args.checkpoint_dir, "best.pt"),
                model, optimizer, epoch, config, best_val_loss,
            )

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f"epoch_{epoch:04d}.pt"),
                model, optimizer, epoch, config, val_total,
            )

        # Log to CSV
        log_writer.writerow([
            epoch, f"{beta:.3f}",
            f"{train_total:.5f}", f"{train_recon:.5f}", f"{train_kl:.5f}",
            f"{val_total:.5f}", f"{val_recon:.5f}", f"{val_kl:.5f}",
            f"{current_lr:.2e}",
        ])
        log_file.flush()

        # Print
        best_marker = "*" if is_best else " "
        print(
            f"{epoch:>6}  {beta:>5.2f}  {train_total:>8.5f}  {train_recon:>8.5f}"
            f"  {train_kl:>7.4f}  {val_total:>8.5f}  {best_val_loss:>8.5f} {best_marker}"
        )

    log_file.close()
    print(f"\nDone. Best val loss: {best_val_loss:.5f}")
    print(f"Best model saved to: {os.path.join(args.checkpoint_dir, 'best.pt')}")
    print(f"Training log:        {log_path}")


if __name__ == "__main__":
    main()
