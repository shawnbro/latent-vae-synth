"""
vae.py — Variational Autoencoder for single-cycle waveforms

Architecture:
  Encoder: 1D strided convolutions → (mu, logvar) of shape (batch, latent_dim)
  Decoder: Linear → 1D transposed convolutions → waveform (batch, 1, 2048)

At runtime only the Decoder is used: a 2D latent point goes in, a waveform comes out.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

WAVEFORM_LENGTH = 2048


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def conv_block(in_ch, out_ch, stride=4):
    """Strided conv → BatchNorm → LeakyReLU (encoder step)."""
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=8, stride=stride, padding=2),
        nn.BatchNorm1d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def deconv_block(in_ch, out_ch, stride=4):
    """Transposed conv → BatchNorm → LeakyReLU (decoder step)."""
    return nn.Sequential(
        nn.ConvTranspose1d(in_ch, out_ch, kernel_size=8, stride=stride, padding=2),
        nn.BatchNorm1d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Maps a waveform (batch, 1, 2048) to (mu, logvar) each of shape (batch, latent_dim).

    Downsampling path (stride=4 each step):
      (1, 2048) → (32, 512) → (64, 128) → (128, 32) → (256, 8)
    Then flatten → two linear heads for mu and logvar.
    """

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            conv_block(1,   32),   # (batch, 32,  512)
            conv_block(32,  64),   # (batch, 64,  128)
            conv_block(64,  128),  # (batch, 128,  32)
            conv_block(128, 256),  # (batch, 256,   8)
        )

        # 256 channels × 8 time steps
        self.flat_dim = 256 * 8

        self.fc_mu     = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        """x: (batch, 1, 2048) → mu, logvar: (batch, latent_dim)"""
        h = self.conv(x)               # (batch, 256, 8)
        h = h.flatten(start_dim=1)     # (batch, 2048)
        return self.fc_mu(h), self.fc_logvar(h)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Maps a latent vector (batch, latent_dim) to a waveform (batch, 1, 2048).

    Upsampling path (stride=4 each step):
      Linear → (256, 8) → (128, 32) → (64, 128) → (32, 512) → (1, 2048)
    Final activation: Tanh → output in [-1, 1].

    This is the only module needed at runtime / ONNX export.
    """

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.flat_dim   = 256 * 8

        self.fc = nn.Linear(latent_dim, self.flat_dim)

        self.deconv = nn.Sequential(
            deconv_block(256, 128),  # (batch, 128,  32)
            deconv_block(128, 64),   # (batch, 64,  128)
            deconv_block(64,  32),   # (batch, 32,  512)
            # Final layer: no BatchNorm, Tanh instead of LeakyReLU
            nn.ConvTranspose1d(32, 1, kernel_size=8, stride=4, padding=2),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, latent_dim) → waveform: (batch, 1, 2048)"""
        h = self.fc(z)                          # (batch, 2048)
        h = h.view(-1, 256, 8)                  # (batch, 256, 8)
        out = self.deconv(h)                    # (batch, 1, ~2048)

        # Ensure exact output length — transposed convs can be off by ±1
        if out.shape[-1] != WAVEFORM_LENGTH:
            out = F.interpolate(out, size=WAVEFORM_LENGTH, mode="linear", align_corners=False)

        return out


# ---------------------------------------------------------------------------
# Full VAE (encoder + decoder, used during training)
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    """
    Full VAE: encodes a waveform to (mu, logvar), samples z via the
    reparameterization trick, then decodes z back to a waveform.

    Args:
        latent_dim: dimensionality of the latent space (default 2 for XY pad)
    """

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample z = mu + eps * std  where eps ~ N(0, 1).

        During inference (eval mode) we skip the noise and return mu directly,
        giving deterministic, clean output for a given latent point.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor):
        """
        x: (batch, 1, 2048)
        Returns: recon (batch, 1, 2048), mu (batch, latent_dim), logvar (batch, latent_dim)
        """
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decoder(z)
        return recon, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Convenience method: decode a latent point directly."""
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def vae_loss(
    recon:  torch.Tensor,
    target: torch.Tensor,
    mu:     torch.Tensor,
    logvar: torch.Tensor,
    beta:   float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined VAE loss = reconstruction loss + β * KL divergence.

    Reconstruction: MSE averaged over all elements (per-sample, per-timestep).
    KL divergence:  closed-form for Gaussian vs standard normal,
                    summed over latent dims, averaged over batch.

    Args:
        recon:   reconstructed waveform  (batch, 1, 2048)
        target:  original waveform       (batch, 1, 2048)
        mu:      latent mean             (batch, latent_dim)
        logvar:  latent log variance     (batch, latent_dim)
        beta:    KL weight (β > 1 encourages smoother latent space)

    Returns:
        total_loss, recon_loss, kl_loss  — all scalar tensors
    """
    recon_loss = F.mse_loss(recon, target, reduction="mean")

    # KL divergence: -0.5 * sum(1 + logvar - mu² - exp(logvar))
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    batch = 8
    latent_dim = 2
    x = torch.randn(batch, 1, WAVEFORM_LENGTH)

    model = VAE(latent_dim=latent_dim)
    model.train()

    recon, mu, logvar = model(x)

    print(f"Input:   {x.shape}")
    print(f"Recon:   {recon.shape}")
    print(f"mu:      {mu.shape}")
    print(f"logvar:  {logvar.shape}")

    total, recon_l, kl_l = vae_loss(recon, x, mu, logvar, beta=1.0)
    print(f"\nLoss breakdown:")
    print(f"  Recon:  {recon_l.item():.4f}")
    print(f"  KL:     {kl_l.item():.4f}")
    print(f"  Total:  {total.item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    enc_params   = sum(p.numel() for p in model.encoder.parameters())
    dec_params   = sum(p.numel() for p in model.decoder.parameters())
    print(f"\nParameters:")
    print(f"  Encoder: {enc_params:,}")
    print(f"  Decoder: {dec_params:,}")
    print(f"  Total:   {total_params:,}")

    # Verify decoder alone works (runtime path)
    model.eval()
    z = torch.tensor([[0.0, 0.0]])  # center of latent space
    with torch.no_grad():
        waveform = model.decode(z)
    print(f"\nDecoder-only output shape: {waveform.shape}")
    print(f"Output range: [{waveform.min():.3f}, {waveform.max():.3f}]")
