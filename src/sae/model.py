"""
Top-K Sparse Autoencoder.

- Encoder: linear → top-k activation (exactly k features per sample)
- Decoder: linear (L2-normalised columns)
- Loss: MSE reconstruction

Adapted from the Latent-Scope project's TopKSparseAutoencoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import SAEConfig


class TopKSparseAutoencoder(nn.Module):
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        d_in = config.d_input
        d_hid = config.d_hidden

        self.encoder = nn.Linear(d_in, d_hid, bias=True)
        self.decoder = nn.Linear(d_hid, d_in, bias=True)

        # Normalise decoder columns at init
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

        # Running stats for analysis
        self.register_buffer("feature_counts", torch.zeros(d_hid))
        self.register_buffer("total_samples", torch.tensor(0, dtype=torch.long))

    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return sparse latents with exactly k non-zero entries per sample."""
        z = self.encoder(x)                            # (B, d_hid)
        top_vals, top_idx = torch.topk(z, self.config.k, dim=-1)
        sparse = torch.zeros_like(z)
        sparse.scatter_(-1, top_idx, F.relu(top_vals))
        return sparse

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (x_hat, z, loss)."""
        # Centre input by decoder bias
        x_centred = x - self.decoder.bias
        z = self.encode(x_centred)
        x_hat = self.decode(z)
        loss = F.mse_loss(x_hat, x)

        # Update running stats (no grad)
        with torch.no_grad():
            active = (z > 0).float()
            self.feature_counts += active.sum(0)
            self.total_samples += x.size(0)

        return x_hat, z, loss

    # ------------------------------------------------------------------
    # Analysis helpers

    def get_feature_activation_frequencies(self) -> torch.Tensor:
        """Fraction of samples each feature was active in."""
        n = max(self.total_samples.item(), 1)
        return self.feature_counts / n

    def get_dead_features(self, threshold: float = 1e-4) -> torch.Tensor:
        freqs = self.get_feature_activation_frequencies()
        return (freqs < threshold).nonzero(as_tuple=True)[0]

    def get_feature_decoder_vectors(self) -> torch.Tensor:
        """Normalised decoder directions for each feature. Shape: (d_hid, d_in)."""
        return F.normalize(self.decoder.weight.T, dim=-1)

    @torch.no_grad()
    def normalise_decoder(self):
        """Re-normalise decoder columns (call after each optimiser step)."""
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
