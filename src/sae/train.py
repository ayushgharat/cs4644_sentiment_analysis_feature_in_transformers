"""
Train the Sparse Autoencoder on Layer 3 residual stream activations.

Steps:
  1. Load the trained transformer from checkpoints/transformer_best.pt
  2. Run training data through the transformer; collect layer-3 activations
  3. Train the TopKSparseAutoencoder on those activations
  4. Report final reconstruction error and save checkpoint
"""

import json
import os
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src.config import DataConfig, SAEConfig, SAETrainingConfig, TransformerConfig
from src.data import get_lm_dataloaders, load_amazon_reviews
from src.sae.model import TopKSparseAutoencoder
from src.transformer.model import TinyTransformer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_activations(
    model: TinyTransformer,
    loader: DataLoader,
    layer_idx: int,
    device: torch.device,
    max_activations: int = 500_000,
) -> torch.Tensor:
    """
    Run batches through the transformer and collect all token-level
    activations from `layer_idx`.  Returns shape (N, d_model).
    """
    model.eval()
    chunks = []
    total = 0
    print(f"  Collecting layer-{layer_idx} activations…")

    for x, _ in loader:
        if total >= max_activations:
            break
        x = x.to(device)
        acts = model.get_layer_activations(x, layer_idx)    # (B, T, d_model)
        acts = acts.reshape(-1, acts.size(-1)).cpu()        # (B*T, d_model)
        chunks.append(acts)
        total += acts.size(0)

    activations = torch.cat(chunks, dim=0)[:max_activations]
    print(f"  Collected {activations.size(0):,} activation vectors")
    return activations


# ---------------------------------------------------------------------------
# Feature resampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def resample_dead_features(
    sae: TopKSparseAutoencoder,
    activations: torch.Tensor,
    device: torch.device,
    dead_threshold: float = 1e-4,
    scan_size: int = 8192,
) -> int:
    """
    Reinitialize dead feature encoder directions toward high-reconstruction-
    error samples. Returns number of features resampled.
    """
    freqs = sae.get_feature_activation_frequencies()
    dead_mask = freqs < dead_threshold
    n_dead = int(dead_mask.sum().item())
    if n_dead == 0:
        return 0

    # Score a random subset of activations by per-sample reconstruction error.
    indices = torch.randperm(len(activations))[:scan_size]
    sample = activations[indices].to(device)

    sae.eval()
    x_centred = sample - sae.decoder.bias
    z = sae.encode(x_centred)
    x_hat = sae.decode(z)
    per_sample_err = ((sample - x_hat) ** 2).mean(dim=-1).cpu()  # (scan_size,)
    sae.train()

    # Sample n_dead activation vectors weighted by reconstruction error.
    probs = (per_sample_err / per_sample_err.sum()).numpy()
    chosen = torch.tensor(
        __import__("numpy").random.choice(scan_size, size=n_dead, replace=True, p=probs)
    )
    candidates = activations[indices[chosen]].to(device)

    # Normalize and use as new encoder directions.
    directions = F.normalize(candidates - sae.decoder.bias, dim=-1)
    dead_idx = dead_mask.nonzero(as_tuple=True)[0]
    sae.encoder.weight.data[dead_idx] = directions
    sae.encoder.bias.data[dead_idx] = 0.0

    # Reset running stats so resampled features start fresh.
    sae.feature_counts[dead_idx] = 0.0

    return n_dead


# ---------------------------------------------------------------------------
# SAE training
# ---------------------------------------------------------------------------

def train_sae(
    num_samples: int = 50_000,
    results_dir: str = "results",
) -> dict:
    data_cfg = DataConfig(num_samples=num_samples)
    model_cfg = TransformerConfig()
    sae_cfg = SAEConfig()
    train_cfg = SAETrainingConfig()
    device = get_device()
    print(f"Device: {device}")

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load transformer
    # ------------------------------------------------------------------
    ckpt_path = os.path.join(train_cfg.checkpoint_dir, "transformer_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(train_cfg.checkpoint_dir, "transformer_latest.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "No transformer checkpoint found. Run transformer training first."
        )

    print(f"Loading transformer from {ckpt_path}…")
    transformer = TinyTransformer(model_cfg).to(device)
    transformer.load_state_dict(torch.load(ckpt_path, map_location=device))
    for p in transformer.parameters():
        p.requires_grad_(False)      # Freeze transformer
    transformer.eval()

    # ------------------------------------------------------------------
    # Collect activations
    # ------------------------------------------------------------------
    print("Loading data for activation collection…")
    train_samples, _ = load_amazon_reviews(
        num_samples=data_cfg.num_samples,
        test_split=data_cfg.test_split,
    )
    act_loader, _ = get_lm_dataloaders(
        train_samples, [],             # don't need val here
        data_cfg,
        batch_size=train_cfg.batch_size,
    )

    activations = collect_activations(
        transformer, act_loader, sae_cfg.layer_idx, device,
        max_activations=train_cfg.max_activations,
    )

    # Compute normalisation stats
    act_mean = activations.mean(0)
    act_std = activations.std(0).clamp(min=1e-8)

    act_ds = TensorDataset(activations)
    act_loader_sae = DataLoader(
        act_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # SAE
    # ------------------------------------------------------------------
    sae = TopKSparseAutoencoder(sae_cfg).to(device)
    optimizer = AdamW(sae.parameters(), lr=train_cfg.learning_rate, weight_decay=0.0)

    best_loss = float("inf")
    history = []

    for epoch in range(1, train_cfg.num_epochs + 1):
        sae.train()
        epoch_loss, epoch_steps = 0.0, 0
        t0 = time.time()

        for (batch,) in act_loader_sae:
            batch = batch.to(device)
            _, _, loss = sae(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalise_decoder()

            epoch_loss += loss.item()
            epoch_steps += 1

            if epoch_steps % train_cfg.log_every == 0:
                print(
                    f"  epoch={epoch}  step={epoch_steps}  "
                    f"recon_loss={loss.item():.6f}"
                )

        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0

        freqs = sae.get_feature_activation_frequencies().cpu()
        n_dead = int((freqs < 1e-4).sum().item())
        print(
            f"Epoch {epoch}/{train_cfg.num_epochs}  "
            f"avg_recon_loss={avg_loss:.6f}  dead={n_dead}/{sae_cfg.d_hidden}  ({elapsed:.0f}s)"
        )
        history.append({"epoch": epoch, "avg_recon_loss": avg_loss, "dead_features": n_dead})

        # Resample dead features periodically.
        if epoch % train_cfg.resample_every_n_epochs == 0 and epoch < train_cfg.num_epochs:
            n_resampled = resample_dead_features(sae, activations, device)
            if n_resampled > 0:
                print(f"  Resampled {n_resampled} dead features.")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                sae.state_dict(),
                os.path.join(train_cfg.checkpoint_dir, "sae_best.pt"),
            )

    # Save latest
    torch.save(
        sae.state_dict(),
        os.path.join(train_cfg.checkpoint_dir, "sae_latest.pt"),
    )

    # ------------------------------------------------------------------
    # Report dead features
    # ------------------------------------------------------------------
    freqs = sae.get_feature_activation_frequencies().cpu()
    dead = (freqs < 1e-4).sum().item()
    mean_freq = freqs.mean().item()

    print(f"\n--- SAE Results ---")
    print(f"  Best recon loss : {best_loss:.6f}")
    print(f"  Dead features   : {dead}/{sae_cfg.d_hidden}")
    print(f"  Mean act freq   : {mean_freq:.4f}")

    results = {
        "model": f"TopK-SAE (k={sae_cfg.k}, {sae_cfg.d_hidden} features)",
        "layer": sae_cfg.layer_idx,
        "d_hidden": sae_cfg.d_hidden,
        "k": sae_cfg.k,
        "num_activations": activations.size(0),
        "best_recon_loss": best_loss,
        "final_recon_loss": history[-1]["avg_recon_loss"] if history else None,
        "dead_features": dead,
        "mean_activation_frequency": mean_freq,
        "history": history,
    }

    out_path = os.path.join(results_dir, "sae_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    train_sae()
