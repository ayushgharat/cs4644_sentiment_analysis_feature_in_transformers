"""
Train the TinyTransformer on Amazon review text (next-token prediction).
Reports validation perplexity. Saves checkpoint to checkpoints/transformer.pt.
"""

import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import DataConfig, TransformerConfig, TransformerTrainingConfig
from src.data import get_lm_dataloaders, load_amazon_reviews
from src.transformer.model import TinyTransformer, count_parameters


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(
    model: TinyTransformer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, y)
        if loss is not None:
            total_loss += loss.item()
            n += 1
    model.train()
    avg_loss = total_loss / max(n, 1)
    return avg_loss


def train_transformer(
    num_samples: int = 50_000,
    results_dir: str = "results",
) -> dict:
    data_cfg = DataConfig(num_samples=num_samples)
    model_cfg = TransformerConfig()
    train_cfg = TransformerTrainingConfig()
    device = get_device()
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("Loading data…")
    train_samples, test_samples = load_amazon_reviews(
        num_samples=data_cfg.num_samples,
        test_split=data_cfg.test_split,
    )
    train_loader, val_loader = get_lm_dataloaders(
        train_samples, test_samples, data_cfg,
        batch_size=train_cfg.batch_size,
    )
    print(f"  train batches={len(train_loader)}  val batches={len(val_loader)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = TinyTransformer(model_cfg).to(device)
    print(f"  params={count_parameters(model):,}")

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * train_cfg.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    global_step = 0
    history = []

    for epoch in range(1, train_cfg.num_epochs + 1):
        model.train()
        epoch_loss, epoch_steps = 0.0, 0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Warmup
            if global_step < train_cfg.warmup_steps:
                lr_scale = (global_step + 1) / train_cfg.warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = train_cfg.learning_rate * lr_scale

            _, loss, _ = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            optimizer.step()
            if global_step >= train_cfg.warmup_steps:
                scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % train_cfg.log_every == 0:
                print(
                    f"  step={global_step}  loss={loss.item():.4f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            if global_step % train_cfg.eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                val_ppl = math.exp(val_loss)
                print(f"  [eval] val_loss={val_loss:.4f}  ppl={val_ppl:.2f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(train_cfg.checkpoint_dir, "transformer_best.pt"),
                    )

        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        val_loss = evaluate(model, val_loader, device)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{train_cfg.num_epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}  "
            f"({elapsed:.0f}s)"
        )
        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
        })

        # Always save latest
        torch.save(
            model.state_dict(),
            os.path.join(train_cfg.checkpoint_dir, "transformer_latest.pt"),
        )

    # Final eval
    final_val_loss = evaluate(model, val_loader, device)
    final_ppl = math.exp(final_val_loss)

    results = {
        "model": "TinyTransformer (4L-256d-8H)",
        "params": count_parameters(model),
        "final_val_loss": final_val_loss,
        "final_val_perplexity": final_ppl,
        "best_val_loss": best_val_loss,
        "best_val_perplexity": math.exp(best_val_loss),
        "epochs": train_cfg.num_epochs,
        "history": history,
    }

    out_path = os.path.join(results_dir, "transformer_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Final val perplexity: {final_ppl:.2f}")

    return results


if __name__ == "__main__":
    train_transformer()
