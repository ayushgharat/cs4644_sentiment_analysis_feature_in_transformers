"""
Analyze sentiment association of SAE latent features.

This script:
  1) Loads trained transformer + SAE checkpoints
  2) Encodes Amazon review samples into review-level SAE latents
  3) Computes per-feature sentiment association metrics
  4) Trains lightweight linear probes on pooled SAE latents
  5) Saves analysis results to results/sentiment_feature_analysis.json

Usage:
  uv run python scripts/analyze_sentiment_features.py
  uv run python scripts/analyze_sentiment_features.py --num-samples 50000 --top-k 30
"""

import argparse
import json
import os
import sys
from typing import Iterable

import numpy as np
import tiktoken
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Add project root to path so `src` is importable when run as script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DataConfig, SAEConfig, TransformerConfig
from src.data import load_amazon_reviews
from src.sae.model import TopKSparseAutoencoder
from src.transformer.model import TinyTransformer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_transformer(device: torch.device, checkpoint_dir: str) -> tuple[TinyTransformer, str]:
    model_cfg = TransformerConfig()
    model = TinyTransformer(model_cfg).to(device)

    ckpt_path = os.path.join(checkpoint_dir, "transformer_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, "transformer_latest.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("No transformer checkpoint found.")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt_path


def load_sae(device: torch.device, checkpoint_dir: str) -> tuple[TopKSparseAutoencoder, str]:
    sae_cfg = SAEConfig()
    sae = TopKSparseAutoencoder(sae_cfg).to(device)

    ckpt_path = os.path.join(checkpoint_dir, "sae_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, "sae_latest.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("No SAE checkpoint found.")

    sae.load_state_dict(torch.load(ckpt_path, map_location=device))
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae, ckpt_path


def build_encoded_samples(
    samples: list[dict],
    max_seq_len: int,
    tokenizer: tiktoken.Encoding,
) -> list[tuple[list[int], int]]:
    encoded: list[tuple[list[int], int]] = []
    for s in samples:
        ids = tokenizer.encode(s["text"], disallowed_special=())
        if len(ids) < 2:
            continue
        ids = ids[: max_seq_len + 1]
        input_ids = ids[:-1]
        if len(input_ids) == 0:
            continue
        encoded.append((input_ids, int(s["label"])))
    return encoded


def batched(items: list[tuple[list[int], int]], batch_size: int) -> Iterable[list[tuple[list[int], int]]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


@torch.no_grad()
def pooled_sae_latents_for_samples(
    encoded_samples: list[tuple[list[int], int]],
    transformer: TinyTransformer,
    sae: TopKSparseAutoencoder,
    layer_idx: int,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      z_mean: (N, d_hidden) pooled by token mean
      z_max : (N, d_hidden) pooled by token max
      y     : (N,)
    """
    z_mean_all: list[np.ndarray] = []
    z_max_all: list[np.ndarray] = []
    y_all: list[int] = []

    pad_id = 0
    for batch in batched(encoded_samples, batch_size):
        bsz = len(batch)
        x = torch.full((bsz, max_seq_len), pad_id, dtype=torch.long, device=device)
        lengths = torch.zeros(bsz, dtype=torch.long, device=device)
        labels = []

        for i, (ids, label) in enumerate(batch):
            n = min(len(ids), max_seq_len)
            x[i, :n] = torch.tensor(ids[:n], dtype=torch.long, device=device)
            lengths[i] = n
            labels.append(label)

        acts = transformer.get_layer_activations(x, layer_idx)  # (B, T, d_model)

        # Match SAE training path: centre by decoder bias before encode().
        acts_centered = acts - sae.decoder.bias.view(1, 1, -1)
        z_tokens = sae.encode(acts_centered)                    # (B, T, d_hidden)

        positions = torch.arange(max_seq_len, device=device).unsqueeze(0)
        valid_mask = positions < lengths.unsqueeze(1)           # (B, T)
        valid_mask_f = valid_mask.unsqueeze(-1).float()         # (B, T, 1)

        # Mean pool over valid tokens.
        denom = valid_mask_f.sum(dim=1).clamp(min=1.0)
        z_mean = (z_tokens * valid_mask_f).sum(dim=1) / denom

        # Max pool over valid tokens.
        neg_inf = torch.full_like(z_tokens, float("-inf"))
        z_tokens_masked = torch.where(valid_mask.unsqueeze(-1), z_tokens, neg_inf)
        z_max = z_tokens_masked.max(dim=1).values
        z_max = torch.where(torch.isfinite(z_max), z_max, torch.zeros_like(z_max))

        z_mean_all.append(z_mean.cpu().numpy().astype(np.float32))
        z_max_all.append(z_max.cpu().numpy().astype(np.float32))
        y_all.extend(labels)

    return (
        np.concatenate(z_mean_all, axis=0),
        np.concatenate(z_max_all, axis=0),
        np.asarray(y_all, dtype=np.int64),
    )


def safe_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if np.all(y == y[0]):
        return 0.5
    if np.all(scores == scores[0]):
        return 0.5
    return float(roc_auc_score(y, scores))


def feature_sentiment_stats(z: np.ndarray, y: np.ndarray, top_k: int) -> dict:
    pos = y == 1
    neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        raise ValueError("Need both sentiment classes for analysis.")

    mean_pos = z[pos].mean(axis=0)
    mean_neg = z[neg].mean(axis=0)
    delta = mean_pos - mean_neg

    feature_aurocs = np.array([safe_auroc(y, z[:, j]) for j in range(z.shape[1])], dtype=np.float32)

    order_pos = np.argsort(delta)[::-1]
    order_neg = np.argsort(delta)
    order_auc = np.argsort(feature_aurocs)[::-1]

    def rows(indices: np.ndarray) -> list[dict]:
        out = []
        for j in indices[:top_k]:
            out.append(
                {
                    "feature_idx": int(j),
                    "mean_positive": float(mean_pos[j]),
                    "mean_negative": float(mean_neg[j]),
                    "delta_pos_minus_neg": float(delta[j]),
                    "auroc": float(feature_aurocs[j]),
                }
            )
        return out

    return {
        "n_features": int(z.shape[1]),
        "top_positive_features": rows(order_pos),
        "top_negative_features": rows(order_neg),
        "top_auroc_features": rows(order_auc),
        "global_stats": {
            "mean_abs_delta": float(np.abs(delta).mean()),
            "max_abs_delta": float(np.abs(delta).max()),
            "mean_feature_auroc": float(feature_aurocs.mean()),
            "max_feature_auroc": float(feature_aurocs.max()),
        },
    }


def train_probe(z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray, y_test: np.ndarray) -> dict:
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(z_train, y_train)
    pred = clf.predict(z_test)
    proba = clf.predict_proba(z_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_score": float(f1_score(y_test, pred, average="binary")),
        "auroc": float(roc_auc_score(y_test, proba)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=None,
        help="Layer to extract activations from (defaults to SAEConfig.layer_idx).",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    data_cfg = DataConfig(num_samples=args.num_samples)
    sae_cfg = SAEConfig()
    layer_idx = sae_cfg.layer_idx if args.layer_idx is None else args.layer_idx

    transformer, transformer_ckpt = load_transformer(device, args.checkpoint_dir)
    sae, sae_ckpt = load_sae(device, args.checkpoint_dir)
    print(f"Loaded transformer: {transformer_ckpt}")
    print(f"Loaded SAE: {sae_ckpt}")
    print(f"Using layer index: {layer_idx}")

    print("Loading samples...")
    train_samples, test_samples = load_amazon_reviews(
        num_samples=data_cfg.num_samples,
        test_split=data_cfg.test_split,
    )

    tokenizer = tiktoken.get_encoding("cl100k_base")
    train_encoded = build_encoded_samples(train_samples, data_cfg.max_seq_len, tokenizer)
    test_encoded = build_encoded_samples(test_samples, data_cfg.max_seq_len, tokenizer)
    print(f"Encoded samples: train={len(train_encoded)}  test={len(test_encoded)}")

    print("Extracting pooled SAE latents for train split...")
    z_train_mean, z_train_max, y_train = pooled_sae_latents_for_samples(
        encoded_samples=train_encoded,
        transformer=transformer,
        sae=sae,
        layer_idx=layer_idx,
        max_seq_len=data_cfg.max_seq_len,
        batch_size=args.batch_size,
        device=device,
    )

    print("Extracting pooled SAE latents for test split...")
    z_test_mean, z_test_max, y_test = pooled_sae_latents_for_samples(
        encoded_samples=test_encoded,
        transformer=transformer,
        sae=sae,
        layer_idx=layer_idx,
        max_seq_len=data_cfg.max_seq_len,
        batch_size=args.batch_size,
        device=device,
    )

    print("Computing feature-level sentiment association metrics...")
    stats_mean = feature_sentiment_stats(z_test_mean, y_test, top_k=args.top_k)
    stats_max = feature_sentiment_stats(z_test_max, y_test, top_k=args.top_k)

    print("Training linear probes...")
    probe_mean = train_probe(z_train_mean, y_train, z_test_mean, y_test)
    probe_max = train_probe(z_train_max, y_train, z_test_max, y_test)

    results = {
        "analysis": "SAE latent sentiment association",
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "max_seq_len": data_cfg.max_seq_len,
        "layer_idx": layer_idx,
        "checkpoint_dir": args.checkpoint_dir,
        "transformer_checkpoint": transformer_ckpt,
        "sae_checkpoint": sae_ckpt,
        "n_train_reviews": int(len(y_train)),
        "n_test_reviews": int(len(y_test)),
        "pooling": {
            "mean": {
                "feature_stats": stats_mean,
                "probe_metrics": probe_mean,
            },
            "max": {
                "feature_stats": stats_max,
                "probe_metrics": probe_max,
            },
        },
    }

    out_path = os.path.join(args.results_dir, "sentiment_feature_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n--- Sentiment feature analysis summary ---")
    print(f"  Probe (mean-pool)  AUROC: {probe_mean['auroc']:.4f}")
    print(f"  Probe (max-pool)   AUROC: {probe_max['auroc']:.4f}")
    print(f"  Top single-feature AUROC (mean): {stats_mean['global_stats']['max_feature_auroc']:.4f}")
    print(f"  Top single-feature AUROC (max) : {stats_max['global_stats']['max_feature_auroc']:.4f}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
