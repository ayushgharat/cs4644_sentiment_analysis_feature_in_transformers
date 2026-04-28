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
import heapq
import json
import os
import sys
from typing import Iterable

import numpy as np
import tiktoken
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      z_mean: (N, d_hidden) SAE latents pooled by token mean
      z_max : (N, d_hidden) SAE latents pooled by token max
      h_mean: (N, d_model)  raw transformer activations pooled by token mean
      h_max : (N, d_model)  raw transformer activations pooled by token max
      y     : (N,)
    """
    z_mean_all: list[np.ndarray] = []
    z_max_all: list[np.ndarray] = []
    h_mean_all: list[np.ndarray] = []
    h_max_all: list[np.ndarray] = []
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
        h_mean = (acts * valid_mask_f).sum(dim=1) / denom

        # Max pool over valid tokens.
        neg_inf_z = torch.full_like(z_tokens, float("-inf"))
        z_tokens_masked = torch.where(valid_mask.unsqueeze(-1), z_tokens, neg_inf_z)
        z_max = z_tokens_masked.max(dim=1).values
        z_max = torch.where(torch.isfinite(z_max), z_max, torch.zeros_like(z_max))

        neg_inf_h = torch.full_like(acts, float("-inf"))
        acts_masked = torch.where(valid_mask.unsqueeze(-1), acts, neg_inf_h)
        h_max = acts_masked.max(dim=1).values
        h_max = torch.where(torch.isfinite(h_max), h_max, torch.zeros_like(h_max))

        z_mean_all.append(z_mean.cpu().numpy().astype(np.float32))
        z_max_all.append(z_max.cpu().numpy().astype(np.float32))
        h_mean_all.append(h_mean.cpu().numpy().astype(np.float32))
        h_max_all.append(h_max.cpu().numpy().astype(np.float32))
        y_all.extend(labels)

    return (
        np.concatenate(z_mean_all, axis=0),
        np.concatenate(z_max_all, axis=0),
        np.concatenate(h_mean_all, axis=0),
        np.concatenate(h_max_all, axis=0),
        np.asarray(y_all, dtype=np.int64),
    )


def vectorized_auroc(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute AUROC for all features simultaneously via rank-sum statistic."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.full(z.shape[1], 0.5, dtype=np.float32)

    # Double argsort gives ordinal ranks per feature in one vectorized pass.
    ranks = np.argsort(np.argsort(z, axis=0, kind="stable"), axis=0, kind="stable").astype(np.float64) + 1
    pos_rank_sum = ranks[y == 1].sum(axis=0)
    aurocs = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    # Zero-variance features (e.g. dead) get 0.5
    aurocs = np.where(z.var(axis=0) == 0, 0.5, aurocs)
    return np.clip(aurocs, 0, 1).astype(np.float32)


def feature_sentiment_stats(z: np.ndarray, y: np.ndarray, top_k: int) -> dict:
    pos = y == 1
    neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        raise ValueError("Need both sentiment classes for analysis.")

    mean_pos = z[pos].mean(axis=0)
    mean_neg = z[neg].mean(axis=0)
    delta = mean_pos - mean_neg

    feature_aurocs = vectorized_auroc(z, y)

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
    # lbfgs converges fast for small feature spaces; saga handles large ones.
    solver = "lbfgs" if z_train.shape[1] <= 200 else "saga"
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=5000,
                    solver=solver,
                    random_state=42,
                ),
            ),
        ]
    )
    clf.fit(z_train, y_train)
    pred = clf.predict(z_test)
    proba = clf.predict_proba(z_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_score": float(f1_score(y_test, pred, average="binary")),
        "auroc": float(roc_auc_score(y_test, proba)),
    }


@torch.no_grad()
def find_top_activating_examples(
    feature_indices: list[int],
    encoded_samples: list[tuple[list[int], int]],
    transformer: TinyTransformer,
    sae: TopKSparseAutoencoder,
    layer_idx: int,
    max_seq_len: int,
    device: torch.device,
    tokenizer: tiktoken.Encoding,
    n_examples: int = 10,
    context_window: int = 8,
    max_scan: int = 5000,
    batch_size: int = 64,
) -> dict[int, list[dict]]:
    """Batched single pass over samples collecting top activating examples per feature."""
    # Heap entries: (activation_value, counter, entry_dict). Counter breaks ties
    # so Python never falls back to comparing dicts.
    heaps: dict[int, list] = {fi: [] for fi in feature_indices}
    counter = 0
    samples_to_scan = encoded_samples[:max_scan]
    all_ids = [ids for ids, _ in samples_to_scan]
    all_labels = [label for _, label in samples_to_scan]
    pad_id = 0

    for batch_start in range(0, len(samples_to_scan), batch_size):
        batch = samples_to_scan[batch_start: batch_start + batch_size]
        bsz = len(batch)
        lengths = [min(len(ids), max_seq_len) for ids, _ in batch]

        x = torch.full((bsz, max_seq_len), pad_id, dtype=torch.long, device=device)
        for i, (ids, _) in enumerate(batch):
            n = lengths[i]
            x[i, :n] = torch.tensor(ids[:n], dtype=torch.long, device=device)

        lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
        positions = torch.arange(max_seq_len, device=device).unsqueeze(0)
        valid_mask = positions < lengths_t.unsqueeze(1)              # (B, T)

        acts = transformer.get_layer_activations(x, layer_idx)       # (B, T, d_model)
        acts_centered = acts - sae.decoder.bias.view(1, 1, -1)
        z = sae.encode(acts_centered)                                 # (B, T, d_hidden)

        for fi in feature_indices:
            feat_acts = z[:, :, fi].masked_fill(~valid_mask, float("-inf"))  # (B, T)
            max_vals, max_toks = feat_acts.max(dim=1)                        # (B,)

            for i in range(bsz):
                max_val = max_vals[i].item()
                if max_val <= 0 or not torch.isfinite(max_vals[i]):
                    continue

                max_tok = max_toks[i].item()
                sample_idx = batch_start + i
                ids = all_ids[sample_idx]
                label = all_labels[sample_idx]
                n = lengths[i]

                start = max(0, max_tok - context_window)
                end = min(n, max_tok + context_window + 1)
                entry = {
                    "activation_value": max_val,
                    "sentiment": "positive" if label == 1 else "negative",
                    "peak_token": tokenizer.decode([ids[max_tok]]),
                    "context": tokenizer.decode(ids[start:end]),
                }

                heap = heaps[fi]
                if len(heap) < n_examples:
                    heapq.heappush(heap, (max_val, counter, entry))
                elif max_val > heap[0][0]:
                    heapq.heapreplace(heap, (max_val, counter, entry))
                counter += 1

    return {
        fi: [e for _, _, e in sorted(heaps[fi], key=lambda t: -t[0])]
        for fi in feature_indices
    }


def h1_concentration_analysis(delta: np.ndarray, target_fraction: float = 0.80) -> dict:
    """
    H1: does ≤5% of features account for ≥80% of total |Δ| polarity signal?
    Returns the concentration curve and whether H1 is supported.
    """
    abs_delta = np.abs(delta)
    total = abs_delta.sum()
    if total == 0:
        return {"h1_supported": False, "error": "all deltas are zero"}

    sorted_desc = np.sort(abs_delta)[::-1]
    cumsum = np.cumsum(sorted_desc) / total

    n_features_needed = int(np.searchsorted(cumsum, target_fraction)) + 1
    fraction_of_features = n_features_needed / len(delta)

    # Sample curve sparsely for storage (dense at low counts, sparse at high)
    n = len(delta)
    sample_points = np.unique(np.concatenate([
        np.arange(1, min(201, n + 1)),
        np.linspace(200, n, 100).astype(int),
    ]))
    curve = [
        {"n_features": int(k), "cumulative_signal_fraction": float(cumsum[k - 1])}
        for k in sample_points if k <= n
    ]

    return {
        "total_features": int(n),
        "target_fraction": float(target_fraction),
        "features_needed": int(n_features_needed),
        "fraction_of_features_needed": float(fraction_of_features),
        "h1_threshold_pct": 5.0,
        "h1_supported": bool(fraction_of_features <= 0.05),
        "concentration_curve": curve,
    }


def h3_sparse_vs_dense(
    z_train: np.ndarray,
    z_test: np.ndarray,
    h_train: np.ndarray,
    h_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    delta_train: np.ndarray,
    m_values: list[int] | None = None,
    n_random_seeds: int = 5,
) -> dict:
    """
    H3: probe on top-m sparse features (by |Δ| on train split) vs random
    m-dim projection of raw transformer activations.
    """
    if m_values is None:
        m_values = [10, 20, 50, 100, 200]

    top_m_indices = np.argsort(np.abs(delta_train))[::-1]
    d_model = h_train.shape[1]
    results = []

    for m in m_values:
        if m > z_train.shape[1]:
            continue

        # Sparse: top-m SAE features selected by train-split |Δ|
        indices = top_m_indices[:m]
        sparse_metrics = train_probe(z_train[:, indices], y_train, z_test[:, indices], y_test)

        # Dense: random m-dim projection of raw transformer activations
        dense_aurocs = []
        for seed in range(n_random_seeds):
            rng = np.random.default_rng(seed)
            R = rng.standard_normal((d_model, m)).astype(np.float32)
            R /= np.linalg.norm(R, axis=0, keepdims=True)
            dense_metrics = train_probe(h_train @ R, y_train, h_test @ R, y_test)
            dense_aurocs.append(dense_metrics["auroc"])

        results.append({
            "m": m,
            "sparse_auroc": sparse_metrics["auroc"],
            "sparse_accuracy": sparse_metrics["accuracy"],
            "sparse_f1": sparse_metrics["f1_score"],
            "dense_auroc_mean": float(np.mean(dense_aurocs)),
            "dense_auroc_std": float(np.std(dense_aurocs)),
            "sparse_beats_dense": bool(sparse_metrics["auroc"] > float(np.mean(dense_aurocs))),
        })

    return {"m_sweep": results}


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
    z_train_mean, z_train_max, h_train_mean, h_train_max, y_train = pooled_sae_latents_for_samples(
        encoded_samples=train_encoded,
        transformer=transformer,
        sae=sae,
        layer_idx=layer_idx,
        max_seq_len=data_cfg.max_seq_len,
        batch_size=args.batch_size,
        device=device,
    )

    print("Extracting pooled SAE latents for test split...")
    z_test_mean, z_test_max, h_test_mean, h_test_max, y_test = pooled_sae_latents_for_samples(
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

    # Full delta on train split — used for H1 and H3 feature selection (no leakage).
    pos_mask = y_train == 1
    neg_mask = y_train == 0
    delta_train_full = (
        z_train_mean[pos_mask].mean(axis=0) - z_train_mean[neg_mask].mean(axis=0)
    )

    print("Training linear probes...")
    probe_mean = train_probe(z_train_mean, y_train, z_test_mean, y_test)
    probe_max = train_probe(z_train_max, y_train, z_test_max, y_test)

    print("Running H1 concentration analysis...")
    h1_results = h1_concentration_analysis(delta_train_full)

    print("Running H3 sparse vs dense comparison...")
    h3_results_mean = h3_sparse_vs_dense(
        z_train_mean, z_test_mean,
        h_train_mean, h_test_mean,
        y_train, y_test,
        delta_train_full,
    )
    h3_results_max = h3_sparse_vs_dense(
        z_train_max, z_test_max,
        h_train_max, h_test_max,
        y_train, y_test,
        delta_train_full,
    )

    print(f"Finding top activating examples for top-{args.top_k} features...")
    top_feature_indices = [
        f["feature_idx"] for f in stats_mean["top_auroc_features"]
    ]
    top_examples = find_top_activating_examples(
        feature_indices=top_feature_indices,
        encoded_samples=test_encoded,
        transformer=transformer,
        sae=sae,
        layer_idx=layer_idx,
        max_seq_len=data_cfg.max_seq_len,
        device=device,
        tokenizer=tokenizer,
        n_examples=10,
        context_window=8,
        max_scan=5000,
        batch_size=args.batch_size,
    )
    # Attach examples to each feature's stats entry
    top_auroc_features_with_examples = []
    for feat in stats_mean["top_auroc_features"]:
        fi = feat["feature_idx"]
        top_auroc_features_with_examples.append({
            **feat,
            "top_examples": top_examples.get(fi, []),
        })

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
                "feature_stats": {
                    **stats_mean,
                    "top_auroc_features": top_auroc_features_with_examples,
                },
                "probe_metrics": probe_mean,
            },
            "max": {
                "feature_stats": stats_max,
                "probe_metrics": probe_max,
            },
        },
        "h1_concentration": h1_results,
        "h3_sparse_vs_dense": {
            "mean_pool": h3_results_mean,
            "max_pool": h3_results_max,
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
    h1 = h1_results
    print(f"\n  H1 concentration: {h1['features_needed']} / {h1['total_features']} features "
          f"({h1['fraction_of_features_needed']*100:.1f}%) account for "
          f"{h1['target_fraction']*100:.0f}% of |Δ| signal  "
          f"→ H1 {'SUPPORTED' if h1['h1_supported'] else 'NOT supported'}")
    print(f"\n  H3 sparse vs dense (mean-pool, m=50):")
    for row in h3_results_mean["m_sweep"]:
        if row["m"] == 50:
            print(f"    sparse AUROC: {row['sparse_auroc']:.4f}  "
                  f"dense AUROC: {row['dense_auroc_mean']:.4f} ± {row['dense_auroc_std']:.4f}  "
                  f"→ sparse {'beats' if row['sparse_beats_dense'] else 'loses to'} dense")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
