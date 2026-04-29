"""
H2 Causal Ablation: does zeroing top polarity-aligned SAE features
cause a larger drop in sentiment-token log-probability than zeroing
an equal number of random features?

For each test review:
  1. Run through transformer -> SAE -> sparse latents z (T, d_hidden)
  2. Full SAE reconstruction as baseline (controls for SAE reconstruction error)
  3. Zero top-k polarity-aligned features in z, reconstruct, propagate -> delta_targeted
  4. Zero k random features n_random_trials times -> delta_random
  5. Paired t-test: targeted < random (one-sided) -> H2 supported if p < 0.05

Usage:
  uv run python scripts/causal_ablation.py
  uv run python scripts/causal_ablation.py --n-ablation-samples 500 --k-values 5 10 20 50
"""

import argparse
import json
import os
import sys

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DataConfig, SAEConfig
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
    from src.config import TransformerConfig
    model = TinyTransformer(TransformerConfig()).to(device)
    ckpt = os.path.join(checkpoint_dir, "transformer_best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(checkpoint_dir, "transformer_latest.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def load_sae(device: torch.device, checkpoint_dir: str) -> tuple[TopKSparseAutoencoder, str]:
    sae = TopKSparseAutoencoder(SAEConfig()).to(device)
    ckpt = os.path.join(checkpoint_dir, "sae_best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(checkpoint_dir, "sae_latest.pt")
    sae.load_state_dict(torch.load(ckpt, map_location=device))
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae, ckpt


@torch.no_grad()
def forward_from_layer(
    transformer: TinyTransformer,
    patched_acts: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Continue forward pass from layer_idx+1 to logits. (B, T, d_model) -> (B, T, vocab)"""
    x = patched_acts
    for i in range(layer_idx + 1, len(transformer.blocks)):
        x = transformer.blocks[i](x, transformer.causal_mask)
    x = transformer.norm_f(x)
    return transformer.lm_head(x)


def get_sentiment_token_ids(tokenizer: tiktoken.Encoding) -> tuple[list[int], list[int]]:
    """Return single-token IDs for positive and negative sentiment words."""
    pos_words = [
        "great", "excellent", "wonderful", "perfect", "amazing", "fantastic",
        "brilliant", "outstanding", "loved", "superb", " great", " excellent",
        " wonderful", " perfect", " amazing", " fantastic", " brilliant",
        " outstanding", " loved", " superb",
    ]
    neg_words = [
        "terrible", "awful", "horrible", "disappointing", "worst", "boring",
        "useless", "pathetic", "dreadful", "poor", " terrible", " awful",
        " horrible", " disappointing", " worst", " boring", " useless",
        " pathetic", " dreadful", " poor",
    ]

    def single_ids(words: list[str]) -> list[int]:
        ids: set[int] = set()
        for w in words:
            toks = tokenizer.encode(w, disallowed_special=())
            if len(toks) == 1:
                ids.add(toks[0])
        return sorted(ids)

    return single_ids(pos_words), single_ids(neg_words)


def encode_samples(
    samples: list[dict],
    max_seq_len: int,
    tokenizer: tiktoken.Encoding,
) -> list[tuple[list[int], int]]:
    encoded = []
    for s in samples:
        ids = tokenizer.encode(s["text"], disallowed_special=())
        if len(ids) < 2:
            continue
        ids = ids[: max_seq_len + 1]
        input_ids = ids[:-1]
        if input_ids:
            encoded.append((input_ids, int(s["label"])))
    return encoded


@torch.no_grad()
def compute_delta(
    encoded_samples: list[tuple[list[int], int]],
    transformer: TinyTransformer,
    sae: TopKSparseAutoencoder,
    layer_idx: int,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
    max_samples: int = 5000,
) -> np.ndarray:
    """Compute mean(z|pos) - mean(z|neg) (mean-pooled over tokens) on a training subset."""
    d = sae.config.d_hidden
    z_pos_sum = np.zeros(d, dtype=np.float64)
    z_neg_sum = np.zeros(d, dtype=np.float64)
    n_pos = n_neg = 0
    pad_id = 0

    for i in range(0, min(max_samples, len(encoded_samples)), batch_size):
        batch = encoded_samples[i: i + batch_size]
        bsz = len(batch)
        lengths = [min(len(ids), max_seq_len) for ids, _ in batch]

        x = torch.full((bsz, max_seq_len), pad_id, dtype=torch.long, device=device)
        for j, (ids, _) in enumerate(batch):
            x[j, : lengths[j]] = torch.tensor(ids[: lengths[j]], dtype=torch.long, device=device)

        lengths_t = torch.tensor(lengths, device=device).unsqueeze(1)
        valid_mask = (torch.arange(max_seq_len, device=device).unsqueeze(0) < lengths_t).unsqueeze(-1).float()

        acts = transformer.get_layer_activations(x, layer_idx)
        acts_centered = acts - sae.decoder.bias.view(1, 1, -1)
        z = sae.encode(acts_centered)

        z_mean = ((z * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)).cpu().numpy()

        for j, (_, label) in enumerate(batch):
            if label == 1:
                z_pos_sum += z_mean[j]
                n_pos += 1
            else:
                z_neg_sum += z_mean[j]
                n_neg += 1

    if n_pos == 0 or n_neg == 0:
        return np.zeros(d, dtype=np.float32)
    return (z_pos_sum / n_pos - z_neg_sum / n_neg).astype(np.float32)


@torch.no_grad()
def ablate_sample(
    ids: list[int],
    label: int,
    transformer: TinyTransformer,
    sae: TopKSparseAutoencoder,
    layer_idx: int,
    max_seq_len: int,
    device: torch.device,
    pos_features: np.ndarray,
    neg_features: np.ndarray,
    pos_token_ids: torch.Tensor,
    neg_token_ids: torch.Tensor,
    k_values: list[int],
    n_random_trials: int,
    rng: np.random.Generator,
) -> dict[int, dict]:
    """
    Run all ablations for one sample. Returns {k: {targeted_delta, random_deltas}}.
    All ablations for a given sample are batched into a single forward pass.
    """
    n = min(len(ids), max_seq_len)
    x = torch.tensor(ids[:n], dtype=torch.long, device=device).unsqueeze(0)

    acts = transformer.get_layer_activations(x, layer_idx)       # (1, n, d_model)
    z = sae.encode(acts - sae.decoder.bias.view(1, 1, -1)).squeeze(0)  # (n, d_hidden)

    sentiment_ids = pos_token_ids if label == 1 else neg_token_ids
    if len(sentiment_ids) == 0:
        return {}

    def lp_sentiment(z_batch: torch.Tensor) -> torch.Tensor:
        """z_batch: (B, n, d_hidden) -> (B,) mean log-prob over sentiment tokens at last pos."""
        logits = forward_from_layer(transformer, sae.decode(z_batch), layer_idx)
        lp = F.log_softmax(logits[:, -1, :], dim=-1)
        return lp[:, sentiment_ids].mean(dim=-1)

    # Baseline: full reconstruction, no ablation
    lp_base = lp_sentiment(z.unsqueeze(0)).item()

    feature_indices = pos_features if label == 1 else neg_features
    d_hidden = z.shape[-1]

    # Build one big batch: [baseline reconstruction already computed above]
    # one targeted + n_random_trials random, for each k
    ablated_list: list[torch.Tensor] = []
    meta: list[tuple[int, str]] = []  # (k, "targeted"|"random")

    for k in k_values:
        z_t = z.clone()
        z_t[:, feature_indices[:k]] = 0.0
        ablated_list.append(z_t)
        meta.append((k, "targeted"))

        for _ in range(n_random_trials):
            z_r = z.clone()
            z_r[:, rng.choice(d_hidden, k, replace=False)] = 0.0
            ablated_list.append(z_r)
            meta.append((k, "random"))

    lp_all = lp_sentiment(torch.stack(ablated_list)).cpu().numpy()

    results: dict[int, dict] = {}
    for (k, kind), lp in zip(meta, lp_all):
        if k not in results:
            results[k] = {"targeted_delta": None, "random_deltas": []}
        delta = float(lp) - lp_base
        if kind == "targeted":
            results[k]["targeted_delta"] = delta
        else:
            results[k]["random_deltas"].append(delta)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--n-ablation-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--n-random-trials", type=int, default=10)
    parser.add_argument("--delta-samples", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    sae_cfg = SAEConfig()
    data_cfg = DataConfig(num_samples=args.num_samples)

    transformer, t_ckpt = load_transformer(device, args.checkpoint_dir)
    sae, s_ckpt = load_sae(device, args.checkpoint_dir)
    print(f"Loaded transformer: {t_ckpt}")
    print(f"Loaded SAE: {s_ckpt}")
    print(f"Using layer index: {sae_cfg.layer_idx}")

    tokenizer = tiktoken.get_encoding("cl100k_base")
    pos_ids, neg_ids = get_sentiment_token_ids(tokenizer)
    print(f"Sentiment tokens — positive: {len(pos_ids)}  negative: {len(neg_ids)}")
    pos_ids_t = torch.tensor(pos_ids, dtype=torch.long, device=device)
    neg_ids_t = torch.tensor(neg_ids, dtype=torch.long, device=device)

    print("Loading samples...")
    train_samples, test_samples = load_amazon_reviews(
        num_samples=data_cfg.num_samples,
        test_split=data_cfg.test_split,
    )
    train_encoded = encode_samples(train_samples, data_cfg.max_seq_len, tokenizer)
    test_encoded  = encode_samples(test_samples,  data_cfg.max_seq_len, tokenizer)
    print(f"Encoded: train={len(train_encoded)}  test={len(test_encoded)}")

    # Compute delta on training split to select features (no leakage)
    print(f"Computing delta on {args.delta_samples} training samples...")
    delta = compute_delta(
        train_encoded, transformer, sae, sae_cfg.layer_idx,
        data_cfg.max_seq_len, args.batch_size, device,
        max_samples=args.delta_samples,
    )
    pos_features = np.argsort(delta)[::-1].copy()   # highest Δ → most positive
    neg_features = np.argsort(delta).copy()          # lowest Δ → most negative
    print(f"Top positive feature (Δ={delta[pos_features[0]]:.4f}): idx {pos_features[0]}")
    print(f"Top negative feature (Δ={delta[neg_features[0]]:.4f}): idx {neg_features[0]}")

    # Ablation loop
    n_samples = min(args.n_ablation_samples, len(test_encoded))
    print(f"\nRunning ablation on {n_samples} test samples  k={args.k_values}  trials={args.n_random_trials}")
    rng = np.random.default_rng(42)
    agg: dict[int, dict] = {k: {"targeted": [], "random_mean": []} for k in args.k_values}

    for i, (ids, label) in enumerate(test_encoded[:n_samples]):
        if i % 200 == 0:
            print(f"  {i}/{n_samples}...")
        res = ablate_sample(
            ids=ids, label=label,
            transformer=transformer, sae=sae,
            layer_idx=sae_cfg.layer_idx,
            max_seq_len=data_cfg.max_seq_len,
            device=device,
            pos_features=pos_features, neg_features=neg_features,
            pos_token_ids=pos_ids_t, neg_token_ids=neg_ids_t,
            k_values=args.k_values,
            n_random_trials=args.n_random_trials,
            rng=rng,
        )
        for k, r in res.items():
            if r["targeted_delta"] is not None:
                agg[k]["targeted"].append(r["targeted_delta"])
                agg[k]["random_mean"].append(float(np.mean(r["random_deltas"])))

    # Statistical summary
    print("\n--- H2 Causal Ablation Results ---")
    k_results = []
    for k in args.k_values:
        targeted    = np.array(agg[k]["targeted"])
        random_mean = np.array(agg[k]["random_mean"])
        diff = targeted - random_mean  # negative → targeted hurts more

        t_stat, p_value = stats.ttest_rel(targeted, random_mean, alternative="less")
        supported = bool(diff.mean() < 0 and p_value < 0.05)

        entry = {
            "k": k,
            "n_samples": int(len(targeted)),
            "targeted_delta_mean": float(targeted.mean()),
            "targeted_delta_std":  float(targeted.std()),
            "random_delta_mean":   float(random_mean.mean()),
            "random_delta_std":    float(random_mean.std()),
            "difference_mean":     float(diff.mean()),
            "t_statistic":         float(t_stat),
            "p_value":             float(p_value),
            "h2_supported":        supported,
        }
        k_results.append(entry)
        print(
            f"  k={k:3d}: targeted={entry['targeted_delta_mean']:+.4f}  "
            f"random={entry['random_delta_mean']:+.4f}  "
            f"diff={entry['difference_mean']:+.4f}  "
            f"p={entry['p_value']:.4f}  "
            f"→ {'SUPPORTED' if supported else 'not supported'}"
        )

    out = {
        "analysis": "H2 causal ablation",
        "transformer_checkpoint": t_ckpt,
        "sae_checkpoint": s_ckpt,
        "layer_idx": sae_cfg.layer_idx,
        "n_ablation_samples": n_samples,
        "n_random_trials": args.n_random_trials,
        "n_pos_sentiment_tokens": len(pos_ids),
        "n_neg_sentiment_tokens": len(neg_ids),
        "k_results": k_results,
    }
    out_path = os.path.join(args.results_dir, "causal_ablation.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
