# dl-sae-sentiment-analysis

Sparse Autoencoder for sentiment feature analysis on Amazon Reviews.

## Repo overview
- `scripts/run_all.py`: runs the full pipeline (baseline -> transformer -> SAE) and merges results into `results/all_results.json`.
- `scripts/analyze_sentiment_features.py`: evaluates whether SAE latents are associated with sentiment labels and writes `results/sentiment_feature_analysis.json`.
- `src/baseline.py`: TF-IDF + Logistic Regression sentiment baseline. Outputs `results/baseline_results.json`.
- `src/transformer/*`: a small decoder-only transformer trained for next-token prediction. Outputs `results/transformer_results.json` and saves checkpoints under `checkpoints/`.
- `src/sae/*`: a Top-K sparse autoencoder trained on layer-3 residual stream activations from the transformer. Outputs `results/sae_results.json` and saves checkpoints under `checkpoints/`.

## Requirements
- Python `>= 3.11`

## Setup
This project includes `uv.lock`, so `uv` is the simplest way to install dependencies.

```bash
uv sync --dev
```

## Run
Full pipeline:
```bash
uv run python scripts/run_all.py
```

Optional flags:
```bash
uv run python scripts/run_all.py --skip-transformer --skip-sae
uv run python scripts/run_all.py --skip-baseline
uv run python scripts/run_all.py --num-samples 20000
```

Sentiment-latent analysis (after training transformer + SAE checkpoints):
```bash
uv run python scripts/analyze_sentiment_features.py --num-samples 50000
```

Optional analysis flags:
```bash
uv run python scripts/analyze_sentiment_features.py --num-samples 50000 --layer-idx 2 --top-k 30
```

### Notes
- The transformer training and SAE training can be compute-heavy.
- SAE training expects a transformer checkpoint at `checkpoints/transformer_best.pt` (or `checkpoints/transformer_latest.pt` if the best one is missing).
- Sentiment-feature analysis expects both transformer and SAE checkpoints in `checkpoints/`.

## Outputs
- `results/`: JSON result files tracked in git (`.png`/`.pdf` excluded). Key files:
  - `baseline_results.json` — TF-IDF + LR baseline
  - `transformer_results.json` — transformer training stats
  - `sentiment_feature_analysis.json` — H1/H3 results, probe metrics, top feature examples
  - `causal_ablation.json` — H2 results across k values
- `checkpoints/`: model checkpoints saved during training (ignored by git, not needed for analysis).

## Design notes

### SAE feature count: 1024 vs 2048
The original project spec called for a 2048-feature SAE (8× expansion). After training, 66% of features were dead (never activated), which artificially inflated apparent concentration metrics for H1. We reduced to a 4× expansion (1024 features), raised `max_activations` to 2M, and added Anthropic-style dead feature resampling every 2 epochs. This brought dead features down to 3.4% (35/1024) and produced more reliable results. All reported numbers use the 1024-feature SAE.

