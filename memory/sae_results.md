---
name: SAE training results and fixes
description: SAE dead feature problem, fixes applied, current checkpoint stats, all hypothesis results
type: project
---

## Current best SAE checkpoint (sae_best.pt)

- Architecture: TopK-SAE, 4× expansion (1024 features), k=32, layer 3
- Dead features: 35 / 1024 (3.4%) — down from 1361/2048 (66%) before fix
- Mean activation frequency: 0.0312 (= 32/1024, features using full capacity)
- Best recon loss: 0.355 (down from 0.439)

**Fix applied:** Reduced expansion_factor 8×→4×, raised max_activations 500k→2M, added Anthropic-style feature resampling every 2 epochs (reinitialize dead encoder directions toward high-loss samples).

## Transformer checkpoint (transformer_best.pt)

- 4-layer, 256-dim, 8-head decoder-only transformer
- Trained on 50k Amazon reviews, next-token prediction, no sentiment labels
- Final val perplexity: 227.72 (on cl100k_base 100k vocab)

## Canonical sentiment analysis results (1024-feature SAE)

- Probe AUROC (mean-pool): 0.8621
- Probe AUROC (max-pool): 0.8546
- Top single-feature AUROC: 0.6407 (mean) / 0.6565 (max)
- TF-IDF + LR baseline: 90.2% accuracy / 90.2% F1

## H1 — Sparse Localization: NOT SUPPORTED (honest result)

- 109/1024 features (10.6%) account for 80% of |Δ| signal
- Threshold was ≤5% (51 features) — not met
- Key insight: old 2048-feature SAE showed 5.6% but 66% dead features inflated apparent concentration
- 10.6% is the correct result — sentiment is moderately localized, not sharply isolated
- Paper framing: reframe as "moderate concentration" rather than binary pass/fail

## H2 — Causal Functionality: SUPPORTED (p≈0 at all k)

Metric: logit difference = mean_logit(pos_tokens) - mean_logit(neg_tokens) at last position.
Baseline: full SAE reconstruction. Random ablation control has Δ≈0.

| k | Targeted Δ | Random Δ | Difference | p-value |
|---|---|---|---|---|
| 5  | -0.0621 | 0.0000 | -0.0621 | ≈0 |
| 10 | -0.0588 | +0.0001 | -0.0589 | ≈0 |
| 20 | -0.0290 | -0.0002 | -0.0289 | ≈0 |
| 50 | -0.0511 | +0.0003 | -0.0514 | ≈0 |

Non-monotonic with k: k=5,10 strongest; k=20 weakest; k=50 rebounds. Suggests a core of ~5-10 highly functional polarity features.

Note: first attempt used raw log-prob at last position — caused spurious positive delta (entropy effect). Fixed by using logit difference which cancels entropy changes.

## H3 — Sparse > Dense Subspace: SUPPORTED

- m=50: sparse AUROC 0.7906 vs dense 0.7500 ± 0.006 → sparse beats dense
- Gap widened vs old unhealthy SAE (was 0.025, now 0.041)
- Full m-sweep in results/sentiment_feature_analysis.json

## Notable features (top by AUROC, from analyze_sentiment_features.py)

- Feature 885: Top positive feature by Δ (Δ=1.3704) — identified by causal ablation
- Feature 671: Top negative feature by Δ (Δ=-1.5774) — identified by causal ablation
- Feature 1152 (old SAE): Superlative adjectives ('Best', 'Perfect') — cleanest single-feature
- Feature 1812 (old SAE): Concessive positive — fires on 'good'/'excellent' in negative reviews
