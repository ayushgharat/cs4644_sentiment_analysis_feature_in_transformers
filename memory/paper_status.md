---
name: Paper status and remaining tasks
description: What's done, what's missing, and priority order for the paper
type: project
---

Paper: "Can Sentiment Be Isolated? Sparse Decomposition of Transformer Activations"
Authors: Ayush Gharat, Rohan Bhasin, Aarav Yadav

## Three hypotheses

- **H1** Sparse Localization: ≤5% of features account for ≥80% of |Δ| polarity signal
- **H2** Causal Functionality: ablating top-k features → significant logit diff drop vs random ablation
- **H3** Sparse > Dense Subspace: sparse top-m probe beats random m-dim dense subspace probe

## Status

| Item | Status |
|---|---|
| Transformer training | ✅ Done (ppl 227.72) |
| SAE training | ✅ Done (35/1024 dead, 3.4%) |
| Baseline (TF-IDF + LR) | ✅ Done (90.2% acc / F1) |
| Linear probes on SAE latents | ✅ Done (AUROC 0.8621 mean-pool) |
| H1 concentration analysis | ✅ Done — 109/1024 (10.6%), NOT supported at ≤5% |
| H2 causal ablation | ✅ Done — SUPPORTED at all k∈[5,10,20,50], p≈0 |
| H3 sparse vs dense | ✅ Done — sparse 0.791 vs dense 0.750 at m=50, SUPPORTED |
| find_top_activating_examples | ✅ Done — examples in sentiment_feature_analysis.json |
| Figures (concentration curve, H3 sweep, ablation curve) | ❌ Not done |
| Mutual information with bootstrap CIs | ❌ Not done (low priority, can drop) |

## Remaining for paper

1. **Figures** — three needed:
   - H1: cumulative |Δ| concentration curve (data already in results JSON)
   - H3: sparse vs dense AUROC across m values (data already in results JSON)
   - H2: logit diff drop across k values (data in causal_ablation.json)
2. **MI with bootstrap CIs** — promised in Section 3.4; can simplify or drop if time-pressed
3. **Write up results section** — all numbers are in hand

## Key results for paper

- Baseline: 90.2% acc (TF-IDF + LR)
- SAE probe: 0.862 AUROC — sentiment linearly decodable from sparse features
- H1: 10.6% of features carry 80% of signal — moderate concentration, not sharp isolation
- H2: p≈0 at all k — top polarity features are causally functional (not just correlated)
- H3: sparse probe beats dense at all m — SAE induces better-aligned coordinate system
- Best single feature AUROC: 0.641 — no single "sentiment neuron", distributed encoding
- Most interesting feature: concessive positive (fires on positive words in negative reviews)
