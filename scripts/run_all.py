"""
Run the full pipeline in order:
  1. TF-IDF + Logistic Regression baseline
  2. Transformer training          (skip with --skip-transformer)
  3. SAE training                  (skip with --skip-sae)

All results are written to the results/ directory and merged into
results/all_results.json at the end.

Usage:
  uv run python scripts/run_all.py
  uv run python scripts/run_all.py --skip-transformer --skip-sae
"""

import argparse
import json
import os
import sys

# Add project root to path so `src` is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-transformer", action="store_true")
    parser.add_argument("--skip-sae", action="store_true")
    parser.add_argument("--num-samples", type=int, default=50_000)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    # ------------------------------------------------------------------
    # 1. Baseline
    # ------------------------------------------------------------------
    if not args.skip_baseline:
        print("=" * 60)
        print("STEP 1: TF-IDF + Logistic Regression Baseline")
        print("=" * 60)
        from src.baseline import run_baseline
        baseline_results = run_baseline(
            num_samples=args.num_samples,
            results_dir=RESULTS_DIR,
        )
        all_results["baseline"] = baseline_results
        print()
    else:
        print("Skipping baseline (--skip-baseline)")

    # ------------------------------------------------------------------
    # 2. Transformer
    # ------------------------------------------------------------------
    if not args.skip_transformer:
        print("=" * 60)
        print("STEP 2: Transformer Training")
        print("=" * 60)
        from src.transformer.train import train_transformer
        transformer_results = train_transformer(
            num_samples=args.num_samples,
            results_dir=RESULTS_DIR,
        )
        all_results["transformer"] = transformer_results
        print()
    else:
        print("Skipping transformer training (--skip-transformer)")

    # ------------------------------------------------------------------
    # 3. SAE
    # ------------------------------------------------------------------
    if not args.skip_sae:
        print("=" * 60)
        print("STEP 3: SAE Training")
        print("=" * 60)
        from src.sae.train import train_sae
        sae_results = train_sae(
            num_samples=args.num_samples,
            results_dir=RESULTS_DIR,
        )
        all_results["sae"] = sae_results
        print()
    else:
        print("Skipping SAE training (--skip-sae)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    all_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(all_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "baseline" in all_results:
        b = all_results["baseline"]
        print(f"  Baseline accuracy : {b['accuracy']:.4f}")
        print(f"  Baseline F1       : {b['f1_score']:.4f}")
    if "transformer" in all_results:
        t = all_results["transformer"]
        print(f"  Val perplexity    : {t['final_val_perplexity']:.2f}")
    if "sae" in all_results:
        s = all_results["sae"]
        print(f"  SAE recon loss    : {s['best_recon_loss']:.6f}")
    print(f"\nAll results saved to {all_path}")


if __name__ == "__main__":
    main()
