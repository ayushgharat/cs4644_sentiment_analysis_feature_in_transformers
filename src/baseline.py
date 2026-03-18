"""
Baseline: TF-IDF + Logistic Regression on Amazon Review Polarity.

Outputs to results/baseline_results.json.
"""

import json
import os
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from src.data import load_amazon_reviews


def run_baseline(
    num_samples: int = 50_000,
    test_split: float = 0.2,
    results_dir: str = "results",
) -> dict:
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("Loading Amazon Review Polarity dataset…")
    t0 = time.time()
    train_samples, test_samples = load_amazon_reviews(
        num_samples=num_samples, test_split=test_split
    )
    print(f"  train={len(train_samples)}  test={len(test_samples)}  "
          f"({time.time()-t0:.1f}s)")

    X_train = [s["text"] for s in train_samples]
    y_train = [s["label"] for s in train_samples]
    X_test = [s["text"] for s in test_samples]
    y_test = [s["label"] for s in test_samples]

    # ------------------------------------------------------------------
    # TF-IDF vectorisation
    # ------------------------------------------------------------------
    print("Fitting TF-IDF vectorizer…")
    t1 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"  vocab={len(vectorizer.vocabulary_)}  ({time.time()-t1:.1f}s)")

    # ------------------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------------------
    print("Training Logistic Regression…")
    t2 = time.time()
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
    )
    clf.fit(X_train_tfidf, y_train)
    print(f"  done  ({time.time()-t2:.1f}s)")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    y_pred = clf.predict(X_test_tfidf)

    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="binary"))
    cm = confusion_matrix(y_test, y_pred).tolist()

    # ------------------------------------------------------------------
    # Top features by LR coefficient (positive class = label 1 = positive)
    # ------------------------------------------------------------------
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]                          # shape: (n_features,)
    top_n = 20
    top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
    top_neg_idx = np.argsort(coefs)[:top_n]

    top_positive_features = [
        {"feature": feature_names[i], "coefficient": float(coefs[i])}
        for i in top_pos_idx
    ]
    top_negative_features = [
        {"feature": feature_names[i], "coefficient": float(coefs[i])}
        for i in top_neg_idx
    ]

    print(f"\n--- Baseline Results ---")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    {cm[0]}")
    print(f"    {cm[1]}")

    print(f"\n  Top {top_n} positive features (→ positive sentiment):")
    for r in top_positive_features:
        print(f"    {r['feature']:<30s}  {r['coefficient']:+.4f}")

    print(f"\n  Top {top_n} negative features (→ negative sentiment):")
    for r in top_negative_features:
        print(f"    {r['feature']:<30s}  {r['coefficient']:+.4f}")

    results = {
        "model": "TF-IDF + Logistic Regression",
        "num_train": len(train_samples),
        "num_test": len(test_samples),
        "tfidf_vocab_size": len(vectorizer.vocabulary_),
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": cm,
        "label_map": {"0": "negative", "1": "positive"},
        "top_positive_features": top_positive_features,
        "top_negative_features": top_negative_features,
    }

    out_path = os.path.join(results_dir, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    run_baseline()
