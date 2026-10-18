"""
Baseline: train a single model on ALL the data (no sharding).
We need this to show that SISA doesn't lose much accuracy vs the naive approach,
and to compare retraining times (full retrain vs one shard).
"""

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def train_baseline(X_train, y_train, X_test, y_test):
    """Train one big model on the full dataset. Returns dict with model + metrics."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    print(f"\nTraining baseline on {len(X_train)} samples...")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, proba[:, 1])
    except ValueError:
        auc = float("nan")

    print(f"Baseline — acc: {acc:.4f}, auc: {auc:.4f}, time: {train_time:.2f}s")

    return {
        "model": model,
        "accuracy": acc,
        "auc": auc,
        "train_time": train_time,
    }


def full_retrain(X_train, y_train):
    """
    Simulate what you'd have to do WITHOUT SISA: retrain on everything
    just to remove one person's data. Returns (model, time).
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"Full retrain: {elapsed:.2f}s on {len(X_train)} samples")
    return model, elapsed
