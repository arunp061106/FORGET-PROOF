"""
Shard sweep experiment: train with N=2,4,8,16 shards and compare
accuracy, AUC, and training time against the baseline.

This produces two key charts:
1. Accuracy vs shard count (the privacy-utility tradeoff)
2. Training time vs shard count
"""

import sys
import os
import time
import numpy as np

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import prepare_data
from models.shard_trainer import train_all_shards
from models.ensemble import ShardedEnsemble
from models.baseline import train_baseline


def run_sweep(raw_df=None, shard_counts=[2, 4, 8, 16]):
    """Run the full shard count sweep. Returns results for plotting."""

    results = []

    # first, get baseline
    data = prepare_data(n_shards=2, raw_df=raw_df)  # n_shards doesn't matter for baseline
    baseline = train_baseline(
        data["X_train"], data["y_train"],
        data["X_test"], data["y_test"]
    )

    print(f"\nBaseline: acc={baseline['accuracy']:.4f}, time={baseline['train_time']:.2f}s")
    results.append({
        "n_shards": 1,
        "accuracy": baseline["accuracy"],
        "auc": baseline["auc"],
        "train_time": baseline["train_time"],
        "label": "baseline",
    })

    # now try each shard count
    for n in shard_counts:
        print(f"\n{'='*50}")
        print(f"Testing N={n} shards")
        print(f"{'='*50}")

        data = prepare_data(n_shards=n, raw_df=raw_df)
        shard_result = train_all_shards(
            data["train_df"], data["feature_names"],
            n_shards=n, n_jobs=-1
        )

        ensemble = ShardedEnsemble(models=shard_result["models"])
        metrics = ensemble.evaluate(data["X_test"], data["y_test"],
                                    label=f"N={n}")

        results.append({
            "n_shards": n,
            "accuracy": metrics["accuracy"],
            "auc": metrics["auc"],
            "train_time": shard_result["total_time"],
            "shard_accs": metrics["shard_accuracies"],
            "label": f"N={n}",
        })

    # print summary table
    print(f"\n{'='*60}")
    print(f"SHARD SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"{'Shards':>8}  {'Accuracy':>10}  {'AUC':>10}  {'Time (s)':>10}")
    print(f"{'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
    for r in results:
        print(f"{r['n_shards']:>8}  {r['accuracy']:>10.4f}  "
              f"{r['auc']:>10.4f}  {r['train_time']:>10.2f}")

    return results


if __name__ == "__main__":
    from data.download import download_dataset
    raw = download_dataset()
    results = run_sweep(raw_df=raw)
