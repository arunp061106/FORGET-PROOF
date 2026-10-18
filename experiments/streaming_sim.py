"""
Streaming withdrawal simulator.

Simulates 50+ forget requests arriving over time and tracks:
- Cumulative compute time (SISA shard retrain vs full retrain)
- Accuracy drift as records are removed
- Number of shards retrained vs full retrains

This is our ORIGINAL CHART — the one that shows the real-world value
of SISA over naive retraining. With 50 requests, the savings add up
dramatically because most requests only touch one shard.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.shard_trainer import train_all_shards, retrain_shard
from models.ensemble import ShardedEnsemble
from models.baseline import full_retrain


def run_simulation(data, n_requests=50, seed=42):
    """
    Simulate a stream of forget requests and compare costs.

    For each request we:
    1. Actually do the SISA forget (retrain one shard)
    2. Estimate what a full retrain would cost (time the full model)
    3. Track accuracy after each removal

    We do the full retrain timing once at the start and scale it,
    because actually retraining 50 times would take forever.
    """
    rng = np.random.RandomState(seed)
    train_df = data["train_df"].copy()
    feature_names = data["feature_names"]
    n_shards = data["n_shards"]

    # train the initial ensemble
    print("Training initial ensemble...")
    shard_result = train_all_shards(train_df, feature_names, n_shards)
    ensemble = ShardedEnsemble(models=shard_result["models"])

    # baseline: how long does a full retrain take?
    print("Timing a full retrain for comparison...")
    _, full_retrain_time = full_retrain(data["X_train"], data["y_train"])
    print(f"Full retrain baseline: {full_retrain_time:.2f}s")

    # pick random records to "forget"
    all_ids = train_df["record_id"].values
    forget_ids = rng.choice(all_ids, size=min(n_requests, len(all_ids)),
                            replace=False)

    # tracking arrays
    log = []
    cumulative_sisa = 0.0
    cumulative_full = 0.0

    print(f"\nSimulating {len(forget_ids)} forget requests...")
    print(f"{'#':>4}  {'Record':>8}  {'Shard':>6}  {'SISA (s)':>10}  "
          f"{'Full (s)':>10}  {'Saved':>8}  {'Acc':>8}")
    print("-" * 70)

    for i, rid in enumerate(forget_ids):
        # find which shard this record is in
        match = train_df[train_df["record_id"] == rid]
        if len(match) == 0:
            continue
        shard_id = int(match["shard_id"].iloc[0])

        # remove the record
        train_df = train_df[train_df["record_id"] != rid].reset_index(drop=True)

        # retrain just the affected shard
        new_model, shard_time = retrain_shard(
            shard_id, train_df, feature_names
        )
        ensemble.set_shard_model(shard_id, new_model)

        # estimate full retrain cost
        # scale by current dataset size (smaller data = slightly faster)
        scale = len(train_df) / len(data["X_train"])
        est_full_time = full_retrain_time * scale

        cumulative_sisa += shard_time
        cumulative_full += est_full_time

        # check accuracy
        preds = ensemble.predict(data["X_test"])
        acc = np.mean(preds == data["y_test"])

        saved = 1 - (cumulative_sisa / cumulative_full) if cumulative_full > 0 else 0

        entry = {
            "request_num": i + 1,
            "record_id": int(rid),
            "shard_id": shard_id,
            "shard_retrain_time": shard_time,
            "est_full_retrain_time": est_full_time,
            "cumulative_sisa_time": cumulative_sisa,
            "cumulative_full_time": cumulative_full,
            "compute_saved_pct": saved * 100,
            "accuracy": acc,
            "remaining_records": len(train_df),
        }
        log.append(entry)

        # print every 5th request + first and last
        if i < 3 or i >= len(forget_ids) - 2 or (i + 1) % 10 == 0:
            print(f"{i+1:>4}  {rid:>8}  {shard_id:>6}  {shard_time:>10.3f}  "
                  f"{est_full_time:>10.2f}  {saved:>7.1%}  {acc:>8.4f}")

    # final summary
    print(f"\n{'='*60}")
    print(f"SIMULATION COMPLETE — {len(log)} requests processed")
    print(f"{'='*60}")
    print(f"Total SISA time:     {cumulative_sisa:.2f}s")
    print(f"Total full retrain:  {cumulative_full:.2f}s (estimated)")
    print(f"Compute saved:       {(1 - cumulative_sisa/cumulative_full)*100:.1f}%")
    print(f"Final accuracy:      {log[-1]['accuracy']:.4f}")

    return log


if __name__ == "__main__":
    from data.download import download_dataset
    from data.preprocess import prepare_data

    raw = download_dataset()
    data = prepare_data(n_shards=8, raw_df=raw)
    log = run_simulation(data, n_requests=50)
