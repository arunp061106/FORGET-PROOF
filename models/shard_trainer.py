"""
Shard trainer — the core of the SISA approach.

Each shard gets its own RandomForestClassifier. We train them in parallel
with joblib to take full advantage of LinuxONE's multiple cores.

We intentionally use unconstrained trees (max_depth=None) so the models
can fit individual records deeply. In standard deployment you would tune
depth to avoid overfitting, but here deep trees ensure that canary records
are genuinely memorized. This creates a clear, observable before/after
gap in the membership inference attack when a record is forgotten.
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed


DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": 1,  # 1 per worker so outer joblib parallelizes across shards
}


def make_model(params=None):
    """Create a fresh shard model instance."""
    p = {**DEFAULT_PARAMS}
    if params:
        p.update(params)
    return RandomForestClassifier(**p)


def _train_one(shard_id, X, y, params):
    """Train a single shard model. Called by joblib in parallel."""
    model = make_model(params)
    t0 = time.time()
    model.fit(X, y)
    elapsed = time.time() - t0
    print(f"  shard {shard_id}: {len(X)} samples, {elapsed:.2f}s")
    return shard_id, model, elapsed


def train_all_shards(train_df, feature_names, n_shards, n_jobs=-1, params=None):
    """
    Train all shards in parallel.

    Returns a dict with:
      models     — {shard_id: model}
      times      — {shard_id: seconds}
      total_time — wall-clock time for all parallel training
      sizes      — {shard_id: n_samples}
    """
    print(f"\nTraining {n_shards} shards in parallel (n_jobs={n_jobs})...")

    shard_data = []
    sizes = {}
    for s in range(n_shards):
        mask = train_df["shard_id"] == s
        X_s = train_df.loc[mask, feature_names].values
        y_s = train_df.loc[mask, "target"].values
        shard_data.append((s, X_s, y_s))
        sizes[s] = len(X_s)

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(_train_one)(s, X, y, params)
        for s, X, y in shard_data
    )
    wall_time = time.time() - t0

    models = {}
    times = {}
    for sid, model, t in results:
        models[sid] = model
        times[sid] = t

    serial_sum = sum(times.values())
    print(f"\nAll done in {wall_time:.2f}s wall-clock")
    if wall_time > 0:
        print(f"Serial equivalent would be {serial_sum:.2f}s ({serial_sum/wall_time:.1f}x speedup)")

    return {
        "models": models,
        "times": times,
        "total_time": wall_time,
        "sizes": sizes,
    }


def retrain_shard(shard_id, train_df, feature_names, params=None):
    """
    Retrain ONE shard after a record was removed from it.
    This is where the SISA speedup comes from — we don't touch the other shards.
    """
    mask = train_df["shard_id"] == shard_id
    X = train_df.loc[mask, feature_names].values
    y = train_df.loc[mask, "target"].values

    print(f"\nRetraining shard {shard_id} ({len(X)} samples after removal)...")
    _, model, elapsed = _train_one(shard_id, X, y, params)
    return model, elapsed
