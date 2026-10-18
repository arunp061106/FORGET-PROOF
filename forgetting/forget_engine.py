"""
The forgetting engine — this is the heart of ForgetProof.

When someone withdraws consent:
1. Find which shard their record lives in
2. Remove it from the training data
3. Retrain ONLY that shard (not the whole model)
4. Log everything for the audit trail

The key insight from SISA: because we isolated data into shards,
the record's influence is contained to one sub-model. So we only
need to retrain ~1/N of the model instead of everything.
"""

import time
import copy
import numpy as np
import pandas as pd


def find_record_shard(record_id, train_df):
    """Look up which shard a record belongs to. Returns shard_id or None."""
    match = train_df[train_df["record_id"] == record_id]
    if len(match) == 0:
        return None
    return int(match["shard_id"].iloc[0])


def remove_record(record_id, train_df):
    """
    Remove a record from the training dataframe.
    Returns the updated df and the removed row (for logging).
    """
    mask = train_df["record_id"] == record_id
    if mask.sum() == 0:
        print(f"WARNING: record {record_id} not found in training data")
        return train_df, None

    removed = train_df[mask].iloc[0].to_dict()
    train_df = train_df[~mask].reset_index(drop=True)
    return train_df, removed


def forget(record_id, train_df, ensemble, feature_names, shard_params=None):
    """
    The main forget function. This is what gets called when a patient
    (or their parent) withdraws consent.

    Steps:
    1. Find the shard
    2. Remove the record
    3. Retrain just that shard
    4. Update the ensemble

    Returns:
        result dict with shard_id, retrain_time, old/new shard sizes, etc.
        updated train_df (record removed)
    """
    from models.shard_trainer import retrain_shard

    # step 1: which shard?
    shard_id = find_record_shard(record_id, train_df)
    if shard_id is None:
        return {"error": f"record {record_id} not found"}, train_df

    old_size = (train_df["shard_id"] == shard_id).sum()
    print(f"\n{'='*60}")
    print(f"FORGET REQUEST: record {record_id} (shard {shard_id})")
    print(f"{'='*60}")

    # step 2: remove the record
    train_df, removed = remove_record(record_id, train_df)
    new_size = (train_df["shard_id"] == shard_id).sum()
    print(f"Removed from shard {shard_id}: {old_size} → {new_size} records")

    # step 3: retrain only this shard
    new_model, retrain_time = retrain_shard(
        shard_id, train_df, feature_names, params=shard_params
    )

    # step 4: swap the model into the ensemble
    ensemble.set_shard_model(shard_id, new_model)

    result = {
        "record_id": record_id,
        "shard_id": shard_id,
        "retrain_time": retrain_time,
        "old_shard_size": old_size,
        "new_shard_size": new_size,
        "removed_record": removed,
    }

    print(f"Shard {shard_id} retrained in {retrain_time:.3f}s")
    return result, train_df


def forget_batch(record_ids, train_df, ensemble, feature_names, shard_params=None):
    """
    Process multiple forget requests. Groups by shard so we only retrain
    each affected shard once even if multiple records come from it.

    This is more realistic — in practice withdrawals arrive in batches.
    """
    results = []

    # figure out which shards are affected
    shard_records = {}
    for rid in record_ids:
        sid = find_record_shard(rid, train_df)
        if sid is not None:
            shard_records.setdefault(sid, []).append(rid)
        else:
            results.append({"record_id": rid, "error": "not found"})

    print(f"\n{len(record_ids)} forget requests affecting {len(shard_records)} shards")

    # remove all records first, then retrain each shard once
    for sid, rids in shard_records.items():
        for rid in rids:
            train_df, removed = remove_record(rid, train_df)
            results.append({
                "record_id": rid,
                "shard_id": sid,
                "removed_record": removed,
            })

    # now retrain each affected shard
    from models.shard_trainer import retrain_shard
    for sid in shard_records:
        new_model, retrain_time = retrain_shard(
            sid, train_df, feature_names, params=shard_params
        )
        ensemble.set_shard_model(sid, new_model)
        # tag the retrain time onto each result from this shard
        for r in results:
            if r.get("shard_id") == sid and "error" not in r:
                r["retrain_time"] = retrain_time

    return results, train_df
