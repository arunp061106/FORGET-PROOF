"""
Canary records — synthetic patients we plant into the training data
to test whether the model memorizes individual records.

The idea: if the model has memorized a canary, a membership inference
attack will flag it as "definitely in the training set" with high
confidence. After we forget it and retrain, the attack should drop
to roughly chance level.

We make canaries stand out by giving them random/noisy labels and
slightly unusual feature values, so the model has to "work harder"
to fit them — which means more memorization.
"""

import numpy as np
import pandas as pd


def create_canaries(train_df, feature_names, n_canaries=20, seed=42):
    """
    Generate synthetic canary records.

    Strategy:
    - Start from real records (so feature distributions look plausible)
    - Perturb some features by 1-2 std devs
    - Flip the label randomly (this forces memorization)
    - Give them unique record_ids in a high range so they don't collide

    Returns:
        canary_df: DataFrame ready to be appended to train_df
        canary_ids: list of the canary record_ids
    """
    rng = np.random.RandomState(seed)

    # sample real records as templates
    templates = train_df.sample(n=n_canaries, random_state=seed).copy()
    templates = templates.reset_index(drop=True)

    # give them unique IDs that won't clash
    max_id = train_df["record_id"].max()
    canary_ids = list(range(max_id + 1000, max_id + 1000 + n_canaries))
    templates["record_id"] = canary_ids

    # perturb ~30% of features by 1-2 standard deviations
    numeric_features = [f for f in feature_names if f in templates.columns]
    for col in numeric_features:
        if rng.random() < 0.3:
            std = train_df[col].std()
            noise = rng.uniform(1, 2) * std * rng.choice([-1, 1])
            templates[col] = templates[col] + noise

    # flip labels randomly — this is the key trick
    # a random label on a slightly unusual record = hard to fit without memorizing
    templates["target"] = rng.randint(0, 2, size=n_canaries)

    print(f"Created {n_canaries} canary records (IDs: {canary_ids[0]}..{canary_ids[-1]})")
    print(f"  Label distribution: {templates['target'].value_counts().to_dict()}")

    return templates, canary_ids


def inject_canaries(train_df, canary_df):
    """Add canaries to the training data. They inherit the shard of their template."""
    combined = pd.concat([train_df, canary_df], ignore_index=True)
    print(f"Injected canaries. Training set: {len(train_df)} → {len(combined)}")
    return combined


def get_canary_data(canary_ids, train_df, feature_names):
    """Extract feature/label arrays for canary records."""
    mask = train_df["record_id"].isin(canary_ids)
    X = train_df.loc[mask, feature_names].values
    y = train_df.loc[mask, "target"].values
    present_ids = train_df.loc[mask, "record_id"].values
    return X, y, present_ids
