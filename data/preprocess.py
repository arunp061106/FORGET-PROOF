"""
Preprocessing pipeline for the diabetes readmission dataset.

What we do here:
1. Drop useless / heavily-missing columns
2. Encode categoricals (simple label encoding — good enough for tree models)
3. Build a binary target: readmitted within 30 days or not
4. Add a few derived features that make clinical sense
5. Split into train/test
6. Assign each training record to a "consent batch" (= shard)
   These batches simulate real-world data arriving from different hospitals
   or consent waves. The shard assignment is what makes SISA work.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def clean(df):
    """Basic cleaning. Returns a copy."""
    df = df.copy()
    df = df.reset_index(drop=True)
    df["record_id"] = df.index  # we'll need this to track who's in which shard

    # '?' means missing in this dataset
    df = df.replace("?", np.nan)

    # these columns are either mostly missing or not useful for prediction
    junk = ["weight", "payer_code", "medical_specialty",
            "examide", "citoglipton",       # near-zero variance drugs
            "encounter_id", "patient_nbr"]  # patient identifiers — we shouldn't train on these anyway
    df = df.drop(columns=[c for c in junk if c in df.columns])

    # just drop remaining rows with NaN — it's a small fraction after the above
    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df)} rows with missing values ({100*(before-len(df))/before:.1f}%)")

    return df


def build_features(df):
    """Feature engineering + target creation."""
    df = df.copy()

    # target: was the patient readmitted within 30 days?
    df["target"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # encode all the string columns — label encoding is fine for gradient boosting
    encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # a few derived features that a clinician might care about
    if "num_medications" in df.columns and "num_procedures" in df.columns:
        df["meds_per_procedure"] = df["num_medications"] / (df["num_procedures"] + 1)

    if "num_lab_procedures" in df.columns and "time_in_hospital" in df.columns:
        df["labs_per_day"] = df["num_lab_procedures"] / (df["time_in_hospital"] + 1)

    if "number_diagnoses" in df.columns:
        # 7+ diagnoses = complex case
        df["high_complexity"] = (df["number_diagnoses"] >= 7).astype(int)

    return df, encoders


def assign_shards(df, n_shards, seed=42):
    """
    Randomly assign each record to a shard (consent batch).

    In a real system these would come from actual consent groups or
    data source partitions. We simulate it with random assignment
    to keep things reproducible.
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["shard_id"] = rng.randint(0, n_shards, size=len(df))

    print(f"\nShard distribution (N={n_shards}):")
    for s in range(n_shards):
        n = (df["shard_id"] == s).sum()
        print(f"  shard {s}: {n} records ({100*n/len(df):.1f}%)")

    return df


def prepare_data(n_shards=8, test_size=0.2, seed=42, raw_df=None):
    """
    Run the full pipeline. Returns a dict with everything downstream needs.

    If raw_df is None, loads from the cached CSV (you need to run download.py first).
    """
    if raw_df is None:
        path = os.path.join(DATA_DIR, "raw_diabetes.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No dataset found at {path}. Run download.py first."
            )
        raw_df = pd.read_csv(path)

    print(f"Starting with {len(raw_df)} rows")

    df = clean(raw_df)
    df, encoders = build_features(df)
    print(f"After cleaning + features: {df.shape}")

    # figure out which columns are actual features
    non_features = {"target", "record_id", "shard_id"}
    feature_cols = [c for c in df.columns if c not in non_features]

    # stratified split so both sets have similar readmission rates
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["target"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # only assign shards to training data — test data doesn't belong to any shard
    train_df = assign_shards(train_df, n_shards, seed)

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df["target"].values
    y_test = test_df["target"].values

    pos_rate_train = y_train.mean()
    pos_rate_test = y_test.mean()
    print(f"\nTrain: {len(X_train)} samples, {100*pos_rate_train:.1f}% positive")
    print(f"Test:  {len(X_test)} samples, {100*pos_rate_test:.1f}% positive")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
        "feature_names": feature_cols,
        "encoders": encoders,
        "n_shards": n_shards,
    }


if __name__ == "__main__":
    data = prepare_data(n_shards=8)
    print(f"\n{len(data['feature_names'])} features: {data['feature_names'][:8]}...")
