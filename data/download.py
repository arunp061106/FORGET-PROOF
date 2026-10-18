"""
Download the UCI Diabetes 130-US Hospitals dataset.
We use this because the organizers didn't provide a dataset and this one
has ~100k rows of real hospital readmission data (CC BY 4.0).

Ref: https://archive.ics.uci.edu/dataset/296
"""

import os
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(DATA_DIR, "raw_diabetes.csv")


def download_dataset():
    # don't re-download if we already have it
    if os.path.exists(RAW_PATH):
        print(f"Already have the dataset at {RAW_PATH}, loading from cache")
        return pd.read_csv(RAW_PATH)

    print("Downloading UCI Diabetes 130-US Hospitals dataset...")

    # try the nice library first
    try:
        from ucimlrepo import fetch_ucirepo
        result = fetch_ucirepo(id=296)
        df = result.data.original
        print(f"Got {len(df)} rows via ucimlrepo")
    except Exception as e:
        # fallback: grab the zip directly from UCI
        print(f"ucimlrepo didn't work ({e}), trying direct download...")
        import zipfile, io, urllib.request

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
        resp = urllib.request.urlopen(url)
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        df = pd.read_csv(z.open(csv_files[0]))
        print(f"Got {len(df)} rows via direct download")

    df.to_csv(RAW_PATH, index=False)
    print(f"Saved to {RAW_PATH}")
    return df


if __name__ == "__main__":
    df = download_dataset()
    print(f"Shape: {df.shape}")
    print(df.head(3))
