import json
import os

import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
ARTIFACTS = os.path.join(PROJECT_ROOT, "artifacts")
MODELS = os.path.join(PROJECT_ROOT, "models")
REPORTS = os.path.join(PROJECT_ROOT, "reports")

for _d in (DATA_RAW, DATA_PROCESSED, ARTIFACTS, MODELS, REPORTS):
    os.makedirs(_d, exist_ok=True)


def save_json(obj, filename, folder=ARTIFACTS):
    path = os.path.join(folder, filename)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def load_json(filename, folder=ARTIFACTS):
    path = os.path.join(folder, filename)
    with open(path, "r") as f:
        return json.load(f)


def save_parquet(df: pd.DataFrame, filename, folder=DATA_PROCESSED):
    path = os.path.join(folder, filename)
    df.to_parquet(path, index=False)
    return path


def load_parquet(filename, folder=DATA_PROCESSED):
    path = os.path.join(folder, filename)
    return pd.read_parquet(path)


def save_object(obj, filename, folder=ARTIFACTS):
    """Simpan objek sklearn (scaler, imputer, dll) dengan joblib."""
    path = os.path.join(folder, filename)
    joblib.dump(obj, path)
    return path


def load_object(filename, folder=ARTIFACTS):
    path = os.path.join(folder, filename)
    return joblib.load(path)
