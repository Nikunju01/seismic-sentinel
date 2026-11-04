"""Seismic Sentinel - data preparation for the UCI seismic-bumps dataset.

Run from repo root:  python model/train.py
Outputs: test_data.csv (held-out 25% split used later by the app)
"""
from pathlib import Path

import pandas as pd
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
TARGET = "class"
CATEGORICAL = ["seismic", "seismoacoustic", "shift", "ghazard"]
SEED = 26


def load_shifts() -> pd.DataFrame:
    raw, _ = arff.loadarff(ROOT / "raw" / "seismic-bumps.arff")
    df = pd.DataFrame(raw)
    for col in CATEGORICAL + [TARGET]:
        df[col] = df[col].str.decode("utf-8")
    df[TARGET] = df[TARGET].astype(int)
    return df


def build_pipeline(estimator, numeric_cols):
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", StandardScaler(), numeric_cols),
    ])
    return Pipeline([("prep", prep), ("clf", estimator)])


def main():
    df = load_shifts()
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    pd.concat([X_te, y_te], axis=1).to_csv(ROOT / "test_data.csv", index=False)

    print(f"rows={len(df)} features={X.shape[1]} hazard_rate={y.mean():.3f}")
    print(f"train={len(X_tr)} test={len(X_te)} test_hazards={int(y_te.sum())}")
    print("missing values:", int(df.isna().sum().sum()))


if __name__ == "__main__":
    main()
