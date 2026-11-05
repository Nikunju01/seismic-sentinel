"""Train 5 classifiers to forecast hazardous seismic bumps in a coal mine (UCI seismic-bumps).

Run from repo root:  python model/train.py
Outputs: model/*.joblib (full preprocessing+model pipelines), model/metrics.json, test_data.csv
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.io import arff
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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


def score(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# Hazards are ~6.6% of shifts, so the weighted models are told to care about the rare class.
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                            class_weight="balanced", random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=9, weights="distance"),
    "Naive Bayes": GaussianNB(),
    # probability=True is deprecated in sklearn 1.9 but still works; calibrated wrapper would predict at p>0.5
    # and never flag the rare class. shortcut: pinned sklearn==1.9.1, revisit before 1.11.
    "Support Vector Machine": SVC(kernel="rbf", C=1.0, class_weight="balanced",
                                  probability=True, random_state=SEED),
}


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    df = load_shifts()
    X, y = df.drop(columns=TARGET), df[TARGET]
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    pd.concat([X_te, y_te], axis=1).to_csv(ROOT / "test_data.csv", index=False)

    results = {}
    for name, est in MODELS.items():
        pipe = build_pipeline(est, numeric_cols).fit(X_tr, y_tr)
        results[name] = score(y_te, pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1])
        joblib.dump(pipe, ROOT / "model" / f"{name.lower().replace(' ', '_')}.joblib")

    (ROOT / "model" / "metrics.json").write_text(json.dumps(results, indent=2))
    table = pd.DataFrame(results).T.round(4)
    print(f"rows={len(df)} features={X.shape[1]} hazard_rate={y.mean():.3f}")
    print(table.to_markdown())


if __name__ == "__main__":
    main()
