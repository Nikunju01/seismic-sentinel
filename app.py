"""Seismic Sentinel - scores the saved models on an uploaded test CSV of coal-mine shifts."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
TARGET = "class"
FEATURES = ["seismic", "seismoacoustic", "shift", "genergy", "gpuls", "gdenergy", "gdpuls", "ghazard",
            "nbumps", "nbumps2", "nbumps3", "nbumps4", "nbumps5", "nbumps6", "nbumps7", "nbumps89",
            "energy", "maxenergy"]
MODELS = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree", "KNN": "knn",
          "Naive Bayes": "naive_bayes", "Support Vector Machine": "support_vector_machine"}
LABELS = ["Calm shift", "Hazard shift"]

st.set_page_config(page_title="Seismic Sentinel", page_icon="⛏️", layout="wide")
st.title("⛏️ Seismic Sentinel")
st.caption("Forecasting hazardous rock bursts in a coal mine from one shift of seismic monitoring data.")


@st.cache_resource
def load_model(slug):
    # Loads only our own trained pipelines from model/, never user-supplied files.
    return joblib.load(MODEL_DIR / f"{slug}.joblib")


def score(y, pred, prob):
    return {"Accuracy": accuracy_score(y, pred),
            "AUC": roc_auc_score(y, prob) if y.nunique() > 1 else float("nan"),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "MCC": matthews_corrcoef(y, pred)}


c_up, c_pick = st.columns([3, 2], gap="large")
with c_up:
    upload = st.file_uploader("Upload test shifts (CSV with the 18 features + class)", type="csv")
    with open(ROOT / "test_data.csv", "rb") as f:
        st.download_button("Download sample test_data.csv", f, "test_data.csv", "text/csv")
with c_pick:
    model_name = st.selectbox("Choose a model", list(MODELS))
    st.caption("Pre-trained models are loaded from `model/`. Nothing is retrained.")

if upload is None:
    st.info("Upload a CSV to see the evaluation.")
    st.stop()

df = pd.read_csv(upload)
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"CSV is missing required columns: {', '.join(missing)}")
    st.stop()
X, y = df[FEATURES], df[TARGET].astype(int)

pipe = load_model(MODELS[model_name])
pred = pipe.predict(X)
prob = pipe.predict_proba(X)[:, 1]

st.subheader(f"Evaluation metrics: {model_name}")
for col, (k, v) in zip(st.columns(6), score(y, pred, prob).items()):
    col.metric(k, f"{v:.3f}")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    ConfusionMatrixDisplay.from_predictions(y, pred, labels=[0, 1], display_labels=LABELS,
                                            cmap="YlOrBr", colorbar=False, ax=ax)
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    rep = pd.DataFrame(classification_report(y, pred, labels=[0, 1], target_names=LABELS,
                                             output_dict=True, zero_division=0)).T
    st.dataframe(rep.style.format("{:.3f}"), width="stretch")
