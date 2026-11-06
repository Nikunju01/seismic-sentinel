"""Seismic Sentinel - a control-room console that scores saved models on uploaded coal-mine shift logs."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
TARGET = "class"
FEATURES = ["seismic", "seismoacoustic", "shift", "genergy", "gpuls", "gdenergy", "gdpuls", "ghazard",
            "nbumps", "nbumps2", "nbumps3", "nbumps4", "nbumps5", "nbumps6", "nbumps7", "nbumps89",
            "energy", "maxenergy"]
MODELS = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree", "KNN": "knn",
          "Naive Bayes": "naive_bayes", "Support Vector Machine": "support_vector_machine"}
CODENAMES = {"Logistic Regression": "LR-01 · LINEAR PROBE", "Decision Tree": "DT-02 · RULE DRILL",
             "KNN": "KN-03 · NEIGHBOUR ECHO", "Naive Bayes": "NB-04 · PROBABILITY LAMP",
             "Support Vector Machine": "SV-05 · MARGIN WALL"}
AMBER, RED, TEAL, INK = "#FFB000", "#FF4D3D", "#3DD6C6", "#E8E4DA"

st.set_page_config(page_title="Seismic Sentinel", page_icon="⛏️", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;600&display=swap');
#MainMenu, footer, header[data-testid="stHeader"] {{visibility:hidden;}}
html, body, [class*="css"], .stMarkdown, p, label {{font-family:'JetBrains Mono', monospace;}}
.block-container {{padding:1.2rem 2.5rem 2rem; max-width:100%;}}
.dial {{min-height:92px;}}
.stripe {{height:10px; background:repeating-linear-gradient(-45deg,{AMBER} 0 14px,#0E0F11 14px 28px); border-radius:3px;}}
.hero {{border:1px solid #2A2D31; border-radius:6px; padding:26px 30px 8px; margin:14px 0 22px;
        background:radial-gradient(1200px 300px at 10% -40%, rgba(255,176,0,.14), transparent), #121417;}}
.hero h1 {{font-family:'Space Grotesk'; font-size:3.1rem; margin:0; letter-spacing:-1px; color:{INK};}}
.hero h1 span {{color:{AMBER};}}
.hero .tag {{color:#9A958A; font-size:.9rem; margin-top:4px;}}
.pill {{display:inline-block; border:1px solid {AMBER}; color:{AMBER}; padding:2px 10px; border-radius:20px;
        font-size:.72rem; letter-spacing:2px; margin-bottom:10px;}}
.wave path {{stroke:{AMBER}; stroke-width:2; fill:none; stroke-dasharray:1400; animation:draw 4s linear infinite;}}
@keyframes draw {{from{{stroke-dashoffset:1400}} to{{stroke-dashoffset:0}}}}
.sec {{font-family:'Space Grotesk'; font-size:1.25rem; color:{INK}; margin:56px 0 18px; border-left:4px solid {AMBER}; padding-left:10px;}}
.sec small {{color:#7C786F; font-family:'JetBrains Mono'; font-size:.75rem; margin-left:8px;}}
.dial {{background:#15171A; border:1px solid #2A2D31; border-radius:6px; padding:14px 16px;}}
.dial .k {{color:#8C887F; font-size:.72rem; letter-spacing:2px;}}
.dial .v {{font-family:'Space Grotesk'; font-size:2rem; color:{INK};}}
.bar {{height:6px; background:#26292D; border-radius:3px; overflow:hidden;}}
.bar i {{display:block; height:100%;}}
/* breathing room between stacked panels */
.verdict {{margin-top:18px; border:1px dashed #3A3D41; border-radius:6px; padding:12px 16px; color:#BDB8AD; font-size:.85rem;}}
div[data-testid="stFileUploader"] section {{background:#15171A; border:1px dashed {AMBER};}}
</style>
<div class="stripe"></div>
<div class="hero">
  <div class="pill">UNDERGROUND HAZARD EARLY-WARNING CONSOLE</div>
  <h1>SEISMIC <span>SENTINEL</span></h1>
  <div class="tag">Will the next 8-hour shift in a Polish longwall coal mine bring a rock-burst above 10⁴ J?
  Upload a shift log and five trained models answer, then the console scores them against reality.</div>
  <svg class="wave" viewBox="0 0 1200 70" width="100%" height="70" preserveAspectRatio="none">
    <path d="M0 35 L120 35 L140 30 L160 40 L180 35 L320 35 L335 18 L350 52 L365 12 L380 58 L395 25 L410 45
    L425 35 L600 35 L615 28 L630 42 L645 35 L760 35 L770 5 L782 66 L794 10 L806 60 L818 20 L830 50 L842 35
    L1000 35 L1012 30 L1024 40 L1036 35 L1200 35"/>
  </svg>
</div>
""", unsafe_allow_html=True)


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


def dial(label, value, color):
    pct = 0 if np.isnan(value) else max(value, 0) * 100
    shown = "n/a" if np.isnan(value) else f"{value:.3f}"
    return (f'<div class="dial"><div class="k">{label.upper()}</div><div class="v">{shown}</div>'
            f'<div class="bar"><i style="width:{pct:.0f}%;background:{color}"></i></div></div>')


def plot_base(fig, h=400):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#15171A",
                      font=dict(family="JetBrains Mono", color=INK), margin=dict(l=10, r=10, t=60, b=20))
    return fig


# ---------- 01 · Feed ----------
st.markdown('<div class="sec">01 · FEED THE SENSORS<small>upload test shifts (CSV with the 18 features + class)</small></div>',
            unsafe_allow_html=True)
c_up, c_pick, c_kpi = st.columns([2.2, 1.6, 2.4], gap="large")
with c_up:
    upload = st.file_uploader("Shift log CSV", type="csv", label_visibility="collapsed")
    with open(ROOT / "test_data.csv", "rb") as f:
        st.download_button("⬇ Download sample test_data.csv", f, "test_data.csv", "text/csv")
with c_pick:
    model_name = st.selectbox("Choose a detector model", list(MODELS))
    st.caption(f"`{CODENAMES[model_name]}` · pre-trained, loaded from `model/` - no retraining")

if upload is None:
    st.markdown('<div class="verdict">⏳ Awaiting a shift log. Grab the sample file above and drop it in the bay.</div>',
                unsafe_allow_html=True)
    st.stop()

df = pd.read_csv(upload)
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"CSV is missing required columns: {', '.join(missing)}")
    st.stop()
X, y = df[FEATURES], df[TARGET].astype(int)

hazards = int(y.sum())
with c_kpi:
    k1, k2 = st.columns(2)
    k3, k4 = st.columns(2)
for col, lab, val in [(k1, "shifts logged", len(df)), (k2, "hazard shifts", hazards),
                      (k3, "hazard rate", f"{hazards / len(df):.1%}"), (k4, "features", len(FEATURES))]:
    col.markdown(f'<div class="dial"><div class="k">{lab.upper()}</div><div class="v">{val}</div></div>',
                 unsafe_allow_html=True)

# ---------- 02 · Selected detector ----------
pipe = load_model(MODELS[model_name])
pred = pipe.predict(X)
prob = pipe.predict_proba(X)[:, 1]
m = score(y, pred, prob)

st.markdown(f'<div class="sec">02 · DETECTOR READOUT<small>{CODENAMES[model_name]}</small></div>',
            unsafe_allow_html=True)
cols = st.columns(6)
palette = [AMBER, AMBER, TEAL, RED, AMBER, TEAL]
for col, (k, v), c in zip(cols, m.items(), palette):
    col.markdown(dial(k, v, c), unsafe_allow_html=True)
st.markdown('<div style="height:36px"></div>', unsafe_allow_html=True)

left, mid, right = st.columns([1, 1, 1.3], gap="large")
with left:
    cm = confusion_matrix(y, pred, labels=[0, 1])
    labels = ["Calm shift", "Hazard shift"]
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale=[[0, "#15171A"], [1, AMBER]],
                               text=cm, texttemplate="<b>%{text}</b>", textfont=dict(size=22), showscale=False))
    fig.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="Actual",
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(plot_base(fig), width='stretch')
with mid:
    rep = pd.DataFrame(classification_report(y, pred, labels=[0, 1], target_names=labels,
                                             output_dict=True, zero_division=0)).T
    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
    st.markdown("**Classification report**")
    st.dataframe(rep.style.format("{:.3f}").background_gradient(cmap="YlOrBr", subset=["f1-score"]),
                 width='stretch')
    missed, false_alarm = cm[1, 0], cm[0, 1]
    st.markdown(f'<div class="verdict">🚨 <b>{cm[1,1]}</b> hazards caught · <b style="color:{RED}">{missed}</b> '
                f'missed (miners unwarned) · <b>{false_alarm}</b> false alarms (shifts paused needlessly)</div>',
                unsafe_allow_html=True)
with right:
    order = np.argsort(prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=prob[order], mode="lines", line=dict(color=AMBER), name="risk score"))
    hz = np.where(y.values[order] == 1)[0]
    fig.add_trace(go.Scatter(x=hz, y=prob[order][hz], mode="markers", name="actual hazard",
                             marker=dict(color=RED, size=7, symbol="x")))
    fig.update_layout(title="Risk seismogram (shifts sorted by predicted risk)", xaxis_title="shift rank",
                      yaxis_title="P(hazard)", legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(plot_base(fig), width='stretch')

# ---------- 03 · All detectors ----------
st.markdown('<div class="sec">03 · DETECTOR SHOOT-OUT<small>all five saved models on your uploaded shifts</small></div>',
            unsafe_allow_html=True)
board = {}
for name, slug in MODELS.items():
    p = load_model(slug)
    board[name] = score(y, p.predict(X), p.predict_proba(X)[:, 1])
board = pd.DataFrame(board).T
best = board["MCC"].idxmax()

b1, b2 = st.columns([1.3, 1], gap="large")
with b1:
    st.dataframe(board.style.format("{:.4f}").highlight_max(color="#5A4300"), width='stretch')
    st.markdown(f'<div class="verdict">🏆 Best balance on this log (by MCC): <b style="color:{AMBER}">{best}</b>. '
                f'Accuracy alone misleads here: a model that always says "calm" would score '
                f'{1 - hazards / len(df):.1%}.</div>', unsafe_allow_html=True)
with b2:
    axes = list(board.columns)
    fig = go.Figure()
    for i, (name, row) in enumerate(board.iterrows()):
        fig.add_trace(go.Scatterpolar(r=list(row.clip(lower=0)) + [max(row.iloc[0], 0)], theta=axes + [axes[0]],
                                      name=name, line=dict(width=3 if name == best else 1)))
    fig.update_layout(polar=dict(bgcolor="#15171A", radialaxis=dict(range=[0, 1], gridcolor="#2A2D31"),
                                 angularaxis=dict(gridcolor="#2A2D31")), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(plot_base(fig, 380), width='stretch')

with st.expander("Training-time metrics (held-out 25% split, from model/metrics.json)"):
    st.dataframe(pd.DataFrame(json.loads((MODEL_DIR / "metrics.json").read_text())).T.style.format("{:.4f}"),
                 width='stretch')
st.markdown('<div class="stripe" style="margin-top:30px"></div>', unsafe_allow_html=True)
