# ⛏️ Seismic Sentinel: early warning of rock bursts in coal mines

> 🚧 Work in progress. A basic Streamlit app is in place; the custom UI is coming next.

## Problem statement
Deep underground, the rock around a longwall coal face stores stress until it suddenly releases it as a **seismic bump**, or rock burst. A bump above 10⁴ J can collapse workings and kill miners. Mines record seismic and acoustic activity on every 8-hour shift, but the hazard-assessment methods used today miss many of these events.

**Goal:** given the monitoring data from one shift, predict whether the **next** shift will see a high-energy bump (`class = 1`, hazardous) or not (`class = 0`). Hazardous shifts make up only about 6.6% of the data, so this is an **imbalanced binary classification** problem. Missing a real hazard costs far more than raising a false alarm.

## Dataset description
- **Source:** UCI Machine Learning Repository, *Seismic-bumps* (Sikora & Wróbel, 2010). Data comes from the "Zabrze-Bielszowice" coal mine in Poland.
  Download: https://archive.ics.uci.edu/dataset/266/seismic+bumps
- **Instances:** 2,584 shifts. **Features:** 18. **Target:** `class` (1 = hazardous bump in the next shift; 170 positives, 6.6%).

| Feature | Meaning |
|---|---|
| `seismic`, `seismoacoustic` | Hazard level (a = lack, b = low, c = high, d = danger) from the seismic and seismoacoustic methods |
| `shift` | W = coal-getting shift, N = preparation shift |
| `genergy`, `gpuls` | Seismic energy and pulse count from the geophone in the previous shift |
| `gdenergy`, `gdpuls` | Deviation of energy and pulses from the average of the last 8 shifts |
| `ghazard` | Hazard level from the geophone alone |
| `nbumps`, `nbumps2` … `nbumps7`, `nbumps89` | Number of bumps in the previous shift, overall and per energy band (10² … 10⁹ J) |
| `energy`, `maxenergy` | Total and maximum energy of the bumps in the previous shift |

**Preprocessing:** the 4 categorical columns are one-hot encoded and the 14 numeric columns are standardised, inside one `Pipeline` per model, so each saved model takes raw CSV columns as input. The data is split 75/25 with stratification (seed 26). The 25% test split (646 shifts, 43 hazards) is saved as `test_data.csv`.

## Models used
All five models are trained on the same split. Logistic Regression, Decision Tree and SVM use `class_weight="balanced"` to counter the 1:14 class imbalance.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7523 | 0.7475 | 0.1579 | 0.6279 | 0.2523 | 0.2198 |
| Decision Tree | 0.8050 | 0.7343 | 0.1832 | 0.5581 | 0.2759 | 0.2360 |
| KNN | 0.9257 | 0.6648 | 0.2222 | 0.0465 | 0.0769 | 0.0742 |
| Naive Bayes | 0.4241 | 0.7289 | 0.0958 | 0.9070 | 0.1733 | 0.1532 |
| Support Vector Machine | 0.8406 | 0.7027 | 0.2115 | 0.5116 | 0.2993 | 0.2548 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Has the **highest AUC (0.748)**, so it ranks risky shifts best. With balanced weights it catches 63% of hazards, at the cost of many false alarms (precision 0.16). It is a solid, interpretable baseline. |
| Decision Tree | A depth-limited tree (max_depth 5, 20 samples per leaf) finds threshold rules on energy and bump counts. It balances recall (0.56) and precision better than LR, which gives it the second-best MCC. Without those limits it overfits badly. |
| KNN | Shows the **accuracy trap**: 92.6% accuracy, yet it detects only 2 of 43 hazards (recall 0.05). The rare hazards are swamped by calm neighbours, so its MCC is close to zero. It is useless as a warning system despite the headline number. |
| Naive Bayes | Has the opposite bias: it flags almost everything and catches **91% of hazards** (best recall), but accuracy falls to 42%. The strongly skewed, correlated energy features break its independence assumption. It could act as a cautious first-stage screen. |
| Support Vector Machine | The RBF kernel with balanced weights gives the **best F1 (0.299) and MCC (0.255)**. It catches about half the hazards and has the highest precision among models with useful recall. |

**Overall winner for this dataset: Support Vector Machine.** On a 6.6%-positive problem, accuracy is misleading, because always predicting "calm" scores 93.4%. MCC and F1 are the fair measures, and SVM leads on both. Logistic Regression is the runner-up with the best AUC. The low absolute scores across all models match the published literature for this dataset: bumps are hard to forecast from one shift of monitoring data.

## Run locally
```bash
pip install -r requirements.txt
python model/train.py      # trains 5 models -> model/*.joblib, model/metrics.json
streamlit run app.py
```
