# ⛏️ Seismic Sentinel: early warning of rock bursts in coal mines

> 🚧 Work in progress. Models and the Streamlit app are coming in later commits.

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

## Run locally
```bash
pip install -r requirements.txt
python model/train.py
```
