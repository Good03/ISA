# NASDAQ-100 Stock Price Predictor — ISA Mini-Project 3

**Course:** Intelligent System Applications (ISA)  
**Authors:** Nikita Koliasnikov  
**Deployment target:** Mini-Project 3 — Production-ready ISA deployment

---

## Overview

A web application that deploys two stock-price prediction models trained on NASDAQ-100 historical data (2000–2026, ~514 000 rows, ~100 tickers):

| Model | Technique | Train period | Features |
|---|---|---|---|
| **LSTM** | Deep learning (Keras) | 2000–2018 | Close price sequences |
| **Ridge** | Linear regression + L2 | 2016–2018 | 12 OHLCV-derived features |

The app lets you compare model predictions interactively, filter by date range, and re-train with custom hyperparameters and data splits — all from the browser.

---

## Requirements

| Requirement | Minimum version | Notes |
|---|---|---|
| Docker | 24.x | [docs.docker.com/get-docker](https://docs.docker.com/get-docker/) |
| Docker Compose | 2.x (plugin) | Bundled with Docker Desktop |
| RAM | 4 GB free | TensorFlow CPU needs ~2 GB |
| Disk | 3 GB free | Image ~1.8 GB, data 28 MB |
| Dataset | — | See **Dataset setup** below |

Docker Desktop covers both requirements on Windows and macOS.

---

## Installation

### 1 — Clone / download the project

```bash
git clone <repo-url>
cd ISA
```

Or download and unzip the submission archive, then `cd` into the extracted folder.

### 2 — Place the dataset

The CSV file is **not** included in the Docker image (it is mounted at runtime).  
Copy `NASDAQ100_Historical_Data.csv` into the `data/` folder:

```
ISA/
└── data/
    └── NASDAQ100_Historical_Data.csv   ← place it here
```

The file must be named exactly `NASDAQ100_Historical_Data.csv`.

### 3 — Build and start

```bash
docker-compose up --build
```

**First run only** — after the image is built, the container trains LSTM and Ridge models for five tickers (AAPL, MSFT, NVDA, AMZN, META). Training runs on CPU and takes **10–20 minutes** depending on hardware.

Trained models are saved to a Docker named volume (`model_cache`) and reused on every subsequent start — subsequent starts take **< 30 seconds**.

### 4 — Open the app

Navigate to **http://localhost:8080** in any modern browser.

To stop the app:

```bash
docker-compose down
```

To stop and delete all trained models (forces full retraining on next start):

```bash
docker-compose down -v
```

---

## Project structure

```
ISA/
├── docker-compose.yml          # Orchestration: app service + model_cache volume
├── data/
│   └── NASDAQ100_Historical_Data.csv   # Dataset (not included, must be provided)
├── project/                    # Flask application (Docker build context)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── entrypoint.sh           # Trains models on first run, then starts server
│   ├── train.py                # Standalone training script (LSTM + Ridge)
│   ├── app.py                  # Flask API
│   └── static/
│       └── index.html          # Single-page frontend (Bootstrap 5 + Chart.js)
├── models/                     # MP1 model artefacts (SVD, LightGBM, etc.)
├── artefacts/                  # MP1 preprocessing artefacts
├── 004_ISA_AIS_MP2.ipynb       # Development notebook (MP2)
└── README.md                   # This file
```

---

## User manual

### Selecting a ticker and model

The **control bar** at the top of the page contains two groups of buttons:

- **Ticker** — choose one of the five pre-trained stocks: `AAPL`, `AMZN`, `META`, `MSFT`, `NVDA`.
- **Model** — switch between `LSTM` (red) and `Ridge` (blue). The chart, metrics, and next-day prediction update immediately.

### Reading the chart

The main chart shows the **test period** (default: Jan–Jun 2019):

| Line | Meaning |
|---|---|
| White solid | Actual closing price |
| Coloured dashed | Model prediction |

Hover over any point to see exact values in a tooltip.

### Filtering by date

The **Date range** row sits directly above the chart:

```
Date range:  [2019-01-02]  →  [2019-06-28]   [Reset]
```

Change either date to zoom into a sub-period. Metrics (R², RMSE, MAE) recalculate instantly on the visible window. Click **Reset** to restore the full test period.

### Reading the metrics panel

| Metric | Meaning |
|---|---|
| **R²** | Proportion of variance explained (1.0 = perfect) |
| **RMSE** | Root-mean-square error in dollars — average prediction error magnitude |
| **MAE** | Mean absolute error in dollars |

When a custom validation split is active (see below), a second **Validation set** block appears below the Test set block. A large gap between validation and test R² indicates overfitting.

### Next-day prediction

The **Next-Day Prediction** card shows the model's forecast for the trading day after the last test date (end of Jun 2019). It uses the final 5 actual prices from the test period as input.

> **Note:** Current market prices (2024–2026) are outside the model's training distribution due to stock splits, so live inference is intentionally limited to the evaluation window.

### Tuning parameters

Click **⚙ Tune Parameters** to expand the panel.

#### Data Splits table

Defines which part of the dataset is used for each role:

| Split | Default (LSTM) | Default (Ridge) |
|---|---|---|
| Train | 2000-01-01 → 2018-12-31 | 2016-01-01 → 2018-12-31 |
| Validate | 2019-01-01 → 2019-03-31 | 2019-01-01 → 2019-03-31 |
| Test | 2019-04-01 → 2019-06-30 | 2019-04-01 → 2019-06-30 |

Edit any date and then click **▶ Apply** (Ridge) or **▶ Train** (LSTM) to use the new split.

Try setting the test period to **2020-02-01 → 2020-04-30** (COVID crash) to see how both models perform on an out-of-distribution shock.

#### Ridge parameters

| Parameter | Range | Effect |
|---|---|---|
| **Alpha** (log scale) | 10⁻⁴ – 10⁴ | L2 regularisation strength. Higher alpha = smoother, more biased predictions. |

Click **▶ Apply** — results appear in under one second.

#### LSTM parameters

| Parameter | Range | Effect |
|---|---|---|
| **Look-back** | 1–60 days | How many past days the model sees. Smaller = less memory, faster; larger = captures longer trends. |
| **Units** | 2–128 | LSTM hidden size. Larger = more capacity but more overfitting risk on small data. |
| **Epochs** | 5–200 | Maximum training passes. Early stopping (patience=5) may stop earlier. |

Click **▶ Train** — a progress bar tracks epoch-by-epoch completion. Training on CPU takes 2–10 minutes depending on look-back, units, and epochs.

---

## API reference

The Flask backend exposes a REST API consumed by the frontend. All endpoints return JSON.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/tickers` | List trained tickers and which models are ready |
| `GET` | `/api/predictions/<ticker>` | LSTM test-set predictions and metrics |
| `GET` | `/api/predictions/ridge/<ticker>` | Ridge test-set predictions and metrics |
| `GET` | `/api/next_day/lstm/<ticker>` | LSTM next-day forecast |
| `GET` | `/api/next_day/ridge/<ticker>` | Ridge next-day forecast |
| `POST` | `/api/tune/ridge/<ticker>` | Retrain Ridge with custom params + date splits |
| `POST` | `/api/tune/lstm/<ticker>` | Start async LSTM retraining job |
| `GET` | `/api/tune/status/<job_id>` | Poll async LSTM training progress |

---

## Technology stack

| Layer | Library / Tool | Version |
|---|---|---|
| Web framework | Flask | 3.0.3 |
| Production server | Waitress | 3.0.1 |
| Deep learning | TensorFlow CPU | 2.15.0 |
| Classical ML | scikit-learn | 1.4.2 |
| Data processing | pandas / NumPy | 2.2.2 / 1.26.4 |
| Frontend charting | Chart.js (CDN) | 4.4.3 |
| Styling | Bootstrap (CDN) | 5.3.3 |
| Containerisation | Docker + Compose | 24+ / 2+ |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Port 8080 already in use | Change `"8080:8080"` to e.g. `"8081:8080"` in `docker-compose.yml` |
| Training never finishes | Check `docker-compose logs` — low RAM may cause OOM; try closing other apps |
| `No such file: NASDAQ100_Historical_Data.csv` | Ensure the CSV is in `data/` with the exact filename |
| Chart shows no data | Open browser DevTools → Console for error details; check `/api/tickers` returns non-empty lists |
| Want to retrain from scratch | Run `docker-compose down -v` then `docker-compose up` |
