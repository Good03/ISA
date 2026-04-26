"""Flask API for the LSTM + Ridge stock price predictor."""

import glob
import json
import os
import pickle
import threading
import uuid

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
DATA_PATH = os.getenv("DATA_PATH", "/app/data/NASDAQ100_Historical_Data.csv")

TEST_START  = "2019-01-01"
TEST_END    = "2019-06-30"

LSTM_TRAIN_START  = "2000-01-01"
LSTM_TRAIN_END    = "2017-06-30"
LSTM_VAL_START    = "2017-07-01"
LSTM_VAL_END      = "2018-12-31"
RIDGE_TRAIN_START = "2016-01-01"
RIDGE_TRAIN_END   = "2018-12-31"

LOOK_BACK    = 10
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "Close_Lag1", "Close_Lag2", "Close_Lag3",
    "Volume_Lag1", "SMA_5", "SMA_20", "Daily_Range",
]

app = Flask(__name__, static_folder="static")

# ── Module-level caches ───────────────────────────────────────────────────────

_lstm_results:  dict = {}
_ridge_results: dict = {}
_lstm_models:   dict = {}
_lstm_scalers:  dict = {}
_ridge_models:  dict = {}
_df: pd.DataFrame | None = None

_jobs: dict = {}               # job_id -> {status, progress, epoch, result, error}
_train_lock = threading.Lock() # one LSTM job at a time


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        _df = _df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return _df


def _create_sequences(data: np.ndarray, target: np.ndarray, look_back: int):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        y.append(target[i + look_back])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _prepare_ridge_features(ticker_df: pd.DataFrame) -> pd.DataFrame:
    d = ticker_df.copy().sort_values("Date")
    d["Close_Lag1"]  = d["Close"].shift(1)
    d["Close_Lag2"]  = d["Close"].shift(2)
    d["Close_Lag3"]  = d["Close"].shift(3)
    d["Volume_Lag1"] = d["Volume"].shift(1)
    d["SMA_5"]       = d["Close"].rolling(5).mean()
    d["SMA_20"]      = d["Close"].rolling(20).mean()
    d["Daily_Range"] = d["High"] - d["Low"]
    d["Target"]      = d["Close"].shift(-1)
    return d.dropna()


def _metrics(true, pred) -> dict:
    r2   = float(r2_score(true, pred))
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae  = float(np.mean(np.abs(np.array(true) - np.array(pred))))
    return {"r2": r2, "rmse": rmse, "mae": mae}


# ── Boot ──────────────────────────────────────────────────────────────────────

def _boot():
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "*_results.json"))):
        if "_ridge_results.json" in path:
            continue
        ticker = os.path.basename(path).replace("_results.json", "")
        with open(path) as fh:
            _lstm_results[ticker] = json.load(fh)
        m_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.keras")
        s_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
        if os.path.exists(m_path) and os.path.exists(s_path):
            _lstm_models[ticker] = tf.keras.models.load_model(m_path)
            with open(s_path, "rb") as fh:
                _lstm_scalers[ticker] = pickle.load(fh)

    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "*_ridge_results.json"))):
        ticker = os.path.basename(path).replace("_ridge_results.json", "")
        with open(path) as fh:
            _ridge_results[ticker] = json.load(fh)
        r_path = os.path.join(MODEL_DIR, f"{ticker}_ridge.pkl")
        if os.path.exists(r_path):
            with open(r_path, "rb") as fh:
                _ridge_models[ticker] = pickle.load(fh)

    print(f"LSTM  models: {sorted(_lstm_results.keys())}", flush=True)
    print(f"Ridge models: {sorted(_ridge_results.keys())}", flush=True)


# ── Standard routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/tickers")
def tickers():
    return jsonify({
        "tickers":     sorted(set(_lstm_results) | set(_ridge_results)),
        "lstm_ready":  sorted(_lstm_results.keys()),
        "ridge_ready": sorted(_ridge_results.keys()),
    })


@app.route("/api/summary")
def summary():
    """Aggregate metrics for all trained tickers — used by the Risk Assessment tab."""
    rows = []
    for ticker in sorted(set(_lstm_results) | set(_ridge_results)):
        row = {"ticker": ticker}
        if ticker in _lstm_results:
            r = _lstm_results[ticker]
            row["lstm_r2"]   = round(r["r2"],   4)
            row["lstm_rmse"] = round(r["rmse"],  4)
            row["lstm_mae"]  = round(r["mae"],   4)
        if ticker in _ridge_results:
            r = _ridge_results[ticker]
            row["ridge_r2"]   = round(r["r2"],   4)
            row["ridge_rmse"] = round(r["rmse"],  4)
            row["ridge_mae"]  = round(r["mae"],   4)
        rows.append(row)

    # Macro averages
    def avg(key):
        vals = [r[key] for r in rows if key in r]
        return round(sum(vals) / len(vals), 4) if vals else None

    return jsonify({
        "tickers": rows,
        "avg": {
            "lstm_r2":    avg("lstm_r2"),
            "lstm_rmse":  avg("lstm_rmse"),
            "lstm_mae":   avg("lstm_mae"),
            "ridge_r2":   avg("ridge_r2"),
            "ridge_rmse": avg("ridge_rmse"),
            "ridge_mae":  avg("ridge_mae"),
        },
    })


@app.route("/api/predictions/<ticker>")
def predictions_lstm(ticker: str):
    if ticker not in _lstm_results:
        return jsonify({"error": f"No LSTM model for {ticker}"}), 404
    return jsonify(_lstm_results[ticker])


@app.route("/api/predictions/ridge/<ticker>")
def predictions_ridge(ticker: str):
    if ticker not in _ridge_results:
        return jsonify({"error": f"No Ridge model for {ticker}"}), 404
    return jsonify(_ridge_results[ticker])


@app.route("/api/next_day/lstm/<ticker>")
def next_day_lstm(ticker: str):
    if ticker not in _lstm_models:
        return jsonify({"error": f"No LSTM model for {ticker}"}), 404
    try:
        true_prices = _lstm_results[ticker]["true"]
        look_back   = _lstm_results[ticker].get("look_back", LOOK_BACK)
        recent = np.array(true_prices[-look_back:], dtype=np.float32)
        last   = float(true_prices[-1])

        scaler = _lstm_scalers[ticker]
        model  = _lstm_models[ticker]
        scaled = scaler.transform(recent.reshape(-1, 1)).flatten()
        pred   = float(scaler.inverse_transform(
            model.predict(scaled.reshape(1, look_back, 1), verbose=0)
        )[0, 0])

        return jsonify({
            "ticker":     ticker,
            "last_price": last,
            "last_date":  _lstm_results[ticker]["dates"][-1],
            "predicted":  pred,
            "change":     pred - last,
            "change_pct": (pred - last) / last * 100,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/next_day/ridge/<ticker>")
def next_day_ridge(ticker: str):
    if ticker not in _ridge_models:
        return jsonify({"error": f"No Ridge model for {ticker}"}), 404
    try:
        res   = _ridge_results[ticker]
        feats = np.array(res["last_features"], dtype=np.float64).reshape(1, -1)
        pred  = float(_ridge_models[ticker]["model"].predict(feats)[0])
        last  = res["last_price"]
        return jsonify({
            "ticker":     ticker,
            "last_price": last,
            "last_date":  res["last_date"],
            "predicted":  pred,
            "change":     pred - last,
            "change_pct": (pred - last) / last * 100,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Tuning routes ─────────────────────────────────────────────────────────────

@app.route("/api/tune/ridge/<ticker>", methods=["POST"])
def tune_ridge(ticker: str):
    """Retrain Ridge with custom alpha and optional date splits — fast, returns immediately."""
    body  = request.get_json() or {}
    alpha = float(body.get("alpha", 1.0))
    alpha = max(1e-4, min(alpha, 1e4))

    train_start = body.get("train_start", RIDGE_TRAIN_START)
    train_end   = body.get("train_end",   RIDGE_TRAIN_END)
    val_start   = body.get("val_start",   "")
    val_end     = body.get("val_end",     "")
    test_start  = body.get("test_start",  TEST_START)
    test_end    = body.get("test_end",    TEST_END)

    try:
        df       = _get_df()
        prepared = _prepare_ridge_features(df[df["Ticker"] == ticker].copy())

        train = prepared[(prepared["Date"] >= train_start) & (prepared["Date"] <= train_end)]
        test  = prepared[(prepared["Date"] >= test_start)  & (prepared["Date"] <= test_end)]

        if len(train) < 10 or len(test) < 3:
            return jsonify({"error": "Not enough data in the selected split"}), 400

        X_tr, y_tr = train[FEATURE_COLS].values, train["Target"].values
        X_te, y_te = test[FEATURE_COLS].values,  test["Target"].values

        model = Ridge(alpha=alpha)
        model.fit(X_tr, y_tr)

        test_pred = model.predict(X_te)
        m_test    = _metrics(y_te, test_pred)

        result = {
            "ticker": ticker, "model": "ridge",
            **m_test,
            "train_period": f"{train_start} – {train_end}",
            "dates": [str(d)[:10] for d in test["Date"].values],
            "true":  y_te.tolist(),
            "pred":  test_pred.tolist(),
            "config": {"alpha": alpha},
        }

        # Validation metrics if a validation split was supplied
        if val_start and val_end:
            val = prepared[(prepared["Date"] >= val_start) & (prepared["Date"] <= val_end)]
            if len(val) >= 3:
                val_pred = model.predict(val[FEATURE_COLS].values)
                m_val    = _metrics(val["Target"].values, val_pred)
                result["val_r2"]   = m_val["r2"]
                result["val_rmse"] = m_val["rmse"]
                result["val_mae"]  = m_val["mae"]
                result["val_period"] = f"{val_start} – {val_end}"
                result["val_dates"]  = [str(d)[:10] for d in val["Date"].values]
                result["val_true"]   = val["Target"].values.tolist()
                result["val_pred"]   = val_pred.tolist()

        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def _lstm_job(job_id: str, ticker: str, look_back: int, units: int, epochs: int,
              train_start: str, train_end: str,
              val_start: str, val_end: str,
              test_start: str, test_end: str):
    """Runs in a background thread; updates _jobs[job_id] as training progresses."""
    with _train_lock:
        try:
            def _phase(p):
                _jobs[job_id]["phase"] = p

            _jobs[job_id].update({"status": "running", "progress": 0,
                                   "epoch": 0, "phase": "loading"})

            _phase("loading")
            df  = _get_df()
            tdf = df[df["Ticker"] == ticker].sort_values("Date")

            train   = tdf[(tdf["Date"] >= train_start) & (tdf["Date"] <= train_end)]
            test    = tdf[(tdf["Date"] >= test_start)  & (tdf["Date"] <= test_end)]
            has_val = bool(val_start and val_end)

            if len(train) < look_back + 10 or len(test) < look_back + 3:
                raise ValueError("Not enough data in the selected split")

            _phase("scaling")
            scaler       = MinMaxScaler((0, 1))
            train_scaled = scaler.fit_transform(train["Close"].values.reshape(-1, 1))
            test_scaled  = scaler.transform(test["Close"].values.reshape(-1, 1))

            X_tr, y_tr = _create_sequences(train_scaled, train_scaled.flatten(), look_back)
            X_te, y_te = _create_sequences(test_scaled,  test_scaled.flatten(),  look_back)

            fit_kwargs: dict = {"validation_split": 0.1}
            if has_val:
                val = tdf[(tdf["Date"] >= val_start) & (tdf["Date"] <= val_end)]
                if len(val) >= look_back + 3:
                    val_scaled = scaler.transform(val["Close"].values.reshape(-1, 1))
                    X_val, y_val = _create_sequences(val_scaled, val_scaled.flatten(), look_back)
                    fit_kwargs = {"validation_data": (X_val, y_val)}

            _phase("building")
            model = Sequential([Input(shape=(look_back, 1)), LSTM(units), Dense(1)])
            model.compile(optimizer=Adam(0.001), loss="mse")

            _phase("training")

            class _Prog(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    _jobs[job_id].update({
                        "epoch":      epoch + 1,
                        "progress":   round((epoch + 1) / epochs, 3),
                        "train_loss": round(float(logs.get("loss", 0)), 6),
                        "val_loss":   round(float(logs.get("val_loss", 0)), 6),
                    })

            _jobs[job_id].update({
                "train_samples": int(len(X_tr)),
                "val_samples":   int(len(fit_kwargs.get("validation_data", [[]])[0])
                                     if "validation_data" in fit_kwargs
                                     else int(len(X_tr) * 0.1)),
            })

            model.fit(
                X_tr, y_tr,
                epochs=epochs, batch_size=16,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                    _Prog(),
                ],
                verbose=0,
                **fit_kwargs,
            )

            _phase("evaluating")
            pred  = scaler.inverse_transform(model.predict(X_te, verbose=0)).flatten()
            true  = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
            dates = test["Date"].values[look_back:]
            m     = _metrics(true, pred)

            result = {
                "ticker": ticker, "model": "lstm",
                **m,
                "train_period": f"{train_start} – {train_end}",
                "dates": [str(d)[:10] for d in dates],
                "true":  true.tolist(),
                "pred":  pred.tolist(),
                "config": {"look_back": look_back, "units": units, "epochs": epochs},
            }

            # Validation metrics (if explicit val set was used)
            if has_val and "validation_data" in fit_kwargs:
                val_pred = scaler.inverse_transform(model.predict(X_val, verbose=0)).flatten()
                val_true = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
                m_val    = _metrics(val_true, val_pred)
                result["val_r2"]    = m_val["r2"]
                result["val_rmse"]  = m_val["rmse"]
                result["val_mae"]   = m_val["mae"]
                result["val_period"] = f"{val_start} – {val_end}"

            _jobs[job_id].update({"status": "done", "progress": 1.0, "result": result})

        except Exception as exc:
            _jobs[job_id].update({"status": "error", "error": str(exc)})


@app.route("/api/tune/lstm/<ticker>", methods=["POST"])
def tune_lstm(ticker: str):
    """Start async LSTM retraining; returns a job_id to poll."""
    body      = request.get_json() or {}
    look_back = max(1,  min(int(body.get("look_back", LOOK_BACK)), 60))
    units     = max(2,  min(int(body.get("units",     5)),          200))
    epochs    = max(5,  min(int(body.get("epochs",    30)),          200))

    train_start = body.get("train_start", LSTM_TRAIN_START)
    train_end   = body.get("train_end",   LSTM_TRAIN_END)
    val_start   = body.get("val_start",   LSTM_VAL_START)
    val_end     = body.get("val_end",     LSTM_VAL_END)
    test_start  = body.get("test_start",  TEST_START)
    test_end    = body.get("test_end",    TEST_END)

    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "queued", "progress": 0, "epoch": 0, "result": None, "error": None,
                     "total_epochs": epochs}

    t = threading.Thread(
        target=_lstm_job,
        args=(job_id, ticker, look_back, units, epochs,
              train_start, train_end, val_start, val_end, test_start, test_end),
        daemon=True,
    )
    t.start()
    return jsonify({"job_id": job_id, "total_epochs": epochs})


@app.route("/api/tune/status/<job_id>")
def tune_status(job_id: str):
    if job_id not in _jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(_jobs[job_id])


# ── Startup ───────────────────────────────────────────────────────────────────

_boot()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
