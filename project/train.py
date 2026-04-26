"""
Train LSTM and Ridge models for NASDAQ-100 stock price prediction (Mini-Project 2).

LSTM best config (9-experiment sweep):
  look_back=5, units=5, epochs=50, train 2000-2018

Ridge config (from notebook):
  12 OHLCV features, alpha=1.0, train 2016-2018
"""

import os
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
DATA_PATH = os.getenv("DATA_PATH", "/app/data/NASDAQ100_Historical_Data.csv")

# Shared test window
TEST_START = "2019-01-01"
TEST_END   = "2019-06-30"

# LSTM hyper-parameters
LSTM_TRAIN_START = "2000-01-01"
LSTM_TRAIN_END   = "2018-12-31"
LOOK_BACK = 5
UNITS     = 5
EPOCHS    = 50
BATCH     = 16

# Ridge hyper-parameters
RIDGE_TRAIN_START = "2016-01-01"
RIDGE_TRAIN_END   = "2018-12-31"
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "Close_Lag1", "Close_Lag2", "Close_Lag3",
    "Volume_Lag1", "SMA_5", "SMA_20", "Daily_Range",
]

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def create_sequences(data: np.ndarray, target: np.ndarray, look_back: int):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        y.append(target[i + look_back])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_features(ticker_df: pd.DataFrame) -> pd.DataFrame:
    """Build the 12-feature DataFrame used by Ridge (same as notebook)."""
    d = ticker_df.copy().sort_values("Date")
    d["Close_Lag1"]  = d["Close"].shift(1)
    d["Close_Lag2"]  = d["Close"].shift(2)
    d["Close_Lag3"]  = d["Close"].shift(3)
    d["Volume_Lag1"] = d["Volume"].shift(1)
    d["SMA_5"]       = d["Close"].rolling(5).mean()
    d["SMA_20"]      = d["Close"].rolling(20).mean()
    d["Daily_Range"] = d["High"] - d["Low"]
    d["Target"]      = d["Close"].shift(-1)   # next-day close
    return d.dropna()


# ── LSTM training ─────────────────────────────────────────────────────────────

def train_lstm(ticker: str, df: pd.DataFrame) -> bool:
    out = os.path.join(MODEL_DIR, f"{ticker}_results.json")   # keep original name
    if os.path.exists(out):
        print(f"  [{ticker}] LSTM already trained — skipping", flush=True)
        return True

    print(f"  [{ticker}] Training LSTM...", flush=True)
    tdf   = df[df["Ticker"] == ticker].sort_values("Date")
    train = tdf[(tdf["Date"] >= LSTM_TRAIN_START) & (tdf["Date"] <= LSTM_TRAIN_END)]
    test  = tdf[(tdf["Date"] >= TEST_START)        & (tdf["Date"] <= TEST_END)]

    if len(train) < 50 or len(test) < LOOK_BACK + 5:
        print(f"  [{ticker}] LSTM: insufficient data", flush=True)
        return False

    scaler       = MinMaxScaler((0, 1))
    train_scaled = scaler.fit_transform(train["Close"].values.reshape(-1, 1))
    test_scaled  = scaler.transform(test["Close"].values.reshape(-1, 1))

    X_tr, y_tr = create_sequences(train_scaled, train_scaled.flatten(), LOOK_BACK)
    X_te, y_te = create_sequences(test_scaled,  test_scaled.flatten(),  LOOK_BACK)

    model = Sequential([Input(shape=(LOOK_BACK, 1)), LSTM(UNITS), Dense(1)])
    model.compile(optimizer=Adam(0.001), loss="mse")
    model.fit(
        X_tr, y_tr,
        epochs=EPOCHS, batch_size=BATCH, validation_split=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=0,
    )

    pred  = scaler.inverse_transform(model.predict(X_te, verbose=0)).flatten()
    true  = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
    dates = test["Date"].values[LOOK_BACK:]

    r2   = float(r2_score(true, pred))
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae  = float(np.mean(np.abs(true - pred)))
    print(f"  [{ticker}] LSTM  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}", flush=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, f"{ticker}_lstm.keras"))
    with open(os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl"), "wb") as fh:
        pickle.dump(scaler, fh)
    with open(out, "w") as fh:
        json.dump({
            "ticker": ticker, "model": "lstm",
            "r2": r2, "rmse": rmse, "mae": mae,
            "look_back": LOOK_BACK,
            "train_period": f"{LSTM_TRAIN_START} – {LSTM_TRAIN_END}",
            "dates": [str(d)[:10] for d in dates],
            "true":  true.tolist(),
            "pred":  pred.tolist(),
        }, fh)
    return True


# ── Ridge training ────────────────────────────────────────────────────────────

def train_ridge(ticker: str, df: pd.DataFrame) -> bool:
    out = os.path.join(MODEL_DIR, f"{ticker}_ridge_results.json")
    if os.path.exists(out):
        print(f"  [{ticker}] Ridge already trained — skipping", flush=True)
        return True

    print(f"  [{ticker}] Training Ridge...", flush=True)
    tdf       = df[df["Ticker"] == ticker].copy()
    prepared  = prepare_features(tdf)

    train = prepared[(prepared["Date"] >= RIDGE_TRAIN_START) & (prepared["Date"] <= RIDGE_TRAIN_END)]
    test  = prepared[(prepared["Date"] >= TEST_START)        & (prepared["Date"] <= TEST_END)]

    if len(train) < 50 or len(test) < 10:
        print(f"  [{ticker}] Ridge: insufficient data", flush=True)
        return False

    X_tr, y_tr = train[FEATURE_COLS].values, train["Target"].values
    X_te, y_te = test[FEATURE_COLS].values,  test["Target"].values

    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    r2   = float(r2_score(y_te, pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    mae  = float(np.mean(np.abs(y_te - pred)))
    print(f"  [{ticker}] Ridge R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}", flush=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, f"{ticker}_ridge.pkl"), "wb") as fh:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS}, fh)

    # Last test row features for next-day prediction
    last_row = test.iloc[-1]
    last_features = [float(last_row[f]) for f in FEATURE_COLS]

    with open(out, "w") as fh:
        json.dump({
            "ticker": ticker, "model": "ridge",
            "r2": r2, "rmse": rmse, "mae": mae,
            "train_period": f"{RIDGE_TRAIN_START} – {RIDGE_TRAIN_END}",
            "dates": [str(d)[:10] for d in test["Date"].values],
            "true":  y_te.tolist(),
            "pred":  pred.tolist(),
            "last_date":     str(last_row["Date"])[:10],
            "last_price":    float(last_row["Close"]),
            "last_features": last_features,
        }, fh)
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading NASDAQ-100 dataset...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    available = set(df["Ticker"].unique())
    tickers   = [t for t in DEFAULT_TICKERS if t in available]
    print(f"Tickers: {', '.join(tickers)}\n", flush=True)

    for ticker in tickers:
        train_lstm(ticker, df)
        train_ridge(ticker, df)

    print("\nAll done!", flush=True)


if __name__ == "__main__":
    main()
