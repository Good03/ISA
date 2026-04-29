#!/bin/bash
set -e

MODEL_DIR="${MODEL_DIR:-/app/model}"
mkdir -p "${MODEL_DIR}"

# Models are trained on-demand from the UI — no pre-training step.
# (Set PRETRAIN=1 to keep the legacy behaviour of training the default 5 tickers up-front.)
if [ "${PRETRAIN:-0}" = "1" ]; then
  LSTM_TRAINED=$(find "${MODEL_DIR}" -name "*_results.json" -not -name "*_ridge_results.json" 2>/dev/null | wc -l)
  RIDGE_TRAINED=$(find "${MODEL_DIR}" -name "*_ridge_results.json" 2>/dev/null | wc -l)
  if [ "${LSTM_TRAINED}" -eq 0 ] || [ "${RIDGE_TRAINED}" -eq 0 ]; then
    echo "========================================================"
    echo " PRETRAIN=1 - training default tickers"
    echo "========================================================"
    python train.py
  fi
fi

echo "Starting server -> http://localhost:8080"
exec waitress-serve --host=0.0.0.0 --port=8080 app:app
