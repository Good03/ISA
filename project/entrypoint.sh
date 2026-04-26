#!/bin/bash
set -e

MODEL_DIR="${MODEL_DIR:-/app/model}"
mkdir -p "${MODEL_DIR}"

LSTM_TRAINED=$(find "${MODEL_DIR}" -name "*_results.json" -not -name "*_ridge_results.json" 2>/dev/null | wc -l)
RIDGE_TRAINED=$(find "${MODEL_DIR}" -name "*_ridge_results.json" 2>/dev/null | wc -l)

if [ "${LSTM_TRAINED}" -eq 0 ] || [ "${RIDGE_TRAINED}" -eq 0 ]; then
  echo "========================================================"
  echo " Training models (LSTM trained: ${LSTM_TRAINED}, Ridge trained: ${RIDGE_TRAINED})"
  echo " First run may take ~10-20 minutes on CPU."
  echo "========================================================"
  python train.py
  echo "========================================================"
  echo " Training complete!"
  echo "========================================================"
else
  echo "All models ready (LSTM: ${LSTM_TRAINED}, Ridge: ${RIDGE_TRAINED}) — skipping training."
fi

echo "Starting server → http://localhost:8080"
exec waitress-serve --host=0.0.0.0 --port=8080 app:app
