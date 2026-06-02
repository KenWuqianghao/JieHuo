#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="logs/retry_train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Retry train/eval/export $(date) ==="
PYTHON=".venv/bin/python"

run() {
  echo ""
  echo ">>> [$1] $(date +%H:%M:%S)"
  shift
  "$@"
}

run "06 train" "$PYTHON" ml/06_train.py --epochs 5 --batch-size 32 --lr 2e-5
run "07 eval" "$PYTHON" ml/07_eval.py
run "08 export onnx" "$PYTHON" ml/08_export_onnx.py

echo ""
echo "=== Done $(date) ==="
echo "Upload: $PYTHON ml/09_upload_hf.py --repo KenWu/multilingual-query-router"
