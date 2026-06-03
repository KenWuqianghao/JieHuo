#!/usr/bin/env bash
# Reproduce the current best multilingual checkpoint (requires OPENAI_API_KEY in .env).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== JieHuo pipeline started $(date) ==="
echo "Log: $LOG"

PYTHON=".venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing .venv — run: python3 -m venv .venv && $PYTHON -m pip install -e ."
  exit 1
fi

run() {
  echo ""
  echo ">>> [$1] $(date +%H:%M:%S)"
  shift
  "$@"
}

run "02 synth" "$PYTHON" ml/02_synth_queries.py --provider openai --model gpt-4.1-mini --per-bucket 50
run "03 heuristic" "$PYTHON" ml/03_heuristic_labels.py
run "04 llm label" "$PYTHON" ml/04_llm_label.py \
  --input data/labeled/heuristic_labeled.jsonl \
  --output data/labeled/llm_labeled_api.jsonl \
  --provider openai \
  --model gpt-4.1-mini \
  --batch-size 60
run "05 merge split" "$PYTHON" ml/05_merge_split.py \
  --input data/labeled/llm_labeled_api.jsonl \
  --min-confidence 0.70 \
  --gold-size 240
run "06 train" "$PYTHON" ml/06_train.py \
  --output checkpoints/best_multilingual \
  --epochs 8 \
  --batch-size 32 \
  --lr 1e-5 \
  --min-train-confidence 0.70 \
  --max-train-per-language-label 1500 \
  --label-smoothing 0.05 \
  --language-weight-power 0.50 \
  --disagreement-weight 0.75 \
  --seed 42
run "07 eval" "$PYTHON" ml/07_eval.py \
  --model checkpoints/best_multilingual \
  --output checkpoints/eval_multilingual \
  --calibration data/labeled/val.jsonl \
  --confidence-thresholds 0.50,0.60,0.70,0.80,0.85,0.90,0.95 \
  --bootstrap-samples 500
run "08 export onnx" "$PYTHON" ml/08_export_onnx.py \
  --model checkpoints/best_multilingual \
  --temperature 1.25

echo ""
echo "=== Pipeline complete $(date) ==="
echo "Upload: $PYTHON ml/09_upload_hf.py --repo KenWu/multilingual-query-router"
