#!/usr/bin/env bash
# Full Ollama pipeline — no shortcuts. Expect many hours.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG="logs/pipeline_ollama_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== JieHuo full pipeline started $(date) ==="
echo "Log: $LOG"

PYTHON=".venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing .venv — run: python3 -m venv .venv && pip install -r requirements.txt"
  exit 1
fi

run() {
  echo ""
  echo ">>> [$1] $(date +%H:%M:%S)"
  shift
  "$@"
}

run "01 collect public" "$PYTHON" ml/01_collect_public.py
run "02 synth ollama" "$PYTHON" ml/02_synth_queries.py --provider ollama --model qwen2.5:1.5b --per-bucket 25
run "03 heuristic" "$PYTHON" ml/03_heuristic_labels.py
run "04 ollama label ALL" "$PYTHON" ml/04_llm_label.py --provider ollama --model qwen2.5:1.5b --batch-size 15
run "05 merge split" "$PYTHON" ml/05_merge_split.py --min-confidence 0.55 --gold-size 500
run "06 train" "$PYTHON" ml/06_train.py --epochs 5 --batch-size 32 --lr 2e-5
run "07 eval" "$PYTHON" ml/07_eval.py
run "08 export onnx" "$PYTHON" ml/08_export_onnx.py

echo ""
echo "=== Pipeline complete $(date) ==="
echo "Upload: $PYTHON ml/09_upload_hf.py --repo KenWu/multilingual-query-router"
