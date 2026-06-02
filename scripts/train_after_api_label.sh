#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="logs/train_after_api_label_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

LABEL_LOG="${1:-logs/api_label_gpt41mini.log}"
INPUT="${2:-data/labeled/llm_labeled_api.jsonl}"
EXPECTED="${3:-33796}"

echo "=== Waiting for API relabel $(date) ==="
echo "  label log: $LABEL_LOG"
echo "  output:    $INPUT (expect ~$EXPECTED rows)"

while pgrep -f "04_llm_label.py.*llm_labeled_api" >/dev/null 2>&1; do
  lines=$(wc -l < "$INPUT" 2>/dev/null || echo 0)
  echo "  $(date +%H:%M:%S) labeling… $lines / $EXPECTED"
  sleep 60
done

lines=$(wc -l < "$INPUT" 2>/dev/null || echo 0)
if [[ "$lines" -lt $((EXPECTED - 100)) ]]; then
  echo "API relabel incomplete ($lines rows); not training."
  exit 1
fi

if [[ -f "$LABEL_LOG" ]] && ! grep -q "LLM labeled" "$LABEL_LOG"; then
  echo "Note: label log missing completion line; proceeding from row count."
fi

PYTHON=".venv/bin/python"
run() {
  echo ""
  echo ">>> [$1] $(date +%H:%M:%S)"
  shift
  "$@"
}

run "05 merge split api" "$PYTHON" ml/05_merge_split.py --input "$INPUT" --min-confidence 0.70 --gold-size 500
run "06 train api" "$PYTHON" ml/06_train.py --output checkpoints/best_api --epochs 5 --batch-size 32 --lr 1e-5 --min-train-confidence 0.70 --label-smoothing 0.05 --language-weight-power 0.25 --disagreement-weight 0.75
run "07 eval api" "$PYTHON" ml/07_eval.py --model checkpoints/best_api --output checkpoints/eval_api

if "$PYTHON" - <<'PY'
import json
from pathlib import Path

baseline_test_f1 = 0.604
baseline_gold_f1 = 0.644
metrics_path = Path("checkpoints/eval_api/metrics.json")
metrics = json.loads(metrics_path.read_text())
test_f1 = metrics.get("test", {}).get("macro_f1", 0.0)
gold_f1 = metrics.get("gold", {}).get("macro_f1", 0.0)
print(f"API teacher test macro F1: {test_f1:.4f} (baseline {baseline_test_f1:.4f})")
print(f"API teacher gold macro F1: {gold_f1:.4f} (baseline {baseline_gold_f1:.4f})")
raise SystemExit(0 if (test_f1 >= baseline_test_f1 or gold_f1 >= baseline_gold_f1) else 1)
PY
then
  run "08 export api onnx" "$PYTHON" ml/08_export_onnx.py --model checkpoints/best_api
else
  echo "API model did not beat baseline; leaving current web model unchanged."
fi

echo ""
echo "=== API teacher training/eval done $(date) ==="
