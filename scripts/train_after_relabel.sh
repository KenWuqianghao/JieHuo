#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="logs/train_after_relabel_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Waiting for stale relabel $(date) ==="
RELABEL_PID=40067
while kill -0 "$RELABEL_PID" 2>/dev/null; do
  sleep 60
done

if ! grep -q "LLM labeled" logs/relabel_stale_llama31_8b.log; then
  echo "Relabel did not finish cleanly; not training."
  exit 1
fi

PYTHON=".venv/bin/python"
run() {
  echo ""
  echo ">>> [$1] $(date +%H:%M:%S)"
  shift
  "$@"
}

run "05 merge split repaired" "$PYTHON" ml/05_merge_split.py --min-confidence 0.70 --gold-size 500
run "06 train repaired" "$PYTHON" ml/06_train.py --output checkpoints/best_repaired --epochs 5 --batch-size 32 --lr 1e-5 --min-train-confidence 0.70 --label-smoothing 0.05 --language-weight-power 0.25
run "07 eval repaired" "$PYTHON" ml/07_eval.py --model checkpoints/best_repaired --output checkpoints/eval_repaired

if "$PYTHON" - <<'PY'
import json
from pathlib import Path

baseline_test_f1 = 0.6064
baseline_gold_f1 = 0.6404
metrics_path = Path("checkpoints/eval_repaired/metrics.json")
metrics = json.loads(metrics_path.read_text())
test_f1 = metrics.get("test", {}).get("macro_f1", 0.0)
gold_f1 = metrics.get("gold", {}).get("macro_f1", 0.0)
print(f"Repaired test macro F1: {test_f1:.4f} (baseline {baseline_test_f1:.4f})")
print(f"Repaired gold macro F1: {gold_f1:.4f} (baseline {baseline_gold_f1:.4f})")
raise SystemExit(0 if (test_f1 >= baseline_test_f1 or gold_f1 >= baseline_gold_f1) else 1)
PY
then
  run "08 export repaired onnx" "$PYTHON" ml/08_export_onnx.py --model checkpoints/best_repaired
else
  echo "Repaired model did not beat baseline; leaving current web model unchanged."
fi

echo ""
echo "=== Repaired training/eval done $(date) ==="
