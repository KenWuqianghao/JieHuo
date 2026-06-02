#!/usr/bin/env bash
# Relabel public queries with OpenAI-compatible API (see .env).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs data/labeled

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv — run: python3 -m venv .venv && pip install -e ."
  exit 1
fi

LOG="${API_LABEL_LOG:-logs/api_label_gpt41mini.log}"
OUTPUT="${1:-data/labeled/llm_labeled_api.jsonl}"
DETACH="${API_LABEL_DETACH:-0}"
BATCH_SIZE="${API_LABEL_BATCH_SIZE:-20}"
SLEEP_SECONDS="${API_LABEL_SLEEP:-0.05}"

echo "Log: $LOG"
echo "Output: $OUTPUT"

run_label() {
  PYTHONUNBUFFERED=1 .venv/bin/python -u ml/04_llm_label.py \
    --provider openai \
    --model "${OPENAI_MODEL:-openai/gpt-4.1-mini}" \
    --output "$OUTPUT" \
    --batch-size "$BATCH_SIZE" \
    --sleep "$SLEEP_SECONDS" \
    --skip-synthetic \
    2>&1 | tee -a "$LOG"
}

if [[ "$DETACH" == "1" ]]; then
  echo "Detaching — tail -f $LOG to monitor"
  nohup env PYTHONUNBUFFERED=1 .venv/bin/python -u ml/04_llm_label.py \
    --provider openai \
    --model "${OPENAI_MODEL:-openai/gpt-4.1-mini}" \
    --output "$OUTPUT" \
    --batch-size "$BATCH_SIZE" \
    --sleep "$SLEEP_SECONDS" \
    --skip-synthetic >> "$LOG" 2>&1 &
  echo "PID=$!"
  exit 0
fi

run_label
