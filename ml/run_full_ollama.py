#!/usr/bin/env python3
"""Run the full Ollama pipeline: collect → synth → label → train → export.

No shortcuts. Expect several hours for synth + labeling on a MacBook.
Logs to logs/pipeline_ollama.log
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import ml._bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
PYTHON = sys.executable


def run_step(name: str, script: str, args: list[str]) -> None:
    cmd = [PYTHON, str(ROOT / "ml" / script)] + args
    print(f"\n{'='*60}\n[{datetime.now():%H:%M:%S}] {name}\n{'='*60}")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"FAILED: {name} (exit {result.returncode})")


def main() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    steps = [
        ("Collect public data", "01_collect_public.py", []),
        (
            "Generate synthetic (Ollama)",
            "02_synth_queries.py",
            ["--provider", "ollama", "--model", "qwen2.5:1.5b", "--per-bucket", "25"],
        ),
        ("Heuristic labels", "03_heuristic_labels.py", []),
        (
            "Ollama LLM labels (ALL queries)",
            "04_llm_label.py",
            ["--provider", "ollama", "--model", "qwen2.5:1.5b", "--batch-size", "15"],
        ),
        ("Merge & split", "05_merge_split.py", ["--min-confidence", "0.55", "--gold-size", "500"]),
        ("Train e5-small", "06_train.py", ["--epochs", "5", "--batch-size", "32", "--lr", "2e-5"]),
        ("Evaluate", "07_eval.py", []),
        ("Export ONNX", "08_export_onnx.py", []),
    ]

    for name, script, args in steps:
        run_step(name, script, args)

    print(f"\n{'='*60}\nPipeline complete at {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("Upload: python3 ml/09_upload_hf.py --repo KenWu/multilingual-query-router")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
