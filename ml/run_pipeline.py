#!/usr/bin/env python3
"""Run the full ML pipeline end-to-end."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    ("01_collect_public.py", []),
    ("02_synth_queries.py", ["--provider", "fallback", "--per-bucket", "20"]),
    ("03_heuristic_labels.py", []),
    ("04_llm_label.py", ["--provider", "fallback"]),
    ("05_merge_split.py", []),
    ("06_train.py", []),
    ("07_eval.py", []),
    ("08_export_onnx.py", []),
]


def main() -> None:
    ml_dir = Path(__file__).parent
    for script, extra_args in SCRIPTS:
        path = ml_dir / script
        print(f"\n{'='*60}")
        print(f"Running {script}")
        print(f"{'='*60}")
        cmd = [sys.executable, str(path)] + extra_args
        result = subprocess.run(cmd, cwd=ml_dir.parent)
        if result.returncode != 0:
            print(f"FAILED: {script} (exit code {result.returncode})")
            sys.exit(result.returncode)
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
