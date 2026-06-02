#!/usr/bin/env python3
"""Upload exported ONNX model to Hugging Face Hub for CDN serving."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import ml._bootstrap  # noqa: F401
from ml.common import ROOT, ensure_dirs

load_dotenv()

# Only upload files needed by transformers.js (skip duplicate ONNX variants)
UPLOAD_PATTERNS = [
    "README.md",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "label_config.json",
    "router_config.json",
    "onnx/model_quantized.onnx",
]


def collect_upload_files(model_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in UPLOAD_PATTERNS:
        path = model_dir / pattern
        if path.exists():
            files.append(path)
        else:
            print(f"Warning: missing {path}")
    if not any("onnx" in str(f) for f in files):
        raise FileNotFoundError(
            f"No ONNX model found under {model_dir}/onnx/. Run ml/08_export_onnx.py first."
        )
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument(
        "--repo",
        type=str,
        default=os.environ.get("HF_MODEL_REPO", "YOUR_USERNAME/multilingual-query-router"),
        help="HF repo id, e.g. username/multilingual-query-router",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(ROOT / "web" / "public" / "models" / "multilingual-router"),
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    api = HfApi(token=token)
    try:
        user = api.whoami()
        print(f"Hugging Face auth: {user.get('name', 'unknown')}")
    except Exception:
        print("Error: Hugging Face authentication not found.")
        print("  Run: huggingface-cli login")
        print("  or export HF_TOKEN=hf_... from https://huggingface.co/settings/tokens")
        sys.exit(1)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Model dir not found: {model_dir}. Run ml/08_export_onnx.py first.")
        sys.exit(1)

    upload_files = collect_upload_files(model_dir)
    total_mb = sum(f.stat().st_size for f in upload_files) / (1024 * 1024)
    print(f"Repo:  {args.repo}")
    print(f"Files: {len(upload_files)} ({total_mb:.1f} MB total)")
    for f in upload_files:
        print(f"  - {f.relative_to(model_dir)} ({f.stat().st_size / (1024 * 1024):.1f} MB)")

    print("\nCreating repo (if needed)...")
    api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)

    print(f"Uploading to https://huggingface.co/{args.repo} ...")
    print("(This may take several minutes for ~113 MB — do not interrupt.)\n")

    for f in upload_files:
        rel = f.relative_to(model_dir).as_posix()
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  ↑ {rel} ({size_mb:.1f} MB)...", flush=True)
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=rel,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"Upload {rel}",
        )
        print("    done.", flush=True)

    print("\n✓ Upload complete!")
    print("\nNext steps:")
    print(f"  1. Verify: https://huggingface.co/{args.repo}")
    print(f"  2. Set Vercel env var: NEXT_PUBLIC_MODEL_REPO={args.repo}")
    print("  3. Redeploy: cd web && npx vercel --prod")


if __name__ == "__main__":
    ensure_dirs()
    main()
