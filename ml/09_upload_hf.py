#!/usr/bin/env python3
"""Upload exported ONNX model to Hugging Face Hub for CDN serving."""

from __future__ import annotations

import ml._bootstrap  # noqa: F401

import argparse

load_dotenv()

from ml.common import ROOT, ensure_dirs

# Only upload files needed by transformers.js (skip duplicate ONNX variants)
UPLOAD_PATTERNS = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "label_config.json",
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

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("Error: HF token not found.")
        print("  export HF_TOKEN=hf_...   # from https://huggingface.co/settings/tokens")
        print("  # or add HF_TOKEN=... to a .env file in the repo root")
        sys.exit(1)

    from huggingface_hub import HfApi

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

    api = HfApi(token=token)
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
        print(f"    done.", flush=True)

    print(f"\n✓ Upload complete!")
    print(f"\nNext steps:")
    print(f"  1. Verify: https://huggingface.co/{args.repo}")
    print(f"  2. Set Vercel env var: NEXT_PUBLIC_MODEL_REPO={args.repo}")
    print(f"  3. Redeploy: cd web && npx vercel --prod")


if __name__ == "__main__":
    ensure_dirs()
    main()
