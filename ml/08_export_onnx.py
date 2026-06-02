#!/usr/bin/env python3
"""Export fine-tuned model to ONNX and INT8 quantize for browser deployment."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import ml._bootstrap  # noqa: F401
from ml.common import CHECKPOINT_DIR, E5_QUERY_PREFIX, GOLD_DIR, ROOT, ensure_dirs, read_jsonl


def export_onnx(model_path: Path, output_dir: Path) -> Path:
    """Export model to ONNX using optimum."""
    from optimum.onnxruntime import ORTModelForSequenceClassification

    print(f"Exporting {model_path} to ONNX...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(model_path), export=True
    )
    ort_model.save_pretrained(str(output_dir))

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer.save_pretrained(str(output_dir))

    # Copy label config
    label_config = model_path / "label_config.json"
    if label_config.exists():
        shutil.copy(label_config, output_dir / "label_config.json")

    onnx_files = list(output_dir.glob("**/*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX file found in {output_dir}")
    return onnx_files[0]


def quantize_onnx(input_path: Path, output_path: Path) -> None:
    """Dynamic INT8 quantization."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"Quantizing {input_path} -> {output_path}")
    quantize_dynamic(
        str(input_path),
        str(output_path),
        weight_type=QuantType.QInt8,
    )


def verify_parity(
    pytorch_path: Path, onnx_dir: Path, gold_path: Path, max_samples: int = 200
) -> dict:
    """Compare PyTorch vs ONNX predictions on gold set."""
    from optimum.onnxruntime import ORTModelForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(pytorch_path))

    pt_model = AutoModelForSequenceClassification.from_pretrained(str(pytorch_path)).to(device)
    pt_model.eval()

    ort_model = ORTModelForSequenceClassification.from_pretrained(str(onnx_dir))

    records = list(read_jsonl(gold_path))[:max_samples]
    queries = [r["query"] for r in records]
    labels = [r["label_id"] for r in records]

    pt_preds, ort_preds = [], []
    for q in queries:
        text = E5_QUERY_PREFIX + q
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            pt_out = pt_model(**enc)
            pt_pred = pt_out.logits.argmax(dim=-1).item()
        pt_preds.append(pt_pred)

        ort_enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        ort_out = ort_model(**ort_enc)
        ort_pred = ort_out.logits.argmax(dim=-1).item()
        ort_preds.append(ort_pred)

    pt_preds = np.array(pt_preds)
    ort_preds = np.array(ort_preds)
    labels = np.array(labels)

    agreement = (pt_preds == ort_preds).mean()
    pt_acc = (pt_preds == labels).mean()
    ort_acc = (ort_preds == labels).mean()

    return {
        "n_samples": len(records),
        "pt_onnx_agreement": float(agreement),
        "pytorch_accuracy": float(pt_acc),
        "onnx_accuracy": float(ort_acc),
        "accuracy_drop": float(pt_acc - ort_acc),
    }


def prepare_web_model(onnx_dir: Path, web_model_dir: Path, temperature: float) -> None:
    """Copy model files to web/public/models/multilingual-router/."""
    web_model_dir.mkdir(parents=True, exist_ok=True)

    # Copy tokenizer and config files
    for pattern in ["*.json", "*.txt", "*.model"]:
        for f in onnx_dir.glob(pattern):
            shutil.copy(f, web_model_dir / f.name)

    # Copy ONNX model(s) into onnx/ subdirectory (transformers.js convention)
    onnx_subdir = web_model_dir / "onnx"
    onnx_subdir.mkdir(exist_ok=True)

    for onnx_file in onnx_dir.glob("**/*.onnx"):
        dest_name = onnx_file.name
        # Only ship quantized model to the browser bundle
        if "quantized" not in dest_name.lower():
            continue
        dest_name = "model_quantized.onnx"
        shutil.copy(onnx_file, onnx_subdir / dest_name)

    # Write config for transformers.js
    config = {
        "model_type": "bert",
        "architectures": ["BertForSequenceClassification"],
        "num_labels": 2,
        "id2label": {"0": "google", "1": "perplexity"},
        "label2id": {"google": 0, "perplexity": 1},
    }
    with (web_model_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    router_config = {
        "temperature": temperature,
        "confidence_thresholds": {
            "balanced_gold_90_accuracy": 0.80,
            "heldout_test_90_accuracy": 0.85,
        },
    }
    with (web_model_dir / "router_config.json").open("w") as f:
        json.dump(router_config, f, indent=2)

    print(f"Web model prepared at {web_model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and quantize model for web")
    parser.add_argument("--model", type=str, default=str(CHECKPOINT_DIR / "best"))
    parser.add_argument("--output", type=str, default=str(ROOT / "web" / "public" / "models" / "multilingual-router"))
    parser.add_argument("--gold", type=str, default=str(GOLD_DIR / "gold.jsonl"))
    parser.add_argument("--skip-quantize", action="store_true")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Optional confidence calibration temperature exported to the web runtime",
    )
    args = parser.parse_args()

    ensure_dirs()
    model_path = Path(args.model)
    output_dir = Path(args.output)
    export_dir = CHECKPOINT_DIR / "onnx_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Model not found: {model_path}. Run 06_train.py first.")
        return

    onnx_path = export_onnx(model_path, export_dir)

    if not args.skip_quantize:
        quantized_path = export_dir / "onnx" / "model_quantized.onnx"
        quantized_path.parent.mkdir(exist_ok=True)
        quantize_onnx(onnx_path, quantized_path)

    gold_path = Path(args.gold)
    if gold_path.exists():
        parity = verify_parity(model_path, export_dir, gold_path)
        print("\nParity check:")
        print(f"  PT-ONNX agreement: {parity['pt_onnx_agreement']:.4f}")
        print(f"  PyTorch accuracy:  {parity['pytorch_accuracy']:.4f}")
        print(f"  ONNX accuracy:     {parity['onnx_accuracy']:.4f}")
        print(f"  Accuracy drop:     {parity['accuracy_drop']:.4f}")

        with (export_dir / "parity.json").open("w") as f:
            json.dump(parity, f, indent=2)

    prepare_web_model(export_dir, output_dir, args.temperature)
    print(f"\nDone. Model ready for web deployment at {output_dir}")


if __name__ == "__main__":
    main()
