#!/usr/bin/env python3
"""Evaluate model with per-language F1, confusion matrix, and ECE calibration."""

from __future__ import annotations

import ml._bootstrap  # noqa: F401

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.common import (
    CHECKPOINT_DIR,
    E5_QUERY_PREFIX,
    GOLD_DIR,
    LABELED_DIR,
    LABEL_NAMES,
    ensure_dirs,
    read_jsonl,
)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def predict_batch(
    model, tokenizer, queries: list[str], device: str, max_length: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_probs = [], []

    batch_size = 64
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        texts = [E5_QUERY_PREFIX + q for q in batch]
        enc = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        preds = probs.argmax(axis=-1)
        all_preds.extend(preds)
        all_probs.extend(probs)

    return np.array(all_preds), np.array(all_probs)


def evaluate_split(
    model, tokenizer, records: list[dict], device: str, split_name: str, output_dir: Path
) -> dict:
    queries = [r["query"] for r in records]
    labels = np.array([r["label_id"] for r in records])
    languages = [r.get("language", "unknown") for r in records]

    preds, probs = predict_batch(model, tokenizer, queries, device)
    pred_probs = probs[np.arange(len(preds)), preds]

    metrics = {
        "split": split_name,
        "n": len(records),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "ece": float(expected_calibration_error(labels, pred_probs)),
    }

    # Per-language F1
    lang_metrics: dict[str, dict] = {}
    by_lang: dict[str, list] = defaultdict(lambda: {"labels": [], "preds": []})
    for lang, label, pred in zip(languages, labels, preds):
        by_lang[lang]["labels"].append(label)
        by_lang[lang]["preds"].append(pred)

    for lang, data in sorted(by_lang.items()):
        lang_metrics[lang] = {
            "n": len(data["labels"]),
            "accuracy": float(accuracy_score(data["labels"], data["preds"])),
            "macro_f1": float(f1_score(data["labels"], data["preds"], average="macro", zero_division=0)),
        }
    metrics["per_language"] = lang_metrics

    # Confusion matrix plot
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["google", "perplexity"],
        yticklabels=["google", "perplexity"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {split_name}")
    fig.savefig(output_dir / f"confusion_matrix_{split_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Calibration histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pred_probs, bins=20, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence Distribution — {split_name} (ECE={metrics['ece']:.3f})")
    fig.savefig(output_dir / f"calibration_{split_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n=== {split_name.upper()} ({len(records)} samples) ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"ECE:      {metrics['ece']:.4f}")
    print("\nPer-language:")
    for lang, lm in lang_metrics.items():
        print(f"  {lang}: n={lm['n']}, acc={lm['accuracy']:.3f}, f1={lm['macro_f1']:.3f}")

    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=["google", "perplexity"]))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate query router model")
    parser.add_argument("--model", type=str, default=str(CHECKPOINT_DIR / "best"))
    parser.add_argument("--test", type=str, default=str(LABELED_DIR / "test.jsonl"))
    parser.add_argument("--gold", type=str, default=str(GOLD_DIR / "gold.jsonl"))
    parser.add_argument("--output", type=str, default=str(CHECKPOINT_DIR / "eval"))
    args = parser.parse_args()

    ensure_dirs()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}. Run 06_train.py first.")
        return

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(device)

    all_metrics = {}
    for split_name, path in [("test", args.test), ("gold", args.gold)]:
        p = Path(path)
        if p.exists():
            records = list(read_jsonl(p))
            all_metrics[split_name] = evaluate_split(model, tokenizer, records, device, split_name, output_dir)

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nMetrics saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
