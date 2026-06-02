#!/usr/bin/env python3
"""Evaluate model with per-language F1, confusion matrix, and ECE calibration."""

from __future__ import annotations

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
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import ml._bootstrap  # noqa: F401
from ml.common import (
    CHECKPOINT_DIR,
    E5_QUERY_PREFIX,
    GOLD_DIR,
    LABELED_DIR,
    ensure_dirs,
    read_jsonl,
)

VALID_LABELS = {"google": 0, "perplexity": 1}


def expected_calibration_error(y_correct: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_correct[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_correct) * abs(bin_acc - bin_conf)
    return ece


def selective_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    pred_probs: np.ndarray,
    thresholds: list[float],
) -> list[dict]:
    """Metrics when the router only auto-routes predictions above a confidence threshold."""
    rows: list[dict] = []
    total = len(labels)
    for threshold in thresholds:
        mask = pred_probs >= threshold
        covered = int(mask.sum())
        if covered == 0:
            rows.append(
                {
                    "threshold": threshold,
                    "coverage": 0.0,
                    "n": 0,
                    "accuracy": None,
                    "macro_f1": None,
                    "abstained": total,
                }
            )
            continue
        rows.append(
            {
                "threshold": threshold,
                "coverage": covered / total,
                "n": covered,
                "accuracy": float(accuracy_score(labels[mask], preds[mask])),
                "macro_f1": float(
                    f1_score(labels[mask], preds[mask], labels=[0, 1], average="macro", zero_division=0)
                ),
                "abstained": total - covered,
            }
        )
    return rows


def operating_points(selective: list[dict], targets: list[float]) -> dict:
    """Summarize routing coverage available at target accuracy levels."""
    points = {}
    valid_rows = [r for r in selective if r["accuracy"] is not None]
    for target in targets:
        eligible = [r for r in valid_rows if r["accuracy"] >= target]
        if not eligible:
            points[f"accuracy>={target:.2f}"] = None
            continue
        best = max(eligible, key=lambda r: (r["coverage"], r["macro_f1"]))
        points[f"accuracy>={target:.2f}"] = {
            "threshold": best["threshold"],
            "coverage": best["coverage"],
            "accuracy": best["accuracy"],
            "macro_f1": best["macro_f1"],
            "n": best["n"],
        }
    return points


def choose_coverage_floor(selective: list[dict], min_coverage: float) -> dict | None:
    eligible = [r for r in selective if r["accuracy"] is not None and r["coverage"] >= min_coverage]
    if not eligible:
        return None
    return max(eligible, key=lambda r: (r["accuracy"], r["macro_f1"]))


def bootstrap_interval(
    labels: np.ndarray,
    preds: np.ndarray,
    metric: str,
    samples: int,
    seed: int,
) -> dict | None:
    if samples <= 0 or len(labels) == 0:
        return None

    rng = np.random.default_rng(seed)
    scores = []
    n = len(labels)
    for _ in range(samples):
        idx = rng.integers(0, n, n)
        if metric == "accuracy":
            score = accuracy_score(labels[idx], preds[idx])
        else:
            score = f1_score(labels[idx], preds[idx], labels=[0, 1], average="macro", zero_division=0)
        scores.append(float(score))

    lo, hi = np.percentile(scores, [2.5, 97.5])
    return {"low": float(lo), "high": float(hi), "samples": samples}


def predict_batch(
    model, tokenizer, queries: list[str], device: str, max_length: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_probs, all_logits = [], [], []

    batch_size = 64
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        texts = [E5_QUERY_PREFIX + q for q in batch]
        enc = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits.cpu().numpy()
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        preds = probs.argmax(axis=-1)
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_logits.extend(logits)

    return np.array(all_preds), np.array(all_probs), np.array(all_logits)


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def fit_temperature(
    model,
    tokenizer,
    records: list[dict],
    device: str,
    max_length: int,
) -> dict:
    """Fit a scalar temperature on calibration records by minimizing NLL."""
    if not records:
        return {"temperature": 1.0, "n": 0, "nll": None}

    labels = np.array([r["label_id"] for r in records])
    _, _, logits = predict_batch(model, tokenizer, [r["query"] for r in records], device, max_length)
    candidates = np.concatenate(
        [
            np.linspace(0.35, 1.0, 27),
            np.linspace(1.05, 3.0, 40),
            np.linspace(3.1, 6.0, 30),
        ]
    )
    best_temp = 1.0
    best_nll = float("inf")
    for temp in candidates:
        probs = softmax_with_temperature(logits, float(temp))
        nll = float(log_loss(labels, probs, labels=[0, 1]))
        if nll < best_nll:
            best_temp = float(temp)
            best_nll = nll

    return {"temperature": best_temp, "n": len(records), "nll": best_nll}


def heuristic_predictions(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    preds, confs = [], []
    for r in records:
        label = str(r.get("heuristic_label", "")).strip().lower()
        preds.append(VALID_LABELS.get(label, -1))
        confs.append(float(r.get("heuristic_confidence", 0) or 0))
    return np.array(preds), np.array(confs)


def hybrid_metrics(
    labels: np.ndarray,
    neural_preds: np.ndarray,
    neural_conf: np.ndarray,
    records: list[dict],
    heuristic_threshold: float,
    selective_thresholds: list[float],
) -> dict | None:
    """Evaluate a deterministic-intent override followed by neural routing."""
    h_preds, h_conf = heuristic_predictions(records)
    use_heuristic = (h_preds >= 0) & (h_conf >= heuristic_threshold)
    if not use_heuristic.any():
        return None

    preds = neural_preds.copy()
    conf = neural_conf.copy()
    preds[use_heuristic] = h_preds[use_heuristic]
    conf[use_heuristic] = h_conf[use_heuristic]

    correct = (preds == labels).astype(float)
    return {
        "heuristic_threshold": heuristic_threshold,
        "heuristic_coverage": float(use_heuristic.mean()),
        "heuristic_n": int(use_heuristic.sum()),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=[0, 1], average="macro", zero_division=0)),
        "ece": float(expected_calibration_error(correct, conf)),
        "selective": selective_metrics(labels, preds, conf, selective_thresholds),
    }


def evaluate_split(
    model,
    tokenizer,
    records: list[dict],
    device: str,
    split_name: str,
    output_dir: Path,
    thresholds: list[float],
    bootstrap_samples: int,
    temperature: float,
    heuristic_threshold: float,
    target_accuracy: list[float],
    target_coverage: float,
) -> dict:
    queries = [r["query"] for r in records]
    labels = np.array([r["label_id"] for r in records])
    languages = [r.get("language", "unknown") for r in records]

    raw_preds, raw_probs, logits = predict_batch(model, tokenizer, queries, device)
    probs = softmax_with_temperature(logits, temperature)
    preds = probs.argmax(axis=-1)
    pred_probs = probs[np.arange(len(preds)), preds]
    correct = (preds == labels).astype(float)
    raw_pred_probs = raw_probs[np.arange(len(raw_preds)), raw_preds]
    raw_correct = (raw_preds == labels).astype(float)
    selective = selective_metrics(labels, preds, pred_probs, thresholds)

    metrics = {
        "split": split_name,
        "n": len(records),
        "temperature": temperature,
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=[0, 1], average="macro", zero_division=0)),
        "ece": float(expected_calibration_error(correct, pred_probs)),
        "raw_uncalibrated": {
            "accuracy": float(accuracy_score(labels, raw_preds)),
            "macro_f1": float(f1_score(labels, raw_preds, labels=[0, 1], average="macro", zero_division=0)),
            "ece": float(expected_calibration_error(raw_correct, raw_pred_probs)),
        },
        "selective": selective,
        "operating_points": operating_points(selective, target_accuracy),
        "best_at_min_coverage": choose_coverage_floor(selective, target_coverage),
    }
    metrics["confidence_intervals"] = {
        "accuracy": bootstrap_interval(labels, preds, "accuracy", bootstrap_samples, seed=17),
        "macro_f1": bootstrap_interval(labels, preds, "macro_f1", bootstrap_samples, seed=29),
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
    metrics["hybrid"] = hybrid_metrics(
        labels,
        preds,
        pred_probs,
        records,
        heuristic_threshold,
        thresholds,
    )

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
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"ECE:      {metrics['ece']:.4f}")
    if temperature != 1.0:
        raw = metrics["raw_uncalibrated"]
        print(f"Temp:     {temperature:.3f} (raw ECE={raw['ece']:.4f})")
    print("\nSelective auto-routing:")
    for row in metrics["selective"]:
        if row["accuracy"] is None:
            print(f"  conf≥{row['threshold']:.2f}: coverage=0.0%, no routed samples")
        else:
            print(
                f"  conf≥{row['threshold']:.2f}: coverage={100 * row['coverage']:.1f}%, "
                f"acc={row['accuracy']:.3f}, f1={row['macro_f1']:.3f}"
            )
    if metrics["hybrid"]:
        h = metrics["hybrid"]
        print(
            "\nHybrid deterministic-intent + neural:"
            f" heuristic coverage={100 * h['heuristic_coverage']:.1f}%, "
            f"acc={h['accuracy']:.3f}, f1={h['macro_f1']:.3f}"
        )
    print("\nPer-language:")
    for lang, lm in lang_metrics.items():
        print(f"  {lang}: n={lm['n']}, acc={lm['accuracy']:.3f}, f1={lm['macro_f1']:.3f}")

    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=["google", "perplexity"]))

    return metrics


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.1f}%"


def write_showcase_summary(metrics: dict, output_dir: Path, calibration: dict | None) -> None:
    lines = ["# JieHuo showcase metrics", ""]
    if calibration:
        lines.extend(
            [
                "## Calibration",
                "",
                f"- Temperature: `{calibration['temperature']:.3f}` fit on `{calibration['n']}` validation rows.",
                "",
            ]
        )

    for split_name, m in metrics.items():
        best = m.get("best_at_min_coverage")
        lines.extend(
            [
                f"## {split_name}",
                "",
                f"- Full coverage: accuracy `{m['accuracy']:.4f}`, macro F1 `{m['macro_f1']:.4f}`, ECE `{m['ece']:.4f}`.",
            ]
        )
        if best:
            lines.append(
                f"- Best selective point above the coverage floor: threshold `{best['threshold']:.2f}`, "
                f"coverage `{pct(best['coverage'])}`, accuracy `{best['accuracy']:.4f}`, "
                f"macro F1 `{best['macro_f1']:.4f}`."
            )
        for target, point in m.get("operating_points", {}).items():
            if point:
                lines.append(
                    f"- `{target}`: coverage `{pct(point['coverage'])}` at threshold "
                    f"`{point['threshold']:.2f}` with macro F1 `{point['macro_f1']:.4f}`."
                )
        hybrid = m.get("hybrid")
        if hybrid:
            lines.append(
                f"- Hybrid deterministic-intent override: {pct(hybrid['heuristic_coverage'])} handled by rules, "
                f"accuracy `{hybrid['accuracy']:.4f}`, macro F1 `{hybrid['macro_f1']:.4f}`."
            )
        lines.append("")

    (output_dir / "showcase.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate query router model")
    parser.add_argument("--model", type=str, default=str(CHECKPOINT_DIR / "best"))
    parser.add_argument("--test", type=str, default=str(LABELED_DIR / "test.jsonl"))
    parser.add_argument("--gold", type=str, default=str(GOLD_DIR / "gold.jsonl"))
    parser.add_argument("--output", type=str, default=str(CHECKPOINT_DIR / "eval"))
    parser.add_argument("--confidence-thresholds", type=str, default="0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--calibration", type=str, default=None, help="Optional validation JSONL for temperature scaling")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hybrid-heuristic-threshold", type=float, default=0.9)
    parser.add_argument("--target-accuracy", type=str, default="0.85,0.90,0.925")
    parser.add_argument("--target-coverage", type=float, default=0.85)
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
    thresholds = [float(t) for t in args.confidence_thresholds.split(",") if t.strip()]
    target_accuracy = [float(t) for t in args.target_accuracy.split(",") if t.strip()]

    calibration = None
    temperature = 1.0
    if args.calibration:
        calibration_path = Path(args.calibration)
        if calibration_path.exists():
            calibration_records = list(read_jsonl(calibration_path))
            calibration = fit_temperature(
                model,
                tokenizer,
                calibration_records,
                device,
                args.max_length,
            )
            temperature = calibration["temperature"]
            print(
                f"Calibration temperature: {temperature:.3f} "
                f"(n={calibration['n']}, nll={calibration['nll']:.4f})"
            )
        else:
            print(f"Calibration file not found: {calibration_path}")

    all_metrics = {}
    for split_name, path in [("test", args.test), ("gold", args.gold)]:
        p = Path(path)
        if p.exists():
            records = list(read_jsonl(p))
            all_metrics[split_name] = evaluate_split(
                model,
                tokenizer,
                records,
                device,
                split_name,
                output_dir,
                thresholds,
                args.bootstrap_samples,
                temperature,
                args.hybrid_heuristic_threshold,
                target_accuracy,
                args.target_coverage,
            )

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(all_metrics, f, indent=2)
    write_showcase_summary(all_metrics, output_dir, calibration)

    print(f"\nMetrics saved to {output_dir / 'metrics.json'}")
    print(f"Showcase summary saved to {output_dir / 'showcase.md'}")


if __name__ == "__main__":
    main()
