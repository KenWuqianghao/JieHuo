#!/usr/bin/env python3
"""Fine-tune multilingual-e5-small for query routing."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import ml._bootstrap  # noqa: F401
from ml.common import (
    CHECKPOINT_DIR,
    E5_QUERY_PREFIX,
    LABELED_DIR,
    MODEL_NAME,
    ensure_dirs,
    read_jsonl,
)


class QueryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int = 128,
        sample_weights: list[float] | None = None,
        use_soft_labels: bool = False,
        soft_label_floor: float = 0.02,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample_weights = sample_weights
        self.use_soft_labels = use_soft_labels
        self.soft_label_floor = soft_label_floor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        text = E5_QUERY_PREFIX + r["query"]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(r["label_id"], dtype=torch.long),
        }
        if self.sample_weights is not None:
            item["sample_weight"] = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
        if self.use_soft_labels:
            confidence = max(0.5, min(1.0, float(r.get("confidence", 1.0))))
            confidence = min(confidence, 1.0 - self.soft_label_floor)
            target = torch.full((2,), self.soft_label_floor, dtype=torch.float32)
            target[r["label_id"]] = confidence
            target[1 - r["label_id"]] = 1.0 - confidence
            item["soft_labels"] = target / target.sum()
        return item


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "google_f1": f1_score(labels, preds, labels=[0], average="macro", zero_division=0),
        "perplexity_f1": f1_score(labels, preds, labels=[1], average="macro", zero_division=0),
    }


def compute_class_weights(records: list[dict]) -> torch.Tensor:
    counts = [0, 0]
    for r in records:
        counts[r["label_id"]] += 1
    total = sum(counts)
    weights = [total / (2 * c) if c > 0 else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(
    records: list[dict],
    language_weight_power: float = 0.0,
    disagreement_weight: float = 1.0,
) -> list[float]:
    """Weight examples by confidence, with optional damped language-label balancing."""
    counts = Counter((r.get("language", "unknown"), r["label_id"]) for r in records)
    raw_weights: list[float] = []
    for r in records:
        balance = 1.0
        if language_weight_power > 0:
            key = (r.get("language", "unknown"), r["label_id"])
            balance = (len(records) / (len(counts) * counts[key])) ** language_weight_power
            balance = min(balance, 3.0)
        confidence = 0.5 + 0.5 * float(r.get("confidence", 1.0))
        agreement = 1.0 if r.get("teacher_agreement", True) else disagreement_weight
        raw_weights.append(balance * confidence * agreement)

    mean_weight = sum(raw_weights) / max(len(raw_weights), 1)
    return [w / mean_weight for w in raw_weights]


def cap_records_per_bucket(records: list[dict], max_per_bucket: int, seed: int) -> list[dict]:
    """Cap overrepresented language/label buckets while keeping sparse buckets intact."""
    if max_per_bucket <= 0:
        return records

    rng = random.Random(seed)
    buckets: dict[tuple[str, int], list[dict]] = {}
    for record in records:
        key = (record.get("language", "unknown"), record["label_id"])
        buckets.setdefault(key, []).append(record)

    capped: list[dict] = []
    for bucket_records in buckets.values():
        shuffled = list(bucket_records)
        rng.shuffle(shuffled)
        capped.extend(shuffled[:max_per_bucket])

    rng.shuffle(capped)
    return capped


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, label_smoothing: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        sample_weight = inputs.pop("sample_weight", None)
        soft_labels = inputs.pop("soft_labels", None)
        outputs = model(**inputs)
        logits = outputs.logits
        if soft_labels is not None:
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(soft_labels.to(logits.device) * log_probs)
            if self.class_weights is not None:
                loss = loss * self.class_weights.to(logits.device)
            loss = loss.sum(dim=-1)
        elif self.class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device),
                label_smoothing=self.label_smoothing,
                reduction="none" if sample_weight is not None else "mean",
            )
            loss = loss_fn(logits, labels)
        else:
            loss_fn = torch.nn.CrossEntropyLoss(
                label_smoothing=self.label_smoothing,
                reduction="none" if sample_weight is not None else "mean",
            )
            loss = loss_fn(logits, labels)
        if sample_weight is not None:
            loss = (loss * sample_weight.to(logits.device)).mean()
        elif loss.ndim > 0:
            loss = loss.mean()
        return (loss, outputs) if return_outputs else loss


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train query router model")
    parser.add_argument("--train", type=str, default=str(LABELED_DIR / "train.jsonl"))
    parser.add_argument("--val", type=str, default=str(LABELED_DIR / "val.jsonl"))
    parser.add_argument("--output", type=str, default=str(CHECKPOINT_DIR / "best"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--min-train-confidence", type=float, default=0.0)
    parser.add_argument(
        "--max-train-per-language-label",
        type=int,
        default=0,
        help="Cap each (language, label) training bucket; 0 keeps all rows.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--language-weight-power", type=float, default=0.0)
    parser.add_argument("--disagreement-weight", type=float, default=1.0)
    parser.add_argument("--soft-labels", action="store_true")
    parser.add_argument("--soft-label-floor", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_reproducible_seed(args.seed)
    ensure_dirs()

    train_path = Path(args.train)
    val_path = Path(args.val)
    if not train_path.exists():
        print(f"Train file not found: {train_path}. Run 05_merge_split.py first.")
        return

    train_records = list(read_jsonl(train_path))
    val_records = list(read_jsonl(val_path)) if val_path.exists() else []
    if args.min_train_confidence > 0:
        before = len(train_records)
        train_records = [
            r for r in train_records if float(r.get("confidence", 0.0)) >= args.min_train_confidence
        ]
        print(
            f"Filtered train by confidence >= {args.min_train_confidence}: "
            f"{len(train_records)} / {before}"
        )
    if args.max_train_per_language_label > 0:
        before = len(train_records)
        train_records = cap_records_per_bucket(
            train_records,
            args.max_train_per_language_label,
            args.seed,
        )
        print(
            f"Capped train buckets at {args.max_train_per_language_label}: "
            f"{len(train_records)} / {before}"
        )

    print(f"Training on {len(train_records)} samples, validating on {len(val_records)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.config.id2label = {0: "google", 1: "perplexity"}
    model.config.label2id = {"google": 0, "perplexity": 1}

    sample_weights = compute_sample_weights(
        train_records,
        args.language_weight_power,
        args.disagreement_weight,
    )
    train_ds = QueryDataset(
        train_records,
        tokenizer,
        args.max_length,
        sample_weights=sample_weights,
        use_soft_labels=args.soft_labels,
        soft_label_floor=args.soft_label_floor,
    )
    val_ds = QueryDataset(val_records, tokenizer, args.max_length) if val_records else None

    class_weights = compute_class_weights(train_records)
    print(f"Class weights: google={class_weights[0]:.3f}, perplexity={class_weights[1]:.3f}")

    use_fp16 = torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR / "runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        eval_strategy="epoch" if val_ds else "no",
        save_strategy="epoch" if val_ds else "no",
        save_total_limit=1,
        load_best_model_at_end=bool(val_ds),
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_fp16 and torch.cuda.is_available(),
        logging_steps=50,
        seed=args.seed,
        report_to="none",
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)] if val_ds else []

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics if val_ds else None,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        callbacks=callbacks,
    )

    trainer.train()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save label mapping
    with (output_dir / "label_config.json").open("w") as f:
        json.dump({"id2label": model.config.id2label, "label2id": model.config.label2id}, f)

    train_config = {
        "base_model": MODEL_NAME,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "label_counts": dict(Counter(r["label"] for r in train_records)),
        "language_counts": dict(Counter(r.get("language", "unknown") for r in train_records)),
        "args": vars(args),
    }
    with (output_dir / "training_config.json").open("w") as f:
        json.dump(train_config, f, indent=2, ensure_ascii=False)

    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    main()
