#!/usr/bin/env python3
"""Fine-tune multilingual-e5-small for query routing."""

from __future__ import annotations

import ml._bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from ml.common import (
    CHECKPOINT_DIR,
    E5_QUERY_PREFIX,
    LABEL_NAMES,
    LABELED_DIR,
    MODEL_NAME,
    ensure_dirs,
    read_jsonl,
)


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int = 128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(r["label_id"], dtype=torch.long),
        }


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


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fn(logits, labels)
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train query router model")
    parser.add_argument("--train", type=str, default=str(LABELED_DIR / "train.jsonl"))
    parser.add_argument("--val", type=str, default=str(LABELED_DIR / "val.jsonl"))
    parser.add_argument("--output", type=str, default=str(CHECKPOINT_DIR / "best"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()

    train_path = Path(args.train)
    val_path = Path(args.val)
    if not train_path.exists():
        print(f"Train file not found: {train_path}. Run 05_merge_split.py first.")
        return

    train_records = list(read_jsonl(train_path))
    val_records = list(read_jsonl(val_path)) if val_path.exists() else []

    print(f"Training on {len(train_records)} samples, validating on {len(val_records)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.config.id2label = {0: "google", 1: "perplexity"}
    model.config.label2id = {"google": 0, "perplexity": 1}

    train_ds = QueryDataset(train_records, tokenizer, args.max_length)
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
        eval_strategy="epoch" if val_ds else "no",
        save_strategy="epoch" if val_ds else "no",
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

    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    main()
