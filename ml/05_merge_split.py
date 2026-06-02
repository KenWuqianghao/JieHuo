#!/usr/bin/env python3
"""Merge labels, deduplicate, resolve disagreements, and create train/val/test splits."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import ml._bootstrap  # noqa: F401
from ml.common import (
    GOLD_DIR,
    LABEL_GOOGLE,
    LABEL_NAMES,
    LABEL_PERPLEXITY,
    LABELED_DIR,
    LANGUAGES,
    ensure_dirs,
    normalize_query,
    read_jsonl,
    write_jsonl,
)

VALID_LABELS = set(LABEL_NAMES.values())


def normalize_label(label: object) -> str:
    return str(label or "").strip().lower()


def label_id(label: str) -> int:
    return LABEL_PERPLEXITY if label == "perplexity" else LABEL_GOOGLE


def resolved_record(record: dict, label: str, confidence: float, resolution: str) -> dict | None:
    label = normalize_label(label)
    if label not in VALID_LABELS:
        return None
    h_label = normalize_label(record.get("heuristic_label"))
    h_conf = float(record.get("heuristic_confidence", 0) or 0)
    l_conf = float(record.get("llm_confidence", 0) or 0)
    return {
        **record,
        "label": label,
        "label_id": label_id(label),
        "confidence": max(0.0, min(1.0, float(confidence))),
        "resolution": resolution,
        "teacher_agreement": bool(h_label == label),
        "teacher_margin": abs(h_conf - l_conf),
    }


def resolve_label(record: dict, min_confidence: float = 0.6) -> dict | None:
    """Resolve final label from heuristic + LLM labels."""
    h_label = normalize_label(record.get("heuristic_label"))
    h_conf = float(record.get("heuristic_confidence", 0) or 0)
    l_label = normalize_label(record.get("llm_label"))
    l_conf = float(record.get("llm_confidence", 0) or 0)

    # Synthetic queries have an explicit intended routing bucket.
    expected_label = normalize_label(record.get("expected_label"))
    if record.get("source") == "synthetic" and expected_label in VALID_LABELS:
        return resolved_record(record, expected_label, 0.85, "synthetic_expected")

    # LLM-labeled rows: trust Ollama/OpenAI at the configured confidence threshold.
    if record.get("llm_reason") and not str(record.get("llm_reason", "")).startswith("heuristic_fallback"):
        if l_label in VALID_LABELS and l_conf >= min_confidence:
            resolution = "teacher_agreement" if h_label == l_label else "llm_primary"
            return resolved_record(record, l_label, l_conf, resolution)

    # Agreement
    if h_label == l_label and h_label in VALID_LABELS:
        conf = max(h_conf, l_conf)
        if conf >= min_confidence:
            return resolved_record(record, h_label, conf, "agreement")

    # Disagreement — trust higher confidence
    if h_label in VALID_LABELS and h_conf >= l_conf and h_conf >= min_confidence:
        return resolved_record(record, h_label, h_conf * 0.85, "heuristic_wins")
    if l_label in VALID_LABELS and l_conf >= min_confidence:
        return resolved_record(record, l_label, l_conf * 0.85, "llm_wins")

    # Low confidence disagreement — drop
    return None


def stratified_split(
    records: list[dict], train_ratio: float = 0.8, val_ratio: float = 0.1
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified split by (language, label)."""
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        buckets[(r["language"], r["label"])].append(r)

    train, val, test = [], [], []
    for bucket_records in buckets.values():
        random.shuffle(bucket_records)
        n = len(bucket_records)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(bucket_records[:n_train])
        val.extend(bucket_records[n_train : n_train + n_val])
        test.extend(bucket_records[n_train + n_val :])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def create_gold_set(records: list[dict], size: int = 500) -> list[dict]:
    """Sample a language- and label-balanced held-out set from resolved records."""
    labels = list(LABEL_NAMES.values())
    per_bucket = max(1, size // (len(LANGUAGES) * len(labels)))
    gold: list[dict] = []
    by_bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_bucket[(r["language"], r["label"])].append(r)

    for lang in LANGUAGES:
        for label in labels:
            pool = by_bucket.get((lang, label), [])
            random.shuffle(pool)
            gold.extend(pool[:per_bucket])

    # Fill remaining from any language
    if len(gold) < size:
        gold_keys = {(g.get("language", ""), normalize_query(g["query"])) for g in gold}
        remaining = [
            r
            for r in records
            if (r.get("language", ""), normalize_query(r["query"])) not in gold_keys
        ]
        random.shuffle(remaining)
        gold.extend(remaining[: size - len(gold)])

    return gold[:size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and split labeled data")
    parser.add_argument("--input", type=str, default=str(LABELED_DIR / "llm_labeled.jsonl"))
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--gold-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    ensure_dirs()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}. Run 04_llm_label.py first.")
        return

    raw_records = list(read_jsonl(input_path))

    # Deduplicate within each language
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for r in raw_records:
        key = (r.get("language", "unknown"), normalize_query(r["query"]))
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    # Resolve labels
    resolved: list[dict] = []
    dropped = 0
    for r in deduped:
        result = resolve_label(r, min_confidence=args.min_confidence)
        if result:
            resolved.append(result)
        else:
            dropped += 1

    # Create gold set before splitting (held out from train)
    gold_size = min(args.gold_size, max(10, len(resolved) // 10))
    gold = create_gold_set(resolved, size=gold_size)
    gold_queries = {(g.get("language", ""), normalize_query(g["query"])) for g in gold}
    train_pool = [
        r
        for r in resolved
        if (r.get("language", ""), normalize_query(r["query"])) not in gold_queries
    ]

    if len(train_pool) < 10:
        # Small dataset: use all for train, gold is a copy for eval only
        train_pool = resolved
        gold = create_gold_set(resolved, size=min(gold_size, len(resolved)))

    train, val, test = stratified_split(train_pool)

    write_jsonl(LABELED_DIR / "train.jsonl", train)
    write_jsonl(LABELED_DIR / "val.jsonl", val)
    write_jsonl(LABELED_DIR / "test.jsonl", test)
    write_jsonl(GOLD_DIR / "gold.jsonl", gold)

    print(f"Deduplicated: {len(deduped)} (from {len(raw_records)})")
    print(f"Resolved: {len(resolved)}, Dropped (low confidence): {dropped}")
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}, gold={len(gold)}")

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        g = sum(1 for r in split_data if r["label"] == "google")
        p = len(split_data) - g
        print(f"  {split_name}: google={g} ({100*g/max(len(split_data),1):.1f}%), perplexity={p}")


if __name__ == "__main__":
    main()
