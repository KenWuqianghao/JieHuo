#!/usr/bin/env python3
"""Merge labels, deduplicate, resolve disagreements, and create train/val/test splits."""

from __future__ import annotations

import ml._bootstrap  # noqa: F401

import argparse
import random
from collections import defaultdict
from pathlib import Path

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


def resolve_label(record: dict, min_confidence: float = 0.6) -> dict | None:
    """Resolve final label from heuristic + LLM labels."""
    h_label = record.get("heuristic_label")
    h_conf = record.get("heuristic_confidence", 0)
    l_label = record.get("llm_label")
    l_conf = record.get("llm_confidence", 0)

    # Synthetic queries with expected_label get priority if high confidence
    if record.get("source") == "synthetic" and record.get("expected_label"):
        return {
            **record,
            "label": record["expected_label"],
            "label_id": LABEL_PERPLEXITY if record["expected_label"] == "perplexity" else LABEL_GOOGLE,
            "confidence": 0.85,
            "resolution": "synthetic_expected",
        }

    # Agreement
    if h_label == l_label:
        conf = max(h_conf, l_conf)
        if conf >= min_confidence:
            return {
                **record,
                "label": h_label,
                "label_id": LABEL_PERPLEXITY if h_label == "perplexity" else LABEL_GOOGLE,
                "confidence": conf,
                "resolution": "agreement",
            }

    # Disagreement — trust higher confidence
    if h_conf >= l_conf and h_conf >= min_confidence:
        return {
            **record,
            "label": h_label,
            "label_id": LABEL_PERPLEXITY if h_label == "perplexity" else LABEL_GOOGLE,
            "confidence": h_conf * 0.85,
            "resolution": "heuristic_wins",
        }
    if l_conf >= min_confidence:
        return {
            **record,
            "label": l_label,
            "label_id": LABEL_PERPLEXITY if l_label == "perplexity" else LABEL_GOOGLE,
            "confidence": l_conf * 0.85,
            "resolution": "llm_wins",
        }

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
    """Sample stratified gold eval set from resolved records."""
    per_lang = max(1, size // len(LANGUAGES))
    gold: list[dict] = []
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_lang[r["language"]].append(r)

    for lang in LANGUAGES:
        pool = by_lang.get(lang, [])
        random.shuffle(pool)
        gold.extend(pool[:per_lang])

    # Fill remaining from any language
    if len(gold) < size:
        remaining = [r for r in records if r not in gold]
        random.shuffle(remaining)
        gold.extend(remaining[: size - len(gold)])

    return gold[:size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and split labeled data")
    parser.add_argument("--input", type=str, default=str(LABELED_DIR / "llm_labeled.jsonl"))
    parser.add_argument("--min-confidence", type=float, default=0.6)
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
    gold_queries = {normalize_query(g["query"]) for g in gold}
    train_pool = [r for r in resolved if normalize_query(r["query"]) not in gold_queries]

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
