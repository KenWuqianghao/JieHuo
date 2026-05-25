"""Shared constants and utilities for the ML pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

# Label mapping: 0 = google, 1 = perplexity
LABEL_GOOGLE = 0
LABEL_PERPLEXITY = 1
LABEL_NAMES = {LABEL_GOOGLE: "google", LABEL_PERPLEXITY: "perplexity"}

LANGUAGES = [
    "en",
    "zh-CN",
    "zh-TW",
    "es",
    "fr",
    "de",
    "ja",
    "ko",
    "pt",
    "ru",
    "ar",
    "hi",
]

INTENT_BUCKETS = [
    "navigational",
    "transactional",
    "local",
    "realtime",
    "factoid",
    "comparison",
    "explanation",
    "research",
    "code",
    "news",
]

E5_QUERY_PREFIX = "query: "
MODEL_NAME = "intfloat/multilingual-e5-small"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTH_DIR = DATA_DIR / "synthetic"
LABELED_DIR = DATA_DIR / "labeled"
GOLD_DIR = DATA_DIR / "gold"
CHECKPOINT_DIR = ROOT / "checkpoints"


def ensure_dirs() -> None:
    for d in [RAW_DIR, SYNTH_DIR, LABELED_DIR, GOLD_DIR, CHECKPOINT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def normalize_query(text: str) -> str:
    """Normalize query text for deduplication."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
