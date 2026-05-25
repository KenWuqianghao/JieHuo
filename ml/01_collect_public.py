#!/usr/bin/env python3
"""Collect raw queries from public datasets via HuggingFace datasets."""

from __future__ import annotations

import ml._bootstrap  # noqa: F401

import argparse
import random

from tqdm import tqdm

from ml.common import LANGUAGES, RAW_DIR, ensure_dirs, write_jsonl


def collect_ms_marco(limit: int) -> list[dict]:
    """Collect English queries from MS MARCO."""
    from datasets import load_dataset

    records: list[dict] = []
    try:
        ds = load_dataset("microsoft/ms_marco", "v1.1", split="train", streaming=True)
        for row in tqdm(ds, desc="MS-MARCO", total=limit):
            query = row.get("query", "").strip()
            if query:
                records.append({"query": query, "language": "en", "source": "ms_marco"})
            if len(records) >= limit:
                break
    except Exception as e:
        print(f"Warning: MS-MARCO failed: {e}")
    return records


def collect_natural_questions(limit: int) -> list[dict]:
    """Collect English questions from Natural Questions."""
    from datasets import load_dataset

    records: list[dict] = []
    try:
        ds = load_dataset("google-research-datasets/natural_questions", "default", split="train", streaming=True)
        for row in tqdm(ds, desc="NaturalQuestions", total=limit):
            question = row.get("question", {}).get("text", "") if isinstance(row.get("question"), dict) else row.get("question", "")
            question = str(question).strip()
            if question:
                records.append({"query": question, "language": "en", "source": "natural_questions"})
            if len(records) >= limit:
                break
    except Exception as e:
        print(f"Warning: Natural Questions failed: {e}")
    return records


def collect_mlqa(limit_per_lang: int) -> list[dict]:
    """Collect multilingual questions from MLQA."""
    from datasets import load_dataset

    lang_map = {
        "zh-CN": "zh",
        "es": "es",
        "de": "de",
        "ar": "ar",
        "hi": "hi",
        "vi": "vi",
    }

    records: list[dict] = []
    for lang_code, mlqa_code in lang_map.items():
        try:
            config = f"mlqa-translate-train.{mlqa_code}"
            ds = load_dataset("mlqa", config, split="train", streaming=True)
            count = 0
            for row in tqdm(ds, desc=f"MLQA-{mlqa_code}", total=limit_per_lang):
                question = row.get("question", "").strip()
                if question:
                    records.append({"query": question, "language": lang_code, "source": "mlqa"})
                    count += 1
                if count >= limit_per_lang:
                    break
        except Exception as e:
            print(f"Warning: MLQA {mlqa_code} failed: {e}")
    return records


def collect_mmarco(limit_per_lang: int) -> list[dict]:
    """Collect multilingual queries from mMARCO."""
    from datasets import load_dataset

    lang_map = {
        "es": "spanish",
        "fr": "french",
        "de": "german",
        "ja": "japanese",
        "pt": "portuguese",
        "ru": "russian",
        "zh-CN": "chinese_simplified",
        "zh-TW": "chinese_traditional",
    }

    records: list[dict] = []
    for lang_code, config in lang_map.items():
        try:
            ds = load_dataset("unicamp-dl/mmarco", config, split="train", streaming=True)
            count = 0
            for row in tqdm(ds, desc=f"mMARCO-{config}", total=limit_per_lang):
                query = row.get("query", "").strip()
                if query:
                    records.append({"query": query, "language": lang_code, "source": "mmarco"})
                    count += 1
                if count >= limit_per_lang:
                    break
        except Exception as e:
            print(f"Warning: mMARCO {config} failed: {e}")
    return records


def collect_orcas(limit: int) -> list[dict]:
    """Collect English queries from ORCAS (Bing click logs)."""
    from datasets import load_dataset

    records: list[dict] = []
    try:
        ds = load_dataset("microsoft/orcas", split="train", streaming=True)
        for row in tqdm(ds, desc="ORCAS", total=limit):
            query = row.get("Query", row.get("query", "")).strip()
            if query:
                records.append({"query": query, "language": "en", "source": "orcas"})
            if len(records) >= limit:
                break
    except Exception as e:
        print(f"Warning: ORCAS failed: {e}")
    return records


def collect_miracl(limit_per_lang: int) -> list[dict]:
    """Collect multilingual queries from MIRACL."""
    from datasets import load_dataset

    miracl_langs = {
        "ar": "ar",
        "de": "de",
        "es": "es",
        "fr": "fr",
        "hi": "hi",
        "ja": "ja",
        "ko": "ko",
        "pt": "pt",
        "ru": "ru",
        "zh-CN": "zh",
    }

    records: list[dict] = []
    for lang_code, miracl_code in miracl_langs.items():
        try:
            ds = load_dataset("miracl/miracl", miracl_code, split="dev", streaming=True)
            count = 0
            for row in tqdm(ds, desc=f"MIRACL-{miracl_code}", total=limit_per_lang):
                query = row.get("query", "").strip()
                if query:
                    records.append({"query": query, "language": lang_code, "source": "miracl"})
                    count += 1
                if count >= limit_per_lang:
                    break
        except Exception as e:
            print(f"Warning: MIRACL {miracl_code} failed: {e}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect public dataset queries")
    parser.add_argument("--ms-marco-limit", type=int, default=5000)
    parser.add_argument("--mmarco-limit", type=int, default=1000, help="Per language")
    parser.add_argument("--orcas-limit", type=int, default=3000)
    parser.add_argument("--miracl-limit", type=int, default=500, help="Per language")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    ensure_dirs()

    all_records: list[dict] = []
    all_records.extend(collect_ms_marco(args.ms_marco_limit))
    all_records.extend(collect_natural_questions(min(3000, args.orcas_limit)))
    all_records.extend(collect_mlqa(args.miracl_limit))
    all_records.extend(collect_mmarco(args.mmarco_limit))
    all_records.extend(collect_orcas(args.orcas_limit))
    all_records.extend(collect_miracl(args.miracl_limit))

    seen: set[str] = set()
    unique: list[dict] = []
    for r in all_records:
        key = r["query"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)

    random.shuffle(unique)
    out_path = RAW_DIR / "public_queries.jsonl"
    write_jsonl(out_path, unique)

    lang_counts: dict[str, int] = {}
    for r in unique:
        lang_counts[r["language"]] = lang_counts.get(r["language"], 0) + 1

    print(f"\nCollected {len(unique)} unique queries -> {out_path}")
    print("By language:")
    for lang in LANGUAGES:
        if lang in lang_counts:
            print(f"  {lang}: {lang_counts[lang]}")


if __name__ == "__main__":
    main()
