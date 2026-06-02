#!/usr/bin/env python3
"""Collect raw queries from public datasets via HuggingFace datasets."""

from __future__ import annotations

import argparse
import random

from tqdm import tqdm

import ml._bootstrap  # noqa: F401
from ml.common import LANGUAGES, RAW_DIR, ensure_dirs, normalize_query, write_jsonl


def _collect_from_dataset(
    name: str,
    limit: int,
    language: str,
    source: str,
    query_field: str,
    config: str | None = None,
    split: str = "train",
    desc: str | None = None,
) -> list[dict]:
    from datasets import load_dataset

    records: list[dict] = []
    try:
        kwargs: dict = {"split": split, "streaming": True}
        if config:
            kwargs["name"] = config
        ds = load_dataset(name, **kwargs)
        for row in tqdm(ds, desc=desc or source, total=limit):
            query = row.get(query_field, "")
            if isinstance(query, dict):
                query = query.get("text", "")
            query = str(query).strip()
            if len(query) >= 3:
                records.append({"query": query, "language": language, "source": source})
            if len(records) >= limit:
                break
    except Exception as e:
        print(f"Warning: {source} failed: {e}")
    return records


def collect_ms_marco(limit: int) -> list[dict]:
    return _collect_from_dataset(
        "microsoft/ms_marco", limit, "en", "ms_marco", "query", config="v1.1"
    )


def collect_ms_marco_v2(limit: int) -> list[dict]:
    return _collect_from_dataset(
        "microsoft/ms_marco", limit, "en", "ms_marco_v2", "query", config="v2.1"
    )


def collect_natural_questions(limit: int) -> list[dict]:
    from datasets import load_dataset

    records: list[dict] = []
    try:
        ds = load_dataset(
            "google-research-datasets/natural_questions", "default", split="train", streaming=True
        )
        for row in tqdm(ds, desc="natural_questions", total=limit):
            question = row.get("question", {})
            text = question.get("text", "") if isinstance(question, dict) else str(question)
            text = text.strip()
            if text:
                records.append({"query": text, "language": "en", "source": "natural_questions"})
            if len(records) >= limit:
                break
    except Exception as e:
        print(f"Warning: natural_questions failed: {e}")
    return records


def collect_trivia_qa(limit: int) -> list[dict]:
    return _collect_from_dataset(
        "trivia_qa", limit, "en", "trivia_qa", "question", config="rc"
    )


def collect_squad(limit: int) -> list[dict]:
    return _collect_from_dataset(
        "rajpurkar/squad_v2", limit, "en", "squad_v2", "question"
    )


def collect_wiki_qa(limit: int) -> list[dict]:
    return _collect_from_dataset("wiki_qa", limit, "en", "wiki_qa", "question")


def collect_gooaq(limit: int) -> list[dict]:
    return _collect_from_dataset(
        "sentence-transformers/gooaq", limit, "en", "gooaq", "question"
    )


def collect_xquad(limit_per_lang: int) -> list[dict]:
    """Multilingual questions from XQuAD."""
    lang_map = {
        "en": "xquad.en",
        "de": "xquad.de",
        "es": "xquad.es",
        "fr": "xquad.fr",
        "hi": "xquad.hi",
        "ar": "xquad.ar",
        "ja": "xquad.ja",
        "ru": "xquad.ru",
        "zh-CN": "xquad.zh",
        "zh-TW": "xquad.zh",  # closest available
        "pt": "xquad.es",  # fallback — no pt split
        "ko": "xquad.ja",  # fallback
    }

    records: list[dict] = []
    seen_configs: set[str] = set()
    for lang_code, config in lang_map.items():
        if config in seen_configs and lang_code not in ("zh-CN", "zh-TW"):
            continue
        seen_configs.add(config)
        records.extend(
            _collect_from_dataset(
                "xquad",
                limit_per_lang,
                lang_code,
                f"xquad_{config.split('.')[-1]}",
                "question",
                config=config,
                split="validation",
                desc=f"xquad-{config}",
            )
        )
    return records


def collect_mmarco(limit_per_lang: int) -> list[dict]:
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
            for row in tqdm(ds, desc=f"mmarco-{config}", total=limit_per_lang):
                query = row.get("query", "").strip()
                if query:
                    records.append({"query": query, "language": lang_code, "source": "mmarco"})
                    count += 1
                if count >= limit_per_lang:
                    break
        except Exception as e:
            print(f"Warning: mmarco {config} failed: {e}")
    return records


def collect_miracl(limit_per_lang: int) -> list[dict]:
    from datasets import load_dataset

    miracl_langs = {
        "ar": "ar", "de": "de", "es": "es", "fr": "fr", "hi": "hi",
        "ja": "ja", "ko": "ko", "pt": "pt", "ru": "ru", "zh-CN": "zh",
    }
    records: list[dict] = []
    for lang_code, code in miracl_langs.items():
        try:
            ds = load_dataset("miracl/miracl", code, split="dev", streaming=True)
            count = 0
            for row in tqdm(ds, desc=f"miracl-{code}", total=limit_per_lang):
                query = row.get("query", "").strip()
                if query:
                    records.append({"query": query, "language": lang_code, "source": "miracl"})
                    count += 1
                if count >= limit_per_lang:
                    break
        except Exception as e:
            print(f"Warning: miracl {code} failed: {e}")
    return records


def dedupe_records(records: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for r in records:
        key = (r.get("language", "unknown"), normalize_query(r["query"]))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect public dataset queries")
    parser.add_argument("--ms-marco-limit", type=int, default=8000)
    parser.add_argument("--ms-marco-v2-limit", type=int, default=5000)
    parser.add_argument("--nq-limit", type=int, default=5000)
    parser.add_argument("--trivia-limit", type=int, default=3000)
    parser.add_argument("--squad-limit", type=int, default=3000)
    parser.add_argument("--wiki-qa-limit", type=int, default=2000)
    parser.add_argument("--gooaq-limit", type=int, default=5000)
    parser.add_argument("--xquad-limit", type=int, default=400, help="Per XQuAD language config")
    parser.add_argument("--mmarco-limit", type=int, default=800, help="Per language")
    parser.add_argument("--miracl-limit", type=int, default=400, help="Per language")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    ensure_dirs()

    all_records: list[dict] = []
    all_records.extend(collect_ms_marco(args.ms_marco_limit))
    all_records.extend(collect_ms_marco_v2(args.ms_marco_v2_limit))
    all_records.extend(collect_natural_questions(args.nq_limit))
    all_records.extend(collect_trivia_qa(args.trivia_limit))
    all_records.extend(collect_squad(args.squad_limit))
    all_records.extend(collect_wiki_qa(args.wiki_qa_limit))
    all_records.extend(collect_gooaq(args.gooaq_limit))
    all_records.extend(collect_xquad(args.xquad_limit))
    all_records.extend(collect_mmarco(args.mmarco_limit))
    all_records.extend(collect_miracl(args.miracl_limit))

    unique = dedupe_records(all_records)
    random.shuffle(unique)
    out_path = RAW_DIR / "public_queries.jsonl"
    write_jsonl(out_path, unique)

    lang_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for r in unique:
        lang_counts[r["language"]] = lang_counts.get(r["language"], 0) + 1
        source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1

    print(f"\nCollected {len(unique)} unique queries -> {out_path}")
    print("By language:")
    for lang in LANGUAGES:
        if lang in lang_counts:
            print(f"  {lang}: {lang_counts[lang]}")
    print("By source:")
    for src, n in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {n}")


if __name__ == "__main__":
    main()
