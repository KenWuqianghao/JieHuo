#!/usr/bin/env python3
"""Batch LLM labeling with strict JSON rubric."""

from __future__ import annotations

import ml._bootstrap  # noqa: F401

import argparse
import json
import os
import time
from pathlib import Path

from tqdm import tqdm

from ml.common import LABELED_DIR, ensure_dirs, read_jsonl, write_jsonl

RUBRIC = """You classify search queries into exactly one routing destination:

GOOGLE — route to Google Search when the user wants:
- A specific website, app, or brand (navigational)
- To buy, download, or complete a transaction
- Local/nearby results (maps, directions, "near me")
- Real-time data (weather, stocks, sports scores, flight status)
- Simple factoid with a short answer
- Code syntax/snippets or image/video results

PERPLEXITY — route to Perplexity when the user wants:
- Open-ended explanation or "how/why" questions
- Comparison or evaluation of options
- Research synthesis across multiple sources
- News analysis or current affairs summary
- Multi-step reasoning or opinion/recommendation

Return JSON array with one object per query:
[{"query": "...", "label": "google"|"perplexity", "confidence": 0.0-1.0, "reason": "brief reason"}]"""

BATCH_SIZE = 20


def call_openai_batch(queries: list[str], model: str) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"Classify these queries:\n{numbered}"},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    if isinstance(parsed, list):
        return parsed
    for key in ("results", "queries", "classifications", "data"):
        if key in parsed and isinstance(parsed[key], list):
            return parsed[key]
    return []


def call_anthropic_batch(queries: list[str], model: str) -> list[dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=RUBRIC,
        messages=[
            {
                "role": "user",
                "content": f"Classify these queries:\n{numbered}\n\nReturn: {{\"results\": [...]}}",
            }
        ],
    )
    content = response.content[0].text
    parsed = json.loads(content)
    if isinstance(parsed, list):
        return parsed
    return parsed.get("results", [])


def label_with_heuristic_fallback(records: list[dict]) -> list[dict]:
    """When no API key, promote heuristic labels to LLM labels."""
    labeled = []
    for r in records:
        labeled.append(
            {
                **r,
                "llm_label": r.get("heuristic_label", "google"),
                "llm_confidence": r.get("heuristic_confidence", 0.5),
                "llm_reason": "heuristic_fallback: " + ", ".join(r.get("heuristic_reasons", [])),
            }
        )
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM batch labeling")
    parser.add_argument("--input", type=str, default=str(LABELED_DIR / "heuristic_labeled.jsonl"))
    parser.add_argument("--output", type=str, default=str(LABELED_DIR / "llm_labeled.jsonl"))
    parser.add_argument("--provider", choices=["openai", "anthropic", "fallback"], default="fallback")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="Limit records for testing")
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    ensure_dirs()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}. Run 03_heuristic_labels.py first.")
        return

    records = list(read_jsonl(input_path))
    if args.limit:
        records = records[: args.limit]

    has_api = (
        (args.provider == "openai" and os.environ.get("OPENAI_API_KEY"))
        or (args.provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"))
    )

    if not has_api or args.provider == "fallback":
        print("No API key or fallback mode — using heuristic labels as LLM labels")
        labeled = label_with_heuristic_fallback(records)
        write_jsonl(Path(args.output), labeled)
        print(f"Labeled {len(labeled)} queries -> {args.output}")
        return

    labeled: list[dict] = []
    for i in tqdm(range(0, len(records), args.batch_size), desc="LLM labeling"):
        batch = records[i : i + args.batch_size]
        queries = [r["query"] for r in batch]

        try:
            if args.provider == "openai":
                results = call_openai_batch(queries, model=args.model or "gpt-4o-mini")
            else:
                results = call_anthropic_batch(queries, model=args.model or "claude-3-5-haiku-20241022")

            result_map = {r.get("query", "").strip().lower(): r for r in results}

            for rec in batch:
                llm = result_map.get(rec["query"].strip().lower(), {})
                labeled.append(
                    {
                        **rec,
                        "llm_label": llm.get("label", rec.get("heuristic_label", "google")),
                        "llm_confidence": float(llm.get("confidence", 0.7)),
                        "llm_reason": llm.get("reason", ""),
                    }
                )
            time.sleep(args.sleep)
        except Exception as e:
            print(f"Warning: batch {i} failed: {e}")
            labeled.extend(label_with_heuristic_fallback(batch))

    write_jsonl(Path(args.output), labeled)
    print(f"\nLLM labeled {len(labeled)} queries -> {args.output}")


if __name__ == "__main__":
    main()
