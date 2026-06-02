#!/usr/bin/env python3
"""Batch LLM labeling with strict JSON rubric."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

import ml._bootstrap  # noqa: F401
from ml.common import LABELED_DIR, ensure_dirs, read_jsonl, write_jsonl
from ml.ollama_client import DEFAULT_OLLAMA_MODEL, call_ollama, extract_json, ollama_available

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


def call_ollama_batch(queries: list[str], model: str) -> list[dict]:
    numbered = "\n".join(f'{i + 1}. "{q}"' for i, q in enumerate(queries))
    content = call_ollama(
        RUBRIC + '\nReturn JSON: {"results": [{"query": "...", "label": "google"|"perplexity", "confidence": 0.0-1.0, "reason": "..."}]}',
        f"Classify these queries:\n{numbered}",
        model=model,
        temperature=0.1,
        json_mode=True,
    )
    parsed = extract_json(content)
    if isinstance(parsed, list):
        return parsed
    return parsed.get("results", [])


def call_openai_batch(queries: list[str], model: str) -> list[dict]:
    from openai import OpenAI

    kwargs: dict = {"api_key": os.environ.get("OPENAI_API_KEY")}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

    numbered = "\n".join(f'{i + 1}. "{q}"' for i, q in enumerate(queries))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RUBRIC},
            {
                "role": "user",
                "content": (
                    f"Classify these queries:\n{numbered}\n\n"
                    'Return JSON: {"results": [{"query": "...", "label": "google"|"perplexity", '
                    '"confidence": 0.0-1.0, "reason": "..."}]}'
                ),
            },
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


def is_stale_label(record: dict) -> bool:
    """Labels from an earlier fallback run should be replaced when Ollama is available."""
    reason = str(record.get("llm_reason", ""))
    return not reason or reason.startswith("heuristic_fallback")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM batch labeling")
    parser.add_argument("--input", type=str, default=str(LABELED_DIR / "heuristic_labeled.jsonl"))
    parser.add_argument("--output", type=str, default=str(LABELED_DIR / "llm_labeled.jsonl"))
    parser.add_argument("--provider", choices=["openai", "anthropic", "ollama", "fallback"], default="ollama")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip Ollama for synthetic rows (use expected_label instead)",
    )
    parser.add_argument(
        "--public-limit",
        type=int,
        default=None,
        help="Only Ollama-label first N public queries (rest get heuristics)",
    )
    parser.add_argument(
        "--relabel-stale",
        action="store_true",
        help="Re-run provider labels for existing heuristic_fallback/empty rows",
    )
    args = parser.parse_args()

    load_dotenv()
    ensure_dirs()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}. Run 03_heuristic_labels.py first.")
        return

    records = list(read_jsonl(input_path))
    ollama_model = args.model or DEFAULT_OLLAMA_MODEL
    output_path = Path(args.output)

    # Resume support
    labeled: list[dict] = []
    done_keys: set[tuple[str, str]] = set()
    if output_path.exists():
        existing = list(read_jsonl(output_path))
        if args.relabel_stale:
            stale_count = sum(1 for r in existing if is_stale_label(r))
            existing = [r for r in existing if not is_stale_label(r)]
            print(f"Relabel stale rows: {stale_count}")
        labeled = existing
        done_keys = {
            (r.get("language", ""), r["query"].strip().lower()) for r in labeled
        }
        print(f"Resuming: {len(labeled)} already labeled")

    # Build work queue
    to_label: list[dict] = []
    pre_labeled: list[dict] = []

    public_seen = 0
    for r in records:
        key = (r.get("language", ""), r["query"].strip().lower())
        if key in done_keys:
            continue

        is_synthetic = r.get("source") == "synthetic"
        is_public = not is_synthetic

        if args.skip_synthetic and is_synthetic:
            pre_labeled.append(
                {
                    **r,
                    "llm_label": r.get("expected_label", r.get("heuristic_label", "google")),
                    "llm_confidence": 0.85,
                    "llm_reason": "synthetic_expected",
                }
            )
            continue

        if is_public and args.public_limit is not None and public_seen >= args.public_limit:
            pre_labeled.append({**r, **label_with_heuristic_fallback([r])[0]})
            continue

        if is_public:
            public_seen += 1
        to_label.append(r)

    labeled.extend(pre_labeled)

    has_api = (
        (args.provider == "openai" and os.environ.get("OPENAI_API_KEY"))
        or (args.provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"))
        or (args.provider == "ollama" and ollama_available())
    )

    if not has_api or args.provider == "fallback":
        print("Using heuristic labels as LLM labels")
        labeled.extend(label_with_heuristic_fallback(to_label))
        write_jsonl(output_path, labeled)
        print(f"Labeled {len(labeled)} queries -> {output_path}")
        return

    if args.provider == "ollama":
        print(f"Using Ollama model: {ollama_model}")
        print(f"  To Ollama-label: {len(to_label)} queries")
        print(f"  Pre-labeled (synthetic/heuristic/resume): {len(pre_labeled)}")
    elif args.provider == "openai":
        model_name = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        print(f"Using OpenAI model: {model_name}")
        if os.environ.get("OPENAI_BASE_URL"):
            print(f"  Base URL: {os.environ.get('OPENAI_BASE_URL')}")
        print(f"  To label: {len(to_label)} queries")
        print(f"  Pre-labeled: {len(pre_labeled)}")

    start = len(labeled)
    for i in tqdm(range(0, len(to_label), args.batch_size), desc="LLM labeling"):
        batch = to_label[i : i + args.batch_size]
        queries = [r["query"] for r in batch]

        try:
            if args.provider == "openai":
                model_name = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
                results = call_openai_batch(queries, model=model_name)
            elif args.provider == "ollama":
                results = call_ollama_batch(queries, model=ollama_model)
            else:
                results = call_anthropic_batch(queries, model=args.model or "claude-3-5-haiku-20241022")

            result_map = {r.get("query", "").strip().lower(): r for r in results if isinstance(r, dict)}

            for j, rec in enumerate(batch):
                llm = result_map.get(rec["query"].strip().lower(), {})
                if not llm and j < len(results) and isinstance(results[j], dict):
                    llm = results[j]
                reason = llm.get("reason", "")
                if reason and args.provider == "openai":
                    reason = f"api:{reason}"
                labeled.append(
                    {
                        **rec,
                        "llm_label": llm.get("label", rec.get("heuristic_label", "google")),
                        "llm_confidence": float(llm.get("confidence", 0.7)),
                        "llm_reason": reason,
                    }
                )
            write_jsonl(output_path, labeled)
            time.sleep(args.sleep)
        except Exception as e:
            print(f"Warning: batch {i} failed: {e}")
            labeled.extend(label_with_heuristic_fallback(batch))
            write_jsonl(output_path, labeled)

    print(f"\nLLM labeled {len(labeled)} queries ({len(labeled) - start} new) -> {output_path}")


if __name__ == "__main__":
    main()
