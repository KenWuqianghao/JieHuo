#!/usr/bin/env python3
"""Generate synthetic multilingual queries via LLM."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

from dotenv import load_dotenv
from tqdm import tqdm

import ml._bootstrap  # noqa: F401
from ml.common import (
    INTENT_BUCKETS,
    LANGUAGES,
    SYNTH_DIR,
    ensure_dirs,
    normalize_query,
    read_jsonl,
    write_jsonl,
)
from ml.ollama_client import (
    DEFAULT_OLLAMA_MODEL,
    call_ollama,
    extract_json,
    ollama_available,
    parse_query_list,
)

# Intent -> expected label hint for generation
INTENT_LABEL_HINT = {
    "navigational": "google",
    "transactional": "google",
    "local": "google",
    "realtime": "google",
    "factoid": "google",
    "comparison": "perplexity",
    "explanation": "perplexity",
    "research": "perplexity",
    "code": "google",  # code snippets often go to Google
    "news": "perplexity",
}

LANGUAGE_NAMES = {
    "en": "English",
    "zh-CN": "Simplified Chinese",
    "zh-TW": "Traditional Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
}

SYSTEM_PROMPT = """You generate realistic search queries that users would type into a search engine or AI assistant.
Return ONLY a JSON array of strings, no other text. Each query should be natural, varied in length (2-15 words),
and authentic to how real users search in the given language."""

USER_PROMPT_TEMPLATE = """Generate {count} realistic search queries in {language_name} for this intent category: {intent}.

Intent description:
- navigational: user wants a specific website, app, or brand (e.g. "facebook login", "github")
- transactional: user wants to buy, download, or complete an action (e.g. "buy iphone 16", "download spotify")
- local: user wants nearby places or directions (e.g. "coffee shops near me", "restaurants 附近")
- realtime: user wants current/live info (weather, stocks, sports scores, flight status)
- factoid: simple factual lookup with a short answer (e.g. "height of mount everest", "capital of france")
- comparison: user wants to compare options (e.g. "iphone vs samsung", "python vs javascript")
- explanation: user wants something explained (e.g. "how does photosynthesis work", "什么是区块链")
- research: open-ended research question requiring synthesis (e.g. "impact of AI on healthcare")
- code: user wants code snippet or syntax help (e.g. "python list comprehension example")
- news: user wants news summary or current events analysis (e.g. "latest developments in ukraine")

Expected routing hint: {label_hint} (but make queries realistic, not forced)

Return JSON: {{"queries": ["query1", "query2", ...]}} with exactly {count} strings.

IMPORTANT: Every query MUST be written entirely in {language_name}. Do not use English unless the language is English."""


def call_ollama_queries(prompt: str, model: str) -> list[str]:
    content = call_ollama(
        system=SYSTEM_PROMPT + ' Always respond with JSON: {"queries": ["...", "..."]}',
        user=prompt,
        model=model,
        temperature=0.9,
        json_mode=True,
    )
    parsed = extract_json(content)
    return parse_query_list(parsed)


def call_openai(prompt: str, model: str = "gpt-4o-mini") -> list[str]:
    from openai import OpenAI

    kwargs: dict = {"api_key": os.environ.get("OPENAI_API_KEY")}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("queries", "results", "data"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
    return []


def call_anthropic(prompt: str, model: str = "claude-3-5-haiku-20241022") -> list[str]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt + "\n\nReturn format: {\"queries\": [\"...\", \"...\"]}"}],
    )
    content = response.content[0].text
    parsed = json.loads(content)
    if isinstance(parsed, list):
        return parsed
    return parsed.get("queries", [])


def generate_fallback_queries(language: str, intent: str, count: int) -> list[str]:
    """Fallback templates when no API key is available."""
    lang_prefix = {
        "en": "",
        "zh-CN": "北京 ",
        "zh-TW": "台北 ",
        "es": "madrid ",
        "fr": "paris ",
        "de": "berlin ",
        "ja": "東京 ",
        "ko": "서울 ",
        "pt": "lisboa ",
        "ru": "москва ",
        "ar": "دبي ",
        "hi": "दिल्ली ",
    }

    templates = {
        "navigational": [
            "{brand} login",
            "{brand} official site",
            "{brand} app download",
            "open {brand}",
        ],
        "transactional": [
            "buy {product}",
            "download {product}",
            "{product} price",
            "cheap {product} deal",
        ],
        "local": [
            "restaurants near me",
            "coffee shops nearby",
            "pharmacy near me",
            "directions to airport",
        ],
        "realtime": [
            "weather today",
            "stock price AAPL",
            "live score football",
            "flight status AA100",
        ],
        "factoid": [
            "capital of france",
            "population of tokyo",
            "who invented telephone",
            "height of mount everest",
        ],
        "comparison": [
            "iphone vs android",
            "python vs javascript",
            "compare tesla vs bmw",
            "macbook vs thinkpad",
        ],
        "explanation": [
            "how does wifi work",
            "why is the sky blue",
            "what is blockchain",
            "explain quantum computing",
        ],
        "research": [
            "impact of climate change on agriculture",
            "history of artificial intelligence",
            "future of renewable energy",
            "trends in remote work",
        ],
        "code": [
            "python sort list",
            "javascript async await example",
            "sql join example",
            "rust ownership explained briefly",
        ],
        "news": [
            "latest tech news",
            "current events summary",
            "recent developments in AI",
            "today's world news overview",
        ],
    }

    localized = {
        "zh-CN": {
            "local": ["附近餐厅", "周边咖啡店", "离我最近的药店", "怎么去机场"],
            "realtime": ["北京今天天气", "苹果股价", "足球比分", "航班状态"],
            "factoid": ["法国首都", "东京人口", "谁发明了电话", "珠穆朗玛峰高度"],
            "comparison": ["iphone和安卓对比", "python和javascript比较", "特斯拉和宝马哪个好"],
            "explanation": ["wifi是怎么工作的", "为什么天空是蓝的", "什么是区块链", "解释量子计算"],
            "research": ["气候变化对农业的影响", "人工智能的历史", "可再生能源的未来"],
            "transactional": ["购买iphone", "下载spotify", "笔记本电脑价格", "耳机优惠"],
            "navigational": ["微信登录", "淘宝官网", "下载支付宝", "打开微博"],
        },
        "ja": {
            "local": ["近くのレストラン", "近所のカフェ", "最寄りの薬局", "空港への行き方"],
            "realtime": ["今日の天気", "アップル株価", "サッカーのスコア", "フライト状況"],
            "explanation": ["WiFiの仕組み", "空が青い理由", "ブロックチェーンとは", "量子コンピュータの説明"],
            "comparison": ["iphoneとandroid比較", "pythonとjavascriptの違い"],
        },
    }

    brands = ["google", "facebook", "amazon", "netflix", "spotify", "github", "twitter"]
    products = ["iphone", "laptop", "headphones", "book", "camera", "tablet"]

    prefix = lang_prefix.get(language, "")
    base = localized.get(language, {}).get(intent) or templates.get(intent, ["search query {n}"])
    queries = []
    for i in range(count):
        t = base[i % len(base)]
        q = t.format(
            brand=brands[i % len(brands)],
            product=products[i % len(products)],
            n=i + 1,
        )
        if prefix and language not in localized:
            q = prefix + q
        queries.append(q)
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic queries via LLM")
    parser.add_argument("--per-bucket", type=int, default=20, help="Queries per language×intent")
    parser.add_argument("--provider", choices=["openai", "anthropic", "ollama", "fallback"], default="ollama")
    parser.add_argument("--model", type=str, default=None, help="Ollama model name (default: qwen2.5:1.5b)")
    parser.add_argument("--sleep", type=float, default=0.1, help="Pause between Ollama calls")
    args = parser.parse_args()

    load_dotenv()
    ensure_dirs()

    ollama_model = args.model or DEFAULT_OLLAMA_MODEL
    if args.provider == "ollama":
        if not ollama_available():
            print("Error: Ollama not running. Start with: ollama serve")
            print(f"Then pull model: ollama pull {ollama_model}")
            return
        print(f"Using Ollama model: {ollama_model}")

    records: list[dict] = []
    out_path = SYNTH_DIR / "synthetic_queries.jsonl"
    # Resume/top-up: load existing if present, then fill each language×intent bucket to target.
    if out_path.exists():
        records = list(read_jsonl(out_path))
        print(f"Resuming with {len(records)} existing synthetic queries")

    existing_queries = {
        (r.get("language", ""), normalize_query(r.get("query", "")))
        for r in records
        if r.get("query")
    }

    for lang in tqdm(LANGUAGES, desc="Languages"):
        for intent in INTENT_BUCKETS:
            bucket_counts = Counter(
                (r.get("language"), r.get("intent")) for r in records if r.get("source") == "synthetic"
            )
            existing_count = bucket_counts[(lang, intent)]
            missing = max(0, args.per_bucket - existing_count)
            if missing == 0:
                continue
            prompt = USER_PROMPT_TEMPLATE.format(
                count=missing,
                language_name=LANGUAGE_NAMES[lang],
                intent=intent,
                label_hint=INTENT_LABEL_HINT[intent],
            )

            try:
                if args.provider == "openai" and os.environ.get("OPENAI_API_KEY"):
                    queries = call_openai(prompt, model=args.model or "gpt-4o-mini")
                elif args.provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
                    queries = call_anthropic(prompt, model=args.model or "claude-3-5-haiku-20241022")
                elif args.provider == "ollama":
                    queries = call_ollama_queries(prompt, model=ollama_model)
                else:
                    queries = generate_fallback_queries(lang, intent, missing)

                time.sleep(args.sleep)
            except Exception as e:
                print(f"Warning: generation failed for {lang}/{intent}: {e}")
                queries = generate_fallback_queries(lang, intent, missing)

            added = 0
            for q in queries:
                if isinstance(q, str) and q.strip():
                    key = (lang, normalize_query(q))
                    if key in existing_queries:
                        continue
                    records.append(
                        {
                            "query": q.strip(),
                            "language": lang,
                            "source": "synthetic",
                            "intent": intent,
                            "expected_label": INTENT_LABEL_HINT[intent],
                        }
                    )
                    existing_queries.add(key)
                    added += 1
                    if added >= missing:
                        break

            write_jsonl(out_path, records)

    print(f"\nGenerated {len(records)} synthetic queries -> {out_path}")


if __name__ == "__main__":
    main()
