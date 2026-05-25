#!/usr/bin/env python3
"""Apply heuristic rules to label queries with confidence scores."""

from __future__ import annotations

import ml._bootstrap  # noqa: F401

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from ml.common import (
    LABEL_GOOGLE,
    LABEL_NAMES,
    LABEL_PERPLEXITY,
    LABELED_DIR,
    RAW_DIR,
    SYNTH_DIR,
    ensure_dirs,
    read_jsonl,
    write_jsonl,
)


@dataclass
class HeuristicResult:
    label: int
    confidence: float
    reasons: list[str]


# --- Google patterns ---
GOOGLE_PATTERNS: list[tuple[str, float, str]] = [
    # URLs and domains
    (r"https?://|www\.|\.com\b|\.org\b|\.net\b|site:", 0.95, "url_or_domain"),
    # Navigational
    (r"\b(login|sign in|sign up|download|app store|play store)\b", 0.90, "navigational"),
    (r"\b(official site|official website|homepage)\b", 0.90, "navigational"),
    # Local
    (r"\b(near me|nearby|directions to|map to|open now)\b", 0.92, "local_en"),
    (r"(附近|周边|怎么走|地图|导航|离我最近)", 0.92, "local_zh"),
    (r"(近く|近所|地図|道順|周辺)", 0.92, "local_ja"),
    (r"(cerca de|près de|in der nähe|perto de|рядом|قريب)", 0.90, "local_multilingual"),
    # Real-time
    (r"\b(weather|forecast|temperature|stock price|stock quote|live score|flight status)\b", 0.93, "realtime_en"),
    (r"(天气|气温|预报|股价|股票|比分|航班)", 0.93, "realtime_zh"),
    (r"(天気|気温|予報|株価|スコア|フライト)", 0.93, "realtime_ja"),
    (r"(météo|temps|börse|aktie|wetter|tempo|clima)", 0.90, "realtime_multilingual"),
    # Transactional
    (r"\b(buy|purchase|order|price of|cost of|cheap|discount|coupon|deal)\b", 0.88, "transactional_en"),
    (r"(购买|买|价格|多少钱|优惠|折扣)", 0.88, "transactional_zh"),
    (r"( kaufen| acheter| comprar| купить)", 0.88, "transactional_multilingual"),
    # Simple factoids (short definitional)
    (r"^(who is|what is the capital|population of|height of|birthday of|age of)\b", 0.85, "factoid_en"),
    (r"^(谁是|什么是|人口|首都|身高)", 0.85, "factoid_zh"),
    # Code snippets
    (r"\b(syntax for|example of|code for|how to .{0,20} in python|how to .{0,20} in javascript)\b", 0.82, "code_en"),
    (r"(代码|语法|示例|python|javascript|sql)", 0.80, "code"),
    # Image/video intent
    (r"\b(image of|picture of|photo of|video of|gif of)\b", 0.90, "media_en"),
    (r"(图片|照片|视频)", 0.90, "media_zh"),
]

# --- Perplexity patterns ---
PERPLEXITY_PATTERNS: list[tuple[str, float, str]] = [
    # Open-ended questions
    (r"^(how does|how do|how can|how should|why does|why do|why is|why are|what are the)\b", 0.88, "question_en"),
    (r"^(explain|describe|summarize|summarise|compare|contrast|analyze|analyse|evaluate|discuss)\b", 0.90, "research_en"),
    (r"^(what are the (pros|cons|advantages|disadvantages|differences|benefits|risks))\b", 0.92, "comparison_en"),
    (r"( vs | versus | compared to |better than| worse than)", 0.90, "comparison"),
    # Chinese
    (r"(为什么|怎么|如何|对比|比较|解释|分析|总结|优缺点|区别|影响|趋势|综述)", 0.88, "question_zh"),
    (r"(是什么原理|有什么影响|有什么区别|哪个更好)", 0.90, "research_zh"),
    # Japanese
    (r"(なぜ|どうやって|比較|違い|説明|分析|まとめ|影響)", 0.88, "question_ja"),
    # Korean
    (r"(왜|어떻게|비교|차이|설명|분석|요약|영향)", 0.88, "question_ko"),
    # Spanish/French/German/Portuguese/Russian/Arabic/Hindi patterns
    (r"(por qué|cómo|comparar|explicar|analizar|resumir|ventajas|desventajas)", 0.88, "question_es"),
    (r"(pourquoi|comment|comparer|expliquer|analyser|résumer|avantages|inconvénients)", 0.88, "question_fr"),
    (r"(warum|wie|vergleich|erklären|analysieren|zusammenfassen|vor- und nachteile)", 0.88, "question_de"),
    (r"(por que|como|comparar|explicar|analisar|resumir|vantagens|desvantagens)", 0.88, "question_pt"),
    (r"(почему|как|сравн|объясн|анализ|итог|преимущ|недостат)", 0.88, "question_ru"),
    (r"(لماذا|كيف|مقارنة|شرح|تحليل|ملخص|مزايا|عيوب)", 0.88, "question_ar"),
    (r"(क्यों|कैसे|तुलना|व्याख्या|विश्लेषण|सारांश|फायदे|नुकसान)", 0.88, "question_hi"),
    # Long research queries
    (r"(impact of|history of|future of|trends in|state of|overview of|implications of)", 0.87, "research_en"),
    (r"(latest developments|current situation|recent changes|what happened)", 0.85, "news_en"),
]


def apply_heuristics(query: str) -> HeuristicResult:
    """Apply regex heuristics and return best label with confidence."""
    query_lower = query.lower().strip()
    token_count = len(query.split())

    google_scores: list[tuple[float, str]] = []
    perplexity_scores: list[tuple[float, str]] = []

    for pattern, conf, reason in GOOGLE_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            google_scores.append((conf, reason))

    for pattern, conf, reason in PERPLEXITY_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            perplexity_scores.append((conf, reason))

    # Long open-ended queries without strong Google signals -> Perplexity
    if token_count >= 8 and "?" in query and not google_scores:
        perplexity_scores.append((0.75, "long_question"))

    if google_scores and not perplexity_scores:
        best = max(google_scores, key=lambda x: x[0])
        return HeuristicResult(LABEL_GOOGLE, best[0], [best[1]])

    if perplexity_scores and not google_scores:
        best = max(perplexity_scores, key=lambda x: x[0])
        return HeuristicResult(LABEL_PERPLEXITY, best[0], [best[1]])

    if google_scores and perplexity_scores:
        g_best = max(google_scores, key=lambda x: x[0])
        p_best = max(perplexity_scores, key=lambda x: x[0])
        if g_best[0] > p_best[0]:
            return HeuristicResult(LABEL_GOOGLE, g_best[0] * 0.9, [g_best[1], f"conflict:{p_best[1]}"])
        return HeuristicResult(LABEL_PERPLEXITY, p_best[0] * 0.9, [p_best[1], f"conflict:{g_best[1]}"])

    # Default: short queries -> Google, longer -> Perplexity
    if token_count <= 4:
        return HeuristicResult(LABEL_GOOGLE, 0.55, ["default_short"])
    return HeuristicResult(LABEL_PERPLEXITY, 0.55, ["default_long"])


def label_records(records: list[dict]) -> list[dict]:
    labeled = []
    for r in records:
        result = apply_heuristics(r["query"])
        labeled.append(
            {
                **r,
                "heuristic_label": LABEL_NAMES[result.label],
                "heuristic_confidence": result.confidence,
                "heuristic_reasons": result.reasons,
            }
        )
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply heuristic labels to queries")
    parser.add_argument("--input", type=str, nargs="*", help="Input JSONL files (default: raw + synthetic)")
    args = parser.parse_args()

    ensure_dirs()

    input_files: list[Path] = []
    if args.input:
        input_files = [Path(p) for p in args.input]
    else:
        for p in [RAW_DIR / "public_queries.jsonl", SYNTH_DIR / "synthetic_queries.jsonl"]:
            if p.exists():
                input_files.append(p)

    if not input_files:
        print("No input files found. Run 01_collect_public.py and 02_synth_queries.py first.")
        return

    all_records: list[dict] = []
    for f in input_files:
        all_records.extend(list(read_jsonl(f)))

    labeled = label_records(all_records)
    out_path = LABELED_DIR / "heuristic_labeled.jsonl"
    write_jsonl(out_path, labeled)

    google_count = sum(1 for r in labeled if r["heuristic_label"] == "google")
    print(f"Labeled {len(labeled)} queries -> {out_path}")
    print(f"  Google: {google_count} ({100*google_count/len(labeled):.1f}%)")
    print(f"  Perplexity: {len(labeled) - google_count} ({100*(len(labeled)-google_count)/len(labeled):.1f}%)")


if __name__ == "__main__":
    main()
