export type RouteLabel = "google" | "perplexity";
export type InferenceSource = "neural" | "heuristic";

export interface ClassificationResult {
  label: RouteLabel;
  confidence: number;
  scores: Record<RouteLabel, number>;
  source: InferenceSource;
  rawLabels?: string[];
}

export interface HeuristicExplanation {
  label: RouteLabel;
  confidence: number;
  reasons: string[];
}

const GOOGLE_PATTERNS: Array<[RegExp, number, string]> = [
  [/https?:\/\/|www\.|\.com\b|\.org\b|site:/i, 0.95, "url_or_domain"],
  [/\b(login|sign in|sign up|download|app store)\b/i, 0.9, "navigational"],
  [/\b(near me|nearby|directions to|open now)\b/i, 0.92, "local_en"],
  [/(附近|周边|怎么走|地图|导航)/, 0.92, "local_zh"],
  [/(近く|近所|地図|道順)/, 0.92, "local_ja"],
  [/\b(weather|forecast|stock price|live score|flight status)\b/i, 0.93, "realtime_en"],
  [/(天气|气温|预报|股价|比分|航班)/, 0.93, "realtime_zh"],
  [/\b(buy|purchase|order|price of|coupon|discount)\b/i, 0.88, "transactional"],
  [/(购买|买|价格|多少钱|优惠)/, 0.88, "transactional_zh"],
  [/^(who is|what is the capital|population of|height of)\b/i, 0.85, "factoid"],
  [/\b(syntax for|code for|example of)\b/i, 0.82, "code"],
  [/\b(image of|picture of|photo of|video of)\b/i, 0.9, "media"],
];

const PERPLEXITY_PATTERNS: Array<[RegExp, number, string]> = [
  [/^(how does|how do|why does|why is|what are the)\b/i, 0.88, "question_en"],
  [/^(explain|describe|summarize|compare|contrast|analyze|evaluate|discuss)\b/i, 0.9, "research_en"],
  [/( vs | versus | compared to |better than)/i, 0.9, "comparison"],
  [/(为什么|怎么|如何|对比|比较|解释|分析|总结|优缺点|区别|影响)/, 0.88, "question_zh"],
  [/(なぜ|どうやって|比較|違い|説明|分析)/, 0.88, "question_ja"],
  [/(왜|어떻게|비교|차이|설명|분석)/, 0.88, "question_ko"],
  [/(por qué|cómo|comparar|explicar|analizar)/i, 0.88, "question_es"],
  [/(pourquoi|comment|comparer|expliquer)/i, 0.88, "question_fr"],
  [/(warum|wie|vergleich|erklären)/i, 0.88, "question_de"],
  [/(impact of|history of|future of|overview of|implications of)/i, 0.87, "research_en"],
  [/(latest developments|current situation|recent changes)/i, 0.85, "news_en"],
];

export function explainHeuristics(query: string): HeuristicExplanation {
  const q = query.toLowerCase().trim();
  const tokens = q.split(/\s+/).filter(Boolean);

  const googleHits: Array<[number, string]> = [];
  const perplexityHits: Array<[number, string]> = [];

  for (const [re, conf, reason] of GOOGLE_PATTERNS) {
    if (re.test(q)) googleHits.push([conf, reason]);
  }
  for (const [re, conf, reason] of PERPLEXITY_PATTERNS) {
    if (re.test(q)) perplexityHits.push([conf, reason]);
  }

  if (tokens.length >= 8 && query.includes("?") && googleHits.length === 0) {
    perplexityHits.push([0.75, "long_question"]);
  }

  if (googleHits.length && !perplexityHits.length) {
    const best = googleHits.sort((a, b) => b[0] - a[0])[0];
    return { label: "google", confidence: best[0], reasons: [best[1]] };
  }
  if (perplexityHits.length && !googleHits.length) {
    const best = perplexityHits.sort((a, b) => b[0] - a[0])[0];
    return { label: "perplexity", confidence: best[0], reasons: [best[1]] };
  }
  if (googleHits.length && perplexityHits.length) {
    const g = googleHits.sort((a, b) => b[0] - a[0])[0];
    const p = perplexityHits.sort((a, b) => b[0] - a[0])[0];
    if (g[0] >= p[0]) {
      return { label: "google", confidence: g[0] * 0.9, reasons: [g[1], `conflict:${p[1]}`] };
    }
    return { label: "perplexity", confidence: p[0] * 0.9, reasons: [p[1], `conflict:${g[1]}`] };
  }

  if (tokens.length <= 4) {
    return { label: "google", confidence: 0.55, reasons: ["default_short"] };
  }
  return { label: "perplexity", confidence: 0.55, reasons: ["default_long"] };
}

export function normalizeModelLabel(raw: string): RouteLabel | null {
  const s = raw.toLowerCase();
  if (s === "google" || s === "perplexity") return s;
  // transformers.js often returns LABEL_0 / LABEL_1 even when id2label is set
  if (s === "label_0" || s === "0") return "google";
  if (s === "label_1" || s === "1") return "perplexity";
  return null;
}

export function resultFromHeuristics(query: string): ClassificationResult {
  const h = explainHeuristics(query);
  return {
    label: h.label,
    confidence: h.confidence,
    scores: {
      google: h.label === "google" ? h.confidence : 1 - h.confidence,
      perplexity: h.label === "perplexity" ? h.confidence : 1 - h.confidence,
    },
    source: "heuristic",
  };
}

export function buildSearchUrls(query: string) {
  const encoded = encodeURIComponent(query);
  return {
    google: `https://www.google.com/search?q=${encoded}`,
    perplexity: `https://www.perplexity.ai/search?q=${encoded}`,
  };
}
