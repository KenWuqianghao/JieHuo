import { buildSearchUrls, normalizeModelLabel, resultFromHeuristics } from "./heuristics";
import type { ClassificationResult, RouteLabel } from "./heuristics";

const E5_PREFIX = "query: ";
const DEFAULT_MODEL_REPO = "KenWu/multilingual-query-router";
const DEFAULT_TIMEOUT_MS = 2500;
const MAX_QUERY_LENGTH = 500;

type PipelineFn = (
  text: string,
  options: { topk: number }
) => Promise<Array<{ label: string; score: number }> | Array<Array<{ label: string; score: number }>>>;

type RouterConfig = {
  temperature?: number;
};

export type RoutedSearch = ClassificationResult & {
  query: string;
  targetUrl: string;
};

let classifierPromise: Promise<PipelineFn> | null = null;
let routerConfigPromise: Promise<RouterConfig> | null = null;

export function sanitizeSearchQuery(value: string | null): string {
  return String(value ?? "").trim().slice(0, MAX_QUERY_LENGTH);
}

export function targetUrlForLabel(label: RouteLabel, query: string): string {
  return buildSearchUrls(query)[label];
}

function routerMode(): "model" | "heuristic" {
  return process.env.JIEHUO_ROUTER_MODE === "heuristic" ? "heuristic" : "model";
}

function modelRepo(): string {
  return process.env.JIEHUO_MODEL_REPO || DEFAULT_MODEL_REPO;
}

function modelTimeoutMs(): number {
  const parsed = Number(process.env.JIEHUO_MODEL_TIMEOUT_MS);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_TIMEOUT_MS;
}

async function getClassifier(): Promise<PipelineFn> {
  if (!classifierPromise) {
    classifierPromise = (async () => {
      const { pipeline, env } = await import("@huggingface/transformers");
      env.allowRemoteModels = true;
      env.allowLocalModels = false;
      env.cacheDir = "/tmp/jiehuo-transformers";

      const createTextClassifier = pipeline as unknown as (
        task: "text-classification",
        model: string,
        options: { dtype: "q8" }
      ) => Promise<PipelineFn>;

      return createTextClassifier("text-classification", modelRepo(), {
        dtype: "q8",
      });
    })();
  }

  return classifierPromise;
}

async function getRouterConfig(): Promise<RouterConfig> {
  if (!routerConfigPromise) {
    routerConfigPromise = (async () => {
      const res = await fetch(
        `https://huggingface.co/${modelRepo()}/resolve/main/router_config.json`,
        { cache: "force-cache" }
      );
      if (!res.ok) return {};
      return (await res.json()) as RouterConfig;
    })().catch(() => ({}));
  }

  return routerConfigPromise;
}

function applyTemperature(
  scores: Record<RouteLabel, number>,
  temperature: number
): Record<RouteLabel, number> {
  if (!Number.isFinite(temperature) || Math.abs(temperature - 1) < 0.001) {
    return scores;
  }

  const googleLogit = Math.log(Math.max(scores.google, 1e-8)) / temperature;
  const perplexityLogit = Math.log(Math.max(scores.perplexity, 1e-8)) / temperature;
  const maxLogit = Math.max(googleLogit, perplexityLogit);
  const googleExp = Math.exp(googleLogit - maxLogit);
  const perplexityExp = Math.exp(perplexityLogit - maxLogit);
  const total = googleExp + perplexityExp;

  return {
    google: googleExp / total,
    perplexity: perplexityExp / total,
  };
}

function parseModelOutput(
  output: Array<{ label: string; score: number }>,
  temperature: number
): ClassificationResult {
  const rawScores: Record<RouteLabel, number> = { google: 0, perplexity: 0 };
  const rawLabels: string[] = [];

  for (const item of output) {
    rawLabels.push(item.label);
    const mapped = normalizeModelLabel(item.label);
    if (mapped) {
      rawScores[mapped] = item.score;
    }
  }

  const total = rawScores.google + rawScores.perplexity;
  if (total > 0 && total < 0.999) {
    if (rawScores.google > 0 && rawScores.perplexity === 0) {
      rawScores.perplexity = 1 - rawScores.google;
    } else if (rawScores.perplexity > 0 && rawScores.google === 0) {
      rawScores.google = 1 - rawScores.perplexity;
    }
  }

  const scores = applyTemperature(rawScores, temperature);
  const label: RouteLabel = scores.perplexity >= scores.google ? "perplexity" : "google";

  return {
    label,
    confidence: Math.max(scores.google, scores.perplexity),
    scores,
    source: "neural",
    rawLabels,
  };
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) => {
      setTimeout(() => reject(new Error("JieHuo model route timed out")), timeoutMs);
    }),
  ]);
}

async function routeWithModel(query: string): Promise<ClassificationResult> {
  const [classifier, config] = await Promise.all([getClassifier(), getRouterConfig()]);
  const raw = await classifier(E5_PREFIX + query, { topk: 2 });
  const output = Array.isArray(raw[0])
    ? (raw[0] as Array<{ label: string; score: number }>)
    : (raw as Array<{ label: string; score: number }>);

  return parseModelOutput(output, config.temperature ?? 1);
}

export async function routeSearchQuery(query: string): Promise<RoutedSearch> {
  let result: ClassificationResult;

  if (routerMode() === "heuristic") {
    result = resultFromHeuristics(query);
  } else {
    try {
      result = await withTimeout(routeWithModel(query), modelTimeoutMs());
    } catch {
      result = resultFromHeuristics(query);
    }
  }

  return {
    ...result,
    query,
    targetUrl: targetUrlForLabel(result.label, query),
  };
}
