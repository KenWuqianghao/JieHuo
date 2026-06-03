import { buildSearchUrls, resultFromHeuristics } from "./heuristics";
import type { ClassificationResult, RouteLabel } from "./heuristics";
import { classifyWithHfInference, hfInferenceToken } from "./hf-inference-router";
import { classifyWithNeuralModelTimed } from "./neural-router";

const MAX_QUERY_LENGTH = 500;

export type RoutedSearch = ClassificationResult & {
  query: string;
  targetUrl: string;
};

export type RoutingMode = "heuristic" | "model";

export function sanitizeSearchQuery(value: string | null): string {
  return String(value ?? "").trim().slice(0, MAX_QUERY_LENGTH);
}

export function targetUrlForLabel(label: RouteLabel, query: string): string {
  return buildSearchUrls(query)[label];
}

async function classifyOnServer(query: string) {
  if (process.env.VERCEL === "1" && hfInferenceToken()) {
    return classifyWithHfInference(query);
  }
  return classifyWithNeuralModelTimed(query);
}

export async function routeSearchQuery(
  query: string,
  mode: RoutingMode = process.env.JIEHUO_ROUTER_MODE === "model" ? "model" : "heuristic"
): Promise<RoutedSearch> {
  let result: ClassificationResult;

  if (mode === "heuristic") {
    result = resultFromHeuristics(query);
  } else {
    try {
      result = await classifyOnServer(query);
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
