import { buildSearchUrls } from "./heuristics";
import type { ClassificationResult, RouteLabel } from "./heuristics";
import { classifyWithNeuralModelTimed } from "./neural-router";

const MAX_QUERY_LENGTH = 500;

export type RoutedSearch = ClassificationResult & {
  query: string;
  targetUrl: string;
};

export function sanitizeSearchQuery(value: string | null): string {
  return String(value ?? "").trim().slice(0, MAX_QUERY_LENGTH);
}

export function targetUrlForLabel(label: RouteLabel, query: string): string {
  return buildSearchUrls(query)[label];
}

export async function routeSearchQuery(query: string): Promise<RoutedSearch> {
  const result = await classifyWithNeuralModelTimed(query);

  return {
    ...result,
    query,
    targetUrl: targetUrlForLabel(result.label, query),
  };
}
