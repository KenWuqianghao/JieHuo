import { buildSearchUrls, resultFromHeuristics } from "./heuristics";
import type { ClassificationResult, RouteLabel } from "./heuristics";

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

/**
 * Server-side routing for /search and /api/route.
 * Uses heuristics only so Vercel serverless functions stay under the 250 MB limit.
 * The browser UI still runs the INT8 neural model in a Web Worker.
 */
export async function routeSearchQuery(query: string): Promise<RoutedSearch> {
  const result = resultFromHeuristics(query);

  return {
    ...result,
    query,
    targetUrl: targetUrlForLabel(result.label, query),
  };
}
