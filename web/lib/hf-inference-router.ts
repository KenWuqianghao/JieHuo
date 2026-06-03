import { normalizeModelLabel } from "./heuristics";
import type { ClassificationResult } from "./heuristics";
import { modelRepo } from "./neural-router";

const E5_PREFIX = "query: ";

type HfLabelScore = { label: string; score: number };

function parseHfResponse(body: unknown): HfLabelScore[] {
  if (Array.isArray(body) && body.length > 0) {
    const first = body[0];
    if (Array.isArray(first)) {
      return first as HfLabelScore[];
    }
    if (typeof first === "object" && first !== null && "label" in first) {
      return body as HfLabelScore[];
    }
  }
  throw new Error("Unexpected Hugging Face inference response");
}

export function hfInferenceToken(): string | undefined {
  return process.env.HF_TOKEN || process.env.HUGGINGFACE_HUB_TOKEN;
}

export async function classifyWithHfInference(query: string): Promise<ClassificationResult> {
  const token = hfInferenceToken();
  if (!token) {
    throw new Error("HF_TOKEN is required for Hugging Face inference routing");
  }

  const res = await fetch(`https://api-inference.huggingface.co/models/${modelRepo()}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ inputs: E5_PREFIX + query }),
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Hugging Face inference failed (${res.status}): ${detail.slice(0, 200)}`);
  }

  const output = parseHfResponse(await res.json());
  const rawScores = { google: 0, perplexity: 0 };
  const rawLabels: string[] = [];

  for (const item of output) {
    rawLabels.push(item.label);
    const mapped = normalizeModelLabel(item.label);
    if (mapped) {
      rawScores[mapped] = item.score;
    }
  }

  const label = rawScores.perplexity >= rawScores.google ? "perplexity" : "google";
  return {
    label,
    confidence: Math.max(rawScores.google, rawScores.perplexity),
    scores: rawScores,
    source: "neural",
    rawLabels,
  };
}
