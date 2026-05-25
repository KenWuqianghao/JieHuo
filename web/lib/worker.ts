import { pipeline, env } from "@huggingface/transformers";
import type { ClassificationResult, RouteLabel } from "./heuristics";
import { normalizeModelLabel } from "./heuristics";

const E5_PREFIX = "query: ";
const LOCAL_MODEL = "multilingual-router";
const REMOTE_MODEL = process.env.NEXT_PUBLIC_MODEL_REPO;

// Prefer local model in dev; use HF Hub in production when NEXT_PUBLIC_MODEL_REPO is set
if (REMOTE_MODEL) {
  env.allowRemoteModels = true;
  env.allowLocalModels = true;
} else {
  env.allowRemoteModels = false;
  env.localModelPath = "/models/";
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let classifier: any = null;
const modelId = REMOTE_MODEL || LOCAL_MODEL;

type WorkerRequest =
  | { type: "init" }
  | { type: "classify"; id: number; query: string };

type WorkerResponse =
  | { type: "ready" }
  | { type: "error"; message: string }
  | { type: "result"; id: number; result: ClassificationResult };

async function initClassifier() {
  if (classifier) return classifier;

  classifier = await pipeline("text-classification", modelId, {
    dtype: "q8",
  });

  return classifier;
}

function parseResult(
  output: Array<{ label: string; score: number }>
): ClassificationResult {
  const scores: Record<RouteLabel, number> = { google: 0, perplexity: 0 };
  const rawLabels: string[] = [];

  for (const item of output) {
    rawLabels.push(item.label);
    const mapped = normalizeModelLabel(item.label);
    if (mapped) {
      scores[mapped] = item.score;
    }
  }

  // If only one class scored (topk quirk), infer the other from softmax sum ≈ 1
  const total = scores.google + scores.perplexity;
  if (total > 0 && total < 0.999) {
    if (scores.google > 0 && scores.perplexity === 0) {
      scores.perplexity = 1 - scores.google;
    } else if (scores.perplexity > 0 && scores.google === 0) {
      scores.google = 1 - scores.perplexity;
    }
  }

  const label: RouteLabel = scores.perplexity >= scores.google ? "perplexity" : "google";
  const confidence = Math.max(scores.google, scores.perplexity);

  return { label, confidence, scores, source: "neural", rawLabels };
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;

  try {
    if (msg.type === "init") {
      await initClassifier();
      self.postMessage({ type: "ready" } satisfies WorkerResponse);
      return;
    }

    if (msg.type === "classify") {
      const pipe = await initClassifier();
      const text = E5_PREFIX + msg.query;
      const raw = await pipe(text, { topk: 2 });
      const output = Array.isArray(raw) ? (Array.isArray(raw[0]) ? raw[0] : raw) : [raw];
      const result = parseResult(output as Array<{ label: string; score: number }>);
      self.postMessage({ type: "result", id: msg.id, result } satisfies WorkerResponse);
    }
  } catch (err) {
    self.postMessage({
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    } satisfies WorkerResponse);
  }
};

export {};
