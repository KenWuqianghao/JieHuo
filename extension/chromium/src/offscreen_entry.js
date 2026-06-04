import { env, pipeline } from "@huggingface/transformers";

const E5_PREFIX = "query: ";
const LOCAL_MODEL = "multilingual-router";

let status = "idle";
let errorMessage = null;
let classifierPromise = null;
let routerConfigPromise = null;

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = chrome.runtime.getURL("models/");
env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL("wasm/");
env.backends.onnx.wasm.proxy = false;
env.backends.onnx.wasm.numThreads = 1;

function normalizeModelLabel(raw) {
  const label = String(raw).toLowerCase();
  if (label === "google" || label === "perplexity") return label;
  if (label === "label_0" || label === "0") return "google";
  if (label === "label_1" || label === "1") return "perplexity";
  return null;
}

function applyTemperature(scores, temperature) {
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

function parseModelOutput(raw, temperature) {
  const output = Array.isArray(raw?.[0]) ? raw[0] : Array.isArray(raw) ? raw : [raw];
  const rawScores = { google: 0, perplexity: 0 };
  const rawLabels = [];

  for (const item of output) {
    rawLabels.push(item.label);
    const label = normalizeModelLabel(item.label);
    if (label) rawScores[label] = item.score;
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
  const label = scores.perplexity >= scores.google ? "perplexity" : "google";

  return {
    label,
    confidence: Math.max(scores.google, scores.perplexity),
    scores,
    source: "extension-neural",
    rawLabels,
  };
}

async function loadRouterConfig() {
  if (!routerConfigPromise) {
    routerConfigPromise = fetch(chrome.runtime.getURL(`models/${LOCAL_MODEL}/router_config.json`))
      .then((response) => (response.ok ? response.json() : {}))
      .catch(() => ({}));
  }
  return routerConfigPromise;
}

async function loadClassifier() {
  if (!classifierPromise) {
    status = "loading";
    classifierPromise = pipeline("text-classification", LOCAL_MODEL, {
      dtype: "q8",
    })
      .then((classifier) => {
        status = "ready";
        errorMessage = null;
        return classifier;
      })
      .catch((err) => {
        status = "error";
        errorMessage = err instanceof Error ? err.message : String(err);
        classifierPromise = null;
        throw err;
      });
  }
  return classifierPromise;
}

async function classify(query) {
  const [classifier, config] = await Promise.all([loadClassifier(), loadRouterConfig()]);
  const raw = await classifier(`${E5_PREFIX}${query}`, { topk: 2 });
  return parseModelOutput(raw, config.temperature ?? 1);
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.target !== "offscreen") return false;

  if (message.type === "status") {
    loadClassifier()
      .then(() => sendResponse({ ok: true, status }))
      .catch((err) =>
        sendResponse({
          ok: false,
          status,
          error: err instanceof Error ? err.message : String(err),
        })
      );
    return true;
  }

  if (message.type === "classify") {
    classify(String(message.query || "").trim().slice(0, 500))
      .then((result) => sendResponse({ ok: true, status, result }))
      .catch((err) =>
        sendResponse({
          ok: false,
          status,
          error: err instanceof Error ? err.message : String(err),
        })
      );
    return true;
  }

  return false;
});
