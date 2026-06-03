import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

const ORT_SYMBOL = Symbol.for("onnxruntime");

function installOnnxWebGlobal(): void {
  if (ORT_SYMBOL in globalThis) return;
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const ort = require("onnxruntime-web");
  const onnx = ort as { default?: unknown };
  (globalThis as unknown as Record<symbol, unknown>)[ORT_SYMBOL] = onnx.default ?? ort;
}

type TransformersWeb = typeof import("@huggingface/transformers");

let runtimePromise: Promise<TransformersWeb> | null = null;

/** Dynamic import avoids hoisting transformers before ONNX is registered. */
export function loadNeuralRuntime(): Promise<TransformersWeb> {
  if (!runtimePromise) {
    runtimePromise = (async () => {
      installOnnxWebGlobal();
      return import("@huggingface/transformers");
    })();
  }
  return runtimePromise;
}
