import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

const ORT_SYMBOL = Symbol.for("onnxruntime");
if (!(ORT_SYMBOL in globalThis)) {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const ort = require("onnxruntime-web");
  const onnx = ort as { default?: unknown };
  (globalThis as unknown as Record<symbol, unknown>)[ORT_SYMBOL] = onnx.default ?? ort;
}

export { env, pipeline } from "@huggingface/transformers";
