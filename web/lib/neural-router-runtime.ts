import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
// eslint-disable-next-line @typescript-eslint/no-require-imports
const ort = require("onnxruntime-web");

const ORT_SYMBOL = Symbol.for("onnxruntime");
if (!(ORT_SYMBOL in globalThis)) {
  const onnx = ort as typeof ort & { default?: typeof ort };
  (globalThis as unknown as Record<symbol, unknown>)[ORT_SYMBOL] = onnx.default ?? onnx;
}

export { env, pipeline } from "@huggingface/transformers";
