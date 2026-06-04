import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(extensionRoot, "dist");
const manifest = JSON.parse(fs.readFileSync(path.join(dist, "manifest.json"), "utf8"));

assert.equal(manifest.manifest_version, 3);
assert.ok(manifest.permissions.includes("offscreen"));
assert.ok(manifest.permissions.includes("declarativeNetRequestWithHostAccess"));
assert.ok(fs.existsSync(path.join(dist, "offscreen.js")));
assert.ok(fs.existsSync(path.join(dist, "route.html")));
assert.ok(fs.existsSync(path.join(dist, "models/multilingual-router/onnx/model_quantized.onnx")));
assert.ok(fs.existsSync(path.join(dist, "models/multilingual-router/tokenizer.json")));
assert.ok(fs.readdirSync(path.join(dist, "wasm")).some((file) => file.endsWith(".wasm")));

const serviceWorker = fs.readFileSync(path.join(dist, "service_worker.js"), "utf8");
assert.doesNotMatch(serviceWorker, /Heuristic|heuristic|GOOGLE_PATTERNS|PERPLEXITY_PATTERNS/);
assert.match(serviceWorker, /declarativeNetRequest/);

console.log("extension dist checks passed");
