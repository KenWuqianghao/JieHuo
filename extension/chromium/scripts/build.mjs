import { build } from "esbuild";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = path.resolve(extensionRoot, "../..");
const webRoot = path.join(repoRoot, "web");
const dist = path.join(extensionRoot, "dist");

const staticFiles = [
  "manifest.json",
  "service_worker.js",
  "popup.html",
  "popup.js",
  "options.html",
  "options.js",
  "offscreen.html",
  "route.html",
  "route.js",
  "status.html",
  "status.js",
  "styles.css",
];

async function copyStaticFiles() {
  await fs.rm(dist, { recursive: true, force: true });
  await fs.mkdir(dist, { recursive: true });

  for (const file of staticFiles) {
    await fs.copyFile(path.join(extensionRoot, file), path.join(dist, file));
  }
}

async function copyModel() {
  await fs.cp(
    path.join(webRoot, "public/models/multilingual-router"),
    path.join(dist, "models/multilingual-router"),
    { recursive: true }
  );
}

async function copyWasmRuntime() {
  const wasmDist = path.join(dist, "wasm");
  await fs.mkdir(wasmDist, { recursive: true });

  const candidates = [
    path.join(extensionRoot, "node_modules/onnxruntime-web/dist"),
    path.join(extensionRoot, "node_modules/@huggingface/transformers/dist"),
    path.join(webRoot, "node_modules/onnxruntime-web/dist"),
    path.join(webRoot, "node_modules/@huggingface/transformers/dist"),
  ];

  const copied = new Set();
  for (const dir of candidates) {
    const files = await fs.readdir(dir).catch(() => []);
    for (const file of files) {
      if (!/^ort-.*\.(wasm|mjs)$/.test(file) || copied.has(file)) continue;
      await fs.copyFile(path.join(dir, file), path.join(wasmDist, file));
      copied.add(file);
    }
  }
}

await copyStaticFiles();
await build({
  entryPoints: [path.join(extensionRoot, "src/offscreen_entry.js")],
  outfile: path.join(dist, "offscreen.js"),
  bundle: true,
  format: "esm",
  platform: "browser",
  target: ["chrome116"],
  minify: true,
  sourcemap: false,
  absWorkingDir: webRoot,
});
await copyModel();
await copyWasmRuntime();

console.log(`Built ${dist}`);
