#!/usr/bin/env node
/**
 * Point @huggingface/transformers Node exports at the WASM web bundle so Vercel
 * never loads transformers.node.mjs (which requires onnxruntime-node).
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const pkgPath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
  "node_modules",
  "@huggingface",
  "transformers",
  "package.json"
);

const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf8"));
const webEntry = "./dist/transformers.web.js";

if (pkg.exports?.node?.import) {
  pkg.exports.node.import.default = webEntry;
}
if (pkg.exports?.node?.require) {
  pkg.exports.node.require.default = webEntry;
}

fs.writeFileSync(pkgPath, `${JSON.stringify(pkg, null, 2)}\n`);
console.log(`Patched ${pkgPath} -> ${webEntry}`);
