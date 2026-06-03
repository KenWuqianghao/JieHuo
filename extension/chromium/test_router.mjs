import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(new URL("./router.js", import.meta.url), "utf8");
const context = vm.createContext({ globalThis: {} });
context.globalThis = context;
vm.runInContext(source, context);

const { JieHuoRouter } = context;

const weather = JieHuoRouter.route("weather in Tokyo today");
assert.equal(weather.label, "google");
assert.match(weather.targetUrl, /^https:\/\/www\.google\.com\/search\?q=/);

const compare = JieHuoRouter.route("compare perplexity and google for research");
assert.equal(compare.label, "perplexity");
assert.match(compare.targetUrl, /^https:\/\/www\.perplexity\.ai\/search\?q=/);

const zh = JieHuoRouter.route("比较 perplexity 和 google 做学术研究的优缺点");
assert.equal(zh.label, "perplexity");

const capped = JieHuoRouter.route("x".repeat(600));
assert.equal(capped.query.length, 500);

console.log("extension router checks passed");
