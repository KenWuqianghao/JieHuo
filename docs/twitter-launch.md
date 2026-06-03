# JieHuo Twitter Launch Copy

## Primary post (BaseLayer-style)

Most search UIs still send every query to one engine. I built a multilingual routing layer that picks the right one first.

Introducing JieHuo (解惑): a browser-runnable Google vs Perplexity query router across 12 languages.

0.883 macro F1 on a balanced multilingual gold set (240 queries, equal labels per language). 0.962 macro F1 when auto-routing only high-confidence queries — 57% coverage, ambiguous queries stay with the user.

GPT-4.1-mini teacher labels → fine-tuned multilingual-e5-small → temperature scaling → INT8 ONNX → transformers.js Web Worker. No server inference on the classification hot path.

try it: https://jiehuo.vercel.app

## Alternate hooks

**Eval-forward**

Fixed an English-heavy eval bug, rebuilt a 12-language gold set, and retrained. Result: 0.883 macro F1 full coverage, 0.911 at 89% auto-route coverage on held-out test.

**Product-forward**

The routing decision is becoming a product primitive. JieHuo temperature-calibrates confidence and only auto-routes when the model is sure — otherwise you choose.

## Thread

1. I built JieHuo (解惑): a multilingual query router that predicts whether a search belongs on Google or Perplexity.

It runs in the browser, not behind an API: multilingual-e5-small → INT8 ONNX → transformers.js Web Worker.

2. Training stack: GPT-4.1-mini as a routing teacher, language-balanced sampling, confidence-aware weighting, and temperature scaling fit on validation data.

3. Eval fix worth calling out: the original gold set was English-heavy because label balancing compared string labels to numeric dict keys. The new gold set is 20 queries × 12 languages, split evenly between Google and Perplexity.

4. Metrics on balanced multilingual gold:

- 0.883 macro F1 at full coverage
- 0.911 macro F1 at 88.8% auto-route coverage (conf ≥ 0.80)
- 0.962 macro F1 at 56.7% high-confidence coverage (conf ≥ 0.90)

5. Product metric > classifier metric.

The app temperature-calibrates model confidence and only auto-routes high-confidence queries. Ambiguous queries stay user-controlled.

6. Links

Live demo: https://jiehuo.vercel.app  
GitHub: https://github.com/KenWuqianghao/JieHuo  
Model: https://huggingface.co/KenWu/multilingual-query-router

## Character counts

Primary post: ~580 characters (fits one tweet with link; trim the third paragraph if you need ≤280 for a single tweet).

**Single-tweet version (≤280)**

Built JieHuo (解惑): 12-language browser router for Google vs Perplexity. 0.883 macro F1 on balanced gold; 0.962 when auto-routing high-confidence only. INT8 ONNX in-browser, no server on the hot path.

https://jiehuo.vercel.app
