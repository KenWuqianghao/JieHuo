# JieHuo Twitter Launch Copy

## Short post

Built JieHuo: a 12-language browser-runnable router that decides whether a query belongs on Google or Perplexity.

GPT-4.1-mini teacher-distilled, temperature-calibrated, exported to INT8 ONNX.

Balanced multilingual gold:
- 0.883 macro F1 at full coverage
- 0.911 macro F1 at 89% auto-route coverage
- 0.962 macro F1 at 57% high-confidence coverage

GitHub: https://github.com/KenWuqianghao/JieHuo  
Model: https://huggingface.co/KenWu/multilingual-query-router

## Thread

1. I built JieHuo: a multilingual query router that predicts whether a search belongs on Google or Perplexity.

It runs in the browser, not behind an API: multilingual-e5-small -> INT8 ONNX -> transformers.js Web Worker.

2. The interesting part is the training/eval process.

I used GPT-4.1-mini as a routing teacher, then trained a small multilingual classifier with language-balanced sampling and confidence-aware weighting.

3. I found and fixed a bad evaluation bug: the original gold set was English-heavy because label balancing was checking string labels against numeric dict keys.

The new gold set is balanced across 12 languages with equal Google/Perplexity labels.

4. Metrics on balanced multilingual gold:

- 0.883 macro F1 at full coverage
- 0.911 macro F1 at 88.8% auto-route coverage
- 0.962 macro F1 at 56.7% high-confidence coverage

5. Product metric > classifier metric.

The app temperature-calibrates model confidence and only auto-routes high-confidence queries. Ambiguous queries can stay user-controlled.

6. Repo and model:

GitHub: https://github.com/KenWuqianghao/JieHuo  
Model: https://huggingface.co/KenWu/multilingual-query-router

I built this because the routing decision itself is becoming a product primitive for search.
