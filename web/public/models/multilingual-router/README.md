---
language:
  - en
  - zh
  - es
  - fr
  - de
  - ja
  - ko
  - pt
  - ru
  - ar
  - hi
license: apache-2.0
library_name: transformers.js
tags:
  - query-routing
  - search
  - perplexity
  - google
  - multilingual
  - onnx
  - transformers.js
---

# JieHuo Multilingual Query Router

JieHuo is a browser-runnable query router that predicts whether a search query belongs on **Google** or **Perplexity**.

It is a fine-tuned `intfloat/multilingual-e5-small` sequence classifier distilled from GPT-4.1-mini labels, exported to INT8 ONNX for transformers.js, and calibrated with a scalar temperature for selective auto-routing.

## Metrics

Evaluation uses a corrected 12-language split and a language+label balanced 240-query gold set.

| Split | n | Accuracy | Balanced Acc | Macro F1 | ECE |
|---|---:|---:|---:|---:|---:|
| Held-out test | 3,820 | 0.8275 | 0.8283 | 0.8229 | 0.0638 |
| Balanced gold | 240 | 0.8833 | 0.8833 | 0.8828 | 0.0680 |

Selective auto-routing:

| Split | Threshold | Coverage | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Held-out test | conf >= 0.70 | 89.2% | 0.8589 | 0.8547 |
| Held-out test | conf >= 0.85 | 70.3% | 0.9103 | 0.9064 |
| Balanced gold | conf >= 0.80 | 88.8% | 0.9108 | 0.9106 |
| Balanced gold | conf >= 0.90 | 56.7% | 0.9632 | 0.9623 |

## Browser Use

```ts
import { pipeline, env } from "@huggingface/transformers";

env.allowRemoteModels = true;
const classifier = await pipeline(
  "text-classification",
  "KenWu/multilingual-query-router",
  { dtype: "q8" }
);

const result = await classifier("query: compare perplexity and google for research", {
  topk: 2,
});
```

The companion `router_config.json` contains the calibration temperature used by the JieHuo web app.
