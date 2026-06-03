# JieHuo (解惑)

JieHuo is a browser-runnable neural router that decides whether a query should go to **Google** or **Perplexity**. It is built for multilingual search behavior: navigational, transactional, local, and realtime lookups usually belong on Google; open-ended synthesis, comparison, and reasoning-heavy questions usually belong on Perplexity.

The current model is a fine-tuned `intfloat/multilingual-e5-small` classifier distilled from GPT-4.1-mini labels, calibrated on a validation split, and exported as an INT8 ONNX model for transformers.js.

## Current result

Latest checkpoint: `checkpoints/best_multilingual`

Live model: [KenWu/multilingual-query-router](https://huggingface.co/KenWu/multilingual-query-router)

Evaluation uses a corrected 12-language split and a language+label balanced 240-query gold set. Gold contains 20 examples per language, split evenly between Google and Perplexity labels.

| Split | n | Accuracy | Balanced Acc | Macro F1 | ECE |
|---|---:|---:|---:|---:|---:|
| Held-out test | 3,820 | 0.8275 | 0.8283 | 0.8229 | 0.0638 |
| Balanced gold | 240 | 0.8833 | 0.8833 | 0.8828 | 0.0680 |

Selective auto-routing is the more useful product metric: route high-confidence queries automatically and leave ambiguous ones to the user.

| Split | Threshold | Coverage | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Held-out test | conf >= 0.70 | 89.2% | 0.8589 | 0.8547 |
| Held-out test | conf >= 0.85 | 70.3% | 0.9103 | 0.9064 |
| Held-out test | conf >= 0.90 | 39.4% | 0.9641 | 0.9588 |
| Balanced gold | conf >= 0.80 | 88.8% | 0.9108 | 0.9106 |
| Balanced gold | conf >= 0.90 | 56.7% | 0.9632 | 0.9623 |

## Why this is interesting

- **Teacher-distilled routing rubric:** GPT-4.1-mini labels queries against a concrete Google-vs-Perplexity rubric instead of relying only on brittle keyword rules.
- **Language-balanced evaluation:** the showcase metric is not an English-heavy random holdout; the gold set is balanced across `en`, `zh-CN`, `zh-TW`, `es`, `fr`, `de`, `ja`, `ko`, `pt`, `ru`, `ar`, and `hi`.
- **Calibrated selective routing:** a temperature-scaled confidence threshold converts the model into a product decision system, not just a classifier score.
- **Runs in the browser:** the PyTorch model is exported and dynamically quantized to INT8 ONNX for local transformers.js inference in a Web Worker.

## Architecture

```text
Query
  -> multilingual-e5-small sequence classifier
  -> temperature-calibrated P(google) / P(perplexity)
  -> browser Web Worker
  -> route button + feedback loop
```

## Quick start

### 1. Python setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
```

### 2. Reproduce the current training path

The strongest checkpoint used OpenAI labeling. Put `OPENAI_API_KEY` in `.env`, then run:

```bash
.venv/bin/python ml/02_synth_queries.py --provider openai --model gpt-4.1-mini --per-bucket 50
.venv/bin/python ml/03_heuristic_labels.py
.venv/bin/python ml/04_llm_label.py --provider openai --model gpt-4.1-mini --batch-size 60
.venv/bin/python ml/05_merge_split.py --input data/labeled/llm_labeled_api.jsonl --min-confidence 0.70 --gold-size 240
.venv/bin/python ml/06_train.py \
  --output checkpoints/best_multilingual \
  --epochs 8 \
  --batch-size 32 \
  --lr 1e-5 \
  --min-train-confidence 0.70 \
  --max-train-per-language-label 1500 \
  --label-smoothing 0.05 \
  --language-weight-power 0.50 \
  --disagreement-weight 0.75 \
  --seed 42
.venv/bin/python ml/07_eval.py \
  --model checkpoints/best_multilingual \
  --output checkpoints/eval_multilingual \
  --calibration data/labeled/val.jsonl \
  --confidence-thresholds 0.50,0.60,0.70,0.80,0.85,0.90,0.95 \
  --bootstrap-samples 500
.venv/bin/python ml/08_export_onnx.py --model checkpoints/best_multilingual --temperature 1.25
```

The exported browser model is written to:

```text
web/public/models/multilingual-router/
```

### 3. Run the web app

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:3000`.

**Live demo:** [jiehuo.vercel.app](https://jiehuo.vercel.app)

## Use as a Comet/Chromium search engine

JieHuo can take over the browser address bar through a custom search engine URL. It classifies the query server-side, then immediately redirects to Google or Perplexity.

In `comet://settings/searchEngines` or Chromium search engine settings, add:

```text
Name: JieHuo
Shortcut: jh
URL: https://jiehuo.vercel.app/search?q=%s
```

Expected behavior:

- Navigational, local, realtime, and transactional queries go to Google.
- Research, comparison, explanation, and synthesis queries go to Perplexity.
- If the neural router is cold or unavailable, JieHuo falls back to deterministic heuristics instead of failing the search.

Or run the full training pipeline (requires `OPENAI_API_KEY`):

```bash
bash scripts/run_pipeline.sh
```

## Pipeline

```text
ml/01_collect_public.py      -> data/raw/
ml/02_synth_queries.py       -> data/synthetic/
ml/03_heuristic_labels.py    -> data/labeled/heuristic_labeled.jsonl
ml/04_llm_label.py           -> data/labeled/llm_labeled*.jsonl
ml/05_merge_split.py         -> train/val/test + balanced gold
ml/06_train.py               -> checkpoints/
ml/07_eval.py                -> calibrated metrics + showcase summary
ml/08_export_onnx.py         -> web/public/models/multilingual-router/
ml/09_upload_hf.py           -> optional Hugging Face model repo
```

## Label rubric

| Route to Google | Route to Perplexity |
|---|---|
| Navigational site, app, or brand queries | Open-ended how/why questions |
| Transactional buy, download, or booking intent | Comparison and evaluation |
| Local and near-me searches | Research synthesis |
| Realtime weather, stocks, scores, and fresh facts | News or context analysis |
| Simple factoids with a direct answer | Multi-step reasoning |

## Deployment note

Deploy the **Next.js app in `web/`**, not the repository root. On Vercel (Git integration), set **Root Directory** to `web` — otherwise `npm install` fails looking for a root `package.json`.

The exported INT8 ONNX model is published on Hugging Face:

[https://huggingface.co/KenWu/multilingual-query-router](https://huggingface.co/KenWu/multilingual-query-router)

For production deployment, set `NEXT_PUBLIC_MODEL_REPO`:

```bash
.venv/bin/python ml/09_upload_hf.py --repo KenWu/multilingual-query-router
```

Then configure:

```text
NEXT_PUBLIC_MODEL_REPO=KenWu/multilingual-query-router
JIEHUO_MODEL_REPO=KenWu/multilingual-query-router
JIEHUO_MODEL_TIMEOUT_MS=2500
JIEHUO_ROUTER_MODE=model
```

`JIEHUO_ROUTER_MODE=heuristic` is an emergency fallback that keeps `/search?q=%s` working without model inference.

## Launch

- Live demo: https://jiehuo.vercel.app
- Twitter/X copy: `docs/twitter-launch.md`
- Link preview image: `web/app/opengraph-image.png` (1200×630, `summary_large_image` on X)
- Regenerate the card: `.venv/bin/python scripts/make_social_card.py`

## Feedback loop

Thumbs up/down in the UI POST to `/api/feedback`. With Supabase configured, feedback is stored in `query_feedback` for future retraining. Run `supabase/schema.sql` in Supabase to create the optional table.
