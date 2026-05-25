# JieHuo (解惑)

Multilingual neural query router that decides whether a search query should go to **Google** or **Perplexity**. Inspired by Perplexity mobile's automatic routing — extended to 12 languages including Chinese.

Runs inference **entirely in the browser** via transformers.js + INT8-quantized ONNX (~30 MB).

## Architecture

```
Query → multilingual-e5-small (fine-tuned) → P(google) vs P(perplexity)
                                              ↓
                                    Client-side Web Worker (transformers.js)
```

## Quick start

### 1. ML pipeline (train + export)

```bash
# Install Python deps (requires Python 3.11+)
pip install -e .

# Run full pipeline (uses fallback synthetic data if no API keys)
python ml/run_pipeline.py

# Or step-by-step:
python ml/02_synth_queries.py --provider fallback --per-bucket 20
python ml/03_heuristic_labels.py
python ml/04_llm_label.py --provider fallback
python ml/05_merge_split.py
python ml/06_train.py
python ml/07_eval.py
python ml/08_export_onnx.py
```

With API keys in `.env`, use real LLM labeling:

```bash
export OPENAI_API_KEY=sk-...
python ml/02_synth_queries.py --provider openai --per-bucket 200
python ml/04_llm_label.py --provider openai
```

### 2. Web app

```bash
cd web
npm install
npm run dev
# → http://localhost:3000
```

The exported model must exist at `web/public/models/multilingual-router/` (created by `ml/08_export_onnx.py`).

### 3. Deploy to Vercel

The quantized ONNX model (~118 MB) exceeds Vercel's 100 MB file limit. Host it on Hugging Face Hub:

```bash
pip install huggingface_hub
export HF_TOKEN=hf_...
python ml/09_upload_hf.py --repo YOUR_USERNAME/multilingual-query-router
```

Then deploy the web app (model excluded via `.vercelignore`, loaded from HF at runtime):

```bash
cd web
# Set in Vercel dashboard → Environment Variables:
# NEXT_PUBLIC_MODEL_REPO=YOUR_USERNAME/multilingual-query-router

npx vercel --prod
```

For **local dev**, the model in `web/public/models/` is used automatically (no env var needed).

## Label rubric

| Route to Google | Route to Perplexity |
|---|---|
| Navigational (site/app/brand) | Open-ended how/why questions |
| Transactional (buy/download) | Comparison & evaluation |
| Local / near me | Research synthesis |
| Real-time (weather, stocks) | News analysis |
| Simple factoids | Multi-step reasoning |

## Supported languages

`en`, `zh-CN`, `zh-TW`, `es`, `fr`, `de`, `ja`, `ko`, `pt`, `ru`, `ar`, `hi`

## Project structure

```
ml/           Training pipeline scripts
data/         Raw, synthetic, and labeled datasets
web/          Next.js app with client-side inference
checkpoints/  Fine-tuned PyTorch model
supabase/     Optional feedback table schema
```

## Feedback / active learning

Thumbs up/down in the UI POST to `/api/feedback`. With Supabase configured, feedback is stored in `query_feedback` for future retraining.

Run `supabase/schema.sql` in your Supabase project to create the table.
