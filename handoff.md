# JieHuo handoff

**Repo:** `/Users/kenwu/Documents/Github/JieHuo`  
**Goal:** make JieHuo presentable as a novel, hiring-worthy multilingual Google-vs-Perplexity query router.

## Current status

| Item | Status |
|---|---|
| Best checkpoint | `checkpoints/best_multilingual` |
| Public metric | Corrected 12-language held-out test + language/label balanced 240-query gold |
| Web model | Exported to `web/public/models/multilingual-router/` |
| Hugging Face model | `https://huggingface.co/KenWu/multilingual-query-router` |
| Browser runtime | transformers.js Web Worker, INT8 ONNX, temperature-calibrated confidence |
| Best story | GPT-4.1-mini teacher-distilled multilingual router with calibrated selective auto-routing |

The old `checkpoints/best_api` metric should not be used as the headline anymore. Its gold set was English-heavy because `ml/05_merge_split.py` previously failed to balance labels correctly. The fixed split now creates a balanced gold set with 20 examples per language and 10 examples per label per language.

## Headline metrics

Source: `checkpoints/eval_multilingual/metrics.json` and `checkpoints/eval_multilingual/showcase.md`.

Temperature calibration: `1.250` fit on `3,781` validation rows.

| Split | n | Accuracy | Balanced Acc | Macro F1 | ECE |
|---|---:|---:|---:|---:|---:|
| Held-out test | 3,820 | 0.8275 | 0.8283 | 0.8229 | 0.0638 |
| Balanced gold | 240 | 0.8833 | 0.8833 | 0.8828 | 0.0680 |

Selective auto-routing:

| Split | Threshold | Coverage | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Held-out test | conf >= 0.70 | 89.2% | 0.8589 | 0.8547 |
| Held-out test | conf >= 0.85 | 70.3% | 0.9103 | 0.9064 |
| Held-out test | conf >= 0.90 | 39.4% | 0.9641 | 0.9588 |
| Balanced gold | conf >= 0.80 | 88.8% | 0.9108 | 0.9106 |
| Balanced gold | conf >= 0.90 | 56.7% | 0.9632 | 0.9623 |

Tweet-sized framing:

> Built JieHuo: a 12-language browser-runnable router that decides whether a query belongs on Google or Perplexity. It is GPT-4.1-mini teacher-distilled, temperature-calibrated, and exported to INT8 ONNX. On a balanced multilingual gold set: 0.883 macro F1 at full coverage, 0.911 macro F1 at 89% auto-route coverage, 0.962 macro F1 at 57% high-confidence coverage.

## Important fixes landed

### `ml/05_merge_split.py`

- Fixed the label balancing bug: code was checking string labels against `LABEL_NAMES` dict keys, so labels like `"google"` and `"perplexity"` did not match as intended.
- Added label normalization before resolving labels.
- Rebuilt split from `data/labeled/llm_labeled_api.jsonl`:
  - train: `30,353`
  - val: `3,781`
  - test: `3,820`
  - gold: `240`

### `ml/06_train.py`

- Added `--max-train-per-language-label` to cap English dominance while keeping sparse non-English examples.
- Current run trained on `9,371` curated rows with:

```bash
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
```

### `ml/07_eval.py`

- Added temperature scaling via `--calibration`.
- Added raw uncalibrated metrics, selective operating points, confidence intervals, and `showcase.md`.

### `ml/08_export_onnx.py`

- Added `--temperature`.
- Export now writes `router_config.json` for browser confidence calibration.

### `web/lib/worker.ts`

- Loads `router_config.json`.
- Applies temperature scaling to model probabilities before displaying route confidence.

## Exact reproduction commands

```bash
cd /Users/kenwu/Documents/Github/JieHuo

.venv/bin/python ml/02_synth_queries.py --provider openai --model gpt-4.1-mini --per-bucket 50 --sleep 0.02
.venv/bin/python ml/03_heuristic_labels.py
.venv/bin/python ml/04_llm_label.py \
  --input data/labeled/heuristic_labeled.jsonl \
  --output data/labeled/llm_labeled_api.jsonl \
  --provider openai \
  --model gpt-4.1-mini \
  --batch-size 60 \
  --sleep 0.02
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

## Verification commands

```bash
.venv/bin/python -m ruff check ml scripts
.venv/bin/python -m compileall ml
bash -n scripts/*.sh
cd web && npm run build
```

`pytest` is installed, but the repo currently has no tests.

## Deployment note

The browser artifact is published at:

```text
https://huggingface.co/KenWu/multilingual-query-router
```

Set this in production:

```text
NEXT_PUBLIC_MODEL_REPO=KenWu/multilingual-query-router
```

Then deploy the Next.js app normally.

## Launch assets

- Social card: `assets/jiehuo-twitter-card.png`
- Twitter/X launch copy: `docs/twitter-launch.md`
- Regenerate the card: `.venv/bin/python scripts/make_social_card.py`
